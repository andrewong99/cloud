# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
The cloud-resolving model on the GPU: a line-by-line port of crm.CPUModel to
OpenGL 4.3 compute shaders.

Every field lives in one shader-storage buffer per time level (u, v, w, th',
qv, qc, qr, qi, qs packed end to end), so no kernel needs more than eight
bindings - the minimum GL 4.3 guarantees.  One step is

    3 x [ scalar, u, v, w tendencies -> divergence -> FFT x -> FFT y
          -> Thomas in z per wavenumber -> inverse FFT y -> inverse FFT x
          -> pressure-gradient correction ]
    microphysics (one thread per column) -> Sun's optical depth -> surface

The pressure solve is the exact FFT + tridiagonal method of the CPU model:
radix-2 FFTs in shared memory, one workgroup per grid row, one thread per
horizontal wavenumber for the tridiagonal.  The self-test runs this model and
the NumPy one from the same initial state and compares them field by field.
"""

from __future__ import annotations

import math

import numpy as np
import moderngl

from . import crm

LOCAL_X = 64


def compute_supported(ctx: moderngl.Context) -> bool:
    try:
        return ctx.version_code >= 430
    except Exception:
        return False


# ------------------------------------------------------------------ GLSL ----
COMMON = r"""
#define MX (NX-1)
#define MY (NY-1)
#define NCELL (NX*NY*NZ)
#define NWF (NX*NY*(NZ+1))
#define OFF_U 0
#define OFF_V NCELL
#define OFF_W (2*NCELL)
#define OFF_TH (2*NCELL+NWF)
#define OFF_QV (OFF_TH+NCELL)
#define OFF_QC (OFF_QV+NCELL)
#define OFF_QR (OFF_QC+NCELL)
#define OFF_QI (OFF_QR+NCELL)
#define OFF_QS (OFF_QI+NCELL)
#define PI 3.14159265358979

int idx(int i, int j, int k) { return (k * NY + (j & MY)) * NX + (i & MX); }
int cidx(int i, int j) { return (j & MY) * NX + (i & MX); }

float flux5(float am3, float am2, float am1, float a0, float ap1, float ap2, float vel) {
    return (vel * (37.0 * (a0 + am1) - 8.0 * (ap1 + am2) + (ap2 + am3))
            - abs(vel) * ((ap2 - am3) - 5.0 * (ap1 - am2) + 10.0 * (a0 - am1))) / 60.0;
}
float flux3(float am2, float am1, float a0, float ap1, float vel) {
    return (vel * (7.0 * (a0 + am1) - (ap1 + am2))
            + abs(vel) * ((ap1 - am2) - 3.0 * (a0 - am1))) / 12.0;
}
// flux through interface m of a column with n entries, with the order
// reduced where the stencil would leave the column (CPU flux_vertical)
float fluxV(int m, int n, float am3, float am2, float am1, float a0, float ap1, float ap2, float vel) {
    if (m <= 0 || m >= n) return 0.0;
    if (m >= 3 && m <= n - 3) return flux5(am3, am2, am1, a0, ap1, ap2, vel);
    if (m >= 2 && m <= n - 2) return flux3(am2, am1, a0, ap1, vel);
    return vel * 0.5 * (a0 + am1);
}
"""

BUFFERS = r"""
layout(std430, binding = 0) readonly buffer Cur { float c[]; } cur;
layout(std430, binding = 1) readonly buffer S0  { float c[]; } s0;
layout(std430, binding = 2) buffer Out { float c[]; } outb;
layout(std430, binding = 3) readonly buffer Base { float c[]; } base;
layout(std430, binding = 4) buffer Maps { float c[]; } maps;

float B(int off, int k) { return base.c[off + k]; }
float C(int off, int i, int j, int k) { return cur.c[off + idx(i, j, k)]; }
float Cw(int i, int j, int k) { return cur.c[OFF_W + idx(i, j, k)]; }
"""

# divergence / rho0 at a cell, from the stage input
DIVFUN = r"""
float divc(int i, int j, int k) {
    float du = C(OFF_U, i + 1, j, k) - C(OFF_U, i, j, k);
    float dv = C(OFF_V, i, j + 1, k) - C(OFF_V, i, j, k);
    float dw = B(B_RHOF, k + 1) * Cw(i, j, k + 1) - B(B_RHOF, k) * Cw(i, j, k);
    return du / DX + dv / DY + dw / (B(B_RHO0, k) * DZ);
}
"""

TEND_SCALARS = r"""
layout(local_size_x = LOCAL_X) in;
uniform float uDts;
uniform int uLateral;

float advScalar(int off, int i, int j, int k, float ux0, float ux1, float vy0, float vy1,
                float m0, float m1, float dv) {
    float a[7];
    for (int t = 0; t < 7; ++t) a[t] = C(off, i - 3 + t, j, k);
    float fx0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], ux0);
    float fx1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], ux1);
    for (int t = 0; t < 7; ++t) a[t] = C(off, i, j - 3 + t, k);
    float fy0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], vy0);
    float fy1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], vy1);
    float z[8];
    for (int t = 0; t < 8; ++t) z[t] = C(off, i, j, clamp(k - 4 + t, 0, NZ - 1));
    float fz0 = fluxV(k, NZ, z[1], z[2], z[3], z[4], z[5], z[6], m0);
    float fz1 = fluxV(k + 1, NZ, z[2], z[3], z[4], z[5], z[6], z[7], m1);
    float s = z[4];
    return -(fx1 - fx0) / DX - (fy1 - fy0) / DY - (fz1 - fz0) / (B(B_RHO0, k) * DZ) + s * dv;
}

void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y), k = int(gl_GlobalInvocationID.z);
    if (i >= NX || j >= NY || k >= NZ) return;
    float ux0 = C(OFF_U, i, j, k), ux1 = C(OFF_U, i + 1, j, k);
    float vy0 = C(OFF_V, i, j, k), vy1 = C(OFF_V, i, j + 1, k);
    float m0 = B(B_RHOF, k) * Cw(i, j, k), m1 = B(B_RHOF, k + 1) * Cw(i, j, k + 1);
    float dv = divc(i, j, k);
    float r0 = B(B_RHO0, k);
    // advection of the reference theta (see CPUModel._setup_constants)
    float flo = m1 * B(B_ALO, k + 1) - abs(m1) * B(B_DZC, k + 1);
    float fhi = m0 * B(B_AHI, k) - abs(m0) * B(B_DZC, k);
    float thBase = -(flo - fhi) / (r0 * DZ);
    int c = idx(i, j, k);
    float ls = (uLateral != 0) ? maps.c[M_LAT + cidx(i, j)] : 0.0;
    float damp = B(B_DAMPC, k);

    float t_th = advScalar(OFF_TH, i, j, k, ux0, ux1, vy0, vy1, m0, m1, dv) + thBase;
    t_th -= damp * cur.c[OFF_TH + c];
    t_th -= ls * cur.c[OFF_TH + c];
    outb.c[OFF_TH + c] = s0.c[OFF_TH + c] + uDts * t_th;

    float t_qv = advScalar(OFF_QV, i, j, k, ux0, ux1, vy0, vy1, m0, m1, dv);
    t_qv -= ls * (cur.c[OFF_QV + c] - B(B_QV0, k));
    outb.c[OFF_QV + c] = s0.c[OFF_QV + c] + uDts * t_qv;

    float t;
    t = advScalar(OFF_QC, i, j, k, ux0, ux1, vy0, vy1, m0, m1, dv) - ls * cur.c[OFF_QC + c];
    outb.c[OFF_QC + c] = s0.c[OFF_QC + c] + uDts * t;
    t = advScalar(OFF_QR, i, j, k, ux0, ux1, vy0, vy1, m0, m1, dv) - ls * cur.c[OFF_QR + c];
    outb.c[OFF_QR + c] = s0.c[OFF_QR + c] + uDts * t;
    t = advScalar(OFF_QI, i, j, k, ux0, ux1, vy0, vy1, m0, m1, dv) - ls * cur.c[OFF_QI + c];
    outb.c[OFF_QI + c] = s0.c[OFF_QI + c] + uDts * t;
    t = advScalar(OFF_QS, i, j, k, ux0, ux1, vy0, vy1, m0, m1, dv) - ls * cur.c[OFF_QS + c];
    outb.c[OFF_QS + c] = s0.c[OFF_QS + c] + uDts * t;
}
"""

TEND_U = r"""
layout(local_size_x = LOCAL_X) in;
uniform float uDts;
uniform int uLateral;
uniform int uDrag;
uniform float uCd;
uniform float uCor;
uniform vec2 uMove;

void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y), k = int(gl_GlobalInvocationID.z);
    if (i >= NX || j >= NY || k >= NZ) return;
    float a[7];
    for (int t = 0; t < 7; ++t) a[t] = C(OFF_U, i - 3 + t, j, k);
    // x: interfaces are the cell centres i-1 and i
    float fx0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], 0.5 * (a[2] + a[3]));
    float fx1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], 0.5 * (a[3] + a[4]));
    float u = a[3];
    // y: interfaces are y faces j and j+1 on this x face
    for (int t = 0; t < 7; ++t) a[t] = C(OFF_U, i, j - 3 + t, k);
    float vy0 = 0.5 * (C(OFF_V, i - 1, j, k) + C(OFF_V, i, j, k));
    float vy1 = 0.5 * (C(OFF_V, i - 1, j + 1, k) + C(OFF_V, i, j + 1, k));
    float fy0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], vy0);
    float fy1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], vy1);
    // z: interfaces are w levels k and k+1 on this x face
    float z[8];
    for (int t = 0; t < 8; ++t) z[t] = C(OFF_U, i, j, clamp(k - 4 + t, 0, NZ - 1));
    float mz0 = 0.5 * B(B_RHOF, k) * (Cw(i - 1, j, k) + Cw(i, j, k));
    float mz1 = 0.5 * B(B_RHOF, k + 1) * (Cw(i - 1, j, k + 1) + Cw(i, j, k + 1));
    float fz0 = fluxV(k, NZ, z[1], z[2], z[3], z[4], z[5], z[6], mz0);
    float fz1 = fluxV(k + 1, NZ, z[2], z[3], z[4], z[5], z[6], z[7], mz1);
    float r0 = B(B_RHO0, k);
    float dvf = 0.5 * (divc(i - 1, j, k) + divc(i, j, k));
    float tend = -(fx1 - fx0) / DX - (fy1 - fy0) / DY - (fz1 - fz0) / (r0 * DZ) + u * dvf;
    tend -= B(B_DAMPC, k) * (u - B(B_U0, k));
    if (uLateral != 0) tend -= maps.c[M_LAT + cidx(i, j)] * (u - B(B_U0, k));
    if (uCor != 0.0) {
        float vu = 0.25 * (C(OFF_V, i, j, k) + C(OFF_V, i - 1, j, k) + C(OFF_V, i, j + 1, k) + C(OFF_V, i - 1, j + 1, k));
        tend += uCor * (vu - B(B_VG0, k));
    }
    if (uDrag != 0 && k == 0) {
        float ug = u + uMove.x;
        float vg = C(OFF_V, i, j, 0) + uMove.y;
        float spd = sqrt(ug * ug + vg * vg + 1.0);
        tend -= uCd * spd * ug / DZ;
    }
    int c = idx(i, j, k);
    outb.c[OFF_U + c] = s0.c[OFF_U + c] + uDts * tend;
}
"""

TEND_V = r"""
layout(local_size_x = LOCAL_X) in;
uniform float uDts;
uniform int uLateral;
uniform int uDrag;
uniform float uCd;
uniform float uCor;
uniform vec2 uMove;

void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y), k = int(gl_GlobalInvocationID.z);
    if (i >= NX || j >= NY || k >= NZ) return;
    float a[7];
    for (int t = 0; t < 7; ++t) a[t] = C(OFF_V, i - 3 + t, j, k);
    float ux0 = 0.5 * (C(OFF_U, i, j - 1, k) + C(OFF_U, i, j, k));
    float ux1 = 0.5 * (C(OFF_U, i + 1, j - 1, k) + C(OFF_U, i + 1, j, k));
    float fx0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], ux0);
    float fx1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], ux1);
    for (int t = 0; t < 7; ++t) a[t] = C(OFF_V, i, j - 3 + t, k);
    float fy0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], 0.5 * (a[2] + a[3]));
    float fy1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], 0.5 * (a[3] + a[4]));
    float v = a[3];
    float z[8];
    for (int t = 0; t < 8; ++t) z[t] = C(OFF_V, i, j, clamp(k - 4 + t, 0, NZ - 1));
    float mz0 = 0.5 * B(B_RHOF, k) * (Cw(i, j - 1, k) + Cw(i, j, k));
    float mz1 = 0.5 * B(B_RHOF, k + 1) * (Cw(i, j - 1, k + 1) + Cw(i, j, k + 1));
    float fz0 = fluxV(k, NZ, z[1], z[2], z[3], z[4], z[5], z[6], mz0);
    float fz1 = fluxV(k + 1, NZ, z[2], z[3], z[4], z[5], z[6], z[7], mz1);
    float r0 = B(B_RHO0, k);
    float dvf = 0.5 * (divc(i, j - 1, k) + divc(i, j, k));
    float tend = -(fx1 - fx0) / DX - (fy1 - fy0) / DY - (fz1 - fz0) / (r0 * DZ) + v * dvf;
    tend -= B(B_DAMPC, k) * (v - B(B_V0, k));
    if (uLateral != 0) tend -= maps.c[M_LAT + cidx(i, j)] * (v - B(B_V0, k));
    if (uCor != 0.0) {
        float uv = 0.25 * (C(OFF_U, i, j, k) + C(OFF_U, i + 1, j, k) + C(OFF_U, i, j - 1, k) + C(OFF_U, i + 1, j - 1, k));
        tend -= uCor * (uv - B(B_UG0, k));
    }
    if (uDrag != 0 && k == 0) {
        float ug = C(OFF_U, i, j, 0) + uMove.x;
        float vg = v + uMove.y;
        float spd = sqrt(ug * ug + vg * vg + 1.0);
        tend -= uCd * spd * vg / DZ;
    }
    int c = idx(i, j, k);
    outb.c[OFF_V + c] = s0.c[OFF_V + c] + uDts * tend;
}
"""

TEND_W = r"""
layout(local_size_x = LOCAL_X) in;
uniform float uDts;
uniform int uLateral;

float buoy(int i, int j, int k) {
    int c = idx(i, j, k);
    float cond = cur.c[OFF_QC + c] + cur.c[OFF_QR + c] + cur.c[OFF_QI + c] + cur.c[OFF_QS + c];
    return G * (cur.c[OFF_TH + c] / B(B_TH0, k) + REPSM1 * (cur.c[OFF_QV + c] - B(B_QV0, k)) - cond);
}

void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y), k = int(gl_GlobalInvocationID.z);
    if (i >= NX || j >= NY || k > NZ) return;
    int c = idx(i, j, k);
    if (k == 0 || k == NZ) { outb.c[OFF_W + c] = 0.0; return; }
    float rf = B(B_RHOF, k);
    float a[7];
    for (int t = 0; t < 7; ++t) a[t] = Cw(i - 3 + t, j, k);
    float r0l = B(B_RHO0, k - 1), r0u = B(B_RHO0, k);
    float mx0 = 0.5 * (r0l * C(OFF_U, i, j, k - 1) + r0u * C(OFF_U, i, j, k));
    float mx1 = 0.5 * (r0l * C(OFF_U, i + 1, j, k - 1) + r0u * C(OFF_U, i + 1, j, k));
    float fx0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], mx0);
    float fx1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], mx1);
    float w = a[3];
    for (int t = 0; t < 7; ++t) a[t] = Cw(i, j - 3 + t, k);
    float my0 = 0.5 * (r0l * C(OFF_V, i, j, k - 1) + r0u * C(OFF_V, i, j, k));
    float my1 = 0.5 * (r0l * C(OFF_V, i, j + 1, k - 1) + r0u * C(OFF_V, i, j + 1, k));
    float fy0 = flux5(a[0], a[1], a[2], a[3], a[4], a[5], my0);
    float fy1 = flux5(a[1], a[2], a[3], a[4], a[5], a[6], my1);
    // z: w array has NZ+1 entries; interface m sits at cell centre m-1
    float z[8];
    for (int t = 0; t < 8; ++t) z[t] = Cw(i, j, clamp(k - 4 + t, 0, NZ));
    float mc0 = 0.5 * (B(B_RHOF, k - 1) * z[3] + B(B_RHOF, k) * z[4]);       // centre k-1
    float mc1 = 0.5 * (B(B_RHOF, k) * z[4] + B(B_RHOF, k + 1) * z[5]);       // centre k
    float fz0 = fluxV(k, NZ + 1, z[1], z[2], z[3], z[4], z[5], z[6], mc0);
    float fz1 = fluxV(k + 1, NZ + 1, z[2], z[3], z[4], z[5], z[6], z[7], mc1);
    float dvf = 0.5 * (divc(i, j, k - 1) + divc(i, j, k));
    float tend = -(fx1 - fx0) / (DX * rf) - (fy1 - fy0) / (DY * rf) - (fz1 - fz0) / (DZ * rf) + w * dvf;
    tend += 0.5 * (buoy(i, j, k - 1) + buoy(i, j, k));
    tend -= B(B_DAMPF, k) * w;
    if (uLateral != 0) tend -= maps.c[M_LAT + cidx(i, j)] * w;
    outb.c[OFF_W + c] = s0.c[OFF_W + c] + uDts * tend;
}
"""

# right-hand side of the Poisson equation, from the unprojected output
DIVERGENCE = r"""
layout(local_size_x = LOCAL_X) in;
layout(std430, binding = 2) readonly buffer Out { float c[]; } outb;
layout(std430, binding = 3) readonly buffer Base { float c[]; } base;
layout(std430, binding = 5) buffer F { vec2 c[]; } fft;
uniform float uDts;
float B(int off, int k) { return base.c[off + k]; }
void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y), k = int(gl_GlobalInvocationID.z);
    if (i >= NX || j >= NY || k >= NZ) return;
    float du = outb.c[OFF_U + idx(i + 1, j, k)] - outb.c[OFF_U + idx(i, j, k)];
    float dv = outb.c[OFF_V + idx(i, j + 1, k)] - outb.c[OFF_V + idx(i, j, k)];
    float dw = B(B_RHOF, k + 1) * outb.c[OFF_W + idx(i, j, k + 1)] - B(B_RHOF, k) * outb.c[OFF_W + idx(i, j, k)];
    float rhs = (B(B_RHO0, k) * (du / DX + dv / DY) + dw / DZ) / uDts;
    fft.c[idx(i, j, k)] = vec2(rhs, 0.0);
}
"""

FFT = r"""
layout(local_size_x = HALFN) in;
layout(std430, binding = 5) buffer F { vec2 c[]; } fft;
uniform float uSign;
shared vec2 sh[LEN];
void main() {
    uint t = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.x;
    uint k = gl_WorkGroupID.y;
#ifdef ALONG_X
    uint base0 = (k * uint(NY) + row) * uint(NX);
    uint stride = 1u;
#else
    uint base0 = k * uint(NY) * uint(NX) + row;
    uint stride = uint(NX);
#endif
    for (uint e = t; e < uint(LEN); e += uint(HALFN)) {
        uint r = bitfieldReverse(e) >> (32u - uint(LOG2LEN));
        sh[r] = fft.c[base0 + e * stride];
    }
    memoryBarrierShared();
    barrier();
    for (uint len = 2u; len <= uint(LEN); len <<= 1u) {
        uint hl = len >> 1u;
        uint grp = t / hl, pos = t % hl;
        uint i0 = grp * len + pos, i1 = i0 + hl;
        float ang = uSign * 2.0 * PI * float(pos) / float(len);
        vec2 w = vec2(cos(ang), sin(ang));
        vec2 a = sh[i0];
        vec2 b = sh[i1];
        vec2 bw = vec2(b.x * w.x - b.y * w.y, b.x * w.y + b.y * w.x);
        sh[i0] = a + bw;
        sh[i1] = a - bw;
        memoryBarrierShared();
        barrier();
    }
    for (uint e = t; e < uint(LEN); e += uint(HALFN)) fft.c[base0 + e * stride] = sh[e];
}
"""

TRIDIAG = r"""
layout(local_size_x = LOCAL_X) in;
layout(std430, binding = 3) readonly buffer Base { float c[]; } base;
layout(std430, binding = 5) buffer F { vec2 c[]; } fft;
layout(std430, binding = 6) buffer S { float c[]; } scratch;
float B(int off, int k) { return base.c[off + k]; }
void main() {
    int l = int(gl_GlobalInvocationID.x), m = int(gl_GlobalInvocationID.y);
    if (l >= NX || m >= NY) return;
    float lam = 2.0 * (cos(2.0 * PI * float(l) / float(NX)) - 1.0) / (DX * DX)
              + 2.0 * (cos(2.0 * PI * float(m) / float(NY)) - 1.0) / (DY * DY);
    bool zero = (l == 0 && m == 0);
    float b0 = zero ? 1.0 : B(B_PBZ, 0) + lam;
    float c0 = zero ? 0.0 : B(B_PC, 0);
    vec2 r0 = zero ? vec2(0.0) : fft.c[idx(l, m, 0)];
    float cp = c0 / b0;
    vec2 dp = r0 / b0;
    scratch.c[idx(l, m, 0)] = cp;
    fft.c[idx(l, m, 0)] = dp;
    for (int k = 1; k < NZ; ++k) {
        float a = B(B_PA, k);
        float den = B(B_PBZ, k) + lam - a * cp;
        cp = B(B_PC, k) / den;
        dp = (fft.c[idx(l, m, k)] - a * dp) / den;
        scratch.c[idx(l, m, k)] = cp;
        fft.c[idx(l, m, k)] = dp;
    }
    vec2 x = dp;
    for (int k = NZ - 2; k >= 0; --k) {
        x = fft.c[idx(l, m, k)] - scratch.c[idx(l, m, k)] * x;
        fft.c[idx(l, m, k)] = x;
    }
}
"""

PROJECT = r"""
layout(local_size_x = LOCAL_X) in;
layout(std430, binding = 2) buffer Out { float c[]; } outb;
layout(std430, binding = 3) readonly buffer Base { float c[]; } base;
layout(std430, binding = 5) readonly buffer F { vec2 c[]; } fft;
uniform float uDts;
float B(int off, int k) { return base.c[off + k]; }
float phi(int i, int j, int k) { return fft.c[idx(i, j, k)].x * (1.0 / float(NX * NY)) / B(B_RHO0, k); }
void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y), k = int(gl_GlobalInvocationID.z);
    if (i >= NX || j >= NY || k > NZ) return;
    if (k < NZ) {
        float p = phi(i, j, k);
        outb.c[OFF_U + idx(i, j, k)] -= uDts * (p - phi(i - 1, j, k)) / DX;
        outb.c[OFF_V + idx(i, j, k)] -= uDts * (p - phi(i, j - 1, k)) / DY;
    }
    if (k >= 1 && k < NZ) {
        outb.c[OFF_W + idx(i, j, k)] -= uDts * (phi(i, j, k) - phi(i, j, k - 1)) / DZ;
    }
}
"""

MICRO = r"""
layout(local_size_x = LOCAL_X) in;
layout(std430, binding = 0) buffer St { float c[]; } st;
layout(std430, binding = 3) readonly buffer Base { float c[]; } base;
layout(std430, binding = 4) buffer Maps { float c[]; } maps;
layout(std430, binding = 7) buffer Diag { float c[]; } diag;
layout(rgba16f, binding = 0) uniform writeonly image3D imgOpt;
layout(rgba16f, binding = 1) uniform writeonly image3D imgAux;
uniform float uDt;
uniform int uIce;
uniform int uForcing;       // bit 1: radiation profile, 2: moisture advection, 4: subsidence,
                            // 8: DYCOMS-II long-wave
uniform vec2 uMove;
uniform float uQAuto;
uniform float uQiAuto;
uniform float uGraupel;
uniform float uLwF0, uLwF1, uLwKappa, uLwDiv, uLwZiQt, uLwRhoi;
uniform float uNudgeTau;
uniform int uNudgeUV;
layout(std430, binding = 6) readonly buffer Prof { float c[]; } prof;
float B(int off, int k) { return base.c[off + k]; }

float esl(float t) { return 611.2 * exp(17.67 * (t - T0K) / (t - 29.65)); }
float esi(float t) { return 611.2 * exp(21.8745584 * (t - T0K) / (t - 7.66)); }
float qsl(float t, float p) { float e = min(esl(t), 0.5 * p); return EPS * e / (p - e); }
float qsi(float t, float p) { float e = min(esi(t), 0.5 * p); return EPS * e / (p - e); }
float dqsl(float t, float p, float q) { float e = min(esl(t), 0.5 * p); return q * (p / (p - e)) * 17.67 * (T0K - 29.65) / ((t - 29.65) * (t - 29.65)); }
float dqsi(float t, float p, float q) { float e = min(esi(t), 0.5 * p); return q * (p / (p - e)) * 21.8745584 * (T0K - 7.66) / ((t - 7.66) * (t - 7.66)); }
float fliq(float t) { return clamp((t - 253.16) / 20.0, 0.0, 1.0); }
float fgrp(float t) { return uGraupel * clamp((t - 223.16) / 60.0, 0.0, 1.0); }
float lamMP(float rq, float rx, float n0) { return pow(PI * rx * n0 / max(rq, 1e-14), 0.25); }
float vRain(float q, float rho) { return q > 1e-12 ? VR_COEF * pow(max(rho * q, 0.0), VR_EXP) * sqrt(1.15 / rho) : 0.0; }
float vIce(float q, float rho, float t) {
    if (q <= 1e-12) return 0.0;
    float fg = fgrp(t);
    float rq = max(rho * q, 0.0);
    return (fg * VG_COEF * pow(rq, B_G / 4.0) + (1.0 - fg) * VS_COEF * pow(rq, B_S / 4.0)) * sqrt(RHO_REF_ICE / rho);
}
float extMP(float rq, float rx, float n0) {
    if (rq <= 1e-12) return 0.0;
    float l = lamMP(rq, rx, n0);
    return PI * n0 / (l * l * l);
}

float th[NZ], qv[NZ], qc[NZ], qr[NZ], qi[NZ], qs[NZ], vf[NZ];

void pdef(int which, inout float deficit) {
    // which: 0 qc 1 qr 2 qi 3 qs 4 qv
    float before = 0.0, after = 0.0;
    for (int k = 0; k < NZ; ++k) {
        float q = which == 0 ? qc[k] : which == 1 ? qr[k] : which == 2 ? qi[k] : which == 3 ? qs[k] : qv[k];
        before += B(B_RHO0, k) * q;
        after += B(B_RHO0, k) * max(q, 0.0);
    }
    if (which == 4) before -= deficit;
    float scale = after > 0.0 ? max(before, 0.0) / max(after, 1e-30) : 0.0;
    if (which != 4) deficit += max(-before, 0.0);
    for (int k = 0; k < NZ; ++k) {
        if (which == 0) qc[k] = max(qc[k], 0.0) * scale;
        else if (which == 1) qr[k] = max(qr[k], 0.0) * scale;
        else if (which == 2) qi[k] = max(qi[k], 0.0) * scale;
        else if (which == 3) qs[k] = max(qs[k], 0.0) * scale;
        else qv[k] = max(qv[k], 0.0) * scale;
    }
}

float sediment(bool rain) {
    // fall speeds once, copied down below the precipitation edge
    for (int k = 0; k < NZ; ++k) {
        float t = (B(B_TH0, k) + th[k]) * B(B_PI0, k);
        vf[k] = rain ? vRain(qr[k], B(B_RHO0, k)) : vIce(qs[k], B(B_RHO0, k), t);
    }
    for (int k = NZ - 2; k >= 0; --k) if (!(vf[k] > 0.0)) vf[k] = vf[k + 1];
    float cr = 0.0;
    for (int k = 0; k < NZ; ++k) cr = max(cr, vf[k] * uDt / DZ);
    int n = int(floor(cr / 0.8)) + 1;
    float dtf = uDt / float(n);
    float fell = 0.0;
    for (int it = 0; it < n; ++it) {
        float fabove = 0.0;
        // top-down so each level uses the flux of the level above from THIS sub-step
        float fprev = 0.0;
        for (int k = NZ - 1; k >= 0; --k) {
            float r0 = B(B_RHO0, k);
            float q = rain ? qr[k] : qs[k];
            float f = r0 * vf[k] * q;
            float nq = q + dtf * (fprev - f) / (r0 * DZ);
            if (rain) qr[k] = nq; else qs[k] = nq;
            fprev = f;
            if (k == 0) fell += dtf * f;
        }
    }
    return fell;
}

void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y);
    if (i >= NX || j >= NY) return;
    for (int k = 0; k < NZ; ++k) {
        int c = idx(i, j, k);
        th[k] = st.c[OFF_TH + c]; qv[k] = st.c[OFF_QV + c]; qc[k] = st.c[OFF_QC + c];
        qr[k] = st.c[OFF_QR + c]; qi[k] = st.c[OFF_QI + c]; qs[k] = st.c[OFF_QS + c];
    }
    // ---- large-scale forcing (CPUModel._forcing) ----
    if ((uForcing & 1) != 0) for (int k = 0; k < NZ; ++k) th[k] += uDt * B(B_RAD, k) / B(B_PI0, k);
    if ((uForcing & 8) != 0) {
        // DYCOMS-II long-wave (CPUModel.lw_heating)
        float ztop = 1e30;
        if (uLwDiv > 0.0) {
            int kf = -1;
            for (int k = 0; k < NZ; ++k) { if (qv[k] + qc[k] + qi[k] < uLwZiQt) { kf = k; break; } }
            if (kf > 0) {
                float qlo = qv[kf - 1] + qc[kf - 1] + qi[kf - 1];
                float qhi = qv[kf] + qc[kf] + qi[kf];
                float f = clamp((qlo - uLwZiQt) / max(qlo - qhi, 1e-12), 0.0, 1.0);
                ztop = Z0 + (float(kf - 1) + 0.5) * DZ + f * DZ;
            }
        }
        float qtot = 0.0;
        for (int k = 0; k < NZ; ++k) qtot += uLwKappa * B(B_RHO0, k) * (max(qc[k], 0.0) + LW_ICE_RATIO * max(qi[k], 0.0)) * DZ;
        float qb = 0.0;
        float fprev = uLwF0 * exp(-qtot) + uLwF1;
        {
            float d = Z0 - ztop;
            if (d > 0.0) fprev += uLwRhoi * CP * uLwDiv * (pow(d, 4.0 / 3.0) / 4.0 + ztop * pow(d, 1.0 / 3.0));
        }
        for (int k = 0; k < NZ; ++k) {
            float r0 = B(B_RHO0, k);
            qb += uLwKappa * r0 * (max(qc[k], 0.0) + LW_ICE_RATIO * max(qi[k], 0.0)) * DZ;
            float fnext = uLwF0 * exp(-max(qtot - qb, 0.0)) + uLwF1 * exp(-qb);
            float d = Z0 + float(k + 1) * DZ - ztop;
            if (d > 0.0) fnext += uLwRhoi * CP * uLwDiv * (pow(d, 4.0 / 3.0) / 4.0 + ztop * pow(d, 1.0 / 3.0));
            th[k] += uDt * (-(fnext - fprev) / (r0 * CP * DZ)) / B(B_PI0, k);
            fprev = fnext;
        }
    }
    if (uNudgeTau > 0.0) {
        // relax the horizontal means of theta_l and q_t (PROFILE) to the
        // initial profile (CPUModel._forcing)
        for (int k = 0; k < NZ; ++k) {
            float wk = B(B_NUDGE, k) / uNudgeTau;
            th[k] += uDt * wk * (B(B_THLT, k) - prof.c[4 * k]);
            qv[k] += uDt * wk * (B(B_QTT, k) - prof.c[4 * k + 1]);
            if (uNudgeUV != 0) {
                int c = idx(i, j, k);
                st.c[OFF_U + c] += uDt * wk * (B(B_U0, k) - prof.c[4 * k + 2]);
                st.c[OFF_V + c] += uDt * wk * (B(B_V0, k) - prof.c[4 * k + 3]);
            }
        }
    }
    if ((uForcing & 2) != 0) for (int k = 0; k < NZ; ++k) qv[k] += uDt * B(B_MADV, k);
    if ((uForcing & 4) != 0) {
        float gth[NZ];
        for (int k = 0; k < NZ; ++k) gth[k] = (k < NZ - 1) ? ((B(B_TH0, k + 1) + th[k + 1]) - (B(B_TH0, k) + th[k])) / DZ : 0.0;
        for (int k = 0; k < NZ; ++k) th[k] -= uDt * B(B_WLS, k) * gth[k];
        for (int k = 0; k < NZ; ++k) gth[k] = (k < NZ - 1) ? (qv[k + 1] - qv[k]) / DZ : 0.0;
        for (int k = 0; k < NZ; ++k) qv[k] -= uDt * B(B_WLS, k) * gth[k];
        for (int k = 0; k < NZ; ++k) gth[k] = (k < NZ - 1) ? (qc[k + 1] - qc[k]) / DZ : 0.0;
        for (int k = 0; k < NZ; ++k) qc[k] -= uDt * B(B_WLS, k) * gth[k];
        for (int k = 0; k < NZ; ++k) gth[k] = (k < NZ - 1) ? (qi[k + 1] - qi[k]) / DZ : 0.0;
        for (int k = 0; k < NZ; ++k) qi[k] -= uDt * B(B_WLS, k) * gth[k];
    }
    // ---- positivity ----
    float deficit = 0.0;
    pdef(0, deficit); pdef(1, deficit); pdef(2, deficit); pdef(3, deficit); pdef(4, deficit);
    // ---- sedimentation ----
    float fell = sediment(true);
    if (uIce != 0) fell += sediment(false);
    int cc = cidx(i, j);
    maps.c[M_RAIN + cc] += fell;
    maps.c[M_RATE + cc] = fell / uDt;
    // ---- conversions (CPUModel._conversions) and saturation adjustment ----
    for (int k = 0; k < NZ; ++k) {
        float pi0 = B(B_PI0, k), p = B(B_P0, k), rho = B(B_RHO0, k);
        float t = (B(B_TH0, k) + th[k]) * pi0;
        float lv = LV1 - LV2 * t;
        float lf = (LS1 - LS2 * t) - lv;
        float Qv = qv[k], Qc = qc[k], Qr = qr[k], Qi = qi[k], Qs = qs[k], Th = th[k];
        float ar = max(K_AUTO * (Qc - uQAuto), 0.0) * uDt;
        float cr = K_ACCR * Qc * pow(max(Qr, 0.0), 0.875) * uDt;
        float tot = ar + cr;
        if (tot > Qc) { float lim = Qc / max(tot, 1e-30); ar *= lim; cr *= lim; }
        float qvs = qsl(t, p);
        float rq = max(rho * Qr, 0.0);
        float er = 0.0;
        if (Qr > 0.0 && Qv < qvs)
            er = (1.6 + 30.3922 * pow(rq, 0.2046)) * (1.0 - Qv / qvs) * pow(rq, 0.525)
                 / ((2.03e4 + 9.584e6 / (qvs * p)) * rho);
        er = min(er * uDt, Qr);
        if (Qv + er > qvs) er = max(qvs - Qv, 0.0);
        Qv += er; Qc -= ar + cr; Qr += ar + cr - er;
        Th -= er * lv / (CP * pi0);
        if (uIce != 0) {
            float tc = t - T0K;
            float ai = max(K_AUTO_ICE * exp(0.025 * tc) * (Qi - uQiAuto), 0.0) * uDt;
            ai = min(ai, Qi);
            Qi -= ai; Qs += ai;
            float fg = fgrp(t);
            float rqs = max(rho * Qs, 0.0);
            float coll = (fg * CG_COEF * pow(rqs, (3.0 + B_G) / 4.0)
                          + (1.0 - fg) * CS_COEF * pow(rqs, (3.0 + B_S) / 4.0)) * sqrt(RHO_REF_ICE / rho);
            float dqc = min(Qc, coll * E_CLOUD_BY_ICE_PRECIP * Qc * uDt);
            float dqi = min(Qi, coll * E_ICE_BY_ICE_PRECIP * Qi * uDt);
            bool cold = t < T0K;
            Qc -= dqc; Qi -= dqi;
            Qs += dqi + (cold ? dqc : 0.0);
            Qr += cold ? 0.0 : dqc;
            if (cold) Th += dqc * lf / (CP * pi0);
            float n0 = fg * N0G + (1.0 - fg) * N0S;
            float rx = fg * RHO_G + (1.0 - fg) * RHO_S;
            float lam = lamMP(rqs, rx, n0);
            float melt = (!cold && Qs > 1e-12) ? 2.0 * PI * n0 * F_VENT * K_AIR * (t - T0K) / (lam * lam * lf * rho) * uDt : 0.0;
            melt = min(melt, Qs);
            Qs -= melt; Qr += melt;
            Th -= melt * lf / (CP * pi0);
            float rqr = max(rho * Qr, 0.0);
            float lamr = lamMP(rqr, RHOW, N0R);
            float frz = 0.0;
            if (cold && Qr > 1e-12) {
                float l2 = lamr * lamr, l7 = l2 * l2 * l2 * lamr;
                frz = 20.0 * PI * PI * BIGG_B * N0R * (RHOW / rho) * (exp(BIGG_A * min(T0K - t, 60.0)) - 1.0) / l7 * uDt;
            }
            frz = min(frz, Qr);
            Qr -= frz; Qs += frz;
            Th += frz * lf / (CP * pi0);
            float qvi = qsi(t, p);
            float ls = LS1 - LS2 * t;
            float es_i = min(esi(t), 0.5 * p);
            float dv = DV0 * pow(t / T0K, 1.81) * (1.0e5 / p);
            float aa = (ls / (RV * t) - 1.0) * ls / (K_AIR * t);
            float bb = RV * t / (dv * es_i);
            float sub = (Qs > 1e-12 && Qv < qvi) ? 2.0 * PI * (1.0 - Qv / qvi) * n0 * F_VENT / (lam * lam * (aa + bb) * rho) * uDt : 0.0;
            sub = min(sub, Qs);
            sub = min(sub, max(qvi - Qv, 0.0));
            Qs -= sub; Qv += sub;
            Th -= sub * ls / (CP * pi0);
        }
        // ---- saturation adjustment (CPUModel._saturation_adjust) ----
        float T = (B(B_TH0, k) + Th) * pi0;
        float Lv = LV1 - LV2 * T;
        float Lf = (LS1 - LS2 * T) - Lv;
        float qn = Qc + Qi;
        float qt = Qv + qn;
        float tl = T - (Lv * Qc + (Lv + Lf) * Qi) / CP;
        float qsm, dq, fl, dfl;
        // qsat over the mixed cloud at tl
        {
            float qw = qsl(tl, p);
            if (uIce != 0) { fl = fliq(tl); qsm = fl * qw + (1.0 - fl) * qsi(tl, p); }
            else { qsm = qw; }
        }
        bool cloudy = qt > qsm;
        float tn = tl;
        if (cloudy) {
            for (int it = 0; it < 4; ++it) {
                float qw = qsl(tn, p);
                if (uIce != 0) {
                    float qii = qsi(tn, p);
                    fl = fliq(tn);
                    dfl = (tn > 253.16 && tn < 273.16) ? 1.0 / 20.0 : 0.0;
                    qsm = fl * qw + (1.0 - fl) * qii;
                    dq = fl * dqsl(tn, p, qw) + (1.0 - fl) * dqsi(tn, p, qii) + dfl * (qw - qii);
                } else {
                    fl = 1.0; dfl = 0.0; qsm = qw; dq = dqsl(tn, p, qw);
                }
                float leff = Lv + (1.0 - fl) * Lf;
                float cond = qt - qsm;
                float f = tn - leff * cond / CP - tl;
                float fp = 1.0 + (Lf * dfl * cond + leff * dq) / CP;
                tn = tn - f / fp;
            }
        }
        float qnn = 0.0;
        fl = 1.0;
        if (cloudy) {
            float qw = qsl(tn, p);
            if (uIce != 0) { fl = fliq(tn); qsm = fl * qw + (1.0 - fl) * qsi(tn, p); }
            else qsm = qw;
            qnn = max(qt - qsm, 0.0);
        }
        qc[k] = fl * qnn;
        qi[k] = (1.0 - fl) * qnn;
        qv[k] = qt - qnn;
        th[k] = tn / pi0 - B(B_TH0, k);
        qr[k] = Qr; qs[k] = Qs;
    }
    // ---- write back, optics for the renderer, column diagnostics ----
    float tauUp = 0.0;
    float colMaxU = 0.0, colMaxV = 0.0, colMaxW = 0.0, wmax = -1e9, wmin = 1e9;
    float top = 0.0, bot = 1e9, lwp = 0.0, water = 0.0, icesum = 0.0, condsum = 0.0;
    float charge = 0.0, chargeZ = 0.0, chargeMax = 0.0;
    float pbot = 1e9;
    for (int k = NZ - 1; k >= 0; --k) {
        int c = idx(i, j, k);
        st.c[OFF_TH + c] = th[k]; st.c[OFF_QV + c] = qv[k]; st.c[OFF_QC + c] = qc[k];
        st.c[OFF_QR + c] = qr[k]; st.c[OFF_QI + c] = qi[k]; st.c[OFF_QS + c] = qs[k];
        float rho = B(B_RHO0, k);
        float t = (B(B_TH0, k) + th[k]) * B(B_PI0, k);
        float sl = 3.0 * rho * max(qc[k], 0.0) / (2.0 * RHOW * R_EFF_LIQ);
        float si = 3.0 * rho * max(qi[k], 0.0) / (2.0 * RHOI * R_EFF_ICE);
        float sr = extMP(rho * max(qr[k], 0.0), RHOW, N0R);
        float fg = fgrp(t);
        float rqs = rho * max(qs[k], 0.0);
        float ss = fg * extMP(rqs, RHO_G, N0G) + (1.0 - fg) * extMP(rqs, RHO_S, N0S);
        float sc = sl + si;
        float tauMid = tauUp + 0.5 * sc * DZ;
        tauUp += sc * DZ;
        imageStore(imgOpt, ivec3(i, j, k), vec4(sl, si, sr, ss));
        float uc = 0.5 * (st.c[OFF_U + c] + st.c[OFF_U + idx(i + 1, j, k)]) + uMove.x;
        float vc = 0.5 * (st.c[OFF_V + c] + st.c[OFF_V + idx(i, j + 1, k)]) + uMove.y;
        float wc = 0.5 * (st.c[OFF_W + c] + st.c[OFF_W + idx(i, j, k + 1)]);
        imageStore(imgAux, ivec3(i, j, k), vec4(tauMid, uc, vc, wc));
        colMaxU = max(colMaxU, abs(st.c[OFF_U + c]));
        colMaxV = max(colMaxV, abs(st.c[OFF_V + c]));
        float wk = st.c[OFF_W + idx(i, j, k + 1)];
        colMaxW = max(colMaxW, abs(wk));
        wmax = max(wmax, wk); wmin = min(wmin, wk);
        float cond = qc[k] + qi[k];
        if (cond > 1e-5) { top = max(top, Z0 + (float(k) + 1.0) * DZ); bot = min(bot, Z0 + float(k) * DZ); }
        lwp += rho * cond * DZ;
        // non-inductive charging needs graupel/snow meeting cloud water or
        // ice in an updraught, between about -5 C and -40 C
        float tcel = t - T0K;
        if (tcel < -5.0 && tcel > -40.0 && wc > 1.0) {
            float ch = rho * qs[k] * cond * 1e6;
            charge += ch * DZ;
            if (ch > chargeMax) { chargeMax = ch; chargeZ = Z0 + (float(k) + 0.5) * DZ; }
        }
        water += rho * (qv[k] + qc[k] + qr[k] + qi[k] + qs[k]) * DZ;
        icesum += qi[k]; condsum += cond;
        if (sr + ss > PRECIP_VISIBLE) pbot = Z0 + float(k) * DZ;
    }
    int d = cc * NDIAG;
    diag.c[d + 0] = colMaxU; diag.c[d + 1] = colMaxV; diag.c[d + 2] = colMaxW;
    diag.c[d + 3] = wmax; diag.c[d + 4] = wmin; diag.c[d + 5] = top; diag.c[d + 6] = bot;
    diag.c[d + 7] = lwp; diag.c[d + 8] = water; diag.c[d + 9] = icesum; diag.c[d + 10] = condsum;
    diag.c[d + 11] = maps.c[M_RATE + cc];
    diag.c[d + 12] = charge;
    diag.c[d + 13] = chargeZ;
    diag.c[d + 14] = pbot;
}
"""

# Optical depth from each ground column to the Sun (CPUModel.sun_optical_depth)
SUNTAU = r"""
layout(local_size_x = LOCAL_X) in;
layout(std430, binding = 4) buffer Maps { float c[]; } maps;
uniform sampler3D tOpt;
uniform vec3 uSunDir;          // east, north, up
void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y);
    if (i >= NX || j >= NY) return;
    float tau = 0.0;
    for (int k = 0; k < NZ; ++k) {
        float z = Z0 + (float(k) + 0.5) * DZ;
        float shx = z * uSunDir.x / uSunDir.z / DX;
        float shy = z * uSunDir.y / uSunDir.z / DY;
        int ix = int(floor(shx)), iy = int(floor(shy));
        float fx = shx - float(ix), fy = shy - float(iy);
        float a = dot(texelFetch(tOpt, ivec3((i + ix) & MX, (j + iy) & MY, k), 0).xy, vec2(1.0));
        float b = dot(texelFetch(tOpt, ivec3((i + ix + 1) & MX, (j + iy) & MY, k), 0).xy, vec2(1.0));
        float c = dot(texelFetch(tOpt, ivec3((i + ix) & MX, (j + iy + 1) & MY, k), 0).xy, vec2(1.0));
        float d = dot(texelFetch(tOpt, ivec3((i + ix + 1) & MX, (j + iy + 1) & MY, k), 0).xy, vec2(1.0));
        tau += ((1.0 - fx) * (1.0 - fy) * a + fx * (1.0 - fy) * b + (1.0 - fx) * fy * c + fx * fy * d) * DZ / uSunDir.z;
    }
    maps.c[M_SUNTAU + cidx(i, j)] = tau;
}
"""

SURFACE = r"""
layout(local_size_x = LOCAL_X) in;
layout(std430, binding = 0) buffer St { float c[]; } st;
layout(std430, binding = 3) readonly buffer Base { float c[]; } base;
layout(std430, binding = 4) buffer Maps { float c[]; } maps;
uniform float uDt;
uniform int uMode;            // 1 sun, 2 fixed, 3 bulk over a surface of known temperature
uniform float uSw;            // clear-sky shortwave, W m^-2 (0 at night)
uniform float uBowen;
uniform float uSwAbs;         // share of the sunlight the ground absorbs
uniform float uHet;
uniform float uFixedWth, uFixedWqv;
uniform float uThS, uQsS, uCh;
uniform vec2 uMove;
float B(int off, int k) { return base.c[off + k]; }
void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y);
    if (i >= NX || j >= NY) return;
    int cc = cidx(i, j);
    float wth, wqv;
    if (uMode == 2) { wth = uFixedWth; wqv = uFixedWqv; }
    else if (uMode == 3) {
        float uc = 0.5 * (st.c[OFF_U + idx(i, j, 0)] + st.c[OFF_U + idx(i + 1, j, 0)]) + uMove.x;
        float vc = 0.5 * (st.c[OFF_V + idx(i, j, 0)] + st.c[OFF_V + idx(i, j + 1, 0)]) + uMove.y;
        float spd = sqrt(uc * uc + vc * vc + 1.0);
        float th1 = B(B_TH0, 0) + st.c[OFF_TH + idx(i, j, 0)];
        wth = uCh * spd * (uThS - th1) * B(B_PI0, 0);
        wqv = uCh * spd * (uQsS - st.c[OFF_QV + idx(i, j, 0)]);
    }
    else {
        float rn;
        if (uSw <= 0.0) rn = -70.0;
        else rn = uSwAbs * uSw / (1.0 + 0.75 * 0.14 * maps.c[M_SUNTAU + cc]) - 70.0;
        bool day = rn > 0.0;
        float avail = day ? 0.9 * rn : 0.8 * rn;
        float h = day ? avail * uBowen / (1.0 + uBowen) : avail;
        float le = day ? avail / (1.0 + uBowen) : 0.0;
        float het = 1.0 + uHet * maps.c[M_HET + cc];
        float rhos = B(B_RHO0, 0);
        wth = h * het / (rhos * CP);
        wqv = le * het / (rhos * (LV1 - LV2 * B(B_T0, 0)));
    }
    int c = idx(i, j, 0);
    st.c[OFF_TH + c] += uDt * wth / (DZ * B(B_PI0, 0));
    st.c[OFF_QV + c] += uDt * wqv / DZ;
}
"""

REDUCE = r"""
layout(local_size_x = 256) in;
layout(std430, binding = 7) buffer Diag { float c[]; } diag;
shared float sh[256 * 13];
void main() {
    uint t = gl_LocalInvocationID.x;
    float r[13];
    r[0] = 0.0; r[1] = 0.0; r[2] = 0.0; r[3] = -1e9; r[4] = 1e9; r[5] = 0.0; r[6] = 1e9;
    r[7] = 0.0; r[8] = 0.0; r[9] = 0.0; r[10] = 0.0; r[11] = 0.0; r[12] = 1e9;
    float cover = 0.0;
    for (uint col = t; col < uint(NX * NY); col += 256u) {
        uint d = col * uint(NDIAG);
        r[0] = max(r[0], diag.c[d]); r[1] = max(r[1], diag.c[d + 1]); r[2] = max(r[2], diag.c[d + 2]);
        r[3] = max(r[3], diag.c[d + 3]); r[4] = min(r[4], diag.c[d + 4]);
        r[5] = max(r[5], diag.c[d + 5]); r[6] = min(r[6], diag.c[d + 6]);
        cover += diag.c[d + 7] > 0.02 ? 1.0 : 0.0;
        r[8] += diag.c[d + 8]; r[9] += diag.c[d + 9]; r[10] += diag.c[d + 10];
        r[11] = max(r[11], diag.c[d + 11]);
        r[12] = min(r[12], diag.c[d + 14]);
    }
    r[7] = cover;
    for (int q = 0; q < 13; ++q) sh[t * 13u + uint(q)] = r[q];
    memoryBarrierShared();
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            uint a = t * 13u, b = (t + s) * 13u;
            sh[a + 0] = max(sh[a + 0], sh[b + 0]); sh[a + 1] = max(sh[a + 1], sh[b + 1]);
            sh[a + 2] = max(sh[a + 2], sh[b + 2]); sh[a + 3] = max(sh[a + 3], sh[b + 3]);
            sh[a + 4] = min(sh[a + 4], sh[b + 4]); sh[a + 5] = max(sh[a + 5], sh[b + 5]);
            sh[a + 6] = min(sh[a + 6], sh[b + 6]); sh[a + 7] += sh[b + 7];
            sh[a + 8] += sh[b + 8]; sh[a + 9] += sh[b + 9]; sh[a + 10] += sh[b + 10];
            sh[a + 11] = max(sh[a + 11], sh[b + 11]);
            sh[a + 12] = min(sh[a + 12], sh[b + 12]);
        }
        memoryBarrierShared();
        barrier();
    }
    if (t == 0u) for (int q = 0; q < 13; ++q) diag.c[uint(NX * NY * NDIAG) + uint(q)] = sh[q];
}
"""

# Horizontal means of theta_l and q_t on every level, for the nudging:
# one workgroup per level (CPUModel.mean_thl_qt)
PROFILE = r"""
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer St { float c[]; } st;
layout(std430, binding = 3) readonly buffer Base { float c[]; } base;
layout(std430, binding = 6) buffer Prof { float c[]; } prof;
shared vec4 s1[256];
float B(int off, int k) { return base.c[off + k]; }
void main() {
    int k = int(gl_WorkGroupID.x);
    uint t = gl_LocalInvocationID.x;
    float pi0 = B(B_PI0, k);
    float lv = B(B_LVREF, k), ls = B(B_LSREF, k);
    vec4 a = vec4(0.0);
    for (int c = int(t); c < NX * NY; c += 256) {
        int id = k * NX * NY + c;
        float qc = st.c[OFF_QC + id], qi = st.c[OFF_QI + id];
        a += vec4(st.c[OFF_TH + id] - (lv * qc + ls * qi) / (CP * pi0),
                  st.c[OFF_QV + id] + qc + qi, st.c[OFF_U + id], st.c[OFF_V + id]);
    }
    s1[t] = a;
    memoryBarrierShared();
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) s1[t] += s1[t + s];
        memoryBarrierShared();
        barrier();
    }
    if (t == 0u) {
        vec4 m = s1[0] / float(NX * NY);
        prof.c[4 * k] = B(B_TH0, k) + m.x;
        prof.c[4 * k + 1] = m.y;
        prof.c[4 * k + 2] = m.z;
        prof.c[4 * k + 3] = m.w;
    }
}
"""

NDIAG = 15


# ------------------------------------------------------------ the model ----
class GPUModel:
    """Same physics and interface as crm.CPUModel, on the GPU."""

    def __init__(self, ctx: moderngl.Context, sc: crm.Scenario, init: crm.CPUModel | None = None):
        self.ctx = ctx
        self.sc = sc
        self.g = sc.grid
        g = self.g
        for n in (g.nx, g.ny):
            if n & (n - 1):
                raise ValueError("GPU model needs power-of-two horizontal sizes")
        self.ref = init if init is not None else crm.CPUModel(sc, dtype="f8")
        self._build_layout()
        self._compile()
        self._alloc()
        self.upload_state(self.ref)
        self.time = self.ref.time
        self.steps = 0
        self.sun_elev = 0.9
        self.sun_az = 3.14
        self._diag_cache = None

    # ------------------------------------------------------------ setup --
    def _build_layout(self):
        g, r = self.g, self.ref
        nz = g.nz
        arrays = [
            ("B_RHO0", r.b.rho0), ("B_RHOF", r.b.rhof), ("B_TH0", r.b.th0), ("B_QV0", r.b.qv0),
            ("B_PI0", r.b.pi0), ("B_P0", r.b.p0), ("B_T0", r.b.t0),
            ("B_U0", r.u0[:, 0, 0]), ("B_V0", r.v0[:, 0, 0]),
            ("B_UG0", r.ug0[:, 0, 0]), ("B_VG0", r.vg0[:, 0, 0]),
            ("B_DAMPC", r.damp_c[:, 0, 0]), ("B_DAMPF", r.damp_f[:, 0, 0]),
            ("B_PA", r.pa), ("B_PC", r.pc), ("B_PBZ", r.pb_z),
            ("B_ALO", r.th0_alo), ("B_AHI", r.th0_ahi), ("B_DZC", r.th0_d),
        ]
        zc = g.zc
        sc = self.sc
        rad = np.array([sc.rad_cool(z) for z in zc]) if sc.rad_cool else np.zeros(nz)
        madv = np.array([sc.moist_adv(z) for z in zc]) if sc.moist_adv else np.zeros(nz)
        wls = np.array([sc.subsidence(z) for z in zc]) if sc.subsidence else np.zeros(nz)
        arrays += [("B_RAD", rad), ("B_MADV", madv), ("B_WLS", wls)]
        arrays += [("B_LVREF", r.lv_ref), ("B_LSREF", r.ls_ref), ("B_THLT", r.thl_target),
                   ("B_QTT", r.qt_target), ("B_NUDGE", r.nudge_mask)]
        offs = {}
        chunks = []
        pos = 0
        for name, arr in arrays:
            a = np.asarray(arr, "f8").ravel()
            offs[name] = pos
            chunks.append(a)
            pos += a.size
        self.base_offsets = offs
        self.base_data = np.concatenate(chunks).astype("f4")
        ncol = g.nx * g.ny
        self.map_offsets = {"M_LAT": 0, "M_HET": ncol, "M_RAIN": 2 * ncol, "M_RATE": 3 * ncol,
                            "M_SUNTAU": 4 * ncol}
        maps = np.zeros(5 * ncol, "f4")
        if r.lat_sponge is not None:
            maps[0:ncol] = r.lat_sponge.ravel()
        maps[ncol:2 * ncol] = r.het.ravel()
        self.maps_init = maps

    def _header(self, extra=""):
        g = self.g
        d = {"NX": g.nx, "NY": g.ny, "NZ": g.nz, "DX": f"{g.dx:.8f}", "DY": f"{g.dy:.8f}",
             "DZ": f"{g.dz:.8f}", "Z0": f"{g.z0:.8f}", "LOCAL_X": LOCAL_X, "NDIAG": NDIAG,
             "LW_ICE_RATIO": crm.LW_ICE_RATIO,
             "G": crm.G, "REPSM1": crm.REPSM1, "EPS": crm.EPS, "CP": crm.CP, "RV": crm.RV,
             "T0K": crm.T0K, "LV1": crm.LV1, "LV2": crm.LV2, "LS1": crm.LS1, "LS2": crm.LS2,
             "RHOW": crm.RHOW, "RHOI": crm.RHOI, "K_AUTO": crm.K_AUTO, "Q_AUTO": crm.Q_AUTO,
             "K_ACCR": crm.K_ACCR, "VR_COEF": crm.VR_COEF, "VR_EXP": crm.VR_EXP,
             "N0R": crm.N0R, "N0S": crm.N0S, "N0G": crm.N0G, "RHO_S": crm.RHO_S,
             "RHO_G": crm.RHO_G, "B_S": crm.B_S, "B_G": crm.B_G,
             "E_CLOUD_BY_ICE_PRECIP": crm.E_CLOUD_BY_ICE_PRECIP,
             "E_ICE_BY_ICE_PRECIP": crm.E_ICE_BY_ICE_PRECIP, "QI_AUTO": crm.QI_AUTO, "PRECIP_VISIBLE": crm.PRECIP_VISIBLE,
             "K_AUTO_ICE": crm.K_AUTO_ICE, "RHO_REF_ICE": crm.RHO_REF_ICE,
             "BIGG_A": crm.BIGG_A, "BIGG_B": crm.BIGG_B, "K_AIR": crm.K_AIR, "DV0": crm.DV0,
             "F_VENT": crm.F_VENT, "R_EFF_LIQ": crm.R_EFF_LIQ, "R_EFF_ICE": crm.R_EFF_ICE,
             "VS_COEF": crm.VS_COEF, "VG_COEF": crm.VG_COEF, "CS_COEF": crm.CS_COEF,
             "CG_COEF": crm.CG_COEF}
        lines = ["#version 430"]
        for k, v in d.items():
            if isinstance(v, float):
                v = repr(float(v))
                if "e" in v or "E" in v:
                    v = f"{float(v):.10e}"
                if "." not in v and "e" not in v:
                    v += ".0"
            lines.append(f"#define {k} {v}")
        for k, v in self.base_offsets.items():
            lines.append(f"#define {k} {v}")
        for k, v in self.map_offsets.items():
            lines.append(f"#define {k} {v}")
        lines.append(extra)
        return "\n".join(lines) + "\n" + COMMON

    def _compile(self):
        ctx = self.ctx
        h = self._header()
        full = BUFFERS + DIVFUN
        self.k_scal = ctx.compute_shader(h + full + TEND_SCALARS)
        self.k_u = ctx.compute_shader(h + full + TEND_U)
        self.k_v = ctx.compute_shader(h + full + TEND_V)
        self.k_w = ctx.compute_shader(h + full + TEND_W)
        self.k_div = ctx.compute_shader(h + DIVERGENCE)
        g = self.g

        def fft_src(n, along):
            return self._header(f"#define LEN {n}\n#define HALFN {n // 2}\n#define LOG2LEN {int(math.log2(n))}\n"
                                + ("#define ALONG_X\n" if along == "x" else "")) + FFT
        self.k_fftx = ctx.compute_shader(fft_src(g.nx, "x"))
        self.k_ffty = ctx.compute_shader(fft_src(g.ny, "y"))
        self.k_tri = ctx.compute_shader(h + TRIDIAG)
        self.k_proj = ctx.compute_shader(h + PROJECT)
        self.k_micro = ctx.compute_shader(h + MICRO)
        self.k_suntau = ctx.compute_shader(h + SUNTAU)
        self.k_surface = ctx.compute_shader(h + SURFACE)
        self.k_reduce = ctx.compute_shader(h + REDUCE)
        self.k_profile = ctx.compute_shader(h + PROFILE)

    def _alloc(self):
        ctx, g = self.ctx, self.g
        n = g.nx * g.ny * g.nz
        nw = g.nx * g.ny * (g.nz + 1)
        self.state_size = 8 * n + nw
        nbytes = self.state_size * 4
        self.buf = [ctx.buffer(reserve=nbytes) for _ in range(3)]
        self.b_base = ctx.buffer(self.base_data.tobytes())
        self.b_maps = ctx.buffer(self.maps_init.tobytes())
        self.b_fft = ctx.buffer(reserve=n * 8)
        self.b_scratch = ctx.buffer(reserve=n * 4)
        self.b_diag = ctx.buffer(reserve=(g.nx * g.ny * NDIAG + 16) * 4)
        self.b_prof = ctx.buffer(reserve=max(4 * g.nz, 4) * 4)
        size = (g.nx, g.ny, g.nz)
        self.t_opt = [ctx.texture3d(size, 4, dtype="f2") for _ in range(2)]
        self.t_aux = [ctx.texture3d(size, 4, dtype="f2") for _ in range(2)]
        for t in self.t_opt + self.t_aux:
            t.repeat_x = True
            t.repeat_y = True
            t.repeat_z = False
        for t in self.t_opt:
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        for t in self.t_aux:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.cur_tex = 0                 # index of the most recently written optics
        self.i0 = 0                      # buffer index holding the state at time n

    # ------------------------------------------------------------ state --
    def pack(self, m: crm.CPUModel) -> np.ndarray:
        parts = [m.u, m.v, m.w, m.th, m.qv, m.qc, m.qr, m.qi, m.qs]
        return np.concatenate([np.ascontiguousarray(p, "f4").ravel() for p in parts])

    def unpack(self, data: np.ndarray) -> dict:
        g = self.g
        n = g.nx * g.ny * g.nz
        nw = g.nx * g.ny * (g.nz + 1)
        out = {}
        pos = 0
        for name in ("u", "v", "w", "th", "qv", "qc", "qr", "qi", "qs"):
            size = nw if name == "w" else n
            shp = (g.nz + 1 if name == "w" else g.nz, g.ny, g.nx)
            out[name] = data[pos:pos + size].reshape(shp)
            pos += size
        return out

    def upload_state(self, m: crm.CPUModel):
        self.buf[self.i0].write(self.pack(m).tobytes())
        rain = np.zeros(self.g.nx * self.g.ny, "f4")
        ncol = self.g.nx * self.g.ny
        self.b_maps.write(rain.tobytes(), offset=self.map_offsets["M_RAIN"] * 4)

    def download_state(self) -> dict:
        data = np.frombuffer(self.buf[self.i0].read(), "f4")
        return self.unpack(data)

    def rain_accum(self) -> np.ndarray:
        g = self.g
        ncol = g.nx * g.ny
        d = np.frombuffer(self.b_maps.read(size=ncol * 4, offset=self.map_offsets["M_RAIN"] * 4), "f4")
        return d.reshape(g.ny, g.nx)

    # ------------------------------------------------------------- step --
    def _groups3(self, nzp=0):
        g = self.g
        return ((g.nx + LOCAL_X - 1) // LOCAL_X, g.ny, g.nz + nzp)

    def _groups2(self):
        g = self.g
        return ((g.nx + LOCAL_X - 1) // LOCAL_X, g.ny, 1)

    def _bind_dyn(self, cur, s0, out):
        cur.bind_to_storage_buffer(0)
        s0.bind_to_storage_buffer(1)
        out.bind_to_storage_buffer(2)
        self.b_base.bind_to_storage_buffer(3)
        self.b_maps.bind_to_storage_buffer(4)
        self.b_fft.bind_to_storage_buffer(5)
        self.b_scratch.bind_to_storage_buffer(6)
        self.b_diag.bind_to_storage_buffer(7)

    @staticmethod
    def _set(prog, name, value):
        if name in prog:
            prog[name].value = value

    def _barrier(self):
        self.ctx.memory_barrier()

    def step(self, dt: float):
        g = self.g
        sc = self.sc
        um, vm = sc.translate
        cd = float(self.ref.cd)
        i0 = self.i0
        ia, ib = (i0 + 1) % 3, (i0 + 2) % 3
        stages = [(i0, ia, 1.0 / 3.0), (ia, ib, 0.5), (ib, ia, 1.0)]
        for cur, out, frac in stages:
            dts = dt * frac
            self._bind_dyn(self.buf[cur], self.buf[i0], self.buf[out])
            for prog in (self.k_scal, self.k_u, self.k_v, self.k_w):
                self._set(prog, "uDts", dts)
                self._set(prog, "uLateral", 1 if sc.lateral_sponge else 0)
                self._set(prog, "uDrag", 1 if cd > 0.0 else 0)
                self._set(prog, "uCd", cd)
                self._set(prog, "uCor", float(sc.coriolis))
                self._set(prog, "uMove", (float(um), float(vm)))
            self.k_scal.run(*self._groups3())
            self.k_u.run(*self._groups3())
            self.k_v.run(*self._groups3())
            self.k_w.run(*self._groups3(1))
            self._barrier()
            self._set(self.k_div, "uDts", dts)
            self.k_div.run(*self._groups3())
            self._barrier()
            self._set(self.k_fftx, "uSign", -1.0)
            self.k_fftx.run(g.ny, g.nz, 1)
            self._barrier()
            self._set(self.k_ffty, "uSign", -1.0)
            self.k_ffty.run(g.nx, g.nz, 1)
            self._barrier()
            self.k_tri.run(*self._groups2())
            self._barrier()
            self._set(self.k_ffty, "uSign", 1.0)
            self.k_ffty.run(g.nx, g.nz, 1)
            self._barrier()
            self._set(self.k_fftx, "uSign", 1.0)
            self.k_fftx.run(g.ny, g.nz, 1)
            self._barrier()
            self._set(self.k_proj, "uDts", dts)
            self.k_proj.run(*self._groups3(1))
            self._barrier()
        # the new state is in buffer ia; make it the state at time n
        self.i0 = ia
        st = self.buf[ia]
        # microphysics
        st.bind_to_storage_buffer(0)
        self.b_base.bind_to_storage_buffer(3)
        self.b_maps.bind_to_storage_buffer(4)
        self.b_diag.bind_to_storage_buffer(7)
        self.cur_tex ^= 1
        self.t_opt[self.cur_tex].bind_to_image(0, read=False, write=True)
        self.t_aux[self.cur_tex].bind_to_image(1, read=False, write=True)
        lw = sc.lw_f0 > 0.0 or sc.lw_f1 > 0.0
        forcing = ((1 if sc.rad_cool else 0) | (2 if sc.moist_adv else 0)
                   | (4 if sc.subsidence else 0) | (8 if lw else 0))
        if sc.nudge_tau > 0.0:
            st.bind_to_storage_buffer(0)
            self.b_base.bind_to_storage_buffer(3)
            self.b_prof.bind_to_storage_buffer(6)
            self.k_profile.run(g.nz, 1, 1)
            self._barrier()
        self.b_prof.bind_to_storage_buffer(6)
        km = self.k_micro
        self._set(km, "uNudgeTau", float(sc.nudge_tau))
        self._set(km, "uNudgeUV", 1 if sc.nudge_uv else 0)
        self._set(km, "uDt", dt)
        self._set(km, "uIce", 1 if sc.micro_ice else 0)
        self._set(km, "uForcing", forcing)
        self._set(km, "uMove", (float(um), float(vm)))
        self._set(km, "uQAuto", float(sc.q_auto))
        self._set(km, "uQiAuto", float(sc.qi_auto))
        self._set(km, "uGraupel", 1.0 if sc.graupel else 0.0)
        self._set(km, "uLwF0", float(sc.lw_f0))
        self._set(km, "uLwF1", float(sc.lw_f1))
        self._set(km, "uLwKappa", float(sc.lw_kappa))
        self._set(km, "uLwDiv", float(sc.lw_div))
        self._set(km, "uLwZiQt", float(sc.lw_zi_qt))
        self._set(km, "uLwRhoi", float(sc.lw_rhoi))
        km.run(*self._groups2())
        self._barrier()
        self.t_opt[self.cur_tex].build_mipmaps()
        if sc.flux_mode in ("sun", "fixed", "bulk") and g.z0 <= 0.0:
            if sc.flux_mode == "sun":
                el = max(self.sun_elev, 0.05)
                sd = (math.sin(self.sun_az) * math.cos(el), math.cos(self.sun_az) * math.cos(el),
                      math.sin(el))
                self.t_opt[self.cur_tex].use(0)
                self._set(self.k_suntau, "tOpt", 0)
                self._set(self.k_suntau, "uSunDir", tuple(float(x) for x in sd))
                self.b_maps.bind_to_storage_buffer(4)
                self.k_suntau.run(*self._groups2())
                self._barrier()
            se = math.sin(max(self.sun_elev, 0.0))
            sw = 1098.0 * se * math.exp(-0.057 / se) if se > 0.01 else 0.0
            st.bind_to_storage_buffer(0)
            self.b_base.bind_to_storage_buffer(3)
            self.b_maps.bind_to_storage_buffer(4)
            k = self.k_surface
            self._set(k, "uDt", dt)
            self._set(k, "uMode", {"sun": 1, "fixed": 2, "bulk": 3}[sc.flux_mode])
            self._set(k, "uSw", float(sw))
            self._set(k, "uBowen", float(sc.bowen))
            self._set(k, "uSwAbs", float(sc.sw_absorb))
            self._set(k, "uHet", float(sc.heterogeneity))
            self._set(k, "uFixedWth", float(sc.fixed_wth))
            self._set(k, "uFixedWqv", float(sc.fixed_wqv))
            if sc.flux_mode == "bulk":
                zc0 = 0.5 * g.dz
                self._set(k, "uCh", float((0.4 / math.log((zc0 + sc.z0) / sc.z0)) ** 2))
                self._set(k, "uThS", float(sc.sst * (crm.P00 / sc.base.psfc) ** (crm.RD / crm.CP)))
                self._set(k, "uQsS", float(crm.qsat_liq(np.float64(sc.sst), sc.base.psfc)))
                self._set(k, "uMove", (float(um), float(vm)))
            k.run(*self._groups2())
            self._barrier()
        self.time += dt
        self.steps += 1
        self._diag_cache = None

    # ------------------------------------------------------ diagnostics --
    def reduce(self) -> np.ndarray:
        self.b_diag.bind_to_storage_buffer(7)
        self.k_reduce.run(1, 1, 1)
        self._barrier()
        g = self.g
        off = g.nx * g.ny * NDIAG * 4
        return np.frombuffer(self.b_diag.read(size=13 * 4, offset=off), "f4").astype("f8")

    def max_dt(self, courant: float = 0.9, dt_max: float | None = None) -> float:
        r = self.reduce()
        g = self.g
        cu = r[0] / g.dx + r[1] / g.dy + r[2] / g.dz
        dtl = 10.0                                                  # as CPUModel.max_dt
        if dt_max is not None:
            dtl = min(dtl, dt_max)
        return float(min(courant / max(cu, 1e-6), dtl))

    def diagnostics(self) -> dict:
        r = self.reduce()
        g = self.g
        ncol = g.nx * g.ny
        return {"time": self.time, "w_max": float(r[3]), "w_min": float(r[4]),
                "cloud_top": float(r[5]), "cloud_base": float(r[6]) if r[6] < 1e8 else 0.0,
                "cloud_cover": float(r[7] / ncol), "rain_rate_max": float(r[11] * 3600.0),
                "rain_total": float(self.rain_accum().mean()),
                "water": float(r[8] * g.dx * g.dy),
                "ice_frac": float(r[9] / max(r[10], 1e-20)),
                "precip_bottom": float(r[12]) if r[12] < 1e8 else -1.0}

    def column_diag(self) -> np.ndarray:
        g = self.g
        d = np.frombuffer(self.b_diag.read(size=g.nx * g.ny * NDIAG * 4), "f4")
        return d.reshape(g.ny, g.nx, NDIAG)

    def optics_textures(self):
        """(previous, current) optics and aux textures for the renderer."""
        return (self.t_opt[self.cur_tex ^ 1], self.t_opt[self.cur_tex],
                self.t_aux[self.cur_tex ^ 1], self.t_aux[self.cur_tex])

    def release(self):
        for b in self.buf + [self.b_base, self.b_maps, self.b_fft, self.b_scratch, self.b_diag,
                             self.b_prof]:
            b.release()
        for t in self.t_opt + self.t_aux:
            t.release()
