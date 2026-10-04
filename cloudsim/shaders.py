# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
GLSL for the sky: the atmosphere, the Atlas cloud decks, the simulated
cloud field, the Sun's shadow map, lightning, and the display pipeline.

Everything here needs only OpenGL 3.3; the cloud-resolving model's compute
shaders live in crm_gpu.py.
"""

VERT = """#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Shared by the sky pass and the shadow-map pass
# ---------------------------------------------------------------------------
COMMON = """
#define PI 3.14159265359
#define MAXDECKS 12
#define DECKTEX 18
#define MAPN 512.0
#define GAP_SOFT 0.18

uniform sampler2D  tTransmittance;   // (mu, altitude) -> RGB
uniform sampler2D  tMultiScatter;    // (mu_sun, altitude) -> RGB
uniform sampler3D  tShape;           // RGBA perlin-worley
uniform sampler3D  tDetail;          // RGB worley
uniform sampler2D  tBlue;            // dither
uniform sampler2DArray tDeckMaps;    // per deck, three realizations of its pattern (layers
                                     // 3d .. 3d+2): z, z^2, element tops or turret domes, gaps
layout(std140) uniform DeckBlock {   // DECKTEX vec4 per layer group
    vec4 deckP[MAXDECKS * DECKTEX];
};
uniform sampler2D  tShadow;          // Beer shadow map: front depth, mean ext, max tau
uniform sampler2D  tRainPhase;       // Mie phase functions, RGB, theta = pi u^2: row 0 rain,
                                     // rows 1-16 nacreous ice spheres of 0.6-3 um
uniform sampler3D  tSimOptA;         // simulated extinction: liquid, ice, rain, snow
uniform sampler3D  tSimOptB;
uniform sampler3D  tSimAuxA;         // tau above, u, v, w
uniform sampler3D  tSimAuxB;
uniform sampler2DArray tRegionWarp;  // displacement of the model field (east, north, m) at the
                                     // forecast's two hours around the clock (xy, zw); layer 1:
                                     // the same of the forecast before a refresh
uniform sampler2DArray tRegionCover; // the forecast's cover (%) of the low, middle and high
                                     // étage at the two hours (layers 0, 1; 2, 3: the forecast
                                     // before a refresh)
uniform sampler2D  tSimKeep;         // the model's column optical depth, widened to whole cells
uniform int   uKeepMapOn;
uniform float uSimFade;              // how much of the model's field is shown (a handover)

uniform vec3  uSunDir;
uniform vec3  uLightDir;             // what lights the clouds: the Sun, or the Moon at night
uniform vec3  uLightColor;           // 1 for the Sun; the Moon's irradiance relative to it
uniform int   uLightIsSun;
uniform float uObserverAlt;          // metres above the ground
uniform float uGroundElev;           // the ground's height above sea level, m: the air
                                     // above a high station is thinner and cleaner
uniform float uSnow;                 // fraction of the ground under lying snow
uniform float uDry;                  // 0 green .. 1 bare sand, from the soil's water
uniform float uTime;                 // seconds of world time
uniform int   uNumDecks;
uniform float uGroundAlbedo;
uniform float uHaze;                 // aerosol extinction at the ground, m^-1 (from visibility)
uniform vec3  uSkyIrrLow;            // downwelling sky irradiance at the ground
uniform vec3  uSkyIrrHigh;           // ... and at 8 km

// the simulation
uniform int   uSimOn;
uniform float uSimMix;               // 0 = state A, 1 = state B
uniform vec3  uSimSize;              // domain size x, y and top, metres
uniform vec3  uSimCell;              // dx, dy, dz
uniform vec2  uSimObs;               // observer's position in the domain, metres
uniform int   uSimTile;              // periodic continuation: bit 1 east-west, bit 2 north-south
uniform float uSimTime;              // model seconds, animates the sub-grid detail
uniform vec4  uJdELiq;               // Jendersie-d'Eon Mie fit for the model's droplets
uniform float uSimZ0;                // height of the domain's floor above the ground, m
uniform float uSimZLo, uSimZHi;      // heights between which the model has anything to draw
uniform float uSimZBase;             // height of the model's lowest cloud
uniform vec2  uSimMove;              // the domain's drift with the wind, m/s

// the weather around: the satellite's view of where the cloud is
uniform int   uRegionOn;
uniform float uRegionHalf;           // half-width of the regional maps, m
uniform vec4  uReg[8];               // per forecast (now, before a refresh), four rows: the
                                     // clock's fraction of the way between the two hours and
                                     // the cover here now per étage; how far each étage's field
                                     // has been carried since the first hour (A) and will be
                                     // until the second (B): A0 A1, A2 B0, B1 B2
uniform float uRegNew;               // how far a refresh has gone over to the new forecast
uniform int   uRegSimEt;             // the étage of the model's cloud layer
uniform float uTauTh[17];            // column optical depth below which the model's cloud is
                                     // dropped, for forecast/local cover ratio 0, 1/16 .. 1

// the Beer shadow map
uniform int   uShadowOn;
uniform vec3  uShE1;
uniform vec3  uShE2;
uniform float uShExtent;
uniform float uShD0;

const float RG = 6360000.0;          // m
const float RT = 6460000.0;
const vec3  RAY_S = vec3(5.802, 13.558, 33.1) * 1e-6;
const vec3  MIE_S = vec3(3.996) * 1e-6;
const vec3  MIE_E = vec3(4.40) * 1e-6;
const vec3  OZO_A = vec3(0.650, 1.881, 0.085) * 1e-6;
const float RAY_H = 8000.0;
const float MIE_H = 1200.0;
const float MIE_G = 0.80;

// ---------------------------------------------------------------- helpers --
float remap(float x, float a, float b, float c, float d) {
    return c + clamp((x - a) / max(b - a, 1e-6), 0.0, 1.0) * (d - c);
}
float saturate(float x) { return clamp(x, 0.0, 1.0); }

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

vec2 raySphere(vec3 ro, vec3 rd, float rad) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad * rad;
    float h = b * b - c;
    if (h < 0.0) return vec2(-1.0);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

vec3 sampleTransmittance(float r, float mu) {
    float x = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float y = clamp((r - RG) / (RT - RG), 0.0, 1.0);
    return texture(tTransmittance, vec2(x, y)).rgb;
}

vec3 sampleMultiScatter(float r, float mu) {
    float x = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float y = clamp((r - RG) / (RT - RG), 0.0, 1.0);
    return texture(tMultiScatter, vec2(x, y)).rgb;
}

// The ground is a sphere of radius RG + uGroundElev.  Clouds, the model and
// the shadow map work in heights above the ground; the air (density,
// transmittance tables) in heights above the sea, so that a station at
// 5 km has half the atmosphere above it and not all of it.
float groundR() { return RG + uGroundElev; }
vec3 earthCentre() { return vec3(0.0, -(RG + uGroundElev + uObserverAlt), 0.0); }
float altitudeOf(vec3 p) { return length(p - earthCentre()) - groundR(); }

// Sunlight arriving at a point: the atmosphere's transmittance along the
// Sun's ray from THIS point (so a tower's top is still white when its base
// is already orange), and nothing if the Earth is in the way.
// The Earth's shadow: a point at radius r sees the light source only above
// its own geometric horizon, mu > -sqrt(1 - (R/r)^2), R the ground's
// radius.  (Intersecting the
// sphere instead fails for points ON the ground, where rounding puts the
// first intersection a hair in front and shadows the whole landscape.)
bool earthShadowed(float r, float mu) {
    float q = groundR() / max(r, groundR());
    return mu < -sqrt(max(1.0 - q * q, 0.0));
}
vec3 sunAt(vec3 p) {
    vec3 c = p - earthCentre();
    float r = length(c);
    float mu = dot(c / r, uSunDir);
    if (earthShadowed(r, mu)) return vec3(0.0);
    return sampleTransmittance(r, mu);
}
// the same for whatever is lighting the clouds (the Moon at night)
vec3 lightAt(vec3 p) {
    vec3 c = p - earthCentre();
    float r = length(c);
    float mu = dot(c / r, uLightDir);
    if (earthShadowed(r, mu)) return vec3(0.0);
    return sampleTransmittance(r, mu) * uLightColor;
}

vec3 skyIrradiance(float alt) {
    return mix(uSkyIrrLow, uSkyIrrHigh, saturate(alt / 8000.0));
}

// ---------------------------------------------------------------- phases --
float phaseRayleigh(float c) { return 3.0 / (16.0 * PI) * (1.0 + c * c); }

// 1 + g^2 - 2 g c written as (1-g)^2 + 2 g (1-c): the textbook form loses
// everything to cancellation in single precision when g is 0.99
float phaseHG(float c, float g) {
    float den = (1.0 - g) * (1.0 - g) + 2.0 * g * (1.0 - c);
    return (1.0 - g * g) / (4.0 * PI * den * sqrt(den));
}

float phaseDraine(float c, float g, float a) {
    float den = (1.0 - g) * (1.0 - g) + 2.0 * g * (1.0 - c);
    return (1.0 - g * g) * (1.0 + a * c * c)
         / (4.0 * PI * (1.0 + a * (1.0 + 2.0 * g * g) / 3.0) * den * sqrt(den));
}

// Jendersie & d'Eon (2023): Mie scattering by cloud droplets of a given
// diameter as a blend of a very sharp Henyey-Greenstein lobe (diffraction)
// and a Draine lobe.  P = (gHG, gD, alpha, wD), computed on the CPU.
float phaseJdE(float c, vec4 P) {
    return mix(phaseHG(c, P.x), phaseDraine(c, P.y, P.z), P.w);
}

// Rain: the Mie phase function averaged over a Marshall-Palmer spectrum and
// the visible spectrum (mie.py).  This is what holds the rainbows.
vec3 phaseRain(float c) {
    float th = acos(clamp(c, -1.0, 1.0));
    float u = sqrt(th / PI);
    return texture(tRainPhase, vec2(u, 0.5 / 17.0)).rgb;
}
// Nearly equal ice spheres of mean radius a (0.6-3 um): Mie theory at each
// wavelength, the nacreous cloud's colours (mie.build_psc_phase).
vec3 phasePsc(float c, float a) {
    float th = acos(clamp(c, -1.0, 1.0));
    float u = sqrt(th / PI);
    float row = clamp((a - 0.6) / 2.4 * 15.0, 0.0, 15.0);
    return texture(tRainPhase, vec2(u, (1.5 + row) / 17.0)).rgb;
}

// ------------------------------------------------------ Beer shadow map ----
// Optical depth from the Sun to x through every cloud, as Hillaire's Beer
// Shadow Map stores it: tau = min(maxTau, meanExt * max(0, depth - front)).
float shadowTau(vec3 x) {
    if (uShadowOn == 0) return 0.0;
    vec2 uv = vec2(dot(x, uShE1), dot(x, uShE2)) / (2.0 * uShExtent) + 0.5;
    if (uv.x <= 0.0 || uv.y <= 0.0 || uv.x >= 1.0 || uv.y >= 1.0) return 0.0;
    vec3 s = texture(tShadow, uv).rgb;
    float d = uShD0 - dot(x, uLightDir);
    return min(s.b, s.g * max(0.0, d - s.r));
}
float cloudShadowT(vec3 x) { return exp(-shadowTau(x)); }

// ---------------------------------------------------------------- region --
// The forecast's cloud around the place, on maps centred on the observer.
vec2 regionUV(vec2 xz) { return xz / (2.0 * uRegionHalf) + 0.5; }
// Where the model's field is continued beyond its own domain, it is
// continued displaced: the air at a place tens of kilometres away has been
// carried by a different wind for the last hours (the displacement is that
// difference times the time, zero here).  The field stays continuous and
// never repeats in view.
vec2 regionWarp(vec2 xz) {
    if (uRegionOn == 0) return vec2(0.0);
    vec2 uv = regionUV(xz);
    vec4 w = texture(tRegionWarp, vec3(uv, 0.0));
    vec2 r = mix(w.xy, w.zw, uReg[0].x);
    if (uRegNew < 0.999) {
        vec4 wo = texture(tRegionWarp, vec3(uv, 1.0));
        r = mix(mix(wo.xy, wo.zw, uReg[4].x), r, uRegNew);
    }
    return r;
}
// The forecast's cover of étage e at xz now, relative to the cover here: the
// field of the hour before carried along the étage's wind for the time since
// that hour and the field of the hour after carried back for the time until
// it, blended by time (advection-corrected interpolation, region.py) - so a
// band of cloud travels across the map, and the map changes continuously
// with the clock, not once a minute.
float regionEtageSet(vec2 xz, int e, int k) {
    vec4 h = uReg[k];
    vec2 sa = (e == 0) ? uReg[k + 1].xy : ((e == 1) ? uReg[k + 1].zw : uReg[k + 2].xy);
    vec2 sb = (e == 0) ? uReg[k + 2].zw : ((e == 1) ? uReg[k + 3].xy : uReg[k + 3].zw);
    float lay = (k == 0) ? 0.0 : 2.0;
    float a = texture(tRegionCover, vec3(regionUV(xz - sa), lay))[e];
    float b = texture(tRegionCover, vec3(regionUV(xz - sb), lay + 1.0))[e];
    float here = (e == 0) ? h.y : ((e == 1) ? h.z : h.w);
    return clamp(mix(a, b, h.x) / here, 0.0, 1.5);
}
float regionEtage(vec2 xz, int e) {
    if (uRegionOn == 0) return 1.0;
    float r = regionEtageSet(xz, e, 0);
    if (uRegNew < 0.999) r = mix(regionEtageSet(xz, e, 4), r, uRegNew);
    return r;
}
// A deck takes its étage's cover; across the boundaries of the étages (2 and
// 7 km) the two étages' are blended over 600 m and 1 km of base height, so a
// layer that rises or sinks through one changes gradually (region.etage_weights).
float regionDeck(vec2 xz, float base) {
    if (uRegionOn == 0) return 1.0;
    float kLM = smoothstep(1700.0, 2300.0, base);
    float kMH = smoothstep(6500.0, 7500.0, base);
    if (kLM <= 0.0) return regionEtage(xz, 0);
    if (kMH >= 1.0) return regionEtage(xz, 2);
    if (kMH > 0.0) return mix(regionEtage(xz, 1), regionEtage(xz, 2), kMH);
    if (kLM >= 1.0) return regionEtage(xz, 1);
    return mix(regionEtage(xz, 0), regionEtage(xz, 1), kLM);
}

// Up to two stretches of a ray inside the spherical shell between the
// heights h0 and h1 (a ray from above a layer can pass through it, under
// it and out through it again), clipped to [0, tmax].  Returns how many.
int shellSegments(vec3 ro, vec3 rd, float h0, float h1, float tmax, out vec4 seg) {
    seg = vec4(0.0);
    vec2 o = raySphere(ro, rd, groundR() + h1);
    if (o.y <= 0.0) return 0;
    vec2 i = raySphere(ro, rd, groundR() + max(h0, 0.0));
    bool hitInner = h0 > 0.0 && i.y > 0.0 && i.x < i.y;
    float a0 = max(o.x, 0.0);
    float a1 = hitInner ? (i.x > 0.0 ? i.x : -1.0) : o.y;
    int n = 0;
    if (a1 > a0 && a0 < tmax) { seg.xy = vec2(a0, min(a1, tmax)); n = 1; }
    if (hitInner) {
        float b0 = max(i.y, 0.0), b1 = min(o.y, tmax);
        if (b1 > b0) {
            if (n == 0) seg.xy = vec2(b0, b1); else seg.zw = vec2(b0, b1);
            n += 1;
        }
    }
    return n;
}

// ------------------------------------------------------------------ decks --
vec4 dp(int d, int i) { return deckP[d * DECKTEX + i]; }

// Castellanus (Atlas: turrets rising from a common horizontal base).  A
// flat, sharp base; the layer's own elements round off at tb; the turrets
// standing on it are solid to 70% of their height and round off to their
// own tops.  tops is the cloud's top there as a fraction of the deck's depth
// (tb on the layer).  The shoulder moves continuously with the top (0.55 tb
// for an element of the layer, tb + 0.7 of the turret above it for a tall
// turret): a switch between the two drew each turret's contours as terraces.
float profileCastellanus(int d, float hf, float tops, float tb) {
    float cum = dp(d, 0).w;
    float strat = smoothstep(0.0, 0.035, hf) * (1.0 - smoothstep(0.75 * tb, tb, hf));
    float bottom = saturate(remap(hf, 0.0, 0.035, 0.0, 1.0));
    float k = 0.7 + 0.45 * tb / max(1.0 - tb, 0.05);
    float shoulder = 0.55 * tb + max(tops - tb, 0.0) * k;
    float top = saturate(remap(hf, shoulder, max(tops, shoulder + 1e-3), 1.0, 0.0));
    return mix(strat, bottom * top, cum);
}

// vertical shape of a deck, hf in 0..1 (below 0 is virga / precipitation)
float profileFor(int d, float hf, float tops) {
    vec4 a = dp(d, 0);                       // base top sigma cumuliformity
    vec4 g = dp(d, 5);                       // anvil cavum fluctus arcus
    float cum = a.w;
    float strat = smoothstep(0.0, 0.10, hf) * (1.0 - smoothstep(0.72, 1.0, hf));
    float bottom = saturate(remap(hf, 0.0, 0.14, 0.0, 1.0));
    // A cloud that carries an anvil stays dense right up to it; an ordinary
    // cumulus rounds off well below its highest turret.
    float shoulder = mix(tops * 0.55, tops * 0.90, g.x);
    float top = saturate(remap(hf, shoulder, tops, 1.0, 0.0));
    float cumul = bottom * top;
    return mix(strat, cumul, cum);
}

// Below the base the cloud has become falling particles.  Virga evaporates
// within a few hundred metres; praecipitatio survives to the ground.  The
// shaft has its own, much lower extinction: the same water carried by a few
// big drops instead of very many small ones extinguishes far less light.
float fallStreakDensity(int d, float metresBelow) {
    vec4 f = dp(d, 4);
    float amount = max(f.x, f.y);
    if (amount <= 0.0 || metresBelow <= 0.0) return 0.0;
    float scaleH = (f.y > 0.5) ? 2500.0 : 320.0;
    float target = (f.y > 0.5) ? 0.0060 : 0.0022;
    float scale = clamp(target / max(dp(d, 0).z, 1e-5), 0.0, 1.0);
    return amount * exp(-metresBelow / scaleH) * scale;
}

// horizontal width of the elements as a function of height: narrow at the
// base, widest in the middle, spreading again in an anvil
float widthFor(int d, float hf) {
    vec4 a = dp(d, 0);
    vec4 g = dp(d, 5);
    float cum = a.w;
    float w = mix(1.0, 0.45 + 0.75 * (1.0 - abs(hf - 0.42) * 1.45), cum);
    w = mix(w, 1.25, smoothstep(0.68, 0.90, hf) * g.x);
    return clamp(w, 0.05, 1.25);
}

// The standard normal distribution function (tanh form, within 3e-4).
float normCdf(float x) {
    // (the argument held to +-10, where tanh is 1 in single precision: a
    // tanh taken as (e^2a - 1)/(e^2a + 1) overflows past a = 44, and a layer
    // at full cover sits 40 units over its threshold)
    return 0.5 + 0.5 * tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0));
}

// A deck's pattern at q (metres, east and north of the eye): three
// realizations blended with weights whose squares sum to one, so the blend
// is exactly standard normal (timeline.py); cloud where it exceeds zstar.
// Each realization's texels carry z and z^2, so a mip level knows how much
// the pattern varies inside it and gives the fraction of it that is cloud,
// Phi((E z - zstar) / sqrt(var + s^2)), rather than thresholding an average
// (LEAN mapping's moments, Olano & Baker 2010).  Returns that fraction;
// tops: the element tops (regular decks) or the tops over the turrets
// (castellanus, absolute); kT: how much of the blend has turrets.
float deckPattern(int d, vec2 q, float pixSize, float zDrop, out float topsReg, out float topsCas,
                  out float kT, out float core) {
    vec4 wz = dp(d, 10);                     // weights of the three slots, zstar
    vec4 ex = dp(d, 11);                     // their tile sizes, soft edge (z units)
    vec4 o01 = dp(d, 12);                    // drift of slots 0 and 1, reduced to their tiles
    vec4 o2g = dp(d, 13);                    // drift of slot 2, gap amount, gap level
    vec4 tu = dp(d, 14);                     // which slots have turrets, common layer top
    vec4 gf = dp(d, 15);                     // which slots have gaps, top variation
    float z = 0.0, var = 0.0, u = 0.0, g = 0.0, gw2 = 0.0, dome = 0.0, lodMax = 0.0;
    float zc = 0.0, varc = 0.0;
    kT = 0.0;
    for (int s = 0; s < 3; ++s) {
        float w = (s == 0) ? wz.x : ((s == 1) ? wz.y : wz.z);
        if (w <= 1e-6) continue;
        float e = (s == 0) ? ex.x : ((s == 1) ? ex.y : ex.z);
        vec2 off = (s == 0) ? o01.xy : ((s == 1) ? o01.zw : o2g.xy);
        float lod = log2(clamp(pixSize / max(e / MAPN, 0.5), 1.0, 512.0));
        vec4 m = textureLod(tDeckMaps, vec3((q - off) / e, float(d * 3 + s)), min(lod, 9.0));
        z += w * m.r;
        var += w * w * max(m.g - m.r * m.r, 0.0);
        if (zDrop > 0.0) {
            // the pattern around: its mean and spread over a coarser texel
            vec4 mc = textureLod(tDeckMaps, vec3((q - off) / e, float(d * 3 + s)),
                                 min(max(lod + 2.0, 4.0), 9.0));
            zc += w * mc.r;
            varc += w * w * max(mc.g - mc.r * mc.r, 0.0);
        }
        float isT = (s == 0) ? tu.x : ((s == 1) ? tu.y : tu.z);
        if (isT > 0.5) { dome = max(dome, min(w * w * 1.125, 1.0) * m.b); kT += w * w; }
        else u += w * m.b;
        float isG = (s == 0) ? gf.x : ((s == 1) ? gf.y : gf.z);
        if (isG > 0.5) { g += w * m.a; gw2 += w * w; }
        lodMax = max(lodMax, lod);
    }
    // undulatus: the layer's waves, one field moving with the layer, added
    // to its evolving pattern (zstar is the level for the pattern and the
    // waves together: noise.zstar_with_wave)
    vec4 wv = dp(d, 17);
    if (wv.x > 0.0) z += wv.x * sin(dot(q, wv.yz) + wv.w);
    float sd = max(ex.w, 1e-4) * 0.3989423;  // a linear ramp of width s rises as a Phi of sd
    // zDrop: a lower threshold, where the cloud spreads wider (an anvil) -
    // only around a tower that rises well above the threshold (the highest
    // of the pattern around, its mean plus 1.5 spreads): a rise of the
    // pattern that stays under the threshold has no tower, and an anvil
    // there floated over clear air
    float zs = wz.w;
    if (zDrop > 0.0)
        zs -= zDrop * smoothstep(wz.w + 0.2, wz.w + 0.9, zc + 1.5 * sqrt(varc));
    float frac = normCdf((z - zs) / sqrt(var + sd * sd));
    // how far inside its element a point is: 0 at the edge, 1 near its
    // middle - the pattern's margin over its threshold (held at -1.2 below,
    // so a sheet at full cover still has its cells, thinner between them).
    // Over 2.2 standard deviations the water thins over the outer half of an
    // element: blurred edges (over 1.2 they were pebbles)
    core = smoothstep(0.0, 2.2, z - max(zs, -1.2));
    // a turret stands on its layer's cloud: where the layer is cloud now, so
    // the turrets thin and sink with the layer and go when it goes (a turret
    // over no cloud popped in the moment the layer's cover rose above zero)
    dome *= smoothstep(0.5, 0.8, frac);
    if (gw2 > 1e-6 && o2g.z > 0.0) {
        // perlucidus's interstices / lacunosus's holes; far away, where a
        // texel holds many of them, the share of the area they take
        float inside = normCdf((g / sqrt(gw2) - o2g.w) / GAP_SOFT);
        float local = o2g.z * gw2 * (1.0 - inside);
        float mean = o2g.z * gw2 * normCdf(o2g.w);
        frac *= 1.0 - mix(local, mean, smoothstep(1.0, 3.0, lodMax));
    }
    if (kT > 0.0) frac = max(frac, clamp(dome * 400.0, 0.0, 1.0));   // a turret stands on cloud
    float uu = clamp(u * 1.6 + 0.5, 0.0, 1.0);
    topsReg = clamp(1.0 - gf.w * uu * (1.15 - 0.5 * frac), 0.22, 1.0);
    float tb = tu.w;
    topsCas = tb + (1.0 - tb) * dome;
    kT = (tb > 0.0) ? clamp(kT, 0.0, 1.0) : 0.0;
    return frac;
}

// pixSize sets the level of detail: without it a deck seen edge-on near the
// horizon aliases into radial streaks.  isRain returns whether the sample is
// falling precipitation below the base rather than cloud.
float deckDensity(int d, vec3 p, float pixSize, out float hf, out float isRain) {
    vec4 a = dp(d, 0);   // base, top, sigma, cumuliformity
    vec4 b = dp(d, 1);   // cellularity, fibrosity, detail, edgeSoftness
    vec4 e = dp(d, 3);   // shearU, shearV, fallSpeed, shapeScale
    vec4 f = dp(d, 4);   // virga, precip, mamma, asperitas
    vec4 g = dp(d, 5);   // anvil, cavum, fluctus, arcus
    vec4 h = dp(d, 6);   // rounded masses (Deck.cell_dome), lens, roll, ragged

    isRain = 0.0;
    if (a.z <= 0.0) return 0.0;
    float alt = altitudeOf(p);
    float thick = max(a.y - a.x, 50.0);
    hf = (alt - a.x) / thick;
    float below = a.x - alt;
    float reach = (f.y > 0.5) ? a.x : min(900.0, a.x * 0.75);
    if (hf > 1.02 || below > reach || alt < -5.0) return 0.0;

    // Where on the pattern (which drifts with the layer's own wind, lens
    // and roll clouds standing: the drift is integrated in timeline.py):
    // shear tilts the deck, and a particle falling dz metres through a wind
    // that shears at e.xy per metre is carried sideways by shear*dz^2/(2*w)
    // - the bend of an uncinus hook and the lean of a virga trail.
    vec2 q = p.xz + vec2(e.x, e.y) * (hf * thick);
    {
        float fall = max(e.z, 0.2);
        float dz = max(a.x + thick * 0.12 - alt, 0.0);
        q -= vec2(e.x, e.y) * (dz * dz / (2.0 * fall));
    }
    // a cumulonimbus's anvil: at the tropopause its plume spreads out, the
    // cloud wider there than its tower (Deck.tower, Deck.anvil)
    float tw = dp(d, 9).y;
    if (tw > 0.0 && below <= 0.0) {
        // A tower is a stack of rising bubbles: its outline bulges in and out
        // with height (turrets), rather than standing as the cell's outline
        // extruded into a wall.  The pattern is read where a 3D noise moves
        // it, by up to a fifth of the noise's scale.
        float ws = max(e.w, 40.0) * 1.7;
        float wl = log2(clamp(pixSize / max(ws / 128.0, 0.5), 1.0, 128.0));
        vec4 wn = textureLod(tShape, p / ws + vec3(0.31, 0.17, 0.53), min(wl, 7.0));
        q += (wn.gb - 0.5) * (2.0 * 0.20 * ws) * tw;
    }
    float zDrop = (below > 0.0) ? 0.0 : tw * g.x * 1.2 * smoothstep(0.64, 0.86, hf);
    float topsReg, topsCas, kT, core;
    float cov = deckPattern(d, q, pixSize, zDrop, topsReg, topsCas, kT, core);
    if (uRegionOn != 0) {
        // the forecast's cover of this deck's étage there, relative to here
        cov *= regionDeck(p.xz, a.x);
    }
    if (cov <= 0.002) return 0.0;

    if (hf < 0.30 && (f.z + f.w) > 0.0) {          // mamma, asperitas
        float lobes = textureLod(tDetail, vec3(p.xz / 260.0, alt / 900.0), 0.0).g;
        float dip = (f.z * 0.10 + f.w * 0.16) * (lobes - 0.5) * 2.0;
        hf += dip;
    }

    float prof;
    if (below > 0.0) { prof = fallStreakDensity(d, below); isRain = 1.0; }
    else {
        float tops = mix(1.0, topsReg, a.w);
        prof = profileFor(d, hf, tops);
        if (h.x > 0.0) {
            // a layer's elements are rounded masses: flat-based, as deep as
            // the layer only in their middle and thin at their edges (a slab
            // of one depth read as coins overhead and pucks toward the
            // horizon)
            float topL = tops * mix(0.32, 1.0, core);
            float roof = saturate(remap(hf, topL * 0.45, topL, 1.0, 0.0));
            prof = mix(prof, saturate(remap(hf, 0.0, 0.10, 0.0, 1.0)) * roof, h.x);
        }
        if (tw > 0.0) {
            // A cumulus is the saturated part of rising plumes: flat at its
            // condensation level, highest over its strongest updraught (its
            // middle), lower toward its edges, where drier air is mixed in -
            // a dome on a flat base, narrowing upward.  (The cell's outline
            // stood up as a wall drew boxes.)  A cumulonimbus's plume spreads
            // at the tropopause into its anvil: flat-topped, wider than the
            // tower (zDrop above), thinning outward; its tower stays broad up
            // to the anvil.
            float dmc = sqrt(saturate(1.0 - (1.0 - core) * (1.0 - core)));
            float topP = tops * mix(mix(0.30, 0.55, g.x), 1.0, dmc);
            float plume = saturate(remap(hf, 0.0, 0.06, 0.0, 1.0))
                        * saturate(remap(hf, topP * 0.55, topP, 1.0, 0.0));
            float anvil = g.x * smoothstep(0.66, 0.82, hf) * (1.0 - smoothstep(0.95, 1.0, hf))
                        * mix(0.35, 1.0, core);
            prof = mix(prof, max(plume, anvil), tw);
        }
        if (kT > 0.0) prof = mix(prof, profileCastellanus(d, hf, topsCas, dp(d, 14).w), kT);
    }
    if (prof <= 0.0) return 0.0;
    // (a plume's width is its dome's: widthFor's narrow base and wide middle
    // drew mushrooms on stalks)
    float wid = (below > 0.0) ? 0.75 : mix(widthFor(d, hf), 1.0, max(kT, tw));

    float scale = max(e.w, 40.0);
    float shapeLod = log2(clamp(pixSize / max(scale / 128.0, 0.5), 1.0, 128.0));
    vec4 sh = textureLod(tShape, p / scale, min(shapeLod, 7.0));
    float wfbm = sh.g * 0.625 + sh.b * 0.25 + sh.a * 0.125;
    float fibEff = (a.w > 0.7) ? b.y * smoothstep(0.50, 0.78, hf) : b.y;
    float shape = mix(sh.r, saturate(wfbm * 1.15), 0.30 * (1.0 - fibEff));
    float influence = mix(0.34, 1.0, a.w) * (1.0 - 0.55 * fibEff);
    influence = mix(influence, 0.80, h.x);           // a layer's elements: lumpy, not smooth
    float shp = mix(1.0, shape, influence);

    float thr = max(1.0 - cov * wid, 0.02);
    float dens = remap(shp, thr, 1.0, 0.0, 1.0) * prof;
    dens *= smoothstep(0.0, 0.10, cov);
    // and their water is thinnest at their edges, thickest in their middle:
    // grey cores and bright, translucent, blurred edges seen from below
    dens *= mix(1.0, mix(0.25, 1.5, core), h.x);
    if (dens <= 0.0) return 0.0;

    if (shapeLod < 1.2 && b.z > 0.0) {
        // explicit level of detail: the Worley detail is finer than a pixel
        // on a distant deck, and sampled without a mip it aliases into a mesh
        float dscale = max(scale * 0.16, 8.0);
        float detLod = log2(clamp(pixSize / (dscale / 32.0), 1.0, 32.0));
        vec3 det = textureLod(tDetail, p / dscale, detLod).rgb;
        float dfbm = det.r * 0.625 + det.g * 0.25 + det.b * 0.125;
        float e0 = mix(dfbm, 1.0 - dfbm, saturate(hf * 2.5));
        dens = remap(dens, e0 * b.z * 0.45 * (0.35 + 0.65 * h.w), 1.0, 0.0, 1.0);
    }
    if (g.y > 0.5) {                                  // cavum
        vec4 cv = dp(d, 16);                          // its drift and spacing
        vec2 qc = (p.xz - cv.xy) / (cv.z * 0.55);
        float r2 = length(fract(qc) - 0.5);
        dens *= smoothstep(0.10, 0.20, r2);
    }
    return saturate(dens);
}

// ------------------------------------------------------------ simulation --
// World point (relative to the eye: x east, y up, z north) -> model
// coordinates (x east, y north, z above the domain's floor).
vec3 simCoord(vec3 p) {
    vec2 xy = vec2(uSimObs.x + p.x, uSimObs.y + p.z) - regionWarp(p.xz);
    return vec3(xy, altitudeOf(p) - uSimZ0);
}
bool simInside(vec3 s) {
    if (s.z < 0.0 || s.z > uSimSize.z) return false;
    if ((uSimTile & 1) == 0 && (s.x < 0.0 || s.x > uSimSize.x)) return false;
    if ((uSimTile & 2) == 0 && (s.y < 0.0 || s.y > uSimSize.y)) return false;
    return true;
}
vec4 simOpt(vec3 s, float lod) {
    vec3 uvw = s / uSimSize;
    return mix(textureLod(tSimOptA, uvw, lod), textureLod(tSimOptB, uvw, lod), uSimMix);
}
// Tricubic B-spline filtering from eight trilinear fetches (Sigg &
// Hadwiger, GPU Gems 2 ch. 20).  Trilinear interpolation of a coarse field
// draws its iso-surfaces as boxes; the cubic B-spline rounds them, which is
// what a cloud's resolved envelope should look like up close.
vec4 cubic3(sampler3D tex, vec3 uvw, vec3 N) {
    vec3 x = uvw * N - 0.5;
    vec3 i = floor(x);
    vec3 f = x - i;
    vec3 f2 = f * f, f3 = f2 * f;
    vec3 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec3 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec3 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    vec3 w3 = f3 / 6.0;
    vec3 g0 = w0 + w1;
    vec3 g1 = w2 + w3;
    vec3 h0 = (i - 0.5 + w1 / g0) / N;
    vec3 h1 = (i + 1.5 + w3 / g1) / N;
    vec4 a = mix(textureLod(tex, vec3(h1.x, h0.y, h0.z), 0.0), textureLod(tex, vec3(h0.x, h0.y, h0.z), 0.0), g0.x);
    vec4 b = mix(textureLod(tex, vec3(h1.x, h1.y, h0.z), 0.0), textureLod(tex, vec3(h0.x, h1.y, h0.z), 0.0), g0.x);
    vec4 c = mix(textureLod(tex, vec3(h1.x, h0.y, h1.z), 0.0), textureLod(tex, vec3(h0.x, h0.y, h1.z), 0.0), g0.x);
    vec4 d = mix(textureLod(tex, vec3(h1.x, h1.y, h1.z), 0.0), textureLod(tex, vec3(h0.x, h1.y, h1.z), 0.0), g0.x);
    return mix(mix(d, c, g0.y), mix(b, a, g0.y), g0.z);
}
vec4 simOptCubic(vec3 s) {
    vec3 uvw = s / uSimSize;
    vec3 N = uSimSize / uSimCell;
    return mix(cubic3(tSimOptA, uvw, N), cubic3(tSimOptB, uvw, N), uSimMix);
}
vec4 simAux(vec3 s) {
    vec3 uvw = s / uSimSize;
    return mix(textureLod(tSimAuxA, uvw, 0.0), textureLod(tSimAuxB, uvw, 0.0), uSimMix);
}
float simTotal(vec4 o) { return o.x + o.y + o.z + o.w; }

// Where the forecast has less of this cloud than there is here, the
// model's thinnest columns are left out first - the cloud field thins and
// ends where the satellite would show it thinning and ending.  r is the
// forecast cover there over the cover here; uTauTh[] holds, for r = k/16,
// the column optical depth below which the model's own columns make up the
// fraction of cloud to drop.
//
// uSimFade < 1 takes the model's field out the same way, everywhere: when
// the sky hands over from the model's clouds to the Atlas layers or back
// (a model starting, restarting or changing), its thinnest columns go
// first and its thickest last - clouds dissolving, not fading.
float simKeep(vec3 p, vec3 s) {
    float r = clamp(regionEtage(p.xz, uRegSimEt), 0.0, 1.0) * uSimFade;
    if (r >= 0.999) return 1.0;
    if (r <= 0.001) return 0.0;
    float f = r * 16.0;
    int k = int(floor(f));
    float th = mix(uTauTh[k], uTauTh[min(k + 1, 16)], f - float(k));
    float col = (uKeepMapOn == 1) ? texture(tSimKeep, s.xy / uSimSize.xy).r
                                  : simAux(vec3(s.xy, 0.25 * uSimCell.z)).x;
    return smoothstep(0.7 * th, 1.3 * th + 1e-3, col);
}
// Precipitation below a cloud is kept or left out with the cloud it fell
// from: its column is traced back up its trajectory (falling at vfall
// through the wind relative to the drifting field) to the cloud base.
float simKeepPrecip(vec3 p, vec3 s, vec4 aux, float snowy) {
    if (uRegionOn == 0 && uSimFade >= 0.999) return 1.0;
    float vfall = mix(6.0, 1.2, snowy);
    float drop = max(uSimZBase - uSimZ0 - s.z, 0.0);
    vec2 rel = aux.yz - uSimMove;
    return simKeep(p, vec3(s.xy - rel * drop / vfall, s.z));
}
"""

# ---------------------------------------------------------------------------
# Cloud sub-grid detail for the simulated field
# ---------------------------------------------------------------------------
SIM_DETAIL = """
// The model resolves a cloud's shape down to a few grid lengths.  On the
// large-eddy grids (tens of metres) that IS the cloud: its cells, rows,
// turrets and lobes are the model's, and nothing is added but the texture of
// the turbulence finer than the grid, at the cloud's edge only.  On the
// coarse grids of the deep-convection runs (hundreds of metres) the billows
// a few hundred metres across are below the model's reach, and procedural
// Perlin-Worley billows are laid over the resolved envelope - more the
// coarser the grid.  Both are carried by the model's own wind (a two-phase
// flow map), and the base is left flat: it is the condensation level, which
// the model resolves.
const float SIG_CORE = 0.035;        // m^-1: about 0.25 g/kg of 10 um droplets
const float SIG_EDGE = 0.03;         // m^-1: where the tricubic field is 'inside'
const float SIG_EDGE_ICE = 0.004;    // ... for ice cloud
const float FLOW_T = 40.0;           // s, flow-map period (|grad u| T << 1)

// how much of the shape is left to procedural billows: none at LES spacing
float subgridWeight() { return clamp((uSimCell.x - 80.0) / 420.0, 0.0, 1.0); }
float simShapeScale() { return max(uSimCell.x * 6.5, 1300.0); }

float shapeAt(vec3 q, float lod) {
    vec4 sh = textureLod(tShape, q / simShapeScale(), lod);
    float billow = sh.g * 0.5 + sh.b * 0.3 + sh.a * 0.2;
    // remapped by the texture's own 5th and 95th percentiles (0.21, 0.55)
    return saturate((0.3 * sh.r + 0.7 * billow - 0.21) / 0.34);
}

float simCloudDensity(vec3 s, vec4 o, vec4 aux, float pixSize, out float cov) {
    // Coverage: 0 outside the resolved cloud, 1 inside, ramping across the
    // tricubic edge (about a grid cell wide).  Ice cloud (an anvil) is
    // optically thinner for the same size - the crystals are bigger - so it
    // counts as 'inside' at a lower extinction.
    float sc = o.x + o.y;
    float ice = o.y / max(sc, 1e-9);
    cov = saturate(sc / mix(SIG_EDGE, SIG_EDGE_ICE, ice));
    if (cov <= 0.0) return 0.0;
    float sw = subgridWeight();
    vec3 vel = aux.yzw;
    float ph = uSimTime / FLOW_T;
    float f1 = fract(ph);
    float f2 = fract(ph + 0.5);
    float w1 = 1.0 - abs(2.0 * f1 - 1.0);
    vec3 s1 = s - vel * (f1 * FLOW_T);
    vec3 s2 = s - vel * (f2 * FLOW_T) + vec3(731.0, 419.0, 263.0);
    float d = 1.0;
    if (sw > 0.0) {
        float P1 = simShapeScale();
        float lod = log2(clamp(pixSize / (P1 / 128.0), 1.0, 128.0));
        float n = mix(shapeAt(s2, lod), shapeAt(s1, lod), w1);
        // is this the base?  cloud above, clear air below
        vec3 hz = vec3(0.0, 0.0, 0.5 * uSimCell.z);
        vec4 ob = simOpt(s - hz, 0.0);
        vec4 oa = simOpt(s + hz, 0.0);
        float cb = ob.x + ob.y, ca = oa.x + oa.y;
        float baseness = saturate((ca - cb) / max(ca, 1e-6)) * saturate(1.0 - cb / SIG_CORE);
        float amp = sw * (1.0 - 0.8 * baseness) * (1.0 - 0.55 * ice);
        d = remap(mix(1.0, n, amp), 1.0 - cov, 1.0, 0.0, 1.0);
    }
    d *= smoothstep(0.04, 0.2, cov);         // no isolated specks outside the cloud
    if (d <= 0.0) return 0.0;
    // turbulence finer than the grid, at the edge of the resolved cloud
    float dscale = max(uSimCell.x * 1.6, 25.0);
    float dlod = log2(clamp(pixSize / (dscale / 32.0), 1.0, 32.0));
    if (dlod < 4.5) {
        vec3 det = textureLod(tDetail, s1 / dscale, dlod).rgb;
        float dfbm = det.r * 0.625 + det.g * 0.25 + det.b * 0.125;
        float edge = 1.0 - cov * 0.85;
        float e0 = (1.0 - dfbm) * mix(0.30, 0.45, sw) * edge * (1.0 - 0.6 * ice) * (1.0 - dlod / 4.5);
        d = remap(d, e0, 1.0, 0.0, 1.0);
    }
    return d;
}

// Precipitation falls in fibres: each generating cell drops its particles
// along one trajectory, and the trajectories lean with the wind relative to
// the cell (snow falls ~1.5 m/s, rain ~6 m/s).  The model gives the shaft;
// this gives the striation inside it, stretched along the trajectory.
float rainStreaks(vec3 s, vec4 aux, float snowy) {
    float vfall = mix(6.0, 1.5, snowy);
    vec2 rel = aux.yz - uSimMove;                 // wind relative to the drifting field
    float h = uSimSize.z - s.z;                   // distance fallen from the top
    vec2 q = s.xy - rel * h / vfall;
    vec3 det = textureLod(tDetail, vec3(q / 90.0, (s.z + vfall * uSimTime) / 2400.0), 0.0).rgb;
    float n = det.r * 0.55 + det.g * 0.30 + det.b * 0.15;
    return 0.55 + 0.9 * n;
}
"""

# ---------------------------------------------------------------------------
# The Beer shadow map: one texel per ray from the Sun
# ---------------------------------------------------------------------------
SHADOW_FRAG = """#version 330
in vec2 v_uv;
out vec4 f_colour;
""" + COMMON + """
uniform int uShSteps;
uniform float uShTop;
void main() {
    vec2 q = v_uv * 2.0 - 1.0;
    vec3 P0 = uShE1 * (q.x * uShExtent) + uShE2 * (q.y * uShExtent) + uLightDir * uShD0;
    vec3 dir = -uLightDir;
    vec3 ro = P0 - earthCentre();
    vec2 top = raySphere(ro, dir, groundR() + uShTop);
    if (top.y <= 0.0 || uLightDir.y < -0.12) { f_colour = vec4(1e9, 0.0, 0.0, 1.0); return; }
    float t0 = max(top.x, 0.0);
    float t1 = top.y;
    vec2 gnd = raySphere(ro, dir, groundR());
    if (gnd.x > 0.0) t1 = min(t1, gnd.x);
    t1 = min(t1, uShD0 + 60000.0);
    float dt = (t1 - t0) / float(uShSteps);
    float front = -1.0, back = 0.0, tau = 0.0;
    float jit = hash13(vec3(gl_FragCoord.xy, 3.1)) - 0.5;
    for (int i = 0; i < 256; ++i) {
        if (i >= uShSteps) break;
        float t = t0 + (float(i) + 0.5 + 0.8 * jit) * dt;
        vec3 x = P0 + dir * t;
        float sig = 0.0;
        for (int d = 0; d < MAXDECKS; ++d) {
            if (d >= uNumDecks) break;
            vec4 a = dp(d, 0);
            float hf, isr;
            float dd = deckDensity(d, x, max(dt, 150.0), hf, isr);
            sig += dd * a.z;
        }
        if (uSimOn != 0) {
            vec3 s = simCoord(x);
            if (simInside(s)) {
                float st = simTotal(simOpt(s, 0.0));
                if (st > 0.0) sig += st * simKeep(x, s);
            }
        }
        if (sig > 2e-5) {
            if (front < 0.0) front = t - 0.5 * dt;
            back = t + 0.5 * dt;
            tau += sig * dt;
        }
    }
    if (front < 0.0) { f_colour = vec4(1e9, 0.0, 0.0, 1.0); return; }
    float depth0 = uShD0 - dot(P0, uLightDir);   // depth of P0 itself (0 by construction)
    f_colour = vec4(front + depth0, tau / max(back - front, dt), tau, 1.0);
}
"""

# ---------------------------------------------------------------------------
# The sky
# ---------------------------------------------------------------------------
SKY_FRAG = """#version 330
in vec2 v_uv;
out vec4 f_colour;
""" + COMMON + SIM_DETAIL + """
uniform vec2  uResolution;
uniform vec3  uCamRight, uCamUp, uCamFwd;
uniform float uTanHalfFov;
uniform float uAspect;

uniform vec3  uMoonDir;
uniform float uSunAngRad;
uniform float uMoonAngRad;
uniform float uMoonIllum;
uniform vec3  uMoonBrightDir;
uniform mat3  uHorToEqu;

uniform float uExposure;
uniform int   uSteps;
uniform int   uLightSteps;
uniform int   uAtmSteps;
uniform int   uSimSteps;
uniform float uFrameJitter;
uniform vec2  uPixJitter;            // sub-pixel ray offset, NDC units (0 unless accumulating)
uniform int   uShowStars;
uniform float uHaloStrength;
uniform float uCoronaStrength;

// clouds above the weather (upper.py)
uniform float uNlc;                  // noctilucent strength, 0 = none
uniform int   uNlcTypes;             // bits: 1 veils, 2 bands, 4 billows, 8 whirls
uniform float uNacreous;             // polar stratospheric type II (ice)
uniform float uPscLens;              // 1: lenticular, 0: cirriform, between: going over
uniform float uNat;                  // polar stratospheric type I (nitric acid)
uniform vec2  uUpperWind;            // mesospheric wind, m/s
uniform vec2  uPscWind;              // stratospheric wind, m/s
uniform float uUpperTime;            // s

// lightning
uniform vec3  uFlashPos;             // relative to the eye
uniform float uFlashPower;           // 0 when dark
uniform int   uBoltN;
uniform vec4  uBolt[48];             // polyline: xyz, brightness

// What the air scatters toward the eye of a diffuse field coming down from
// above (even radiance over the upper hemisphere), as a fraction of all it
// scatters: the haze's phase function (Henyey-Greenstein, g 0.7) integrated
// over the upper hemisphere.  mu is the upward part of the light's way to the
// eye: -1 for an eye looking straight up into the field (forward scattering,
// 0.916), 0 looking level (0.5), +1 for an eye looking straight down on it
// (back scattering, 0.084).  Fitted to the integral within 0.005.
float hemiDown(float mu) {
    float a = abs(mu);
    return 0.5 - sign(mu) * (1.5448 * a / (1.0745 + a) - 0.3324 * a);
}

// ------------------------------------------------------------- atmosphere --
// Marches the air between d0 and d1.  Steps crowd toward d0 (the eye) where
// crepuscular rays live, and each sample is jittered from frame to frame.
// The single-scattered sunlight is multiplied by the cloud shadow map - that
// is what makes the shafts - but the multiply-scattered part is not, as in
// Hillaire's sky model.
void atmosphere(vec3 rd, float d0, float d1, int steps, float jit,
                out vec3 L, out vec3 T) {
    L = vec3(0.0);
    T = vec3(1.0);
    if (d1 <= d0) return;
    vec3 ro = -earthCentre();
    float cosT = dot(rd, uSunDir);
    float pr = phaseRayleigh(cosT);
    float pm = phaseHG(cosT, MIE_G);
    float ph = phaseHG(cosT, 0.70);
    float prev = d0;
    for (int i = 0; i < 64; ++i) {
        if (i >= steps) break;
        float s1 = float(i + 1) / float(steps);
        float t1 = d0 + (d1 - d0) * s1 * s1;
        float ds = t1 - prev;
        float d = prev + ds * jit;
        vec3 x = rd * d;
        vec3 p = ro + x;
        float r = length(p);
        float h = max(r - RG, 0.0);
        float dr = exp(-h / RAY_H);
        float dm = exp(-h / MIE_H);
        float doz = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
        // boundary-layer haze on top of the clean-air aerosol of the tables:
        // grey, single-scattering albedo 0.92, forward-scattering (g = 0.7),
        // scale height 1.5 km.  It is what hazes the horizon and what makes
        // crepuscular rays visible at all.
        float dh = uHaze * exp(-max(h - uGroundElev, 0.0) / 1500.0);
        vec3 scat = RAY_S * dr + MIE_S * dm + vec3(dh * 0.92);
        vec3 ext = RAY_S * dr + MIE_E * dm + OZO_A * doz + vec3(dh);
        float muS = dot(p / r, uSunDir);
        vec3 trSun = sampleTransmittance(r, muS);
        float shadow = earthShadowed(r, muS) ? 0.0 : 1.0;
        // Under cloud the air sees neither the Sun nor the blue sky, but the
        // grey light the cloud lets through: the sky's diffuse light and the
        // Sun's beam each thinned by the two-stream transmission of the
        // cloud above (vertical optical depth ~ slant tau x mu0).  Of the
        // beam's scattered part, the droplets' forward peak (delta-M: the
        // fraction g^2 = 0.74 of what is scattered, g 0.86) is turned by a
        // few degrees only and goes on with the beam, so the air scatters it
        // as it scatters sunlight; the rest is a diffuse field coming down
        // from the cloud, and what the air sends of it toward the eye is the
        // phase function over the upper hemisphere (hemiDown) - most of it
        // to an eye below looking up into it, little to an eye above looking
        // down on it.  (All of it was taken as scattered evenly in every
        // direction: seen from above, the shadows of clouds in hazy air came
        // out up to four times brighter than the sunlit air beside them -
        // white trails behind every cloud, radiating from the point opposite
        // the Sun - and the air under a thin veil several times too bright.)
        float tauSh = (uLightIsSun != 0) ? shadowTau(x) : 0.0;
        float mu0 = max(muS, 0.05);
        float tDiff = 1.0 / (1.0 + 0.105 * tauSh * mu0);
        float tBeam = exp(-tauSh);
        float tScat = max(1.0 / (1.0 + 0.105 * tauSh) - tBeam, 0.0);
        float tFwd = min(max(exp(-0.26 * tauSh) - tBeam, 0.0), tScat);
        // the diffuse field's radiance, the beam's diffusely transmitted flux / pi
        vec3 sunDiff = trSun * shadow * mu0 * (tScat - tFwd) * (1.0 / PI);
        shadow *= tBeam + tFwd;
        float hd = hemiDown(-dot(rd, p / r));
        vec3 inscat = (RAY_S * dr * pr + MIE_S * dm * pm + vec3(dh * 0.92) * ph) * trSun * shadow;
        inscat += (RAY_S * dr + MIE_S * dm) * sampleMultiScatter(r, muS) * tDiff;
        inscat += vec3(dh * 0.92) * skyIrradiance(max(h - uGroundElev, 0.0)) * (1.6 / (4.0 * PI)) * tDiff;
        // (Rayleigh scatters as much forward as back: half of the field)
        inscat += (RAY_S * dr * 0.5 + (MIE_S * dm + vec3(dh * 0.92)) * hd) * sunDiff;
        vec3 stepT = exp(-ext * ds);
        L += T * (inscat - inscat * stepT) / max(ext, vec3(1e-9));
        T *= stepT;
        prev = t1;
    }
}

// ------------------------------------------------------------ cloud light --
// Light inside a cloud, as a source function S (radiance scattered toward
// the eye per unit extinction):
//   * single scattering with the droplets' true Mie phase function
//     (Jendersie-d'Eon), attenuated along the path to the light;
//   * two octaves of low-order forward scattering (Wrenninge's octave model
//     as SimonDev shows it): thinner cloud, flatter phase function - what
//     makes a silver lining glow;
//   * the diffuse field of high orders.  In a thick cloud the light no
//     longer decays exponentially, it diffuses; the Eddington solution for a
//     conservative slab gives a mean intensity falling linearly from the lit
//     face to the far face,
//         J = E/pi (1 + 3/4 (t* - s*)) / (1 + 3/4 t*),
//     with s* the similarity-scaled optical depth (1-g)tau from the lit face
//     and t* the whole thickness along the light.  That is what keeps the lit
//     side of a cumulus white from every direction (clouds reflect 70-90%)
//     and its base grey instead of black.
vec3 cloudRadiance(float tauS, float tauB, float cosT, vec4 jde, float ice, float g0,
                   vec3 sunC, float powderDens, float occ) {
    float ph0 = mix(phaseJdE(cosT, jde), phaseHG(cosT, 0.76), ice);
    float single = ph0 * exp(-tauS);
    float o1 = 0.62 * phaseHG(cosT, g0 * 0.60) * exp(-tauS * 0.52);
    float o2 = 0.38 * phaseHG(cosT, g0 * 0.36) * exp(-tauS * 0.27);
    float ts = tauS * (1.0 - g0);
    float tt = (tauS + tauB) * (1.0 - g0);
    float D = (1.0 + 0.75 * (tt - ts)) / (1.0 + 0.75 * tt) * (1.0 - exp(-1.5 * tt));
    // powder: in-scattering needs something to in-scatter, so the thinnest
    // edges facing away from the light are darker (Schneider 2015)
    float powder = 1.0 - exp(-powderDens);
    powder = mix(1.0, powder, saturate(cosT * 0.5 + 0.5) * 0.6);
    return sunC * ((single + o1 + o2) * powder + D * occ / PI);
}

// The light of a layer of rounded masses (Altocumulus, Stratocumulus,
// Cirrocumulus: Deck.cell_dome).  The diffuse term above is the two-stream
// solution of a layer without edges.  An element a few hundred metres
// across is not one: its diffused light leaks out of its flanks into the
// clear air between the elements, and what is left travels on away from the
// Sun.  Seen from below, Monte Carlo transport through such elements (flat-
// based half-ellipsoids two wide for one deep, droplet phase g = 0.86, the
// Sun 38.7 degrees up, seen 40-50 degrees up toward, across and away from
// the Sun; alone, 34 % and 60 % of the sky covered, optical depths 3-80)
// gave elements 2-10 times darker than that term on their shaded side and
// a sunlit side twice as bright as their shaded one, where it drew them
// evenly lit: no shadow beside the lit part.  Fitted to those runs (rms
// error a factor 1.28, from 2.5):
//   D = 0.48 e^(-0.10 s*) [1 + 3/4 (t* - s*) + 0.72 cos] / (1 + 3/4 t*)
//       x (1 - e^(-1.33 max(t*, (1-g) tauEl)))^4.5
// with s*, t* as above, cos the cosine of the angle to the Sun, tauEl the
// column through an element's middle (the deck's optical depth); the near-
// forward orders gain (1 + 2.3 cos^4.7) for the light the element does not
// turn aside.  The layer's own two-stream term returns as the elements
// close into an overcast (weight cover^4.4: a layer without edges).
// dome weighs it in, so a layer that changes its kind changes its light
// smoothly; dome 0 is cloudRadiance exactly.
// gl_sky.DOME_COLUMN: by how much a layer of rounded masses' extinction is
// raised so the column through an element's middle is the deck's optical depth
const float DOME_COLUMN = 3.0;

vec3 deckRadiance(float tauS, float tauB, float tauEl, float cover, float dome, float cosT,
                  vec4 jde, float ice, float g0, vec3 sunC, float powderDens) {
    float ph0 = mix(phaseJdE(cosT, jde), phaseHG(cosT, 0.76), ice);
    float single = ph0 * exp(-tauS);
    float o1 = 0.62 * phaseHG(cosT, g0 * 0.60) * exp(-tauS * 0.52);
    float o2 = 0.38 * phaseHG(cosT, g0 * 0.36) * exp(-tauS * 0.27);
    float ts = tauS * (1.0 - g0);
    float tt = (tauS + tauB) * (1.0 - g0);
    float Dlayer = (1.0 + 0.75 * (tt - ts)) / (1.0 + 0.75 * tt) * (1.0 - exp(-1.5 * tt));
    float tEl = max(tt, tauEl * (1.0 - g0));
    float Delem = 0.481 * exp(-0.104 * ts)
                * max(1.0 + 0.75 * (tt - ts) + 0.724 * cosT, 0.0) / (1.0 + 0.75 * tt)
                * pow(1.0 - exp(-1.329 * tEl), 4.465);
    float D = mix(Dlayer, mix(Delem, Dlayer, pow(saturate(cover), 4.363)), dome);
    float fwd = 1.0 + dome * 2.275 * pow(max(cosT, 0.0), 4.717);
    float powder = 1.0 - exp(-powderDens);
    powder = mix(1.0, powder, saturate(cosT * 0.5 + 0.5) * 0.6);
    return sunC * ((single + (o1 + o2) * fwd) * powder + D / PI);
}

float thick_of(int d) { vec4 a = dp(d, 0); return max(a.y - a.x, 50.0); }

vec3 deckLighting(int d, vec3 p, float density, float hf, float cosT,
                  vec3 skyAmbient, float pixSize, float isRain) {
    vec4 a = dp(d, 0);
    vec4 c = dp(d, 2);
    float sigma = a.z;
    float ice = c.w;
    vec4 jde = dp(d, 8);
    float g0 = mix(0.86, 0.76, ice);
    vec3 sunC = lightAt(p);

    // march toward the light through this deck, then the shadow map beyond
    float tau = 0.0;
    float stepLen = max((a.y - a.x) / float(uLightSteps), 40.0);
    float tl = 0.0;
    for (int i = 0; i < 12; ++i) {
        if (i >= uLightSteps) break;
        float t = (float(i) + 0.5) * stepLen;
        vec3 q = p + uLightDir * t;
        float hq, rq;
        float dq = deckDensity(d, q, pixSize * 2.0, hq, rq);
        tau += dq * sigma * stepLen;
        tl = t + 0.5 * stepLen;
        if (altitudeOf(q) > a.y) break;
    }
    tau += shadowTau(p + uLightDir * tl);
    // and away from the light, through the rest of the deck
    float tauB = 0.0;
    for (int i = 0; i < 4; ++i) {
        float t = (float(i) + 0.5) * stepLen * 1.5;
        vec3 q = p - uLightDir * t;
        float hq, rq;
        tauB += deckDensity(d, q, pixSize * 2.0, hq, rq) * sigma * stepLen * 1.5;
    }
    // a layer of rounded masses (Deck.cell_dome) is lit as one (deckRadiance)
    vec4 h6 = dp(d, 6);
    float dome = h6.x;
    if (isRain > 0.5) {
        // Under the deck the rain sees the light the deck lets through: the
        // sky's and the Sun's, each thinned by the conservative two-stream
        // diffuse transmission 1/(1 + 0.75 (1-g) tau) of the deck above.
        float tauDeck = sigma * thick_of(d);
        float tDiff = 1.0 / (1.0 + 0.75 * (1.0 - g0) * tauDeck);
        float mu0 = max(uLightDir.y, 0.05);
        vec3 amb = skyAmbient * tDiff + sunC * mu0 / PI * max(tDiff - exp(-tauDeck / mu0), 0.0);
        return sunC * exp(-tau) * phaseRain(cosT) * 0.95 + amb;
    }
    vec3 sun;
    if (dome > 0.0) {
        // the column through an element's middle: the deck's optical depth
        // (its extinction was raised by DOME_COLUMN for that, gl_sky)
        float tauEl = sigma * max(a.y - a.x, 50.0) / (1.0 + dome * (DOME_COLUMN - 1.0));
        sun = deckRadiance(tau, tauB, tauEl, dp(d, 16).w, dome, cosT, jde, ice, g0, sunC,
                           density * sigma * 800.0);
    }
    else sun = cloudRadiance(tau, tauB, cosT, jde, ice, g0, sunC, density * sigma * 800.0, 1.0);
    float tauStar = tau * (1.0 - g0);
    float openness = exp(-tauStar * 0.20);
    vec3 amb = skyAmbient * (0.25 + 0.75 * saturate(hf)) * mix(openness, 1.0, saturate(hf))
             + skyAmbient * uGroundAlbedo * 0.6 * (1.0 - saturate(hf));
    return sun + amb;
}

// haloW and coronaW: the halo and corona strengths of the decks this ray
// sees (thin cirrostratus, thin altocumulus), each weighted by how much of
// what reaches the eye through the decks in front of it that deck gives - a
// halo is not seen through a stratus overcast below it.  Each deck's
// strength is continuous in time (gl_sky.deck_optics), so a halo fades as
// its layer thickens or changes its name.
void marchDecks(vec3 rd, float maxDist, float dither, out vec3 L, out float T, out float meanDist,
                out float haloW, out float coronaW) {
    L = vec3(0.0);
    T = 1.0;
    meanDist = 0.0;
    haloW = 0.0;
    coronaW = 0.0;
    float wsum = 0.0;
    vec3 ro = -earthCentre();
    float cosT = dot(rd, uLightDir);
    // The decks in the order this ray meets them, nearest first, so that a
    // nearer deck hides a farther one: from the ground the lowest deck is in
    // front, from an aircraft or a satellite the highest.
    vec4 segs[MAXDECKS];
    int nsegs[MAXDECKS];
    int order[MAXDECKS];
    float entry[MAXDECKS];
    int nd = 0;
    for (int d = 0; d < MAXDECKS; ++d) {
        if (d >= uNumDecks) break;
        vec4 a = dp(d, 0);
        vec4 f = dp(d, 4);
        nsegs[d] = 0;
        if (a.z <= 0.0 || dp(d, 10).w > 30.0) continue;     // an empty group, or nothing of it there
        float thick = max(a.y - a.x, 50.0);
        float lowest = a.x - ((f.x + f.y) > 0.0 ? thick * mix(0.35, 0.9, f.y) : 0.0);
        vec4 sg;
        int ns = shellSegments(ro, rd, max(lowest, 20.0), a.y, maxDist, sg);
        segs[d] = sg;
        nsegs[d] = ns;
        if (ns == 0) continue;
        int k = nd;
        while (k > 0 && entry[k - 1] > sg.x) {
            entry[k] = entry[k - 1];
            order[k] = order[k - 1];
            --k;
        }
        entry[k] = sg.x;
        order[k] = d;
        ++nd;
    }
    int tail = 0;                            // unlit steps taken once opaque
    for (int j = 0; j < MAXDECKS; ++j) {
        if (j >= nd) break;
        if (T < 1e-4 || tail >= 32) break;
        int d = order[j];
        vec4 seg = segs[d];
        int nseg = nsegs[d];
        vec4 a = dp(d, 0);
        float thick = max(a.y - a.x, 50.0);
        vec4 optics = dp(d, 9);              // r_eff, ice, halo strength, corona strength
        vec3 amb = skyIrradiance((a.x + a.y) * 0.5) / PI;
        float feature = max(dp(d, 3).w, 60.0);
        float pixAng = 2.0 * uTanHalfFov / max(uResolution.y, 1.0);
        float minStep = clamp(feature * 0.09, 14.0, 190.0);
        float maxStep = max(thick * 0.8, 600.0);
        const float BIG = 3.0;
        float t = seg.x;
        float d1 = seg.y;
        int part = 0;
        float ds = clamp(max(t * 0.018, t * pixAng * 0.9), minStep, maxStep);
        // a stretch starts with single steps: its first sample lies within one
        // step of its entry, or a thin layer could be stepped over whole
        float scaleStep = 1.0;
        float lastEmpty = -1.0;
        for (int i = 0; i < 640; ++i) {
            if (i >= uSteps || T < 1e-4 || tail >= 32) break;
            if (t > d1) {
                if (part == 0 && nseg == 2) { part = 1; t = max(t, seg.z); d1 = seg.w; lastEmpty = -1.0; scaleStep = 1.0; }
                else break;
            }
            ds = clamp(max(t * 0.018, t * pixAng * 0.9), minStep, maxStep);
            // Each sample stands anywhere in the stretch it stands for (the
            // dither changes from frame to frame).  Through empty air the
            // stride grows to three steps; a sample only dithered within one
            // step never looked at the other two, so an element thinner than
            // that along the ray was hit on some rows and missed on the next -
            // small steep elements (castellanus turrets) came out as stacks of
            // discs.
            float stride = min(ds * scaleStep, max(d1 - t, 1.0));   // never past the stretch's end
            float ts = t + stride * dither;
            vec3 p = rd * ts;
            float hf, isRain;
            float pixSize = max(ts * pixAng, 1.0);
            float dens = deckDensity(d, p, pixSize, hf, isRain);
            if (dens > 0.0015) {
                if (lastEmpty >= 0.0 && scaleStep > 1.5) {
                    // crossed into cloud on a long stride: find the edge
                    float lo = lastEmpty, hi = ts;
                    for (int b = 0; b < 5; ++b) {
                        float mid = (lo + hi) * 0.5;
                        float hh, rr;
                        if (deckDensity(d, rd * mid, pixSize, hh, rr) > 0.0015) hi = mid;
                        else lo = mid;
                    }
                    t = hi;
                    scaleStep = 1.0;
                    lastEmpty = -1.0;
                    continue;
                }
                float sigma = a.z * dens;
                float stepT = exp(-sigma * ds);
                if (T < 0.012) {
                    // As good as opaque: no more light worth lighting, but the
                    // transmittance is followed on (up to 32 unlit steps), or
                    // the Sun's disc would show through a nimbostratus at 1%
                    // of its brightness.
                    T *= stepT;
                    t += ds;
                    ++tail;
                    continue;
                }
                vec3 lit = deckLighting(d, p, dens, hf, cosT, amb, pixSize, isRain);
                float seen = (1.0 - stepT) * T;
                L += lit * seen;
                meanDist += ts * seen;
                wsum += seen;
                haloW += seen * optics.z;
                coronaW += seen * optics.w;
                T *= stepT;
                scaleStep = 1.0;
                lastEmpty = -1.0;
                t += ds;
            } else {
                lastEmpty = ts;
                t += stride;
                scaleStep = min(scaleStep + 0.5, BIG);
            }
        }
    }
    meanDist = (wsum > 1e-5) ? meanDist / wsum : -1.0;
}

// ------------------------------------------------------- simulated clouds --
// Optical depth toward the light: two short steps through the full,
// detailed density - the billow-scale self-shadowing that makes a cumulus
// look like a cauliflower rather than a cotton ball - then the grid.
float simLightTau(vec3 x, vec4 aux, float pixSize, out float tl, out float tauNear) {
    float tau = 0.0;
    tl = 0.0;
    float dl = 28.0;
    for (int i = 0; i < 2; ++i) {
        vec3 sp = simCoord(x + uLightDir * (tl + 0.5 * dl));
        if (simInside(sp)) {
            vec4 o = simOpt(sp, 0.0);
            float sc = o.x + o.y;
            float cv;
            float dd = (sc > 1e-6) ? simCloudDensity(sp, o, aux, pixSize, cv) : 0.0;
            tau += (dd * max(sc, mix(SIG_CORE * 0.3, SIG_EDGE_ICE * 0.5, o.y / max(sc, 1e-9))) + o.z + o.w) * dl;
        }
        tl += dl;
        dl *= 2.6;
    }
    tauNear = tau;
    dl = max(uSimCell.z * 0.35, 40.0);
    for (int i = 0; i < 10; ++i) {
        if (i >= uLightSteps) break;
        vec3 xp = x + uLightDir * (tl + 0.5 * dl);
        vec3 sp = simCoord(xp);
        if (sp.z > uSimSize.z) { tl += dl; break; }
        if (simInside(sp)) tau += simTotal(simOpt(sp, 0.0)) * dl;
        tl += dl;
        dl *= 1.7;
    }
    return tau;
}

float simBackTau(vec3 x) {
    float tau = 0.0;
    float dl = max(uSimCell.z * 0.35, 40.0);
    float t = 0.0;
    for (int i = 0; i < 6; ++i) {
        vec3 sp = simCoord(x - uLightDir * (t + 0.5 * dl));
        if (sp.z < 0.0 || sp.z > uSimSize.z) break;
        if (simInside(sp)) tau += simTotal(simOpt(sp, 0.0)) * dl;
        t += dl;
        dl *= 1.7;
    }
    return tau;
}

vec3 flashLight(vec3 x) {
    if (uFlashPower <= 0.0) return vec3(0.0);
    float d = length(x - uFlashPos);
    // the diffuse glow of a flash seen through a cloud (Nubis 3's potential-
    // energy falloff, softened into a diffusion profile)
    return vec3(0.80, 0.86, 1.0) * uFlashPower * exp(-d / 1800.0) / (1.0 + d * d / 250000.0);
}

void marchSim(vec3 rd, float maxDist, float dither, out vec3 L, out float T, out float meanDist) {
    L = vec3(0.0);
    T = 1.0;
    meanDist = -1.0;
    if (uSimOn == 0) return;
    vec3 ro = -earthCentre();
    // the stretches of the ray inside the model's layer of the atmosphere:
    // from below, from inside it, or from above it (the view from a plane)
    vec4 seg;
    int nseg = shellSegments(ro, rd, uSimZLo, uSimZHi, min(maxDist, 300000.0), seg);
    if (nseg == 0) return;
    float cosT = dot(rd, uLightDir);
    vec3 pRain = phaseRain(cosT);
    float pSnow = phaseHG(cosT, 0.55);
    float pixAng = 2.0 * uTanHalfFov / max(uResolution.y, 1.0);
    float cell = max(uSimCell.x, uSimCell.z);
    float t = seg.x + dither * min(uSimCell.z, 120.0);
    float tEnd = seg.y;
    // no step longer than a twelfth of the stretch of the ray inside the
    // layer: a thin layer seen steeply from far above (a satellite's view)
    // is crossed in a few hundred metres, and a step of t/50 would jump it
    float capStep = max((seg.y - seg.x) / 12.0, 15.0);
    int part = 0;
    int tail = 0;
    float wsum = 0.0, dsum = 0.0;
    float lastEmpty = -1.0;
    bool refined = false;
    for (int i = 0; i < 1000; ++i) {
        if (i >= uSimSteps || T < 1e-4 || tail >= 32) break;
        if (t > tEnd) {
            if (part == 0 && nseg == 2) {
                part = 1; t = max(t, seg.z); tEnd = seg.w; lastEmpty = -1.0;
                capStep = max((seg.w - seg.z) / 12.0, 15.0);
            }
            else break;
        }
        vec3 x = rd * t;
        vec3 s = simCoord(x);
        float pixSize = max(t * pixAng, 1.0);
        if (!simInside(s)) {
            t += max(cell, t * 0.02);
            continue;
        }
        // empty-space skipping on a coarser mip the further away, so a ray
        // near the horizon can cross a hundred kilometres of the field
        float lev = clamp(2.0 + floor(log2(max(t, 1.0) / 15000.0)), 2.0, 4.0);
        vec4 oc = simOpt(s, lev);
        if (simTotal(oc) < 1e-6) {
            lastEmpty = t;
            t += min(max(0.5 * exp2(lev) * cell, t * 0.02), capStep * 1.5);
            continue;
        }
        // the field at the pixel's footprint: tricubic up close, the
        // matching mip level far away (a thin layer seen edge-on near the
        // horizon would otherwise alias into moire)
        float flod = log2(max(pixSize / uSimCell.x, 1.0));
        vec4 o = (pixSize < uSimCell.x * 0.6) ? simOptCubic(s) : simOpt(s, min(flod, 3.0));
        float keep = simKeep(x, s);
        vec4 aux = simAux(s);
        o.xy *= keep;
        if (o.z + o.w > 1e-6)
            o.zw *= simKeepPrecip(x, s, aux, o.w / max(o.z + o.w, 1e-9));
        float sc = o.x + o.y;
        float cov;
        float dcl = (sc > 1e-6) ? simCloudDensity(s, o, aux, pixSize, cov) : 0.0;
        float sigC = dcl * max(sc, mix(SIG_CORE * 0.3, SIG_EDGE_ICE * 0.5, o.y / max(sc, 1e-9)) * keep);
        if (o.z + o.w > 1e-6) {
            float snowy = o.w / max(o.z + o.w, 1e-9);
            float st = rainStreaks(s, aux, snowy);
            o.z *= st;
            o.w *= st;
        }
        float sigma = sigC + o.z + o.w;
        // The step: no finer than the pixel, and in thin cloud (an anvil
        // seen from underneath) as long as ~0.4 optical depths
        float ds = clamp(max(t * pixAng * 1.2, 0.4 / max(sigma, 1e-4)), min(18.0, uSimCell.z),
                         max(max(uSimCell.z, 60.0), t * 0.02));
        ds = min(ds, capStep);
        if (sigma < 2e-5) {
            lastEmpty = t;
            t += min(ds * 2.0, capStep * 1.5);
            refined = false;
            continue;
        }
        if (lastEmpty >= 0.0 && !refined && sigC > 1e-3) {
            // the edge of a cloud: bisect back from the last empty sample
            float lo = lastEmpty, hi = t;
            for (int b = 0; b < 4; ++b) {
                float mid = 0.5 * (lo + hi);
                vec3 xm = rd * mid;
                vec3 sm = simCoord(xm);
                vec4 om = simOpt(sm, 0.0) * keep;
                float cm;
                float dm = (om.x + om.y > 1e-6) ? simCloudDensity(sm, om, simAux(sm), pixSize, cm) : 0.0;
                if (dm * (om.x + om.y) + om.z + om.w > 2e-4) hi = mid; else lo = mid;
            }
            t = hi;
            refined = true;
            lastEmpty = -1.0;
            continue;
        }
        if (T < 0.01) {
            // opaque already: follow the transmittance only (see marchDecks)
            T *= exp(-sigma * ds);
            t += ds;
            ++tail;
            continue;
        }
        // --- lighting ---
        float tl, tauNear;
        float tauS = (simLightTau(x, aux, pixSize, tl, tauNear) + shadowTau(x + uLightDir * tl)) * keep;
        tauNear *= keep;
        // the diffuse field is thinned where the billow itself shades the point
        float occ = 0.35 + 0.65 * exp(-tauNear * 0.25);
        vec3 sunC = lightAt(x);
        float ice = o.y / max(sc, 1e-9);
        float g0 = mix(0.86, 0.76, ice);
        float alt = s.z + uSimZ0;
        // diffuse light from the sky above and the ground below, each
        // thinned by the cloud between (conservative two-stream transmission)
        float tauAbove = aux.x * keep;
        float tauDown = max(simAux(vec3(s.xy, 0.25 * uSimCell.z)).x - aux.x, 0.0) * keep;
        float amb1 = 1.0 / (1.0 + 0.75 * (1.0 - g0) * tauAbove);
        float amb2 = 1.0 / (1.0 + 0.75 * (1.0 - g0) * tauDown);
        vec3 skyA = skyIrradiance(alt) / PI * amb1;
        vec3 gndE = skyIrradiance(0.0) + lightAt(x - (ro + x) * (alt / length(ro + x))) * max(uLightDir.y, 0.0) * 0.7;
        vec3 gndA = gndE * uGroundAlbedo / PI * 0.5 * amb2;
        vec3 flash = flashLight(x);
        vec3 S = vec3(0.0);
        if (sigC > 0.0) {
            vec3 Lc = cloudRadiance(tauS, simBackTau(x) * keep, cosT, uJdELiq, ice, g0, sunC, sigC * 600.0, occ)
                    + skyA * (0.35 + 0.65 * saturate(1.0 - tauAbove * 0.02)) + gndA + flash;
            S += sigC * Lc;
        }
        // precipitation also sees the sunlight the cloud above lets through
        // as diffuse light: the two-stream total transmission less the part
        // that is still the direct beam (which the single-scattering term
        // already counts)
        float mu0 = max(uLightDir.y, 0.05);
        vec3 diffSun = sunC * mu0 / PI * max(amb1 - exp(-tauAbove / mu0), 0.0);
        if (o.z > 0.0) {
            vec3 Lr = sunC * exp(-tauS) * pRain * 0.95 + (skyA + diffSun + gndA) * 0.9 + flash;
            S += o.z * Lr;
        }
        if (o.w > 0.0) {
            vec3 Ls = sunC * exp(-tauS) * pSnow + skyA + diffSun + gndA + flash;
            S += o.w * Ls;
        }
        S /= sigma;
        float stepT = exp(-sigma * ds);
        L += T * S * (1.0 - stepT);
        dsum += t * (1.0 - stepT) * T;
        wsum += (1.0 - stepT) * T;
        T *= stepT;
        lastEmpty = -1.0;
        t += ds;
    }
    meanDist = (wsum > 1e-5) ? dsum / wsum : -1.0;
}

// ------------------------------------------------------------ sun and moon --
float hash21(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// A star field with the real magnitude distribution: the number of stars
// brighter than magnitude m goes as 10^(0.6 m), about 9000 to magnitude 6.5.
vec3 starField(vec3 rd) {
    if (uShowStars == 0) return vec3(0.0);
    vec3 e = normalize(uHorToEqu * rd);
    float dec = asin(clamp(e.z, -1.0, 1.0));
    float ra = atan(e.y, e.x);
    const float CELL = 0.0087266;
    const float MMIN = -1.4, MMAX = 6.5;
    float a0 = pow(10.0, 0.6 * MMIN);
    float a1 = pow(10.0, 0.6 * MMAX);
    float row = floor(dec / CELL);
    float pixAng = 2.0 * uTanHalfFov / max(uResolution.y, 1.0);
    float sigma = max(0.00055, pixAng * 0.55);
    vec3 acc = vec3(0.0);
    for (int j = -1; j <= 1; ++j) {
        float r0 = row + float(j);
        float d0 = (r0 + 0.5) * CELL;
        if (abs(d0) > 1.5708) continue;
        float cosd = max(cos(d0), 0.02);
        float dra = CELL / cosd;
        float col = floor(ra / dra);
        for (int i = -1; i <= 1; ++i) {
            vec2 cid = vec2(col + float(i), r0);
            float h = hash21(cid);
            if (h > 0.06) continue;
            float u = max(hash21(cid + 31.7), 1e-4);
            float mag = log(u * (a1 - a0) + a0) / log(10.0) / 0.6;
            float jr = (hash21(cid + 7.1) - 0.5) * dra;
            float jd = (hash21(cid + 13.3) - 0.5) * CELL;
            float sra = (col + float(i) + 0.5) * dra + jr;
            float sd = d0 + jd;
            vec3 sp = vec3(cos(sd) * cos(sra), cos(sd) * sin(sra), sin(sd));
            float ang = acos(clamp(dot(sp, e), -1.0, 1.0));
            float bright = 2.0e-4 * pow(10.0, -0.4 * mag);
            vec3 tint = mix(vec3(0.72, 0.82, 1.0), vec3(1.0, 0.82, 0.62), hash21(cid + 23.9));
            acc += tint * bright * exp(-(ang * ang) / (2.0 * sigma * sigma));
        }
    }
    return acc;
}

vec3 sunAndMoon(vec3 rd) {
    vec3 c = vec3(0.0);
    float aSun = acos(clamp(dot(rd, uSunDir), -1.0, 1.0));
    if (aSun < uSunAngRad) {
        float x = aSun / uSunAngRad;
        float limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - x * x, 0.0)));
        c += vec3(120.0) * limb;
    }
    float aMoon = acos(clamp(dot(rd, uMoonDir), -1.0, 1.0));
    if (aMoon < uMoonAngRad) {
        vec3 up = normalize(uMoonBrightDir - uMoonDir * dot(uMoonBrightDir, uMoonDir));
        vec3 rt = normalize(cross(uMoonDir, up));
        vec3 loc = normalize(rd) - uMoonDir;
        float sx = dot(loc, up) / uMoonAngRad;
        float sy = dot(loc, rt) / uMoonAngRad;
        float cosG = 2.0 * uMoonIllum - 1.0;
        float sinG = sqrt(max(1.0 - cosG * cosG, 0.0));
        float nz = sqrt(max(1.0 - sx * sx - sy * sy, 0.0));
        float lit = smoothstep(-0.04, 0.04, sx * sinG + nz * cosG);
        float shade = 0.55 + 0.45 * nz;
        c += vec3(0.060, 0.058, 0.052) * (lit * shade + 0.012);
    }
    c += starField(rd);
    return c;
}

// ------------------------------------------------ clouds above the weather --
// Noctilucent cloud: the wave field of the mesopause, in the four forms of
// Fogle & Haurwitz (1966) - veils, bands (gravity-wave crests tens of km
// apart), billows (short waves of a few km) and whirls - drifting with the
// summer mesospheric wind.
uniform float uLatSign;              // +1 northern hemisphere, -1 southern
// fp: the pixel's footprint on the shell, m - a wave finer than about a
// third of it is replaced by its mean (it would alias into a moire)
float wave(float phase, float lambda, float fp) {
    float w = clamp(fp / (0.33 * lambda) - 0.5, 0.0, 1.0);
    return mix(0.5 + 0.5 * cos(phase), 0.5, w);
}
float nlcField(vec2 q, float fp) {
    vec2 x = q - uUpperWind * uUpperTime;
    // The display: a patch of the noctilucent belt pole-ward of the
    // observer, seen low over the twilight horizon as in most sightings.
    vec2 c = vec2(0.0, 550000.0 * uLatSign);
    float patch = exp(-pow(length((q - c) / vec2(750000.0, 300000.0)), 2.0));
    if (patch < 1e-3) return 0.0;
    // Brightness varies over hundreds of km with the gravity-wave packets
    // that cool the mesopause (and so grow the ice) more in some places.
    float e1 = 0.5 + 0.5 * sin(dot(x, vec2(0.83, 0.55)) * 2.0 * PI / 260000.0 + 0.4);
    float e2 = 0.5 + 0.5 * sin(dot(x, vec2(-0.45, 0.89)) * 2.0 * PI / 190000.0 + 2.2);
    // Type I, the veil: the ice sheet itself, present wherever the display is.
    float f = ((uNlcTypes & 1) != 0 ? 0.55 : 0.30) * (0.45 + 0.55 * e1 * (0.6 + 0.4 * e2));
    if ((uNlcTypes & 2) != 0) {
        // Type II, bands: the veil thickened in the crests of gravity waves
        // 20-60 km apart, two sets crossing (the herringbone).  They modulate
        // the veil; they are not separate lines in an empty sky.
        // the crests are not straight: the larger waves the bands ride on
        // bend them (a smooth warp of their phase)
        float warp = textureLod(tDetail, vec3(x / 400000.0, 0.83), 0.0).g - 0.5;
        float warp2 = textureLod(tDetail, vec3(x / 250000.0 + 0.5, 0.21), 0.0).b - 0.5;
        float b1 = wave(dot(x, normalize(vec2(0.35, 1.0))) * 2.0 * PI / 42000.0 + 9.0 * warp,
                        42000.0, fp);
        float b2 = wave(dot(x, normalize(vec2(-0.55, 1.0))) * 2.0 * PI / 27000.0 + 1.3 + 9.0 * warp2,
                        27000.0, fp);
        f *= 0.45 + 0.55 * mix(b1, b2, 0.35 + 0.3 * e2) * (0.8 + 0.4 * e1);
    }
    if ((uNlcTypes & 4) != 0) {
        // Type III, billows: short waves of 3-10 km where the bands break
        // (Kelvin-Helmholtz and convective instabilities at the mesopause);
        // their fine turbulent texture from the cellular noise texture.
        float env = e2 * (0.5 + 0.5 * cos(x.x * 2.0 * PI / 90000.0 + 0.7));
        float bl = wave(dot(x, normalize(vec2(1.0, 0.2))) * 2.0 * PI / 6000.0, 6000.0, fp);
        float lod = log2(clamp(fp / (24000.0 / 32.0), 1.0, 32.0));
        float tex = textureLod(tDetail, vec3(x / 24000.0, 0.37), lod).r;
        f *= 1.0 + 0.45 * env * (bl - 0.5) + 0.35 * (tex - 0.5);
    }
    if ((uNlcTypes & 8) != 0) {
        // Type IV, whirls: an eddy tens of km across wound into the sheet.
        vec2 d = x - vec2(60000.0, 520000.0 * uLatSign);
        float r = length(d);
        float a = atan(d.y, d.x);
        f *= 1.0 + 0.6 * exp(-r / 60000.0) * (wave(r * 2.0 * PI / 18000.0 - 2.0 * a, 18000.0, fp) - 0.4);
    }
    return max(f, 0.0) * patch;
}

// Nacreous cloud.  Lenticular: separate lenses standing in the crests of
// mountain lee waves that reach the stratosphere - crests 20-30 km apart
// along the wind, each lens some tens of km long across it and a few km
// wide, thickest in its middle, where its crystals have grown largest.
// Cirriform: a thin fibrous sheet drawn out along the wind.
float pscField(vec2 q, out float grow) {
    vec2 w = normalize(uPscWind + vec2(1e-3, 0.0));
    vec2 nrm = vec2(-w.y, w.x);
    grow = 0.0;
    // the display: the lee of one mountain range, some 100 km across,
    // centred 75 km from the observer toward the Sun's side of the sky
    vec2 sunH = normalize(vec2(uSunDir.x, uSunDir.z) + vec2(1e-4, 0.0));
    float region = exp(-pow(length((q - sunH * 75000.0) / vec2(55000.0, 45000.0)), 2.0));
    if (region < 0.02) return 0.0;
    // uPscLens: how much the stratospheric wind makes lee-wave lenses of it
    // (it goes over from one form to the other as the wind changes)
    float fL = 0.0, gL = 0.0, fC = 0.0, gC = 0.0;
    if (uPscLens > 0.0) {
        // the lenses stand in the wave crests while the air blows through
        float a = dot(q, w), b = dot(q, nrm);
        const float CREST = 20000.0, SEG = 45000.0;
        float ia = floor(a / CREST + 0.5);
        float fa = a - ia * CREST;
        // each crest broken into lenses; look at this segment and its neighbours
        for (int k = -1; k <= 1; ++k) {
            float jb = floor(b / SEG) + float(k);
            float h1 = hash21(vec2(ia, jb) + 0.37);
            float h2 = hash21(vec2(ia, jb) + 5.11);
            float h3 = hash21(vec2(ia, jb) + 9.73);
            if (h1 < 0.25) continue;                      // not every crest has cloud
            float len = SEG * (0.5 + 0.5 * h2);
            float cen = (jb + 0.5 + 0.3 * (h3 - 0.5)) * SEG;
            float wid = 2500.0 + 3000.0 * h3;
            // the crests bend with the ridge below and shift with the wave's
            // phase from one lens to the next
            float slope = 0.35 * (h1 - 0.6);
            float u = (b - cen) / (0.5 * len);
            float v = (fa - 6000.0 * (h2 - 0.5) - slope * (b - cen)) / wid;
            float r2 = u * u + v * v;
            if (r2 < 1.0) {
                float lens = pow(1.0 - r2, 1.2) * (0.6 + 0.4 * h1);
                if (lens > fL) { fL = lens; gL = sqrt(1.0 - r2); }
            }
        }
    }
    if (uPscLens < 1.0) {
        // the cirriform sheet drifts with the wind
        vec2 x = q - uPscWind * uUpperTime;
        float a = dot(x, w), b = dot(x, nrm);
        float fib = textureLod(tDetail, vec3(a / 90000.0, b / 12000.0, 0.61), 0.0).r;
        float env = 0.5 + 0.5 * sin(b * 2.0 * PI / 70000.0 + 0.3 * sin(a * 2.0 * PI / 150000.0));
        gC = fib;
        fC = env * (0.35 + 0.65 * fib) * 0.6;
    }
    float k = clamp(uPscLens, 0.0, 1.0);
    grow = mix(gC, gL, k);
    return mix(fC, fL, k) * region;
}

// Single scattering by an optically thin shell at height h: the sunlight
// that reaches it (Earth's shadow and the atmosphere's colour included)
// times optical depth over the slant.
vec3 upperClouds(vec3 rd) {
    vec3 L = vec3(0.0);
    if (uNlc <= 0.0 && uNacreous <= 0.0 && uNat <= 0.0) return L;
    vec3 ro = -earthCentre();
    float cosT = dot(rd, uSunDir);
    if (uNlc > 0.0) {
        vec2 h = raySphere(ro, rd, RG + 82500.0);
        if (h.y > 0.0) {
            vec3 p = rd * h.y;
            vec3 n = normalize(p + ro);
            float mu = max(dot(rd, n), 0.05);
            float pixAng = 2.0 * uTanHalfFov / max(uResolution.y, 1.0);
            float f = nlcField(p.xz, h.y * pixAng / mu);
            float tau = 3.0e-3 * uNlc * f;
            float ph = 0.6 * phaseHG(cosT, 0.2) + 0.4 * phaseRayleigh(cosT);
            // particles of ~50 nm scatter as lambda^-3.5: blue
            L += sunAt(p) * vec3(0.72, 1.0, 1.45) * ph * tau / mu;
        }
    }
    if (uNacreous > 0.0) {
        vec2 h = raySphere(ro, rd, RG + 22000.0);
        if (h.y > 0.0) {
            vec3 p = rd * h.y;
            vec3 n = normalize(p + ro);
            float mu = max(dot(rd, n), 0.05);
            float grow;
            float f = pscField(p.xz, grow);
            // ~5 ice particles per cm3 of 1-3 um in a layer a few km deep
            float tau = 0.25 * uNacreous * f;
            // iridescence: nearly equal ice spheres whose radius grows from
            // ~1 um at the lens's edge, where they have just formed, to
            // ~2.5 um in its middle; each size sends each wavelength to its
            // own angles (the Mie table), so the colours run in bands along
            // the lens and change with the angle from the Sun
            float arad = 1.1 + 1.1 * grow;
            vec3 ph = phasePsc(cosT, arad);
            L += sunAt(p) * ph * tau / mu;
        }
    }
    if (uNat > 0.0) {
        vec2 h = raySphere(ro, rd, RG + 20000.0);
        if (h.y > 0.0) {
            vec3 p = rd * h.y;
            vec3 n = normalize(p + ro);
            float mu = max(dot(rd, n), 0.05);
            float f = 0.7 + 0.3 * sin(dot(p.xz, vec2(0.6, 0.8)) * 2.0 * PI / 120000.0);
            L += sunAt(p) * phaseHG(cosT, 0.6) * 0.004 * uNat * f / mu;
        }
    }
    return L;
}

// ---------------------------------------------------------------- ground ----
// Fields, woods and roads at a few scales, so that the shadows of the clouds
// have something to fall on.
vec3 groundAlbedo(vec2 xz, float pixSize) {
    // fields and woods at three scales, bent by a slow warp and each grid
    // turned its own way, so that from an aircraft or a satellite they are
    // a patchwork and not a checkerboard; each scale fades to its mean once
    // a pixel covers several patches (else it aliases)
    vec2 wv = vec2(textureLod(tDetail, vec3(xz / 23000.0, 0.13), 0.0).r,
                   textureLod(tDetail, vec3(xz / 23000.0 + 0.37, 0.61), 0.0).g) - 0.5;
    vec2 q = xz + wv * 3000.0;
    vec2 q1 = mat2(0.94, 0.34, -0.34, 0.94) * q;
    vec2 q2 = mat2(0.81, -0.59, 0.59, 0.81) * q;
    vec2 q3 = mat2(0.99, 0.14, -0.14, 0.99) * q;
    float fa = 1.0 - smoothstep(600.0, 1800.0, pixSize);
    float fb = 1.0 - smoothstep(140.0, 420.0, pixSize);
    float fc = 1.0 - smoothstep(30.0, 95.0, pixSize);
    float a = mix(0.5, hash13(vec3(floor(q1 / 1800.0), 1.0)), fa);
    float b = mix(0.5, hash13(vec3(floor(q2 / 420.0), 2.0)), fb);
    float c = mix(0.5, hash13(vec3(floor(q3 / 95.0), 3.0)), fc);
    vec3 grass = vec3(0.20, 0.30, 0.10);
    vec3 wood  = vec3(0.07, 0.13, 0.05);
    vec3 crop  = vec3(0.30, 0.28, 0.14);
    float isWood = mix(0.25, smoothstep(0.55, 0.75, a), fa);
    float isCrop = mix(0.2, smoothstep(0.62, 0.80, b), fb) * (1.0 - isWood);
    vec3 col = mix(grass, wood, isWood);
    col = mix(col, crop, isCrop);
    col *= 0.85 + 0.3 * c;
    if (uDry > 0.0) {
        // Dry ground (the forecast's root-zone soil water; the forecast has no
        // land cover, so this is the program's proxy): the fields turn to dry
        // grass and then bare sand (albedo 0.35), the woods thin to scrub and
        // are gone in a desert.
        vec3 grassDry = vec3(0.36, 0.31, 0.18) * (0.9 + 0.2 * b);
        vec3 sand = vec3(0.60, 0.49, 0.34) * (0.92 + 0.16 * c);
        vec3 scrub = vec3(0.24, 0.21, 0.13);
        vec3 dryCol = mix(grassDry, sand, smoothstep(0.45, 1.0, uDry));
        dryCol = mix(dryCol, scrub, isWood * (1.0 - smoothstep(0.6, 1.0, uDry)));
        col = mix(col, dryCol, smoothstep(0.0, 0.5, uDry));
    }
    if (uSnow > 0.0) {
        // Lying snow (the forecast's snow depth; cover = depth / 0.1 m as in
        // the ECMWF land surface, Dutra et al. 2010).  Open ground turns white
        // - albedo 0.75, between the 0.5 of old snow and the 0.85 of fresh -
        // woods only grey, ~0.3, their crowns shading the snow under them.
        // A partial cover lies field by field, not as a grey wash.
        float nb = mix(0.5, hash13(vec3(floor(q2 / 420.0), 7.0)), fb);
        float patchy = smoothstep(nb - 0.05, nb + 0.05, uSnow * 1.1 - 0.05);
        float snowy = mix(uSnow, patchy, fb);
        vec3 snowCol = mix(vec3(0.73, 0.75, 0.79) * (0.94 + 0.12 * c), vec3(0.28, 0.30, 0.33), isWood);
        col = mix(col, snowCol, snowy);
    }
    return col;
}

vec3 groundColour(vec3 rd, float dist) {
    vec3 p = rd * dist;
    vec3 n = normalize(p - earthCentre());
    float ndl = max(dot(n, uLightDir), 0.0);
    float pixAngG = 2.0 * uTanHalfFov / max(uResolution.y, 1.0);
    vec3 albedo = groundAlbedo(p.xz, dist * pixAngG / max(abs(rd.y), 0.02));
    vec3 sky = skyIrradiance(0.0);
    // Under cloud the sky is dimmed by the two-stream diffuse transmission of
    // the column overhead, and part of the blocked sunbeam comes back as
    // grey, diffuse light scattered through and around the cloud.
    float tDiff = 1.0;
    if (uSimOn != 0) {
        vec3 s = simCoord(p);
        s.z = 0.25 * uSimCell.z;
        if (simInside(s)) tDiff = 1.0 / (1.0 + 0.75 * 0.14 * simAux(s).x * simKeep(p, s));
    }
    vec3 sun0 = lightAt(p);
    float tauSh = shadowTau(p);
    float tsh = exp(-tauSh);
    // the Atlas layers overhead dim the sky's light too: their vertical
    // optical depth from the shadow map's slant one (slant tau x mu0, as for
    // the air under cloud) - without it the ground under a stratus overcast
    // kept the whole blue sky's light and turned blue under snow
    tDiff = min(tDiff, 1.0 / (1.0 + 0.105 * tauSh * max(uLightDir.y, 0.05)));
    // the part of the blocked beam that the cloud forward-scatters on to the
    // ground: two-stream total transmission along the Sun's path, less the
    // direct beam itself
    float tDiffSun = 1.0 / (1.0 + 0.75 * 0.14 * tauSh);
    // Light goes back and forth between the ground and the cloud base above
    // it: the ground receives 1 / (1 - albedo x R) times more, R = 1 - tDiff
    // the cloud's diffuse reflectance.  Over snow under an overcast that is
    // 1.6 x - why a snowfield under stratus is nearly as bright as the sky.
    float bounce = 1.0 / (1.0 - uGroundAlbedo * (1.0 - tDiff));
    vec3 diffuse = (sky * tDiff + sun0 * ndl * max(tDiffSun - tsh, 0.0)) * bounce;
    return albedo / PI * (sun0 * tsh * ndl + diffuse) + flashLight(p) * albedo * 0.05;
}

// ------------------------------------------------------------- lightning ---
vec3 boltEmission(vec3 rd, float pixAng, out float boltDist) {
    boltDist = -1.0;
    if (uBoltN < 2) return vec3(0.0);
    vec3 acc = vec3(0.0);
    float best = 1e9;
    for (int i = 0; i < 47; ++i) {
        if (i + 1 >= uBoltN) break;
        vec3 a = uBolt[i].xyz, b = uBolt[i + 1].xyz;
        float bri = uBolt[i + 1].w;
        if (bri <= 0.0) continue;
        // closest approach between the view ray and the segment
        vec3 u = b - a;
        float uu = dot(u, u);
        float ud = dot(u, rd);
        float ua = dot(u, a);
        float da = dot(rd, a);
        float den = uu - ud * ud;
        float s = (den > 1e-6) ? clamp((ud * da - ua) / den, 0.0, 1.0) : 0.0;
        vec3 q = a + u * s;
        float tq = max(dot(q, rd), 1.0);
        float ang = length(q - rd * tq) / tq;
        float w = pixAng * 0.9;
        acc += vec3(0.85, 0.9, 1.0) * bri * (exp(-(ang * ang) / (w * w)) * 60.0
                                               + 0.6 * exp(-ang / 0.02));
        if (ang < w * 2.0) best = min(best, tq);
    }
    if (best < 1e8) boltDist = best;
    return acc;
}

// ------------------------------------------------------------------ main ---
void main() {
    // uPixJitter moves the ray within its pixel from one accumulated frame to
    // the next, so that a still frame converges to the pixel's average and
    // not to its centre: without it the edges of moonlit cloudlets against a
    // black sky stay stair-stepped however many frames are averaged
    vec2 ndc = (v_uv * 2.0 - 1.0) + uPixJitter;
    vec3 rd = normalize(uCamFwd
                        + uCamRight * ndc.x * uTanHalfFov * uAspect
                        + uCamUp * ndc.y * uTanHalfFov);
    vec3 ro = -earthCentre();
    vec2 gnd = raySphere(ro, rd, groundR());
    float groundDist = (gnd.y > 0.0 && gnd.x > 0.0) ? gnd.x : -1.0;
    vec2 topHit = raySphere(ro, rd, RT);
    float far = (groundDist > 0.0) ? groundDist : max(topHit.y, 0.0);
    float dither = fract(texture(tBlue, gl_FragCoord.xy / 64.0).r + uFrameJitter);
    float pixAng = 2.0 * uTanHalfFov / max(uResolution.y, 1.0);

    // 1. clouds: the Atlas decks and the simulated field
    vec3 Ld, Ls; float Td, Ts, dd, dsim, haloW, coronaW;
    float maxD = (groundDist > 0.0) ? groundDist : 400000.0;
    marchDecks(rd, maxD, dither, Ld, Td, dd, haloW, coronaW);
    marchSim(rd, maxD, dither, Ls, Ts, dsim);
    vec3 Lc;
    float Tc = Td * Ts;
    float dCloud;
    if (dsim >= 0.0 && (dd < 0.0 || dsim < dd)) Lc = Ls + Ts * Ld;
    else Lc = Ld + Td * Ls;
    float wd = (dd >= 0.0) ? (1.0 - Td) : 0.0;
    float ws = (dsim >= 0.0) ? (1.0 - Ts) : 0.0;
    dCloud = (wd + ws > 1e-5) ? (wd * max(dd, 0.0) + ws * max(dsim, 0.0)) / (wd + ws) : -1.0;
    if (dCloud < 0.0) dCloud = min(far, 40000.0);
    dCloud = min(dCloud, far);

    // 2. the air in front of the clouds and behind them
    vec3 Ln, Tn, Lf, Tf;
    float jit = fract(dither * 1.618 + 0.37);
    atmosphere(rd, 0.0, dCloud, uAtmSteps, jit, Ln, Tn);
    atmosphere(rd, dCloud, far, max(uAtmSteps / 2, 6), jit, Lf, Tf);

    // 3. what is behind everything
    vec3 back = (groundDist > 0.0) ? groundColour(rd, groundDist) : sunAndMoon(rd) + upperClouds(rd);

    // 4. optical phenomena made by thin ice and water decks
    // Each is weighted by how much of this ray's light comes from a deck
    // that makes it, through whatever is in front (haloW, coronaW); a model
    // cloud nearer than the decks hides them as well.
    vec3 optic = vec3(0.0);
    float aSun = degrees(acos(clamp(dot(rd, uSunDir), -1.0, 1.0)));
    if (uHaloStrength > 0.0 && haloW > 0.0) {
        float inner = smoothstep(21.2, 22.2, aSun);
        float halo = inner * exp(-pow((aSun - 22.6) / 1.5, 2.0));
        float outer = smoothstep(22.5, 27.0, aSun) * 0.16 * exp(-(aSun - 22.5) / 16.0);
        optic += (vec3(1.35, 1.02, 0.80) * halo * 0.42 + vec3(0.55) * outer) * uHaloStrength * haloW;
    }
    if (uCoronaStrength > 0.0 && coronaW > 0.0) {
        float cor = exp(-pow(aSun / 2.2, 2.0)) * 0.55
                  + exp(-pow((aSun - 4.6) / 1.5, 2.0)) * 0.20;
        optic += vec3(1.0, 0.86, 1.12) * cor * uCoronaStrength * coronaW;
    }
    if (dsim >= 0.0 && (dd < 0.0 || dsim < dd)) optic *= Ts;
    // the colour of the sunlight (not its strength), and gone with the Sun:
    // it fades over the Sun's last two degrees (it was cut off at the
    // horizon, and a halo still at full strength vanished in a frame)
    optic *= sunAt(vec3(0.0)) / max(sunAt(vec3(0.0)).g, 1e-3) * smoothstep(0.0, 0.035, uSunDir.y);

    // 5. lightning channels, hidden by any cloud in front of them
    float bd;
    vec3 bolt = boltEmission(rd, pixAng, bd);
    if (bd > 0.0 && dCloud < bd) bolt *= Tc;

    vec3 colour = Ln + Tn * (Lc + optic + Tc * (Lf + Tf * back)) + bolt;
    f_colour = vec4(max(colour, vec3(0.0)) * uExposure, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Display: temporal accumulation, bloom, tone mapping
# ---------------------------------------------------------------------------
ACCUM_FRAG = """#version 330
in vec2 v_uv; out vec4 f_colour;
uniform sampler2D tPrev; uniform sampler2D tNew;
uniform float uMix, uAspect;
uniform vec3 cR, cU, cF, pR, pU, pF;
uniform float cT, pT;
void main(){
    vec3 nw = texture(tNew, v_uv).rgb;
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(cF + cR * ndc.x * cT * uAspect + cU * ndc.y * cT);
    float z = dot(rd, pF);
    if (z <= 1e-4) { f_colour = vec4(nw, 1.0); return; }
    vec2 pn = vec2(dot(rd, pR) / (z * pT * uAspect), dot(rd, pU) / (z * pT));
    vec2 puv = pn * 0.5 + 0.5;
    if (any(lessThan(puv, vec2(0.002))) || any(greaterThan(puv, vec2(0.998)))) {
        f_colour = vec4(nw, 1.0); return;
    }
    vec3 old = texture(tPrev, puv).rgb;
    f_colour = vec4(mix(old, nw, uMix), 1.0);
}
"""

# Jimenez (SIGGRAPH 2014) bloom as in LearnOpenGL's physically based bloom:
# a 13-tap downsample (Karis-averaged on the first level) and a tent upsample.
BLOOM_DOWN = """#version 330
in vec2 v_uv; out vec4 f_colour;
uniform sampler2D tSrc;
uniform vec2 uTexel;
uniform int uKaris;
vec3 k(vec3 c) { float l = dot(c, vec3(0.2126, 0.7152, 0.0722)) * 0.25; return c / (1.0 + l); }
void main() {
    vec2 x = uTexel;
    vec3 a = texture(tSrc, v_uv + vec2(-2, 2) * x).rgb;
    vec3 b = texture(tSrc, v_uv + vec2(0, 2) * x).rgb;
    vec3 c = texture(tSrc, v_uv + vec2(2, 2) * x).rgb;
    vec3 d = texture(tSrc, v_uv + vec2(-2, 0) * x).rgb;
    vec3 e = texture(tSrc, v_uv).rgb;
    vec3 f = texture(tSrc, v_uv + vec2(2, 0) * x).rgb;
    vec3 g = texture(tSrc, v_uv + vec2(-2, -2) * x).rgb;
    vec3 h = texture(tSrc, v_uv + vec2(0, -2) * x).rgb;
    vec3 i = texture(tSrc, v_uv + vec2(2, -2) * x).rgb;
    vec3 j = texture(tSrc, v_uv + vec2(-1, 1) * x).rgb;
    vec3 kk = texture(tSrc, v_uv + vec2(1, 1) * x).rgb;
    vec3 l = texture(tSrc, v_uv + vec2(-1, -1) * x).rgb;
    vec3 m = texture(tSrc, v_uv + vec2(1, -1) * x).rgb;
    // Jimenez's Karis average is left out on purpose: it exists to stop
    // single bright pixels from flickering, and the brightest single pixel
    // here is the Sun, whose glare is exactly what the bloom is for.  The
    // first level clamps instead, and temporal accumulation handles flicker.
    vec3 o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + kk + l + m) * 0.125;
    if (uKaris == 1) o = min(o, vec3(4000.0));
    f_colour = vec4(max(o, vec3(0.0)), 1.0);
}
"""

BLOOM_UP = """#version 330
in vec2 v_uv; out vec4 f_colour;
uniform sampler2D tSrc;
uniform vec2 uTexel;
uniform float uRadius;
void main() {
    vec2 x = uTexel * uRadius;
    vec3 a = texture(tSrc, v_uv + vec2(-x.x, x.y)).rgb;
    vec3 b = texture(tSrc, v_uv + vec2(0.0, x.y)).rgb;
    vec3 c = texture(tSrc, v_uv + vec2(x.x, x.y)).rgb;
    vec3 d = texture(tSrc, v_uv + vec2(-x.x, 0.0)).rgb;
    vec3 e = texture(tSrc, v_uv).rgb;
    vec3 f = texture(tSrc, v_uv + vec2(x.x, 0.0)).rgb;
    vec3 g = texture(tSrc, v_uv + vec2(-x.x, -x.y)).rgb;
    vec3 h = texture(tSrc, v_uv + vec2(0.0, -x.y)).rgb;
    vec3 i = texture(tSrc, v_uv + vec2(x.x, -x.y)).rgb;
    vec3 o = e * 4.0 + (b + d + f + h) * 2.0 + (a + c + g + i);
    f_colour = vec4(o / 16.0, 1.0);
}
"""

# AgX (Troy Sobotka), in the three.js / iolite form; Khronos PBR Neutral as
# the alternative; exact sRGB transfer function on the way out.
TONEMAP = """#version 330
in vec2 v_uv; out vec4 f_colour;
uniform sampler2D tHdr;
uniform sampler2D tBloom;
uniform float uBloom;
uniform int uTone;                  // 0 AgX, 1 AgX punchy, 2 PBR Neutral
uniform float uDither;

const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(vec3(0.6274, 0.0691, 0.0164),
    vec3(0.3293, 0.9195, 0.0880), vec3(0.0433, 0.0113, 0.8956));
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(vec3(1.6605, -0.1246, -0.0182),
    vec3(-0.5876, 1.1329, -0.1006), vec3(-0.0728, -0.0083, 1.1187));

vec3 agxContrast(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
         + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 agx(vec3 color, bool punchy) {
    const mat3 AgXInsetMatrix = mat3(vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
    const mat3 AgXOutsetMatrix = mat3(vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
    const float AgxMinEv = -12.47393;
    const float AgxMaxEv = 4.026069;
    color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
    color = AgXInsetMatrix * color;
    color = max(color, 1e-10);
    color = log2(color);
    color = (color - AgxMinEv) / (AgxMaxEv - AgxMinEv);
    color = clamp(color, 0.0, 1.0);
    color = agxContrast(color);
    if (punchy) {
        // Filament's PUNCHY look: power 1.35, saturation 1.4 (ASC CDL)
        color = pow(max(color, 0.0), vec3(1.35));
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = luma + 1.4 * (color - luma);
    }
    color = AgXOutsetMatrix * color;
    color = pow(max(vec3(0.0), color), vec3(2.2));
    color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
    return clamp(color, 0.0, 1.0);
}

vec3 pbrNeutral(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;
    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, newPeak * vec3(1.0), g);
}

vec3 srgbOETF(vec3 c) {
    return mix(pow(c, vec3(0.41666)) * 1.055 - 0.055, c * 12.92, lessThanEqual(c, vec3(0.0031308)));
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec3 hdr = texture(tHdr, v_uv).rgb;
    vec3 bl = texture(tBloom, v_uv).rgb;
    vec3 c = mix(hdr, bl, uBloom);
    vec3 o;
    if (uTone == 2) o = clamp(pbrNeutral(c), 0.0, 1.0);
    else o = agx(c, uTone == 1);
    o = srgbOETF(o);
    // a least-significant-bit dither hides banding in the smooth sky
    o += (hash(gl_FragCoord.xy + uDither) - 0.5) / 255.0;
    f_colour = vec4(clamp(o, 0.0, 1.0), 1.0);
}
"""

PRESENT = """#version 330
in vec2 v_uv; out vec4 f_colour;
uniform sampler2D tSrc;
void main(){ f_colour = vec4(texture(tSrc, v_uv).rgb, 1.0); }
"""
