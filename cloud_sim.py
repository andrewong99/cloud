#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""CloudSim - a ground-to-sky cloud and wind simulator based on the WMO
International Cloud Atlas.

    python cloud_sim.py                     live weather for Kuala Lumpur: the sky
                                            its data call for, grown by the cloud
                                            model, with the reasons shown
    python cloud_sim.py --place 51.5,-0.13 --name London
    python cloud_sim.py --place Paris       any of the world's 1,584 major cities, its
                                            own clock (summer time included); in the
                                            program the place, the date and the hour
                                            are dropdowns on the panel
    python cloud_sim.py --offline           no network, synthetic sounding
    python cloud_sim.py --sim streets       one cloud form's physics (see --help)
    python cloud_sim.py --sim none          Atlas layers only, no model
    python cloud_sim.py --selftest          run the checks and exit
"""
import argparse
import re
import sys
from datetime import datetime, timezone

# Windows consoles still default to a legacy code page; the self-test prints
# "étage" and degree signs, which would otherwise raise UnicodeEncodeError.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", default=None,
                    help='a city (Paris, "New York") or latitude,longitude (48.86,2.35; '
                         '3.14N 101.69E); default Kuala Lumpur')
    ap.add_argument("--name", default=None,
                    help="the place's name on the panel (default: the city's, or the "
                         "nearest city's)")
    ap.add_argument("--tz", type=float, default=None,
                    help="hours from UTC for the clock (default: the place's own time zone, "
                         "summer time included)")
    ap.add_argument("--time", default=None, help="UTC start time, e.g. 2026-09-10T06:00")
    ap.add_argument("--quality", default="medium",
                    choices=["low", "medium", "high", "photo"])
    ap.add_argument("--size", default="1280x760")
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    ap.add_argument("--sim", default=None,
                    choices=["auto", "none", "weather", "supercell", "squall", "bomex",
                             "stratocumulus",
                             "altocumulus", "altocumulus_ra", "altocumulus_un", "streets",
                             "asperitas", "virga"],
                    help="what the cloud-resolving model grows (default: auto, the regime "
                         "the weather data call for)")
    ap.add_argument("--sim-size", default=None, choices=["fast", "standard", "fine"],
                    help="model grid (default: standard on a GPU, fast on the CPU)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # "--place -33.87,151.21": a latitude south is a value, not an option
    argv = list(sys.argv[1:])
    for i in range(len(argv) - 1):
        if argv[i] == "--place" and re.match(r"-\d", argv[i + 1]):
            argv[i:i + 2] = ["--place=" + argv[i + 1]]
            break
    args = ap.parse_args(argv)

    if args.selftest:
        from cloudsim import selftest
        return selftest.main(verbose=args.verbose)

    from cloudsim import places, splash
    if args.place is None:
        name, lat, lon, clock = "Kuala Lumpur", 3.1390, 101.6869, "Asia/Kuala_Lumpur"
    else:
        c = places.parse_coords(args.place)
        if c is not None:
            lat, lon, hrs = c
            name = places.name_for(lat, lon)
            clock = hrs if hrs is not None else places.zone_at(lat, lon)[0]
        else:
            city = places.find(args.place)
            if city is None:
                print(f"--place {args.place!r}: no such city in the list, and not "
                      "latitude,longitude (e.g. 48.86,2.35)", file=sys.stderr)
                return 2
            name, lat, lon, clock = city.label, city.lat, city.lon, city.zone
    if args.name:
        name = args.name
    if args.tz is not None:
        clock = args.tz                      # fixed hours from UTC, as asked
    w, h = (int(x) for x in args.size.lower().split("x"))
    when = None
    if args.time:
        when = datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)

    # the loading window first: it is up while the program imports, opens
    # its window, compiles its shaders, fetches the weather and builds the
    # sky, and closes the moment the program's window shows
    coords = f"{lat:.3f}, {lon:.3f}"
    loader = splash.Loader.start(coords if name == coords else f"{name}  ·  {coords}")
    loader.step(0.02, "Starting CloudSim", 0.08)
    try:
        from cloudsim import app
        app.run(width=w, height=h, quality=args.quality, offline=args.offline,
                place=(name, lat, lon, clock), when=when, sim=args.sim,
                sim_size=args.sim_size, loader=loader)
    finally:
        loader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
