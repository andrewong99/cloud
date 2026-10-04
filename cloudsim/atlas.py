# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
WMO International Cloud Atlas taxonomy.

Everything in this module is driven by data/atlas.json, whose classification
follows cloudatlas.wmo.int (its "sources" key lists the 230 pages used; its
descriptions are this program's own wording, see its "about" key).  Nothing
here is invented: the genus/species/variety/feature associations are the
Atlas's tables of the genera each form occurs with most frequently,
cross-checked against its Table 2.

Those tables are titled for the genera a form occurs with *most frequently*,
not the only ones it may occur with.  So a combination outside a table is
reported as UNUSUAL, not as an error.  Only the rules the Atlas states as
rules (one genus, at most one species, translucidus/opacus mutually
exclusive) are hard errors.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data", "atlas.json")

with open(_DATA, "r", encoding="utf-8") as _f:
    RAW = json.load(_f)

# ---------------------------------------------------------------- lookups ---

GENERA = {g["abbr"]: g for g in RAW["genera"]}
GENUS_ORDER = [g["abbr"] for g in RAW["genera"]]          # Ci Cc Cs Ac As Ns Sc St Cu Cb
SPECIES = {s["abbr"]: s for s in RAW["species"]}
VARIETIES = {v["abbr"]: v for v in RAW["varieties"]}
SUPPLEMENTARY = {s["abbr"]: s for s in RAW["supplementary_features"]}
ACCESSORY = {a["abbr"]: a for a in RAW["accessory_clouds"]}
SPECIAL = {s["abbr"]: s for s in RAW["special_clouds"]}
OTHER_CLOUDS = {o["name"]: o for o in RAW["other_clouds"]}
CONSTITUTION = {c["genus"]: c for c in RAW["constitution"]}
ETAGES = RAW["etages"]
CODE_FIGURES = RAW["code_figures"]
COMBINATION_RULES = RAW["combination_rules"]
DISCRIMINATORS = RAW["discriminators"]
PHOTOMETEORS = RAW["photometeors"]
ANGULAR_CRITERIA = RAW["angular_criteria"]
MOTHER = RAW["mother_clouds"]
ETAGE_EXCEPTIONS = {e["genus"]: e["note"] for e in RAW["etage_exceptions"]}

GENITUS = {m["mother"]: m["abbr"] for m in MOTHER["genitus"] if m.get("mother")}
MUTATUS = {m["mother"]: m["abbr"] for m in MOTHER["mutatus"] if m.get("mother")}
TRANSITIONS = MOTHER["transitions"]

#: Latin stem used to build "<mother>genitus"/"<mother>mutatus" names.
_LATIN_STEM = {
    "Ci": "cirro", "Cc": "cirrocumulo", "Cs": "cirrostrato",
    "Ac": "altocumulo", "As": "altostrato", "Ns": "nimbostrato",
    "Sc": "stratocumulo", "St": "strato", "Cu": "cumulo", "Cb": "cumulonimbo",
}

# The Atlas states element apparent-width criteria only for these three genera
# (cc-compared-with-ac, ac-compared-with-sc, sc-compared-with-ac), each on the
# condition that the elements are seen more than 30 degrees above the horizon.
ELEMENT_WIDTH_DEG = {}
for _c in ANGULAR_CRITERIA:
    g = _c.get("genus")
    if g in ("Cc", "Ac", "Sc") and _c.get("observed_elevation_deg") == 30:
        ELEMENT_WIDTH_DEG[g] = (_c.get("min_deg") or 0.0, _c.get("max_deg") or 12.0)
#: Reference elevation for the apparent-width rule, degrees above the horizon.
ELEMENT_WIDTH_REF_ELEV_DEG = 30.0


def genus_name(abbr: str) -> str:
    return GENERA[abbr]["name"]


def phase_of(abbr: str) -> str:
    """'ice' | 'water' | 'mixed' - the Atlas's physical constitution."""
    return CONSTITUTION[abbr]["phase"]


def etage_of(abbr: str) -> str:
    return GENERA[abbr]["etage"]


def latitude_band(lat_deg: float) -> str:
    """Atlas étage bands.  The Atlas gives polar / temperate / tropical without
    numeric latitudes; the conventional split is used here."""
    a = abs(lat_deg)
    if a >= 60.0:
        return "polar"
    if a >= 23.5:
        return "temperate"
    return "tropical"


def etage_range_m(abbr: str, lat_deg: float) -> tuple[float, float]:
    """Base-height range in metres for a genus at this latitude."""
    return tuple(ETAGES[latitude_band(lat_deg)][etage_of(abbr)])


# ---------------------------------------------------------------- CloudSpec --

@dataclass
class CloudSpec:
    """One fully-qualified Atlas cloud name.

    genus         : 'Cu'
    species       : 'con' or None            (at most one - Atlas rule)
    varieties     : ['ra', 'du']             (several permitted)
    supplementary : ['pra', 'arc']
    accessory     : ['pil', 'vel']
    mother        : ('Ac', 'genitus') or None
    special       : 'flgen' or None
    """
    genus: str
    species: str | None = None
    varieties: list[str] = field(default_factory=list)
    supplementary: list[str] = field(default_factory=list)
    accessory: list[str] = field(default_factory=list)
    mother: tuple[str, str] | None = None
    special: str | None = None

    # -- naming ------------------------------------------------------------
    def latin(self) -> str:
        """Full Latin name in the Atlas's component order:
        genus, species, variety, supplementary feature, then mother/special."""
        parts = [GENERA[self.genus]["name"]]
        if self.species:
            parts.append(SPECIES[self.species]["name"])
        parts += [VARIETIES[v]["name"] for v in self.varieties]
        parts += [SUPPLEMENTARY[s]["name"] for s in self.supplementary]
        parts += [ACCESSORY[a]["name"] for a in self.accessory]
        if self.special:
            parts.append(SPECIAL[self.special]["name"])
        elif self.mother:
            parts.append(_LATIN_STEM[self.mother[0]] + self.mother[1])
        return " ".join(parts)

    def abbrev(self) -> str:
        parts = [self.genus]
        if self.species:
            parts.append(self.species)
        parts += list(self.varieties) + list(self.supplementary) + list(self.accessory)
        if self.special:
            parts.append(SPECIAL[self.special]["abbr"])
        elif self.mother:
            parts.append((GENITUS if self.mother[1] == "genitus" else MUTATUS)
                         .get(self.mother[0], self.mother[0].lower() + self.mother[1][:3]))
        return " ".join(parts)

    def __str__(self) -> str:
        return self.latin()

    # -- validation --------------------------------------------------------
    def validate(self) -> tuple[list[str], list[str]]:
        """Return (errors, warnings).

        errors   - break a rule the Atlas states as a rule.
        warnings - outside the Atlas's tables of the genera a form most often
                   occurs with: unusual but not forbidden; the renderer still
                   draws it.
        """
        err: list[str] = []
        warn: list[str] = []
        g = self.genus
        if g not in GENERA:
            return [f"{g} is not one of the ten genera"], warn

        # Atlas: an identified cloud carries at most one species name
        if self.species is not None:
            if self.species not in SPECIES:
                err.append(f"{self.species} is not a species")
            elif g not in SPECIES[self.species]["genera"]:
                warn.append(f"{SPECIES[self.species]['name']} is not tabulated for "
                            f"{GENERA[g]['name']}")

        # Atlas: translucidus and opacus are mutually exclusive.
        if "tr" in self.varieties and "op" in self.varieties:
            err.append("translucidus and opacus are mutually exclusive")
        if len(set(self.varieties)) != len(self.varieties):
            err.append("a variety is repeated")
        for v in self.varieties:
            if v not in VARIETIES:
                err.append(f"{v} is not a variety")
            elif g not in VARIETIES[v]["genera"]:
                warn.append(f"{VARIETIES[v]['name']} is not tabulated for {GENERA[g]['name']}")

        for s in self.supplementary:
            if s not in SUPPLEMENTARY:
                err.append(f"{s} is not a supplementary feature")
            elif g not in SUPPLEMENTARY[s]["genera"]:
                warn.append(f"{SUPPLEMENTARY[s]['name']} is not tabulated for {GENERA[g]['name']}")

        for a in self.accessory:
            if a not in ACCESSORY:
                err.append(f"{a} is not an accessory cloud")
            elif g not in ACCESSORY[a]["genera"]:
                warn.append(f"{ACCESSORY[a]['name']} is not tabulated for {GENERA[g]['name']}")

        if self.special is not None:
            if self.special not in SPECIAL:
                err.append(f"{self.special} is not a special cloud")
            elif g not in SPECIAL[self.special]["genera"]:
                warn.append(f"{SPECIAL[self.special]['name']} is not tabulated for "
                            f"{GENERA[g]['name']}")
            if self.mother is not None:
                err.append("a cloud cannot carry both a special-cloud name and a "
                           "mother-cloud name")

        if self.mother is not None:
            mg, form = self.mother
            if mg not in GENERA:
                err.append(f"{mg} is not a genus")
            elif form not in ("genitus", "mutatus"):
                err.append(f"{form} is not genitus or mutatus")
            elif not any(t["from"] == mg and t["to"] == g and t["form"] == form
                         for t in TRANSITIONS):
                warn.append(f"the Atlas does not list {GENERA[g]['name']} forming from "
                            f"{GENERA[mg]['name']} by {form}")
        return err, warn

    def is_valid(self) -> bool:
        return not self.validate()[0]


# ------------------------------------------------------------ enumeration ---

def options_for(genus: str) -> dict:
    """Everything the Atlas tabulates for this genus, for the UI browser."""
    g = GENERA[genus]
    return {
        "species": list(g["species"]),
        "varieties": list(g["varieties"]),
        "supplementary": list(g["supplementary"]),
        "accessory": list(g["accessory"]),
        "special": [k for k, v in SPECIAL.items() if genus in v["genera"]],
        "genitus": [t["from"] for t in TRANSITIONS if t["to"] == genus and t["form"] == "genitus"],
        "mutatus": [t["from"] for t in TRANSITIONS if t["to"] == genus and t["form"] == "mutatus"],
    }


def enumerate_tabulated(max_varieties: int = 9,
                        with_features: bool = True) -> Iterable[CloudSpec]:
    """Every genus+species+variety-set (+ optional single feature) combination
    that the Atlas tabulates.  Used by the self-test to count the space."""
    from itertools import combinations
    for g in GENUS_ORDER:
        o = options_for(g)
        species = [None] + o["species"]
        vs = o["varieties"]
        for sp in species:
            for n in range(0, min(max_varieties, len(vs)) + 1):
                for combo in combinations(vs, n):
                    if "tr" in combo and "op" in combo:
                        continue
                    if not with_features:
                        yield CloudSpec(g, sp, list(combo))
                        continue
                    yield CloudSpec(g, sp, list(combo))
                    for f in o["supplementary"]:
                        yield CloudSpec(g, sp, list(combo), [f])


def parse(text: str) -> CloudSpec:
    """Parse an abbreviation string such as 'Ac len un du' or a Latin name."""
    toks = text.replace(",", " ").split()
    if not toks:
        raise ValueError("empty cloud name")
    lookup_g = {v["name"].lower(): k for k, v in GENERA.items()}
    lookup_g.update({k.lower(): k for k in GENERA})
    g = lookup_g.get(toks[0].lower())
    if g is None:
        raise ValueError(f"{toks[0]!r} is not a genus")
    spec = CloudSpec(g)
    name_sp = {v["name"].lower(): k for k, v in SPECIES.items()}
    name_va = {v["name"].lower(): k for k, v in VARIETIES.items()}
    name_su = {v["name"].lower(): k for k, v in SUPPLEMENTARY.items()}
    name_ac = {v["name"].lower(): k for k, v in ACCESSORY.items()}
    for t in toks[1:]:
        tl = t.lower()
        if tl in SPECIES or tl in name_sp:
            spec.species = SPECIES.get(tl, {}).get("abbr") or name_sp[tl]
        elif tl in VARIETIES or tl in name_va:
            spec.varieties.append(VARIETIES.get(tl, {}).get("abbr") or name_va[tl])
        elif tl in SUPPLEMENTARY or tl in name_su:
            spec.supplementary.append(SUPPLEMENTARY.get(tl, {}).get("abbr") or name_su[tl])
        elif tl in ACCESSORY or tl in name_ac:
            spec.accessory.append(ACCESSORY.get(tl, {}).get("abbr") or name_ac[tl])
        elif tl.endswith("genitus") or tl.endswith("mutatus"):
            form = "genitus" if tl.endswith("genitus") else "mutatus"
            stem = tl[: -len(form)]
            for k, v in _LATIN_STEM.items():
                if v == stem:
                    spec.mother = (k, form)
                    break
        else:
            raise ValueError(f"{t!r} is not an Atlas term")
    return spec


def angular_width_to_metres(width_deg: float, base_m: float,
                            elev_deg: float = ELEMENT_WIDTH_REF_ELEV_DEG) -> float:
    """Convert the Atlas's apparent element width into a physical spacing.

    The Atlas measures apparent width with the cloud seen more than 30 degrees
    above the horizon, so the slant range to a deck whose base is at
    `base_m` is base_m / sin(elev).  A small element of physical width w then
    subtends w / range radians.
    """
    import math
    rng = base_m / math.sin(math.radians(max(elev_deg, 1.0)))
    return 2.0 * rng * math.tan(math.radians(width_deg) * 0.5)


def metres_to_angular_width(width_m: float, base_m: float,
                            elev_deg: float = ELEMENT_WIDTH_REF_ELEV_DEG) -> float:
    import math
    rng = base_m / math.sin(math.radians(max(elev_deg, 1.0)))
    return 2.0 * math.degrees(math.atan2(width_m * 0.5, rng))
