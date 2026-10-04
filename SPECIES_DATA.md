# How likely each cloud species is — the data behind the odds

CloudSim draws the Atlas layers of the forecast (scene.diagnose). For every
layer the panel now shows the odds of the species, for example

> odds: stratiformis or another species 99.96% · castellanus or floccus 0.04%

and draws the most likely one. This note records where those numbers come
from, what they can and cannot say, and how to reproduce them
(`species_study/`).

## What observations can give

A SYNOP report carries three cloud codes, CL, CM and CH (WMO code tables
0513, 0515, 0509). They name only a few species:

| code | meaning | used for |
|---|---|---|
| CM 8 | Altocumulus castellanus or floccus | P(castellanus or floccus) of a middle layer |
| CH 9 | Cirrocumulus alone or predominant | P(Cirrocumulus) of a high layer |
| CL 6, 7 | Stratus (nebulosus, fractus; fractus of bad weather) | P(Stratus) of a low layer |

Castellanus and floccus share one code, so they have one probability. No code
names lenticularis, volutus or the species of Stratocumulus: those stay
physical rules, and the panel says "a rule" instead of giving a percentage.
Nothing here is invented to fill a gap.

## Data

* **Reports**: SYNOP (AAXX) of 2025 from the OGIMET archive (ogimet.com),
  decoded for the 8NhCLCMCH group. Temperate: WMO blocks 115 and 118
  (Czechia, Slovakia), 123 and 125 (Poland), 276 (central Russia) for April,
  June, July, August and October, and January counts for 115, 123, 276, 471,
  476, 545. Tropical: 486 (Malaysia, Singapore), 485 and 484 (Thailand), 489
  (Vietnam, Laos, Cambodia), 984 (Luzon), 432 (southern India), 434 (Sri
  Lanka) for January, April, July and October (484, 960, 967: January).
* **Station positions**: NCEI's ISD station history (isd-history.csv).
* **Forecasts**: the Open-Meteo historical forecast (best_match, the model
  choice the program's own requests make), hourly, 1000-150 hPa: temperature,
  humidity, cloud cover and height of each level, as the program reads them.
* **Pairing**: each report is matched to the forecast hour it was made at,
  at its station, and the program's own layer geometry and physics are
  computed on that sounding (`species_study/feat.js`, a line-by-line port of
  sounding.py and scene.py; on three real soundings it agrees with the
  Python to 3e-5).

## The layer: its cloudy part only

Open-Meteo's level cloud cover is Sundqvist et al. (1989),
cc = 1 - sqrt((1 - RH)/(1 - RHc)): inverting 550-3000 of its own (RH, cc)
pairs per level (Kuala Lumpur, January-June 2025) gives RHc = 0.70 at and
above 700 hPa, rising to 0.89 at 1000 hPa, each within ±0.02
(sounding.RHC_OPEN_METEO). Pressure levels lie 1-2 km apart in the middle
troposphere, and the deployed version interpolated the cloud fraction itself
from a cloudy level to its clear neighbours: one level at 500 hPa with 28%
cloud became a layer 4710-7322 m deep. The cloud fraction is not the
continuous variable; humidity is. The layer is now where the fraction —
computed between the levels from interpolated RH through the model's own
relation — exceeds half its peak (Sounding.cloud_layers): 5432-6084 m for
that sounding.

## Altocumulus castellanus or floccus

**The fit.** 8,457 reports that saw middle cloud (CM 1-9) and had a forecast
middle layer, at the 33 stations of the temperate blocks whose observers use
code 8 at all, June-August 2025; 396 of them CM 8 (4.7%). Of the candidate
quantities, one carries most of the signal: the lapse rate across the cloudy
part of the layer minus the saturated adiabat there (`lapse_in`, K/km) —
convection inside the saturated layer, the textbook origin of the turrets.

    logit P = -3.2994 + 0.8519 · lapse_in

| lapse_in decile mean, K/km | -1.17 | -0.66 | -0.43 | -0.25 | -0.08 | 0.10 | 0.30 | 0.56 | 0.92 | 1.61 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM 8 reported, % | 0.9 | 1.2 | 1.7 | 3.6 | 3.4 | 5.2 | 4.6 | 5.6 | 8.7 | 11.9 |
| fitted, % | 1.4 | 2.1 | 2.5 | 2.9 | 3.3 | 3.9 | 4.6 | 5.6 | 7.5 | 13.1 |

(each decile ~846 reports; the "fitted" row is the mean prediction in each
decile of predicted probability.) Deviance -187.5 for one parameter; area
under the ROC curve 0.70; with each of the five countries left out of the
fit in turn and predicted, 0.76 (Czechia), 0.82 (Poland 123), 0.64 (Russia),
0.73 (Poland 125), 0.64 (Slovakia). Adding a second quantity — humidity below
the layer, the change of equivalent potential temperature across it, the CAPE
of a parcel lifted from it, its cover or depth, its temperature — improves
the held-out log-likelihood by at most 0.4 percentage points, so it is left
out.

Even the most unstable decile gives 12%: in a central-European summer
castellanus/floccus is never the most likely species of a forecast layer.
Observers also report it in 5.8% of the hours when the forecast has no
middle layer at all (737 of 12,695): what the forecast does not have, the
program cannot draw.

**The climate.** The fit is for the training climate. Elsewhere the program
shifts the log-odds by the climate's own base rate (Bayes' rule, assuming
the distribution of lapse_in among castellanus and non-castellanus layers is
the same):

    logit P = -3.2994 + 0.8519 · lapse_in + logit(rate) - logit(0.04461)

| climate (scene.climate_of) | CM 8 share of middle-cloud reports | stations |
|---|---|---|
| training: temperate, June-August | 4.46% (1,136 of 25,466) | 44 |
| temperate summer half (April-September north) | 4.08% (1,310 of 32,124) | 44 |
| temperate winter half | 0.47% (50 of 10,732) | 66 |
| tropical (within 23.5° of the equator) | 0.018% (8 of 43,640) | 164 |

The rates count stations that report "no middle cloud" (CM 0) at least 2%
of the time. Observers who almost never do — 11 of the 12 stations in
peninsular Malaysia and Singapore (Chuping is the exception), all ten in
northern Sumatra, seven of nine in Java, five in southern India and one in
Thailand — give a middle-cloud code in nearly every report, day and night,
under any sky (Subang in 2025: CM 3 in 53% of 2,918 reports, CM 4 in 33%,
never 0 or "hidden"); they code by habit, and are left out. Observing practice matters as much as the weather: in June
2025, 23 of Praha-Ruzyne's reports were CM 8 and none of Praha-Libus's. Counting only the stations
that used code 8 at least once gives 0.22% for the tropics (19 of 8,549) —
an upper bound, since picking stations by the rare event favours those that
happened to see it. The tropical rate rests on 8 events: its 95% interval is
about 0.008-0.036%.

**The deployed rule**, for the same layers and hours: it drew
castellanus/floccus for 37% of the central-European middle layers (observers:
4.7%) and for 70% of the Malaysian ones (observers: 7 of 4,992, 0.14%). For
Kuala Lumpur's live week 22-29 September 2026 it drew them in 82 of 192
hours; the fitted odds for that week's 103 middle layers have a median of
0.022% and a maximum of 0.045%.

**Diurnal cycle** (training stations, June-August, all 21,152 reports with
middle cloud): CM 8 is 10.2% at 06-09 local solar time, 7.3% at 09-12, 4-5%
in the afternoon and 1.6-1.9% at night — the morning castellanus of the
textbooks, and the night's darkness. The fit uses all hours.

## Cirrocumulus

CH 9 among reports that saw high cloud, with a forecast high layer, at the
same stations and months: 82 of 6,819 (1.2%). The temperature of the layer's
base carries the signal — a warm, low ice layer is likelier to be
Cirrocumulus:

    logit P = -2.3824 + 0.0763 · t_base (C)

| base temperature, sextile mean | -52.8 | -45.1 | -38.9 | -30.4 | -21.6 | -12.2 °C |
|---|---|---|---|---|---|---|
| CH 9 reported | 0.4% | 0.2% | 0.4% | 1.1% | 1.7% | 3.5% |
| fitted | 0.2% | 0.3% | 0.5% | 0.9% | 1.7% | 3.5% |

Left-out-country areas under the curve 0.58-0.92; held-out log-likelihood
+5.4%. Instability in or above the layer predicts it in sample but not from
one country to the next (held-out skill below zero), and is left out. Base
rates (credible stations, all reports with high cloud): training months
1.42%, temperate summer 1.19% (387 of 32,568), winter 0.98% (45 of 4,576,
October), tropical 0.26% (28 of 10,584). The deployed rule's Cirrocumulus
(instability in a thin high layer) fired for 12 of 6,819 central-European
and 5 of 1,709 Malaysian high layers.

## Stratus

CL 6 or 7 among reports that saw low cloud, by the forecast low layer's base
above the ground (19,760 reports):

| base, m | below 150 | 150-300 | 300-600 | 600-1000 | 1000-1500 | 1500 and up |
|---|---|---|---|---|---|---|
| Stratus reported | 13.4% | 5.7% | 1.2% | 0.9% | 0.6% | 1.1% |
| temperate / tropical | 19.8 / 10.3% | 6.2 / 4.5% | 2.8 / 0.8% | 1.3 / 0.4% | 0.6 / 0.7% | 0.9 / 1.5% |

Stratus is now drawn only for a layer based within 300 m of the ground (the
rules for saturation under a stable layer, or a ragged low layer, are
unchanged); above that a low layer is Stratocumulus. The deployed version's
"Stratus fractus opacus" at 1.65 km is gone.

## Limits

* The probabilities are of what observers report, with the forecast's
  errors folded in. They are calibrated where the reports are many (central
  Europe in summer) and shifted by base rates elsewhere; the shift assumes
  the physics of the layers is the same everywhere.
* One year (2025), four months in the tropics, five in the temperate zone;
  no polar or southern-hemisphere stations (polar latitudes use the
  temperate rates).
* The species not named by any code are rules. The genus of a low layer
  above 300 m is not calibrated either: for layers based 1.5-2.5 km up,
  observers report Stratocumulus (CL 4, 5, 8) for 52% in the tropics and 40%
  in central Europe, cumulus (CL 1-2) for 33% and 45%; the program draws
  Stratocumulus. Calibrating Cu against Sc is the next step this data
  allows.
* The drawn species is the most likely one, so castellanus, floccus and
  Cirrocumulus are not drawn from a forecast in practice; they remain
  available by hand.

## Reproducing

`species_study/feat.js` (the program's geometry and predictors in
JavaScript) and `species_study/species_study.js` (download, decode, pair,
fit) ran in a browser page on www.ogimet.com, which reaches OGIMET, NCEI and
Open-Meteo. The fitted numbers are in scene.SPECIES_DATA; self-test rules
Q1-Q6 and G10 check the geometry, the odds and the drawing.
