# CloudSim

A ground-to-sky cloud simulator based on the WMO International Cloud Atlas and
on a cloud-resolving model. You stand on the ground, rise into the cloud or
look down from above, and look around the way you do in Stellarium: drag to
turn, scroll to zoom. The sky shows what the weather data call for. The
cloud-resolving model grows it from that data, and the program says why it
chose it.

    cloud_sim.bat                        (Windows) installs what is missing, then starts
    cloud_sim_menu.bat                   (Windows) a menu: start, start with choices, self-test
    python cloud_sim.py                  live weather for Kuala Lumpur
    python cloud_sim.py --place 51.5,-0.13 --name London --tz 0
    python cloud_sim.py --place Paris    any of 1,584 cities, on its own clock (see Place and time)
    python cloud_sim.py --offline        no network: synthetic soundings
    python cloud_sim.py --sim streets    one cloud form's physics (see below)
    python cloud_sim.py --selftest       6471 checks (some rules open a hidden window; one
                                         shows the loading window for a moment)

---

## How a sky is made

**1. Weather data → regime.** The program fetches the Open-Meteo forecast
(free, no key) for your place and hour. That covers 19 pressure levels of
temperature, humidity, cloud cover, wind and height, plus the surface fields
hour by hour 12 h either side (weather code, cover low/mid/high,
precipitation, CAPE, pressure, temperature, dew point, wind, visibility), and
the ground's state: the station's height, snow depth and root-zone soil
water. It also fetches a 7 × 7 grid of points 40 km apart around you. From these,
`regimes.py` decides which physical regime is making the clouds overhead:

* a thunderstorm here, or one on its way;
* showers;
* fair-weather cumulus;
* a stratocumulus or stratus deck, or fog;
* an altocumulus layer, with virga if the air below it is dry;
* a high ice veil only, or a clear sky.

Every step keeps its number, and the right-hand panel shows the argument, for
example:

> Chosen: stratocumulus deck — the data put an 85 % layer between 620 and 1050 m …
> · weather code: code 3 – overcast → the sky is covered
> · cloud cover: low 85 % mid 10 % high 0 % → a low deck dominates the view
> · instability: CAPE 0 J/kg … → stable: nothing rises by itself
> · pressure: 1021.3 hPa, +0.8 hPa in 3 h → steady: no change coming
> · temperature: 18.2 °C, dew point 12.9 °C → saturates at about 660 m
> · wind: … low clouds drift toward NE at 25 km/h
> · around: thunderstorm 80 km to the WSW, steered toward you: overhead in about 1.6 h
> not a thunderstorm: …   not fog: visibility 18 km   …

**2. Regime → model.** The cloud-resolving model is built from the same
profile, so its temperature, moisture and wind are the forecast's. Its clouds
form at the forecast's heights and drift, lean and trail their virga with the
forecast's wind. A layer the forecast reports is built as a well-mixed,
cloud-topped layer under its inversion, cooled at its top by long-wave
radiation. Its mean temperature and moisture are held to the forecast's with
a 15-minute relaxation (the standard way an LES of an observed case is
nudged). Its cells, gaps, rows and lobes are the model's own.

**3. Model → the sky, as a satellite would see it.** The model's field is
continued beyond its own domain as one continuous field, never as copies:

* it is displaced by how differently the forecast wind has carried the air
  at each place over the last two hours, plus a smooth long-wave stretch far
  out;
* it is thinned where the forecast grid has less of that cloud: the model's
  thinnest cells are dropped first, each cell whole (every column takes the
  largest optical depth within 300 m before the choice is made), so the deck
  keeps its cells and ends where the forecast's cloud ends. Precipitation is
  kept or dropped with the cloud it fell from, traced back up its trail.

The forecast's cover shapes only the sky chosen from the data (and the
convection grown from your sounding). A reference case shows its mechanism
the same everywhere, or, for virga, inside a composition of its own that is
labelled as such in the panel (bands of mid-level moisture, so the layer has
edges to see its trails against clear sky).

The ground view, the view from inside the cloud and the view from above are
all views of this one field. The eye-height buttons are *ground*,
*in the cloud*, *above it* and *satellite* (20 km, looking down).

**4. The "In this view" panel.** A grid of rays is cast through the current
view, into the same field the renderer draws. For each kind of cloud they
meet first, the panel shows:

* its Atlas name, and the varieties its field actually shows: perlucidus from
  its gaps, translucidus or opacus from its optical depth, radiatus or
  undulatus from the orientation of its bands against the wind;
* where it is (height, distance, part of the frame) and how much of the view
  it fills;
* the physics that gives it this look, with live numbers. For example: "its
  cells are about 1.1 km across, 1.3 times the 840 m depth of the overturning
  layer"; "the base passes about 45 % of the light that falls on the top";
  "a 1.1 km cell 0.7 km away spans 77°; 60 km away it spans 1°, and seen at
  1° elevation only 0.02° tall — which is why the same pattern shrinks,
  flattens and crowds into rows toward the horizon".

**5. The Atlas layers of the forecast, and their odds.** Each cloudy layer of
the forecast is also drawn as an Atlas layer (the SKY list on the right).
Its depth is its cloudy part only — where the cloud fraction, computed
between the pressure levels from interpolated humidity through the model's
own relation, is over half its peak — not the whole gap to the clear levels
1–2 km away. Its species is the most likely one, and the panel gives the
odds, for example

> odds: stratiformis or another species 99.96% · castellanus or floccus 0.04%

The numbers are what observers report. SYNOP reports carry three cloud codes,
and three of their figures name a species: CM 8 (Altocumulus castellanus or
floccus), CH 9 (Cirrocumulus) and CL 6–7 (Stratus). The program's odds for
these are fitted on 2025 SYNOP reports paired with the forecast of the same
hour and place — castellanus/floccus on 8,457 reports from central Europe in
summer, against the instability of the saturated layer, then shifted by each
climate's base rate (4.1 % of middle-cloud reports in a temperate summer,
0.47 % in winter, 0.018 % in the tropics). Species no code names
(lenticularis, the Stratocumulus species) stay physical rules, marked as such.
Opacus needs an extensive sheet (60 % of the sky or more), not a field of
thick elements. [SPECIES_DATA.md](SPECIES_DATA.md) has the data, the fits and
their limits.

Castellanus, when drawn (by hand, or where it is the most likely), is a
common layer with turrets on it: the layer's elements gather on lines along
the shear, and three in four elements on a line carry a dome that rises as
far as a parcel lifted from the layer stays buoyant.

A layer of cells is drawn as it is seen from the ground. Its elements are
flat: the Atlas describes them as plates (laminae), rounded masses and rolls
in Altocumulus, rounded masses and rolls in Stratocumulus, grains and ripples
in Cirrocumulus, and overturning in a layer sets in as cells at least twice as wide as the
layer is deep (the onset of Rayleigh–Bénard convection: 2.0 depths between
rigid boundaries, 2.8 between free ones; Chandrasekhar 1961). So the
elements of Altocumulus, Stratocumulus and Cirrocumulus are drawn at most
half as deep as they are wide (and 50 m at least, the thinnest layer drawn).
The forecast's cloudy band can be far deeper: drawn to its full depth, cells
a few hundred metres apart stood as columns 1–2 km tall, and elements as
deep as wide read as balls. The panel says how much of the band is drawn
(`its elements, 232 m wide, are at most half as deep: 116 m of the 4220 m
cloudy band is drawn`); castellanus turrets still rise as far as the parcel
does. Lenticularis is separate lenses in a strong wind across a stable layer;
a sheet of 60 % of the sky or more is not lens-shaped (a rule, as for
opacus).

Their size follows the one numeric rule the Atlas gives: the elements of a
layer seen more than 30° above the horizon have an apparent width of 1–5°
(Altocumulus), under 1° (Cirrocumulus), over 5° (Stratocumulus). From 30°
up to overhead the layer comes twice as near (the distance is h / sin e),
so the same element looks twice as wide overhead: it is made wide enough
30° up and small enough overhead, measured from the station's own height.
An element's width is the cloudy part of its cell (the diameter of a disc
of the same area). The panel gives both ends (`elements 232 m wide,
2796 m up: 2.4 deg seen 30 deg up, 4.8 overhead (Atlas: 1-5 deg above 30
deg)`); checked at 30° alone, a Paris altocumulus 3.3° wide there was 6.6°
overhead.

The elements are rounded masses, not balls or coins: flat-based, as deep as
the layer only in their middle, their water thinning over their outer half
into blurred edges, each stirred within itself. A layer of them fills its
depth only in part, so its extinction is raised (by `gl_sky.DOME_COLUMN`,
3.0, measured in the renderer) for an element's middle to have the optical
depth the panel gives; before, they were drawn about three times thinner
than the panel said. They are lit as what they are, small separate
elements — see Rendering.

Towering convective clouds drawn as Atlas layers (Cumulus congestus,
Cumulonimbus, when the cloud model is off) are drawn as what makes them:
rising plumes. Each is flat at its condensation level and highest over its
strongest updraught, lower toward its edges where drier air is mixed in, so
it is a dome on a flat base that narrows upward; its outline bulges in and
out with height like a stack of rising bubbles; a cumulonimbus's tower stays
broad up to the tropopause, where its plume spreads into an anvil wider
than the tower and thinning outward — and only around a tower that reaches
it. Before, the cell's outline was stood up as a wall: cumulonimbus came out
as rectangular blocks and slabs with flat tops. Cumulus humilis and
mediocris keep their look.

A deep layer from the low étage into the middle is named by what falls from
it, as the Atlas does: with rain or snow, overcast, it is Nimbostratus; with
showers, or thunder, Cumulonimbus (thunder alone too: the CL = 9
convention); drizzle leaves the name alone (it is Stratus's precipitation).
The weather code is Open-Meteo's (WMO code table 4677: 95–99 thunder, 80–86
showers, 61–75 rain or snow, 51–57 and 77 drizzle). Showers or thunder under
a low overcast still bring their convective cloud. The screen-level (2 m)
humidity makes no cloud short of saturation: fog is the forecast's
visibility. The panel gives CIN only where a parcel has a level of free
convection (`no LFC` where it has none).

## How the sky changes with time

The forecast gives the atmosphere on the hour; the sky between the hours is
continuous. [CLOUD_TRANSITIONS.md](CLOUD_TRANSITIONS.md) has the study, the
design and the stress test; in short:

* every hour of the forecast is kept, and the atmosphere at any moment is
  interpolated between the two hours around it;
* each Atlas layer is followed from hour to hour; its numbers (height,
  cover, depth, wind, look) pass smoothly through the hours on a monotone
  cubic, a layer that ends thins away over the hour and one that begins
  forms from nothing; its name changes at the half hour and its halo or
  corona fades with it;
* a layer's cover is a threshold on a standard-normal pattern, so when the
  forecast's cover changes the elements grow and appear, or shrink and
  vanish, and the drawn cover is the forecast's at every moment;
* the pattern itself lives: it is a blend of independent realizations whose
  weights' squares always sum to one, renewing half of itself in 7.5
  minutes for fair-weather cumulus, 15 for a thunderstorm cell, half an
  hour for stratus and cirrus, an hour for a cirrostratus veil;
* each layer drifts with its own wind, integrated over time;
* the regional maps travel along the wind between the hours, blended on the
  GPU for the exact moment;
* moving the clock up to three hours plays the sky through; further, the
  sky dissolves and the new one forms; a new forecast is merged into the sky
  showing; past the forecast's end the sky holds;
* the program opens paused, in real time (x1): the clouds drift at their
  wind's speed and renew at the rates above; x10, x60 and x300 are
  time-lapses (x60: a simulated minute per second).

Before, the program drew one forecast hour and replaced the whole sky when
the weather was fetched again: on the Kuala Lumpur forecast 47 % of the sky
changed at once at a refresh on the hour.

## Place and time

Under the place's name on the panel, **world cities ▾** opens a list of
1,584 places: every national capital, every city of 300,000 people or more,
the region capitals and the Antarctic stations with a clock of their own.
Type to narrow it — the start of a name or a country (`san f`, `kuala`,
`japan`), accents optional — then Enter or a click. **latitude, longitude**
takes any point: `3.139, 101.687`, `3.14N 101.69E`, `33.87 S, 151.21 E`;
the list shows where it is and the cities nearest it. Every place keeps its
own clock: the time zone of the city, or for a point the time zone map's
(within 15 km of a city, the city's; at sea the coast's, further out the
sea's), summer time included. A number after the coordinates (`+8`,
`-3:30`) fixes the clock instead.

**The date ▾ and the hour ▾** under the clock move it on the place's own
clock: the date list runs from 92 days back to 15 ahead, and says for each
day what the weather data are — the forecast, an archived forecast, the
forecast to a given hour on its last day, or none (Sun and Moon only). A
date typed goes anywhere: `2026-10-05`, `5/10/2026` (day first), `5 Oct`,
`+3`, `tomorrow`. The hour list shows the Sun's height at each hour, and
says where summer time skips one; `15:30`, `1530` and `3pm` go to the
minute. A new place is a cut: its forecast is fetched and its sky built
from the start (the model starts again there). A new date or hour moves the
clock as the time buttons do — played through within three hours, else the
sky dissolves and the new one forms — and the forecast is fetched for it
when the one there does not reach it (so is **now**'s). In a list the keys
type — Esc closes the list, not the program — and Ctrl+V pastes.

Open-Meteo forecasts 16 days ahead (today and 15 more days); its archive of
past forecasts starts around 2022. The program asks for three days back and
ten ahead as before; further ahead, up to the 16th day; further back, the
archive around the date. The window is chosen by the hours it holds (00:00
UTC three days back to 23:00 UTC nine days ahead for the usual one), not by
days from now. A time no forecast reaches keeps the nearest hour's clouds,
and the status says so. While a new sky's patterns are being made the panel
says `the sky is forming ...`, not `clear sky`.

On the command line `--place` takes a city or coordinates (`--place Paris`,
`--place "New York"`, `--place 48.86,2.35`); the name and the clock follow
from it unless `--name` or `--tz` are given. `cloud_sim_menu.bat` asks the
same.

## Loading, the compass and the Sun and Moon

**The loading window.** The start takes seconds: compiling the sky's
shaders, fetching the weather, building the layers' patterns, starting the
model. A window drawn by the program itself cannot move while it waits (it
showed white and frozen), so a small window of its own (`splash.py`, a
separate process) comes up first, over a sky drawn by CloudSim, and counts
the start through: what it is doing, a bar and a percentage that never go
back, the patterns built (`7 of 24`). During a long step — a slow network —
the bar creeps on towards the next step without passing it. The program's
window is made hidden, draws its first frames, and shows; the loading window
closes as soon as the program's window has drawn on the screen. If the
loading window cannot start (no display), the program opens as before.

**The loading card.** In the program, a load that was asked for shows a card
over the sky — fetching the weather of a place or a date, building a new
place's sky, forming the sky of a far moment, rebuilding it (r), a change of
layers by hand — with what it is doing and a bar, and goes the frame the
load is done (one shorter than 0.15 s is not shown). A new place's sky is
made in the background while the sky there is still drawn, then cut to; the
program's own thread builds nothing meanwhile, so the window never freezes
(before, it froze while the new sky was built). The panels keep working
under the card. The program's own refreshes (the forecast fetched again near
its end) show no card.

**The compass.** As in Stellarium, the points of the compass are written on
the horizon: N, E, S and W bigger and in Stellarium's red, NE, SE, SW and NW
smaller. They stand on the horizon the renderer draws: the ground is a
sphere, so from an eye above it the horizon lies acos(R / (R + h)) below the
level (0.04° at 1.7 m, 0.56° at 300 m, 4.5° at 20 km). `q` (Stellarium's key)
or **directions** under VIEW shows and hides them; they are on from the
start.

**The Sun and the Moon** are the real ones for the place and the moment,
computed (`astro.py`, Meeus's algorithms: the Sun's position, the Moon's
from the largest terms of its theory, the Moon's parallax seen from the
ground, refraction near the horizon) — not a fixed path. Against pyephem and
astropy, the Sun's position is within 0.013°, the Moon's within 0.05°, its
lit fraction within 0.003, sunrise and sunset within 4 seconds and moonrise
and moonset within 1.2 minutes (Kuala Lumpur, Paris, Reykjavik at
midsummer, Sydney). The Moon's disc is lit on the side facing the Sun, as
much as its phase. Rules E1–E8.

## The physics of each cloud form

Besides "from the weather", each reference case runs one cloud form's
mechanism (`--sim <key>` or the panel buttons):

| key | what forms, and why |
|---|---|
| `stratocumulus` | DYCOMS-II RF01 (Stevens et al. 2005): long-wave cooling of the cloud top (F0 70, F1 22 W m⁻², κ 85 m² kg⁻¹) overturns the 840 m boundary layer into closed cells. Includes subsidence −Dz, Coriolis and fixed surface fluxes. |
| `altocumulus` | a thin moist layer at 3 km overturning under its own radiative cooling: cloudlets and gaps (perlucidus) |
| `altocumulus_ra` | the same with shear through the layer: rolls, i.e. rows along the wind (radiatus) |
| `altocumulus_un` | a thin shear layer on the inversion (Ri ≈ 0.2 < 0.25): Kelvin–Helmholtz waves across the wind (undulatus, fluctus) |
| `streets` | a cold-air outbreak (0 °C air at 14 m/s over an 8 °C sea, bulk surface fluxes): boundary-layer roll vortices lift cloud in lines 2–3 boundary-layer depths apart |
| `asperitas` | a thick Sc layer (1.6–2.3 km) on a moist, stable, sheared layer: across the 200 m under the base the air warms 1 K while the wind changes 6 m/s and turns 60°, a Richardson number of ~0.2 (below the 0.25 at which it overturns). The Kelvin–Helmholtz waves that grow there lift 85–99 % saturated air into cloud, so they are printed on the base, running in several directions at once as the wind turns: the "rough sea seen from below". Drizzle evaporating under the base makes the troughs sag (Harrison et al. 2017; Ravichandran & Govindarajan 2020, 2022). |
| `virga` | a mixed-phase altocumulus at −12 to −20 °C over 65 % humid air: ice grows at the droplets' expense, falls as snow (all snow: updraughts of 1–3 m/s make no graupel; ice turns to snow from 0.01 g/kg, as in a thin mixed-phase layer) and sublimates within about 1 km. The cloud drifts with the wind at its level while its snow falls into slower air, so each trail hangs back upwind; the panel gives the numbers. |
| `weather` | deep convection from your sounding, heated by the Sun: cumulus → congestus → cumulonimbus, anvils, overshooting tops, rain, lightning (Price & Rind 1992 flash rates) |
| `supercell` | Weisman & Klemp (1982): the textbook supercell sounding and quarter-circle hodograph |
| `squall` | a squall line after Rotunno, Klemp & Weisman (1988). The cold pool's gust front lifts the inflow into a shelf cloud (arcus). |
| `bomex` | trade-wind cumulus (Siebesma et al. 2003) |

The model (`crm.py`, and `crm_gpu.py` on the GPU) is CM1's anelastic
formulation:

* Arakawa C grid, Wicker–Skamarock RK3, fifth-order upwind advection;
* an exact FFT + tridiagonal pressure solve;
* Kessler warm rain plus SAM1MOM ice, snow and graupel (graupel only in deep
  convection, as SAM's dograupel switch allows), with a saturation
  adjustment that conserves energy exactly;
* for layer clouds: domains lifted to the layer (`Grid.z0`), the DYCOMS-II
  long-wave scheme, the Coriolis force with a geostrophic wind, bulk
  sea-surface fluxes, large-scale subsidence of all water, and nudging of the
  mean profiles.

The ground under a sounding heats the air by the forecast's soil water: the
evaporative fraction falls from 0.69 (Bowen ratio 0.45) at field capacity to
0.03 at the wilting point (Noah's loam values, 0.329 and 0.066 m³/m³), so a
desert heats its air without moistening it. Bare dry ground absorbs 65 % of
the sunlight, not 77 % (sand, albedo 0.35), and lying snow 36 %. Without the
forecast's ground (offline), the ground is moist and bare.

The GPU and NumPy models are run side by side in the self-test and agree to
single precision.

## Every cloud in the WMO navbox

* **All 10 genera, 15 species, 9 varieties, 11 supplementary features, 4
  accessory clouds and the special clouds** (Atlas data transcribed from
  cloudatlas.wmo.int, `data/atlas.json`) can be chosen by hand in the cloud
  browser and drawn as Atlas layers.
* **Grown by the model**, as listed above: Sc, Ac, Cu, Cb, virga,
  praecipitatio, asperitas, arcus, incus, fluctus/undulatus, radiatus and
  fog. Mamma is drawn as an Atlas layer (the cloud browser) but not grown: in
  this model's microphysics (cloud water in saturation equilibrium, rain
  falling at Kessler speeds) the evaporating lobes do not form.
* **Named by the in-view census when the model makes them:** the overshooting
  top (an updraught above the tropopause); the hot tower (a tropical tower
  above 14 km); towering cumulus (TCu, the ICAO term, also called Cumulus
  castellanus); trade-wind cumulus (BOMEX).
* **Above the weather** (`upper.py`), lit geometrically by the Sun below the
  horizon:
  * **Noctilucent clouds.** In summer at 50–70° latitude, seen with the Sun
    6–16° below the horizon, silvery blue, low over the pole-ward horizon.
    The four Fogle–Haurwitz forms are the mesopause's waves acting on one
    ice sheet: type I the veil itself, II bands (crossing sets of gravity-wave
    crests), III billows (short waves and turbulence where the bands break),
    IV whirls.
  * **Nacreous polar stratospheric clouds**, when the sounding's stratosphere
    is below −85 °C in polar winter: separate lenses standing in lee-wave
    crests over one mountain range, or a fibrous sheet. Their mother-of-pearl
    colours come from a Mie table of nearly equal ice spheres of 0.6–3 µm
    (`data/psc_phase.npy`): the particles grow toward the middle of each
    lens, so the colours run in bands along it and change with the angle
    from the Sun.
  * **Nitric-acid PSCs**, faint, when the stratosphere is below −78 °C.

  The browser can also show any of them on demand.
* **Named but not simulated:** the horseshoe cloud (a vortex that lives for a
  minute or two at the top of a thermal in shear), actinoform clouds (a
  100–300 km pattern visible only from space) and homogenitus/homomutatus
  (contrails). The first two are beyond any domain the model can run live;
  contrails are not modelled.

## Rendering

OpenGL 3.3 through moderngl (the model's compute shaders need 4.3; without
them it runs on the CPU, on the small grid):

* a spherical atmosphere with Rayleigh, Mie and ozone scattering and
  Hillaire's (2020) multiple-scattering table, standing on the station's
  ground: at 5364 m half the air is below the eye, and the sky is half as
  bright and bluer (checked against the reference integration);
* the ground's fields and woods, under lying snow when the forecast has
  some: cover = depth / 10 cm (Dutra et al. 2010), open ground at albedo
  0.75, snowy woods at 0.3; the light going back and forth between snow and
  a cloud base is counted, so a snowfield under stratus is nearly as bright
  as the sky. Where the forecast's root-zone soil is dry, the fields turn to
  dry grass and then bare sand (albedo 0.35) - green from 0.20 m³/m³ down,
  bare at the wilting point. A forecast has no land cover; this is the
  program's proxy for it;
* the model's cloud water, ice, rain and snow ray-marched with Mie phase
  functions (Jendersie–d'Eon for droplets, a BHMIE rain table that holds the
  rainbows), light marches, a Beer shadow map (ground shadows and
  crepuscular rays), and the Eddington diffuse field for thick cloud;
* a layer of rounded masses (Altocumulus, Stratocumulus, Cirrocumulus) lit
  as separate elements. The Eddington field is a layer's without edges; an
  element a few hundred metres across loses its diffused light through its
  flanks into the clear air around it, and what is left travels on away
  from the Sun. Monte Carlo transport through such elements (flat-based,
  twice as wide as deep, droplet phase g = 0.86; alone, at 34 % and 60 % of
  the sky, optical depths 3–80, seen toward, across and away from the Sun)
  gave a shaded side 2–10 times darker than that field drew, and a sunlit
  side twice as bright as the shaded one — the field drew them evenly lit,
  with no shadow beside the lit part. The diffuse term of these layers is
  fitted to those runs (`shaders.deckRadiance`: rms error a factor 1.28,
  from 2.5) and returns to the layer's own as the elements close into an
  overcast; every other cloud keeps its light;
* air under a cloud deck lit by the grey light the deck lets through. The
  droplets' forward peak (delta-M scaling) goes on with the Sun's beam and
  the air scatters it as sunlight; the rest comes down as a diffuse field,
  of which the air sends the eye what the haze's phase function gives over
  the upper hemisphere — much to an eye below looking up into it, little
  to an eye above looking down on it. (It was taken as scattered evenly in
  every direction: from an aircraft the shadows of clouds in hazy air came
  out brighter than the sunlit air beside them, as white trails radiating
  from the point opposite the Sun);
* Atlas layers composited nearest first along each ray (from the ground the
  lowest is in front, from an aircraft the highest); a halo or corona only
  where its layer is seen through what lies in front of it; a ray that is
  already opaque follows its transmittance on, so the Sun's disc does not
  show through a nimbostratus;
* still frames averaged over the whole pixel (Halton sub-pixel offsets), so
  their edges are anti-aliased; the average starts again when a change of
  the sky ends, so nothing of the sky before it stays in the picture;
* metered auto-exposure within limits set by the Sun's elevation;
* AgX tone mapping and physically based bloom.

On the large-eddy grids the cloud shapes are the model's own. Procedural
billows are added only on the coarse grids of the deep-convection runs, below
their grid scale.

## Stress test

`selftest.py`'s rules P1–P11 and G3–G9 are the stress test's findings, each
checked to fail on the code before its fix. The test drove forecast answers
that are broken or extreme (a 5 km station without 2 m data, the poles, the
date line, a missing hour, NaN and infinity), soundings from a 48 °C desert
to a −45 °C polar night, 600 random weathers, every model case at every grid
size and seven runs of 2–3 hours, the renderer at every eye height, field of
view, place and hour, the in-view census, and the app itself headless (55
actions). It found and fixed:

* forecast levels below a high station kept; region points past the pole; a
  missing hour silently replaced by another; NaN forecast values crashing the
  regime choice;
* NaN regional maps giving a black frame and crashing the census; the
  satellite view losing the model's layer beyond ~20 km;
* from above, a higher Atlas layer hidden behind a lower one; a halo seen
  through a stratus overcast; the Sun's disc through any cloud at 1 %; a
  sea-level sky over a 5 km station; green grass at −45 °C and in a
  bone-dry desert; the ground under an Atlas overcast still lit by the whole
  blue sky; lightning drawn as high above the storm as the eye was;
* a fine asperitas grid the GPU model could not run; the desert ground
  evaporating like a lawn (its air gained 18 % water in 2.5 h); every Atlas
  layer below 20 km (a cirrostratus veil too) left out while the squall-line
  model ran, instead of only its cumulonimbus and cumulus.

A model that becomes numerically unstable is now stopped with a message.

The sky between the forecast's hours has its own stress test
([CLOUD_TRANSITIONS.md](CLOUD_TRANSITIONS.md), `transition_study/`): four real
forecasts played through minute by minute (18,480 minutes), everything the
renderer reads checked for jumps at every hour, half hour and random moment
(772 moments; none left), the pattern's own change against its theory (0.99
of the predicted rate), the drawn cover against the forecast's (mean error
0.005), and the old and new program rendered side by side through three
hours of Kuala Lumpur and Paris. Rules T1–T14 and G11 hold its findings: on
the way it found and fixed halos and coronae switching on and off, the
castellanus turrets' base jumping on the hour and turrets standing on no
cloud, undulatus waves shared by every realization (the cover up to 0.4
wrong), and a threshold solver that could overshoot to "all cloud". A second
round (30 Sep) scripted ten skies with the hardest hourly changes a forecast
can make, fed broken answers through the request path, rendered twilight
frame by frame and drove the app itself: no jump between the hours (354
moments), but it found an hour given twice shifting every later hour, an
infinite wind speed turning into NaN drift, the cover of few-element tiles
off by up to 7.7 %, and six whole-picture jumps at dusk and dawn (refraction
cut off at −2°, the exposure's range stepping at +4° and −14°, moonlight
switched on at −10°) – all fixed, rules T15–T17 and G12. After it was
deployed, the panels turned out not to have been drawn since 28 Sep (the
layers' uniform block sat on the binding point pyglet uses for the panels'
projection); fixed, rule G13. Pictures of the whole window then showed
that since 28 Sep a time shift made while paused, with no model running,
left the panel's clock, Sun and Moon at the old time (the panel is now
redrawn whenever what it shows has changed; rule G15) and a restart of the
model the program made by itself started a paused clock (it now leaves the
clock as it was; a click on a model still starts it; rule G16), and three older
faults, all in the 25 Sep version too: the convective layer read a forecast
of 0 % low cloud as a missing figure and drew Cumulonimbus at 4/8 (rule
P12); the help box (h) came up
without its text – pyglet deletes a label the moment nothing refers to it –
and a wrapped panel text was given a height guessed from its number of
characters, so its last line could run into the next line or button, and
after a turn of the mouse wheel every wrapped text slid down over the ones
below it. The help now holds its labels until they are drawn, in a box as
wide as its longest line, and the panel measures each wrapped text as the
font lays it out; rule G14.

Rules Q1–Q6 and G10 check the species odds: a layer's depth from its cloudy
part, Open-Meteo's cloud cover read as the model computes it, castellanus
only as often as observers report it, opacus only for a sheet, no Stratus
far above the ground, castellanus turrets standing on their layer, and small
steep elements drawn solid — the ray march's dither now covers the whole
stride through empty air, where it covered one step of three and drew them
as stacks of discs.

A third round (3 Oct) began with a Paris altocumulus drawn as hanging
columns and stress-tested the program again: every city's clock against the
tz database (47,520 checks), every clock change of 2026–2027 (51,200), the
time-zone map on a 0.5° grid (259,200 points), every city found by its name
(1,584), 60,000 random strings typed into the date, hour and coordinate
fields, 89,352 request windows, 84 runs at the poles, the date line and the
clocks furthest from UTC, and the window itself under random real input on a
slow, failing network (six runs, 1,150 actions, two with the cloud model
running): no crash, no broken state. Rules X1–X9 and W8 hold what it found
and fixed: layers of cells drawn as deep as their cloudy band (145 of 245
Altocumulus and Stratocumulus layers in four recorded forecasts stood taller
than their cells were wide); deep precipitating layers named by their depth
alone — steady rain from "Stratocumulus", showers and thunder with no
Cumulonimbus (3 of 6 thunder hours); Stratus made from the 2 m humidity,
based under the ground (41 of 54 Stratus layers); the offline Stratus
profile's 21 K/km layer, which turned its deck castellanus; a CIN of −14,000
to −18,500 J/kg in the panel where a parcel has no level of free convection;
overcast opacus Stratocumulus named lenticularis by its wind alone (10 of 14
lenticularis layers); Nimbostratus drawn with gaps; a still picture that kept
the sky from before a change in it (two thirds of it a dozen frames later —
longest on a slow machine); a request nine days ahead late in the UTC day
that missed its hour; `clear sky` in the panel while a new sky formed; and a
self-test rule that depended on the wall clock (T16).

A fourth round (4 Oct) looked at the layers of cells from the ground, as
they are seen there. It found the elements sized at 30° up alone, so that
overhead, twice as near, an altocumulus was 6.6° wide where the Atlas says
1–5°; elements as deep as they were wide or deeper (from below, balls); the
elements drawn about three times thinner than the optical depth the panel
gave; and their light that of a layer without edges, which drew each
element evenly lit, without the shaded side beside its sunlit part; and
cumulonimbus drawn as rectangular blocks (the cell's outline extruded into
walls, flat-topped). All fixed (rules Y1–Y5, Y7, Y8). The window now opens
at a 60° view (rule Y6): at 80° a rectilinear frame stretched the clouds at
its top and bottom edges 1.7 times, at 60° 1.33 times; the slider still
goes from 4° to 140°.

The loading window, the loading card and the compass have rules Z1–Z8: what
the program tells the loading window reaches it as told and its bar never
goes back; it opens, ignores what it cannot read and closes when told to or
when the program's end of the pipe closes; the program's window is hidden
until its first frames are drawn; a sky made ready in the background asks
for exactly what the sky drawn then needs and is cut to with nothing left to
build, the program's thread building nothing; the card is up for every load
asked for and gone the frame it is done; the compass points stand where the
renderer's rays for those directions meet the horizon (400 random views,
eye heights from 1.6 m to 20 km), and the horizon in the rendered picture is
where they stand (to 2 pixels at 30 pixels a degree).

## Files

`cloud_sim.py` (entry), `cloudsim/`:

* `app.py` — window, panels, input; the place, date and hour dropdowns; the
  loading card and the compass
* `splash.py` — the loading window (`data/splash.png`: its picture, a sky
  drawn by CloudSim)
* `places.py` — the cities, their time zones and clocks, the dates and
  hours typed (`data/cities.json`, `data/timezones.bin`: a 0.1° map of the
  time zones)
* `forecast.py` — every hour of the forecast; the atmosphere at any moment
* `timeline.py` — the sky between the hours: layers followed through them,
  their evolving patterns and drift
* `regimes.py` — weather data → regime → model, with reasons
* `region.py` — the satellite's view: regional cover and displacement maps
* `viewinfo.py` — the in-view census
* `upper.py` — noctilucent and polar stratospheric clouds
* `crm.py` and `crm_gpu.py` — the cloud-resolving model
* `simdriver.py` — runs the model alongside the sky
* `gl_sky.py` and `shaders.py` — the renderer
* `atmosphere.py` and `mie.py` — the atmosphere and scattering tables
  (`data/rain_phase.npy`: rain and its rainbows; `data/psc_phase.npy`:
  nacreous iridescence)
* `sounding.py` — Open-Meteo sounding, series and region; the windows asked for
* `scene.py` — Atlas layers and their species odds (`SPECIES_DATA`)
* `atlas.py` — the taxonomy
* `selftest.py` — the checks

`SPECIES_DATA.md` — the observations behind the odds; `species_study/` — the
scripts that paired them with the forecast, and three figures.
`CLOUD_TRANSITIONS.md` — how the sky changes between the forecast's hours;
`transition_study/` — its stress test: the old and new program side by side
through three hours of two real forecasts, and the measurements.
`LICENSE` — the GNU General Public License, version 3.

## Sources

Stevens et al. 2005 (DYCOMS-II RF01, Mon. Wea. Rev. 133); Bryan & Fritsch 2002
and the CM1 source (model numerics); Siebesma et al. 2003 (BOMEX); Weisman &
Klemp 1982; Rotunno, Klemp & Weisman 1988; Ravichandran & Govindarajan 2020
(J. Fluid Mech. 899) and 2022 (Phys. Rev. Fluids 7, 010501) on mammatus and
asperitas; Harrison et al. 2017 (Weather) on asperitas; Khairoutdinov &
Randall 2003 (SAM's one-moment microphysics, its graupel switch); Fogle &
Haurwitz 1966 (noctilucent forms); Warren & Brandt 2008 (the refractive
index of ice); Bohren & Huffman 1983 (Mie theory); Hillaire 2020 (sky);
Jendersie & d'Eon 2023 (droplet phase functions); Price & Rind 1992
(lightning); Dutra et al. 2010 (J. Hydrometeor. 11, snow cover and albedo);
Chen & Dudhia 2001 (Mon. Wea. Rev. 129, the Noah land surface); Sundqvist et
al. 1989 (Mon. Wea. Rev. 117, cloud fraction from humidity); Chandrasekhar
1961 (Hydrodynamic and Hydromagnetic Stability: the onset of convection in a
layer); WMO code tables 0509, 0513, 0515 (the CL, CM, CH codes) and 4677
(present weather, as Open-Meteo's weather code); the WMO International Cloud Atlas;
Open-Meteo; OGIMET's SYNOP archive; NCEI's station history. Places: Natural
Earth's populated places (public domain); the time-zone boundaries of
timezone-boundary-builder 2025b (© OpenStreetMap contributors, ODbL); the
IANA time zone database, 2026c (the clocks from 2020 to 2035 are stored).
Meeus 1998 (Astronomical Algorithms, 2nd ed.: the Sun, the Moon, sidereal
time); Bennett 1982 (refraction); pyephem and astropy (the checks).

## License

CloudSim is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version. It is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE).

`SPDX-License-Identifier: GPL-3.0-or-later` · Copyright (C) 2026 The
CloudSim authors.

The data files carry their sources' terms, not the program's:

* `cloudsim/data/cities.json` — from Natural Earth's populated places
  (public domain).
* `cloudsim/data/timezones.bin` — a 0.1° map made from the boundaries of
  timezone-boundary-builder 2025b (© OpenStreetMap contributors), available
  under the Open Database License 1.0; the clocks are the IANA time zone
  database's (public domain).
* `cloudsim/data/atlas.json` — the cloud classification of the WMO
  International Cloud Atlas: its names, abbreviations, associations, levels,
  code figures and angular criteria (facts, from the Atlas pages listed in
  its `sources` field). Every description, rule and note in it is written in
  CloudSim's own words and is under the program's license; facts were also
  checked against the freely licensed pages in its `references` field. Its
  `about` field says so in the file itself.
* `cloudsim/data/rain_phase.npy`, `psc_phase.npy` and `splash.png` are made
  by the program itself (`mie.py`, the renderer) and are under its license.
* Weather data by [Open-Meteo.com](https://open-meteo.com/), under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The program
  fetches it while it runs and names it on the panel; the recorded answers in
  `transition_study/scripts/data/` (rounded to 16-bit integers) and the few
  soundings and values written into `cloudsim/selftest.py` as test fixtures
  are Open-Meteo data too.
