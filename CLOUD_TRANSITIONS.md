# How the sky changes between forecast hours

A forecast gives the atmosphere on the hour. Clouds do not change on the
hour: they drift with the wind all the time, their elements are born and die
in minutes, and a layer forms, thickens, rises and breaks up over tens of
minutes to hours. This note is the study behind CloudSim's handling of the
time between two forecast hours: what the program did, how real clouds
change, the ways of filling the gap, what the program does now, and how it
was stress-tested.

## 1. What the program did

The deployed program took the forecast hour nearest the clock, diagnosed its
Atlas layers, built one pattern per layer and let the patterns drift with
their wind for as long as the clock ran. A refresh (a re-fetch, or the
R key) replaced every layer and every pattern at once.

Measured on four real Open-Meteo forecasts, with the program's own requests
(Kuala Lumpur, Paris, San Francisco, Chicago; 60–84 hours each), over an
equal-area grid of 3,840 sky directions from 5° to the zenith, a direction
counted as cloudy where it crosses a layer's middle where the layer's map is
cloud:

| | KL | Paris | SF | Chicago |
|---|---|---|---|---|
| sky directions that flip cloud↔clear at a refresh on the hour (mean) | 47 % | 35 % | 16 % | 6 % |
| … worst hour | 82 % | 78 % | 100 % | 100 % |
| cover error of a sky held from the first hour (mean over the run) | 0.25 | 0.25 | 0.11 | 0.73 |

A refresh changed as much of the sky, in one frame, as three minutes of
drift. Holding one hour's sky while the clock played left the cover wrong by
a quarter of the sky on average, and by all of it at worst.

## 2. How real clouds change

Three processes, on three time scales:

1. **Drift.** Over minutes a cloud field is carried by the wind at its own
   height, largely unchanged in shape – Taylor's frozen-turbulence
   hypothesis. Layers at different heights move at different speeds and in
   different directions. This is most of what one sees move in the sky.

2. **The elements' own lives.** The individual elements form, grow and
   evaporate. Shallow cumulus live 15 ± 2 minutes (median of 158 clouds
   tracked with stereo cameras at the ARM Southern Great Plains site; Romps
   et al. 2021). A single thunderstorm cell goes through its life cycle in
   about 30 minutes (NOAA JetStream). Larger organised patterns last
   longer: in radar precipitation fields, how long a pattern stays
   recognisable in the frame moving with it grows with its size (Germann &
   Zawadzki 2002). Layer clouds renew themselves on the time scale of the
   motions that make them: the overturning of a stratocumulus-topped mixed
   layer (depth over velocity, ~1 km at ~1 m/s: a quarter to half an hour),
   the fall of ice crystals through a cirrus layer (1.5 km at ~1 m/s: half
   an hour).

3. **The layer's cover.** A layer's cloud fraction is the part of the
   air's sub-grid distribution of water that exceeds saturation – the
   statistical cloud schemes of Sommeria & Deardorff (1977) and Mellor
   (1977). When the air moistens, the elements grow and new ones appear
   where the air was closest to saturation; when it dries they shrink and
   the smallest vanish. Clouds form and evaporate; they do not fade in and
   out as a whole.

The forecast samples the third on the hour and says nothing about the first
two.

## 3. Ways to fill the time between frames

| method | what it does | what goes wrong |
|---|---|---|
| hold and switch (the old program) | one hour's sky until the next | the sky teleports |
| cross-fade two renderings or two cover maps | blend the before and after | ghosting: both skies half-visible; a blend of two independent patterns has half the variance, so thresholding it gives the wrong cover |
| advection-corrected interpolation (Anagnostou & Krajewski 1999; `pysteps`) | move each frame along the wind toward the moment, then blend | right for fields known at both ends, like the regional cover maps; it cannot invent the elements' lives |
| numbers interpolated over one fixed pattern (game engines' weather maps, e.g. *Horizon Zero Dawn*, Schneider 2015) | cover, height and type change smoothly | the same elements for ever: nothing is born or dies |
| numbers interpolated over an evolving stochastic pattern, blended so its statistics never change (histogram-preserving blending, Heitz & Neyret 2018) | continuous, with a life of its own | needs care to keep the cover exact and the blend independent |
| a cloud model run forward | the physics | needs spin-up; used where CloudSim runs its model, which hands over to the layers by dissolving |

CloudSim now uses the fifth for its Atlas layers, the third for its regional
maps and the sixth where its cloud-resolving model runs.

## 4. What CloudSim does now

**Every hour of the forecast is kept** (`forecast.py`). The atmosphere at
any moment is linear in time between the two hours around it: temperature,
humidity, the heights of the pressure levels, the wind by its components,
the model's cloud cover; categories (weather code) come from the nearer
hour.

**Layers are followed through the hours** (`timeline.py`). The hours are
diagnosed in order from the forecast's first, so the tracks – and the sky at
a moment – do not depend on how the clock got there. A layer of one hour is
the same layer as the one of the next hour whose depth overlaps it most
(each depth widened by what a layer can plausibly move in an hour). A layer
without a successor thins to nothing over the next hour; one without a
predecessor grows from nothing. A track's seed comes from where it begins
(its first hour as a time, its place among that hour's layers, its genus),
so a refreshed forecast that says the same gives the same future.

**Every number changes smoothly through the hours.** Base, top, cover,
extinction, wind, element size and every look parameter lie on the monotone
cubic through the hourly values (Fritsch & Carlson 1980; the slopes of
Fritsch & Butland 1984, as in scipy's `PchipInterpolator`): continuous and
with a continuous rate of change, an hour where a quantity turns is a
turning point of the curve, and nothing overshoots between two hours (a
cover stays within 0–1). The name changes at the half hour; the halo and
corona a name implies do not switch with it – each hour's kind of optics
(halo for Cs, less for Ci; corona for Ac and Cc) is a number that blends, and
the thin-cloud limits (optical depth ~4 for a halo, ~6 for a corona) are
smooth.

**Cover is a threshold on a standard-normal pattern.** Each layer's pattern
is a field *z* with exactly a standard normal distribution (the histogram
transform of Heitz & Neyret 2018, applied to the cellular, fibrous or
turbulent field of the layer's kind). The layer is cloud where
*z* > *z\** = Φ⁻¹(1 − cover): the statistical scheme's rule. When the
forecast's cover changes, elements grow and appear, or shrink and vanish,
where the pattern is highest and lowest – and the drawn cover is exactly the
forecast's at every moment.

**The elements live and die.** The pattern is a blend of independent
realizations, one begun every *hop*, each alive for three:

 *z* = Σ *w_j* *Z_j*, *w_j* = √(8/9) sin²(π(θ − *j*)/3)

The three living weights always satisfy Σ*w_j*² = 1 (sin⁴ at three phases a
third of a period apart sums to 9/8), so *z* stays exactly standard normal
and the cover exact – nothing ghosts, as it would in a cross-fade. Each
weight starts and ends at rest, and Σ(d*w_j*/dθ)² is constant, so the
pattern changes at a steady rate. θ advances at a rate set by the kind of
cloud, so that half the pattern has been replaced after its renewal time:

| | Cu | Cu fra | Cu con | Cb | Ac | Cc | Sc | St | Ci | Cs, As | Ns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| half renewed after (min) | 7.5 | 6 | 12 | 15 | 15 | 8 | 25 | 30 | 30 | 60 | 45 |

Cu and Cb follow the observed lifetimes (a snapshot's elements are on
average half way through their lives, so half are gone after half a
lifetime); the others are this program's estimates from the time scales of
section 2.

**Each layer drifts with its own wind, integrated over time** – the wind
interpolated between the hours – instead of wind × clock time, so a wind
that changes does not make the whole layer jump. Lenticularis and volutus
stand still.

**Castellanus** turrets come with the realizations that carry them and fade
in and out with their weights; they stand on the layer's cloud where it is
cloud now, so they sink and go when the layer thins. **Perlucidus and
lacunosus** gap nets likewise come with their realizations. **Undulatus**
waves are one field of the layer that moves with it, added to its evolving
pattern, with the threshold computed for the pattern and the waves together
(`noise.zstar_with_wave`); built into each realization, as first tried, they
made the realizations correlated (0.7–0.8) and the cover wrong.

**The regional maps** (the forecast's cover on a 7 × 7 grid 40 km apart,
which decides where layers and the model's field end) are carried along each
étage's wind from the hour before and the hour after and blended by time:
advection-corrected interpolation. Both hours' maps are on the GPU and the
blend is done per sample for the exact moment; the CPU makes maps only when
the clock enters another hour (it used to rebuild them once a simulated
minute, in 15 ms on the main thread, and they changed in steps). A layer
passing an étage boundary (2 and 7 km) takes the new étage's cover over 600 m
and 1 km of height instead of at once.

**When patterns are late.** Realizations are built on a background thread.
If one is not ready, the layer's pattern waits at a point where the missing
realization's weight is zero and catches up at up to three hops a second; it
never jumps. More than nine hops behind (a far jump of the clock) the layer
dissolves and forms again with the new pattern.

**Moving the clock.** A shift of up to three hours plays the sky through
every moment in between over 1–3 seconds; further, the layers the new moment
does not have dissolve while its own form. A refreshed forecast is matched
to the sky showing: a layer that continues keeps its pattern and drift and
eases to its new numbers over four seconds; the regional maps of the old
forecast blend into the new ones over the same time. Past the forecast's
last hour the sky holds instead of emptying.

**Above the weather.** Nacreous and nitric-acid polar stratospheric clouds
grow in over 4 °C about their thresholds (−85 °C, −78 °C) and the
lenticular form blends with the cirriform as the stratospheric wind passes
20–30 m/s; the noctilucent forms are those of the local night, changing at
local noon when none can be seen (they changed at 00 UTC).

**The model** (the cloud-resolving model, where it runs) cannot be
interpolated. Its field is handed to the layers and back by dissolving –
its thinnest columns first – and a model that must restart does so hidden.

## 5. Stress test

**Teleport detector.** At every hour boundary, every half hour (where names
change) and 40 random moments of each of the four real forecasts (772
moments),
everything the shader reads – every number of every layer, the halo and
corona strengths, the regional maps on a 23 × 23 grid out to 110 km – and
the sky mask are taken 0.1 s before and after, and 1 s before and after. A
continuous quantity changes ten times less over the shorter interval; a jump
does not.

| | first version of the new code | final |
|---|---|---|
| moments with a jump (KL, Paris, SF, Chicago) | 34 of 772 (7, 12, 6, 9) | 0 of 772 |
| sky directions flipping in 0.1 s across an hour, mean (KL, Paris, SF, Chicago) | – | 0.13 %, 0.32 %, 0.05 %, 0.01 % |
| … at random moments | – | 0.17 %, 0.42 %, 0.06 %, 0.03 % |

What it found in the first version, now fixed: each layer's halo and corona
were packed as on/off flags, so a newborn cirrostratus switched its halo on
at once and a thickening veil lost it at an optical depth of exactly 4 (29
moments); the castellanus layer's top was taken from the hour's deck rather
than from the realizations showing, so the turrets' base jumped on the hour
when the next hour had no castellanus (6 moments). A later probe caught the
threshold of a newborn undulatus layer (below) jumping to "all cloud": an
unguarded Newton step from a tiny cover; the solver is now bracketed.

**On the screen.** Rendered frames 1 s apart across the hour (same camera,
same noise), old program against new:

| hour (local) | before: mean change, grey levels | pixels changing > 25 levels | now: mean change | pixels > 25 |
|---|---|---|---|---|
| Kuala Lumpur 11:00 | 7.7 | 4.8 % | 0.21 | 0.01 % |
| Kuala Lumpur 12:00 | 16.9 | 34.4 % | 0.22 | 0.00 % |
| Kuala Lumpur 13:00 | 10.2 | 3.9 % | 0.22 | 0 |
| Paris 11:00 | 2.0 | 0 | 0.10 | 0 |
| Paris 12:00 | 3.2 | 1.2 % | 0.09 | 0 |
| Paris 13:00 | 8.9 | 14.2 % | 0.005 | 0 |

What is left in a second is the second's drift.

The GPU rule G11 found one more jump the CPU detector could not see: a
castellanus layer whose cover rose from zero drew its turrets at full size at
once (turrets stood on cloud that was not there yet); a second across that
hour changed the picture by 10 %, a tenth of a second by as much. Fixed:
turrets stand on the cloud there is.

**Minute by minute.** Each forecast played through at one-minute steps
(18,480 minutes in all), the sky mask compared minute to minute. The share
of sky directions that change in a minute is the same across an hour as
within one: KL 15.7 % vs 16.4 %, Paris 24.2 % vs 25.1 %, SF 5.9 % vs 5.9 %,
Chicago 1.6 % vs 2.6 % – and close to what the wind alone moves in a
minute over the old program's frozen maps (15.0 %, 23.8 %, 5.0 %, 1.9 %):
most of what changes from minute to minute is drift, as in the sky.

**The pattern's own change** (the change the wind does not explain), in each
layer's drifting frame, against what the theory predicts from the blend's
correlation – a bivariate normal with correlation Σ*w_j*(θ₁)*w_j*(θ₂):

| | Ac | As | Cb | Ci | Cs | Cu | Sc | St | all |
|---|---|---|---|---|---|---|---|---|---|
| measured ÷ predicted, per minute | 0.99 | 1.01 | 0.99 | 1.00 | 0.99 | 1.01 | 1.00 | 0.97 | 0.99 |

(28,035 layer-minutes; correlation 0.92 minute by minute.) Across the hours
the pattern changes by 0.87 % of a layer's area a minute on average, within
them by 0.95 %.

**Cover exactness** – the area over the threshold, on 40,000 random points
over each layer's largest tile (and many wavelengths of its waves), against
the forecast's cover, every minute: mean error 0.005 of the sky, 95 % within
0.019, worst 0.072 (33,300 layer-minutes; the largest errors on the layers
with the largest elements, Cb and St, where a tile holds fewest of them).
Undulatus layers: mean 0.002, worst 0.019 – they were drawn up to 0.4 wrong
when every realization carried the same waves. Castellanus 0.003; the gap
nets of perlucidus and lacunosus take 0.006 of the cover, as they should.

**Time-lapses** (`transition_study/`): the old and the new program side by
side through three hours of the Kuala Lumpur and Paris forecasts, a frame
every two simulated minutes, and the change between frames.

In Kuala Lumpur (10:30–13:30, Cb, Sc, Ac, Ci) the old program's frames
jump on each hour – by 9.8, 10.7 and 27.7 grey levels on average, against
about 4 between other frames – and the new one's change between frames stays
at 2.5–5.8 all through. In Paris (a clearing morning) the old program showed
a clear sky from 13:00 until its next refresh; the new one starts forming
the cumulus congestus of 14:00 from 13:00, so its frames change more after
13:00 than the old one's.

**Self-test.** Rules T1–T14 (the blend's algebra, exact cover with and
without waves, continuity of everything the renderer reads across the hours,
C¹ smoothness and no overshoot, drift and renewal integrals, path
independence, refresh stability, regional maps carried between the hours,
sounding interpolation, perlucidus gaps on the cell walls, holding past the
forecast's end, optics fading, étage blending, the upper atmosphere) and G11
(on the GPU: no jump at the hours; the regional maps continuous) – each
checked to fail on the code without its fix. The whole self-test: 2,789
checks, 0 failed.

## 6. Second stress round (30 September)

Real forecasts rarely change as hard as a forecast can. So the second round
scripted ten skies by hand, hour by hour, through the program's own decks
(`scene.deck_from_spec`) and continuous sky: a warm front (Ci → Cs with a
halo → As → Ns with rain → Sc → Cu, the wind veering 130°), a tropical
afternoon (Cu → Cb capillatus with an anvil → dissipation), fog lifting at
dawn with the eye inside it, a layer whose cover flips 95 % / 5 % every
hour, a layer jumping between 1.2 and 5.5 km every hour, sixteen layers at
once (the renderer holds twelve), castellanus growing and losing turrets,
undulatus whose wind reverses every hour, cirrus in a 60–80 m/s jet, and a
cirrostratus veil thickening into altostratus. On top: nine broken forecast
answers fed through the request path, twilight rendered frame by frame with
the exposure metering as in the app, and the app itself driven under Xvfb
through 21 actions (time shifts played through, a far jump, refetch, R,
layers by hand, offline, ×3600).

| | result |
|---|---|
| teleport detector, 354 moments of the ten skies | 0 jumps; the largest ratio of any number's change in 0.2 s to its change in 2 s: 0.109 |
| minute by minute, 3,720 minutes | each hour's minute changes 0.93–1.11 × as much as its four neighbours |
| the pattern's own change, measured ÷ predicted | 0.996 (106.8 ÷ 107.3); castellanus 0.93 (below) |
| the app, 21 actions in 74 s | 0 exceptions, 0 errors in the pattern worker |

**Found and fixed.**

* *Broken answers.* An hour given twice shifted every later hour by one (the
  forecast's own interpolation and the sky both); an infinite wind speed
  became NaN winds on the layers for three hours and jumps of their drift.
  The answer is now laid on a regular hourly axis first (`forecast.clean_hourly`):
  a repeated hour keeps its first values; a missing hour and every
  non-number (NaN, infinity, null) are gaps, filled in time when at most six
  hours long – linearly, wind directions the short way round, categories
  (weather codes) from the hour before. The status line says how many values
  were filled. Rule T15.
* *Cover on tiles of a few large elements.* The blend Σ*w_j*Z_j* is standard
  normal for independent realizations; on a tile of three 40 km Cs
  nebulosus elements two realizations are correlated by chance (±0.2), the
  blend's spread over the tile is not 1, and the drawn cover was off by up to
  7.7 % (mean 1.8 %). The threshold is now set for the blend's own spread,
  √(Σ*w_i*w_j*ρ_ij*) with ρ measured between the realizations on the tile
  (`Sky.blend_spread`): mean 0.9 %, worst 3.6 % – what remains is the tile's
  higher moments. Castellanus realizations share, by design, the rows their
  turrets stand on (ρ = +0.33); that dependence is not a bivariate normal one
  and a Gaussian spread made of it overcorrects (+0.7 % became −1.2 %), so it
  is left out. The shared rows also make castellanus flip 23 % less often
  than independent fields would; the prediction with the correlation is
  within 7 %. Rule T16.
* *Twilight.* Six whole-picture jumps in the renderer, each at a fixed
  elevation of the Sun, measured as the change of the picture in 2.4 s (the
  Sun moving 0.01°) at dusk in Kuala Lumpur under a full Moon:

  | Sun | what switched | deployed | now |
  |---|---|---|---|
  | +4° | the metering's range (3 × the base by day, 60 × in twilight) | × 6.7 | × 1.01 |
  | 0° | the halo, cut off at the horizon | × 0.90 | × 0.99 |
  | −2° | refraction, cut off while still 0.73°: the Sun's direction fell 0.73° | × 0.41 | × 0.98 |
  | −6° | the exposure's moonlight factor (× 3.5 at full Moon) | × 0.86 | × 1.04 |
  | −10° | the light on the clouds, from the Sun to the full Moon | × 97 | × 1.00 |
  | −14° | the metering's range (60 × to 1.5 × the base) | × 0.075 | × 0.99 |

  Now refraction tapers to nothing between −1° and −3° (no line of sight to
  bend there; the Moon had the same 0.73° jump at moonrise); the metering's
  limits move with the Sun over four degrees about +4° and −14°
  (log-linear); the Moon's share of the light and of the exposure rises from
  0 at −6°, where no cloud is sunlit any more (the Earth's shadow is 20 km
  deep by −4.5°), to 1 at −10°; the halo fades over the Sun's last two
  degrees. Two smaller steps went too: the sky's light on the clouds was held
  for each quarter degree of the Sun (steps of 15–70 % in twilight, once a
  second at ×60) and is now interpolated between the quarter degrees; each
  metering's step (every third frame) is spread over the three frames. Rules
  T17 and G12.

Every rule was run against the deployed code first and failed there. The
self-test: 2,802 checks, 0 failed.

**Found after deployment.** The panels (the controls on the left, the
explanation of the clouds in view on the right, the help) had not been drawn
since the transitions update of 28 September. The layers' parameters were
moved then into a uniform block bound to binding point 0 – the binding
pyglet keeps for the projection its shapes and text are drawn with – so from
the first frame the panels were drawn with the layers' numbers as their
projection, off the screen. The block is now on the context's last binding
point, which pyglet hands out last; rule G13 draws a panel-like quad after
the sky and checks it is there (on the 28 September code it covers 0 % of
its area). My app runs had captured the rendered sky, never the whole
window, so they did not see it. At the same time the program was set to
open paused in real time (×1) instead of playing at ×60; the panel's ×10,
×60 and ×300 are the time-lapses. The self-test: 2,804 checks, 0 failed.

**Found in the whole-window pictures.** Twenty pictures of the whole window
(sky and both panels), taken after that fix by driving the program with real
mouse clicks and key presses, showed five more faults. Two were mine, from
the transitions update of 28 September:

* after −1 h, −10 m, +10 m, +1 h or now, with the clock paused and no model
  running (under a high ice veil or a clear sky the model has nothing to
  grow), the left panel kept the old time, Sun and Moon. Since 28 September
  a time shift is played through over one to three seconds instead of set
  at once, and the panel was redrawn at its start and afterwards only while
  the clock or the model ran. Now the panel is also redrawn, twice a second
  at most, whenever what it shows has changed with no click (the clock of a
  shift, a layer that has formed). Rule G15 runs the program itself, paused
  with no model, shifts an hour and reads the panel (before: the old time).
* the clock started by itself. Since 28 September the program restarts the
  cloud model on its own when the weather calls for another kind of cloud
  (or the clock went back), and the restart went through the same start as
  a click on a model, which starts the clock – so a paused sky began to
  play after a time shift into other weather (Chicago, 29 September:
  paused at 10:30, three clicks on +1 h, and the clock ran on from 13:30).
  A click on a model still starts the clock, as it always has; a restart the
  program makes by itself leaves it as it was. Rule G16.

Three were older – all in the 25 September version as well:

* the layer added for convection (when the sounding has the buoyancy but no
  layer of Cu or Cb) took its cover from the forecast's low cloud – and read
  0 % as a missing figure, giving Cumulonimbus 4/8 (Cumulus 1.8/8), where
  2 % gave 0.16/8. Kuala Lumpur, 24 September 20:00–21:00: 0 % low cloud,
  no rain, CAPE 2,520–2,650 J/kg – and a 4/8 Cumulonimbus in the sky. 0 %
  now gives no cover; only a missing figure takes the defaults. Rule P12.
* the help box (the h key) was drawn empty. Its lines were made as pyglet
  labels that nothing kept, and pyglet (2.0 and 2.1 alike) deletes a label
  the moment nothing refers to it – before the box was drawn. The labels are
  now held until drawn. Drawn, the two longest lines ran past the right edge
  of the 470-px box in a monospaced font at 11 pt; the box now takes the
  width of its longest line.
* a wrapped text in the panels was given the height of a guess (0.7 × the
  font size per character), so where the font's lines broke differently its
  last line ran into the next line or button; and each placing of the panel
  after the first (a turn of the mouse wheel) moved every wrapped text down
  by the height of its extra lines, over the widgets below it, until the
  panel was rebuilt. Each wrapped text is now laid out by pyglet with the
  font the computer has, and its lines hang from the top of a box that tall.

Rule G14 draws the help as the program does and checks that each of its
eleven lines leaves text on the screen (0 of 11 before) and that no text
falls outside the box, and lays out the program's own long lines in both
panels, before and after a scroll, checking that no line leaves its box or
reaches the next widget. The self-test: 2,828 checks, 0 failed.

**Not changed, by design.** A jump of more than three hours dissolves the
clouds but moves the Sun at once – another time was asked for. A layer
waiting for one of the renderer's twelve groups fades in over 0.9 s when one
frees; a drawn layer is never swapped out.

## 7. What it does not do

* The forecast says nothing within the hour. The cubic between the hours is
  an assumption – the simplest smooth one that keeps the hours' values and
  turning points – not information.
* The renewal times of layer clouds (Sc, Ac, Cc, Ci, Cs, As, Ns) are this
  program's estimates from physical time scales, not measurements.
* Layers are followed by the overlap of their depths. Two layers that merge
  or one that splits are drawn as one thinning while the other grows – a
  dissolve, not a jump, but not a merger either.
* On a slow machine a layer's pattern can lag behind the clock and catch up
  (visibly faster evolution for a moment); it never jumps.
* The cloud-resolving model's own clouds are the model's; when it restarts
  its clouds hand over to the layers by dissolving, not by interpolation.

## Sources

* Romps, D. M., Öktem, R., Endo, S., Vogelmann, A. M. (2021). On the life
  cycle of a shallow cumulus cloud: is it a bubble or plume, active or
  forced? *J. Atmos. Sci.* 78(9). doi:10.1175/JAS-D-20-0361.1 –
  <https://davidromps.com/papers/pubdata/2019/lifecycle/19lifecycle.pdf>
* NOAA JetStream: Life cycle of a thunderstorm –
  <https://www.noaa.gov/jetstream/thunderstorms/life-cycle-of-thunderstorm>
* Germann, U., Zawadzki, I. (2002). Scale-dependence of the predictability
  of precipitation from continental radar images. Part I. *Mon. Wea. Rev.*
  130, 2859–2873 –
  <https://journals.ametsoc.org/view/journals/mwre/130/12/1520-0493_2002_130_2859_sdotpo_2.0.co_2.xml>
* Sommeria, G., Deardorff, J. W. (1977). Subgrid-scale condensation in
  models of nonprecipitating clouds. *J. Atmos. Sci.* 34, 344–355 –
  <https://journals.ametsoc.org/view/journals/atsc/34/2/1520-0469_1977_034_0344_sscimo_2_0_co_2.xml>
* Mellor, G. L. (1977). The Gaussian cloud model relations. *J. Atmos.
  Sci.* 34, 356–358 –
  <https://journals.ametsoc.org/view/journals/atsc/34/2/1520-0469_1977_034_0356_tgcmr_2_0_co_2.xml>
* Heitz, E., Neyret, F. (2018). High-performance by-example noise using a
  histogram-preserving blending operator. *Proc. ACM Comput. Graph.
  Interact. Tech.* 1(2) – <https://inria.hal.science/hal-01824773/>
* Anagnostou, E. N., Krajewski, W. F. (1999). Real-time radar rainfall
  estimation. Part I: algorithm formulation. *J. Atmos. Oceanic Technol.*
  16, 189–197 –
  <https://journals.ametsoc.org/jtech/article/16/2/189/1609/Real-Time-Radar-Rainfall-Estimation-Part-I>;
  the same method in pysteps:
  <https://pysteps.readthedocs.io/en/stable/auto_examples/advection_correction.html>
* Fritsch, F. N., Carlson, R. E. (1980). Monotone piecewise cubic
  interpolation. *SIAM J. Numer. Anal.* 17, 238–246. doi:10.1137/0717021;
  Fritsch, F. N., Butland, J. (1984). A method for constructing local
  monotone piecewise cubic interpolants. *SIAM J. Sci. Stat. Comput.* 5,
  300–304. doi:10.1137/0905021 – <https://epubs.siam.org/doi/10.1137/0905021>
* Olano, M., Baker, D. (2010). LEAN mapping. *Proc. I3D 2010*, 181–188 –
  <https://userpages.cs.umbc.edu/olano/papers/lean/lean.pdf> (the
  pattern's mip levels carry *z* and *z*², so a distant texel gives the
  fraction of it that is cloud)
* Schneider, A. (2015). The real-time volumetric cloudscapes of Horizon
  Zero Dawn. SIGGRAPH Advances in Real-Time Rendering –
  <https://advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf>
* Taylor, G. I. (1938). The spectrum of turbulence. *Proc. R. Soc. Lond.
  A* 164, 476–490 (frozen turbulence).
