# PD-AOS Sampler for ComfyUI

An adaptive sampler. Before it samples anything, it measures the model and your
prompt with two small probe passes, and then uses those measurements to:

1. **place the steps of whatever schedule you plug in** where the measurements say
   they are needed, and
2. **decide, for each part of the picture separately, how far ahead it looks** when
   it works out where the image is heading.

> **Tested with Krea 2 and Anima models.** Other flow models are expected to work,
> but have not been verified.

## Provenance — what this is, and what it is not

**This project began as an implementation of A-FloPS** — Jin, Xiao & Gu, *A-FloPS: Accelerating Diffusion Models via
Adaptive Flow Path Sampler*, AAAI 2026 ([arXiv:2509.00036](https://arxiv.org/abs/2509.00036)) — **and became
something only very loosely inspired by it.** The name was changed to PD-AOS because the old one over-claimed, so
here is the split, stated plainly:

- **From the paper:** the idea of working in a reparameterised flow time, and — still included, as two reference
  nodes — the paper's own **Algorithm 2 ("A-Euler")**, a faithful reproduction of the authors' reference
  implementation, kept so the original method can be compared against this sampler directly.
- **This sampler, which is a different thing:** before sampling it measures *your* model and prompt with two small
  probe passes, then uses those measurements to place the steps of the schedule you plug in and to decide, per region
  of the image, how many previous steps to look at. The probes, the per-pixel order control, the Autotuner and the
  local-field machinery are this project's own and have no counterpart in the paper.
- **They are not the same algorithm.** On an exact analytic ODE test the paper's step parameterisation sits closer to
  the true solution than this sampler's own step does. That is a statement about *discretisation accuracy*, not about
  how the images look. If you want A-FloPS itself, use the paper's nodes below or the authors' implementation; if you
  want this, use PD-AOS.
- **The attribution stays**, and so does the comparison: the reference nodes are there so you can check the above for
  yourself rather than take it on trust.

## What it does

- **Measures first, samples second.** The probes run on tiny latents before the
  real sampler starts. Nothing is guessed and nothing is fitted to a formula: the
  schedule and the per-pixel decisions come from what was measured on your model
  and your prompt.
- **Keeps your scheduler.** The Autotuner fine-tunes the sigma list you give it. If
  it has no usable measurements, it passes your list through untouched.
- **Works per pixel.** Some areas of a picture follow a clear direction and can be
  pushed harder; others are busy or detailed and cannot. The sampler decides this
  separately for each area.
- **Refuses what is not safe.** A higher setting is rejected when it would make the
  step come out too large, and a corrected step is dropped when it departs too far
  from the plain one. Raising a setting can therefore never force a bad step through.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

## Nodes

### PD-AOS Autotuner (probe + schedule)

- Inputs: `guider` (the guider your run uses), `sigmas` (any scheduler's output).
- Outputs: `sigmas` (fine-tuned) and `options` (connect this to the Sampler's
  `options+` input).

| setting | what it does |
|---|---|
| `probe_steps` | How many steps the model probe uses. This probe decides the schedule, and its result is kept for every prompt with the same model and settings. **Higher = a more reliable schedule decision and a slower first run; lower = a faster first run.** |
| `probe_resolution` | The picture size the model probe runs at. **Higher = the probe measures at a scale closer to your real output and costs more; lower = cheaper and less representative.** |
| `cond_probe_steps` | How many steps the prompt probe uses. This probe runs again for every new prompt, so this is the setting you feel on each generation. |
| `cond_probe_resolution` | The picture size the prompt probe runs at. |
| `bias` | How the probes' placement is tilted, used only when the derived schedule is on. **1.00 leaves it as measured. Below 1.00 puts more steps late, where detail is built. Above 1.00 concentrates the end into a sharper final drop. The first and last steps never move.** |
| `warp` | Moves the sigma values themselves, leaving the first and last where they are. **1.00 keeps your schedule as it is. Below 1.00 reaches low noise sooner.** |
| `auto_warp` | **ON: the `warp` setting is ignored and the strongest safe value is chosen for you. OFF: your `warp` value is used as set.** |
| `probe_smin_frac` | How far down the noise range the two probes **look**, as a fraction of where your schedule starts. **0.10 is what this node has always used: the probes stop at a tenth of the starting noise, so on a 16-step run nothing below 0.17 was ever measured. Lower measures deeper — 0.0001 reaches the model's own smallest noise level — and this is a MEASUREMENT setting: at the default it changes nothing, and lowering it does not by itself move a single step. Cost: extra probe model calls on every run, roughly four times as many at 0.0001 as at 0.10; the count is written into the run's log.** |
| `probe_schedule` | **OFF (the default) is what this node has always done: your scheduler places the steps and `bias` does nothing. ON: the probes' measurements place them — the hard stretches of the run get more steps and the easy ones fewer — with `bias` tilting that placement.** |

### PD-AOS Sampler

| setting | what it does |
|---|---|
| `log_errors` | Writes what the sampler decides at every step to the console, and fills the Error Report node. For testing. |

### PD-AOS Debug Options

**This node carries the engine's adaptive controls.** Every widget on it defaults to
the engine's own shipped value — the node reads those defaults from `ENGINE_DEFAULTS`
at load time rather than repeating them, so a widget and the engine cannot drift apart.
Leave the node disconnected and the sampler uses the engine defaults.

It has an **`options+` input of its own**, so it can be **chained** with another
options node — for example the Autotuner's `options` output into this node, and this
node's `options` output into the Sampler. Fields from the upstream node pass through
untouched; where two nodes set the same field, the one **closest to the Sampler** wins.

The controls fall into four groups:

| group | what it decides |
|---|---|
| **order** (`order_floor`, `order_max`, `order_by_bound`, `order_pixel_adaptive`, `order_elev_gain`, `order_demote_gain`) | how many previous steps the sampler looks at. **Higher = it follows the trajectory direction harder and costs more per step; lower = flatter and cheaper.** |
| **automatic demand** (`order_floor_adaptive_rtol`, `_graded`, `_lo`, `_hi`, `_r_hi`, `order_demand_q`) | when and how far the sampler raises its own order demand from the run's measured error. **`rtol` is the flip point: the demand acts while the step's error over its tolerance is above it, and asks for nothing below.** |
| **safety / anomaly** (`rtol`, `wmax_adapt`, `wmax_adapt_k`, `guard_ratio`, `guard_sensitivity`, `guard_window`, `guard_quantile`, `guard_floor`, `guard_hold`, `guard_mad_floor`, `an_value`, `an_from`, `corrector_ab`) | what the sampler refuses to do. The guard watches for a step that departs from its own recent behaviour; the anomaly detector damps a region that does not recover. **Raising a threshold lets more through; lowering it refuses more.** |
| **smoothing** (`ord_ema`) | how quickly the per-pixel order reacts. **Higher = smoother and slower to react; lower = twitchier.** |

The shipped values, their meanings and the code that consumes each one are in
`research_samplers/aflops.py` — grep the setting's name; `ENGINE_DEFAULTS` holds the
defaults. They are not repeated here, because a number copied into a README drifts
away from the engine silently.

### PD-AOS Noise (ETA) Options

**Every stochasticity control in one node.** The sampler is deterministic by default;
this node is how you add ancestral noise and test whether it helps. Nothing on it is a
finished setting — the Debug Options node above carries the switches that reached a
decision, and this one carries the ones that have not.

Connect it in the chain, **Autotuner `options` → this node's `options+` → the Sampler**,
and set `eta` above zero. Its defaults are chosen so that merely connecting it does
something: the noise window is opened, and the probe path the escape needs is enabled.
Without that window the engine applies noise only to the first step, which has no
trajectory history for the detector to read — which is why earlier attempts at this
appeared to do nothing.

| setting | what it decides |
|---|---|
| `eta` | **how much** noise, 0 to 1, the same scale as another sampler's eta. It is a ceiling: what is actually injected each step is written to the run log as `eta_eff`. |
| `eta_mode` | **where** the noise goes. `adaptive` injects only on steps where the engine's detector finds the trajectory stalled — that is the escape. `constant` injects everywhere; `cosine`, `linear`, `exponential` and `sigma_power` fade it out. |
| `s_tmin`, `s_tmax` | the sigma window the noise is applied in. `s_tmax` 0 means no upper bound. |
| `escape` | **`auto` or `off`.** `auto` derives the two firing thresholds from the probe's own measurement of your model, so the base is measured rather than copied from another checkpoint. `off` disables the detector. The three numbers it resolves to are in every run log. |
| `escape_gain` | sensitivity, as a multiplier on those derived thresholds: 1.0 is the measurement unchanged, higher fires more readily, 0 never fires. |
| `fade_floor`, `fade_sigma` | `fade_floor` is how much of the **base floor** (`eta_min`) survives the late fade. The shipped `eta_min` is 0 and has no widget, so this control does nothing in a shipped graph; it is meaningful only if a custom Options node sets `eta_min` above zero. It does not fade the escape. `fade_sigma` is where that fade starts; −1 means "use the probe's measured oscillation onset". |
| `noise_survival_probe` | whether the probe measures how big a kick your model can still correct at a given sigma — the budget any escape has to spend. `follow eta` measures it whenever eta is above zero; `force off` saves up to ten model calls on a session's first run. |

### PD-AOS Experimental Options

**A deliberately volatile node.** It exists so a test arm can be added or removed
without the widget-ordering risk that governs every stable node here: ComfyUI stores a
UI graph's widget values as a *positional* list, so dropping a widget from the middle of
a stable node silently re-maps every later value onto the wrong widget. This node is
exempt because it is only ever used for testing, and nothing saved depends on it —
switches may come and go between versions. **Every switch on it defaults to the old
behaviour**, so connecting it changes nothing until you set one, and disconnecting it
returns your graph to the shipped path. Chain it anywhere in the options chain; if you
don't want experimental behaviour, don't add it.

| setting | what it decides |
|---|---|
| `escape_mode` | **which conditions may arm the escape's stall detector.** `flatness` is the shipped behaviour and is left exactly as it was. `momentum` **adds** a second way to arm the same half of the detector — a run-relative test: the per-step movement has fallen to 70% or less of its own median over the last 4 steps *and* the image has not gained more than 15% detail over that window. The reversal half of the detector is untouched, so an A/B changes one thing. **Why it exists:** measured on the first build shipping this sampler's noise options, the two quantities the shipped stall test multiplies move in *opposite* directions as a run converges — the movement falls while the high-frequency detail *rises* as the picture forms — so their product read zero on every run measured, and no sensitivity setting reached it. The extra noise this arm adds was **not** judged on appearance by anything but your own eye: it is a test arm, not an improvement. Both arms are readable from the run log (`escape_mode` in the config, and `eta_momentum`, `eta_momentum_ratio`, `eta_growth_rate` per step, stamped on **both** arms so the two are comparable). |
| `escape_growth_probe` | whether that arm's **growth budget is derived from your model** instead of fixed. OFF is the delivered behaviour. ON makes the budget depend on how consistently the probe measured your model's trajectory moving in *one* direction: a model whose trajectory keeps reversing — this sampler's own definition of being stuck, every pixel moves but back and forth — gets the budget opened up so the escape can fire, while a model that commits cleanly keeps the delivered budget and behaves exactly as it does today. **Why a measurement rather than a number:** over 31 runs, the quantity separating the runs where the escape fired from the runs where it did not was the probe's own direction-consistency curve across its *full* range (firing runs peaked at −0.32 or below, silent runs at +0.60 or above), and the same model sampled in 16 steps builds more detail per step than in 50, so one constant cannot mean the same thing at both step counts. Measured effect: a 50-step checkpoint was untouched (51 armed steps either way) while 16-step turbo checkpoints went from 0 to 18 and 0 to 8. Only does anything when `escape_mode` is `momentum`. The budget used is in every run log as `eta_growth_budget`, and the measurement behind it as `eta_escape_commit`. |
| `noise_survival_curve` | **measurement only — it changes nothing the sampler does.** ON makes the model probe *record* its noise-absorption measurement instead of only reducing it to one number: at **four sigmas spread across your run's own range** it injects a perturbation at **up to four amplitudes** and writes down how far that perturbation still shows in the model's next prediction, as a fraction of an unperturbed reference. **How to read it:** a small value means the model corrected the kick — this model at this sigma will not let noise survive; a value near 1 means the kick persisted. **Why keep it:** that curve is the empirical form of "how much noise can this model absorb here", which is what a *derived* noise schedule would be built from rather than a hand-tuned one. The measurement has always been computed and then immediately compared against a fixed 0.35 to produce a single number, so the only surviving record was that number — and across 10,602 logged profiles it is missing in 69% of them and sits at the clamp or the skipped-probe fallback in most of the rest, so it has rarely carried a usable per-model reading. **The ladder was also too narrow to see a shape:** three separate measurement rounds never sampled above σ 0.66 because it walked the *probe grid's* quartiles, and the amplitude ladder stopped at 0.35 — truncating exactly the region where models stop correcting. It now spans your run's sigma range and records every rung, including the failing one. **Cost:** ~1 model call per sigma/amplitude pair plus 1 reference per sigma — around 20 calls on the first run for a model, on the **probe path only**, then cached; no per-step calls, so generation speed after the first run is unchanged. Read it in the run log as `noise_survival_curve`. |
| `eta_from_probe` | **derive the noise level from the probe's measurement of your model**, instead of inferring it from whether the image looks stuck. OFF is the delivered behaviour. The probe already injects a perturbation one step ahead and measures how much of it still shows in the model's next prediction; that response is **linear in the size of the kick** (measured on five checkpoints, constant to within 0.2–4% across the amplitudes tested), so one number per σ describes how strongly your model lets a kick through. Asking for the level that holds the surviving kick under a residual budget gives **η(σ) = (budget ÷ that number)²**. **Why it replaces the escape rather than adding to it:** it answers "how much noise can this model absorb *here*", which is the question the escape's detector was approximating from a different quantity — so it is a measurement, not another policy. **Measured why it matters:** on one checkpoint that number is ≈1.35 at σ 0.66 and ≈0.31 at σ 0.28, so a 0.10 budget asks for η **0.0055** early and **0.072** late, against a fixed η of 1.0. The model tolerates far *less* noise early than the escape assumed. Applies to every `eta_mode` and replaces that mode's level; `eta` remains the **ceiling**. Falls back silently to the shipped behaviour when the probe has no curve. Both numbers are in the run log: `eta_eff` (used) and `eta_probe` (derived). |
| `eta_budget_from_probe` | whether that budget comes from the **probe** instead of the widget. OFF = the widget governs. ON uses the number the probe already measures on every probed run — how big a kick this model still corrects, at the probe's own 0.35 threshold — which is the same quantity and the same units as the budget, so it replaces an invented default with a measurement. **What it does not do:** find a "golden" setting. Measured across 20 curves, the response stays on its low-kick line all the way to the top of the amplitude ladder on 21 of 27 σ samples, so there is no reachable knee to derive an optimum from — and the quantity says what the model *corrects*, not what a viewer accepts. If the images look under-noised, set `eta_probe_budget` yourself. Falls back to the widget silently when the probe has no value. `_eta_budget_src` in the run config records which was used. |
| `eta_probe_budget` | only used when `eta_from_probe` is ON: how much of an injected perturbation may survive into the next prediction. Lower = less noise. The probe's own pass/fail threshold has always been 0.35, so the 0.10 default sits well inside what that experiment already treats as corrected. This is the one number shaping the derived schedule, and it is a deliberately **safe** default rather than an optimal one — the sweep is yours. |
| `escape_momentum_flatness` | whether the `momentum` arm must **also** see a flat image before it fires — the check the original flatness detector always applied and this newer arm was missing. **Leave ON.** Measured on a matched pair: a 16-bit pixel-art prompt at `eta` 0.0 vs 1.0 (same seed, scheduler and step count) gave a crisper image with a cleaner, more uniform pixel grid at 0.0, while 1.0 produced "half/half pixels" — cells the sampler could not resolve, mushed together. The run log shows why: every step the arm fired had the image's fine detail at **1.4× to 3.3×** the flatness threshold, i.e. the picture was *full* of structure the model was deliberately placing on a pixel grid. The arm fired anyway because it only tested whether detail had stopped *growing*, and on pixel art a grid that stops growing has **locked**, not given up. With this ON those firings are suppressed and the run matches the deterministic arm that looked better. **This is not free:** on the 31-run corpus it also removes 28 of 85 firings, mostly on 16-step turbo checkpoints. The per-step reading is in every log as `eta_detail_abs`, against the threshold stamped as `eta_escape_detail`. |

### PD-AOS Error Report

Optional. Writes the sampler's per-step decisions to a JSON file.

### A-FloPS A-Euler (paper) / A-FloPS A-Euler Schedule (paper)

The reference implementation of Algorithm 2 from Jin et al., AAAI 2026
(arXiv:2509.00036), included so the adaptive sampler can be compared against it.
These are plain samplers: they do not use the probes and they make one model call
per step. They are not needed for normal use.

## Wiring

![Wiring example: BasicScheduler → PD-AOS Autotuner → the PD-AOS Sampler → SamplerCustomAdvanced](docs/wiring-example.png)

1. Build a normal `SamplerCustomAdvanced` graph (model, noise, guider, sampler,
   sigmas).
2. Add **PD-AOS Autotuner**: connect your guider to `guider`, and your scheduler's
   output (for example BasicScheduler "simple") to `sigmas`.
3. Connect the Autotuner's `sigmas` output to `SamplerCustomAdvanced.sigmas`.
4. Add **PD-AOS Sampler**: connect the Autotuner's `options` output to its
   `options+` input, and its `sampler` output to `SamplerCustomAdvanced.sampler`.
   The Sampler's own `model` input is optional: connecting the model there lets it
   read the model's type and settings directly. The Autotuner needs no such input,
   because it receives the model through the guider.
5. Run. The first generation probes the model and the prompt and stores the
   measurements; later generations with the same model and settings reuse them.

Optional nodes that change what the sampler does can be spliced into the same
`options+` chain — the Autotuner's `options` output into their `options+`, and
their `options` output onward to the Sampler. **PD-AOS Noise (ETA) Options** adds
ancestral noise, and **PD-AOS Experimental Options** carries test arms that may be
added or removed between versions. Where two nodes set the same field, the one
closest to the Sampler wins.

## Notes

- **Higher is not automatically better.** Looking further ahead follows the
  trajectory's direction harder, and on some models and prompts that gives a
  cleaner, more finished image, while on others it does not. The sampler measures
  per pixel and refuses what is not safe; the settings above let you push further,
  and the result should be judged by eye.
- **The first run costs more than later ones**, because of the model probe. It is
  cached per model, settings and schedule, so it is paid once per model
  configuration rather than once per generation.
- **A probe cache is per ComfyUI session.** Restarting ComfyUI re-measures it, so two
  runs that straddle a restart are not guaranteed to receive the same calibration.
  Compare runs from one session, or expect the difference.
- **A few experimental switches exist in the code but are not shown in the node
  interface.** They are fixed at their shipped values. They are not described here
  because they are not proven.
