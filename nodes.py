import json
import logging
import comfy.samplers
from .research_samplers import aflops as aflops_mod
from .research_samplers import paper_aeuler as paper_aeuler_mod
from .research_samplers.aflops import aflops_engine, ENGINE_DEFAULTS

OPT_TYPE = "RS_SAMPLER_OPT"

# The probe keys the Noise Options node supplies WHEN NOTHING UPSTREAM DID.
# They are the Autotuner's own defaults (`nodes.py` autotune signature and its
# `opts` dict), so a graph with no Autotuner behaves like one running a default
# Autotuner instead of like a STANDALONE run -- and in a standalone run the
# engine zeroes every escape threshold (`aflops.py:9762-9767`), which would make
# every widget on that node silently inert.  Used with `setdefault`, so a chained
# Autotuner's settings always win and there is still one owner per value.
_NOISE_PROBE_FALLBACK = {
    "probe_enabled": True,
    "probe_steps": 10,
    "probe_resolution": 48,
    "cond_probe_enabled": True,
    "cond_probe_steps": 4,
    "cond_probe_resolution": 16,
    "cond_probe_weight": 0.5,
    "probe_smin_frac": 0.1,
}


def _f(v):
    return float(v[0]) if isinstance(v, (list, tuple)) else float(v)


def _b(v):
    v = v[0] if isinstance(v, (list, tuple)) else v
    return bool(v)


def _opts(v):
    """Normalize the options+ input: None, a single dict, or a list of dicts."""
    if v is None:
        return []
    if isinstance(v, dict):
        return [v]
    return [o for o in v if o]


def _sig_tensor_list(v):
    """SIGMAS input (tensor or list) -> descending float list or None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and v and not isinstance(v[0], (int, float)):
        v = v[0]
    try:
        lst = [float(x) for x in v]
    except Exception as e:
        logging.warning("[PD-AOS] SIGMAS input could not be converted: %s: %s",
                        type(e).__name__, e)
        return None
    if len(lst) < 4 or lst[0] <= lst[1]:
        return None
    return lst




class ResearchAFlopsSampler:
    DESCRIPTION = ("PD-AOS (Adaptive Flow Path Sampler). Implements the core method "
                   "from Jin et al. (AAAI 2026, arXiv:2509.00036): the velocity field "
                   "is decomposed into a linear drift and a slowly-varying residual via "
                   "an adaptive coefficient, then integrated with an exponential "
                   "integrator. Pure single-solver engine: every step runs the "
                   "PD-AOS formula with ZERO extra model calls, and the per-pixel "
                   "adaptive order ladder feeds the residual integrator. Without extra "
                   "inputs it runs fully automatic: noise colour, anomaly "
                   "detection, the local per-region schedule field with its spatial "
                   "fragility prior, and flow/VE detection are all self-tuned from the probes. "
                   "Connect a MODEL for auto-detection. "
                   "Connect Options nodes to override any group of settings.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "log_errors": ("BOOLEAN", {"default": False,
                    "tooltip": "Writes what the sampler decides at every step to the console, and fills the PD-AOS Error Report node. Leave it on while you are testing a prompt; off for normal use."}),
                # THE BOUND A/B WAS HERE AND IS GONE (2026-10-04).  `order_bound_fix` was a Sampler
                # widget for one turn; the A/B ran on three real pairs, and the corrected remainder
                # scale that it toggled is now the ONLY path in `_pixel_order_bound`.  Operator:
                # *"with removal we both mean also implement the new code, right?  Old code goes too."*
                # **A SIDE BENEFIT: removing it restores the shift-safety the five-switch move had
                # won.** The widget order is `[log_errors]` again, which IS a prefix of the published
                # order, so a published-era graph's surplus values are discarded instead of landing on
                # a live switch.  See NOTES for what the A/B measured -- it was INERT on both
                # configurations tried, which says the per-pixel criterion is not the binding
                # constraint, not that the correction is wrong.
                # THE FIVE FINISHED SWITCHES MOVED OFF THIS NODE (2026-10-04): `lf_ab`,
                # `order_by_bound`, `order_floor`, `an_enable` and `wmax_eg_dir` are no longer
                # widgets here.  Each had reached a decision -- by measurement plus the
                # operator's eye -- so each is fixed at its ENGINE DEFAULT and driven, when
                # anyone wants to drive it, from the **PD-AOS Debug Options** node, whose
                # `options` output plugs into this node's `options+` input.  The tooltips moved
                # with the switches; nothing was dropped.
                #
                # WHY THEY ALL MOVED AT ONCE, and this is not tidiness.  ComfyUI stores a UI
                # graph's widget values as a POSITIONAL list (`widgets_values`), so removing a
                # widget from the MIDDLE of this list re-maps every later value onto the wrong
                # widget when an existing saved workflow is loaded -- silently, and onto
                # plausible values.  `wmax_eg_dir` is last in the old order, so keeping it (or
                # any other non-prefix subset) would have handed old graphs `wmax_eg_dir = 1.0`
                # from what used to be `lf_ab = True`: the most aggressive extrapolation, on the
                # knob that changes image STRUCTURE.  `log_errors` is at index 0, so keeping
                # ONLY it means an old list's first value still lands correctly and the surplus
                # is discarded -- existing workflows keep working and pick up the engine
                # defaults, which is the intended outcome of the move.
                #
                # The `if "x" in kw:` blocks in `get_sampler` are kept as BACKWARD-COMPATIBLE
                # OVERRIDES: an API-format graph that still passes the old names, or an Options
                # node that sets the keys, is honoured rather than silently ignored.
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Connect the MODEL for auto-tuning. PD-AOS will detect the model type (flow vs VE), read sigma_data for correct SNR math, detect video latents, and detect distilled models (turbo/schnell/lightning). All auto-tuning can be overridden by Options nodes. Without this, PD-AOS uses a sigma-magnitude heuristic."}),
                "options+": (OPT_TYPE, {"tooltip": "Connect any number of PD-AOS Options nodes here. Each connected group overrides the automatic settings for its fields."}),
                # APPENDED LAST (rule 35).  ComfyUI stores a saved workflow's widget values as a
                # POSITIONAL list, so a widget added anywhere but the END re-maps the values of
                # the widgets after it -- silently, and onto plausible numbers.  Appending here
                # leaves `log_errors` at index 0 and gives an older saved graph a default for
                # this one, which is what makes the change safe to load.
                "use_measured_tolerance": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("use_measured_tolerance", False)),
                    "tooltip": "Which number sets the sampler's tolerance for a step's error: the tolerance the Autotuner MEASURED for your model, or that same value with a fixed MINIMUM of 0.01536 applied on top of it. OFF (the default) applies the minimum, so a measured tolerance below 0.01536 is raised to it -- this is what the sampler has always done. ON uses the measured value exactly as measured, however small it is, which makes the sampler accept less error per step and fall back to the plain step more often. If your model's measured tolerance is already above 0.01536 the two settings behave identically. MEASURED over 439 logged runs on 7 models: the minimum was raising the tolerance on 289 of the 302 runs that had a measured value, and on those runs turning it off moves the typical tolerance by between 1.37x and 2.77x depending on the model, so this is a large change rather than a trim. It is a TEST ARM, not an improvement: both numbers are measurements, they disagree, and nothing here can say which one looks better -- so it stays OFF until you have compared the two on your own model and prompt. The value it used is written into every run's log (`tol_in_force`), together with the measured value it replaced (`rtol_measured`)."}),
                "probe_fuse_per_step": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("probe_fuse_per_step", False)),
                    "tooltip": "How the two probes' measurements are combined. OFF (the default) is what the sampler has always done: the model probe and the prompt probe are each boiled down to one number per setting (tolerance, order cap, guard floor, anomaly thresholds) and those two numbers are averaged 50/50, once, before sampling starts. ON keeps them apart as per-step curves instead: both probes' measured curves of how much the image moves at each step are lined up with YOUR step positions, the sampler watches which probe is actually tracking the run as it goes, and each step's ORDER CAP is worked out from the combined curve -- and when the two probes disagree enough to allow different orders, the sampler takes the SAFER, lower order rather than the average. What ON does NOT change: the guard floor, the anomaly thresholds and the step tolerance. Cost: nothing extra is measured, no extra model calls, and the runtime work is arithmetic on numbers already in memory. It is a TEST ARM, not an improvement: nothing here can say which one looks better on your model and prompt, which is why both are available and why the default is the old behaviour. What it used is written into the run log (`probe_fuse` at the top level, and `fuse_w_in_force` per step)."}),
                "probe_fuse_raise_cap": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("probe_fuse_raise_cap", False)),
                    "tooltip": "Only used when 'probe_fuse_per_step' is ON, and it decides whether the combined curve is allowed to ask for a HIGHER order than the sampler would otherwise use. OFF (the default) means the combined curve can only ever LOWER the order -- the sampler behaves exactly as it does today, just with the two probes combined per step instead of once, so switching the fusion on is a change in ONE thing only and the two settings can be compared fairly. ON lets the combined curve also RAISE the order above the normal cap. Be aware that ON is two changes at once: raising the cap also switches on this sampler's probe-derived order bound in the cases where that bound is normally off, so an image difference under ON cannot be attributed to the probe combination alone. TEST ARM, not a recommendation. Both the setting and the caps it produced are written into the run log."}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"

    def get_sampler(self, **kw):
        log = _b(kw["log_errors"])
        merged = dict(ENGINE_DEFAULTS)
        # THE PER-STEP PROBE FUSION ARM (2026-10-08).  The widget exists so the two behaviours
        # can be compared by EYE, which is the instrument for appearance (the operator cannot
        # read the effect out of a log).  The engine reads `cfg["probe_fuse_per_step"]`; this is
        # the other half of that wiring, and both halves are asserted by
        # scratch_probe_fuse_rig.py (rule 38: the widget is not finished until a rig shows the
        # value reaching the decision it claims to control).
        if "probe_fuse_per_step" in kw:
            merged["probe_fuse_per_step"] = _b(kw["probe_fuse_per_step"])
        # The cap-raise switch, same convention and same reason (one change per switch).
        if "probe_fuse_raise_cap" in kw:
            merged["probe_fuse_raise_cap"] = _b(kw["probe_fuse_raise_cap"])
        # The corrector A/B arm.  NO LONGER A WIDGET (release build): it is fixed at its
        # engine default.  This branch stays so a saved graph, or an Options node that sets
        # the key, is honoured rather than silently ignored -- the normal case now is that
        # it never fires.  Set BEFORE the Options nodes are applied so a connected Options
        # node can still override it.
        if "corrector_ab" in kw:
            merged["corrector_ab"] = _b(kw["corrector_ab"])
        # The order/extrapolation A/B arm (same convention, same reason it is not a widget).
        if "order_ab" in kw:
            merged["order_ab"] = _b(kw["order_ab"])
        # The forced per-pixel order floor.  WAS a Sampler widget; now on the Debug Options node
        # (see the INPUT_TYPES note for why all five moved together).  Kept here as a
        # backward-compatible override, so an API-format graph or an Options node still drives
        # it -- one switch, one place.
        if "order_floor" in kw:
            merged["order_floor"] = int(kw["order_floor"])
        # Which per-pixel ORDER CRITERION runs.  Same convention: Debug Options node now, this
        # override kept for saved graphs and Options nodes.
        if "order_by_bound" in kw:
            merged["order_by_bound"] = _b(kw["order_by_bound"])
        # The SIGN of the endgame-error response.  Moved to the Debug Options node with the other
        # four.  Clamped here as well as in the engine, so a saved graph from before the clamp
        # cannot inject an out-of-range value.
        if "wmax_eg_dir" in kw:
            merged["wmax_eg_dir"] = max(-1.0, min(1.0, _f(kw["wmax_eg_dir"])))
        # The two DERIVED-CONSTANT arms.  Independent, so each can be flipped alone; the
        # derivation logs BOTH the shipped and the derived value regardless, so one run
        # shows the delta even with the switch off.
        # NOTE: pct_derived is NOT A WIDGET ON EITHER NODE in the release build (the
        # Autotuner used to own it).  It is fixed at its engine default and is kept
        # here only so Options nodes and saved graphs can still drive it.  `wmax_bound`
        # moved to the Debug Options node (audit C2: the per-step fusion's lower-cap
        # rule needs it, and a rule no widget can switch is the defect), so the
        # Autotuner's `options` output still carries its own copy at the default; a
        # chained Debug node overrides it, which is the documented order.  The
        # Autotuner's `options` output is applied BELOW, i.e. AFTER these assignments,
        # so an Autotuner copy would silently override a Sampler copy anyway -- two
        # switches with one name, where one of them quietly does nothing.  One switch,
        # one place.
        # DERIVED CONSTANTS and the CUT VARIABLE are NO LONGER WIDGETS -- both A/Bs
        # reached their verdicts, so they are fixed settings in ENGINE_DEFAULTS
        # (`derived_ab = True`, `cut_var = "midpoint"`).  The two blocks below are
        # kept deliberately as BACKWARD-COMPATIBLE OVERRIDES: a saved graph that
        # still carries the old widget value, or an Options node that sets the key,
        # is honoured instead of being silently ignored.  If neither supplies them
        # (the normal case now) these branches never fire and the defaults stand.
        # Both keys remain stamped into `_LAST_CFG`, so every log still states which
        # behaviour produced it.
        if "derived_ab" in kw:
            merged["derived_ab"] = _b(kw["derived_ab"])
        # THE TOLERANCE-MINIMUM ARM (see the INPUT_TYPES note).  A live widget again, so this
        # branch is taken on every run from the UI; the `if "x" in kw:` form is kept so an
        # API-format graph that does not carry the key gets the engine default -- OFF, i.e.
        # today's behaviour -- instead of an error.  Set BEFORE the Options nodes are applied,
        # so a connected Options node can still override it.
        if "use_measured_tolerance" in kw:
            merged["use_measured_tolerance"] = _b(kw["use_measured_tolerance"])
        if "cut_ab" in kw:
            merged["cut_var"] = "midpoint" if _b(kw["cut_ab"]) else "corrected"
        # self-describing about which behaviour produced it.
        # LOCAL-FIELD arm.  Moved to the Debug Options node with the other four; kept as an
        # override so a saved graph or an Options node still drives it.
        if "lf_ab" in kw:
            merged["lf_enable"] = _b(kw["lf_ab"])
        # THE ANOMALY DETECTOR'S ARM.  Moved to the Debug Options node with the other four and
        # DEFAULTING TO ON, on the operator's own A/B: *"i think my preference was on, there was
        # a prompt where it actually didn't fire, but for when it does, it does generally fix
        # things in a way regular sampling doesn't seem to be able to recover in time."*
        # It gates a path that can damp a small region for several steps, so testing it by eye
        # needs the same prompt and seed with the switch flipped -- which the Debug node still
        # provides.  The name matches the engine key exactly, so there is no mapping to get wrong.
        if "an_enable" in kw:
            merged["an_enable"] = _b(kw["an_enable"])
        # THE BOUND A/B IS GONE (2026-10-04): its widget and its mapping were removed in the same
        # change that made the corrected remainder scale the only path in `_pixel_order_bound`.  No
        # `if "order_bound_fix" in kw:` override is kept, because there is no longer a second
        # behaviour for it to select -- a leftover key would silently mean nothing.

        for o in _opts(kw.get("options+", kw.get("options"))):
            for k, v in o.items():
                merged[k] = v
        # Extract model info for auto-tuning (optional MODEL input).
        model = kw.get("model")
        if isinstance(model, (list, tuple)):
            model = model[0] if model else None
        if model is not None:
            mi = aflops_mod._extract_model_info(model)
            if mi is not None:
                merged["_model_info"] = mi
                if log:
                    import logging as _lg
                    _lg.info("[PD-AOS] model info: is_flow=%s sigma_data=%s "
                             "is_video=%s is_distilled=%s image_model=%s",
                             mi.get("is_flow"), mi.get("sigma_data"),
                             mi.get("is_video"), mi.get("is_distilled"),
                             mi.get("image_model"))
        if merged["eta"] is None and kw.get("stochasticity") is not None:
            # The `stochasticity` WIDGET IS NO LONGER EXPOSED in the release build.  It was
            # measured dead before hiding it: across 12 consecutive runs whose cfg.eta reads
            # 0.1 -- and one that reads 1.0 -- EVERY step reports eta_eff 0.0, so the slider
            # never changed the output.  The key is still honoured here for saved graphs and
            # for Options nodes that set it programmatically; with neither, eta keeps its
            # engine default.
            merged["eta"] = _f(kw["stochasticity"])
        return (comfy.samplers.KSAMPLER(aflops_engine,
                                        extra_options={"cfg": merged, "log_errors": log}),)


class ResearchAFlopsDebugOptions:
    DESCRIPTION = ("PD-AOS Debug Options: the FINISHED switches, moved off the Sampler. "
                   "Every one of these reached a decision -- by measurement plus the operator's "
                   "eye -- so none of them belongs on the node you use every day. Connect this "
                   "node's `options` output to the PD-AOS Sampler's `options+` input to drive "
                   "them again: for an A/B on the same prompt and seed, to re-test a verdict on "
                   "a new model, or to reproduce an old run. With this node disconnected the "
                   "Sampler uses the engine defaults, which ARE the decided values. "
                   "The defaults here are READ FROM THE ENGINE, not repeated, so the two cannot "
                   "drift apart.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        # DEFAULTS COME FROM THE ENGINE (rule: one value, one place).  A literal here would be a
        # second copy of a shipped default, and two copies of one value is how this project has
        # shipped a no-op before.  `lf_ab` is the UI name for the engine's `lf_enable`.
        return {
            "required": {
                "order_floor": ("INT", {"default": int(ENGINE_DEFAULTS.get("order_floor", 6)),
                    "min": 0, "max": 6, "step": 1,
                    "tooltip": "0 = AUTOMATIC, AND THE SHIPPED DEFAULT (it won 61 % of head-to-heads against the fixed demand across 26 judged prompts and was never the worst pick): demand the high order only while the run's OWN measured error is high, then let go once it settles -- it reads the step's error over its tolerance, and the threshold it used is stamped in the log. 1 = off, the sampler decides by itself. 2-6 = ASKS for at least that many previous steps per pixel -- a DEMAND the safety path may refuse, NOT a guarantee of that order. Above 1, every pixel is pushed up to at least this value, which makes the sampler follow the direction of the trajectory harder. Two safeguards still apply: an order is refused when the step would come out too large, and a pixel the sampler has just reset stays at 1. DECIDED 6, and measured: the demand is largely REFUSED by the safety path, which is why the order actually used tops out below it -- MEASURED on a 16-prompt batch: demanding 6 produced a mean order of 4.80 (against 2.36 when demanding 1) with a maximum of 5, and about 44% of pixels still below the demand. Read it as 'how hard the order system is asked to push', not as the order that will be used."}),
                "order_floor_adaptive_rtol": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_rtol", 0.05)),
                    "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "ONLY DOES ANYTHING WHEN `order_floor` IS 0 (AUTOMATIC), and it sets that mode's FLIP POINT: the automatic demand asks for the high order while the step's measured error over its tolerance is ABOVE this number, and asks for nothing once the error falls below it. Lower = the gate opens more often, so the run sits nearer order floor 6; higher = rarer, so it sits nearer floor 1. GROUNDED, NOT CHOSEN: on a 16-prompt paired batch (768 step records) that ratio has median 0.013, p75 0.0505, p90 0.294 and max 1.158, and it decays through a run -- so 0.0505 (the shipped default, the p75) demands the high order in roughly the worst quarter of the steps, acting early and releasing. Set 0.013 to open it about half the time, 0.294 for the worst tenth; the top of the range is effectively 'never opens', since the largest ratio ever measured is 1.158. THE SAMPLER'S OWN `rtol` ALSO MOVES THIS FLIP POINT, because the gate compares error OVER tolerance -- both numbers are stamped per step (`ord_floor_rtol`, `ord_floor_prev_ratio`), so the pair is always readable from the log. Measured live: at 0.05 the gate opened on 12-15 of 50 steps and put the result 26-71% of the way from order floor 1 to 6, depending on the prompt."}),
                # APPENDED LAST (rule 35): the bound-derived per-step request.
                "order_demand_q": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_demand_q", 0.0)),
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "ONLY ACTS WHEN `order_floor` IS 0 (AUTOMATIC). 0.0, the default, is OFF: the automatic request comes from the error gate, exactly as before. Above 0 the request for each step becomes the q-QUANTILE OF THE PER-PIXEL VALIDITY BOUND -- the highest order the engine's own remainder criterion considers valid at each pixel's own smoothness. PROVISIONAL AND MEASURED NOT TO DO WHAT IT CLAIMS: on the harness it produced a BINARY request {1, 6} rather than a graded one, and asked for 6 on steps whose own ceiling was 3-5, because the bound is permissive (median 6) and the engine's demand is a scalar floor that discards the per-pixel structure. 0.5 was intended as the median of the bound; 0.0 (OFF) is the only setting that is verified. 0.5 is the MEDIAN of the bound and is the arm worth trying first; 1.0 is its maximum, which is roughly what the fixed demand of 6 already asks for, so it is the least informative setting. WHY A QUANTILE AND NOT THE ERROR RATIO: measured over 544 step records, the error ratio correlates with the order actually granted at only +0.059 and is flat across its own thirds, so it can say WHETHER to ask but not HOW MUCH -- the bound is the engine's own answer to that question. WHY NOT SIMPLY ASK FOR MORE: the ladder's ceiling is 5 and a request above the ceiling is unreachable (0 of 61 steps ever reached a demand of 6), so the useful request lives INSIDE the reachable range, which is what a quantile of the bound gives."}),
                "order_floor_adaptive_graded": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_floor_adaptive_graded", True)),
                    "tooltip": "ONLY ACTS WHEN `order_floor` IS 0 (AUTOMATIC). ON, the shipped default: when the automatic gate opens, the order it ASKS FOR SCALES WITH THE ERROR instead of jumping straight to the maximum -- about 4 at the bottom of the gate's range and 5 near the top, so it asks for what the engine actually uses at that error level. OFF restores the previous behaviour exactly (a flat demand of 6 whenever the gate is open), which is what makes the two comparable in one test. GROUNDED, NOT CHOSEN: measured over the gated runs, the order the engine actually uses above the gate's trigger is p25 4.17 / p50 4.53 / p75 4.88, while at or below the trigger it is 1.81 / 2.41 / 2.85 -- so usage rises with the error, and a flat 6 asks for more order than the engine uses anywhere in that range."}),

                # ---- THE GRADED CURVE'S ENDPOINTS, APPENDED LAST (rule 35).  They were cfg-read constants with
                # no emitter, which meant neither the UI nor a batch graph could set them -- and a value that
                # cannot be varied from the UI cannot be A/B'd by the person with the eyes.
                "order_floor_adaptive_lo": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_lo", 4.0)),
                    "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "ONLY ACTS WHEN `order_floor` IS 0 (AUTOMATIC) AND `graded` IS ON. The order the automatic mode asks for at the BOTTOM of the gate's error range, i.e. just above the trigger -- the curve ramps from here up to `hi` as the run's measured error rises. The shipped 4.0 is DERIVED, not chosen: measured over the gated runs, the order the engine actually uses just above the trigger is p25 4.17 / p50 4.53 / p75 4.88. Setting `lo` equal to `hi` gives a flat demand at that level, i.e. the old behaviour; setting `lo` above `hi` is allowed and inverts the ramp, which is almost certainly wrong."}),
                "order_floor_adaptive_hi": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_hi", 5.0)),
                    "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "ONLY ACTS WHEN `order_floor` IS 0 (AUTOMATIC) AND `graded` IS ON. The order the automatic mode asks for at the TOP of the error range (at or above `r_hi`), i.e. how hard it pushes when the run is struggling most. The shipped 5.0 is DERIVED: it is the p75 of the order the engine actually uses above the trigger, and the reason the old flat demand of 6 was an over-ask is that the engine never uses more than about 5 there. RAISING THIS ABOVE 6 asks for more order than the previous behaviour did -- legitimate to try, but it is the direction that already looked like over-asking."}),
                "order_floor_adaptive_r_hi": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_r_hi", 0.3031)),
                    "min": 0.06, "max": 2.0, "step": 0.01,
                    "tooltip": "ONLY ACTS WHEN `order_floor` IS 0 (AUTOMATIC) AND `graded` IS ON. The run's measured error over tolerance at which the demand reaches `hi` and stops rising -- the WIDTH of the ramp. Smaller means the curve reaches full strength sooner (a more aggressive ramp just above the trigger); larger means it stays gentle for longer. Must stay ABOVE the trigger (`order_floor_adaptive_rtol`), otherwise the ramp is degenerate and the demand jumps straight to `hi`. The shipped 0.3031 is DERIVED: it is the p90 of the observed error ratio above the trigger."}),

                # ---- THE KREA2 TEST SET, APPENDED LAST (rule 35) so every existing workflow's
                # POSITIONAL widget values keep their slots and the new ones simply take their defaults.
                # Each is here because a measured finding named it; the why is in each tooltip and in
                # DEAD_FEATURES.md section KREA2 DEEP DIVE.
                "ord_ema": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("ord_ema", 0.37)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How fast the sampler forgets its own recent innovation estimate (0 follows the newest sample, 1 never moves). Shipped 0.37, DERIVED on the anima corpus as alpha* = 1 - rho_1. **RE-DERIVED ON KREA2 it lands at 0.03-0.17 across four logged fields** -- every field agrees the estimate should lean harder on the newest sample -- which is why this is the first arm to test. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "order_max": ("INT", {
                    "default": int(ENGINE_DEFAULTS.get("order_max", 3)),
                    "min": 0, "max": 8, "step": 1,
                    "tooltip": "The ceiling on how many previous steps the per-pixel order ladder may use. Shipped 3; 0 means 10 (the previous behaviour). 3 is a SAFE DEFAULT, not a measurement of your model: on a synthetic flow it kept fine detail in slow areas about four times better than 4, while 4 and 5 made a fast-changing area overshoot. HIGHER can sharpen detail and overshoot; LOWER is flatter and safer. A per-model value from the probe is not implemented yet."}),
                "order_elev_gain": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_elev_gain", 0.1)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How strongly the ladder is allowed to elevate a pixel's order. Raising it should show up as a higher achieved order on the steps the demand is already asking for. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "order_demote_gain": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_demote_gain", 0.25)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How strongly the ladder may push an order DOWN. This is the counterpart of the elevation gain and the other candidate explanation for the tail's drop to 4. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "guard_ratio": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_ratio", 3.0)),
                    "min": 0, "max": 12, "step": 0.01,
                    "tooltip": "How far outside its own neighbourhood a step must be before the guard acts. MEASURED: on 672 krea2 steps the guard fired on 0 % and `guard_sev` never left 1.0, i.e. it carried no signal at all -- so lowering this is how you find out whether it should be acting. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "guard_sensitivity": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_sensitivity", 2.5)),
                    "min": 0, "max": 10, "step": 0.01,
                    "tooltip": "The guard's sensitivity. Ships at 2.5; paired with guard_ratio, this decides whether anything is ever held back. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "an_value": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("an_value", 2.5)),
                    "min": 0, "max": 10, "step": 0.01,
                    "tooltip": "The anomaly detector's value threshold -- how abrupt a local change must be to count as an anomaly. MEASURED on krea2: it fired on 0.4 % of steps, so this is the knob that decides whether that mechanism exists at all for a given model family. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "an_from": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("an_from", 0.15)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "The noise level at which the anomaly detector starts looking. Ships at 0.15. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "wmax_adapt": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("wmax_adapt", False)),
                    "tooltip": "Turns the ADAPTIVE end-of-run error limit on. Ships OFF, and the krea2 corpus was therefore run entirely with the fixed limit -- an arm that has never been tested."}),
                "wmax_adapt_k": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("wmax_adapt_k", 0.5)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How fast that adaptive limit responds when wmax_adapt is on. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                "corrector_ab": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("corrector_ab", True)),
                    "tooltip": "The corrector's A/B arm. MEASURED on krea2: the corrector's weight is 0.175 on the worst quarter of steps against 0.768 on the best -- it backs off exactly where the error is worst, and whether that is the guard limiting it or a design choice is what this arm tests."}),
                "rtol": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("rtol", 0.01)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "The sampler's own error tolerance. `err/tol` climbs from 0.03 to 5.86 across a 16-step krea2 run because the tolerance shrinks as the run ends; how fast it may shrink is this number, which makes it the most direct lever on the unsettled tail. The range here is deliberately WIDE: the honest bound would be the clamp at the key's own read site, and the engine still applies that -- a test arm must not be silently clipped by a UI limit someone guessed."}),
                # ---- THE GUARD FAMILY AND THE TWO GATES, APPENDED LAST (rule 35).  The guard halves the
                # krea2 tail error when it fires; the two gates below are what make the order-gain widgets
                # and the whole per-pixel order machinery testable at all.  Why: DEAD_FEATURES.md.
                "guard_window": ("INT", {
                    "default": int(ENGINE_DEFAULTS.get("guard_window", 8)),
                    "min": 0, "max": 32, "step": 1,
                    "tooltip": "How many recent steps the guard looks back over to judge whether a step is an outlier. Ships at 8. The guard is the ONLY mechanism in this corpus that measurably improves a krea2 run: acting on ONE step of sixteen HALVES the raw tail error, with the firing threshold between 1.5 and 2.5 (guard_sensitivity). So the width of the window it judges from is a live question, and it was unreachable until now."}),
                "guard_quantile": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_quantile", 0.99)),
                    "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "The quantile of THIS STEP's per-pixel jump signal that counts as the guard's reading. Ships at 0.99 -- i.e. the guard compares against almost the most extreme pixel in the step, which is a conservative choice worth testing in the other direction. It is not a quantile over the recent steps; guard_window is the history."}),
                "guard_floor": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_floor", 0.1)),
                    "min": 0, "max": 4, "step": 0.001,
                    "tooltip": "The floor on the guard's scale, so a very quiet history cannot make it hypersensitive."}),
                "guard_hold": ("INT", {
                    "default": int(ENGINE_DEFAULTS.get("guard_hold", 2)),
                    "min": 0, "max": 8, "step": 1,
                    "tooltip": "How many steps a guard decision is held for before it may change again. Ships at 2; a hold of 1 lets it act every step, a larger hold makes it a slower, steadier response."}),
                "guard_mad_floor": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_mad_floor", 0.1)),
                    "min": 0, "max": 4, "step": 0.001,
                    "tooltip": "The floor on the guard's MAD-based scale estimate, the robust counterpart of guard_floor."}),
                "order_by_bound": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_by_bound", True)),
                    "tooltip": "THE GATE THAT MAKES TWO OTHER WIDGETS REAL. `order_elev_gain` and `order_demote_gain` sit in the `else:` branch of `if _order_by_bound:` (aflops.py 10376-10400), so while this is ON and the bound is available those two widgets CANNOT FIRE -- measured: both produced a bit-identical latent at outlandish values. Turning this OFF is the only way to test them."}),
                "order_pixel_adaptive": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_pixel_adaptive", True)),
                    "tooltip": "The gate on the whole per-pixel order machinery. It defaults ON, and it is the condition that guards several of the order knobs -- including `order_max`, which was measured BELOW REACH. Turning it OFF is how you find out what that machinery is actually worth on a given model family."}),
                # APPENDED LAST (rule 15).  It was frozen OFF and off the UI in the
                # release build (see the note above), which made the per-step fusion's
                # advertised 'take the lower order when the probes disagree' rule
                # unreachable: that rule is judged on caps this derivation produces,
                # so with the key fixed at False the fusion always used the shipped
                # cap and the disagreement branch could not run (audit C2).  Shipping
                # it as a widget, default OFF, keeps today's behaviour and makes the
                # rule switchable -- and testable.
                "wmax_bound": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("wmax_bound", False)),
                    "tooltip": "Which rule decides the sampler's extrapolation ORDER. OFF (the default) is what the sampler has always done: an order is allowed when its largest extrapolation weight is under the cap. ON adds the paper's remainder test -- the order's own truncation error at THIS step's spacing must also stay inside the step tolerance, using the per-order derivative scale the probe measured -- so an order the trajectory's own smoothness does not justify is refused even when its weights look small. It is also what the per-step probe fusion's 'when the two probes disagree, take the lower order' rule needs -- but that rule does not work on either setting: the two caps it compares come out identical, so it cannot lower an order. Treat that rule as NOT shipped. MEASURED, NOT NEUTRAL, AND THE DIRECTION IS MODEL-SPECIFIC: through the per-step fusion on krea2Turbo it refused enough orders to take the mean admitted order from 4-5 down to 1-2, while on mklanKrea2 it admitted order 5 at every step and the run's cap rose from 1.94-2.14 (order 3) to 3.28125 (order 5). OFF is the shipped value; ON is for testing which way your model goes. Toggling it starts a fresh probe on the next run, because the value is part of the probe's cache key."}),
                # APPENDED LAST (rule 15).  The dial the operator asked for after seeing a
                # real ON/OFF difference on a krea checkpoint and being unable to call it.
                "order_bound_blend": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_bound_blend", 1.0)),
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only does anything while `order_by_bound` is ON. It scales how much of each pixel's order comes from the remainder criterion instead of the 'which order predicts x0 best' rule: 0.0 leaves the order exactly as the OFF arm has it, 1.0 is the criterion's own answer (the shipped behaviour), and 0.5 means a pixel whose two answers are order 1 and order 4 runs at order 2. WHY IT EXISTS: the two endpoints can look clearly different without either being obviously better, and a verdict imposed from a metric is not a verdict. Sweep it -- 0.25, 0.5, 0.75, 1.0 at the same seed -- and keep the value that looks best to you; the value in force is written to the run log as `order_bound_blend`, and the order actually used as `ord_mean` per step."}),
                # APPENDED LAST (rule 15): the bound's own map as the request.
                "order_demand_bound": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_demand_bound", True)),
                    "tooltip": "ON, the shipped default: on any step where the per-pixel validity bound was computed, the request is 'raise nothing' -- the bound's own map already says which order each pixel's smoothness justifies, so the error gate's ramp cannot push an order past it. OFF restores the ramp everywhere. It only acts when `order_floor` is 0 (AUTOMATIC) and `order_by_bound` is ON, and an explicit `order_demand_q` above 0 still selects the quantile request. A SAVED node keeps its own value, so this default governs new nodes; the log stamps the policy each step in `ord_demand_policy` ('bound_map' / 'bound' / 'graded') and the level in force in `ord_demand`."}),
                # APPENDED LAST (rule 15): the per-pixel history-consistency gate A/B.
                "hist_cos_per_pixel": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("hist_cos_per_pixel", True)),
                    "tooltip": "ON (shipped): the history-consistency cosine is computed per pixel; only the locally reversing pixels go to order 1, so a volatile corner no longer removes the per-pixel order ladder from a detailed area. OFF: the old whole-image gate -- one negative cosine switches the ladder off for EVERY pixel; on the saved runs that happened on 88.8% of the steps that were about to use order 2 or higher. ON raises the admitted order on those steps and costs some trajectory error; the operator's paired tests preferred it on anima, krea and moss. Only acts while the per-pixel order ladder is on (`order_pixel_adaptive`); with the ladder off, the whole-image gate still applies. The per-step log records the gated fraction as `hist_cos_px_neg`; the run config records `hist_cos_per_pixel`."}),
                # APPENDED LAST (rule 15): the per-pixel gate on the second-order term A/B.
                "second_order_per_pixel": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("second_order_per_pixel", True)),
                    "tooltip": "ON (shipped): the exponential integrator's second-order slope term is applied only to pixels whose admitted per-pixel order is 2 or more; the volatile pixels the order map keeps at order 1 stay on the first-order step, which is the stable one. OFF: the old global term -- applied to EVERY pixel whenever the step's order cap is 2 or more. The operator's paired tests preferred ON. The per-step log records the fraction of pixels that received the term as `second_order_px`; the run config records `second_order_per_pixel`."}),
                # APPENDED LAST (rule 15): the raw-lambda slope A/B.
                "second_order_raw_lambda": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("second_order_raw_lambda", True)),
                    "tooltip": "ON (shipped): the second-order slope term and its phi2 coefficient use the UNCLAMPED lambda, while the first-order decay stays clamped. OFF: the whole slope term uses the clamped lambda. The clamp is active on every step, so the clamped slope carries a residual that does not shrink with the step count; ON removes it. The paired runs measured the two arms as identical where the clamp does not bind and slightly lower step error where it does; OFF restores the clamped form. The per-step log records `second_order_raw`; the run config records `second_order_raw_lambda`."}),
            },
            # CHAINING (2026-10-04).  Until this input existed only the Sampler could RECEIVE
            # options, so two options-producing nodes could not both drive a run -- the operator's
            # own report: *"i check the debug node, but it doesn't have options+ input -> i can't
            # have multiple chained options (scheduler+debugoptions) this way."*  It is OPTIONAL:
            # with nothing connected this node behaves exactly as before.
            "optional": {
                "options+": (OPT_TYPE, {"tooltip": "Connect another PD-AOS Options node here (the Autotuner's `options` output, or another Debug Options node) and this node passes its settings through, adding or overriding its own on top. Chain them in the order you want them applied: the node CLOSEST to the Sampler wins on any field two of them both set."}),
            },
        }

    RETURN_TYPES = (OPT_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "options"

    def options(self, order_floor=int(ENGINE_DEFAULTS.get("order_floor", 0)),
                order_floor_adaptive_rtol=float(
                    ENGINE_DEFAULTS.get("order_floor_adaptive_rtol", 0.05)),
                order_demand_q=float(ENGINE_DEFAULTS.get("order_demand_q", 0.0)),
                order_floor_adaptive_graded=bool(
                    ENGINE_DEFAULTS.get("order_floor_adaptive_graded", True)),
                order_floor_adaptive_lo=float(ENGINE_DEFAULTS.get("order_floor_adaptive_lo", 4.0)),
                order_floor_adaptive_hi=float(ENGINE_DEFAULTS.get("order_floor_adaptive_hi", 5.0)),
                order_floor_adaptive_r_hi=float(
                    ENGINE_DEFAULTS.get("order_floor_adaptive_r_hi", 0.3031)),                ord_ema=ENGINE_DEFAULTS.get("ord_ema", 0.37),
                order_max=ENGINE_DEFAULTS.get("order_max", 3),
                order_elev_gain=ENGINE_DEFAULTS.get("order_elev_gain", 0.1),
                order_demote_gain=ENGINE_DEFAULTS.get("order_demote_gain", 0.25),
                guard_ratio=ENGINE_DEFAULTS.get("guard_ratio", 3.0),
                guard_sensitivity=ENGINE_DEFAULTS.get("guard_sensitivity", 2.5),
                an_value=ENGINE_DEFAULTS.get("an_value", 2.5),
                an_from=ENGINE_DEFAULTS.get("an_from", 0.15),
                wmax_adapt=ENGINE_DEFAULTS.get("wmax_adapt", False),
                wmax_adapt_k=ENGINE_DEFAULTS.get("wmax_adapt_k", 0.5),
                corrector_ab=ENGINE_DEFAULTS.get("corrector_ab", True),
                rtol=ENGINE_DEFAULTS.get("rtol", 0.01),
                guard_window=ENGINE_DEFAULTS.get("guard_window", 8),
                guard_quantile=ENGINE_DEFAULTS.get("guard_quantile", 0.99),
                guard_floor=ENGINE_DEFAULTS.get("guard_floor", 0.1),
                guard_hold=ENGINE_DEFAULTS.get("guard_hold", 2),
                guard_mad_floor=ENGINE_DEFAULTS.get("guard_mad_floor", 0.1),
                order_by_bound=ENGINE_DEFAULTS.get("order_by_bound", True),
                order_pixel_adaptive=ENGINE_DEFAULTS.get("order_pixel_adaptive", True),
                wmax_bound=ENGINE_DEFAULTS.get("wmax_bound", False),
                order_bound_blend=float(
                    ENGINE_DEFAULTS.get("order_bound_blend", 1.0)),
                order_demand_bound=bool(
                    ENGINE_DEFAULTS.get("order_demand_bound", True)),
                hist_cos_per_pixel=bool(
                    ENGINE_DEFAULTS.get("hist_cos_per_pixel", True)),
                second_order_per_pixel=bool(
                    ENGINE_DEFAULTS.get("second_order_per_pixel", True)),
                second_order_raw_lambda=bool(
                    ENGINE_DEFAULTS.get("second_order_raw_lambda", True)),
 **kw):
        # The Sampler merges this dict into its own cfg BEFORE the Autotuner's options, so a
        # Debug node wins over a probe-side default and can be flipped without touching either
        # node's other settings.  `lf_ab` -> `lf_enable` is the ONLY rename here.
        out = {}
        # UPSTREAM FIRST, THEN OURS (see the `options+` tooltip): a chained node passes its fields
        # through untouched, and this node's own deliberate settings win where the two overlap --
        # which today is nowhere, because no other options node sets any of these six keys.
        for o in _opts(kw.get("options+", kw.get("options"))):
            if isinstance(o, dict):
                out.update(o)
        # FOUR SETTLED SWITCHES ARE NOT PARAMETERS HERE -- `lf_enable` (the `lf_ab`
        # widget), `an_enable`, `wmax_eg_dir` and `cond_rtol_loosen`.  Their values come
        # from ENGINE_DEFAULTS, and the engine still READS all four.  `order_by_bound`
        # and `ord_ema` are WIDGETS on this node and ARE emitted below; an earlier
        # version of this comment wrongly listed them as removed (2026-10-17).
        out.update({
            "order_floor": int(_f(order_floor)),
            "order_floor_adaptive_rtol": max(0.0, min(1.0, _f(order_floor_adaptive_rtol))),
            # READ BY THE ENGINE where the automatic demand is computed.  The widget, the parameter and this
            # emission are the three ends that must ALL exist -- see the `ord_ema` note above.
            "order_demand_q": max(0.0, min(1.0, _f(order_demand_q))),
            # READ BY THE ENGINE at the graded-demand branch.  Widget, parameter and emission are the three ends
            # that must ALL exist (see the `ord_ema` note above).
            "order_floor_adaptive_graded": _b(order_floor_adaptive_graded),
            # READ BY THE ENGINE inside `_graded_demand`'s call sites.  Widget, parameter and emission are the
            # three ends that must ALL exist (see the `ord_ema` note) -- three more keys, three more chances to
            # ship half a wire, which is why the rig censuses every widget in both directions automatically.
            "order_floor_adaptive_lo": max(1.0, min(10.0, _f(order_floor_adaptive_lo))),
            "order_floor_adaptive_hi": max(1.0, min(10.0, _f(order_floor_adaptive_hi))),
            "ord_ema": (_f(ord_ema)),
            "order_max": (int(_f(order_max))),
            "order_elev_gain": (_f(order_elev_gain)),
            "order_demote_gain": (_f(order_demote_gain)),
            "guard_ratio": (_f(guard_ratio)),
            "guard_sensitivity": (_f(guard_sensitivity)),
            "an_value": (_f(an_value)),
            "an_from": (_f(an_from)),
            "wmax_adapt": (_b(wmax_adapt)),
            "wmax_adapt_k": (_f(wmax_adapt_k)),
            "corrector_ab": (_b(corrector_ab)),
            "rtol": (_f(rtol)),
            "guard_window": (int(_f(guard_window))),
            "guard_quantile": (_f(guard_quantile)),
            "guard_floor": (_f(guard_floor)),
            "guard_hold": (int(_f(guard_hold))),
            "guard_mad_floor": (_f(guard_mad_floor)),
            "order_by_bound": (_b(order_by_bound)),
            "order_pixel_adaptive": (_b(order_pixel_adaptive)),
            # READ BY THE ENGINE at `_order_gate` and re-derived per run through
            # `_calib_inputs` / `calib_flags` (it also changes the fusion's choice of
            # derivation).  Widget, parameter and emission are the three ends that
            # must ALL exist -- see the `ord_ema` note above; the connectivity rig
            # (`rig_audit_pickup.py`) censuses this key in both directions.
            "wmax_bound": _b(wmax_bound),
            # READ BY THE ENGINE at `_order_bound_target`.  A run-level dial, so the
            # value in force is stamped per run rather than inferred from the order map.
            "order_bound_blend": min(1.0, max(0.0, _f(order_bound_blend))),
            # READ BY THE ENGINE in `_demand_policy`, at the automatic demand branch.  Widget, parameter
            # and emission are the three ends that must ALL exist (see the `ord_ema` note above).
            "order_demand_bound": _b(order_demand_bound),
            "hist_cos_per_pixel": _b(hist_cos_per_pixel),
            "second_order_per_pixel": _b(second_order_per_pixel),
            "second_order_raw_lambda": _b(second_order_raw_lambda),
            "order_floor_adaptive_r_hi": max(0.06, min(2.0, _f(order_floor_adaptive_r_hi))),
        })
        return (out,)


class ResearchAFlopsNoiseOptions:
    DESCRIPTION = (
        "PD-AOS Noise (ETA) Options - every stochasticity control in ONE node. "
        "The Sampler carries only what you touch every day and the Debug Options "
        "node carries only switches that REACHED A DECISION; ancestral noise has "
        "not reached one, so it lives here instead of diluting either. NOTHING "
        "HERE IS A FINISHED SETTING. "
        "Wiring: chain it between the Autotuner and the Sampler "
        "(Autotuner 'options' -> this node's 'options+' -> Sampler 'options+'). "
        "Order matters only where two nodes set the same field, and the node "
        "CLOSEST to the Sampler wins. "
        "Its defaults are chosen so that CONNECTING IT DOES SOMETHING: the noise "
        "window is opened and, when eta is above zero, the probe path the escape "
        "needs is enabled. Connected to a graph with no Autotuner, it behaves "
        "like one running the Autotuner's own defaults rather than like a "
        "standalone run -- in a standalone run the engine zeroes all three escape "
        "thresholds and every widget here would silently do nothing.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "HOW MUCH ancestral noise, on the same scale as every other sampler's eta: 0 is the fully deterministic sampler, 0.1 is the value you found helpful elsewhere. This is the ceiling on what the engine may inject, and it is the SAME key the old 'stochasticity' slider fed, so a saved graph or an Options node that still sets that name sets this one. What it means depends on 'eta_mode': with 'adaptive' the noise actually injected is eta x the detector's stuck signal, so it fires only where the trajectory has stalled and never reaches eta; with the other modes it is the level across the whole run."}),
                "eta_mode": (["adaptive", "constant", "cosine", "linear",
                              "exponential", "sigma_power"], {
                    "default": "adaptive",
                    "tooltip": "WHERE the noise goes, and the single most useful switch here. 'adaptive' injects ONLY on the steps where the engine's local-minimum detector fires (the trajectory's movement has collapsed and the image is still flat, or the update has reversed direction) -- that is the escape, and it needs the probe path to be live. 'constant' injects the full eta on every step, 'cosine' and 'linear' fade eta to zero across the run, 'exponential' and 'sigma_power' fade it on the sigma schedule instead. 'constant' is the arm that was reported as washing out structure; every fading mode keeps the noise out of the endgame. The level actually used each step is stamped in the run log as 'eta_eff', and the detector's own reading as 'eta_stuck'."}),
                "s_tmin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                    "tooltip": "The noise is applied only where sigma is ABOVE this, and the default here is deliberately 0.0 rather than the engine's own 1.0. This single number is what silently disabled every previous attempt: at 1.0 only the very first step qualifies, and that step has no trajectory history, so the detector cannot fire on it -- the only step allowed to make noise was the only step where the detector is blind. Connecting this node therefore OPENS the window. Raise it to keep noise out of the high-sigma steps, where the model is still deciding composition."}),
                "s_tmax": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                    "tooltip": "The upper end of that window: 0.0 means NO upper bound, which is the engine's own convention. Set it above 0 to stop making noise below that sigma -- the noise needed early is often harmful in the endgame, and this is the blunt version of what the fade already does smoothly."}),
                "escape": (["auto", "off"], {
                    "default": "auto",
                    "tooltip": "Whether the local-minimum detector is allowed to fire, and it is a SWITCH rather than two threshold numbers on purpose. 'auto' lets the engine DERIVE the two firing thresholds from the probe's own measurement of your model -- how far a pixel normally moves per step, and how much high-frequency detail the image normally holds -- so the base is measured on the model in front of you rather than copied from another checkpoint. 'off' sets both to zero, which disables the escape without touching the noise level. The three numbers this resolves to are stamped in every run log ('eta_escape_jump', 'eta_escape_detail', 'eta_fade_sigma'), so 'auto' is never a black box. NOTE: if an upstream Options node sets either threshold explicitly, that value is used and this switch does not override it."}),
                "escape_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Sensitivity of the escape, as a multiplier on the two thresholds 'auto' derives: 1.0 leaves the probe's measurement exactly as measured, above 1 makes the detector fire more readily (more steps get noise), below 1 makes it more reluctant, and 0 means it never fires. This exists so sensitivity is ONE number instead of two hand-set thresholds -- a threshold typed in for one model is a value from that model's population, and it would travel badly to the next checkpoint."}),
                "fade_floor": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much of the BASE floor (`eta_min`) survives the late fade. The shipped `eta_min` is 0.0 and has no widget, so this slider does nothing on a graph built from these nodes; it only matters if a custom Options node sets `eta_min` above zero. It does NOT fade the escape: a stuck trajectory still gets its nudge at the end."}),
                "fade_sigma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.001,
                    "tooltip": "Where the late fade begins, in sigma. The default -1 is 'auto': the engine uses the sigma where the probe measured your model's trajectory first reversing direction (or where it converged, if it never reverses), which is a measurement rather than a guess. Set a positive number to anchor the fade yourself."}),
                "noise_survival_probe": (["follow eta", "force on", "force off"], {
                    "default": "follow eta",
                    "tooltip": "Whether the probe measures how much of an injected perturbation still shows in the model's prediction a step later -- the measurement that says how big a kick THIS model can still correct at a given sigma, i.e. the budget any escape has to spend. 'follow eta' (the default) measures it whenever eta is above zero, which is the case that needs it; 'force off' skips it and saves up to ten model calls on the first run of a session, at the cost of not knowing the budget; 'force on' measures it even at eta 0, for inspecting the curve without generating noise."}),
            },
            # CHAINING, same convention as the Debug Options node: a node closest
            # to the Sampler wins where two of them set the same field.
            "optional": {
                "options+": (OPT_TYPE, {"tooltip": "Connect another PD-AOS Options node here -- the Autotuner's 'options' output, or the Debug Options node -- and its settings pass through, with this node's own overriding on any field both set. Anything connected here also supplies the probe settings, which this node only fills in when nothing else did."}),
            },
        }

    RETURN_TYPES = (OPT_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "options"

    def options(self, eta=0.0, eta_mode="adaptive", s_tmin=0.0, s_tmax=0.0,
                escape="auto", escape_gain=1.0, fade_floor=0.25, fade_sigma=-1.0,
                noise_survival_probe="follow eta", **kw):
        out = {}
        # UPSTREAM FIRST, THEN OURS -- the Debug Options convention, so a chained
        # node's fields pass through untouched and this node wins only where both
        # speak.  `options+` cannot be a named parameter (not a valid identifier),
        # which is why it arrives through **kw.
        for o in _opts(kw.get("options+", kw.get("options"))):
            if isinstance(o, dict):
                out.update(o)
        own = {
            "eta": max(0.0, _f(eta)),
            "eta_mode": str(eta_mode),
            "s_tmin": max(0.0, _f(s_tmin)),
            "s_tmax": max(0.0, _f(s_tmax)),
            # 'auto' is the engine's negative sentinel, i.e. "derive from the
            # probe"; 'off' is the explicit zero that disables the detector.  The
            # node never emits a hand-set threshold -- that is what `escape_gain`
            # is for.
            "eta_escape_jump": (-1.0 if str(escape) != "off" else 0.0),
            "eta_escape_detail": (-1.0 if str(escape) != "off" else 0.0),
            "escape_gain": max(0.0, _f(escape_gain)),
            "eta_fade_floor": max(0.0, min(1.0, _f(fade_floor))),
            "eta_fade_sigma": _f(fade_sigma),
        }
        # The probe-measurement switch is THREE-valued: absent means "follow eta"
        # and the engine tests `is not None`, so a string would be read as truthy
        # and 'force off' would silently mean 'force on'.
        _ov = {"force on": True, "force off": False}.get(str(noise_survival_probe))
        if _ov is not None:
            own["probe_noise_survival"] = _ov
        out.update(own)
        # ---- MAKE CONNECTING THIS NODE DO SOMETHING, WHICHEVER WAY IT IS WIRED --
        # The engine treats "no probe" as STANDALONE and zeroes all three escape
        # thresholds (`aflops.py:9762-9767`).  On a graph whose Autotuner
        # `options` output is not wired in -- which is the current workflow's
        # state -- every widget above would then do nothing at all, silently.
        # These are filled in ONLY WHERE NOTHING UPSTREAM SET THEM, so a chained
        # Autotuner's own probe settings are used untouched and this node is a
        # fallback rather than a second owner.  They are the Autotuner's own
        # defaults (`nodes.py:612-616`, and its `opts` dict), so a run with no
        # Autotuner behaves like one running a default Autotuner.
        #
        # ONLY WHEN eta IS ABOVE ZERO: with no noise to place there is no escape
        # to arm, and the engine probe would be latency bought for nothing.
        if own["eta"] > 0.0:
            for _k, _v in _NOISE_PROBE_FALLBACK.items():
                out.setdefault(_k, _v)
        return (out,)


class ResearchAFlopsExperimental:
    DESCRIPTION = (
        "PD-AOS Experimental Options - a deliberately VOLATILE node. "
        "Its whole purpose is to let a test arm be added or REMOVED without the "
        "widget-ordering risk that governs every other node here: ComfyUI stores "
        "a UI graph's widget values POSITIONALLY, so dropping a widget from the "
        "middle of a stable node re-maps every later value onto the wrong widget, "
        "silently. THIS node is exempt because it is only ever used for testing, "
        "and nothing saved depends on it. "
        "EVERY SWITCH HERE DEFAULTS TO THE OLD BEHAVIOUR, so connecting it "
        "changes nothing until you set one, and disconnecting it returns the "
        "graph to the shipped path. "
        "Wiring: chain it anywhere in the options chain "
        "(Autotuner 'options' -> this node's 'options+' -> Sampler 'options+'). "
        "Order matters only where two nodes set the same field, and the node "
        "CLOSEST to the Sampler wins.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "escape_mode": (["flatness", "momentum"], {
                    "default": "flatness",
                    "tooltip": "WHICH CONDITIONS MAY ARM THE ESCAPE'S STALL DETECTOR. 'flatness' is what the sampler has always done, and is left exactly as it was. 'momentum' ADDS a second way to arm it -- it does not replace the first, and it does not touch the reversal detector, so an A/B changes ONE thing. WHY THE SECOND WAY EXISTS, measured on the first build that shipped the Noise (ETA) Options node (7 runs, 5 checkpoints): the sampler's stall test multiplies 'the movement has collapsed' by 'the image is still flat', and those two quantities move in OPPOSITE directions as a run converges -- the movement falls while the high-frequency detail RISES as the picture forms. Their product read exactly 0.0000 on every one of those runs AT THE SHIPPED SENSITIVITY (gain 1.0); raising the gain does reach it -- gains of 1.5 to 4.0 armed 1 to 39 steps across the ten runs measured -- but a gain high enough to arm it also noises steps that did not need it, and centring both thresholds on each run's own measured medians still armed 1 step out of 252. 'momentum' arms the same half of the detector from a run-relative test instead: the per-step movement has fallen to 70% or less of its own median over the last 4 steps AND the detail has not grown by more than 15% over that window. On that same corpus it arms 2-5 steps per run, in the middle of the run rather than the endgame. IT IS A TEST ARM, NOT AN IMPROVEMENT: nothing here can say whether the extra noise looks better, which is exactly what this node is for. Both arms are readable from the run log -- `escape_mode` in the config, and per step `eta_momentum`, `eta_momentum_ratio` and `eta_growth_rate`, which are stamped on BOTH arms so the two can be compared."}),
                "noise_survival_curve": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "MEASUREMENT AND INPUT -- on its own it changes nothing the sampler does, but `eta_from_probe` / `eta_budget_from_probe` READ this curve, and turning either of those on records it automatically. Off is the delivered behaviour. On makes the model probe RECORD the noise-absorption measurement instead of only reducing it to one number: at four sigmas spread across your run's range it injects a perturbation at up to four amplitudes and writes down how far that perturbation still shows in the model's next prediction, as a fraction of an unperturbed reference. WHAT THE NUMBERS MEAN: a small value means the model corrected the kick, i.e. this model at this sigma will not let noise survive; a value near 1 means the kick persisted. That curve is the empirical form of 'how much noise can this model absorb here', which is the quantity a principled noise schedule would be derived from rather than hand-tuned. WHY IT IS NOT ALREADY THERE: the measurement has always been computed and then immediately compared against a fixed 0.35 to produce a single number, so the only surviving record was that number -- and across 10,602 logged profiles it is missing in 69% of them and sits at the clamp or the skipped-probe fallback in most of the rest, so it has rarely carried a usable per-model reading. THE LADDER WAS ALSO TOO NARROW TO SEE THE SHAPE: three separate measurement rounds never sampled above sigma 0.66, and the amplitude ladder stopped at 0.35, which truncated exactly the region where the model stops correcting. It now spans your run's own sigma range and records every rung, including the one that failed. COST: one model call per sigma/amplitude pair plus one reference call per sigma -- around 20 calls on the first run for a model, on the PROBE path only, then cached. It does NOT add per-step model calls, so generation speed after the first run is unchanged. Read it in the run log as 'noise_survival_curve'; each entry carries the sigma, the amplitude, the divergence, the two norms the ratio was built from, and whether it passed the threshold."}),
                "eta_from_probe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Derive the noise level from the probe's own measurement of your model, instead of inferring it from whether the image looks stuck. OFF is the delivered behaviour. HOW IT WORKS: the probe already injects a perturbation one step ahead and measures how much of it still shows in the model's next prediction. That response is LINEAR in the size of the kick -- measured on five checkpoints, the ratio is constant to within 0.2-4% across the amplitudes tested -- so one number per sigma describes your model: how strongly it lets a kick through. Asking for the level that holds the surviving kick under a residual budget gives eta(sigma) = (budget / that number)^2, clamped to 1. WHY THIS REPLACES THE ESCAPE RATHER THAN ADDING TO IT: it answers 'how much noise can this model absorb HERE', which is the question the escape's detector was approximating from a different quantity. It is not a policy and not a per-model table -- it is computed from the measurement taken on the model in front of you, so it travels to the next checkpoint by construction. MEASURED WHY IT MATTERS: on the same checkpoint the response number is about 1.35 at sigma 0.66 and 0.31 at sigma 0.28, so a 0.10 budget asks for eta 0.0055 early in a run and 0.072 late -- against a fixed eta of 1.0. The model tolerates far LESS noise early than the escape policy assumed. It applies to every eta_mode and REPLACES the level that mode would use; your 'eta' stays the CEILING, so a lower setting still gives a lower level. If the probe has no curve for this model the sampler falls back to the shipped behaviour silently. Both numbers are in the run log: 'eta_eff' (what was used) and 'eta_probe' (what was derived), plus '_eta_probe_gains' and '_eta_probe_sigmas' in the config."}),
                "eta_budget_from_probe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether the residual budget used by 'eta_from_probe' comes from the PROBE rather than from the 'eta_probe_budget' widget. OFF is the delivered behaviour and the widget governs. ON uses the number the probe ALREADY measures on every probed run -- 'safe_fresh_frac', how big a kick this model still corrects, at the probe's own fixed 0.35 threshold. That is the same quantity and the same units as the budget widget, so switching this on replaces an invented default with a measurement. WHY IT EXISTS: the budget was the last hand-set number in an otherwise derived schedule, and it did not need to be -- the engine has been measuring exactly this quantity all along, for a different consumer. WHAT IT DOES NOT DO: it does not find a 'golden' setting. Measured across 20 curves, the response stays on its low-kick line all the way to the top of the amplitude ladder on 21 of 27 sigma samples, so there is no reachable knee to derive an optimum from; and the quantity itself says what the model CORRECTS, not what a viewer will accept. So this makes the number MEASURED rather than THE BEST -- if the images look under-noised, the widget is still there and you can set it yourself. If the probe has no usable value, this node silently falls back to 'eta_probe_budget'. Which source was used is stamped in the run config as '_eta_budget_src' ('widget' or 'probe'), and the measured value as '_eta_budget_measured'."}),
                "eta_probe_budget": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Only used when 'eta_from_probe' is ON: how much of an injected perturbation you are willing to let survive into the next prediction, as a fraction of an unperturbed reference. LOWER means less noise and a tighter schedule; HIGHER means more. The probe's own pass/fail threshold has always been 0.35, so the 0.10 default sits well inside what that experiment already treats as corrected. It is the one number that shapes the derived schedule, and it is deliberately a SAFE setting rather than an optimal one -- you cannot verify optimality on your own model, so the default is conservative and the sweep is yours to run."}),
                "escape_momentum_flatness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether the momentum arm must ALSO see a flat image before it fires -- the test the original flatness detector always applied and this newer arm was missing. Leave it ON unless you specifically want the old ungated behaviour. WHY IT EXISTS, measured on a matched pair: a 16-bit pixel-art prompt run at eta 0.0 and eta 1.0 with the same seed, scheduler and step count produced a crisper image with a cleaner, more uniform pixel grid at eta 0.0, while eta 1.0 produced 'half/half pixels' -- cells the sampler could not resolve and mushed together. The cause is visible in the run log: every step the momentum arm fired had the image's fine detail at 1.4x to 3.3x the flatness threshold, i.e. the picture was FULL of structure the model was deliberately placing on a pixel grid. The arm fired anyway because it only tested whether the detail had stopped GROWING, and on pixel art a stopped-growing grid means the grid LOCKED, not that the trajectory gave up. With this ON, all three of those firings are suppressed and the run matches the deterministic arm that looked better; on the runs this arm was built for -- a genuine stall on a still-flat picture -- the check is inert. The per-step reading it tests is in every run log as 'eta_detail_abs', against the threshold stamped as 'eta_escape_detail'."}),
                "escape_growth_probe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether the momentum arm's growth budget is DERIVED FROM THE PROBE rather than fixed. OFF is the shipped behaviour. ON makes the budget depend on how consistently the probe measured YOUR model's trajectory moving in one direction: a model whose trajectory keeps reversing (which is this sampler's own definition of being stuck -- every pixel moves, just back and forth) gets the budget opened up so the escape can fire, while a model that commits cleanly keeps the shipped budget and behaves exactly as it does today. WHY IT EXISTS, measured over 31 runs: the quantity that separates the runs where the escape fired from the runs where it did not is the probe's own direction-consistency curve, across its FULL range -- firing runs peak at -0.32 or below, silent runs at +0.60 or above. A fixed budget cannot express that, because the same model sampled in 16 steps builds more detail per step than in 50, so one number means different things at different step counts. Measured effect of switching it on: the 50-step checkpoint is untouched (51 armed steps either way) while the 16-step turbo checkpoints go from 0 to 18 and from 0 to 8. NOTE: this only does anything when 'escape_mode' is 'momentum'. The budget actually used is written to every run log as 'eta_growth_budget', and the measurement behind it as 'eta_escape_commit'."}),
            },
            # CHAINING, same convention as the other Options nodes: a node closest
            # to the Sampler wins where two of them set the same field.
            "optional": {
                "options+": (OPT_TYPE, {"tooltip": "Connect another PD-AOS Options node here -- the Autotuner's 'options' output, the Noise (ETA) Options node, or the Debug Options node -- and its settings pass through, with this node's own overriding on any field both set."}),
            },
        }

    RETURN_TYPES = (OPT_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "options"

    def options(self, escape_mode="flatness", escape_growth_probe=False,
                escape_momentum_flatness=True, noise_survival_curve=False,
                eta_from_probe=False, eta_probe_budget=0.10,
                eta_budget_from_probe=False, **kw):
        # UPSTREAM FIRST, THEN OURS -- the convention every Options node here
        # follows, so a chained node's fields pass through untouched and this node
        # wins only where both speak.  `options+` cannot be a named parameter (it
        # is not a valid identifier), which is why it arrives through **kw.
        out = {}
        for o in _opts(kw.get("options+", kw.get("options"))):
            if isinstance(o, dict):
                out.update(o)
        # ONLY the keys this node owns.  It deliberately does NOT fill in probe
        # defaults the way the Noise (ETA) Options node does: a test arm must
        # change one thing, and silently enabling a probe path here would confound
        # the comparison it exists to make.
        out["escape_mode"] = str(escape_mode)
        out["escape_growth_probe"] = bool(_b(escape_growth_probe))
        out["escape_momentum_flatness"] = bool(_b(escape_momentum_flatness))
        # MEASUREMENT, NOT A DECISION.  It changes what the probe RECORDS, not
        # what it measures, so it is deliberately NOT in the probe key -- same
        # reasoning as `escape_gain`.  `eta_from_probe` reads the recorded curve.
        out["probe_noise_survival_curve"] = bool(_b(noise_survival_curve))
        # THE DERIVED SCHEDULE.  `eta_from_probe` replaces the escape for the run
        # it is set on; the budget is the one number that shapes it.
        out["eta_from_probe"] = bool(_b(eta_from_probe))
        out["eta_probe_budget"] = max(0.01, min(1.0, _f(eta_probe_budget)))
        out["eta_budget_from_probe"] = bool(_b(eta_budget_from_probe))
        return (out,)


class ResearchPaperAEuler:
    DESCRIPTION = ("A-FloPS A-Euler - the paper's Algorithm 2 (Jin et al., AAAI 2026, "
                   "arXiv:2509.00036) applied DIRECTLY to a native flow model, with no "
                   "extras. This is the faithful reference: it factorizes the FM velocity "
                   "v = lambda*x + h with the adaptive lambda = <dv,dx>/||dx||^2 (Eq.12, "
                   "UNCLAMPED -- neither Algorithm 2 nor the authors' reference clamps it, "
                   "and this node passes no clamp), solves the linear drift exactly, and "
                   "extrapolates the residual from the previous step (exponential "
                   "integrator, first step Euler). One model call per step. The method is "
                   "grid-agnostic: on a native flow model flow time is t = 1 - sigma, so "
                   "each step's span IS the sigma difference and it runs on WHATEVER sigma "
                   "list you feed it; connect it to a SamplerCustomAdvanced and drive it "
                   "with any scheduler. For the paper's exact uniform-flow-time placement, "
                   "use the companion 'A-FloPS A-Euler Schedule (paper)' node as the sigmas "
                   "source.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(paper_aeuler_mod.sample_paper_aeuler),)


class ResearchPaperAEulerSchedule:
    DESCRIPTION = ("The paper's own schedule for A-FloPS A-Euler: sigmas placed at "
                   "UNIFORM flow time t = n/N (noise -> clean), mapped through the model's "
                   "own sigma<->time map: flow time is scaled to the NATIVE TIMESTEP by the "
                   "model's own `multiplier` for the SD3/3.5, Wan and LTXV classes (and "
                   "ModelSamplingAV), `flux_time_shift` for Flux, and `t/(1-t)` for "
                   "Cosmos/RFlow. "
                   "Connect this node's SIGMAS output to SamplerCustomAdvanced (or any "
                   "sampler's sigmas input). This is the schedule Algorithm 2 assumes; the "
                   "sampler itself also accepts any other scheduler's sigmas. Note: on "
                   "cosmos-style sigma=t/(1-t) schedules the noise end is compressed, so "
                   "the first step is very large - you may need more steps than 'simple' "
                   "to match its quality.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model the run will use. Its own sigma<->time map defines the flow-time grid."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise=1.0):
        import torch as _t
        steps = int(steps[0] if isinstance(steps, (list, tuple)) else steps)
        denoise = float(denoise[0] if isinstance(denoise, (list, tuple)) else denoise)
        if denoise <= 0.0:
            return (_t.FloatTensor([]),)
        total_steps = max(int(steps / denoise), steps)
        ms = model.get_model_object("model_sampling")
        sigmas = paper_aeuler_mod._uniform_flow_time_sigmas(ms, total_steps)
        sigmas = sigmas[-(steps + 1):]
        return (sigmas,)


# ---------------------------------------------------------------------------
# options nodes
# ---------------------------------------------------------------------------


class ResearchAFlopsAutotuner:
    DESCRIPTION = ("PD-AOS Autotuner: probes + schedule fine-tune in one node. "
                   "Connect the guider the run will use and ANY scheduler's sigmas "
                   "(e.g. BasicScheduler \"simple\"); the node measures the model and "
                   "this prompt's trajectory at the schedule's noise levels (tiny "
                   "probe latents), then re-allocates the steps of YOUR sigma list "
                   "where the measured curvature needs them (error-equidistributed; "
                   "flat curvature reproduces your list exactly). Outputs the "
                   "fine-tuned sigmas for the sampler and the probe options for the "
                   "PD-AOS Sampler node (connect its options output to the Sampler's "
                   "options+ input). Two probes, two sizes: the MODEL probe is the "
                   "bigger schedule-driving one and is cached per model+params+"
                   "schedule (across prompts); the COND probe is the small per-prompt "
                   "one, so its cost is what you feel each generation. Elementary "
                   "knobs only: those two probe sizes, plus bias and warp, both "
                   "1.0 = neutral (the probe's own error-optimal placement of your "
                   "schedule). Both knobs use the factor 4**(w-1), so +0.5 and -0.5 "
                   "are exact reciprocals of each other: equal power in either "
                   "direction.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER", {"tooltip": "Connect the same guider your run uses, so the Autotuner measures the model and this prompt with the settings the run will actually use. It does its work here, before the sampler starts."}),
                "sigmas": ("SIGMAS", {"tooltip": "Connect any scheduler's output (BasicScheduler \"simple\", Karras, and so on). This is the schedule the Autotuner fine-tunes. When it has no usable measurements the list is passed through unchanged, so your scheduler is always the fallback."}),
                "probe_steps": ("INT", {"default": 10, "min": 5, "max": 16, "step": 1,
                    "tooltip": "How many steps the model probe uses, 5 at the minimum. This probe measures the model and decides the fine-tune of the schedule. Its result is kept and reused for every prompt with the same model and settings, so it costs you on the first run only."}),
                "probe_resolution": ("INT", {"default": 48, "min": 16, "max": 128, "step": 8,
                    "tooltip": "The picture size the model probe runs at. 48 is the established value. Larger sizes are slower and measure more finely."}),
                "cond_probe_steps": ("INT", {"default": 4, "min": 3, "max": 8, "step": 1,
                    "tooltip": "How many steps the prompt probe uses, 3 to 8. This probe runs again for every new prompt, so this is the setting you feel on each generation."}),
                "cond_probe_resolution": ("INT", {"default": 16, "min": 8, "max": 32, "step": 8,
                    "tooltip": "The picture size the prompt probe runs at. 16 is the default; 8 is the fastest and 32 the finest."}),
                "bias": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "How the probes' placement is tilted, used only when 'probe_schedule' is on or this is off 1.00. 1.00 leaves the placement as measured. Below 1.00 puts more of the steps in the late, low-noise part of the run, where detail is built -- measured on the corpus, this moves steps down on every run tested. Above 1.00 concentrates the end of the run into a sharper final drop, so the last step lands at a higher noise level -- measured to do that on every run tested. The first and last steps are never moved. At the default nothing here does anything, because your scheduler places the steps."}),
                "warp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Moves the sigma values themselves, leaving the first and last ones where they are. 1.00 keeps your schedule exactly as it is. Below 1.00 reaches low noise sooner and spends more of the run there. Above 1.00 does the opposite. Does nothing while auto_warp is on, and large values take the run outside the range of noise levels the probe measured."}),
                "auto_warp": ("BOOLEAN", {"default": False,
                    "tooltip": "Chooses the warp for you. ON: the warp setting is ignored and the Autotuner uses the strongest value that keeps every step inside the range of noise levels it measured. OFF: your warp value is used exactly as you set it."}),
                # APPENDED LAST (rule 15).  ComfyUI stores a saved workflow's widget values as a
                # POSITIONAL list, so a widget added anywhere but the END re-maps the values of the
                # widgets after it -- silently, and onto plausible numbers.
                "probe_smin_frac": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "How far down the noise range the two probes LOOK. 0.10 is what this node has always used: the probe stops at a tenth of the starting noise, so on a 16-step run it never measured below 0.17. Lower values measure deeper -- down to 0.0001, which reaches the model's own smallest noise level on the models tested -- and only then can the measurements say anything about where a run should end. This is a MEASUREMENT setting: at the default it changes nothing at all, and lowering it does not by itself move any steps. Cost: every step further down is extra model calls at the probe's own small size, on the model probe once per session and on the prompt probe every generation; going from 0.10 to 0.0001 multiplies the probe's model calls by roughly four. The count for each run is written into its log as the probe's step count."}),
                # APPENDED LAST (rule 15).
                "probe_schedule": ("BOOLEAN", {"default": False,
                    "tooltip": "Whether the steps are PLACED by what the probes measured. OFF (the default) is what this node has always done: your scheduler's list goes through unchanged, and the bias setting does nothing. ON: the probes' measured curvature decides where your steps go -- worth having where curvature is uneven, so the hard stretches get more steps and the easy ones get fewer -- and the bias setting tilts that placement towards structure or detail. This changes the sigma values themselves, so the images will differ from OFF; how much is written into the run's log."}),
                # EXPERIMENTAL SWITCHES, HIDDEN IN THE RELEASE BUILD.  `pct_derived`
                # and `wmax_budget` are no longer WIDGETS: they are fixed at their
                # engine defaults (pct_derived = True, wmax_budget = 1.0), which is
                # the shipped behaviour.  They are still accepted as parameters below
                # and still listed in `opts`, so a saved graph or an Options node can
                # drive them and the research rigs can keep testing them -- they are
                # simply not offered to users as knobs.  Reason: each was measured and
                # neither moves the image on the models we tested (`pct_derived` is a
                # bug fix and is ON; `wmax_budget` was measured inert).
                # `wmax_bound` USED TO BE HERE AND MOVED TO THE DEBUG OPTIONS NODE
                # (2026-10-18): with the key frozen off, the per-step fusion's
                # advertised "take the lower order when the two probes disagree" rule
                # could not run at all, and a rule that no widget can switch is the
                # defect (audit C2).  It is still a parameter here, default False, so
                # the node-side probes keep carrying the key.
            },
        }

    RETURN_TYPES = ("SIGMAS", OPT_TYPE)
    RETURN_NAMES = ("sigmas", "options")
    FUNCTION = "autotune"

    def autotune(self, guider, sigmas, probe_steps, probe_resolution,
                 cond_probe_steps, cond_probe_resolution,
                 bias, warp, auto_warp=False, probe_smin_frac=0.1,
                 probe_schedule=False,
                 pct_derived=True, wmax_bound=False, wmax_budget=1.0):
        import torch as _t
        import logging as _lg
        if isinstance(guider, (list, tuple)):
            guider = guider[0] if guider else None
        ext = _sig_tensor_list(sigmas)

        def _iv(v, default):
            if v is None:
                return default
            return int(v[0] if isinstance(v, (list, tuple)) else v)

        # Two probes, two sizes (restored): the MODEL probe is the bigger,
        # schedule-driving one and is cached across prompts; the COND probe is
        # the small per-prompt one, so its cost is what the user feels on each
        # generation.
        steps = max(5, min(16, _iv(probe_steps, 10)))
        res = max(16, min(128, _iv(probe_resolution, 48)))
        csteps = max(3, min(8, _iv(cond_probe_steps, 4)))
        cres = max(8, min(32, _iv(cond_probe_resolution, 16)))
        # THE PROBE'S REACH BELOW THE SHIPPED BAND.  0.10 is the shipped floor (`_PROBE_SMIN_FLOOR`
        # in the engine) and reproduces the shipped grid exactly; anything lower appends points and
        # is therefore a MEASUREMENT change, which is why it is in the engine's probe cache key.
        # Clamped, not merely documented: the engine re-reads this key from cfg, and a value outside
        # the widget's own range must not arrive by way of an API graph or an Options node.  The
        # lower end is DELIBERATELY below every model's own sigma_min on the models tested, so the
        # probes can be taken to the bottom of the range for measurement; the placement does not
        # follow the probe down unless something separately asks it to.
        sf = _f(float(probe_smin_frac) if probe_smin_frac is not None else 0.1)
        sf = max(0.0001, min(0.1, sf))
        # The lambda clamp is always probe-driven now.  The A/B mode widget was
        # removed once 'auto' proved to be the right setting, so nothing writes a
        # fixed aflops_lam_min/max any more; the engine's own automatic path
        # picks the per-step trust region from the probe's measured x0 jump.
        bi = _f(bias)
        wa = _f(warp)
        # Keep the WIDGET value separate: auto_warp overwrites `wa` with the
        # adopted value, and without both we cannot tell "auto picked 1.0" from
        # "the user asked for 1.0".  The report records neither today, which
        # makes any warp/bias A/B unverifiable from the logs.
        wa_widget = float(wa)

        def _sched_stamp(info):
            info = dict(info or {})
            info["widgets"] = {"bias": round(float(bi), 4),
                               "warp": round(wa_widget, 4),
                               "auto_warp": bool(auto_warp),
                               "warp_used": round(float(wa), 4)}
            return info

        opts = {
            "probe_enabled": True,
            "probe_steps": steps,
            "probe_resolution": res,
            "cond_probe_enabled": True,
            "cond_probe_steps": csteps,
            "cond_probe_resolution": cres,
            "cond_probe_weight": 0.5,
            # THE PROBE'S REACH MUST BE LISTED HERE for the same reason as the two arms below: a key
            # absent from this dict is absent from the engine cfg, and the engine would then read its
            # own default (the shipped 0.10) whatever the widget said.  That is exactly how the
            # Sampler's switch once read ON in the log while every probe ran with it OFF.
            "probe_smin_frac": float(sf),
            # THE DERIVED-CONSTANT ARMS MUST BE LISTED HERE OR THEY DO NOTHING.
            # This dict is hand-written, so a key that is not named here is absent from
            # the node-side probes' cfg -- which is precisely how the Sampler's switch came
            # to read ON in the log while every node/cond probe ran with it OFF.
            "pct_derived": bool(pct_derived),
            "wmax_bound": bool(wmax_bound),
            # THE NUDGE MUST BE LISTED HERE TOO, for the same reason as the two arms above:
            # a key absent from this dict is absent from the engine cfg, and then the gate
            # silently reads its neutral default 1.0 whatever the widget says.
            "wmax_budget": float(wmax_budget),
        }
        # Record the model's ACTUAL sampling + the user's input schedule so
        # the report is self-describing (no more guessing Flux/Cosmos/shift).
        try:
            _p = guider
            if hasattr(guider, "model_patcher"):
                _p = guider.model_patcher
            elif hasattr(guider, "inner_model"):
                _in = guider.inner_model
                if hasattr(_in, "model_patcher"):
                    _p = _in.model_patcher
            _ms = _p.get_model_object("model_sampling")

            def _fval(x):
                try:
                    import torch as _t
                    if _t.is_tensor(x):
                        return float(x.detach().item())
                    return float(x)
                except Exception:
                    return None

            _sigs = getattr(_ms, "sigmas", None)
            _ms_info = {
                "class": type(_ms).__name__,
                "shift": _fval(getattr(_ms, "shift", None)),
                "sigma_min": (_fval(_ms.sigma_min)
                              if hasattr(_ms, "sigma_min") else None),
                "sigma_max": (_fval(_ms.sigma_max)
                              if hasattr(_ms, "sigma_max") else None),
                "multiplier": _fval(getattr(_ms, "multiplier", None)),
                "n_sigmas": (len(_sigs) if _sigs is not None else None),
            }
        except Exception as _e:
            _ms_info = {"error": str(_e)}
        _ms_info["sigmas_in"] = ([round(float(v), 5) for v in ext]
                                 if ext is not None else None)
        opts["_ms_info"] = _ms_info
        if guider is None:
            # Nothing to measure: pass the schedule through untouched.
            _lg.warning("[PD-AOS-autotuner] no guider connected; sigmas "
                        "passed through unmodified")
            opts["_shift_schedule_node"] = _sched_stamp(
                {"mode": "off", "adopted": False, "base": "external"})
            return (sigmas, opts)
        if ext is None:
            _lg.warning("[PD-AOS-autotuner] unusable sigmas input; passed "
                        "through unmodified")
            opts["_shift_schedule_node"] = _sched_stamp(
                {"mode": "off", "adopted": False, "base": "external"})
            return (sigmas, opts)

        # 1) measure: model probe then per-prompt cond probe, both on the
        #    incoming schedule (schedule-keyed caches, no double probe).
        #    (The cond signature was hoisted ABOVE the model probe for a few hours on
        #    2026-10-08, while the model key briefly carried it.  It does not: a model
        #    measurement is model-keyed and shared across prompts by design, so the signature
        #    is needed only from the cond block onwards -- see `_probe_key_env_part`.)
        profile_m = None
        profile_c = None
        try:
            profile_m, _ = aflops_mod._run_node_side_model_probe(
                guider, opts, sigmas=ext)
        except Exception as e:
            _lg.warning("[PD-AOS-autotuner] model probe failed: %s", e)
        try:
            probe_cfg = aflops_mod._guider_probe_cfg(guider)
            if probe_cfg is None:
                probe_cfg = 7.0
            opts["cond_probe_sig"] = aflops_mod._guider_cond_signature(
                guider, probe_cfg)
            profile_c, _ = aflops_mod._run_node_side_cond_probe_guider(
                guider, opts, sigmas=ext)
        except Exception as e:
            _lg.warning("[PD-AOS-autotuner] cond probe failed: %s", e)

        # 2) fine-tune: the per-prompt profile wins when it can drive the
        #    schedule (>= 3 curvature points; guaranteed at probe_steps >= 5),
        #    else the model profile, else untouched passthrough.
        _p = None
        for _c in (profile_c, profile_m):
            if (isinstance(_c, dict) and _c.get("space") == "flow"
                    and len(_c.get("curv_curve") or []) >= 3):
                _p = _c
                break
        sched_info = {"mode": "off", "adopted": False, "base": "external"}
        grid = None
        wa_info = None
        if _p is not None and bool(auto_warp):
            # auto-warp: the probe picks the most aggressive safe warp; the
            # manual widget is disabled while this is on.
            wa, wa_info = aflops_mod._warp_frontier(
                _p, ext, bias=float(bi))
            _lg.info("[PD-AOS-autotuner] auto-warp: adopted %.2f "
                     "(safe range %.2f-1.00, %s; worst-step ratio %.3f, "
                     "steps below measured floor %s)"
                     % (wa, wa_info.get("safe_range", [wa, 1.0])[0],
                        wa_info.get("why", "?"),
                        wa_info.get("worst_ratio", 1.0),
                        wa_info.get("below", "?")))
        # THE DEGENERACY THAT MADE THIS INERT IS FIXED (2026-10-09) -- but the shipped default is
        # staged to stay exactly as it was, so the note above is superseded rather than deleted.
        # Measured on the corpus: the builder's resolution floor solved `mu = mean(max(w, mu))`,
        # whose only solution is `mu = max(w)`, so every weight was identical and the placement
        # reproduced the base for ANY curvature, any `p`, any bias -- 832 builds matched their base
        # to the builder's own rounding, `mu == max(w)` on 104/104 runs.  `engage` is what stages
        # the repair: at bias = 1.0 with `probe_schedule` off the builder now returns the base
        # EARLY and bit-identically, and only a run that turns the derived schedule on (or moves
        # `bias` off neutral) gets the probes' placement.  Passing None when the switch is off
        # leaves the tilt rule in ONE place -- the engine.
        if _p is not None:
            _engage = True if bool(probe_schedule) else None
            grid = aflops_mod._build_curvature_schedule(
                _p, len(ext) - 1, 1.0, bias=float(bi), warp=float(wa),
                base_sigmas=ext, engage=_engage)
        if grid is not None and grid.numel() == len(ext):
            sched_info = {"mode": "curvature", "adopted": True,
                          "base": "external",
                          "bias": round(float(bi), 4),
                          "warp": round(float(wa), 4),
                          # THE VALUE IN FORCE, stamped rather than inferred (rule 14): `adopted`
                          # is True on the identity path too, so without this a reader cannot tell
                          # "the probes placed these steps" from "your scheduler did".
                          "placed_by_probes": bool(probe_schedule)}
            if wa_info is not None:
                sched_info["auto_warp"] = {
                    "safe_range": wa_info.get("safe_range"),
                    "why": wa_info.get("why"),
                    "worst_ratio": wa_info.get("worst_ratio"),
                    "below": wa_info.get("below")}
            _lg.info("[PD-AOS-autotuner] schedule fine-tuned from probe "
                     "curvature (bias=%.2f warp=%.2f%s, %d steps)",
                     float(bi), float(wa),
                     " auto" if wa_info is not None else "", len(ext) - 1)
            out_grid = grid
        else:
            # "adopted" above does NOT mean the schedule CHANGED -- a grid
            # numerically identical to the input lands here too, which is how
            # the inert re-allocation stayed invisible for so long.  A future
            # A/B should stamp the max relative change so "adopted" and
            # "actually moved" can be told apart.
            sched_info = {"mode": "off", "adopted": False, "base": "external"}
            _lg.info("[PD-AOS-autotuner] no usable curvature evidence; "
                     "schedule passed through")
            out_grid = _t.tensor(ext, dtype=_t.float32)
        opts["_shift_schedule_node"] = _sched_stamp(sched_info)
        return (out_grid, opts)


class ResearchAFlopsErrorReport:
    DESCRIPTION = ("Outputs the per-step internal decisions from the last run as JSON: "
                   "solver chosen, errors, guard trips, anomaly verdicts, call counts, "
                   "etc. Connect the sampled latent to 'trigger' to force this node to "
                   "run after sampling. Only updates when the graph re-runs.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"trigger": ("LATENT", {"tooltip": "Connect the sampled latent here to force this node to run after sampling completes."})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("errors_json",)
    FUNCTION = "report"
    OUTPUT_NODE = True

    def report(self, trigger=None):
        # Include only the CURRENT run's own profiles, not the whole cache:
        # every saved/cached profile for every model+prompt+schedule was being
        # dumped into each log, so the JSON grew with the session instead of
        # describing this run (context rot).
        _cfg = aflops_mod._LAST_CFG or {}
        _mk = None
        for _k in ("_probe_key", "probe_key"):
            if _cfg.get(_k):
                _mk = str(_cfg[_k]).split("|s:")[0]
                break
        _psig = None
        for _k in ("_cond_probe_key", "cond_probe_key"):
            if _cfg.get(_k):
                # THE MODEL KEY MAY ITSELF CONTAIN "|" (`|p:<patches_uuid>`), so a fixed
                # field index is no longer the prompt.  One helper owns the parse.
                _psig = aflops_mod._cond_key_prompt_sig(_cfg[_k])
                break

        def _own_model(key):
            if _mk is None:
                # No key means this run cannot identify its own profile; dumping
                # the process cache here is the context-rot bug this filter exists
                # to prevent.
                return False
            return key == _mk or key.startswith(_mk + "|s:")

        def _own_cond(key):
            if _mk is None:
                return False
            if key[0] != _mk:
                return False
            if _psig is None:
                # No prompt signature: the run cannot attribute a cond profile.
                return False
            return _psig in str(key[1])

        probe_summary = {}
        try:
            for key, prof in aflops_mod._PROBE_PROFILES.items():
                if not _own_model(str(key)):
                    continue
                probe_summary[str(key)] = {
                    "version": prof.get("version"),
                    "space": prof.get("space"),
                    "n_probe_steps": prof.get("n_probe_steps"),
                    # ADDED 2026-10-05 with the fields themselves.  These dicts are FIELD PROJECTIONS,
                    # not copies: a field not listed here never reaches a log, which is why the run that
                    # asked for 16 px while consuming a 32 px measurement showed no trace of the resolution.
                    "n_probe_res": prof.get("n_probe_res"),
                    # ADDED 2026-10-08.  `probe_src` says whether this profile is a MODEL
                    # measurement or a per-prompt COND one -- the two can occupy the same
                    # model slot, because the node-side COND probe writes `_PROFILE_BY_MODEL`
                    # and the engine PROMOTES what it adopts into its own `|s:` slot.  The
                    # adoption gate (`_profile_serves_model_probe`) reads it, so a log
                    # without it cannot show which producer a slot's numbers came from.
                    # `probe_device` is what the probe measured ON: the cond paths move
                    # their latent to whatever device `_probe_call_device` chooses, which is
                    # runtime state, not configuration.
                    "probe_src": prof.get("probe_src"),
                    "probe_device": prof.get("probe_device"),
                    "curv_sigma": prof.get("curv_sigma"),
                    # THE PROBE'S REACH.  A whitelist drops any key not named here, and these four
                    # are the difference between "measured to 0.1" and "measured to 0.01" -- two
                    # profiles that are NOT comparable and were previously indistinguishable.
                    "probe_reach_frac": prof.get("probe_reach_frac"),
                    "probe_band_floor": prof.get("probe_band_floor"),
                    "probe_reach_floor": prof.get("probe_reach_floor"),
                    "probe_reach_n_extra": prof.get("probe_reach_n_extra"),
                    "is_distilled_eff": prof.get("is_distilled_eff"),
                    "safe_fresh_frac": prof.get("safe_fresh_frac"),
                    "med_jump": prof.get("med_jump"),
                    "med_curv": prof.get("med_curv"),
                    # the trajectory evidence curves the shift/schedule
                    # recommendation consumes -- without them the "how fast can
                    # we drop sigma" question cannot be answered from the log.
                    "sigma_curve": prof.get("sigma_curve"),
                    "curv_curve": prof.get("curv_curve"),
                    "jump_curve": prof.get("jump_curve"),
                    # per-interval error-model evidence (err vs curv vs gap):
                    # the schedule's density exponent a/p is fitted from
                    # these; without them the exponent cannot be measured
                    # from the report at all.
                    "local_err_curve": prof.get("local_err_curve"),
                    "gap_curve": prof.get("gap_curve"),
                    "real_gap_curve": prof.get("real_gap_curve"),
                    "slope_curve": prof.get("slope_curve"),
                    "whitepoint": prof.get("whitepoint"),
                    "med_local_err": prof.get("med_local_err"),
                    "guard_sig_med": prof.get("guard_sig_med"),
                    "guard_sig_p90": prof.get("guard_sig_p90"),
                    "log_gap_probe": prof.get("log_gap_probe"),
                    "log_gap_real": prof.get("log_gap_real"),
                    # v7: the escape-threshold derivation inputs (stalled =
                    # 1/4 of the measured normal jump, flat = 1/2 of the
                    # measured normal detail) and the late-fade anchor, so
                    # the adaptive-eta math is fully auditable from the
                    # report instead of only from the in-memory profile.
                    "med_jump_rel": prof.get("med_jump_rel"),
                    "med_detail_abs": prof.get("med_detail_abs"),
                    "conv_sigma": prof.get("conv_sigma"),
                    "osc_sigma": prof.get("osc_sigma"),
                    "early_frac": prof.get("early_frac"),
                    "conv_step": prof.get("conv_step"),
                    "fragility": prof.get("fragility"),
                    "calib": prof.get("calib"),
                    # v6: the local-field evidence the model probe
                    # contributes -- trajectory state, the run-level
                    # direction feedback and the cfg scale the probe ran
                    # at.  Without these the derived gains were only half
                    # auditable from this report.
                    "trajectory_state": prof.get("trajectory_state"),
                    "dir_real": prof.get("dir_real"),
                    "probe_cfg": prof.get("probe_cfg"),
                    # THE PROBE-SIDE RESCUE COVERAGE CURVE.  This report is a
                    # WHITELIST, not a dump: any profile key not named here never
                    # reaches the log, whatever the engine computes.  That is how
                    # `probe_diag` and `med_local_err_real` have been produced by
                    # the profile builder for a long time yet are absent from
                    # every log, and it is why the first cov_curve build appeared
                    # to have no effect.  Since the rescue gate's coverage CLIFF
                    # is the pitfall these runs exist to derive a rule for, the
                    # curve and its provenance must be here or the campaign
                    # produces data that cannot answer the question.
                    "cov_curve": prof.get("cov_curve"),

                    "cov_curve_terms": prof.get("cov_curve_terms"),
                    "cov_curve_meta": prof.get("cov_curve_meta"),
                    # THE DERIVED VARIANT + its measured rate.  The synthetic
                    # fields could not decide between the shipped per-step decay
                    # and ref_drop = k*du (scratch_ref_decay_rig.py refuted both
                    # of its fields), so both curves must reach the log and the
                    # real trajectories decide.
                    "cov_curve_kdu": prof.get("cov_curve_kdu"),
                    "cov_curve_k": prof.get("cov_curve_k"),
                    "probe_call": prof.get("probe_call"),
                    # WHICH PATCHER THE PROBE RAN ON.  The clone-path freeze is the
                    # open defect (NOTES.md, "THE FREEZE IS AN INTERACTION") and the
                    # clone was removed to test it -- but that change could NOT be
                    # confirmed from the corpus, because the WHITELIST documented
                    # above is precisely the thing that silently drops a new profile
                    # key.  Stamped by _engine_cond_probe_run; ABSENT on every build
                    # that used a clone, so its presence in a log is the proof that
                    # the running code carries the removal.
                    "probe_patcher": prof.get("probe_patcher"),
                    # HOW MANY TIMES THE SAMPLING WRAPPER WAS ENTERED.  One is normal; a
                    # higher value means the signature-retry path ran, which used to
                    # re-run the whole probe after a post-sampling TypeError.
                    "probe_attempts": prof.get("probe_attempts"),
                    "probe_mopt_swapped": prof.get("probe_mopt_swapped"),
                    # THE RANK LADDER'S DECISION.  The node-side probes fall
                    # back to rank 4, and a 4-D latent through a 5-D model is a
                    # SILENT pass-through (rank 4 -> 9 frozen / 0 healthy on the
                    # corpus, rank 5 -> 0 frozen / 16 healthy).  These record
                    # which rank was used, what was tried, and the ladder order.
                    "probe_rank": prof.get("probe_rank"),
                    "probe_rank_tried": prof.get("probe_rank_tried"),
                    "probe_rank_ladder": prof.get("probe_rank_ladder"),
                    # present in the profile builder, absent from every log so far:
                    "probe_diag": prof.get("probe_diag"),
                    "med_local_err_real": prof.get("med_local_err_real"),
                    # THE INPUTS THE DERIVED THRESHOLDS ARE COMPUTED FROM.  A log that
                    # carries only the OUTPUT (calib) cannot say WHY a value came out
                    # clamped: reading calib alone, guard_floor 0.075 looks like a floor
                    # when it is really 0.15*sqrt(r/4) at r = 1, and wmax_base 2.5 looks
                    # like a ceiling when it is the map saturating below med_curv 1.4775.
                    # `jump_q` and `err_q` are named by _derive_anomaly_calibrations as the
                    # evidence for an_spatial / an_novelty / an_value, and n_jump feeds
                    # conv_step; without them those thresholds are unauditable from a log.
                    # Found by scratch_log_coverage_audit.py, which diffs the profile keys
                    # the engine assigns against this list.
                    "jump_q": prof.get("jump_q"),
                    "err_q": prof.get("err_q"),
                    "n_jump": prof.get("n_jump"),
                    "n_err": prof.get("n_err"),
                    "n_guard": prof.get("n_guard"),
                    "endgame": prof.get("endgame"),
                    "fragile_ranges": prof.get("fragile_ranges"),
                    "dir_cons": prof.get("dir_cons"),
                    "dir_cons_valid": prof.get("dir_cons_valid"),
                    "ojf_curve": prof.get("ojf_curve"),
                    "band_curve": prof.get("band_curve"),
                    # THE NOISE-ABSORPTION SERIES.  This whitelist is the only
                    # route from a cached profile into a run log, and the series
                    # was never added -- so the instrument was invisible even on
                    # the runs where it did measure.  Present only when the
                    # Experimental node asks for the curve; empty otherwise, and
                    # a profile that never carried the key reads back None.
                    "noise_survival_curve": prof.get("noise_survival_curve"),
                    # temporary diagnostic; see the stamp in `_run_probe`
                    "_ns_curve_diag": prof.get("_ns_curve_diag"),
                    # THE PER-ORDER DERIVATIVE SCALE the `wmax_bound` arm consumes, so the
                    # log says which orders this grid could certify and which it could not.
                    "dk_scale": prof.get("dk_scale"),
                    # THE PER-STEP SERIES behind the median.  This whitelist is the ONLY
                    # route from a cached profile into the run log; without this line the
                    # series was measured and then discarded, which is the same defect the
                    # `noise_survival_curve` note above records.
                    "dk_series": prof.get("dk_series"),
                    # WHICH A/B FLAGS THIS PROFILE WAS BUILT UNDER.  Two different nodes can
                    # build probes (the Autotuner's opts dict and the Sampler's cfg), so
                    # without this a disagreement is invisible and the log's cfg block
                    # describes only the Sampler's settings -- which is exactly how a
                    # switch read ON while the probes ran OFF.
                    "calib_flags": prof.get("calib_flags"),
                    # THE TWO ESTIMATORS, SIDE BY SIDE.  So one run shows the shipped-vs-
                    # derived delta without needing a toggle flipped, and `n_guard` says
                    # whether the p90/med collision is in play for this profile.
                    "guard_sig_med_shipped": prof.get("guard_sig_med_shipped"),
                    "guard_sig_p90_shipped": prof.get("guard_sig_p90_shipped"),
                    "guard_sig_med_derived": prof.get("guard_sig_med_derived"),
                    "guard_sig_p90_derived": prof.get("guard_sig_p90_derived"),
                }
        except Exception as _e:
            logging.warning("[PD-AOS] model profile projection failed: %s: %s",
                            type(_e).__name__, _e)
        # Per-prompt (conditioned) probe profiles: refreshed every generation
        # the Cond Probe node runs, keyed by (model, prompt signature).
        cond_probe_summary = {}
        try:
            # Defensive: _COND_PROBE_PROFILES is not defined until the
            # cond-probe engine support lands.  getattr avoids an
            # AttributeError that would silently empty this section.
            _cond_store = getattr(aflops_mod, '_COND_PROBE_PROFILES', {})
            # ONE FORMATTER, SHARED WITH THE ENGINE -- called directly, with NO inline
            # fallback: a fallback that re-spells the key is how the two spellings diverged
            # in the first place (the log's store keys and the stamped `cfg["_cond_probe_key"]`
            # matched in 0 of 443 logs).  If the helper is ever missing, that is an error worth
            # seeing, not one worth papering over.
            for key, prof in _cond_store.items():
                if not _own_cond(key):
                    continue
                cond_probe_summary[aflops_mod._cond_key_str(key)] = {
                    "cond_sig": prof.get("cond_sig"),
                    "space": prof.get("space"),
                    "n_probe_steps": prof.get("n_probe_steps"),
                    # ADDED 2026-10-05 with the fields themselves.  These dicts are FIELD PROJECTIONS,
                    # not copies: a field not listed here never reaches a log, which is why the run that
                    # asked for 16 px while consuming a 32 px measurement showed no trace of the resolution.
                    "n_probe_res": prof.get("n_probe_res"),
                    # ADDED 2026-10-08 -- see the model projection's comment for why.
                    "probe_src": prof.get("probe_src"),
                    "probe_device": prof.get("probe_device"),
                    "curv_sigma": prof.get("curv_sigma"),
                    # the probe's reach -- see the model-projection comment for why these must be
                    # named explicitly rather than assumed present.
                    "probe_reach_frac": prof.get("probe_reach_frac"),
                    "probe_band_floor": prof.get("probe_band_floor"),
                    "probe_reach_floor": prof.get("probe_reach_floor"),
                    "probe_reach_n_extra": prof.get("probe_reach_n_extra"),
                    "version": prof.get("version"),
                    "med_jump": prof.get("med_jump"),
                    "med_curv": prof.get("med_curv"),
                    "sigma_curve": prof.get("sigma_curve"),
                    "curv_curve": prof.get("curv_curve"),
                    "jump_curve": prof.get("jump_curve"),
                    "local_err_curve": prof.get("local_err_curve"),
                    "gap_curve": prof.get("gap_curve"),
                    "real_gap_curve": prof.get("real_gap_curve"),
                    "slope_curve": prof.get("slope_curve"),
                    "safe_fresh_frac": prof.get("safe_fresh_frac"),
                    "is_distilled_eff": prof.get("is_distilled_eff"),
                    "med_local_err": prof.get("med_local_err"),
                    "guard_sig_med": prof.get("guard_sig_med"),
                    "guard_sig_p90": prof.get("guard_sig_p90"),
                    "log_gap_probe": prof.get("log_gap_probe"),
                    "log_gap_real": prof.get("log_gap_real"),
                    "med_jump_rel": prof.get("med_jump_rel"),
                    "med_detail_abs": prof.get("med_detail_abs"),
                    "conv_sigma": prof.get("conv_sigma"),
                    "osc_sigma": prof.get("osc_sigma"),
                    "early_frac": prof.get("early_frac"),
                    "conv_step": prof.get("conv_step"),
                    "fragility": prof.get("fragility"),
                    "gap": prof.get("gap"),
                    "probe_cfg": prof.get("probe_cfg"),
                    "dir_real": prof.get("dir_real"),
                    "trajectory_state": prof.get("trajectory_state"),
                    "calib": prof.get("calib"),
                    "focused": prof.get("focused"),
                    # probe-side rescue coverage curve -- see the model-profile
                    # whitelist above for why this must be named explicitly.
                    "cov_curve": prof.get("cov_curve"),

                    "cov_curve_terms": prof.get("cov_curve_terms"),
                    "cov_curve_meta": prof.get("cov_curve_meta"),
                    # THE DERIVED VARIANT + its measured rate.  The synthetic
                    # fields could not decide between the shipped per-step decay
                    # and ref_drop = k*du (scratch_ref_decay_rig.py refuted both
                    # of its fields), so both curves must reach the log and the
                    # real trajectories decide.
                    "cov_curve_kdu": prof.get("cov_curve_kdu"),
                    "cov_curve_k": prof.get("cov_curve_k"),
                    "probe_call": prof.get("probe_call"),
                    # which patcher the cond probe ran on (see the model-probe
                    # whitelist for why this must be named explicitly)
                    "probe_patcher": prof.get("probe_patcher"),
                    "probe_mopt_swapped": prof.get("probe_mopt_swapped"),
                    # how many times the sampling wrapper was entered (see the
                    # model-probe whitelist)
                    "probe_attempts": prof.get("probe_attempts"),
                    # the rank ladder's decision (see the model-probe whitelist)
                    "probe_rank": prof.get("probe_rank"),
                    "probe_rank_tried": prof.get("probe_rank_tried"),
                    "probe_rank_ladder": prof.get("probe_rank_ladder"),
                    "med_local_err_real": prof.get("med_local_err_real"),
                    # THE INPUTS THE COND THRESHOLDS ARE COMPUTED FROM -- see the
                    # model-probe whitelist for why: a log carrying only `calib` cannot say
                    # why a value landed on a clamp.  Adding a key the cond profile does
                    # not carry is harmless (prof.get returns None).
                    "jump_q": prof.get("jump_q"),
                    "err_q": prof.get("err_q"),
                    "n_jump": prof.get("n_jump"),
                    "n_err": prof.get("n_err"),
                    "n_guard": prof.get("n_guard"),
                    "endgame": prof.get("endgame"),
                    "fragile_ranges": prof.get("fragile_ranges"),
                    "dir_cons": prof.get("dir_cons"),
                    "dir_cons_valid": prof.get("dir_cons_valid"),
                    "ojf_curve": prof.get("ojf_curve"),
                    "band_curve": prof.get("band_curve"),
                    # THE NOISE-ABSORPTION SERIES.  This whitelist is the only
                    # route from a cached profile into a run log, and the series
                    # was never added -- so the instrument was invisible even on
                    # the runs where it did measure.  Present only when the
                    # Experimental node asks for the curve; empty otherwise, and
                    # a profile that never carried the key reads back None.
                    "noise_survival_curve": prof.get("noise_survival_curve"),
                    # temporary diagnostic; see the stamp in `_run_probe`
                    "_ns_curve_diag": prof.get("_ns_curve_diag"),
                    # THE PER-ORDER DERIVATIVE SCALE the `wmax_bound` arm consumes, so the
                    # log says which orders this grid could certify and which it could not.
                    "dk_scale": prof.get("dk_scale"),
                    # THE PER-STEP SERIES behind the median.  This whitelist is the ONLY
                    # route from a cached profile into the run log; without this line the
                    # series was measured and then discarded, which is the same defect the
                    # `noise_survival_curve` note above records.
                    "dk_series": prof.get("dk_series"),
                    # WHICH A/B FLAGS THIS PROFILE WAS BUILT UNDER.  Two different nodes can
                    # build probes (the Autotuner's opts dict and the Sampler's cfg), so
                    # without this a disagreement is invisible and the log's cfg block
                    # describes only the Sampler's settings -- which is exactly how a
                    # switch read ON while the probes ran OFF.
                    "calib_flags": prof.get("calib_flags"),
                    # THE TWO ESTIMATORS, SIDE BY SIDE.  So one run shows the shipped-vs-
                    # derived delta without needing a toggle flipped, and `n_guard` says
                    # whether the p90/med collision is in play for this profile.
                    "guard_sig_med_shipped": prof.get("guard_sig_med_shipped"),
                    "guard_sig_p90_shipped": prof.get("guard_sig_p90_shipped"),
                    "guard_sig_med_derived": prof.get("guard_sig_med_derived"),
                    "guard_sig_p90_derived": prof.get("guard_sig_p90_derived"),
                }
        except Exception as _e:
            logging.warning("[PD-AOS] cond profile projection failed: %s: %s",
                            type(_e).__name__, _e)
        # WHY IS THE COND DUMP EMPTY? (2026-10-04)  On the operator's first `cond_rtol_loosen` A/B the
        # dump was EMPTY in all 12 runs (`cond_probe_profiles: 0`) even though the cond probe had RUN
        # (`node_cond_probe_state: ran`) -- and an empty dump is indistinguishable from "no cond profile
        # exists", which is what left that A/B's null unexplained.  This records WHICH filter rejected
        # them, in the log, instead of leaving it to be inferred: how many cond profiles exist, how many
        # carry this run's model key, how many also pass the prompt-signature test, and how many were
        # actually dumped.  `_LAST_CFG` is what the log serialises, so a diagnostic put here lands in the
        # corpus.  It changes no behaviour.
        if _cfg is not None:
            try:
                _cond_all = getattr(aflops_mod, "_COND_PROBE_PROFILES", {}) or {}
                _same_model = sum(1 for _k in _cond_all
                                  if _mk is None or (isinstance(_k, (tuple, list)) and _k
                                                     and _k[0] == _mk))
                _strict = sum(1 for _k in _cond_all
                              if isinstance(_k, (tuple, list)) and len(_k) >= 2
                              and (_mk is None or _k[0] == _mk)
                              and (_psig is None or _psig in str(_k[1])))
                _cfg["_cond_dump_diag"] = {"store": len(_cond_all), "same_model": _same_model,
                                           "strict_ok": _strict, "dumped": len(cond_probe_summary),
                                           "mk": _mk, "psig": (str(_psig)[:28] if _psig else None)}
            except Exception as _e:
                logging.warning("[PD-AOS] cond dump diagnostic failed: %s: %s",
                                type(_e).__name__, _e)
        return (json.dumps({"aflops_cfg": aflops_mod._LAST_CFG,
                            "probe_profiles": probe_summary,
                            "cond_probe_profiles": cond_probe_summary,
                            "aflops": aflops_mod._LAST_ERRORS}, indent=2),)


NODE_CLASS_MAPPINGS = {
    "ResearchAFlopsSampler": ResearchAFlopsSampler,
    "ResearchAFlopsAutotuner": ResearchAFlopsAutotuner,
    "ResearchAFlopsDebugOptions": ResearchAFlopsDebugOptions,
    "ResearchAFlopsNoiseOptions": ResearchAFlopsNoiseOptions,
    "ResearchAFlopsExperimental": ResearchAFlopsExperimental,
    "ResearchAFlopsErrorReport": ResearchAFlopsErrorReport,
    "ResearchPaperAEuler": ResearchPaperAEuler,
    "ResearchPaperAEulerSchedule": ResearchPaperAEulerSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResearchAFlopsSampler": "PD-AOS Sampler",
    "ResearchAFlopsAutotuner": "PD-AOS Autotuner (probe + schedule)",
    "ResearchAFlopsDebugOptions": "PD-AOS Debug Options (finished switches)",
    "ResearchAFlopsNoiseOptions": "PD-AOS Noise (ETA) Options",
    "ResearchAFlopsExperimental": "PD-AOS Experimental Options",
    "ResearchAFlopsErrorReport": "PD-AOS Error Report",
    "ResearchPaperAEuler": "A-FloPS A-Euler (paper)",
    "ResearchPaperAEulerSchedule": "A-FloPS A-Euler Schedule (paper)",
}
