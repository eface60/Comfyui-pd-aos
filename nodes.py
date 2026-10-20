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
    DESCRIPTION = ("PD-AOS is an adaptive sampler for diffusion models. At each step it "
                   "splits the model's prediction into a straight-line part and a "
                   "correction, solves the straight-line part exactly, and extrapolates the "
                   "correction from earlier steps. That keeps one model call per step while "
                   "following curved paths, and each pixel can use its own number of "
                   "earlier steps when that helps. Most behaviour adapts automatically. "
                   "Connect a MODEL so PD-AOS can detect the model type, noise scale, video "
                   "latents and distilled models. Connect the Autotuner to measure the "
                   "model and tune the schedule, and connect Options nodes to override "
                   "individual settings.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "log_errors": ("BOOLEAN", {"default": False,
                    "tooltip": "Prints the sampler's decisions at every step to the console. It does not change the image. A run longer than 64 steps keeps the first 64 step records in the Error Report when this is off, and the latest 64 when it is on. Turn it on while testing a prompt; leave it off for normal runs."}),
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
                "model": ("MODEL", {"tooltip": "Connect the MODEL so PD-AOS can adapt to it: model type, noise scale, video latents and distilled models (turbo, schnell, lightning) are detected automatically. Any Options node can override what is detected. Without a MODEL, the sampler falls back to a simple noise-level estimate."}),
                "options+": (OPT_TYPE, {"tooltip": "Connect one or more PD-AOS Options nodes here. Their settings override the sampler's automatic choices for the fields they set. Chain them in the order you want applied: the node closest to the sampler wins when two set the same field."}),
                # APPENDED LAST (rule 35).  ComfyUI stores a saved workflow's widget values as a
                # POSITIONAL list, so a widget added anywhere but the END re-maps the values of
                # the widgets after it -- silently, and onto plausible numbers.  Appending here
                # leaves `log_errors` at index 0 and gives an older saved graph a default for
                # this one, which is what makes the change safe to load.
                "use_measured_tolerance": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("use_measured_tolerance", False)),
                    "tooltip": "Chooses the tolerance used to judge each step's error. Off: take the tolerance measured for your model, but never below a fixed minimum (0.01536). On: take the measured value exactly, however small it is, so the sampler refuses its correction more often and falls back to the plain step. When the measured value is already above the minimum, both settings behave the same. Only does anything when the Autotuner has measured the model; otherwise the sampler's own tolerance is used."}),
                "probe_fuse_per_step": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("probe_fuse_per_step", False)),
                    "tooltip": "How the model probe and the prompt probe are combined. Off: each probe is reduced to one number per setting, and the two numbers are averaged once before sampling. On: both probes' per-step curves are kept, lined up with your actual steps, and each step's order limit comes from the combined curve; where the probes disagree, the lower (safer) order is used. It does not change the guard, the anomaly detector or the tolerance. It needs probe measurements, so connect the Autotuner or another Options node that supplies them."}),
                "probe_fuse_raise_cap": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("probe_fuse_raise_cap", False)),
                    "tooltip": "Only used when 'probe_fuse_per_step' is on. Off: the combined curve may only lower the order limit, so turning the combination on changes one thing only. On: it may also raise the limit above the normal cap, which changes two things at once and makes the result harder to compare."}),
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
    DESCRIPTION = ("PD-AOS Debug Options: advanced switches for testing and for reproducing "
                   "a run. They are not needed for normal sampling, because the Sampler "
                   "uses the same values as its defaults. Connect this node's 'options' "
                   "output to the PD-AOS Sampler's 'options+' input, or chain it through "
                   "another Options node; the node closest to the Sampler wins where two "
                   "set the same field. The experimental switches at the bottom are off by "
                   "default and may change or be removed between releases.")
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
                    "tooltip": "How much extrapolation order the sampler demands, i.e. how many previous steps each pixel's update should use. 0 = automatic: ask for more order while the run's own error is high, then release as the error settles. 1 = no demand; the sampler chooses. 2-6 = demand at least that many previous steps for every pixel. The demand is a request, not a guarantee: the safety check can refuse it, and a pixel that was just reset stays at order 1. Higher values push the trajectory harder."}),
                "order_floor_adaptive_rtol": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_rtol", 0.05)),
                    "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Only used when 'order_floor' is 0 (automatic). The error level that opens the automatic demand: the sampler asks for the high order while a step's error divided by its tolerance is above this number. Lower values open the gate more often, so the run stays nearer a high fixed demand; higher values open it less often, so it stays nearer no demand. Roughly 0.01 to 0.3 is the useful range; near the top the gate almost never opens."}),
                # APPENDED LAST (rule 35): the bound-derived per-step request.
                "order_demand_q": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_demand_q", 0.0)),
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used when 'order_floor' is 0 (automatic) and 'order_by_bound' is on. 0 = off: the automatic demand follows the usual rules: the error gate, or the per-pixel validity bound where there is one. Above 0, the demand becomes that quantile of the per-pixel validity bound, i.e. of the highest order the sampler's own error test allows at each pixel. 0.5 asks for the middle of that range, 1.0 for the maximum. Experimental: the demand can jump between low and high instead of varying smoothly, so 0 is the safe setting."}),
                "order_floor_adaptive_graded": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_floor_adaptive_graded", True)),
                    "tooltip": "Only used when 'order_floor' is 0 (automatic) and the demand comes from the error gate rather than the per-pixel validity bound. On: as the run's error rises above the gate threshold, the demanded order ramps from 'order_floor_adaptive_lo' to 'order_floor_adaptive_hi' instead of jumping straight to the maximum. Off: the demand is flat at 6 whenever the gate is open. On asks for roughly the order the sampler actually uses at that error level."}),

                # ---- THE GRADED CURVE'S ENDPOINTS, APPENDED LAST (rule 35).  They were cfg-read constants with
                # no emitter, which meant neither the UI nor a batch graph could set them -- and a value that
                # cannot be varied from the UI cannot be A/B'd by the person with the eyes.
                "order_floor_adaptive_lo": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_lo", 4.0)),
                    "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Only used when 'order_floor' is 0 (automatic), 'order_floor_adaptive_graded' is on and the demand comes from the error gate. The order demanded just above the gate threshold, where the ramp starts. As error rises, the demand moves from here toward 'order_floor_adaptive_hi'. Setting it equal to 'order_floor_adaptive_hi' gives a flat demand; setting it higher than 'order_floor_adaptive_hi' inverts the ramp and is almost certainly wrong."}),
                "order_floor_adaptive_hi": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_hi", 5.0)),
                    "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Only used when 'order_floor' is 0 (automatic), 'order_floor_adaptive_graded' is on and the demand comes from the error gate. The order demanded at the top of the error range, i.e. when the run is struggling most. As error rises, the demand moves from 'order_floor_adaptive_lo' to here. Values above 6 ask for more order than the previous fixed demand did."}),
                "order_floor_adaptive_r_hi": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_floor_adaptive_r_hi", 0.3031)),
                    "min": 0.06, "max": 2.0, "step": 0.01,
                    "tooltip": "Only used when 'order_floor' is 0 (automatic), 'order_floor_adaptive_graded' is on and the demand comes from the error gate. The error-over-tolerance level at which the demand reaches 'order_floor_adaptive_hi'. Smaller values reach full demand sooner; larger values keep the ramp gentle for longer. Must stay above 'order_floor_adaptive_rtol', otherwise the ramp has no width and the demand jumps straight to the high value."}),

                # ---- THE KREA2 TEST SET, APPENDED LAST (rule 35) so every existing workflow's
                # POSITIONAL widget values keep their slots and the new ones simply take their defaults.
                # Each is here because a measured finding named it; the why is in each tooltip and in
                # DEAD_FEATURES.md section KREA2 DEEP DIVE.
                "ord_ema": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("ord_ema", 0.37)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How quickly the sampler forgets its recent estimate of which order predicts the image best. 0 uses only the newest step, 1 never updates, and values in between smooth the estimate: higher is steadier, lower reacts faster. Only used while the per-pixel order ladder is on ('order_pixel_adaptive')."}),
                "order_max": ("INT", {
                    "default": int(ENGINE_DEFAULTS.get("order_max", 3)),
                    "min": 0, "max": 8, "step": 1,
                    "tooltip": "The highest number of previous steps the per-pixel order ladder may use. Higher values can sharpen detail but may overshoot in fast-changing areas; lower values are flatter and safer. 0 means the default ceiling of 10, not zero order."}),
                "order_elev_gain": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_elev_gain", 0.1)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How easily the order ladder raises a pixel's order. The ladder raises an order when the next one predicts clearly better; higher values make it demand less improvement and so raise more readily. Only used when 'order_by_bound' is off; with it on, the validity bound chooses the order and this setting is ignored."}),
                "order_demote_gain": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_demote_gain", 0.25)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How hard it is for the order ladder to drop a pixel's order. The ladder drops an order when the current one predicts clearly worse; higher values demand a bigger difference and so drop less readily. Only used when 'order_by_bound' is off; with it on, the validity bound chooses the order and this setting is ignored."}),
                "guard_ratio": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_ratio", 3.0)),
                    "min": 0, "max": 12, "step": 0.01,
                    "tooltip": "The guard holds back a step that moves far more than recent steps. Besides the step-to-step change test, the guard also flags a step when its movement is more than this many times the recent typical level. Lower values make the guard fire more easily; higher values make it rarer."}),
                "guard_sensitivity": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_sensitivity", 2.5)),
                    "min": 0, "max": 10, "step": 0.01,
                    "tooltip": "The guard's main sensitivity. A step's change is compared with the normal change over recent steps; higher values require a bigger outlier before the guard acts, lower values make it more nervous."}),
                "an_value": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("an_value", 2.5)),
                    "min": 0, "max": 10, "step": 0.01,
                    "tooltip": "The anomaly detector's value test: how strongly a region must stand out from its surroundings in value (roughly, brightness) before that region can be treated as an anomaly. Lower values catch more regions, higher values only stronger ones. The detector starts partway into the run; 'an_from' sets where."}),
                "an_from": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("an_from", 0.15)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How far into the run the anomaly detector starts, as a fraction of the total steps. 0.15 means roughly the first 15% of steps are ignored. Higher values delay detection; lower values let it catch problems earlier. This is a fraction of the run, not a noise level."}),
                "wmax_adapt": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("wmax_adapt", False)),
                    "tooltip": "Lets the cap on extrapolation order fall while a step's error is high. Off: the cap stays at its fixed maximum. On: a step whose previous error was well over tolerance gets a lower cap, so the sampler takes fewer high-order risks while it is struggling. This is the cautious direction; leave it off unless you are testing overshoot."}),
                "wmax_adapt_k": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("wmax_adapt_k", 0.5)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "Only used when 'wmax_adapt' is on. How far the cap may fall when the previous error is high. Higher values allow a larger drop; lower values keep the cap closer to its fixed maximum."}),
                "corrector_ab": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("corrector_ab", True)),
                    "tooltip": "Chooses the corrector. On: the current corrector, which may blend back toward the plain step when the correction looks risky. Off: an older corrector that always takes the corrected step and uses the raw prediction. Off exists for comparison; leave it on unless you are reproducing an old run."}),
                "rtol": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("rtol", 0.01)),
                    "min": 0, "max": 4, "step": 0.01,
                    "tooltip": "How much error the sampler accepts on a step. Lower is stricter: it falls back to the plain step more often, which is safer but can lose detail. Higher lets it take the more ambitious correction more often, which can add detail but risks overshoot. Other error checks compare against this value."}),
                # ---- THE GUARD FAMILY AND THE TWO GATES, APPENDED LAST (rule 35).  The guard halves the
                # krea2 tail error when it fires; the two gates below are what make the order-gain widgets
                # and the whole per-pixel order machinery testable at all.  Why: DEAD_FEATURES.md.
                "guard_window": ("INT", {
                    "default": int(ENGINE_DEFAULTS.get("guard_window", 8)),
                    "min": 0, "max": 32, "step": 1,
                    "tooltip": "How many recent steps the guard remembers when judging whether the current step is an outlier. A longer window is steadier; a short one reacts faster but can be fooled by one unusual step. Values below 4 are treated as 4."}),
                "guard_quantile": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_quantile", 0.99)),
                    "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Which level of the current step's movement the guard judges. The guard reduces the step's per-pixel movement to one number, and this chooses that number as a quantile: 0.99 means almost the most extreme pixel, 0.5 the median. High values make the guard react only to the worst-moving pixels; lower values make it react to more of the image. This is not a count of past steps."}),
                "guard_floor": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_floor", 0.1)),
                    "min": 0, "max": 4, "step": 0.001,
                    "tooltip": "The smallest threshold the guard will use, as a fraction of the recent typical movement. It stops a very quiet stretch from making the guard fire on tiny changes. Raise it to make the guard calmer in quiet runs."}),
                "guard_hold": ("INT", {
                    "default": int(ENGINE_DEFAULTS.get("guard_hold", 2)),
                    "min": 0, "max": 8, "step": 1,
                    "tooltip": "How many steps the guard stays active after it fires before it can change its mind. 1 lets it react every step; larger values give a slower, smoother response."}),
                "guard_mad_floor": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("guard_mad_floor", 0.1)),
                    "min": 0, "max": 4, "step": 0.001,
                    "tooltip": "The smallest spread the guard assumes when measuring how unusual a step is, as a fraction of the recent typical movement. It stops the guard becoming hypersensitive when recent steps barely change. Raise it to make the guard calmer in very steady runs."}),
                "order_by_bound": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_by_bound", True)),
                    "tooltip": "How the per-pixel order is chosen. On: the highest order whose own error estimate still fits the step tolerance (the validity bound). Off: the order whose prediction comes closest to the final image. On is more permissive in smooth areas and stricter in rough ones. While it is on, 'order_elev_gain' and 'order_demote_gain' are not used."}),
                "order_pixel_adaptive": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_pixel_adaptive", True)),
                    "tooltip": "Turns the per-pixel order ladder on or off. On: each pixel gets its own order, so detailed and smooth areas can be treated differently. Off: one order for the whole image, which is simpler and cheaper. This is the main switch for the order settings on this node."}),
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
                    "tooltip": "Adds a second test before an order is allowed. Off: an order is allowed when its largest extrapolation weight is under the cap. On: its own truncation error at this step's spacing must also fit the step tolerance, so a smoothness argument can refuse an order the weight cap would allow. Off is the safer default; on can change the image noticeably, and the direction depends on the model."}),
                # APPENDED LAST (rule 15).  The dial the operator asked for after seeing a
                # real ON/OFF difference on a krea checkpoint and being unable to call it.
                "order_bound_blend": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("order_bound_blend", 1.0)),
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used when 'order_by_bound' is on. Blends the two order choices: 0 uses the order that best predicts the image, 1 uses the validity bound, and values in between mix them. 0.5 means a pixel whose two answers are order 1 and order 4 runs at order 2. Adjust it when both extremes look plausible but neither is clearly better."}),
                # APPENDED LAST (rule 15): the bound's own map as the request.
                "order_demand_bound": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_demand_bound", True)),
                    "tooltip": "Only used when 'order_floor' is 0 (automatic) and 'order_by_bound' is on. On: on steps where the validity bound was computed, the sampler adds no extra demand, because the bound already sets a safe order for each pixel. Off: the error-gate demand applies everywhere. Setting 'order_demand_q' above 0 overrides this."}),
                # APPENDED LAST (rule 15): the per-pixel history-consistency gate A/B.
                "hist_cos_per_pixel": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("hist_cos_per_pixel", True)),
                    "tooltip": "Controls how a change of direction affects the per-pixel order ladder. On: only pixels that actually reverse direction drop to order 1, so one busy corner does not flatten the whole image. Off: a single reversal anywhere drops every pixel to order 1. On gives more order and detail in mixed areas and can cost some accuracy. Only used while 'order_pixel_adaptive' is on; with it off, the whole image is still gated as one."}),
                # APPENDED LAST (rule 15): the per-pixel gate on the second-order term A/B.
                "second_order_per_pixel": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("second_order_per_pixel", True)),
                    "tooltip": "Controls where the second-order correction is applied. On: only to pixels whose own order is 2 or higher, leaving volatile pixels on the safer first-order step. Off: to every pixel whenever the step's order cap is 2 or higher. On is more careful in rough areas."}),
                # APPENDED LAST (rule 15): the raw-lambda slope A/B.
                "second_order_raw_lambda": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("second_order_raw_lambda", True)),
                    "tooltip": "Controls the value used inside the second-order correction. On: use the unclamped estimate, so the correction does not carry a leftover bias from the clamp. Off: use the clamped value. The difference is small when the clamp is not active and can slightly reduce step error where it is."}),

                # ==================================================================================
                # THE 2026-10-19 PER-STEP AUDIT'S A/B ARMS -- APPENDED LAST (rule 15), ALL OFF.
                # These are EXPERIMENTS, not finished settings.  Each one is OFF by default, and
                # with all eight OFF this node emits exactly what it emitted before.  To test one,
                # connect this node to the Sampler's `options+`, turn ONE of them ON, and re-run
                # the same prompt, seed and schedule as the OFF run.  Each arm writes the value it
                # put in force to that run's own log, so the two runs are attributed from their
                # records rather than from a filename.  Findings:
                # findings/2026-10-19-per-step-audit/.
                # ==================================================================================
                "tol_step_du": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("tol_step_du", False)),
                    "tooltip": "Gives each step its own tolerance based on its length: a long step gets a looser tolerance, a short step a tighter one. Off: one tolerance for the whole run. It changes both the corrector's blend and the safety cut. Do not turn it on together with 'tol_from_curve'; if both are on, 'tol_from_curve' wins."}),
                "hist_cos_quantile": ("FLOAT", {
                    "default": float(ENGINE_DEFAULTS.get("hist_cos_quantile", 0.0)),
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Changes the direction-reversal test from a sign check to a quantile. 0 keeps the sign check: every pixel whose direction reverses is sent to order 1. Above 0, that fraction of the most-reversing pixels is sent to order 1, so 0.05 affects only the worst 5% instead of every pixel that reverses. Only has its per-pixel effect when 'hist_cos_per_pixel' is on; otherwise the whole image is gated as one."}),
                "probe_dk_per_side": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("probe_dk_per_side", False)),
                    "tooltip": "Only matters when 'probe_fuse_per_step' is on. Off: both measurements share one error scale, so they always allow the same order and the 'take the lower order' rule cannot fire. On: each measurement carries its own scale, so the two can disagree and the lower order is taken. Off is the safer default."}),
                "dk_series_step": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("dk_series_step", False)),
                    "tooltip": "How the order test's error scale is chosen. Off: use one run-wide median measured by the Autotuner. On: use the Autotuner's value at this step's own noise level, falling back to the median when that value is missing. On adapts to schedules whose steps differ a lot. It only changes the order test, so it does nothing unless 'wmax_bound' is on."}),
                "second_order_jump_mask": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("second_order_jump_mask", False)),
                    "tooltip": "Only used when 'second_order_per_pixel' is on. On: apply the correction only to pixels in the calmer half of this step's movement, keeping it away from sharp changes. Off: apply it to every pixel that qualifies by order. On narrows where the correction acts."}),
                "order_floor_norm": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("order_floor_norm", False)),
                    "tooltip": "Only used when 'order_floor' is 0 (automatic). Off: the gate compares a step's error-over-tolerance with the fixed threshold. On: it compares against what the Autotuner measured as normal for this part of the run, so the gate opens on steps that are worse than expected rather than on steps that merely come early in a decaying run. Falls back to the fixed test when the Autotuner has no curve."}),
                "tol_from_curve": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("tol_from_curve", False)),
                    "tooltip": "Takes each step's tolerance from the Autotuner's measured error curve instead of using one run-wide tolerance. It acts on the same two checks as 'tol_step_du'; if both are on, 'tol_from_curve' wins. Falls back to the run tolerance when the curve is missing or too short."}),
                "fuse_step_h": ("BOOLEAN", {
                    "default": bool(ENGINE_DEFAULTS.get("fuse_step_h", False)),
                    "tooltip": "Only used when 'probe_fuse_per_step' is on. Off: the combination uses the run's average step spacing. On: it uses each step's own spacing, which matters when the end of the schedule has much longer steps than the start. A uniform schedule behaves the same either way."}),
            },
            # CHAINING (2026-10-04).  Until this input existed only the Sampler could RECEIVE
            # options, so two options-producing nodes could not both drive a run -- the operator's
            # own report: *"i check the debug node, but it doesn't have options+ input -> i can't
            # have multiple chained options (scheduler+debugoptions) this way."*  It is OPTIONAL:
            # with nothing connected this node behaves exactly as before.
            "optional": {
                "options+": (OPT_TYPE, {"tooltip": "Connect another PD-AOS Options node here (the Autotuner's options output or another Debug Options node). Its settings pass through, and this node overrides them where both set the same field. The node closest to the sampler wins."}),
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
                tol_step_du=bool(ENGINE_DEFAULTS.get("tol_step_du", False)),
                hist_cos_quantile=float(ENGINE_DEFAULTS.get("hist_cos_quantile", 0.0)),
                probe_dk_per_side=bool(ENGINE_DEFAULTS.get("probe_dk_per_side", False)),
                dk_series_step=bool(ENGINE_DEFAULTS.get("dk_series_step", False)),
                second_order_jump_mask=bool(
                    ENGINE_DEFAULTS.get("second_order_jump_mask", False)),
                order_floor_norm=bool(ENGINE_DEFAULTS.get("order_floor_norm", False)),
                tol_from_curve=bool(ENGINE_DEFAULTS.get("tol_from_curve", False)),
                fuse_step_h=bool(ENGINE_DEFAULTS.get("fuse_step_h", False)),
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
            # THE 2026-10-19 AUDIT ARMS.  Widget, parameter and emission are the three ends
            # that must all exist (rule 14); each is OFF by default, so a graph that never
            # sets it emits the shipped value.  The connectivity rigs census them in both
            # directions, and each arm stamps its value in force in the run log.
            "tol_step_du": _b(tol_step_du),
            "hist_cos_quantile": max(0.0, min(1.0, _f(hist_cos_quantile))),
            "probe_dk_per_side": _b(probe_dk_per_side),
            "dk_series_step": _b(dk_series_step),
            "second_order_jump_mask": _b(second_order_jump_mask),
            "order_floor_norm": _b(order_floor_norm),
            "tol_from_curve": _b(tol_from_curve),
            "fuse_step_h": _b(fuse_step_h),
            "order_floor_adaptive_r_hi": max(0.06, min(2.0, _f(order_floor_adaptive_r_hi))),
        })
        return (out,)


class ResearchAFlopsNoiseOptions:
    DESCRIPTION = (
        "PD-AOS Noise (ETA) Options: all of the sampler's ancestral-noise controls in one "
        "node. Use it to add random noise back during sampling, for example to help a run "
        "whose image looks stuck or flat. 'eta' sets the maximum amount, and 0 turns noise "
        "off; 'eta_mode' chooses where the noise is applied. Connect this node's 'options' "
        "output to the Sampler's 'options+' input, directly or through another Options "
        "node; the node closest to the Sampler wins. When eta is above 0, connecting this "
        "node also opens the noise window and supplies the probe settings the stall "
        "detector needs, so it works without an Autotuner.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much random noise is added back at each step, on the same scale as other samplers' eta: 0 is fully deterministic, 0.1 is a typical gentle setting. It is the ceiling, not always the amount used: the mode decides where and how much is applied. With 'adaptive' the amount is scaled by the stall detector and never reaches eta; the other modes use it across the run. It also caps the level when 'eta_from_probe' is on."}),
                "eta_mode": (["adaptive", "constant", "cosine", "linear",
                              "exponential", "sigma_power"], {
                    "default": "adaptive",
                    "tooltip": "Where the noise goes. 'adaptive': only on steps where the sampler detects a stall, so it acts as an escape and needs a working measurement from the Autotuner. 'constant': the full eta on every step. 'cosine' and 'linear': fade from eta to zero over the run. 'exponential' and 'sigma_power': fade according to the noise level instead. Constant is the harshest; the fading modes keep noise out of the endgame."}),
                "s_tmin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                    "tooltip": "The lowest noise level that may receive noise. Noise is added only while sigma is at or above this value, so raising it stops noise in the late, low-noise part of the run where detail is being locked in. 0 means no lower limit. This is the lower end of the window, not the amount of noise."}),
                "s_tmax": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                    "tooltip": "The highest noise level that may receive noise. Noise is added only while sigma is at or below this value, so setting it above 0 keeps noise out of the early, high-noise part of the run where composition is decided. 0 means no upper limit."}),
                "escape": (["auto", "off"], {
                    "default": "auto",
                    "tooltip": "Whether the stall detector may add its escape nudge. 'auto' derives the firing thresholds from the Autotuner's measurement of your model, so they fit the model in front of you. 'off' disables the detector without changing the noise level. If another Options node sets either threshold explicitly, that value is used and this switch does not override it."}),
                "escape_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Scales the escape's sensitivity by multiplying the thresholds derived in 'auto' mode. 1.0 uses them as measured; above 1 fires more readily, so more steps get noise; below 1 is more reluctant; 0 disables the firing. Only used when 'escape' is 'auto'."}),
                "fade_floor": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much of the base noise level survives the late fade. The base level is normally 0, so this does nothing on its own; it only matters if a custom Options node sets a base level above 0. It does not fade the stall escape: a stuck run still gets its nudge at the end."}),
                "fade_sigma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.001,
                    "tooltip": "Where the late fade begins, in noise level. -1 means automatic: use the point where the Autotuner measured your model's trajectory settling. Set a positive value to choose the point yourself. It also marks the noise level above which a change of direction is not treated as a stall."}),
                "noise_survival_probe": (["follow eta", "force on", "force off"], {
                    "default": "follow eta",
                    "tooltip": "Whether the Autotuner also measures how much of a small kick the model still corrects at a given noise level. 'follow eta' measures it whenever eta is above 0, which is when the budget matters. 'force off' skips it and saves some model calls on the first run. 'force on' measures it even at eta 0, for inspecting the curve."}),
            },
            # CHAINING, same convention as the Debug Options node: a node closest
            # to the Sampler wins where two of them set the same field.
            "optional": {
                "options+": (OPT_TYPE, {"tooltip": "Connect another PD-AOS Options node here (the Autotuner's options output or the Debug Options node). Its settings pass through, and this node overrides them where both set the same field. The node closest to the sampler wins. Anything connected here also supplies the probe settings, which this node fills in only when nothing else did."}),
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
        "PD-AOS Experimental Options: switches for testing one change at a time. Every "
        "switch defaults to the normal behaviour, so connecting this node changes nothing "
        "until you set one. Connect its 'options' output to the Sampler's 'options+' input, "
        "or chain it through another Options node; the node closest to the Sampler wins "
        "where two set the same field. These switches are experimental and may change or "
        "disappear between releases; use the stable Options nodes for settings you want to "
        "keep.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "escape_mode": (["flatness", "momentum"], {
                    "default": "flatness",
                    "tooltip": "Which stall test can trigger the escape noise. 'flatness' is the original test: movement has collapsed and the image is still flat. 'momentum' adds a second test: movement has fallen to about 70% or less of its recent normal while detail has not grown by more than a small allowance, and it also requires a flat image when 'escape_momentum_flatness' is on. The momentum test does not replace the flatness test; either can fire."}),
                "noise_survival_curve": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Records how much of a small kick the model still corrects at several noise levels. It changes nothing by itself; it is the measurement 'eta_from_probe' and 'eta_budget_from_probe' read, and either of those turns it on automatically. Recording it costs extra probe calls on the first run for a model, not per step."}),
                "eta_from_probe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Derives the noise level from the Autotuner's measurement of how much noise your model can absorb at each noise level, instead of the level the chosen mode would use. Off is the normal behaviour. On applies to every mode; your 'eta' still acts as a ceiling. If the Autotuner has no measurement for this model, the sampler uses the normal behaviour. The measurement is recorded automatically when this is on."}),
                "eta_budget_from_probe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only used when 'eta_from_probe' is on. Off: the budget comes from 'eta_probe_budget'. On: it comes from the Autotuner's own measurement of how large a kick your model still corrects. On replaces a fixed number with a measured one, but it does not find a best-looking setting; if the image looks under-noised, set the budget yourself."}),
                "eta_probe_budget": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Only used when 'eta_from_probe' is on and 'eta_budget_from_probe' is off. How much of a kick may still show in the model's next prediction, as a fraction of an undisturbed image. Lower means less noise and a tighter schedule; higher means more. This is the one number that shapes the derived noise level."}),
                "escape_momentum_flatness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Only used when 'escape_mode' is 'momentum'. On: the momentum test also requires the image to be flat before it fires, the rule the original flatness test always had. Keep it on unless you deliberately want the older behaviour, which can fire on pixel art and other images whose fine detail has stopped growing but is not stalled."}),
                "escape_growth_probe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only used when 'escape_mode' is 'momentum'. Off: the momentum test uses a fixed allowance for detail growth. On: that allowance comes from the Autotuner's measurement of how consistently your model's trajectory moves in one direction; a model that keeps reversing gets a looser allowance, while a model that commits keeps the fixed one. Off is the normal behaviour."}),
            },
            # CHAINING, same convention as the other Options nodes: a node closest
            # to the Sampler wins where two of them set the same field.
            "optional": {
                "options+": (OPT_TYPE, {"tooltip": "Connect another PD-AOS Options node here (the Autotuner's options output, the Noise node or the Debug Options node). Its settings pass through, and this node overrides them where both set the same field. The node closest to the sampler wins."}),
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
    DESCRIPTION = ("A-FloPS A-Euler: the paper's Algorithm 2 (Jin et al., AAAI 2026, "
                   "arXiv:2509.00036) applied directly to a native flow model, with no "
                   "extra machinery. It splits the model's velocity into a straight-line "
                   "part with an adaptive slope and a correction, solves the straight-line "
                   "part exactly, then extrapolates the correction from the previous step; "
                   "the first step is plain Euler. The adaptive slope is used unclamped: "
                   "neither Algorithm 2 nor the authors' reference clamps it, and this node "
                   "passes no clamp. One model call per step. It works with any sigma list, "
                   "so connect it to SamplerCustomAdvanced and drive it with any scheduler; "
                   "for the paper's exact uniform-flow-time placement, use the companion "
                   "'A-FloPS A-Euler Schedule (paper)' node as the sigmas source.")
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
    DESCRIPTION = ("The paper's own schedule for A-FloPS A-Euler: sigma values at uniform "
                   "flow time, from noise to clean, mapped through the model's own time "
                   "scaling. Flow time is converted to the model's native timestep using "
                   "its multiplier for SD3/3.5, Wan and LTXV (and ModelSamplingAV), its "
                   "time shift for Flux, and t/(1-t) for Cosmos/RFlow. Connect this node's "
                   "'sigmas' output to SamplerCustomAdvanced or any sampler's sigmas input. "
                   "This is the schedule Algorithm 2 assumes, but the sampler also accepts "
                   "any other scheduler's sigmas. On Cosmos-style t/(1-t) schedules the "
                   "noise end is compressed, so the first step is very large and you may "
                   "need more steps than 'simple' to match its quality.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Connect the model the run will use. Its own noise-to-time mapping defines the flow-time grid this schedule is built on."}),
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
    DESCRIPTION = ("PD-AOS Autotuner: measures the model and this prompt with small test "
                   "runs, then moves your scheduler's steps toward the parts of the run "
                   "that measured as hardest. Connect the guider the run will use and any "
                   "scheduler's sigmas; the tuned sigmas go to your sampler, and the "
                   "'options' output goes to the PD-AOS Sampler's 'options+' input. The "
                   "model measurement is reused for every prompt with the same model and "
                   "settings; the prompt measurement runs on each generation. 'bias' and "
                   "'warp' both start at 1.0, which is neutral: bias tilts where the "
                   "measured placement puts steps, and warp moves the noise levels "
                   "themselves. If the Autotuner cannot measure anything useful, your sigma "
                   "list is passed through unchanged.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER", {"tooltip": "Connect the same guider your run uses, so the Autotuner measures the model and this prompt with the settings the run will actually apply. It does this before sampling starts."}),
                "sigmas": ("SIGMAS", {"tooltip": "Connect any scheduler's output (for example BasicScheduler with 'simple' or Karras). This is the schedule the Autotuner fine-tunes. If it cannot measure anything usable, the list is passed through unchanged, so your scheduler is always the fallback."}),
                "probe_steps": ("INT", {"default": 10, "min": 5, "max": 16, "step": 1,
                    "tooltip": "How many steps the model probe uses, from 5 to 16. The model probe decides how the schedule is fine-tuned. Its result is reused for every prompt with the same model and settings, so it costs extra time only on the first run."}),
                "probe_resolution": ("INT", {"default": 48, "min": 16, "max": 128, "step": 8,
                    "tooltip": "The image size the model probe runs at, from 16 to 128. Larger sizes measure more finely but are slower. 48 is the usual value."}),
                "cond_probe_steps": ("INT", {"default": 4, "min": 3, "max": 8, "step": 1,
                    "tooltip": "How many steps the prompt probe uses, from 3 to 8. This probe runs again for every new prompt, so this is the setting that affects each generation. Fewer steps are faster and measure less."}),
                "cond_probe_resolution": ("INT", {"default": 16, "min": 8, "max": 32, "step": 8,
                    "tooltip": "The image size the prompt probe runs at, from 8 to 32. 16 is the usual value; 8 is fastest and 32 measures most finely."}),
                "bias": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Tilts where the steps are placed. 1.0 is neutral. Below 1.0 puts more steps in the late, low-noise part where detail forms; above 1.0 puts more in the early, high-noise part and leaves a larger final drop. The first and last steps never move. With 'probe_schedule' off, a value other than 1.0 still turns the measured placement on."}),
                "warp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Moves the noise levels themselves, keeping the first and last where they are. 1.0 keeps your schedule exactly. Below 1.0 reaches low noise sooner and spends more of the run there; above 1.0 does the opposite. Ignored when 'auto_warp' is on. Large values can take the run outside the range the Autotuner measured."}),
                "auto_warp": ("BOOLEAN", {"default": False,
                    "tooltip": "Chooses the warp for you. On: 'warp' is ignored and the Autotuner uses the strongest warp that keeps every step inside the range of noise levels it measured. Off: your 'warp' value is used as set."}),
                # APPENDED LAST (rule 15).  ComfyUI stores a saved workflow's widget values as a
                # POSITIONAL list, so a widget added anywhere but the END re-maps the values of the
                # widgets after it -- silently, and onto plausible numbers.
                "probe_smin_frac": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "How far down the noise range the probes measure, as a fraction of the starting noise. 0.10 stops at a tenth of the start; lower values measure closer to zero noise. This is a measurement setting: it does not move any steps by itself. Measuring deeper costs extra probe calls."}),
                # APPENDED LAST (rule 15).
                "probe_schedule": ("BOOLEAN", {"default": False,
                    "tooltip": "On: the probes' measured curvature decides where your steps go, so hard stretches get more steps and easy stretches get fewer, and 'bias' tilts that placement. Off: your scheduler's list is passed through unchanged and 'bias' does nothing. On changes the noise levels themselves, so the image will differ from off."}),
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
    DESCRIPTION = ("Shows what the sampler decided at every step of the last run, as JSON: "
                   "the solver used, step errors, guard trips, anomaly detections and "
                   "model-call counts. The sampler records up to 64 steps; for longer runs, "
                   "turning on the Sampler's 'log_errors' keeps the latest 64 instead of "
                   "the first 64. Connect the sampled latent to 'trigger' so this node runs "
                   "after sampling finishes. It updates when the graph re-runs.")
    CATEGORY = "custom_sampling/research_samplers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"trigger": ("LATENT", {"tooltip": "Connect the sampled latent here so this node runs after sampling finishes and reports the current run."})},
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
