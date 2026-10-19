import math
import logging
import hashlib
import functools
import os
import statistics
import time
import torch
import torch.nn.functional as F
import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler
try:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
except Exception:
    pass
try:
    torch.backends.cudnn.benchmark = False
except Exception:
    pass
try:
    torch.backends.cudnn.deterministic = True
except Exception:
    pass
_AFLOPS_TF32_OFF = os.environ.get("AFLOPS_TF32_OFF", "1") == "1"
if _AFLOPS_TF32_OFF:
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except Exception:
        pass
# ---- attention-backend determinism ----------------------------------------
# The flags above cover cudnn, cublas and matmul reductions, but NOT the
# attention path -- and that is where the remaining non-determinism lives.
# PyTorch's MEMORY-EFFICIENT SDPA backend accumulates with atomics and is
# documented as non-deterministic in the FORWARD pass; the flash and math
# backends are deterministic.
#
# The symptom this targets, measured: two Anima/CFG5 runs with an IDENTICAL
# selected profile and an identical step-1 input still disagree at the FIRST
# model call -- x0_rms reads 1.449564 / 0.657451 / 0.592164 / 0.598357 across
# logs 35-42, which all share profile hash d411cc1f0e -- so the divergence is in
# the forward pass, not in the stepping arithmetic.  Krea2/CFG1 is
# bit-reproducible by contrast (logs 71 == 73, 0 of 16 steps differ).
#
# AFLOPS_DET_SDPA=0 restores the default backends: faster, but not
# reproducible.  Like the block above this is a GLOBAL torch setting and affects
# the whole ComfyUI process; the tradeoff is attention speed, not correctness.
if os.environ.get("AFLOPS_DET_SDPA", "1") == "1":
    for _fn in ("enable_mem_efficient_sdp", "enable_cudnn_sdp"):
        try:
            getattr(torch.backends.cuda, _fn)(False)
        except Exception:
            pass
try:
    from .common import exact_step as _exact_step_ext
except Exception:
    _exact_step_ext = None
_LAST_ERRORS = []
_LAST_CFG = {}
# WHICH BRANCH OF `_model_cache_key` PRODUCED THIS RUN'S KEY.  Stamped into `_LAST_CFG` so a corpus
# query can EXCLUDE runs whose key could not distinguish two models -- without this, a degraded key is
# invisible in exactly the way the `wc:` collision was.  Values: "model_hash", "param_digest",
# "width_code_DEGRADED", "model_id_DEGRADED".
_MODEL_KEY_KIND = None
# THE PATCH IDENTITY OF THE MODEL KEY (LoRA / weight wrappers / hooks).  Stamped into
# `_LAST_CFG` beside `_model_key_kind`; see `_patcher_patch_tag`.
_MODEL_PATCH_TAG = "-"

# ---------------------------------------------------------------------------
# PER-RUN PHASE TIMING.
#
# WHY THIS EXISTS.  PD-AOS logged NO timing at all -- zero `perf_counter`, zero
# `elapsed`, zero `duration_ms` anywhere -- so "the autotuner probes feel
# extremely slow since a while" could not be checked from any artifact this
# project produces.  It was checked from ComfyUI's own log instead, and that
# dated a REAL regression: the per-queue pre-sampling window went from 2.16 s on
# build `fccd7e2a8a18` to 13.2-15.6 s on build `bf9dbade0f8a` (and stayed there
# through `2247cad14dfc`), while the per-step sampler cost did not move at all.
# ComfyUI's clock can say WHICH span grew -- the span following "Model ...
# prepared for dynamic VRAM loading" -- but it cannot say what ran inside it,
# because that span is silent in both logs.
#
# So the engine now states its own phases.  Milliseconds, per run, stamped into
# `aflops_cfg._phase_ms` (a TOP-LEVEL key of `aflops_cfg`, alongside `run_id` and
# `ab_reach` -- the whole object is serialised by nodes.py, so no whitelist has to
# be updated) so a corpus query can read them.  Deliberately cheap and
# exception-proof: timing must never be able to fail a run.
#
# THE DICT IS REPLACED, NOT CLEARED, and the log holds a REFERENCE to it.  The
# run's report dict is built BEFORE the sampling loop (its `ab_reach` entry is a
# live reference for the same reason), so a snapshot taken at that point would
# contain none of the sampling phases -- measured: `pre_sampling_ms` was missing
# from the stamped record while present in `_PHASE_MS`.  Replacing the dict per
# run gives each run an object of its own that nothing later mutates, so a log
# always describes the run that produced it.
# ---------------------------------------------------------------------------
_PHASE_MS = {}

# PHASES RECORDED OUTSIDE AN ENGINE RUN.  The node-side probe runs in the
# Autotuner node BEFORE the sampler, and it is the single largest block of the
# operator's "53 seconds for the autotuner".  It cannot write into `_PHASE_MS`
# directly: at that moment `_PHASE_MS` is the PREVIOUS run's stamped dict, which
# the reporter still holds, so writing there would rewrite history.  It writes
# here instead, and the engine SEEDS its own record from this at entry -- so the
# node's time lands in the log of the run it preceded.
_NODE_PHASE_MS = {}

# The process-wide probe-call counter at the previous engine entry.  The DIFF is
# the node-side probe's forward passes for this queue: the NFE the node spent
# before the sampler ever ran, which no per-run delta can see.
_LAST_ENGINE_CALLS = [0]


def _nphase_set(key, value):
    try:
        _NODE_PHASE_MS[key] = value
    except Exception:
        pass


def _nphase_add(key, t_start):
    if t_start is None:
        return
    try:
        _NODE_PHASE_MS[key] = _NODE_PHASE_MS.get(key, 0.0) + \
            (time.perf_counter() - t_start) * 1e3
    except Exception:
        pass


def _t0():
    try:
        return time.perf_counter()
    except Exception:
        return None


def _phase_add(key, t_start):
    """Add one span to a phase accumulator."""
    if t_start is None:
        return
    try:
        _PHASE_MS[key] = _PHASE_MS.get(key, 0.0) + \
            (time.perf_counter() - t_start) * 1e3
    except Exception:
        pass


def _phase_set(key, value):
    try:
        _PHASE_MS[key] = value
    except Exception:
        pass


_PROBE_PROFILES = {}       # "<model_key>|s:<sched_hash>" -> profile
_PROFILE_BY_MODEL = {}     # "<model_key>" -> most recent profile (scheduler view)
_COND_PROBE_PROFILES = {}  # (model_key, cond_sig) -> per-prompt profile
_COND_PROBE_PROFILES_SCHED = {}
_NODE_PROBE_PROFILES = {}  # (model_key, param_sig, cfg_sig) -> node-side profile
_EXACT_STEP_STATUS = "unverified"


# ---------------------------------------------------------------------------
# Build provenance: ties a saved report / console log to the exact code that
# produced it.
#
# The live node is a COPY of the repo (it has no .git), so a commit hash cannot
# be read at runtime.  The primary key is therefore a hash of the SOURCE FILES
# themselves -- which also catches local edits that never became a commit --
# with the git commit and branch attached as a bonus whenever a repo IS
# reachable (i.e. when running from the workspace checkout).
#
# WARNING, AND IT IS UNGUARDED.  `_BUILD_SRC_FILES` is this file's own list.
# The cross-check tool that once recomputed the same hash from git BLOBS
# (`scratch_build_id_tool.py`) was deleted in the 2026-10-08 documentation
# purge, so nothing now verifies that this list matches what the package
# actually ships.  A path added, renamed or moved without an edit here fails
# SILENTLY and SHRINKS src_hash: see `_build_identity`, which records the
# misses in `src_missing` rather than raising.  Check that field is empty
# before quoting a src_hash.
# ---------------------------------------------------------------------------
_BUILD_SRC_FILES = (
    "__init__.py",
    "nodes.py",
    "research_samplers/aflops.py",
    "research_samplers/common.py",
    "research_samplers/paper_aeuler.py",
    "research_samplers/pd_aos_comfy.py",
)
# `ruff.toml` IS DELIBERATELY ABSENT FROM THE LIST ABOVE, and adding it would be a
# step backwards rather than more thorough.  src_hash answers one question -- "which
# code produced this run?" -- and it is never used as a cache key, only as the build
# identity that scopes runs for comparison.  The lint config cannot change what the
# code DOES, and no runtime code reads it, so folding it in would mint a NEW
# src_hash for behaviourally identical code every time a lint rule changes.  That
# splits runs which should be pooled, and it is the only one of the two errors that
# costs anything: the config is already tracked in git and the stamp carries
# git_commit, so the standard in force is recoverable from the stamp either way.


def _new_run_id():
    """A PER-RUN identifier, unique per invocation, safe to use as a FILENAME.

    WHY THIS EXISTS.  The corpus log and the saved latent are two files that share no key.  Every
    digest the log already carries is a CONFIG identity, not a RUN identity: `sigmas_digest` is the
    schedule, `probe_key`/`cond_probe_key` are the model+prompt, `src_hash` is the build.  Two runs
    that differ only in the RNG -- or two replicates of the same experimental arm -- share ALL of
    them, so a latent named from any of them collides.  That is not hypothetical: measured on build
    `5494efacea47`, anima `log_00187`/`00188`/`00189` are the same config and produce THREE
    different trajectories, so a config-derived name is wrong by construction.  The operator's
    request was to be able to name the latent from the log, which needs exactly this: an identifier
    that is a property of the RUN.

    FORMAT `<utc>Z-<8 hex>`, e.g. `20261003T095831-4f2a9c7e`.  Sortable by time so a directory
    listing reads chronologically, and safe on every filesystem (digits, `T`, `-` and hex only --
    deliberately no `:`, `/` or `|`, which is why the ISO-8601 colons are stripped).

    NOT DETERMINISTIC, ON PURPOSE.  The identifier names a run, and a re-run is a DIFFERENT run
    whose latent differs; a deterministic id would give two different latents the same name.  The
    random suffix is what separates two runs that land in the same second.

    THE LATENT LINK IS ALSO CHECKABLE WITHOUT THIS.  Inverting the latent format's own map
    (`Wan21.process_in`) recovers the tensor the engine held, and its final `x0_rms`/`x0_std` then
    match the log's last step record EXACTLY -- measured to 6 decimals on the operator's first
    saved latent against `krea2TurboOfficialComfy .../log_00016.json`.  So a latent can be paired to
    a log by arithmetic as well as by name, which is what makes the link survive a rename.
    """
    try:
        _ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    except Exception:
        _ts = "00000000T000000"
    try:
        _sfx = os.urandom(4).hex()
    except Exception:
        _sfx = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
    return "%s-%s" % (_ts, _sfx)


def _src_hash_from(items):
    """items: iterable of (relpath, bytes) in _BUILD_SRC_FILES order.

    Line endings are normalised, because git stores LF blobs while a Windows
    worktree (and the copied live node) may be CRLF -- without this, a hash of
    the live files would never match a hash of the committed blobs.
    """
    h = hashlib.sha256()
    for rel, data in items:
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(data.replace(b"\r\n", b"\n"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _git_dir_at(d):
    """The .git directory governing `d`, or None.  Mirrors git's own walk.

    `.git` is a DIRECTORY in an ordinary checkout and a FILE holding
    `gitdir: <path>` in a worktree or a submodule.  The live node ships without
    a .git at all, which is why every failure here is non-fatal.
    """
    g = os.path.join(d, ".git")
    if os.path.isdir(g):
        return g
    if os.path.isfile(g):
        try:
            with open(g, "r", encoding="utf-8") as fh:
                first = (fh.readline() or "").strip()
            if first.startswith("gitdir:"):
                return os.path.abspath(
                    os.path.join(d, first.split(":", 1)[1].strip()))
        except Exception:
            return None
    return None


def _resolve_head_ref(g, ref):
    """Commit pointed at by `ref` inside git dir `g`, loose ref preferred.

    TWO DIRECTORIES, NOT ONE.  In a worktree or a submodule, `g` is the admin
    directory (`.git/worktrees/<name>`), which holds that checkout's HEAD but
    NOT its refs: those live in the MAIN repo, named by `g/commondir`
    (normally `../..`).  Looking in `g` alone returns the branch name with no
    commit -- an empty stamp masquerading as a detached one.
    """
    dirs = [g]
    try:
        with open(os.path.join(g, "commondir"), "r", encoding="utf-8") as fh:
            common = (fh.read() or "").strip()
        if common:
            dirs.append(os.path.abspath(os.path.join(g, common)))
    except Exception:
        pass
    for base in dirs:
        try:
            with open(os.path.join(base, *ref.split("/")), "r",
                      encoding="utf-8") as fh:
                return (fh.read().strip() or None)
        except Exception:
            pass
        try:  # a ref may be packed instead of loose
            with open(os.path.join(base, "packed-refs"), "r",
                      encoding="utf-8") as fh:
                for line in fh:
                    if not line.startswith(("#", "^")) and \
                            line.strip().endswith(ref):
                        return line.split()[0]
        except Exception:
            pass
    return None


def _git_head_of(start_dir):
    """(commit12, branch) of the repo governing `start_dir` -- no subprocess.

    THE **NEAREST** `.git` WINS: this walks up from `start_dir` and stops at the
    first directory that has one.  Depth is deliberate and load-bearing.  The
    deciding question is always "which repo is this copy of the code IN", not
    "is there a repo somewhere above it":

      * package at a repo root   -> that repo, as before;
      * package nested in a repo that is itself a repo -> the INNER one.  A
        fixed depth would read the outer repo's HEAD and stamp every log with a
        commit that does not contain the running code -- silently, because a
        wrong hash looks exactly like a right one;
      * package inside a WORKTREE -> the worktree's own HEAD, through `.git`
        being a gitdir pointer file rather than a directory;
      * no repo anywhere (the shipped node) -> (None, None), and the caller
        falls back to src_hash, which needs no repo.
    """
    d = os.path.abspath(start_dir)
    while True:
        g = _git_dir_at(d)
        if g is not None:
            try:
                with open(os.path.join(g, "HEAD"), "r", encoding="utf-8") as fh:
                    head = fh.read().strip()
            except Exception:
                return None, None
            if not head.startswith("ref:"):
                return (head[:12] or None), "(detached)"
            ref = head.split(" ", 1)[1].strip()
            branch = ref.rsplit("/", 1)[-1]
            commit = _resolve_head_ref(g, ref)
            return (commit[:12] if commit else None), branch
        parent = os.path.dirname(d)
        if parent == d:
            return None, None
        d = parent


def _build_identity():
    """Provenance block stamped into every report (and logged once at import)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items = []
    missing = []
    for rel in _BUILD_SRC_FILES:
        try:
            with open(os.path.join(root, *rel.split("/")), "rb") as fh:
                items.append((rel, fh.read()))
        except Exception:
            missing.append(rel)
    commit, branch = _git_head_of(root)
    try:
        import datetime as _dt
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        ts = None
    return {
        "src_hash": _src_hash_from(items) if items else None,
        "src_files": [r for r, _ in items],
        "src_missing": missing,
        "git_commit": commit,
        "git_branch": branch,
        "stamped_utc": ts,
    }


_BUILD = _build_identity()
logging.info("[PD-AOS] build src=%s git=%s@%s files=%d%s",
             _BUILD.get("src_hash"), _BUILD.get("git_commit"),
             _BUILD.get("git_branch"), len(_BUILD.get("src_files") or []),
             (" missing=%s" % ",".join(_BUILD["src_missing"]))
             if _BUILD.get("src_missing") else "")

def _exact_step_local(x, x0, s, s2):
    r = float(s2) / max(float(s), 1e-12)
    return r * x + (1.0 - r) * x0

def _validate_exact_step():
    global exact_step, _EXACT_STEP_STATUS
    if _exact_step_ext is None:
        exact_step = _exact_step_local
        _EXACT_STEP_STATUS = "fallback: .common.exact_step unavailable"
        logging.warning("[PD-AOS] %s", _EXACT_STEP_STATUS)
        return
    try:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, 8, 8, generator=g).double()
        x0 = torch.randn(2, 3, 8, 8, generator=g).double()
        worst = 0.0
        for s, s2 in ((14.0, 7.0), (1.0, 0.3), (0.5, 0.0), (2.0, 1.999)):
            got = _exact_step_ext(x, x0, s, s2)
            want = _exact_step_local(x, x0, s, s2)
            if not torch.isfinite(got).all():
                raise ValueError("non-finite output for s=%s s2=%s" % (s, s2))
            err = float((got - want).abs().max() /
                        want.abs().max().clamp_min(1e-6))
            worst = max(worst, err)
        if worst <= 1e-4:
            exact_step = _exact_step_ext
            _EXACT_STEP_STATUS = ("ok (matches (s2/s)x+(1-s2/s)x0, "
                                  "worst rel dev %.2e)" % worst)
        else:
            exact_step = _exact_step_local
            _EXACT_STEP_STATUS = ("fallback: .common.exact_step deviates %.2e "
                                  "from (s2/s)x+(1-s2/s)x0" % worst)
            logging.warning("[PD-AOS] %s", _EXACT_STEP_STATUS)
    except Exception as e:
        exact_step = _exact_step_local
        _EXACT_STEP_STATUS = "fallback: validation error (%s)" % e
        logging.warning("[PD-AOS] %s", _EXACT_STEP_STATUS)

_validate_exact_step()

def _detect_space(sigmas, mode):
    if mode != "auto":
        return mode
    return "flow" if float(sigmas[0]) <= 1.5 else "ve_edm"

def _ancestral_split(s, sn, eta, space):
    if eta <= 0.0 or sn <= 1e-8:
        return sn, 0.0
    if space == "flow":
        tf = s / max(1.0 - s, 1e-6)
        tt = sn / max(1.0 - sn, 1e-6)
        up_t = eta * math.sqrt(max(tt * tt * (tf * tf - tt * tt) / (tf * tf), 0.0))
        down_t = math.sqrt(max(tt * tt - up_t * up_t, 0.0))
        s_down = down_t / (1.0 + down_t)
        k = down_t / tt
        s_up = sn * math.sqrt(max(1.0 - k * k, 0.0))
        return s_down, s_up
    else:
        up = min(sn, eta * math.sqrt(sn * sn * (s * s - sn * sn) / (s * s)))
        down = math.sqrt(max(sn * sn - up * up, 0.0))
        return down, up

_NOISE_COLOR_PRESETS = {
    "white": 0.0,
    "pink": 1.0,
    "red": 1.5,
    "brown": 2.0,
    "blue": -1.0,
    "violet": -2.0,
}

def _colored_noise(shape, beta, device, dtype, generator=None,
                   decay_floor=1e-3, temporal_beta=0.0, dc_zero=True):
    eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    if beta == 0.0 and temporal_beta == 0.0:
        return eps
    if len(shape) < 2 or shape[-1] < 2 or shape[-2] < 2:
        return eps
    H, W = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    fmag = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
    fmag = fmag.clamp_min(float(decay_floor))
    weight = fmag.pow(-float(beta) / 2.0)
    if dc_zero:
        weight = weight.clone()
        weight[0, 0] = 0.0
    weight_r = weight[:, :W // 2 + 1]
    shp = [1] * len(shape)
    shp[-2], shp[-1] = H, W // 2 + 1
    weight_r = weight_r.view(*shp).to(dtype=dtype)
    eps_fft = torch.fft.rfft2(eps, norm="ortho")
    eps_fft = eps_fft * weight_r
    out = torch.fft.irfft2(eps_fft, s=(H, W), norm="ortho")
    if temporal_beta != 0.0 and len(shape) == 5 and shape[2] > 2:
        T = shape[2]
        ft = torch.fft.fftfreq(T, device=device).clamp_min(float(decay_floor))
        wt = ft.pow(-float(temporal_beta) / 2.0)
        if dc_zero:
            wt = wt.clone()
            wt[0] = 0.0
        wt = wt.view(1, 1, -1, 1, 1)[:, :, :T // 2 + 1, :, :].to(dtype=dtype)
        out_fft = torch.fft.rfft(out, dim=2, norm="ortho")
        out_fft = out_fft * wt
        out = torch.fft.irfft(out_fft, n=T, dim=2, norm="ortho")
    lead = 1
    for d in shape[:-2]:
        lead *= d
    flat = out.reshape(lead, -1)
    std = flat.std(dim=-1, keepdim=True).clamp_min(1e-8)
    out = out / std.reshape(*shape[:-2], 1, 1)
    return out

def _spectral_concentration(beta, r_min=1.0 / 64.0, r_max=0.5):
    beta = float(beta)
    if abs(beta) < 1e-6:
        return 1.0

    def _radial_pow_int(exp):
        if abs(exp + 1.0) < 1e-9:
            return math.log(r_max / r_min)
        return (r_max ** (exp + 1.0) - r_min ** (exp + 1.0)) / (exp + 1.0)

    total = 2.0 * math.pi * _radial_pow_int(1.0 - beta)
    num_modes = math.pi * (r_max * r_max - r_min * r_min)
    mean_pm = total / max(num_modes, 1e-12)
    peak = r_min ** (-beta) if beta > 0 else r_max ** (-beta)
    return max(peak / max(mean_pm, 1e-12), 1.0)

class _ColorAwareNoiseSampler:

    def __init__(self, x, seed, noise_color="white", noise_beta=1.0,
                 color_schedule="constant", color_beta_max=1.5,
                 color_beta_min=-0.5, color_temporal=False,
                 color_temporal_beta=0.0, color_decay_floor=1e-3,
                 color_dc_zero=True):
        self.shape = x.shape
        self.device = x.device
        self.dtype = x.dtype
        self.color = str(noise_color)
        self.beta = float(noise_beta)
        self.schedule = str(color_schedule)
        self.beta_max = float(color_beta_max)
        self.beta_min = float(color_beta_min)
        self.temporal = bool(color_temporal)
        self.temporal_beta = float(color_temporal_beta)
        self.decay_floor = float(color_decay_floor)
        self.dc_zero = bool(color_dc_zero)
        self.last_beta = 0.0
        self.generator = torch.Generator(device=x.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.s_max = None
        self.s_min = None

    def _base_beta(self):
        if self.color == "custom":
            return self.beta
        return _NOISE_COLOR_PRESETS.get(self.color, 0.0)

    def _beta_at(self, sigma):
        b0 = self._base_beta()
        if self.schedule == "constant":
            return b0
        s_max = self.s_max if self.s_max is not None else max(sigma, 1e-8)
        s_min = self.s_min if self.s_min is not None else max(s_max * 1e-3, 1e-8)
        ls_hi = math.log(max(s_max, 1e-8))
        ls_lo = math.log(max(s_min, 1e-8))
        ls = math.log(max(sigma, 1e-8))
        frac = (ls_hi - ls) / max(ls_hi - ls_lo, 1e-8)
        frac = min(max(frac, 0.0), 1.0)
        if self.schedule == "fade_to_white":
            return b0 * (1.0 - frac)
        if self.schedule == "linear":
            return self.beta_max + (self.beta_min - self.beta_max) * frac
        if self.schedule == "cns_approx":
            if frac < 0.5:
                return b0 * 0.2 * (1.0 - frac / 0.5)
            t = (frac - 0.5) / 0.5
            return (b0 * 0.1) * (1.0 - t) + self.beta_min * t
        return b0

    def __call__(self, sigma, sigma_next, override_beta=None):
        sig = float(sigma) if sigma is not None else 1.0
        sig_n = float(sigma_next) if sigma_next is not None else 0.0
        if self.s_max is None:
            self.s_max = sig
        self.s_max = max(self.s_max, sig)
        if sig_n > 1e-8:
            self.s_min = min(self.s_min if self.s_min is not None else sig_n, sig_n)
        if self.s_min is None:
            self.s_min = max(sig_n, sig * 1e-3, 1e-8)
        if override_beta is not None:
            beta = float(override_beta)
        else:
            beta = self._beta_at(sig)
        self.last_beta = beta
        out = _colored_noise(
            self.shape, beta, self.device, self.dtype,
            generator=self.generator, decay_floor=self.decay_floor,
            temporal_beta=self.temporal_beta if self.temporal else 0.0,
            dc_zero=self.dc_zero)
        return out

def _make_noise_sampler(x, extra_args, cfg):
    color = str(cfg.get("noise_color", "white"))
    schedule = str(cfg.get("color_schedule", "constant"))
    needs_color = color != "white"
    if not needs_color and schedule != "constant":
        if schedule in ("linear", "cns_approx") and (
                float(cfg.get("color_beta_max", 0.0)) != 0.0 or
                float(cfg.get("color_beta_min", 0.0)) != 0.0):
            needs_color = True
    _seed = extra_args.get("seed", None)
    if _seed is None:
        _seed = 0
    if not needs_color:
        return default_noise_sampler(x, _seed)
    return _ColorAwareNoiseSampler(
        x, _seed,
        noise_color=color,
        noise_beta=float(cfg.get("noise_beta", 1.0)),
        color_schedule=schedule,
        color_beta_max=float(cfg.get("color_beta_max", 1.5)),
        color_beta_min=float(cfg.get("color_beta_min", -0.5)),
        color_temporal=bool(cfg.get("color_temporal", False)),
        color_temporal_beta=float(cfg.get("color_temporal_beta", 0.0)),
        color_decay_floor=float(cfg.get("color_decay_floor", 1e-3)),
        color_dc_zero=bool(cfg.get("color_dc_zero", True)),
    )

_CENTROID_BETA_LUT = None

def _build_centroid_lut():
    global _CENTROID_BETA_LUT
    r_min, r_max = 1.0 / 64.0, 0.5
    betas = []
    centroids = []
    for i in range(81):
        beta = -2.0 + i * 0.05
        if abs(beta - 2.0) < 1e-6:
            c = (r_max - r_min) / math.log(r_max / r_min)
        elif abs(beta - 3.0) < 1e-6:
            c = math.log(r_max / r_min) * r_min * r_max / (r_max - r_min)
        else:
            num = (r_max ** (3.0 - beta) - r_min ** (3.0 - beta)) / (3.0 - beta)
            den = (r_max ** (2.0 - beta) - r_min ** (2.0 - beta)) / (2.0 - beta)
            c = num / den if den > 1e-12 else 0.3
        betas.append(beta)
        centroids.append(c)
    _CENTROID_BETA_LUT = (centroids, betas)

def _centroid_to_beta(centroid):
    if _CENTROID_BETA_LUT is None:
        _build_centroid_lut()
    centroids, betas = _CENTROID_BETA_LUT
    c = float(centroid)
    if c >= centroids[0]:
        return -2.0
    if c <= centroids[-1]:
        return 2.0
    lo, hi = 0, len(centroids) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if centroids[mid] >= c:
            lo = mid
        else:
            hi = mid
    c_lo, c_hi = centroids[lo], centroids[hi]
    b_lo, b_hi = betas[lo], betas[hi]
    if abs(c_lo - c_hi) < 1e-8:
        return b_lo
    t = (c - c_lo) / (c_hi - c_lo)
    return b_lo + t * (b_hi - b_lo)

def _blackman_harris_window_1d(N, device=None, dtype=torch.float32):
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    n = torch.arange(N, device=device, dtype=dtype)
    c = (2.0 * math.pi / float(N)) * n
    return a0 - a1 * torch.cos(c) + a2 * torch.cos(2.0 * c) - a3 * torch.cos(3.0 * c)

def _spectral_centroid(jmap):
    if jmap is None:
        return None
    sh = jmap.shape
    if len(sh) < 3:
        return None
    H, W = sh[-2], sh[-1]
    if H < 4 or W < 4:
        return None
    j = jmap.reshape(-1, H, W).float()
    j = j - j.mean(dim=(-2, -1), keepdim=True)
    win_y = _blackman_harris_window_1d(H, device=j.device, dtype=j.dtype)
    win_x = _blackman_harris_window_1d(W, device=j.device, dtype=j.dtype)
    win_2d = (win_y.view(-1, 1) * win_x.view(1, -1)).unsqueeze(0)  # (1, H, W)
    j = j * win_2d
    fft = torch.fft.rfft2(j, norm="ortho")
    power = (fft.real ** 2 + fft.imag ** 2)
    power = power.mean(dim=0)
    fy = torch.fft.fftfreq(H, device=jmap.device)
    fx = torch.fft.rfftfreq(W, device=jmap.device)
    R = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
    w = torch.full((W // 2 + 1,), 2.0, device=jmap.device)
    w[0] = 1.0
    if W % 2 == 0:
        w[-1] = 1.0
    w = w.view(1, -1)
    pw = power * w
    total = pw.sum()
    if total < 1e-12:
        return 0.33
    centroid = float((R * pw).sum() / total)
    return centroid

def _whitepoint_from_x0s(x0s):
    """Spectral centroid of the mid-trajectory x0's detail map (the probe
    'whitepoint'): measures whether the model's natural output structure is
    blue- or red-heavy.  Shared by the model probe, the cond probe and the
    focused probe so EVERY profile carries it (the engine's model-probe
    fallback via _latest_profile may legitimately serve a per-prompt cond
    profile, and the adaptive-noise prior wants the whitepoint either way)."""
    try:
        if x0s:
            mid = len(x0s) // 2
            jm = x0s[mid].pow(2).mean(dim=1).sqrt()
            c = _spectral_centroid(jm)
            if c is not None:
                return float(c)
    except Exception:
        pass
    return 0.33

def _predict_adaptive_beta(jmap, err, eta_eff, s, space, sigma_data,
                           consistency, cfg, corr_used=None, fresh_frac=None):
    if jmap is None:
        return None
    centroid = _spectral_centroid(jmap)
    if centroid is None:
        return None
    beta_raw = min(_centroid_to_beta(centroid), 0.0)
    if fresh_frac is None:
        fresh_frac = 0.25 * float(eta_eff) * float(eta_eff)
    tau_eta = float(cfg.get("adaptive_tau_eta", 0.15))
    s_eta = math.exp(-max(float(fresh_frac), 0.0) / max(tau_eta, 1e-3))
    s_eta = max(0.0, min(s_eta, 1.0))
    tau_err = float(cfg.get("adaptive_tau_err", 0.5))
    err_norm = (err or 0.0) / max(float(s), 0.01)
    s_err = math.exp(-err_norm / max(tau_err, 1e-4))
    s_err = max(0.0, min(s_err, 1.0))
    snr = _compute_snr(float(s), space, sigma_data)
    log_snr = math.log(max(snr, 1e-8))
    tau_snr = float(cfg.get("adaptive_tau_snr", 2.0))
    s_snr = math.exp(-log_snr * log_snr / (2.0 * tau_snr * tau_snr))
    s_snr = max(0.0, min(s_snr, 1.0))
    s_cons = max(0.0, min(float(consistency), 1.0))
    gamma_cons = float(cfg.get("adaptive_cons_gamma", 1.5))
    s_cons = s_cons ** gamma_cons
    s_corr = 0.7
    safety = s_eta * s_err * s_snr * s_cons * s_corr
    user_strength = float(cfg.get("color_strength", 1.0))
    user_strength = max(0.0, min(user_strength, 1.0))
    beta = safety * user_strength * beta_raw
    beta_min = float(cfg.get("color_beta_min", -2.0))
    beta_max = float(cfg.get("color_beta_max", 2.0))
    beta = max(beta_min, min(beta, beta_max))
    return beta

def _lf_get(lf, name, default):
    v = getattr(lf, name, default)
    if callable(v):
        try:
            v = v()
        except Exception:
            return default
    return v if v is not None else default

def _extract_model_info(model):
    if model is None:
        return None
    try:
        patcher = model
        if hasattr(model, "inner_model"):
            inner = model.inner_model
            if hasattr(inner, "model_patcher"):
                patcher = inner.model_patcher
            elif hasattr(inner, "inner_model") and hasattr(inner.inner_model, "model_patcher"):
                patcher = inner.inner_model.model_patcher
        elif hasattr(model, "model_patcher"):
            patcher = model.model_patcher
        ms = patcher.get_model_object("model_sampling")
        lf = patcher.get_model_object("latent_format")
        mt = patcher.get_model_object("model_type")
        mc = patcher.get_model_object("model_config")
        uc = getattr(mc, "unet_config", {}) or {}
        try:
            import comfy.model_sampling as _ms
            flow_classes = (
                _ms.ModelSamplingDiscreteFlow,
                _ms.ModelSamplingFlux,
                _ms.ModelSamplingAV,
                _ms.ModelSamplingCosmosRFlow,
            )
        except Exception:
            flow_classes = ()
        _mt_int = int(getattr(mt, "value", mt)) if mt is not None else None
        is_flow = isinstance(ms, flow_classes) if flow_classes else None
        if is_flow is None:
            try:
                from comfy.model_base import ModelType as _MT
                is_flow = _mt_int in (
                    _MT.FLOW.value, _MT.FLUX.value, _MT.FLOW_AV.value,
                    _MT.FLOW_COSMOS.value, _MT.IMG_TO_IMG_FLOW.value,
                )
            except Exception as _e:
                logging.debug("[PD-AOS] is_flow fallback unavailable: %s", _e)
                is_flow = None
        image_model = uc.get("image_model")
        guidance_embed = uc.get("guidance_embed")
        is_distilled = None
        try:
            diff_model = patcher.get_model_object("diffusion_model")
            if image_model == "flux" and not guidance_embed:
                n_double = len(getattr(diff_model, "double_blocks", []))
                is_distilled = n_double <= 6
        except Exception as _e:
            logging.debug("[PD-AOS] distilled-block probe unavailable: %s", _e)
        if is_distilled is None:
            try:
                init = getattr(patcher, "cached_patcher_init", None)
                if init and len(init) > 1 and init[1]:
                    fname = str(init[1][0]).lower()
                    is_distilled = any(k in fname for k in
                                       ("turbo", "schnell", "lightning",
                                        "dmd", "lcm", "hyper"))
            except Exception as _e:
                logging.debug("[PD-AOS] distilled-filename probe unavailable: %s", _e)
        return {
            "is_flow": is_flow,
            "model_type": _mt_int,
            "sigma_min": float(getattr(ms, "sigma_min", 0.0)),
            "sigma_max": float(getattr(ms, "sigma_max", 1.0)),
            "sigma_data": float(getattr(ms, "sigma_data", 1.0)),
            "shift": getattr(ms, "shift", None),
            "multiplier": getattr(ms, "multiplier", None),
            "latent_channels": int(_lf_get(lf, "latent_channels", 4)),
            "latent_dimensions": int(_lf_get(lf, "latent_dimensions", 2)),
            "spacial_downscale_ratio": int(_lf_get(lf, "spacial_downscale_ratio", 8)),
            "temporal_downscale_ratio": int(_lf_get(lf, "temporal_downscale_ratio", 1)),
            "image_model": image_model,
            "in_channels": uc.get("in_channels"),
            "out_channels": uc.get("out_channels"),
            "guidance_embed": guidance_embed,
            "is_distilled": is_distilled,
            "is_video": int(_lf_get(lf, "latent_dimensions", 2)) >= 3,
        }
    except Exception as _e:
        logging.warning("[PD-AOS] model info extraction failed: %s: %s",
                        type(_e).__name__, _e)
        return None

_PROBE_RANK_CACHE = {}
# Why the probe rank came out the way it did.  The node-invoked probes build a
# 4-D latent while these models need 5-D, and a 4-D input is a pass-through --
# so the rank DECISION is the thing to measure, not the resulting x_shape.
_PROBE_RANK_DIAG = {}
# The most recent shape decision, bound to the latent it produced (see
# _probe_latent_shape).  Read by _probe_model_call so the probe reports the
# reason for ITS input's rank, not whatever ran last.
_LAST_SHAPE_DIAG = {}

def _probe_rank_key(patcher):
    try:
        if patcher is None:
            return None
        return _model_cache_key(patcher)
    except Exception:
        return None

def _remember_probe_rank(patcher, rank):
    try:
        if rank not in (4, 5):
            return
        k = _probe_rank_key(patcher)
        if k is not None:
            _PROBE_RANK_CACHE[k] = int(rank)
    except Exception:
        pass

def _detect_probe_rank(patcher, mi=None):
    # Records WHY it reached its verdict, so the next run MEASURES the cause
    # instead of inferring it from the resulting x_shape.  Written to a SUB-dict
    # so a later detection cannot clobber a resolution another call recorded.
    _d = {"mi_ld": None, "mi_src": None, "lf_ld": None, "uc_temporal": [],
          "uc_keys": None, "pattern_hit": None, "pattern_tried": 0,
          "verdict": None, "rule": None}
    _PROBE_RANK_DIAG["detect"] = _d
    if patcher is None:
        _d["rule"] = "patcher-is-None"
        return None
    try:
        if mi is None:
            try:
                mi = _extract_model_info(patcher)
                # ONLY claim "extracted" when something was actually returned.  This
                # used to be set straight after the CALL, with no check on the result,
                # so it read "extracted" in exactly the failing case -- the corpus pairs
                # `no-detector-fired` with `mi_ld: null` AND `mi_src: "extracted"`,
                # i.e. the provenance stated the opposite of the truth.
                _d["mi_src"] = "extracted" if mi is not None else "extracted-None"
            except Exception:
                mi = None
                _d["mi_src"] = "extract-failed"
        else:
            _d["mi_src"] = "passed-in"
        if mi is not None:
            try:
                ld = int(mi.get("latent_dimensions") or 0)
            except (TypeError, ValueError):
                ld = 0
            _d["mi_ld"] = ld
            if ld >= 3:
                _d["verdict"] = 5
                _d["rule"] = "model_info.latent_dimensions>=3"
                return 5
    except Exception as e:
        _d["mi_src"] = "raised:" + type(e).__name__
    # ---- THE DECLARATION, READ FROM THE MODEL ITSELF ------------------------
    # `latent_dimensions` is declared per format in ComfyUI's OWN source
    # (comfy/latent_formats.py:4-7) and the latent rank is 2 + it: 3 for the
    # video/temporal formats (Wan21, Mochi, LTXV, HunyuanVideo, Cosmos1CV8x8x8,
    # CogVideoX, SeedVR2), 2 for images, 1 for audio/3D.  The block above reads the
    # same field, but only THROUGH `mi` -- and on the real corpus `mi` is None
    # (`mi_ld: null`), so the definitive route was skipped and the detector fell
    # through to the two weak ones and then to the caller's `return 4` fallback: a 4-D
    # latent through a 5-D model is a SILENT PASS-THROUGH, which is the frozen probe,
    # which is every derived calibration computed from zeros.  Reading the declaration
    # from the patcher removes the dependency on a lookup that can return None.
    #
    # IT CAN ONLY PROMOTE.  `>= 3` returns 5; anything else FALLS THROUGH rather than
    # returning 4, so a temporal `unet_config` can still win (a model may carry temporal
    # patching its format does not advertise) and an image model still reaches the
    # caller's fallback unchanged.  Asserted in scratch_rank_declaration_rig.py.
    try:
        _lf = patcher.get_model_object("latent_format")
        _raw = getattr(_lf, "latent_dimensions", None)
        _d["lf_ld"] = _raw
        # COERCE, DO NOT TYPE-CHECK.  The first version of this route tested
        # `isinstance(_ld, int)`, which is FALSE for a float `3.0` and for int-LIKE
        # objects (numpy scalars, enums, wrappers) -- both perfectly legal declarations
        # -- so the route silently declined to fire for them.  A silently-declined
        # route is INDISTINGUISHABLE from a model that genuinely is 4-D, which is the
        # exact failure class this route exists to remove, so the check must be
        # numeric and the raw value must still be recorded.  Caught by the edge-case
        # section of scratch_rank_declaration_rig.py, not by the happy path.
        _ld = None
        try:
            if _raw is not None:
                _ld = int(_raw)          # bool True -> 1, floats/ints/np scalars all work
        except (TypeError, ValueError):
            _ld = None                   # non-numeric: fall through, raw already recorded
        if _ld is not None and _ld >= 3:
            _d["verdict"] = 5
            _d["rule"] = "latent_format.latent_dimensions>=3"
            return 5
    except Exception as e:
        _d["lf_ld"] = "raised:" + type(e).__name__
    try:
        uc = patcher.get_model_object("model_config").unet_config or {}
        if isinstance(uc, dict):
            _d["uc_keys"] = sorted(str(k) for k in uc.keys())[:24]
            _hits = [k for k in ("patch_temporal", "max_frames",
                                 "temporal_patch_size") if k in uc]
            _d["uc_temporal"] = _hits
            if _hits:
                _d["verdict"] = 5
                _d["rule"] = "unet_config:" + ",".join(_hits)
                return 5
    except Exception as e:
        _d["uc_keys"] = "raised:" + type(e).__name__
    try:
        dm = patcher.get_model_object("diffusion_model")
        for _n, mod in dm.named_modules():
            _d["pattern_tried"] += 1
            pat = getattr(mod, "pattern", None)
            if isinstance(pat, str) and "(t" in pat and "(h" in pat \
                    and "(w" in pat:
                _d["pattern_hit"] = pat[:60]
                _d["verdict"] = 5
                _d["rule"] = "module.pattern"
                return 5
    except Exception as e:
        _d["pattern_hit"] = "raised:" + type(e).__name__
    _d["rule"] = "no-detector-fired"
    return None

def _resolve_probe_rank(patcher, ref_x=None, mi=None):
    _rank_src = None
    try:
        k = _probe_rank_key(patcher)
        if k is not None:
            v = _PROBE_RANK_CACHE.get(k)
            if v in (4, 5):
                _rank_src = "cache[{}]".format(k[:24])
                _PROBE_RANK_DIAG["resolve_src"] = _rank_src
                _PROBE_RANK_DIAG["resolve_rank"] = int(v)
                return int(v)
    except Exception:
        pass
    try:
        if torch.is_tensor(ref_x):
            d = int(ref_x.dim())
            if d in (4, 5):
                _PROBE_RANK_DIAG["resolve_src"] = "ref_x.dim()"
                _PROBE_RANK_DIAG["resolve_rank"] = d
                _PROBE_RANK_DIAG["ref_x_shape"] = tuple(int(v) for v in ref_x.shape)
                return d
    except Exception:
        pass
    try:
        r = _detect_probe_rank(patcher, mi)
        if r in (4, 5):
            _PROBE_RANK_DIAG["resolve_src"] = "detect"
            _PROBE_RANK_DIAG["resolve_rank"] = int(r)
            return r
    except Exception:
        pass
    # THE FALLBACK.  A 5-D model given this rank is a pass-through, so record
    # loudly that the decision was made by default and not by detection.
    _PROBE_RANK_DIAG["resolve_src"] = "FALLBACK-4 (no detector fired)"
    _PROBE_RANK_DIAG["resolve_rank"] = 4
    return 4

def _probe_rank_ladder(patcher, ref_x=None, mi=None):
    ladder = [_resolve_probe_rank(patcher, ref_x, mi)]
    try:
        if torch.is_tensor(ref_x):
            d = int(ref_x.dim())
            if d in (4, 5) and d not in ladder:
                ladder.append(d)
    except Exception:
        pass
    try:
        r = _detect_probe_rank(patcher, mi)
        if r in (4, 5) and r not in ladder:
            ladder.append(r)
    except Exception:
        pass
    for r in (4, 5):
        if r not in ladder:
            ladder.append(r)
    return ladder

def _probe_rank_ladder_run(run_at_rank, ladder, degenerate_fn):
    """Try each rank until one yields a NON-DEGENERATE profile.

    WHY THIS IS DRIVEN BY DEGENERACY AND NOT BY EXCEPTIONS
    -----------------------------------------------------
    The node-side probes pass `ref_x=None`, so the rank comes from
    `_detect_probe_rank`, which only recognises a model whose `unet_config`
    advertises temporal keys.  For every other 5-D model it fails and
    `_resolve_probe_rank` returns its `return 4` fallback -- and a 4-D latent
    through a 5-D model RUNS SUCCESSFULLY and returns its input.  Nothing is
    raised, so `_is_shape_error` never fires and an exception-driven ladder
    stalls at rank 4 forever, which is the bug this exists to kill.

    The criterion is the profile's own degeneracy, which is a measurement the
    probe already makes.  On the saved corpus the separation is ~6 orders of
    magnitude and has NO overlap, so the predicate is not a judgement call:

        frozen  (ret_rel == 0, n=12):  med_jump 0.0      .. 1.11e-08
        healthy (ret_rel != 0, n=24):  med_jump 4.87e-02 .. 2.70e-01
        `_probe_degenerate` threshold: 1e-4  (10x above the frozen max,
                                             487x below the healthy min)

    Returns (profile, rank_used, tried).  `profile` is the FIRST profile seen
    (even if all were degenerate) so behaviour degrades to the old one rather
    than to nothing, and `rank_used` is None only if every rank produced None.
    """
    tried = []
    first = None
    first_rank = None
    for r in ladder:
        try:
            prof = run_at_rank(r)
        except Exception as e:
            tried.append(r)
            logging.info("[PD-AOS-probe] rank %s failed: %s", r, e)
            continue
        tried.append(r)
        if prof is None:
            continue
        if first is None:
            first = prof
            first_rank = r
        try:
            bad = bool(degenerate_fn(prof))
        except Exception:
            bad = False
        if not bad:
            if len(tried) > 1:
                logging.info("[PD-AOS-probe] rank ladder: rank %s was "
                             "degenerate, using rank %s (tried %s)",
                             tried[0], r, tried)
            return prof, r, tried
    return first, first_rank, tried

def _probe_latent_shape(ref_x=None, pres=16, batch=1, rank=None,
                        channels=None, patcher=None, mi=None, site=None):
    # PER-CALL record of the shape decision.  This used to be inferred from a
    # global dict that a later call could clobber, which made
    # `resolve_src=ref_x.dim()` appear on a call site that passes ref_x=None.
    # Recording at the build itself binds the reason to the latent it produced.
    _LAST_SHAPE_DIAG.clear()
    _LAST_SHAPE_DIAG.update({
        "site": site, "pres": pres, "batch": batch, "rank_arg": rank,
        "ref_x_is_tensor": bool(torch.is_tensor(ref_x)),
        "ref_x_dim": (int(ref_x.dim()) if torch.is_tensor(ref_x) else None),
        "ref_x_shape": (tuple(int(v) for v in ref_x.shape)
                        if torch.is_tensor(ref_x) else None),
        "channels_arg": channels, "patcher_is_none": patcher is None,
    })
    try:
        batch = int(batch)
        if batch < 1:
            batch = 1
    except Exception:
        batch = 1
    try:
        pres = max(1, int(pres))
    except Exception:
        pres = 16
    if rank not in (4, 5):
        rank = _resolve_probe_rank(patcher, ref_x, mi)
    if channels is None:
        try:
            channels = int(ref_x.shape[1]) if torch.is_tensor(ref_x) else 0
        except Exception:
            channels = 0
    try:
        channels = int(channels)
    except Exception:
        channels = 4
    if channels < 1:
        channels = 4
    _out = (batch, channels, pres, pres)
    if rank == 5:
        _out = (batch, channels, 1, pres, pres)
        try:
            if torch.is_tensor(ref_x) and ref_x.dim() == 5:
                T = int(ref_x.shape[2])
                if 0 < T <= 64:
                    _out = (batch, channels, T, pres, pres)
        except Exception:
            pass
    _LAST_SHAPE_DIAG.update({
        "rank_out": int(rank), "out_shape": tuple(int(v) for v in _out),
        "resolve_src": _PROBE_RANK_DIAG.get("resolve_src"),
        "detect_rule": _PROBE_RANK_DIAG.get("detect", {}).get("rule")
        if isinstance(_PROBE_RANK_DIAG.get("detect"), dict) else None,
    })
    return _out

def _is_shape_error(e):
    try:
        if _is_device_error(e):
            return False
        if isinstance(e, IndexError):
            return True
        msg = str(e).lower()
    except Exception:
        return False
    for token in ("tuple index out of range", "index out of range",
                  "number of dimensions", "dimension mismatch",
                  "dimensions mismatch", "expected 3d", "expected 4d",
                  "expected 5d", "got 3d", "got 4d", "got 5d",
                  "expected 3-dimensional", "expected 4-dimensional",
                  "expected 5-dimensional", "got 3-dimensional",
                  "got 4-dimensional", "got 5-dimensional",
                  "invalid shape", "shape mismatch", "size mismatch",
                  "mat1 and mat2", "calculated padded input size",
                  "output size is too small", "expected input batch_size",
                  "einops", "rearrange", "weight of size"):
        if token in msg:
            return True
    return False

def _navigate_to_patcher(model):
    if model is None:
        return None
    try:
        patcher = model
        if hasattr(model, "inner_model"):
            inner = model.inner_model
            if hasattr(inner, "model_patcher"):
                patcher = inner.model_patcher
            elif hasattr(inner, "inner_model") and hasattr(inner.inner_model, "model_patcher"):
                patcher = inner.inner_model.model_patcher
        elif hasattr(model, "model_patcher"):
            patcher = model.model_patcher
        if hasattr(patcher, "get_model_object"):
            return patcher
        return None
    except Exception:
        return None

def _param_digest(dm, n_tensors=64, per_tensor=64):
    """A collision-resistant digest of a diffusion model's parameters.

    WHY THIS REPLACES THE OLD FALLBACK.  The previous weak key was
    `"wc:%.6f:%d" % (sum of 64 strided floats from EACH of the FIRST FOUR parameters, 4)` -- a
    256-float SUM, rounded to 6 decimals, taken from the earliest layers only.  It is not a digest,
    and it was MEASURED to collide: `mklanKrea2_v10_quantized_int8_00001_` and
    `cielbleuKrea2_v1-int8_convrot` -- two different files -- both returned `wc:-40.044484:4`, so the
    probe cache served one model's calibration to the other.

    WHY NOT HASH EVERYTHING.  These checkpoints are 13 GB; hashing them is not "a few ms" and the key
    is computed once per model, so the design goal is BOUNDED WORK with negligible collision
    probability.  Three things buy that, and none of them is a sum:
      * MANY TENSORS, spread across the whole model rather than the first four.  A finetune differs
        most in the middle and late blocks, which the old key never looked at.
      * BYTES, not a float sum.  Two different tensors can share a sum to 6 decimals; they cannot
        share their bytes.
      * SHAPES AND DTYPES for EVERY parameter, which are free (no data read) and already eliminate
        most pairs; the sampled bytes then resolve the rest.

    COST, MEASURED rather than assumed -- `scratch_model_key_cost_rig.py` wrote the first version of
    this paragraph, and `scratch_param_digest_cost_rig.py` later REFUTED what it said.  Read the
    second one, not the first.

    **THE DEFECT THIS FUNCTION SHIPPED WITH, AND THE RETRACTION (2026-10-03).** This used to do
    `flat = p.detach().reshape(-1)` on each picked parameter.  On an int8 checkpoint `p` is a
    `comfy_kitchen` `QuantizedTensor` -- a `torch.Tensor` SUBCLASS whose `__torch_dispatch__` handles
    `detach`/`clone`/`to`/`contiguous`/`copy_`/`empty_like`, consults a layout table, and OTHERWISE
    calls `_dequant_and_fallback`, which MATERIALISES THE WHOLE TENSOR.  `TensorWiseINT8Layout`
    registers only `t`/`linear`/`mm`/`addmm` -- no `reshape`, no `view` (the FP8 layout registers
    both, `fp8.py:211-216`) -- so `detach().reshape(-1)` was not a view: it DEQUANTISED every picked
    parameter, 28.6 % of the checkpoint (3.473 G int8 elements), and kept 64 values from each.

    `scratch_param_digest_cost_rig.py` measured that at 0.037-0.107 s and declared the hypothesis
    REFUTED as the cause of a measured 14.6 s gap.  **THAT VERDICT WAS WRONG, and the reason is a rig
    error worth recording: the rig built its `QuantizedTensor`s directly in VRAM, where the
    dequantisation is a fast elementwise pass, while the real model's qdata lives in a `fast_disk`
    staging store and every byte has to be faulted in.**  On the operator's own next run the engine's
    new `_phase_ms` record reported `model_key_ms` = **6057 ms and 6386 ms** per queue -- the
    dequantisation reading ~3.5 GB per queue off the staging store.  The rig's ground truth was not
    representative of the thing it claimed to measure (rule 1), and the fix is below.

    **THE FIX: SAMPLE THE QUANTISED STORAGE, NEVER THE DEQUANTISED TENSOR.** A `QuantizedTensor`
    exposes its int8 payload as `_qdata`, a PLAIN tensor.  Sampling that is not merely cheaper
    (1 byte per element instead of 2, and a view rather than a materialisation): it is a BETTER
    digest input, because the raw quantised bytes carry more entropy than their rounded-up float32
    image and no work is discarded.  Plain-float parameters take the `_qdata`-absent branch and
    behave exactly as before.

    NOTE FOR WHOEVER READS AN OLD LOG: this changes the digest VALUE, so `probe_key` changes and
    every probe cache entry is invalidated once.  That is a one-time cost, and the key was already
    changing per build; the separation property is what matters and it is checked by the rig.
    """
    try:
        ps = [p for p in dm.parameters() if getattr(p, "numel", None) and p.numel() > 0]
        if not ps:
            return None
        h = hashlib.sha1()
        # 1) shapes and dtypes for EVERY parameter: no data movement at all.
        for p in ps:
            h.update(("%s|%s;" % (tuple(p.shape), p.dtype)).encode())
        # 2) a strided byte sample from tensors spread across the model.
        step = max(1, len(ps) // max(1, n_tensors))
        picked = ps[::step][:n_tensors]
        for p in picked:
            # SEE THE DOCSTRING: on a quantised parameter, `p` itself must never be
            # reshaped -- that dequantises the whole tensor.  `_qdata` is the storage.
            try:
                src = getattr(p, "_qdata", None)
                if not torch.is_tensor(src):
                    src = p
                flat = src.detach().reshape(-1)
            except Exception as _e:
                logging.warning("[PD-AOS] param digest: cannot flatten %s: %s",
                                getattr(p, "dtype", "?"), _e)
                continue          # exotic storage: the shape and dtype already went in above
            n = flat.numel()
            stride = max(1, n // max(1, per_tensor))
            idx = torch.arange(0, n, stride, device=flat.device)[:per_tensor]
            if idx.numel() == 0:
                continue
            try:
                # dtype-agnostic: 64 int8 bytes for a quantised parameter, 64
                # float32 values for a plain one.  Converting to float32 first
                # would inflate the quantised case 4x AND throw away the byte
                # pattern that makes the sample discriminating.
                # bfloat16/fp8 have no numpy dtype on this torch build, so the
                # raw bytes are read through a uint8 view instead of dropping the
                # sample (which used to leave a shapes-only key labelled healthy).
                sample = flat[idx].to("cpu").contiguous()
                try:
                    sample_bytes = sample.numpy().tobytes()
                except TypeError:
                    sample_bytes = sample.view(torch.uint8).numpy().tobytes()
            except Exception as _e:
                logging.warning("[PD-AOS] param digest: cannot sample %s bytes: %s",
                                getattr(p, "dtype", "?"), _e)
                continue
            h.update(sample_bytes)
        return h.hexdigest()[:16]
    except Exception:
        return None


def _patcher_patch_tag(patcher):
    """The patcher's own identity for the WEIGHTS IT APPLIES, as a tag.

    WHY THIS EXISTS -- LORA WAS INVISIBLE TO THE MODEL KEY (2026-10-18).  The key below
    describes the BASE checkpoint: `model_hash()` (when the host provides it) and
    `_param_digest` both sample `diffusion_model.parameters()`, and a LoRA is a list of
    patches the patcher applies ON LOAD, not a change to those parameter tensors.  So a
    LoRA run and a no-LoRA run produced the SAME key: measured on
    `moss_image_sft_int8_convrot`, the LoRA runs (image `LoraLoaderModelOnly`,
    `MOSS_Image_SFT_8_steps_dmd_v0.1`, strength 1.0; logs 00072/00073) carried
    `pd:61457a9d6c516c11`, exactly the key of the non-LoRA runs 00066-00071.  The
    Autotuner's model and per-prompt caches therefore served a profile measured through a
    DIFFERENT model.  The operator saw it as a LoRA producing bad results with no re-probe.

    `patches_uuid` is the host's own answer to "which weights are applied": ComfyUI rolls a
    fresh UUID in `add_patches`, `add_weight_wrapper` and `set_hooks`, and `clone()` copies
    it to a child patcher.  So it is:
      * STABLE across the node-side and engine-side patchers of one queue (the probe's
        clone inherits it), and
      * CHANGED by a different LoRA, a different strength, or removing the LoRA.
    The FULL UUID is used, not a prefix: a truncated hash would trade a re-probe for a
    possible wrong reuse, which is the failure this key exists to prevent.  It is
    deliberately NOT a content hash: hashing every LoRA tensor per queue would cost more
    than the re-probe it replaces, and a UUID can only cause a redundant re-probe -- never
    a wrong reuse.

    Returns `"-"` when the patcher has no tag, so the key shape stays uniform.
    """
    try:
        v = getattr(patcher, "patches_uuid", None)
        return str(v) if v is not None else "-"
    except Exception:
        return "-"


def _model_cache_key(model):
    """A key that MUST distinguish two different checkpoints.  It used not to; see below.

    THE DEFECT THIS FIXES, measured 2026-10-03: the old fallback was a 256-float SUM over the first
    four parameters, rounded to 6 decimals, and it COLLIDED between
    `mklanKrea2_v10_quantized_int8_00001_` and `cielbleuKrea2_v1-int8_convrot` -- two different files
    (13 148 546 560 against 13 148 995 440 bytes) that both returned `wc:-40.044484:4`.  Their probe
    profiles were therefore the same cached object and their measured fields matched to 15 digits
    (`med_curv` 1.62827205347272).  Since the profile carries `wmax_base`, `rtol`, the guard
    calibration, the LF gains and `is_distilled_eff`, that meant **swapping models silently inherited
    the previous model's entire calibration**.  Third instance of the cache-key defect class in this
    project (rule 28), and the first where the key was WEAK rather than missing.

    ORDER OF PREFERENCE, unchanged in spirit and stronger in fallback:
      1. `ModelPatcher.model_hash()` when the host provides it.
      2. `_param_digest` -- a real digest, bounded work, sample spread across the model.
      3. the old width-code, KEPT ONLY as a last resort so a failure to sample never leaves the
         engine without a key at all.  It is no longer the normal path.
      4. `id(model)`, which is at least unique within a session.
    A key that falls back to (3) or (4) is a DEGRADED key and the log says so: `_LAST_CFG["_model_key_kind"]`
    carries which branch produced it, so a corpus query can exclude runs that could not tell two
    models apart.  Without that, this defect would be invisible in exactly the way it was.

    PATCHES ARE PART OF THE MODEL (2026-10-18).  All four branches above describe the BASE
    checkpoint and are blind to a LoRA, which the patcher applies on load.  The key therefore
    carries `_patcher_patch_tag(patcher)` as a trailing `|p:<tag>`: a different LoRA, strength or
    no-LoRA state is a different measurement, and a profile measured through one must never serve
    another.  `_LAST_CFG["_model_patch_tag"]` stamps the tag in force for the same reason
    `_model_key_kind` is stamped -- a cache miss caused by a LoRA must be readable from the log.
    """
    global _MODEL_KEY_KIND, _MODEL_PATCH_TAG
    patcher = _navigate_to_patcher(model)
    _MODEL_PATCH_TAG = _patcher_patch_tag(patcher)
    _tail = "|p:" + _MODEL_PATCH_TAG
    if patcher is not None:
        try:
            mh = getattr(patcher, "model_hash", None)
            if callable(mh):
                v = mh()
                if v:
                    _MODEL_KEY_KIND = "model_hash"
                    return "mh:" + str(v) + _tail
        except Exception:
            pass
        try:
            dm = patcher.get_model_object("diffusion_model")
            dig = _param_digest(dm)
            if dig:
                _MODEL_KEY_KIND = "param_digest"
                return "pd:" + dig + _tail
        except Exception:
            pass
        try:
            dm = patcher.get_model_object("diffusion_model")
            acc = 0.0
            n = 0
            for p in list(dm.parameters())[:4]:
                flat = p.detach().float().flatten()
                if flat.numel() > 0:
                    acc += float(flat[:: max(1, flat.numel() // 64)][:64].sum())
                    n += 1
            if n > 0:
                _MODEL_KEY_KIND = "width_code_DEGRADED"
                return "wc:%.6f:%d" % (acc, n) + _tail
        except Exception:
            pass
        base = getattr(patcher, "model", None)
        if base is not None:
            _MODEL_KEY_KIND = "model_id_DEGRADED"
            return "bm:" + str(id(base)) + _tail
    _MODEL_KEY_KIND = "model_id_DEGRADED"
    return "id:" + str(id(model)) + _tail

def _schedule_sig(x, sigmas, cfg):
    try:
        h = hashlib.sha1()
        h.update(str(tuple(int(v) for v in x.shape[1:])).encode())
        # THE PROBE GRID, NOT THE RAW SCHEDULE (see `_probe_grid_sig`).  The probe's
        # measurements depend on the points it walks, and `_probe_grid` derives those from
        # the schedule's RANGE -- not from the interior placement.  Hashing the sigma list
        # made every probe-derived placement a new slot, so a scheduler that places steps
        # from the probe re-probed on every prompt.
        try:
            h.update(_probe_grid_sig(sigmas, cfg).encode())
        except Exception:
            st = sigmas.detach().to("cpu", torch.float32).contiguous().flatten()
            h.update(str([round(float(v), 8) for v in st]).encode())
        h.update(("ps=%d,pr=%d,pd=%d,wb=%d,sf=%.4f" % (
            int(cfg.get("probe_steps", 10)),
            int(cfg.get("probe_resolution", 48)),
            int(bool(cfg.get("pct_derived", False))),
            int(bool(cfg.get("wmax_bound", False))),
            # `probe_smin_frac` CHANGES WHAT IS MEASURED -- it decides whether the grid stops at
            # 10% of sigma_max or reaches below it -- so it belongs in the key for exactly the
            # reason the comment above gives: a measurement cannot be re-derived from itself.  A
            # profile measured at one reach must never serve a run asking for another.  The shipped
            # default is the constant, so every pre-existing key is reproduced verbatim.
            float(_probe_reach_eff(cfg)))).encode())
        # THE GUIDANCE THE PROBE MEASURED WITH (U05/1361).  The probe runs THROUGH the
        # guider, so CFG is a measurement input, not a re-derivable constant: a profile
        # taken at CFG 5 must not serve a run at CFG 7.  Added only when known, so a run
        # whose CFG cannot be read keeps the pre-existing key rather than fragmenting it.
        try:
            if cfg.get("_probe_cfg_eff") is not None:
                h.update(("cf=%.4f" % float(cfg["_probe_cfg_eff"])).encode())
        except (TypeError, ValueError):
            pass
        # The A/B experiment modes belong in this digest for the same reason
        # `pct_derived` and `wmax_bound` do: they change the `trajectory_state`
        # that gets CACHED INSIDE the profile.  Leaving them out made the
        # oversmooth A/B unsound, and it is measured rather than theorised --
        # scratch_oversmooth_engine_mode_rig.py reads the corpus and finds 40 of
        # 380 runs at the shipped default that CONSUMED a non-zero `oversmooth`
        # (up to 0.698, driving the `s` gain to 1.1225 against its 0.25 baseline),
        # including 2 runs in mode "off", which is defined to force it to 0.
        # Mechanism: an exact-key hit is trusted as-is and suppresses the probe
        # (see `probe_from_node` below), so the run consumes whatever state was
        # cached when the OTHER arm populated this slot.  Setting the modes at
        # the top of aflops_engine did not fix it -- that only orders the
        # assignment relative to a probe that, on a cache hit, never runs.
        h.update(("osm=%s,ev=%s" % (
            str(cfg.get("_osm_ab_mode", "engine") or "engine"),
            str(cfg.get("_ev_ab_mode", "engine") or "engine"))).encode())
        # `wmax_budget` IS DELIBERATELY ABSENT, and this is a MEASURED decision rather than an
        # omission (it was carried as open item 13 for two days on the assumption that it was one).
        # It reaches only the DERIVED calibration -- never a measurement -- and `_recalibrated`
        # re-derives that from the profile's own budget-independent measurements whenever
        # `_calib_inputs` says the live inputs differ, so a cached profile cannot serve a value
        # derived under another budget.  `scratch_wmax_budget_cover_rig.py` (9/9) checks it, and the
        # load-bearing check is corpus-wide: of 306 runs whose consumed profile records a
        # `wmax_ab.budget`, 306 record THAT RUN's own budget -- including the pair that shares one
        # `probe_key` across two budgets in one session (`log_00162` at 1.0, `log_00163` at 4.0,
        # `tol_eff` 0.005 vs 0.02).  Keying it would fragment the probe cache (a fresh probe, ~10
        # model calls) to protect a value that is already correct.  Same trade as `wmax_eg_dir`.
        #
        # `wmax_budget`'s exemption is NOT a licence to omit the input below, and the distinction
        # is exactly rule 28's: the budget reaches only what is DERIVED from the profile's
        # measurements, whereas the distilled shaping of the probe's grid changes WHAT IS
        # MEASURED (`log_gap_probe` is the grid's own mean log gap).  A measurement cannot be
        # re-derived from itself.  The conditioning was briefly keyed here too and has been
        # REMOVED -- the model measurement is shared across prompts by design, and the evidence
        # that said otherwise pooled builds; `_probe_key_env_part` carries the retraction.
        h.update(_probe_key_env_part(cfg).encode())
        return h.hexdigest()[:16]
    except Exception:
        return "nologic"

def _sigmas_digest(sigmas):
    """Full-precision identity of the sigma tensor the sampler CONSUMED.

    Why this exists (DEAD_FEATURES.md sections 13-15): `_ms_info.sigmas_in` is
    the AUTOTUNER's input rounded to 5 decimals, and each per-step sigma is
    logged rounded to 6.  Two runs whose schedules differ only below those
    precisions are therefore indistinguishable in the log -- yet this was
    measured to be enough to change the image visibly, because the model
    amplifies a ~1e-6 relative step difference to 0.57% within two steps.

    A measured example: two arms of a scheduler A/B produced arrays that agreed
    at every stored decimal and still carried different probe cache keys, which
    forced the whole difference to be *deduced* from a sha1 rather than read.
    This digest reads the raw bytes, so "the same schedule" becomes checkable.
    """
    try:
        st = sigmas.detach().to("cpu", torch.float32).contiguous().flatten()
        n = int(st.numel())
        if n == 0:
            return {"n": 0}
        try:
            payload = st.numpy().tobytes()
        except Exception:
            payload = repr([float(v) for v in st]).encode()
        return {
            "sha1_16": hashlib.sha1(payload).hexdigest()[:16],
            "n": n,
            "dtype": str(getattr(sigmas, "dtype", "?")),
            # full-precision scalars, not rounded: the digest detects a
            # difference, these show how big it was
            "first": float(st[0]),
            "last": float(st[-1]),
            "sum_f64": float(st.double().sum()),
        }
    except Exception as e:
        return {"error": type(e).__name__}


def _prune_probe_cache(mk, keep=8):
    """Trim the SCHEDULE-keyed slots; never evict a node-side MODEL slot.

    A node-side model profile is a cross-prompt measurement filed under
    `<mk>|s:node:<params>|<cfg>` (written at `_run_node_side_model_probe`'s
    `_PROBE_PROFILES` assignment).  The schedule-keyed engine slots are per
    schedule.  Counting both in one `keep=8` list made the node slot the oldest
    entry as soon as eight schedule keys existed, so it was dropped while
    `_NODE_PROBE_PROFILES` still held the measurement.  `_node_model_profile`
    looks in `_PROBE_PROFILES`, found nothing, and the engine re-ran the probe
    on the next prompt: measured on moss `log_00049`, `model_probe_ms` 168 690
    against 78 500 ms of sampling, at 16 steps / 128 px, while the node profile
    was a `cache_hit`.  Excluding the node slots here is what keeps the
    cross-prompt measurement addressable.
    """
    keys = [k for k in _PROBE_PROFILES
            if (k == mk or k.startswith(mk + "|s:"))
            and "|s:node:" not in k]
    for k in keys[:-keep] if len(keys) > keep else []:
        _PROBE_PROFILES.pop(k, None)

def _cond_signature(positive, negative):
    def _tensor_part(t):
        tf = t.detach().float()
        return "T%s:%.6e:%.6e:%.6e" % (
            tuple(tf.shape), float(tf.sum()),
            float(tf.norm()), float(tf.mean()))

    def _entry_part(entry):
        if isinstance(entry, dict):
            keys = sorted(str(k) for k in entry.keys() if k != "uuid")
            parts.append("D%d:%s" % (len(keys), ",".join(keys)))
            for k in keys:
                v = entry.get(k)
                if torch.is_tensor(v):
                    parts.append(str(k) + "=" + _tensor_part(v))
        elif isinstance(entry, (list, tuple)) and entry:
            t = entry[0]
            if torch.is_tensor(t):
                parts.append(_tensor_part(t))
            else:
                parts.append(repr(t)[:64])
            if len(entry) > 1 and isinstance(entry[1], dict):
                parts.append(",".join(sorted(entry[1].keys())))
        else:
            parts.append(repr(entry)[:64])

    parts = []
    try:
        for cond in (positive, negative):
            parts.append("#%d" % (len(cond) if cond else 0))
            for entry in (cond or []):
                _entry_part(entry)
    except Exception:
        parts.append("err")
    return hashlib.sha256("|".join(parts).encode("utf-8", "replace")).hexdigest()[:16]

def _extra_cond_signature(extra_args):
    try:
        if not isinstance(extra_args, dict):
            return None
        pos = extra_args.get("cond")
        neg = extra_args.get("uncond")
        if pos is None and neg is None:
            return None
        return _cond_signature(pos, neg)
    except Exception:
        return None

def _graded_demand(ratio, rtol, lo, hi, r_hi, cap, graded):
    """THE ONE OWNER OF THE AUTOMATIC MODE'S DEMAND LEVEL.  Called by the demand site AND by the per-step stamp,
    so the level the record reports is the level the maths used -- by construction, not by coincidence.  Two copies
    of this formula is how a stamp drifts from the value in force (the `cond_rtol_loosen` defect).

    `ratio` is the CARRIED `err/tol` (None on the first step).  Below the trigger it returns 1.0, i.e. the
    existing gate semantics; with `graded` False it returns `cap` (the previous flat behaviour, kept so the two can
    be compared in one A/B); with `graded` True it ramps from `lo` to `hi` across (rtol, r_hi] and clamps to
    [1, cap].  Monotone non-decreasing in `ratio` by construction.
    """
    if ratio is None or ratio <= rtol:
        return 1.0
    if not graded:
        return float(cap)
    frac = min(max((ratio - rtol) / max(r_hi - rtol, 1e-9), 0.0), 1.0)
    # CLAMP TO `max(cap, hi)`, not to `cap`: with the shipped defaults (hi = 5 < cap = 6) this is exactly the old
    # clamp and the shipped behaviour is unchanged, but a deliberate `hi` ABOVE the flat value is honoured instead
    # of being silently swallowed.  `graded = False` still returns `cap` exactly, so the negative control holds.
    return float(min(max(round(lo + (hi - lo) * frac), 1.0), max(cap, hi)))


def _demand_policy(bound_available, q, by_bound, demand_bound, graded):
    """WHICH REQUEST A STEP MAKES -- ONE OWNER, called by the demand site AND by the per-step stamp.

    The returned strings are the log's `ord_demand_policy` values, so a record cannot name a policy
    the maths did not run (the `cond_rtol_loosen` defect this pattern exists to prevent):

      "bound_map" the per-pixel validity bound was computed this step and `order_demand_bound` is on:
                  the bound's own map IS the request, so the scalar demand is 1.0 (raise nothing).
      "bound"     the quantile arm: `order_demand_q` above 0 with the bound criterion in force.
      "graded"    the error gate's ramp -- the automatic mode's shipped request.
      "flat"      the pre-ramp behaviour (`order_floor_adaptive_graded` off).

    ORDER MATTERS, AND IT IS EXPLICIT-BEATS-DEFAULT.  A user who sets `order_demand_q` above 0 is
    making a deliberate request for the quantile arm, so it wins even though `order_demand_bound`
    ships ON; otherwise flipping that default would silently swallow a widget the user had set.  With
    q at its default 0.0, the map arm wins where a bound exists, and the ramp is the fallback for the
    steps that have no bound to ask with.
    """
    if q > 0.0 and by_bound:
        return "bound"
    if demand_bound and bound_available:
        return "bound_map"
    return "graded" if graded else "flat"


def _cond_probe_dims(cfg):
    """THE PROBE DIMENSIONS, as they belong in every COND cache key (2026-10-05).

    WHY THIS EXISTS.  `cond_probe_steps` and `cond_probe_resolution` change what the cond probe
    MEASURES -- the resolution it measures at and the length of every curve (`aflops.py:1561` records
    `curv_curve` carrying m-2 entries, and `aflops.py:1577` records that a profile measured at 4 steps
    carries a curve of length 2).  By rule 28 a switch that changes what is measured MUST be in the cache
    key, and these two were in NEITHER of the two key paths, so a run asking for 4 steps was served a
    profile measured at 8: the operator's own test changed both widgets between two runs and got
    byte-identical curves out, with `cond_probe_state: cached` in both.

    The values here are the EFFECTIVE ones (the same clamps the probe applies), so two requests that
    clamp to the same measurement share an entry instead of fragmenting the cache.
    """
    return "|k%d/r%d|sf%.4f" % (max(3, int(cfg.get("cond_probe_steps", 4))),
                                max(8, int(cfg.get("cond_probe_resolution", 16))),
                                _probe_reach_eff(cfg))


def _probe_key_env_part(cfg):
    """THE INPUT THAT SHAPES A MODEL PROBE'S GRID BUT IS NOT THE MODEL OR THE SCHEDULE.

    `di=` is `_probe_distilled_mode(cfg)`, and `_probe_grid(sigmas, K, distilled)` spends half
    its points in the top log-sigma half when it is true, which changes the grid's mean log gap
    -- i.e. `log_gap_probe`, which the rtol derivation reads through
    `scale = (gap_real / gap_probe)**2`.  Its inputs are `_guider_live_cfg` and
    `_model_info["is_distilled"]`; the live cfg reaches the NODE key as the trailing `|<cfg>`
    token but reaches the ENGINE key nowhere, and `_model_info` is the one key the engine strips
    before logging, so a grid mismatch was invisible in the corpus.

    ---- RETRACTION: THIS FUNCTION USED TO CARRY `cs=<cond_probe_sig>` TOO (removed 2026-10-08) --
    It was added on the claim that the model probe's values move with the PROMPT, because the
    probe calls the model through a CFGGuider with the run's own conditioning.  The mechanism is
    real -- `comfy/samplers.py:609-613` skips the uncond batch at cfg 1, so what is evaluated is
    the COND branch -- but THE CLAIM WAS FALSE, and three separate errors produced it:

      1. The evidence pooled BUILDS.  Rule 16 exists for this; the group cited as "four clusters
         tracking the prompt" was 19 runs of ONE build whose runs span TWO CHECKPOINTS under one
         `pd:` digest (`anima-base-v1.0` 13 runs, `animaprequantized_00001_` 6) -- see
         `_param_digest`'s collision note.  The same prompt `9eeb84a2d705a270` reads whitepoint
         0.073525 on one checkpoint and 0.177643 on the other.
      2. The values do NOT partition by prompt.  0.177643 appears against prompt `9eeb84a2â€¦`,
         against `1a4d3df7â€¦` and against NO logged prompt at all; 0.155423 appears under both
         prompts.  What they partition by is the SESSION, which is what a cache does.
      3. Re-measured build-scoped: of 417 groups with >=2 runs, 143 have ONE measurement serving
         every prompt in the build (56 across >=4 prompts, one covering 35), while only 3 show a
         prompt-shaped split and those are confounded with session (18 more vary WITHIN one
         prompt).  SCOPED, because the operator's objection applies to this number: those are
         HISTORICAL builds (up to `363c8e1e269f`), and a statistic computed over old code states
         describes old code states -- rule 16's reason for not pooling, applied to the answer
         rather than to the query.  What CURRENT code states show is thinner and is stated as
         such: on `9cf6beb78f1f` (anima-base-v1.0 logs 00034-00039) SIX different prompt
         signatures share one node-slot measurement (whitepoint 0.110707104 / med_curv
         5.2011676), which is the sharing this key is meant to allow.  On the one build where
         the cond term was LIVE (`8c845858d8e7`) no cross-prompt pair was ever run, so "how much
         would a model re-measurement move with the prompt" is OPEN -- and it is not this key's
         job to pre-empt it.

    So the design's separation holds in practice and in intent: the MODEL probe is measured once
    and shared across prompts, and the COND probe is what carries the per-prompt part (its store
    is prompt-keyed and its `gap`/`prior` measurements exist only per prompt).  Keying the model
    probe by the conditioning would have re-measured it for every prompt (~34 model calls), which
    is exactly the cost this project has chased, in exchange for a benefit the corpus does not
    show.  The residual approximation is honest and remains: the model probe IS a conditioned
    forward pass, so its sharing across prompts is an assumption the design makes rather than a
    property it enforces.  Sizing that leak needs one build that re-measures the same model under
    two prompts; it is an OFFER, not something this key should pre-empt.
    """
    try:
        di = 1 if _probe_distilled_mode(cfg) else 0
    except Exception:
        # UNKNOWN, and said so, rather than folded into a value it is not.
        di = -1
    return "di=%d" % di


def _cond_cache_key(cfg, extra_args, model_key):
    cond_sig = cfg.get("cond_probe_sig")
    if not cond_sig:
        cond_sig = _extra_cond_signature(extra_args)
    return (model_key, str(cond_sig) + _cond_probe_dims(cfg)) if cond_sig else None


def _cond_key_str(ckey):
    """THE ONE SPELLING OF A COND CACHE KEY.  Both the log's store keys and the cfg's stamped
    `_cond_probe_key` must use it.
    WHY THIS EXISTS (2026-10-02).  `_COND_PROBE_PROFILES` is keyed by the TUPLE returned from
    `_cond_cache_key`, and two different pieces of code stringified it two different ways:

        nodes.py:698   "%s | prompt %s" % (key[0], key[1])      <- the STORE keys in the log
        aflops.py      "%s|%s"         % (ckey[0], ckey[1])     <- the STAMPED cfg key

    The consequence was measured: `cfg.cond_probe_key` matched a stored key in **0 of 443
    corpus logs**, so a reader that looks up the cond profile by the key the log itself hands
    it finds NOTHING -- silently, and reports "no cond data" as though that were a result.
    That is what a frozen-probe audit rig did on its first run, and what several cond-side
    reads did during this session.  A stamped identifier that cannot be used to find the thing
    it names is worse than no identifier.

    The STORE's spelling is kept as the canonical one, because 443 existing logs use it for
    their store keys and changing that would invalidate them; only the stamped cfg value was
    ever wrong.  Accepts 2- or 3-element keys, since `_cond_cache_key`'s tuple is extended with
    a schedule signature on the schedule-qualified path.
    """
    if isinstance(ckey, (tuple, list)) and len(ckey) >= 2:
        return " | prompt ".join(str(x) for x in ckey[:2]) + \
            "".join(" | " + str(x) for x in ckey[2:])
    return str(ckey)


def _cond_key_prompt_sig(s):
    """The prompt signature out of a `_cond_key_str` spelling, or None.

    WHY THIS IS A HELPER AND NOT A FIELD INDEX (2026-10-18).  The model key now ends in
    `|p:<patches_uuid>` (see `_patcher_patch_tag`), so `str(key).split("|")[1]` is the
    PATCH TAG, not the prompt.  The report node parsed the stamped `_cond_probe_key` that
    way and would have filtered every cond profile out of `cond_probe_profiles` -- the
    exact silent-empty-dump defect `_cond_key_str` was introduced to end.  The separator
    `" | prompt "` is the one `_cond_key_str` writes; parse that, and let both spellings
    share this function.
    """
    try:
        s = str(s or "")
        if " | prompt " in s:
            return s.split(" | prompt ", 1)[1].split("|", 1)[0].strip() or None
        return None
    except Exception:
        return None


def _cond_profile_origin(from_cache, node_cond_state):
    """Which provenance a per-prompt profile's REPORT should carry.

    THE DEFECT THIS NAMES (measured 2026-10-18, moss log_00071): the engine's fallback
    lookup finds the profile the NODE-side probe wrote for THIS queue under the same key,
    so `from_cache` was True for a brand-new prompt.  The log then printed
    "using cached per-prompt profile (model+prompt+schedule unchanged; skip probe)" beside
    `node_cond_probe_state: ran` -- a fresh measurement reported as a cache hit, which sent
    the operator hunting a cache collision that did not exist.

    `node_cond_probe_state` is the producer's own statement (copied into `_PHASE_MS` at
    engine entry, so it is this queue's value): `ran` means the node measured this prompt
    now; `cache_hit` means the NODE found an earlier profile; absent means no node probe
    ran, so a store hit can only be from an earlier engine run.

    Returns `"node_probe"`, `"cached"` or `"adopted_node"`.  `adopted_node` remains for a
    profile present without a cache claim, which is what the old `else` branch meant.
    """
    if from_cache and str(node_cond_state or "") == "ran":
        return "node_probe"
    if from_cache:
        return "cached"
    return "adopted_node"

# ---------------------------------------------------------------------------
# A/B experiment switches -- TESTING ONLY, defaults reproduce the engine exactly.
#
# Set once per run from cfg by aflops_engine.  Module-level because
# _compute_trajectory_state is called from six places (2085, 3122, 4223, 4517,
# 4870, 5573) and every one of them must apply the same rule; threading a
# parameter through six call sites would risk a site being missed, which is
# exactly the class of bug this session spent its time fixing.
#
#   _OSM_AB_MODE  "engine"     -> the original absolute thresholds (default)
#                 "reanchored" -> the anchors MEASURED from real profiles, below
#                 "off"        -> force oversmooth to 0
#   _EV_AB_MODE   "engine"     -> ev = max(ov, osm) (default)
#                 "ov_only"    -> ev = ov, isolating the `s` consumer
#
# The distilled early-out (`if is_distilled: oversmooth = 0.0`) was REMOVED at
# the operator's request, so the A/B can test whether distilled models
# over-smooth at all.  Note the removal alone changes nothing: in "engine" mode
# the original thresholds still gate every distilled profile to 0, so a run with
# default settings behaves exactly as before.
_OSM_AB_MODE = "engine"
_EV_AB_MODE = "engine"

# THE COND-PATH TOLERANCE LOOSENING, AS A TUNABLE MULTIPLIER (2026-10-04).
#
# `_derive_cond_calibrations` loosens the cond profile's `rtol` when `gap.strength < 0.5`.  That
# threshold never discriminates -- measured over every cond profile that carries the input: `strength`
# spans 0.0046..0.0861, so `< 0.5` is TRUE on 214 of 214 and `> 2.0` on 0 of 214
# (`scratch_nonexp_loose_ends_rig.py` Y4a/Y4b).  So the shipped behaviour is a CONSTANT x1.5, and this
# knob is what lets the operator A/B the loosening's EXISTENCE instead of arguing about its condition:
#
#     1.0 = off (no loosening)      1.5 = the historical intent      0.0 = the tightest the blend allows
#
# RANGE, DERIVED rather than chosen (`scratch_loosen_knob_range_rig.py`, 5/5 over ~2045 profiles):
#   * upper 8.0 -- the multiply's own hard bound: above m = 8.0 no profile in the corpus responds,
#     because the result saturates at the 0.04 cap (`m_sat = 0.04 / base` spans 1.00..8.00).  The BLEND
#     downstream clamps sooner (4x ceiling reached at m p5 1.4 / median 2.5 / p95 7.0), so most runs
#     stop moving well before the top.
#   * lower 0.0 -- and NOT below: the multiply clamps only the TOP, and the blend's clamp is guarded by
#     `blended > 0`, so a NEGATIVE multiplier would pass through unclamped and reach cfg as a negative
#     tolerance.  The guard in `_loosen_from_cfg` is therefore a correctness guard, not a taste.
#   * the A/B's effect size in the value the run CONSUMES (m=1.5 vs 1.0) is a median x1.264, because the
#     blend mixes the cond value with the model-derived one at `cond_probe_weight` (0.5 by default).
#
# IT IS NO LONGER A GLOBAL.  It was, and that was the defect: the two functions that DERIVE a cond
# calibration do not call `_apply_ab_modes`, so they read a value some earlier entry point left behind --
# recorded runs show the arithmetic using m = 1.0 while the cfg and the profile's own `calib_flags` said
# 0.0 and 5.0 (`scratch_cond_loosen_provenance_rig.py`, checks I2a/I2b).  The value now travels as an
# argument from the cfg that is deriving (`_loosen_from_cfg`), so there is nothing to inherit.
_COND_RTOL_LOOSEN_MAX = 8.0


def _apply_ab_modes(cfg):
    """Set the A/B experiment modes for THIS unit of work.

    Every entry point that can compute OR CACHE a `trajectory_state` must call
    this first: `aflops_engine`, and both node-side probes (model and cond).

    WHY (measured, not theorised -- scratch_oversmooth_engine_mode_rig.py):
    these are module globals, and nothing reset them after a run, so a node-side
    probe inherited whatever mode the LAST engine run left behind.  The arm
    therefore leaked: run one `reanchored` generation, and the next Autotuner
    probe computed its profile's `oversmooth` under `reanchored` too, cached it,
    and every later default-mode run adopted that profile and consumed the
    other arm's value.  The corpus shows the result: 40 of 380 runs at the
    shipped default consumed a non-zero `oversmooth`, two of them in mode "off",
    which is defined to force it to zero.

    Setting the mode here is necessary but NOT sufficient on its own -- a probe
    that is skipped because of a cache hit never computes anything, which is why
    the modes also belong in both cache-key builders.  Same defect class as the
    calib flags in `_node_probe_param_sig`'s docstring.
    """
    global _OSM_AB_MODE, _EV_AB_MODE
    m = str((cfg or {}).get("_osm_ab_mode", "engine") or "engine")
    _OSM_AB_MODE = m if m in ("engine", "reanchored", "off") else "engine"
    e = str((cfg or {}).get("_ev_ab_mode", "engine") or "engine")
    _EV_AB_MODE = e if e in ("engine", "ov_only") else "engine"
    # `cond_rtol_loosen` IS DELIBERATELY *NOT* SET HERE (2026-10-04).  It used to be a global set at this
    # one point, and a provenance rig then showed the two functions that actually DERIVE a cond
    # calibration never call this setter -- so they read whatever value a previous entry point had left,
    # and the recorded runs show the arithmetic using `m = 1.0` where the cfg and the profile's own
    # `calib_flags` stamp both said 0.0 / 5.0.  The value is now passed explicitly as an argument
    # (`_loosen_from_cfg`, called from `_derive_cond_calibrations`), which removes the inherited-value
    # class rather than patching one instance.  Setting it here as well would be a SECOND source of truth
    # for one number -- the exact shape of the defect it replaces.

# Anchors MEASURED from 9 real non-distilled profiles by
# scratch_oversmooth_inputs_rig.py: each factor's zero point sits at the
# measured p90 (the least-collapsed run) and its width at the p90..p10 span, so
# the ramp runs 0 -> 1 across the range real runs actually occupy.
#
# Why the original thresholds fail (measured ranges in brackets):
#   trend_resid      needs < -3.2   [-0.756 .. -0.530]   a 25x tail-slope
#                                                         collapse where runs
#                                                         deliver ~1.75x
#   conv_ratio       needs <  0.03  [ 0.066 ..  0.123]   asks 33x, sees 8-15x
#   late_curv_ratio  needs <  0.3   [ 0.478 ..  0.680]   asks 3.3x, sees 1.5-2x
#   med_err          needs <  0.003 [0.0006 .. 0.0010]   passes 9/9 -- left
#                                                         alone, and its ramp
#                                                         re-anchored only so
#                                                         the three factors
#                                                         share one rule
_OSM_RA_TREND_ZERO = -0.5299
_OSM_RA_TREND_W = 0.1531
_OSM_RA_CONV_ZERO = 0.12288
_OSM_RA_CONV_W = 0.03508
_OSM_RA_CURV_ZERO = 0.6795
_OSM_RA_CURV_W = 0.1317
_OSM_RA_ERR_ZERO = 0.000860
_OSM_RA_ERR_W = 0.000262


# A probe profile's `curv_curve` carries m-2 entries (m = probe grid points,
# verified on 315 saved profiles), so m >= 5 is what yields the >= 3 curvature
# points the dwell recommender requires.  nodes.py:382 states the same invariant
# in its own words: "(>= 3 curvature points; guaranteed at probe_steps >= 5)".
# Kept here so this guard and that comment cannot drift apart.
_PROBE_MIN_USABLE_STEPS = 5


def _profile_serves_model_probe(prof, cfg=None):
    """Can this profile serve as THIS run's MODEL measurement -- i.e. suppress the probe?

    Adoption is deliberately NOT gated on this: aflops.py:477-483 documents that
    the model-probe fallback may legitimately serve a per-prompt COND profile,
    because every profile carries the whitepoint and the adaptive-noise prior
    wants it either way.  Only the decision to SKIP our own model probe depends
    on it -- and that skip is what starved the curvature consumers: a profile
    produced at cond_probe_steps (default 4) carries `curv_curve` of length 2
    while the dwell recommender needs >= 3, so the engine skipped its own probe
    (K = max(4, probe_steps) >= 5) and never produced a usable one.  Measured:
    70 of 70 saved runs adopted m=4 and never armed the dwell.

    THREE GATES AS OF 2026-10-08, and the last two are new.  The step-count gate
    below was necessary but nowhere near sufficient, because it asks "is there enough
    of this profile?" and never "is this profile the SAME KIND OF MEASUREMENT as the one
    I would make?" -- and those came apart in practice:
      * `probe_src == "model"` -- a per-prompt COND profile measured at
        `cond_probe_steps`/`cond_probe_resolution` reached this predicate through
        `_PROFILE_BY_MODEL`, was accepted, and (as the model probe was then skipped)
        was PROMOTED into the engine's own `|s:` slot.  Measured, `363c8e1e269f`
        `anima-base-v1.0` log_00019/00020: med_curv 0.7218 against the engine probe's
        4.9994, and in force an order cap of 2.5 against 2.15 -- opposite sides of the
        order-4 Lagrange weight 2.1875.
      * `n_probe_steps == max(4, cfg.probe_steps)` -- the K mismatch is inside the
        calibration, because `gap_probe` is `log(10)/(K-1)` and the rtol derivation
        reads `(gap_real/gap_probe)**2`.
    `cfg` is optional so a caller with no cfg in hand gets the provenance gate only;
    the engine passes it.

    See NOTES.md (round 16 root cause, round 17 fix correction) and
    scratch_latest_profile_rig.py, whose case C covers exactly this design.
    scratch_probe_slot_provenance_rig.py covers the two new gates, and runs the same
    census against the pre-fix tree so the gates are shown to be able to fail.
    """
    if not isinstance(prof, dict):
        return False
    try:
        m = int(prof.get("n_probe_steps") or 0)
    except (TypeError, ValueError):
        return False
    if m < _PROBE_MIN_USABLE_STEPS:
        return False
    # POINTS THE REACH EXTENSION APPENDED below the shipped band -- 0 when the profile predates
    # that work, when the reach is at the shipped floor, or when the key is absent.  Read here so
    # the geometry bound below compares the SHIPPED resolution rather than the walked total.
    try:
        _reach_extra = int(prof.get("probe_reach_n_extra") or 0)
    except (TypeError, ValueError):
        _reach_extra = 0
    # AND the profile must not be DEGENERATE.  This was the hole a live run fell
    # through: an m=7 profile with med_jump 2.2e-08 satisfies the step count, so
    # the engine adopted it and SKIPPED its own probe, which flipped the blend to
    # cond-only (w=1.00, ov 0.432 against 0.277) and produced a visibly
    # different image -- log_00068 of the 68-73 A/B set.  A profile whose own
    # dynamics read ~0 cannot serve the curvature consumers, and adopting one is
    # not "safe", it is SILENT.
    # Missing fields count as degenerate, because _probe_degenerate defaults both
    # thresholds to 0.0 -- so an unrecognised profile is re-probed rather than
    # trusted, which is the safe direction.
    if _probe_degenerate(prof):
        return False
    # ---- A COND PROFILE IS NOT A MODEL MEASUREMENT (added 2026-10-08) ----------
    # WHY THIS GATE EXISTS, measured rather than theorised.  `_PROFILE_BY_MODEL` is
    # written by the node-side COND probe as well as by the two model probes, so
    # `_latest_profile` hands the engine a per-prompt COND profile -- measured at
    # `cond_probe_steps` on a `cond_probe_resolution` latent.  This predicate then
    # accepted it, so the engine ADOPTED it, SKIPPED its own probe, and at
    # `aflops_engine`'s promotion site wrote it into its own `|s:` slot.
    #
    # MEASURED, build `363c8e1e269f`, `anima-base-v1.0` log_00018 against
    # log_00019/00020 -- one seed, one `sigmas_digest`, one prompt, one step count:
    #   engine's own probe   n_probe_steps 16, n_probe_res 48, patcher None,
    #                        med_curv 4.99936021203121, med_local_err 0.005736437859013677,
    #                        log_gap_probe 0.1535056728662697
    #   the adopted profile  n_probe_steps  8, n_probe_res 16, patcher clone,
    #                        x_shape [1,16,1,16,16], and BYTE-IDENTICAL to that run's
    #                        own cond slot: med_curv 0.7218011581686318,
    #                        med_local_err 0.008027324685826898,
    #                        log_gap_probe 0.3289407275705779
    # and in force: `rtol` 0.00625 vs 0.0075, `guard_floor` 0.13585 vs 0.1638,
    # `wmax_base` 2.15 vs 2.5.  That last pair is not cosmetic -- the cap is compared
    # against the order's max|Lagrange weight| (order 3 = 1.875, order 4 = 2.1875,
    # computed by `_uniform_halfstep_wmax` from the engine's own `_lagrange_weights`),
    # so those two caps admit DIFFERENT ORDERS.
    #
    # The two `log_gap_probe` values are the arithmetic proof that the grids differ:
    # for a flow schedule floored at 10 % of sigma_max the mean log gap is exactly
    # log(10)/(K-1) = 0.1535057 at K=16 and 0.3289407 at K=8.
    #
    # The white-list check has to be POSITIVE, not a blacklist of cond-only field
    # names: an older profile carries neither `gap` nor `focused`, and the whole point
    # is not to guess.  `probe_src` is stamped by the producer.
    if str(prof.get("probe_src") or "") != "model":
        return False
    if cfg is not None:
        # THE GEOMETRY MUST MATCH TOO, because the grid spacing is inside the
        # calibration: `scale = clamp((gap_real/gap_probe)**2, 0.04, 1)` and
        # `gap_probe` IS the grid's mean log gap, so an 8-step profile can move the
        # derived tolerance by nearly the whole clamp range.  The two expectations
        # below mirror `_run_probe`'s own `pres`/`K` (search `pres = max(16,`).
        try:
            want_res = max(16, int(cfg.get("probe_resolution", 48)))
            want_k = max(4, int(cfg.get("probe_steps", 10)))
        except (TypeError, ValueError):
            want_res = want_k = None
        have_res = prof.get("n_probe_res")
        if want_res is not None and have_res is not None:
            try:
                if int(have_res) != want_res:
                    return False
            except (TypeError, ValueError):
                return False
        # THE REACH MUST MATCH TOO.  A profile measured at 0.1 cannot serve a run
        # asking for 0.002: the deeper points are the measurement's whole purpose.
        # A DEEPER profile may serve a shallower request (its shipped band is a
        # prefix), so only `have > want` is refused.
        try:
            want_reach = float(_probe_reach_eff(cfg))
        except Exception:
            want_reach = None
        have_reach = prof.get("probe_reach_frac")
        if want_reach is not None and have_reach is not None:
            try:
                if float(have_reach) > want_reach + 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        if want_k is not None and (m - _reach_extra) > want_k:
            # THE REACH EXTENSION IS NOT A RESOLUTION CHANGE, and comparing the raw count treated
            # it as one.  `probe_smin_frac` below the shipped floor appends points at the BAND'S
            # OWN log spacing, so `gap_probe` -- the grid's mean log gap, which is the whole reason
            # this bound exists -- is UNCHANGED while `m` rises above `want_k`.  The bound then
            # rejected every profile measured on an extended reach, including the engine's OWN,
            # and `_profile_serves_model_probe` said False for all of them.  Measured on the
            # operator's runs (anima build ea9e27e, `probe_smin_frac` 0.05, logs 00112-00125):
            # all 8 engine model slots carried `n_probe_steps` 13 against `want_k` 10 with
            # `med_curv` 1.09-2.84 and `degenerate` False -- healthy profiles refused by this
            # ONE comparison -- so the engine re-probed on every generation: 45 model calls and
            # ~5.0 s of `model_probe_ms` per run, inside `SamplerCustomAdvanced` and before step 1.
            # Profiles that predate the stamp carry no `probe_reach_n_extra`, subtract 0, and are
            # judged exactly as before.
            # `m <= want_k`, NOT `m == want_k`, and the reason is MEASURED: the probe's grid is
            # built from the SCHEDULE -- `_probe_grid` returns the schedule's own positive sigmas
            # when there are no more than K of them, and the distilled variant spends half its
            # points in the top log-sigma half -- so the grid a probe really measures on can be
            # SHORTER than the requested K.  Measured on build `8c845858d8e7` (mklan
            # log_00008/00009): `probe_steps` 10, `len(sigma_curve)` 9.
            #
            # RETRACTION, same day: the commit that introduced this bound claimed the EQUALITY
            # form was what refused the node-side profile and cost the engine a second probe
            # (~34 calls).  THAT WAS WRONG, and the correction is load-bearing because it names a
            # different cause: the engine re-probes because `_latest_profile` returns the COND
            # entry that the node's cond probe wrote LAST, so the node-side MODEL profile is
            # never consulted at all.  See check S4 of scratch_probe_slot_provenance_rig.py,
            # which reproduces that path, and `_latest_profile`'s docstring.  This bound stands
            # on the grid evidence above; it was not the refusal.
            #
            # The PROVENANCE gate above is what closes the cond hole; this bound only refuses a
            # FINER measurement than this run would make, whose `log_gap_probe` (and therefore
            # the rtol derivation's `scale`) would not be this run's own.
            return False
    # THE PROFILE MUST HAVE BEEN MEASURED AT THIS RUN'S GUIDANCE (U05/1361).
    # `probe_cfg` is part of what was measured, so a profile taken at another CFG does
    # not describe this run.  Compared only when BOTH sides are known: every NODE-side
    # profile records it, the ENGINE-side ones did not until U18/8092, so an older
    # engine profile is judged exactly as before rather than refused wholesale.
    _live_cf = None
    if isinstance(cfg, dict):
        _live_cf = cfg.get("_probe_cfg_eff", cfg.get("_guider_live_cfg"))
    _prof_cf = prof.get("probe_cfg")
    if _live_cf is not None and _prof_cf is not None:
        try:
            if abs(float(_prof_cf) - float(_live_cf)) > 1e-3:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _latest_profile(model):
    """The most recent profile known for this model, for the ADOPTION fallback.

    THE EARLY RETURN IS A MEASURED BEHAVIOUR, NOT AN OPTIMISATION (recorded 2026-10-08).  The
    Autotuner runs its MODEL probe and then its COND probe, and the COND probe is what writes
    `_PROFILE_BY_MODEL[mk]` LAST.  Returning that entry on sight means the node's own MODEL
    profile -- which sits in `_PROBE_PROFILES` under `|s:node:...` and WOULD satisfy
    `_profile_serves_model_probe` -- is never consulted, so `probe_from_node` is False and the
    engine measures again (~34 model calls).  Measured on `8c845858d8e7` (mklan
    log_00008/00009): `model_probe_state: ran` with `probe_calls_node_side` non-zero and a
    serving node profile in the store; reproduced as check S4 of
    scratch_probe_slot_provenance_rig.py.

    ADOPTING THE NODE PROFILE INSTEAD IS A DELIBERATE TRADE, NOT AN OVERSIGHT.  Across one
    restart on those same runs the ENGINE's model probe reproduced bit-for-bit (its slot differs
    only in `probe_call.puuid`), the engine's cond probe moved in 12 leaves (its `endgame`
    quantiles), and the NODE-side model probe moved in 280 -- so today's path consumes the more
    reproducible of the two measurements and pays ~34 calls for it.  Changing that preference
    means importing the noisier measurement into the run, which is a decision with evidence on
    both sides rather than a bug fix.
    """
    try:
        mk = _model_cache_key(model)
    except Exception:
        return None
    prof = _PROFILE_BY_MODEL.get(mk)
    if prof is not None:
        return prof
    hit = None
    for k, v in _PROBE_PROFILES.items():
        if k == mk or k.startswith(mk + "|s:"):
            hit = v
    return hit


def _node_model_profile(mk, cfg):
    """The NODE-side MODEL measurement for this model at these probe parameters, or None.

    WHY THIS EXISTS.  The Autotuner files its model probe under a key of its own shape --
    `<mk>|s:node:<_node_probe_param_sig>|<cfg>` (written at the `_PROBE_PROFILES` assignment in the
    node-side probe) -- and that key is built from the schedule the node was GIVEN.  The engine's
    `pkey` is built from the schedule the node RETURNED.  While the node returns its input
    unchanged those agree by accident; the moment anything re-places the steps they CANNOT agree,
    because the returned grid is derived from the measurement the input grid was probed for.

    So the exact-key lookup misses on every run, `_latest_profile` hands back whatever the node's
    COND probe wrote last, that is not a model measurement, and the engine measures again.  This
    finds the measurement that already exists instead.

    MEASURED on anima build ea9e27e (logs 00112-00125, `probe_schedule` ON): `model_probe_state:
    ran` on every run, 45 engine probe model calls and ~5.0 s of `model_probe_ms` per generation.
    With the Autotuner disconnected the same window collapses -- not only because the node is gone,
    but because `probe_enabled` defaults False and the ENGINE's own pass is switched off with it.

    THIS IS NOT WHAT CAUSED THOSE RUNS, and the correction is worth carrying because it was the
    first hypothesis and it was wrong.  The store DID hold 8 `src=model` slots -- but every one of
    them was REFUSED by the geometry bound in `_profile_serves_model_probe` (`n_probe_steps` 13
    against `want_k` 10), because the reach extension raises the walked count above `probe_steps`.
    So `_latest_profile`'s answer was never the deciding factor: had it returned a model slot, the
    bound would have refused that too.  See the fix and the census at that bound.  This helper
    remains as hardening -- the node slot IS keyed differently and `_latest_profile` DOES return
    whatever was written newest, which is a cond entry -- but it is not the repair.

    THE PARAMETERS ARE MATCHED EXACTLY, and a near-miss is NOT adopted: the node slot names the
    probe steps/resolution it was measured at, and a measurement taken at different settings is
    not this run's evidence.  `probe_src` is required to be `model` for the same reason the
    exact-key path requires it -- the promotion path can leave per-prompt COND values in a model
    slot, measured on anima log_00019/00020.
    """
    try:
        pref = str(mk) + "|s:node:" + str(_node_probe_param_sig(cfg)) + "|"
    except Exception:
        return None
    # A NODE MODEL PROBE IS PER-CFG (U05/1361).  This loop used to take the LAST prefix
    # match, so a run at cfg 3 found the profile measured at cfg 5 -- which the adoption
    # gate then refused, and the engine re-probed on every prompt.  MEASURED on the moss
    # runs (`moss_image_base_int8_convrot`, 50-step, 128-ch): `model_probe_state: ran` on
    # every run, 12.6-17.3 s of `model_probe_ms` per generation, while a node-side model
    # probe at the SAME cfg was sitting in the store and would have been served.
    # Prefer the match at the live CFG; if there is none, return None rather than a
    # measurement taken at another guidance scale.
    _live_cf = None
    if isinstance(cfg, dict):
        _live_cf = cfg.get("_probe_cfg_eff", cfg.get("_guider_live_cfg"))
    hit = None
    for k, v in _PROBE_PROFILES.items():
        if not (str(k).startswith(pref)
                and str((v or {}).get("probe_src") or "") == "model"):
            continue
        if _live_cf is not None and v.get("probe_cfg") is not None:
            try:
                if abs(float(v["probe_cfg"]) - float(_live_cf)) > 1e-3:
                    continue
            except (TypeError, ValueError):
                continue
        hit = v
    return hit


def _lagrange_weights(us, u_t):
    n = len(us)
    lam = []
    for j in range(n):
        d = 1.0
        for k in range(n):
            if k == j:
                continue
            d *= (us[j] - us[k])
        lam.append(1.0 / d if d != 0.0 else 0.0)
    for j in range(n):
        if abs(u_t - us[j]) < 1e-15:
            ws = [0.0] * n
            ws[j] = 1.0
            return ws
    cs = []
    for j in range(n):
        cs.append(lam[j] / (u_t - us[j]))
    s = sum(cs)
    if abs(s) < 1e-300:
        return [1.0 / n] * n
    return [c / s for c in cs]

def _mid_x0(pts, u_mid, order):
    sel = pts[-min(order, len(pts)):]
    us = [p[0] for p in sel]
    if len(set(us)) != len(us):
        return sel[-1][1], 0.0
    ws = _lagrange_weights(us, u_mid)
    out = sel[0][1] * ws[0]
    for w, p in zip(ws[1:], sel[1:]):
        out = out + p[1] * w
    return out, max(abs(w) for w in ws)
_ORD_BIG = 1e30


# ---------------------------------------------------------------------------
# THE EXTRAPOLATION-ORDER GATE (`wmax_bound`).
#
# The engine rejects an order when its largest Lagrange weight exceeds a scalar cap.  That
# cap is a PROXY for the quantity that actually matters -- the extrapolation's truncation
# error, whose exact size is the Lagrange remainder
#
#     |err_k(u_t)| = |f^(k)(xi)| / k! * prod_j |u_t - u_j|
#
# and the scalar form the cap came from is only its UNIFORM case: on a uniform grid with
# u_t half a step past the last node, prod_j |u_t - u_j| / k! == c_k * h^k, verified to
# 1.55e-15 in scratch_extrap_remainder_rig.py -- which also verified that `_mid_x0` IS the
# exact interpolating polynomial (0.0 relative deviation against an independently computed
# Lagrange basis).  One scalar cannot describe a real schedule: the consecutive
# |d log sigma| gaps of the operator's own runs span 28x-76x, so no single h is the gap of
# any particular step.  `_extrap_geom` reads the factor off the ACTUAL nodes instead --
# the same equation without the uniformity assumption, and it is free, because the gate
# already holds the nodes.  Measured on the corpus' own dk_scale/rtol/grids: the shipped
# scalar form admits an order at 0/39 (anima) and 0/15 (krea) extrapolating steps -- i.e.
# it refuses everything and the arm is inert -- while this form admits at 36/39 and 11/15,
# refusing exactly where the grid stretches (du >= 0.25 / >= 0.19).
#
# COMPOSITION, NOT REPLACEMENT.  The arm is `weight test AND remainder test`, so it can
# only ever be TIGHTER than the shipped one: it refuses where the truncation error at THIS
# step's own geometry exceeds the tolerance, and leaves every other decision where it was.
# An order whose derivative scale was not measured (`dk_scale` omits any k that needs more
# samples than the probe grid has) falls back to the weight test alone: we certify, we do
# not assume.
#
# THE RESIDUAL ASSUMPTION, stated where it is consumed: `dk_scale` is measured on the
# PROBE grid and is a trajectory statistic, so it is not a per-step upper bound -- on an
# analytic ground truth the shipping median under-bounds 58 of 204 (step, order) cases,
# by up to 578-1807x at individual steps.  `wmax_budget` is where that residual, and the
# operator's own judgement, enters.  It scales ONLY this gate's tolerance and never
# `rtol`, which the corrector also consumes -- scaling that would put two decisions
# behind one widget.
#
# Rig: scratch_extrap_remainder_rig.py (15/15), output committed alongside it.
# ---------------------------------------------------------------------------
_ORDER_GATE_AB = {"off": 0, "on": 0, "pass": 0, "refuse": 0, "unmeasured": 0,
                  "fallback": 0}
# FLOAT32 EPSILON, READ FROM THE DTYPE THE DIFFERENCES ARE COMPUTED IN rather than written as a
# literal, so the unresolved-difference floor in `_pixel_order_bound` cannot drift from the arithmetic
# it describes.  `_pixel_innovations` and the difference stencils below all work in float32.
_EPS_F32 = float(torch.finfo(torch.float32).eps)
# THE 10-90 SPAN OF A UNIT NORMAL, IN SIGMA.  A curve's p10..p90 spread is 2*Phi^-1(0.9)
# = 2.563 sigma, NOT 1.349 -- that is the inter-quartile range, and dividing an 80 % span
# by it inflated every residual sigma by 2.563/1.349 = 1.90x (U11/4347), which made the
# probes' disagreements look 1.9x smaller than they are in the weight update.  Computed
# from the distribution rather than typed, so it cannot drift from the quantiles it
# describes.
_NORM_SPAN_10_90 = 2.0 * statistics.NormalDist().inv_cdf(0.9)


def _extrap_geom(us, u_t):
    """prod_j |u_t - u_j| / k! -- the geometric factor of the Lagrange remainder.

    Identical to c_k * h^k on a uniform grid with u_t half a step past the last node
    (rig check 1c), so using it is not a change of equation, it is the same equation
    without the uniformity assumption.
    """
    n = len(us)
    if n < 2:
        return 0.0
    p = 1.0
    for u in us:
        p *= abs(float(u_t) - float(u))
    return p / float(math.factorial(n))


def _order_gate(pts, u_mid, order, wmax, limit, cfg, space="log_sigma"):
    """(admitted, info) for one candidate order.  `info` is None on the shipped path.

    Shipped path (the arm OFF, or this order's scale unmeasured): the weight test alone,
    byte-identical to the code before this gate existed.  Arm ON: the weight test AND
    `(1 - exp(-du)) * geom * dk[k] <= rtol * wmax_budget`, where `du` is recovered from
    the midpoint every call site passes (`2*|u_mid - u_last|`).

    THE GATE'S STATISTICS ARE LOG-SIGMA QUANTITIES, whatever space the nodes arrived
    in: `dk_scale` is measured per unit log-sigma (`_measure_dk_scale`), the scalar
    derivation consumes it against `log_gap`, and the step factor `1 - exp(-h)` equals
    `1 - s2/s` only when `h` is a LOG gap.  `space` therefore says which space `pts`
    and `u_mid` are in, and they are converted to log-sigma before anything is
    computed.  Without that, on the sigma-space branch the geometry was sigma^k while
    `dk` stayed per log-sigma^k, so the bound was wrong by the product of the selected
    sigmas -- at the verifier's worked case 0.173x the correctly dimensioned value at
    order 2 and 0.029x at order 5, which ADMITS orders the true remainder refuses.
    The conversion is exact and needs no per-order derivative transform, because the
    remainder is then the one of x0 as a function of log sigma -- the same function
    `dk_scale` describes.
    """
    if not bool(cfg.get("wmax_bound", False)):
        _ORDER_GATE_AB["off"] += 1
        return (float(wmax) <= float(limit)), None
    try:
        sel = pts[-min(int(order), len(pts)):]
        us = [float(p[0]) for p in sel]
        um = float(u_mid)
        if str(space) != "log_sigma":
            us = [math.log(max(u, 1e-12)) for u in us]
            um = math.log(max(um, 1e-12))
        dk = cfg.get("_dk_scale")
        tol = cfg.get("rtol")
        k = int(order)
        dscale = None
        if isinstance(dk, dict):
            dscale = dk.get(k, dk.get(str(k)))
        if dscale is None or tol is None:
            _ORDER_GATE_AB["unmeasured"] += 1
            return (float(wmax) <= float(limit)), None
        budget = float(cfg.get("wmax_budget", 1.0) or 1.0)
        tol_eff = float(tol) * max(budget, 1e-6)
        du = 2.0 * abs(um - us[-1])
        prop = 1.0 - math.exp(-max(du, 0.0))
        geom = _extrap_geom(us, um)
        bound = geom * float(dscale)
        step_err = prop * bound
        ok = (float(wmax) <= float(limit)) and (step_err <= tol_eff)
        _ORDER_GATE_AB["on"] += 1
        _ORDER_GATE_AB["pass" if ok else "refuse"] += 1
        return bool(ok), {"k": k, "geom": geom, "du": du,
                          "prop": prop, "dk": float(dscale), "bound": bound,
                          "step_err": step_err, "tol_eff": tol_eff,
                          "wmax": float(wmax), "limit": float(limit),
                          "bound_ok": bool(step_err <= tol_eff),
                          "weight_ok": bool(float(wmax) <= float(limit)),
                          "gate_space": "log_sigma"}
    except Exception:
        _ORDER_GATE_AB["fallback"] += 1
        return (float(wmax) <= float(limit)), None

# The series/exp branch switch in `_aflops_step`.  The shipped value is the
# hand-set 1e-4; the DERIVED value is sqrt(12*eps) of the working dtype, because
# the truncated Taylor branch's dominant relative error is its phi2 term,
# (z^2/24)/(1/2) = z^2/12, so the switch belongs where that meets the precision.
# scratch_step_constants_rig.py verified the z^2/12 law numerically (law/series
# ratio 0.997..1.001 over z = 1e-5..1e-2) and gives 1.196e-3 for float32 and
# 5.162e-8 for float64.  1e-4 is 12x conservative for f32, 1937x too large for f64.
_AFLOPS_SERIES_SWITCH = 1e-4


def _derived_series_switch(dtype):
    """sqrt(12*eps) for the working dtype -- the derived form of the switch."""
    try:
        if dtype is not None and getattr(dtype, "is_floating_point", False):
            eps = float(torch.finfo(dtype).eps)
        else:
            eps = 1.1920929e-07
    except Exception:
        eps = 1.1920929e-07
    return math.sqrt(12.0 * eps)


# ---------------------------------------------------------------------------
# A/B REACHABILITY INSTRUMENTATION.
#
# WHY THIS EXISTS.  The derived_ab bundle shipped as a NO-OP on every probed run
# and nobody noticed for two rounds of A/B: five model pairs came back with ZERO
# differing per-step fields.  The cause was that one of its three sites was gated
# on a condition that never held (`tol == ENGINE_DEFAULTS["rtol"]`, while every
# probed run derived a different rtol), and the other two are guarded by
# `eff >= 2` / a |z| band that real runs frequently never enter.  A flag that is
# wired to nothing looks EXACTLY like a flag whose effect is below the noise
# floor -- and the corpus cannot tell those apart.
#
# So every A/B site now counts its own invocations and the runs that matter.
# `ab_reach` is stamped into `_LAST_CFG` and therefore into every saved log, so a
# log states outright which A/B paths fired instead of leaving it to be inferred.
# A rig (scratch_ab_reach_rig.py) asserts each site is REACHABLE, i.e. that some
# configuration actually fires it; a site that can never fire is reported rather
# than believed to work.
#
# Cost: one dict lookup + increment per site per step.  Negligible against a
# model call, and it is the difference between "no effect" and "not connected".
# ---------------------------------------------------------------------------
_AB_REACH = {}
# The counters of the CURRENT run that were recorded BEFORE the engine started -- the node-side
# probe runs first and its hits belong to this run, not to the session.  Written at engine entry by
# `_ab_reach_reset` and at engine exit by `_ab_reach_end`; the two docstrings state the per-run
# window, and neither value is read by any sampling decision.
_AB_REACH_CARRY = {}


def _ab_hit(name, val=None):
    """Count one A/B site invocation.  `val` keeps the last value seen."""
    r = _AB_REACH.get(name)
    if r is None:
        _AB_REACH[name] = r = [0, None]
    r[0] += 1
    if val is not None:
        r[1] = val


def _ab_hit_max(name, val):
    """Count an invocation and keep the running MAXIMUM instead of the last value.

    Needed for guard-style sites, where "did it ever trip" is the question and
    the last step's value says nothing: the envelope guard compares a ratio
    against a bound, so the informative record is the largest ratio seen.
    """
    r = _AB_REACH.get(name)
    if r is None:
        _AB_REACH[name] = r = [0, val]
    r[0] += 1
    try:
        if r[1] is None or float(val) > float(r[1]):
            r[1] = val
    except (TypeError, ValueError):
        r[1] = val


def _ab_reach_reset():
    """Start a fresh per-run record WITHOUT losing the node-side phase.

    WHY THIS CARRIES RATHER THAN CLEARS.  The reset exists so a run's record describes THAT run and not
    the session's running total (the comment at the call site records the cumulative-counter bug it
    fixed).  But the NODE-side probe runs BEFORE `aflops_engine` is entered -- that is its whole point,
    to have the profile ready for the Scheduler -- so its hits are part of THIS run's record and clearing
    the dict silently deleted them.

    MEASURED CONSEQUENCE, which is why this is a fix and not a tidy-up: `probe.noise_survival_ran` and
    `probe.noise_survival_skipped` live in `_run_probe`, which the node-side model probe calls; across
    **668 corpus logs not one carries either key**, because every fresh probe happens node-side and every
    node-side hit was wiped before the report was written.  The gate that skips ten model calls was
    therefore invisible -- the same "a switch wired to nothing is indistinguishable from one whose
    effect is below the floor" trap (rule 9) that this instrumentation exists to prevent.

    THE OTHER HALF IS `_ab_reach_end`, CALLED WHEN THE RUN CLOSES.  This reset REPLACES the carry with
    whatever is in `_AB_REACH` at engine entry, which is only correct if `_AB_REACH` holds nothing but
    this run's pre-engine hits.  It did not: nothing emptied `_AB_REACH` when a run finished, so the
    previous run's engine counters were still in it and were carried into this run's log -- a site that
    fires once per engine entry read 2, and a run could report a counter its own configuration had
    switched off.  `_ab_reach_end` is what makes this function's copy mean "this run's node-side phase".
    """
    global _AB_REACH_CARRY
    try:
        _AB_REACH_CARRY = {k: [int(v[0]), v[1]] for k, v in _AB_REACH.items()}
    except Exception:
        _AB_REACH_CARRY = {}
    _AB_REACH.clear()


def _ab_reach_end():
    """Close this run's A/B reach record, so the NEXT run starts from an empty one.

    WHY THIS EXISTS.  `_ab_reach_reset` copies whatever `_AB_REACH` holds into the carry, because the
    sites that fire BEFORE the engine -- the Autotuner's node-side probe -- belong to the run that is
    about to start.  Nothing emptied `_AB_REACH` when a run FINISHED, so at the next run's reset the dict
    still held the PREVIOUS run's engine counters and the carry handed them to the new run's log.
    Measured on the saved corpus (scratch_fix_logging_1_3_out.txt section 4b): a site that fires exactly
    once per engine entry recorded 2 instead of 1 in 899 of the 1397 logs carrying a record, 138 runs
    reported a counter their own configuration had switched off (for example `rtol.arm_probe` with the
    measured-tolerance arm OFF), and 884 runs exceeded their own step count on a once-per-step counter.

    WHAT IT DOES.  The run's whole record -- the pre-engine phase plus the engine phase -- is left in the
    carry, so a snapshot taken after the engine returns still describes the run that just ended, which is
    what the run's own log and every rig read.  `_AB_REACH` itself goes back to empty, so the counters of
    the NEXT run's pre-engine phase accumulate on their own and the next reset carries only those.

    THIS IS BOOKKEEPING, NOT SAMPLING: the dict it writes is the log's `ab_reach` record, and no
    sampling decision in this file reads it.
    """
    global _AB_REACH_CARRY
    try:
        _AB_REACH_CARRY = _ab_reach_snapshot()
    except Exception:
        _AB_REACH_CARRY = {}
    _AB_REACH.clear()


def _ab_reach_snapshot():
    """{name: [hits, last_value]} with `hits` first -- cheap, JSON-safe.

    Includes the pre-engine carry (see `_ab_reach_reset`), SUMMED with anything the engine hit for the
    same site, because a hit count is a count of invocations and both phases belong to one run.  The
    last-value slot keeps the engine's own value when it has one, since that is the later phase.

    Called both by the run's own log (in `_aflops_engine_body`, before `_aflops_close_run`) and by
    `_ab_reach_end` itself to close the record; after the close it returns that run's whole record,
    because the carry then holds it.
    """
    out = {k: [int(v[0]), v[1]] for k, v in _AB_REACH.items()}
    try:
        for k, v in _AB_REACH_CARRY.items():
            if k in out:
                out[k][0] += int(v[0])
                if out[k][1] is None:
                    out[k][1] = v[1]
            else:
                out[k] = [int(v[0]), v[1]]
    except Exception:
        pass
    return {k: out[k] for k in sorted(out)}


_AFLOPS_LAM_MIN = -3.0
_AFLOPS_LAM_MAX = 0.0
_AFLOPS_X0M_ENV = 1.3
# `_ORD_EMA` WAS REMOVED 2026-10-04 (rule 26 -- a vestigial identifier is worse than none).  The EMA
# weight is a per-run value read from cfg (`ord_ema`, `ENGINE_DEFAULTS`), so a module constant holding a
# second copy of it would read as the live default to anyone who grepped for it.  Its three use sites now
# read `_ord_ema`; the value the maths used is stamped as `order.ema_in_force`.
_ORD_Z = 3.5          # robust outlier threshold (modified z-score)
_ORD_R_FLOOR = 1.5    # novelty must ALSO beat the pixel's own envelope
_ORD_ENV_DECAY = 0.7  # motion-envelope decay (was the order_jema_decay knob)
_GATE_ETA = 0.15      # integral rate (per steered step)
_GATE_TARGET = 0.10   # required net efficacy for full authority
_GATE_FLOOR = 0.05    # exploration floor when gated down
_LF_ACT_M = 0.3       # |m| > 0.3 * amp counts as steered (same as C1)
_LF_ACT_RHO = 0.05    # refresh fraction that counts as re-drawn material

def _pixel_innovations(hist_pts, u_t, x0_cur, p_max):
    out = []
    n = len(hist_pts)
    for p in range(1, int(p_max) + 1):
        if n < p:
            out.append(None)
            continue
        sel = hist_pts[-p:]
        us = [q[0] for q in sel]
        if len(set(us)) != len(us):
            out.append(None)
            continue
        ws = _lagrange_weights(us, u_t)
        pred = sel[0][1].float() * ws[0]
        for w, q in zip(ws[1:], sel[1:]):
            pred = pred + q[1].float() * w
        out.append((pred - x0_cur.float()).pow(2).mean(dim=1, keepdim=True)
                   .sqrt())
    return out

def _order_bound_target(o_innov, o_bound, blend):
    """The per-pixel order map after applying the Runge criterion at strength `blend`.

    `blend` is the dial between the two maps the criterion can produce:
      0.0  the innovation map, unchanged       -- the `order_by_bound=OFF` arm exactly
      1.0  the bound's own map                 -- the ON arm exactly
      else the two mixed IN ORDER UNITS and rounded, so a pixel whose innovation order
           is 1 and whose bound is 4 reads 2 at 0.5.
    The endpoints are exact by construction, which is what makes the dial checkable
    against the two arms it interpolates; the clamp keeps a stray value from inverting.
    """
    b = min(max(float(blend), 0.0), 1.0)
    if b >= 1.0:
        return o_bound
    if b <= 0.0:
        return o_innov
    return torch.round((1.0 - b) * o_innov + b * o_bound.float())


def _pixel_order_bound(pts, u_mid, prop, allowed, tol_eff, kmax_cap=8):
    """Per-pixel HIGHEST VALID order, from the Lagrange remainder -- the RUNGE criterion.

    `prop` IS THE STEP FACTOR `1 - s2/s`, PASSED IN RATHER THAN DERIVED FROM A GAP.  The
    nodes arrive in log-sigma OR in sigma depending on `use_space`, so a gap `|u2 - u|`
    means `log(s/s2)` on one branch and `s - s2` on the other -- and `1 - exp(-(s - s2))`
    is not `1 - s2/s`: it is larger by ~1/s, i.e. 2x at sigma 0.5 and 10x at sigma 0.1.
    Measured on this build's corpus, `space` is `sigma` on 1899 of 2210 steps, so a
    gap-derived factor was wrong on 86% of them, and wrong in the LOOSENING direction
    (`tol_x0 = tol / prop` came out too small, admitting orders the step error does not
    justify).  The caller knows `s` and `s2` exactly, so it passes the factor and there is
    nothing left to infer.

    WHY THIS EXISTS.  The shipped ladder picks the order that best PREDICTS x0
    (`_pixel_innovations`' argmin).  That answers a different question from the one the order
    exists for: it is dominated by pixel-level noise, which the higher orders AMPLIFY (their
    Lagrange weights run 1.5, 1.875, 2.19, 3.28, 5.41 for orders 2-6), so on a real model it
    settles near order 1 -- measured on the operator's own run, `ord_want` 1.03-1.23 with
    `|ord_want - ord_mean| <= 5e-4`, i.e. nothing downstream clamps and no cap, budget or gate
    can move it.  The question that matters is VALIDITY: for THIS pixel, how high can the
    order go before the polynomial extrapolation stops describing the trajectory and starts
    oscillating (Runge)?  That is exactly the Lagrange remainder

        |err_k| <= ( prod_j |u_t - u_j| / k! ) * |d^k x0/du^k| / |x0|

    and every factor is measurable from data already in hand: `prod` from the nodes, and
    `|d^k x0/du^k|` from the k-th difference over this pixel's OWN history -- the engine keeps
    up to `hist_cap` samples, so k <= 6 on a 16-step run and higher on a 40-step one.  No
    extra NFE.

    WHY PER-PIXEL IS THE POINT: a pixel with a kink or per-pixel junk blows up its own k-th
    difference and its bound fails at low k, so it keeps order 1; a pixel on a smooth curve
    keeps a small difference and is admitted at high k.  The order map is therefore spatially
    varying by construction, which is what "some parts need order 6, others need order 1
    because of Runge" means as a measurement.

    `allowed` is the step's own admission list (weights + the remainder gate when it is on):
    orders this step refuses for everyone are refused here too, so this criterion composes
    with the safety path rather than bypassing it.

    Returns (order_tensor or None, info).  The tensor is per-pixel, in [1, kmax].
    """
    try:
        n = len(pts)
        if n < 3:
            return None, {"reason": "fewer than 3 samples: no second difference"}
        x0_cur = pts[-1][1].float()
        ref = x0_cur.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        prop = float(prop)
        if not (prop > 0.0):
            return None, {"reason": "step factor prop is not positive"}
        # `tol` bounds the STEP error; the remainder bounds the x0 error that feeds it
        tol_x0 = float(tol_eff) / prop
        us = [float(p[0]) for p in pts]
        _h1 = abs(us[-1] - us[-2])
        # the per-pixel first-difference rate, for the de-bias (see _measure_dk_scale)
        r1 = ((pts[-1][1].float() - pts[-2][1].float())
              .pow(2).mean(dim=1, keepdim=True).sqrt()
              / max(_h1, 1e-12) / ref)
        kmax = int(max(2, min(int(kmax_cap), n - 1, len(allowed) + 1)))
        order = torch.ones_like(ref)
        per_k = []
        # ---- THE REMAINDER SCALE IS EVALUATED AT THE FAR END.  THIS IS THE ONLY PATH. ----------
        # `scratch_rtol_bound_rig.py` measured the previous form against EXACTLY constructed
        # interpolants and found it was not an upper bound: it held on 7 of 12 analytic (case, k)
        # pairs, and when the k-th difference VANISHED it read ~1e-15 against a true error of 1.168 --
        # so the order was admitted with certainty and no `tol` could refuse it, because 0 <= tol for
        # every positive tol.
        #
        # WHY.  The Lagrange remainder needs f^(k)(xi) for some xi in [u_last_node, u_mid], and for
        # EXTRAPOLATION that interval reaches BEYOND the newest node.  Estimating f^(k) AT the newest
        # node was therefore systematically low when f^(k) grows across the reach: measured at
        # 0.706 / 0.441 / 0.153 of the far-end remainder for c = 0.5 / 1.0 / 2.0.
        #
        # THE CHANGE: a first-order extrapolation of f^(k) from the newest node out to the point being
        # extrapolated to, using the NEXT difference --
        #     scale_eff = scale_k + |u_mid - u_newest| * scale_{k+1}
        # Both terms are non-negative (the engine takes abs), so this can only RAISE the bound: it can
        # never admit an order the old form refused, only refuse more.  It also closes the vanishing
        # case for free, because when `scale_k` is 0 the second term still carries the next difference.
        #
        # RIGGED at 7/12 -> 11/12 pairs, vanishing case 1.102e-15 -> 5.419e+00.  THE RESIDUAL IS KNOWN
        # AND NOT HIDDEN: one pair (exp c=2.0, k=2) is still 25% short, because a first-order
        # correction fixes the trend and not the curvature.
        #
        # IT WAS A/B'd AND THE SWITCH WAS REMOVED (2026-10-04).  Operator: *"with removal we both mean
        # also implement the new code, right?  Old code goes too."*  The A/B measured that this
        # correction is INERT on the configurations tried -- it changed the criterion's own admission
        # on up to a third of steps (one case to zero) and the latent was BIT-IDENTICAL -- which is a
        # statement about the criterion not being the binding constraint, NOT about this change being
        # wrong.  See NOTES "THE SWITCH IS LIVE AND THE OUTCOME IS BIT-IDENTICAL" and "RIG 2".
        #
        # `no_history` is the one case the correction cannot reach: at k == kmax the (k+1)-th
        # difference needs one more node than the history holds.  It is recorded per order rather
        # than silently skipped.
        #
        # NOT A CACHED PATH (AGENTS rule 28): this function has exactly ONE call site, in the sampling
        # loop, and nothing probe-derived reads its result.
        def _scale_of(kk):
            """`(dk / rho**kk, dk, h)`: the estimate of f^(kk) at the NEWEST node.

            The RAW `dk` and the spacing travel with it because the top order needs them: at
            `k == kmax` the correction's input does not exist, and whether the near-end scale may be
            trusted at all depends on whether `dk` is resolved above the difference's own rounding
            floor.  Returning them costs nothing -- they are already computed here.
            """
            _sel = pts[-(kk + 1):]
            if len(_sel) < kk + 1:
                return None
            _sus = [float(p[0]) for p in _sel]
            _h = sum(abs(_sus[i + 1] - _sus[i]) for i in range(kk)) / float(kk)
            if not (_h > 0.0):
                return None
            _acc = None
            for j in range(kk + 1):
                _t = _sel[kk - j][1].float() * float(((-1) ** j) * math.comb(kk, j))
                _acc = _t if _acc is None else _acc + _t
            _dk = _acc.pow(2).mean(dim=1, keepdim=True).sqrt() / (_h ** kk) / ref
            # DE-BIAS as in _measure_dk_scale: for x0 = e^{cu} a raw k-th difference over h is
            # low by rho(ch)^k.  Without it the bound is conservative by that factor.
            _ch = r1 * _h
            _rho = torch.where(_ch > 1e-9,
                               (1.0 - torch.exp(-_ch)) / _ch.clamp_min(1e-9),
                               torch.ones_like(_ch))
            return _dk / _rho.pow(kk), _dk, _h

        # COMPUTED ONCE for k = 2 .. kmax+1.  The +1 is the correction's own input, so precomputing
        # avoids computing each order's difference twice (the loop would otherwise recompute k+1's
        # when it reaches k+1).  One extra order of cheap integer-weighted sums, no extra model call.
        _scale_by_k = {}
        for _kk in range(2, kmax + 2):
            _s = _scale_of(_kk)
            if _s is not None:
                _scale_by_k[_kk] = _s

        corrected_n = 0
        for k in range(2, kmax + 1):
            if (k - 1) < len(allowed) and not allowed[k - 1]:
                per_k.append([k, None, None, "refused for this step"])
                continue
            sel = pts[-(k + 1):]
            sus = [float(p[0]) for p in sel]
            h = sum(abs(sus[i + 1] - sus[i]) for i in range(k)) / float(k)
            if not (h > 0.0):
                per_k.append([k, None, None, "degenerate spacing"])
                continue
            scale = _scale_by_k.get(k)
            if scale is None:
                per_k.append([k, None, None, "unmeasurable difference"])
                continue
            scale, _dk_raw, _h_k = scale
            resolved = None
            scale_next = _scale_by_k.get(k + 1)
            if scale_next is not None:
                scale = scale + abs(float(u_mid) - sus[-1]) * scale_next[0]
                corr = "applied"
                corrected_n += 1
            else:
                # NOT ENOUGH HISTORY for the (k+1)-th difference -- this is `k == kmax` at the
                # boundary.  The correction cannot reach the top order, so it is judged on the
                # near-end scale alone and the step SAYS SO (`no_history`) rather than pretending
                # otherwise.
                #
                # AND ONE CASE IS REFUSED RATHER THAN JUDGED, because the near-end scale is not merely
                # imprecise there but carries NO information: a k-th difference that has cancelled to
                # the arithmetic noise floor says nothing about f^(k), so the bound reads ~0 while the
                # true error can be O(1).  Measured IN THE ENGINE by `scratch_order_kmax_hole_rig.py`:
                # in that case the bound is 0 and the true relative error 1.168 -- 73.8x outside the
                # tolerance the order was admitted against, at order 4 of a 5-node history.
                #
                # THE FLOOR IS DERIVED, NOT CHOSEN.  `_dk` is a float32 sum of k+1 terms whose
                # coefficients sum to 2^k in absolute value, so its rounding error is bounded by
                # (k+1) * eps * 2^k * ref -- i.e. (k+1) * eps * 2^k / h^k after the same /h^k/ref
                # normalisation `_dk` itself uses.  Anything at or below that is indistinguishable
                # from zero at this precision.
                #
                # A RESOLVED difference keeps the previous behaviour exactly (judged on the near-end
                # scale), so a run whose top difference is healthy is BIT-IDENTICAL to before this
                # change -- the refusal fires only where the old path was demonstrably unsound.
                corr = "no_history"
                resolved = _dk_raw > (((k + 1) * _EPS_F32 * (2.0 ** k)) / (_h_k ** k))
            geom = 1.0
            for p in pts[-k:]:
                geom *= abs(float(u_mid) - float(p[0]))
            geom /= float(math.factorial(k))
            bound = geom * scale
            ok = (bound <= tol_x0)
            if resolved is not None:
                ok = ok & resolved
            order = torch.where(ok, torch.full_like(order, float(k)), order)
            per_k.append([k, round(float(bound.mean()), 8),
                          round(float(ok.float().mean()), 4), corr])
        # RULE 9: the correction must PROVE it runs.  `corrected_n` counts the orders this step
        # actually corrected (as opposed to `no_history`), and the per-order `corr` marker beside it
        # says which -- so "the correction did nothing here" can be told from "it never applied".
        try:
            _ab_hit("order_bound.corrected", float(corrected_n))
            _ab_meta = {"orders_corrected": corrected_n, "kmax": kmax}
        except Exception:
            _ab_meta = None
        try:
            _dk_run = {int(k): round(float(v[0].median()), 8)
                       for k, v in _scale_by_k.items()}
        except Exception:
            _dk_run = None
        return order, {"kmax": kmax, "n": n, "tol_x0": round(tol_x0, 8),
                       "per_k": per_k, "bound_fix": _ab_meta,
                       "dk_run": _dk_run}
    except Exception as e:
        return None, {"reason": "exception: {}".format(e)}


def _grad_weighted_norm(diff, x0, strength=5.0):
    gx = (x0[..., 1:, :] - x0[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    gy = (x0[..., :, 1:] - x0[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    gx = F.pad(gx, (0, 0, 0, 1))
    gy = F.pad(gy, (0, 1, 0, 0))
    g = gx + gy
    g = g / g.mean().clamp_min(1e-8)
    w = (1.0 + float(strength) * g).clamp(1.0, 10.0)
    dw = diff * w
    return float(dw.norm() / (x0 * w).norm().clamp_min(1e-8))
_PERCEPTUAL_DOG_GAIN = math.sqrt(327680.0 / 329.0)


def _perceptual_error_norm(diff, ref):
    sp = diff.shape
    d2 = diff.reshape(-1, 1, sp[-2], sp[-1]).float()
    g_small = _gaussian_blur_separable(d2, 3)   # [1,2,1]/4       (radius 1)
    g_large = _gaussian_blur_separable(d2, 5)   # [1,4,6,4,1]/16  (radius 2)
    d_hp = ((g_small - g_large) * _PERCEPTUAL_DOG_GAIN).reshape(sp)
    err_lo = diff.flatten().float().norm()                  # ||diff||
    err_hi = d_hp.flatten().float().norm()                  # ||DoG(diff)||
    ref_norm = ref.flatten().float().norm().clamp_min(1e-8)
    return float(torch.sqrt(err_lo * err_lo + 0.5 * err_hi * err_hi) / ref_norm)

def _err_norm(a, b, mode, x0_ref=None, strength=5.0):
    diff = (a - b).float()
    if mode == "grad_l2" and x0_ref is not None:
        return _grad_weighted_norm(diff, x0_ref.float(), strength=strength)
    if mode == "perceptual":
        return _perceptual_error_norm(diff, b)
    diff_flat = diff.flatten()
    if mode == "abs_l2":
        return float(diff_flat.norm() / math.sqrt(max(1, diff_flat.numel())))
    return float(diff_flat.norm() / b.flatten().float().norm().clamp_min(1e-8))

def _eta_schedule(eta, mode, frac, s, s_max, falloff, eta_min):
    if eta <= 0.0:
        return 0.0
    if mode == "linear":
        e = eta * (1.0 - frac)
    elif mode == "exponential":
        e = eta * math.exp(-max(float(falloff), 0.0) * frac)
    elif mode == "cosine":
        e = eta * 0.5 * (1.0 + math.cos(math.pi * frac))
    elif mode == "sigma_power":
        base = min(s / max(s_max, 1e-8), 1.0)
        e = eta * (base ** max(float(falloff), 1e-4))
    else:
        e = eta
    return max(e, min(float(eta_min), eta))

_ETA_ESCAPE_JUMP = 0.005    # fallback: median jump / median |x0| below this => stalled
_ETA_ESCAPE_DETAIL = 0.25   # fallback: ||highpass(x0)|| / ||x0|| below this => flat
_ETA_FADE_FLOOR = 0.25      # fraction of the baseline eta retained after the late fade
_ETA_FADE_SIGMA = 0.35      # fallback convergence sigma (used when the probe gave none)
_ETA_FADE_LO_RATIO = 0.25   # the fade completes at this fraction of the convergence sigma


def _eta_late_fade(s, cfg=None):
    """Decay the baseline ancestral noise through the late stage, anchored at
    the sigma where the probe measured the model's convergence (eta_fade_sigma
    from the profile's conv_step).  The noise is needed early/mid (to stop
    blotchy regions forming), but a constant level keeps oscillating the x0
    trajectory through the endgame; fading it lets structure lock in.  The
    escape (local-minimum) nudge is NOT faded -- a stuck trajectory still gets
    its noise."""
    c = cfg or {}
    conv_sigma = c.get("eta_fade_sigma")
    if conv_sigma is None:
        conv_sigma = _ETA_FADE_SIGMA
    conv_sigma = max(float(conv_sigma), 1e-6)
    floor = float(c.get("eta_fade_floor", _ETA_FADE_FLOOR) or _ETA_FADE_FLOOR)
    lo = conv_sigma * _ETA_FADE_LO_RATIO
    t = (conv_sigma - float(s)) / max(conv_sigma - lo, 1e-9)
    t = min(max(t, 0.0), 1.0)
    ss = t * t * (3.0 - 2.0 * t)  # smoothstep
    return 1.0 - (1.0 - floor) * ss


def _stuck_components(x0, jmap):
    """Return (jump_rel, detail_abs) for the global stuck detector, or
    (None, None) when unavailable.  Exposed separately so the engine can log
    the raw values per step (needed to calibrate the thresholds)."""
    if jmap is None or x0 is None:
        return None, None
    try:
        xm = x0.float().pow(2).mean(dim=1).sqrt()
        med_xm = float(xm.median().clamp_min(1e-8))
        med_j = float(jmap.float().median().clamp_min(1e-8))
        jump_rel = med_j / med_xm
        x4 = x0.float().reshape(-1, x0.shape[1], *x0.shape[-2:])
        hp = x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)
        detail_abs = float(hp.norm() / x4.norm().clamp_min(1e-8))
        return jump_rel, detail_abs
    except Exception:
        return None, None


def _eta_from_probe(profs, budget):
    """DERIVE eta(sigma) from the probe's own noise-absorption measurement.

    THE MEASUREMENT.  `_probe_noise_survival` injects a perturbation of amplitude
    `sqrt(f) * sigma` one step ahead and records `div`, the fraction of it that
    still shows in the model's next prediction relative to an unperturbed
    reference.  Measured across five checkpoints on the ETA-options builds, the
    response is LINEAR in the kick -- `div / sqrt(f)` is constant to within 0.2-4%
    across the amplitude ladder on four of the five (kroma at sigma 0.290: 0.3736
    vs 0.3704; anteros at 0.286: 0.2832 vs 0.2825; velvetmuse: 0.2667 vs 0.2713) --
    so a single per-sigma gain

        k(sigma) = div / sqrt(f)

    describes the model's local response, and it VARIES STRONGLY with sigma in a
    direction that matters: k is ~1.35 at sigma 0.66 and ~0.31 at sigma 0.28 for
    the same checkpoint.  The model tolerates far LESS noise early than late.

    THE DERIVATION.  Asking the measurement for the level that holds the surviving
    perturbation under a residual budget D gives

        k * sqrt(eta) <= D   =>   eta(sigma) = min(1, (D / k(sigma))**2)

    which is a schedule measured on the model in front of the engine, not a
    policy and not a table.

    WHY THIS IS NOT "ANOTHER GATE".  It does not decide WHEN to add noise.  It
    answers HOW MUCH the model can absorb, at every sigma, from the same
    perturbation experiment -- so `f_max(sigma)` on kroma at a 0.10 budget runs
    0.0055, 0.107, 0.072 across the measured range, against a shipped fixed
    `eta = 1.0`.  A schedule that is 180x too high at the start of a run and 14x
    too high in the middle is not a tuning problem.

    WHAT IT DOES NOT DO.  Measured sigma coverage is roughly 0.17..0.66, so
    outside that band the nearest measured k is held (clamped), and the schedule
    is only as good as that extrapolation.  Above the top sample k is large, so
    the clamp holds eta LOW, which is the safe direction.

    Returns a callable `sigma -> eta in [0, 1]`, or None when the probe supplied
    no curve (then the caller keeps the shipped behaviour).
    """
    pts = []
    # THE FIT USES ONLY THE LOW-KICK PART OF THE LADDER, because the response
    # SATURATES above it.  Measured on the widened ladder: `k = div/sqrt(f)` is
    # flat to 4-15% across low amplitudes but drifts +30% to +100% once f reaches
    # 0.5-0.7, so averaging the whole ladder biases k HIGH and therefore biases
    # the derived eta LOW.  The cutoff is the ladder's SECOND rung rather than a
    # literal, so it tracks the ladder if it changes.
    try:
        _ladder = sorted(_NOISE_SURVIVAL_LADDER)
        _f_cut = _ladder[1] if len(_ladder) >= 2 else _ladder[0]
    except Exception:
        _f_cut = 0.35
    for p in (profs or ()):
        if not isinstance(p, dict):
            continue
        cur = p.get("noise_survival_curve")
        if not isinstance(cur, list) or not cur:
            continue
        for e in cur:
            try:
                sig = float(e["sigma"])
                f = float(e["f"])
                div = float(e["div"])
            except (KeyError, TypeError, ValueError):
                continue
            if sig <= 1e-9 or f <= 0.0 or f > _f_cut * (1.0 + 1e-9):
                continue
            k = div / math.sqrt(f)
            if k > 0.0:
                pts.append((sig, k))
    if not pts:
        # NO CURVE AT ALL -> None, so the caller keeps the shipped behaviour.  This
        # is the "not measured" state and it must stay distinguishable from the one
        # below.
        if not any(isinstance(p, dict) and p.get("noise_survival_curve")
                   for p in (profs or ())):
            return None
        # A CURVE THAT MEASURED ZERO RESPONSE is a MEASUREMENT, not an absence:
        # every sampled amplitude was fully corrected.  The gain is 0, so
        # `(D/k)**2` is unbounded and the honest reading is "this model tolerates
        # whatever you ask" -- saturate at the ceiling.  Returning None here would
        # silently fall back to the shipped policy on the best-behaved models,
        # which is the opposite of what the measurement says.
        def _eta_ceiling(sigma):
            return 1.0
        _eta_ceiling.measured_sigmas = ()
        _eta_ceiling.gains = ()
        _eta_ceiling.budget = max(1e-4, float(budget))
        _eta_ceiling.zero_response = True
        return _eta_ceiling
    # one k per sigma (the ladder gives several amplitudes; they agree when the
    # response is linear, and averaging keeps a saturating point from dominating)
    per = {}
    for sig, k in pts:
        per.setdefault(round(sig, 6), []).append(k)
    xs = sorted(per)
    ks = [sum(per[s]) / len(per[s]) for s in xs]
    D = max(1e-4, float(budget))

    def _eta(sigma):
        try:
            s = float(sigma)
        except (TypeError, ValueError):
            return None
        if s <= 0.0:
            return 1.0
        if len(xs) == 1:
            k = ks[0]
        elif s <= xs[0]:
            k = ks[0]
        elif s >= xs[-1]:
            k = ks[-1]
        else:
            j = 0
            while j < len(xs) - 1 and xs[j + 1] < s:
                j += 1
            # interpolate k in log-sigma: sigma is multiplicative along a run
            try:
                t = ((math.log(s) - math.log(xs[j]))
                     / max(math.log(xs[j + 1]) - math.log(xs[j]), 1e-12))
            except ValueError:
                t = 0.0
            k = ks[j] + t * (ks[j + 1] - ks[j])
        if k <= 1e-9:
            return 1.0
        return max(0.0, min(1.0, (D / k) ** 2))

    _eta.measured_sigmas = tuple(xs)
    _eta.gains = tuple(ks)
    _eta.budget = D
    return _eta


def _probe_budget_from_profile(profs):
    """The residual budget the probe ALREADY MEASURED, or None.

    `safe_fresh_frac` is the profile's own answer to "how big a kick can this model
    still correct at a given sigma" -- the divergence at a fixed 0.35 threshold,
    which is the SAME quantity and the same units as the `eta_probe_budget`
    widget.  Its consumer is `calib["safe_eta"]`, a scalar cap on global eta.

    WHY THIS EXISTS RATHER THAN A SECOND MEASUREMENT.  The derived schedule needs a
    residual budget; the `eta_probe_budget` widget is an invented default for it.
    The engine already measures the thing itself and always has, so a user who
    wants the number measured rather than chosen should be able to ask for that.

    Returns a float in (0, 1], or None when no profile carries a usable value --
    in which case the caller keeps the widget's value, so the toggle can never
    leave the sampler without a budget.
    """
    for p in (profs or ()):
        if not isinstance(p, dict):
            continue
        v = p.get("safe_fresh_frac")
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v > 0.0:
            return max(1e-4, min(1.0, v))
    return None


def _probe_commitment(profs, cfg):
    """How much this model COMMITS to a direction, as the probe measured it.

    Returns (committed, raw_mean, n_points) or (None, None, 0).

    `dir_cons` is the per-pixel mean cosine between successive x0 updates, so a
    value below zero means the trajectory REVERSED at that point.  A model whose
    trajectory reverses throughout is exactly the one `_global_stuck_signal`
    describes: "stuck in a local minimum while its median jump is still
    non-trivial -- every pixel moves, just back and forth".

    The mean is taken over the profile's OSCILLATION PHASE when the profile says
    where that starts (`osc_sigma`), because that is the phase the escape exists
    for; without the anchor it is taken over the whole measured curve.  Reported
    clamped to [0, 1] because the caller wants a budget, not a signed reading.
    """
    for p in (profs or ()):
        if not isinstance(p, dict):
            continue
        dc = p.get("dir_cons")
        sg = p.get("sigma_curve")
        if not dc:
            continue
        anchor = p.get("osc_sigma")
        vals = list(dc)
        if sg:
            if anchor is not None:
                sel = [dc[i] for i in range(min(len(dc), len(sg)))
                       if float(sg[i]) <= float(anchor)]
                if sel:
                    vals = sel
        if not vals:
            continue
        m = float(sum(vals) / len(vals))
        return max(0.0, min(1.0, m)), m, len(vals)
    return None, None, 0


def _momentum_signal(jump_rel, detail_abs, jhist, dhist, cfg, jmap=None):
    """Movement-collapse detector: the SECOND way to arm the stall half.

    Returns (score, collapse_ratio, growth_rate), or (None, None, None) when
    there is not enough history yet.  Two conditions, BOTH relative to this run:

      * the per-step movement has fallen to `escape_momentum_collapse` or less of
        its own median over the last `escape_momentum_window` steps -- "not
        moving much, FOR THIS RUN", rather than against an absolute level derived
        from a probe measurement taken on a different grid; and
      * the high-frequency content has not grown by more than
        `escape_momentum_growth` since that same window -- i.e. the picture is
        not still being built while the movement collapses.

    WHY THIS EXISTS: measured on the first build that shipped the Noise Options
    node, the flatness term and the movement term move in OPPOSITE directions as
    a run converges (Spearman -1.00 and +0.99 against step position), so their
    product -- the original `stall` -- read exactly 0.0000 on every run, and no
    threshold placement reaches it.  This branch is armed by `escape_mode` and is
    OFF by default, because whether firing there improves an image is a question
    about appearance and this engine does not decide those.

    THE FLATNESS GATE, and the measured failure it fixes.  The operator generated
    16-bit pixel art at `eta=1.0` with this branch live, and reported it against
    the same prompt and seed at `eta=0.0`: the deterministic run was crisper, with
    a cleaner and more uniform pixel grid, while the noisy run produced
    "half/half pixels" -- cells the sampler could not resolve, mushed together.
    The pair is matched (identical prompt, seed, scheduler, 16 steps) and both
    latent and image are saved, so the cause is measurable rather than inferred:

        step  sigma  ratio  growth  momentum | detail_abs  (d_thr = 0.1449)
          6  0.886  0.429  -0.015     0.387 |     0.2028   <-- fired
         15  0.341  0.406   0.074     0.350 |     0.4732   <-- fired
         16  0.192  0.361   0.016     0.467 |     0.4773   <-- fired

    Every step this branch fired had `detail_abs` at 1.4x to 3.3x the derived
    flatness threshold -- the image was NOT flat, it was FULL of fine structure
    the model was deliberately placing on a pixel grid.  The growth term read
    "flat" because the grid had stopped getting busier, which on pixel art means
    the grid LOCKED, not that the trajectory gave up.  The original flatness
    branch cannot make this mistake: it requires `detail_abs < d_thr` as well.

    So the fix is not new instrumentation, it is applying the test this branch
    was missing.  `escape_momentum_flatness` (default ON for this branch only)
    requires the image to be genuinely flat before the movement collapse may count
    as a stall.  The measured effect on the pixel-art pair: all three firings
    above are suppressed, `eta_eff` stays 0 for the whole run, and the result
    matches the deterministic arm the operator preferred.  On the runs this
    branch was derived for, where the image IS flat at the point of collapse (a
    stall on a still-flat picture), the gate is inert.
    """
    try:
        win = max(2, int(cfg.get("escape_momentum_window", 4)))
        collapse = float(cfg.get("escape_momentum_collapse", 0.7))
        growth = float(cfg.get("escape_momentum_growth", 0.15))
    except (TypeError, ValueError):
        return None, None, None
    if jhist is None or dhist is None or len(jhist) < win or len(dhist) < win:
        return None, None, None
    jw = [v for v in list(jhist)[-win:] if v is not None]
    dw = [v for v in list(dhist)[-win:] if v is not None]
    if len(jw) < win or len(dw) < win:
        return None, None, None
    ref_j = float(sorted(jw)[len(jw) // 2])
    ref_d = float(sorted(dw)[len(dw) // 2])
    if ref_j <= 1e-9 or ref_d <= 1e-9:
        return None, None, None
    collapse_ratio = float(jump_rel) / ref_j
    growth_rate = (float(detail_abs) - ref_d) / ref_d
    if collapse_ratio > collapse or growth_rate > growth:
        return 0.0, collapse_ratio, growth_rate
    # THE FLATNESS GATE.  `_thr` mirrors `_global_stuck_signal`'s reader: a
    # negative value is the "auto" sentinel and means the probe supplied nothing,
    # in which case the constant fallback is used rather than a zero floor that
    # would make the gate unsatisfiable.
    if bool(cfg.get("escape_momentum_flatness", True)):
        try:
            d_thr = float(cfg.get("eta_escape_detail", _ETA_ESCAPE_DETAIL))
        except (TypeError, ValueError):
            d_thr = _ETA_ESCAPE_DETAIL
        if d_thr is None or d_thr < 0:
            d_thr = _ETA_ESCAPE_DETAIL
        if float(detail_abs) >= max(d_thr, 1e-9):
            # full of structure: movement collapsing here is the picture being
            # finished, not a stall -- refuse the arm.
            return 0.0, collapse_ratio, growth_rate
    # All the way to 1.0 at zero movement, scaled by how far under the growth
    # budget the picture sits -- so the branch fades in rather than snapping on,
    # and a trajectory that is collapsing while still gaining detail is not armed.
    js = 1.0 - min(max(collapse_ratio / max(collapse, 1e-9), 0.0), 1.0)
    ds = 1.0 - min(max(growth_rate / max(growth, 1e-9), 0.0) if growth > 1e-9 else 0.0,
                   1.0)
    return js * ds, collapse_ratio, growth_rate


def _global_stuck_signal(x0, jmap, cfg=None, dir_cos=None, s=None,
                         jhist=None, dhist=None):
    """Global local-minimum detector, in [0, 1].

    Fires on EITHER of two stuck states:

    1. STALLED: the median per-pixel x0 jump has collapsed (relative to the x0
       magnitude) AND the image is still flat (low absolute high-frequency
       energy).  The flatness condition is meant to keep the nudge from firing at
       normal convergence, where the jump also collapses but the image is already
       detailed.

    2. COLLAPSING (`escape_mode="momentum"` only): the movement has collapsed
       against THIS RUN's own recent level and the image is not still gaining
       detail -- a second, run-relative way to arm the same half of the detector,
       added because the flatness condition above is measurably unsatisfiable on
       real runs.  See `_momentum_signal`, which carries that measurement.

    3. OSCILLATING: the x0 update reverses the previous update (dir_cos < 0),
       and only BELOW the probe-measured oscillation onset (eta_fade_sigma =
       the profile's osc_sigma).  A trajectory can be stuck in a local minimum
       while its median jump is still non-trivial -- every pixel moves, just
       back and forth.  The early "commit" also reverses direction, but that
       is the model committing, not a wobble, so the reversal is only a stuck
       signal after the oscillation onset.  It scales with the reversal
       strength: monotone contributes nothing, a full reversal saturates.

    Thresholds are cfg-tunable (eta_escape_jump / eta_escape_detail) and
    deliberately absolute rather than median-relative so a global stall (every
    pixel stops at once) is still detected.  NOTE ON THAT CHOICE, measured
    (`rig_eta_tune.py`, the first build shipping the Noise Options node): the
    two inputs this term multiplies move in OPPOSITE directions as a run
    converges -- Spearman against step position was -1.00 for the jump and +0.99
    for the detail, because the high-frequency term RISES as the picture forms
    instead of falling flat.  `stall` therefore read exactly 0.0000 on all 7 runs
    of that build and, at the shipped gain of 1.0, on every one of them.  The gain
    does reach it (the audit's ten runs armed at 1.5-4.0), so the shipped
    sensitivity is below the population's floor, not the term zero by construction;
    the default stays 1.0 because arming it also noises steps that did not need it.
    That is what branch 2 exists to address, and it is why it ships OFF.
    """
    jump_rel, detail_abs = _stuck_components(x0, jmap)
    if jump_rel is None or detail_abs is None:
        return 0.0
    c = cfg or {}

    def _thr(key, fallback):
        v = c.get(key, fallback)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = fallback
        # a negative value is the "auto" sentinel; the engine derives the real
        # threshold from the probe before the loop, so here it only means "no
        # probe measurement was available" -> use the constant fallback.
        if v is None or v < 0:
            v = fallback
        return max(v, 1e-9)

    j_thr = _thr("eta_escape_jump", _ETA_ESCAPE_JUMP)
    d_thr = _thr("eta_escape_detail", _ETA_ESCAPE_DETAIL)
    stag_score = 1.0 - min(max(jump_rel / j_thr, 0.0), 1.0)
    flat_score = 1.0 - min(max(detail_abs / d_thr, 0.0), 1.0)
    stall = stag_score * flat_score
    # The momentum branch is a SECOND way to arm the SAME half of the detector,
    # so it takes `max` with the flatness stall rather than replacing it: with
    # `escape_mode` at its shipped "flatness" the term below is exactly the
    # shipped one, and nothing about the reversal (osc) half changes with it.
    mom = 0.0
    if str(c.get("escape_mode", "flatness")) == "momentum":
        mom, _cr, _gr = _momentum_signal(jump_rel, detail_abs, jhist, dhist, c)
        mom = 0.0 if mom is None else mom
        stall = max(stall, mom)
    osc = 0.0
    if dir_cos is not None and dir_cos < 0.0:
        osc = min(1.0, -float(dir_cos))
        _osc_on = c.get("eta_fade_sigma")
        if _osc_on is not None and s is not None and float(s) > float(_osc_on):
            # above the oscillation onset the reversal is the commit, not a
            # wobble; keep the nudge for the late phase only.
            osc = 0.0
    return max(stall, osc)


def _adaptive_eta(err, consistency, s, sn, s_max, s_min, space, sigma_data,
                  s_noise, model_info, eta_ceiling, cfg, few_step=False,
                  stuck=0.0):
    """Adaptive stochasticity (v15): minimal noise + local-minimum escape.

    The base level is eta_min (default 0 = fully deterministic), so the
    trajectory is free to lock structure in.  Noise is only raised -- toward
    the stochasticity ceiling -- while `stuck` (the global local-minimum
    signal) is non-zero.  The escape is self-limiting: the injected noise makes
    the trajectory move again, which lowers `stuck` on the next step.

    The base (floor) noise is faded through the late stage (see
    _eta_late_fade, anchored at the probe-measured convergence sigma) so the
    endgame can lock in; the escape nudge rides on top of the faded floor and
    is NOT faded, so a genuinely stuck trajectory still gets its noise.
    Replaces the old burn-in / error / SNR / stability multipliers, which
    pegged eta at the ceiling for flow models.
    """
    floor = max(0.0, float(cfg.get("eta_min", 0.0) or 0.0))
    # eta_min is a FLOOR: if the user sets it above the stochasticity slider the
    # effective level is eta_min (constant, no escape headroom) rather than
    # silently collapsing to 0.
    ceil = max(float(eta_ceiling), floor)
    floor = floor * _eta_late_fade(float(s), cfg)
    stuck = max(0.0, min(float(stuck), 1.0))
    return floor + (ceil - floor) * stuck

def _jump_signal(x0, prev_x0, x, q=0.99):
    diff = (x0 - prev_x0).pow(2).mean(dim=1)
    resid = (x - x0).pow(2).mean(dim=1)
    ref = x0.detach().pow(2).mean(dim=1)
    r = (diff + 1e-12).sqrt() / ((resid + 1e-12).sqrt() + 0.02 * (ref + 1e-12).sqrt())
    flat = r.flatten().float()
    n = flat.numel()
    if n > (1 << 22):
        flat = flat[:: n // (1 << 22) + 1]
    return float(torch.quantile(flat, q))


def _guard_quantile_value(cfg):
    """`guard_quantile` clamped to the only range `torch.quantile` accepts.

    The widget used to declare 0..4; a value above 1 raised inside the step loop
    with no enclosing handler.  The clamp lives here so an API graph or an
    Options node is covered too, not only the UI.
    """
    try:
        q = float(cfg.get("guard_quantile", 0.99))
    except (TypeError, ValueError):
        q = 0.99
    return min(max(q, 0.0), 1.0)

class _OutlierGuard:

    def __init__(self, window=8, z=3.0, floor=0.05, ratio=3.0, hold=2,
                 warmup=4, mad_floor=0.1):
        self.window = max(4, int(window))
        self.z = float(z)
        self.floor = float(floor)
        self.ratio = float(ratio)
        self.hold = int(hold)
        self.warmup = int(warmup)
        self.mad_floor = float(mad_floor)
        self.hist = []
        self.deltas = []
        self.left = 0
        self.last_sev = 1.0

    def step(self, signal, z=None):
        z = self.z if z is None else float(z)
        trip = False
        self.last_sev = 1.0
        if self.hist:
            delta = signal - self.hist[-1]
            if len(self.deltas) >= self.warmup:
                arr = sorted(self.deltas)
                dmed = arr[len(arr) // 2]
                dmad = sorted(abs(d - dmed) for d in self.deltas)[len(self.deltas) // 2]
                level = sorted(self.hist)[len(self.hist) // 2]
                lvl = max(level, 1e-6)
                dmad_eff = max(dmad, self.mad_floor * lvl)
                dthr = max(dmed + z * 1.4826 * dmad_eff, self.floor * lvl)
                trip = delta > dthr or signal > self.ratio * lvl
                if trip:
                    self.last_sev = max(signal / max(dthr, 1e-8), 1.0)
            self.deltas.append(delta)
            if len(self.deltas) > self.window:
                self.deltas.pop(0)
        if trip:
            self.left = self.hold + 1
        active = self.left > 0
        self.hist.append(signal)
        if len(self.hist) > self.window:
            self.hist.pop(0)
        if self.left > 0:
            self.left -= 1
        return trip, active

def _gaussian_kernel1d(radius, dtype=torch.float32, device=None):
    from math import comb
    n = 2 * int(radius)
    coeffs = [float(comb(n, k)) for k in range(n + 1)]
    s = sum(coeffs)
    t = torch.tensor(coeffs, device=device, dtype=dtype) / s
    return t

def _gaussian_blur_separable(g, win):
    sh = g.shape
    g4 = g.reshape(-1, 1, sh[-2], sh[-1]).float()
    radius = max(1, int(win) // 2)
    k = _gaussian_kernel1d(radius, dtype=g4.dtype, device=g4.device)
    pad = radius
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    g4 = F.pad(g4, (pad, pad, 0, 0), mode="replicate")
    g4 = F.conv2d(g4, kx)
    g4 = F.pad(g4, (0, 0, pad, pad), mode="replicate")
    g4 = F.conv2d(g4, ky)
    return g4.reshape(sh)

def _spatial_stats(t):
    """(rms, std, grad, hf) of a latent -- cheap per-step logging statistics.

    `hf` is the DoG high-pass energy, reusing the engine's own perceptual
    helper.  Every term is a reduction over a tensor that already exists: no
    model call and no extra sampling work, so this is orders of magnitude
    cheaper than one NFE and can stay on permanently.

    Recorded because scalar error metrics (err/tol) demonstrably do NOT
    discriminate between clamp settings that look very different -- detail
    energy is a candidate for a signal that does.
    """
    f = t.float()
    rms = float(f.pow(2).mean().sqrt())
    try:
        std = float(f.reshape(f.shape[0], -1).std(dim=1).mean())
    except Exception:
        std = 0.0
    grad = float((f[..., 1:, :] - f[..., :-1, :]).abs().mean()
                 + (f[..., :, 1:] - f[..., :, :-1]).abs().mean())
    try:
        hf = float((_gaussian_blur_separable(f, 3)
                    - _gaussian_blur_separable(f, 5)).pow(2).mean().sqrt())
    except Exception:
        hf = 0.0
    return rms, std, grad, hf


def _tile_extremes(t, grid=16):
    """LOCALIZATION of a field: (max tile mean, SHARE of the total, y, x) on a `grid` x `grid` map.

    WHY THIS EXISTS (2026-10-02).  NOTES' artifact entry records that the operator reports SMALL
    LOCALIZED artifacting in some Krea2 generations, and that NOTHING in the logs separates the
    labelled offender runs -- because every spatial channel this engine logged was either a
    whole-tensor aggregate (`x0_hf`, `dx0_grad`, `med_detail_abs`, `oversteer`) or an 8x8 map.
    A reduction averages localized damage away, and an 8x8 tile on a 1024px latent is a 128px
    block, so both are structurally blind to a small artifact.  That is an instrument gap, not a
    mystery.

    This reduces a field to `grid x grid` per-tile means by AVERAGE POOLING -- no model call and
    no sampling work, the same cost class as `_spatial_stats` -- and reports the extremes OF THAT
    MAP, so "the damage is concentrated in one small region" becomes a number.

    THE SECOND RETURNED VALUE IS A SHARE, NOT A RATIO, and that is a CORRECTION.  The first
    version returned `max/median`; `scratch_tile_extremes_rig.py` refuted it against a
    constructed field that is zero except ONE tile -- exactly the artifact case -- where the
    median of the tile map is 0, so the ratio collapsed to 0 and the statistic reported "no
    concentration" for the MOST concentrated field possible.  `max / sum` is bounded in
    [0, 1], defined for every field including an all-zero one (0.0 by convention), scale
    invariant, and monotone in concentration: a completely flat field gives 1/grid^2 (0.0039 at
    grid 16, i.e. "this tile holds its fair share"), and a single-tile field gives 1.0.

    Returns None when the tensor has no 2-D spatial extent to tile.
    """
    try:
        f = t.float().abs()
        if f.dim() < 2 or f.shape[-1] < 2 or f.shape[-2] < 2:
            return None
        hw = f.reshape(-1, f.shape[-2], f.shape[-1])
        pooled = F.adaptive_avg_pool2d(hw.unsqueeze(0), (int(grid), int(grid)))[0]
        per_tile = pooled.reshape(-1, int(grid) * int(grid)).mean(dim=0)
        mx = float(per_tile.max())
        total = float(per_tile.sum())
        gy, gx = divmod(int(per_tile.argmax()), int(grid))
        return mx, (mx / total if total > 1e-12 else 0.0), gy, gx
    except Exception:
        return None


def _hf_field(t):
    """The DoG high-pass FIELD that `_spatial_stats` reduces to a scalar (`hf`).

    Exposed separately so its LOCALIZATION can be measured with `_tile_extremes`: a scalar `hf`
    cannot tell a uniformly detailed image from one with a single damaged patch, which is the
    distinction the artifact investigation needs.
    """
    f = t.float()
    return _gaussian_blur_separable(f, 3) - _gaussian_blur_separable(f, 5)


def _local_mean(g, win):
    return _gaussian_blur_separable(g, win)

def _clean_mask(m, erode=1, dilate=2):
    sh = m.shape

    def pool(x, k):
        return F.max_pool2d(x.reshape(-1, 1, sh[-2], sh[-1]).float(), k,
                            stride=1, padding=k // 2)

    if erode > 0:
        m = ~(pool(~m, 2 * erode + 1) > 0)
    if dilate > 0:
        m = pool(m, 2 * dilate + 1) > 0
    return m.reshape(sh)

def _neighbor_count(m):
    sh = m.shape
    m4 = m.reshape(-1, 1, sh[-2], sh[-1]).float()
    cnt = F.avg_pool2d(m4, 3, stride=1, padding=1) * 9.0
    return cnt.reshape(sh)

def _anomaly_mask(x0, prev_x0, j_prev_max, win, s_thr, n_thr, v_thr,
                  jmap=None, region_z=12.0):
    if jmap is None:
        jmap = (x0 - prev_x0).pow(2).mean(dim=1).sqrt()
    scale = float(jmap.mean()) + 1e-8
    loc = _local_mean(jmap, win)
    spatial = (jmap / (loc + 0.1 * scale)) > float(s_thr)
    v = x0.pow(2).mean(dim=1).sqrt()
    vloc = _local_mean(v, win)
    value = (v / (vloc + 1e-6)) > float(v_thr)
    if j_prev_max is not None:
        nov = (jmap / (j_prev_max + 0.1 * scale)) > float(n_thr)
        m_small = spatial & nov & value
        flat = jmap.reshape(jmap.shape[0], -1).float()
        med = flat.median(dim=1).values.unsqueeze(1)
        mad = (flat - med).abs().median(dim=1).values.unsqueeze(1)
        zr = ((flat - med) / (1.4826 * mad + 0.05 * med + 1e-6)) > float(region_z)
        zr2d = zr.reshape(jmap.shape) & nov
        cnt = _neighbor_count(zr2d)
        m = m_small | (zr2d & (cnt >= 3.0))
    else:
        m = spatial & value
    return _clean_mask(m, erode=1, dilate=2)

def _elevation_mask(jmap, win, v_thr, x0, mult=3.0):
    flat = jmap.reshape(jmap.shape[0], -1).float()
    med = flat.median(dim=1).values.view(-1, *([1] * (jmap.dim() - 1)))
    scale = float(jmap.mean()) + 1e-8
    m = jmap > (float(mult) * med + 0.1 * scale)
    v = x0.pow(2).mean(dim=1).sqrt()
    vloc = _local_mean(v, win)
    return m & ((v / (vloc + 1e-6)) > float(v_thr))
_LF_CENTERING = "weighted_const"
_LF_V_MAX = 1.70
_LF_DRIFT_RELAX = 0.30
# `_LF_BAND_RHO` REMOVED 2026-10-02.  It was the fallback refresh for "no band_curve
# baseline", and BOTH of its sites were unreachable: `_ref <= 1e-9` never occurred in 8444
# corpus step records, and the `_rho_band is None` default could not be reached because the
# only producer of `_rho_band` sets it whenever `_ref > 1e-9`.  A value for a situation that
# does not occur is exactly the "un-explainable constant" this codebase does not keep.  The
# no-baseline case is now handled by declining to quantify the detection at all (see the
# band block: the excess is a RATIO to `_ref`, so with no baseline it is undefined).
_LF_FRAG_GAIN = 0.75    # fallback fragility gain (when no local-error measurement exists)
# GROUNDED 2026-10-02, replacing 0.006 (scratch_batch4_anchors_rig.py, output committed;
# CONSTANTS_AUDIT.md section 11).  The constant's job is stated in this comment: a model at
# this value gets frag_gain ~ 1.0.  The measured median `med_local_err_real` over 401 real
# probe profiles is 0.011194, so at the OLD 0.006 a typical model sat at ratio 1.87 -- above
# `_FRAG_GAIN_MAX` -- and 78.1% of real profiles were pinned at the ceiling, i.e. the constant
# could not deliver the gain ~ 1.0 it promised.  At the measured median the ratio is 1.0 and
# only 4.2% are pinned.  Anchor stability: the per-model median spans 3.5x across 16 models
# (0.006044 .. 0.021395), so one value can serve.
# RE-DERIVE if the probe distribution moves -- this is a corpus quantile, not a natural
# constant, which is the same argument that made "derive at runtime" the answer for `rtol`
# (CONSTANTS_AUDIT.md section 9).
_FRAG_ERR_REF = 0.011194
# GROUNDED 2026-10-02, replacing 0.15.  These clamp the ratio `_me / _FRAG_ERR_REF`; anchored
# to that ratio's own percentiles so they bound OUTLIERS rather than clipping the body.
# With the derived ref above, the ratio's measured range is [0.337, 3.394] with p5 0.411 and
# p95 1.489 over the same 401 profiles.  The old MIN 0.15 sat BELOW the entire measured range
# (0.0% of profiles were ever under it), so it could never bind and had no basis.  `_MAX` is
# unchanged because 1.5 already sits at that p95 -- it was well placed, and measurement keeps
# a constant as readily as it moves one.
_FRAG_GAIN_MIN = 0.411
_FRAG_GAIN_MAX = 1.5
# The band-surgical DETECTION, in one place, because the ramp's zero point must BE the
# detection threshold rather than a second number that approximates it.  Before 2026-10-02
# the detection was `_hi > 1.25*_ref + 0.03` and the ramp started at a separate
# `_BAND_RHO_REF = 0.25`; those two only agree as `_ref` grows, and the comment claimed
# "0 at the detection threshold" while the arithmetic delivered `0.04/_ref` there.  Deriving
# one from the other removes the discrepancy and one constant with it.
_BAND_DET_A = 1.25      # detected when the hi-band exceeds this multiple of its baseline...
_BAND_DET_B = 0.03      # ...by at least this absolute margin
# Ceiling on the injected refresh.  GROUNDED, not chosen: measured 2026-10-02 as the p99 of
# the PRIMARY channel's own refresh (`lf.rho_max`) over the 3366 corpus step records where
# the refresh was applied -- p50 0.0115, p90 0.0299, p99 0.0507, max 0.0869.  The rule: a
# FALLBACK that stands in for the primary channel when it is silent must not be more
# aggressive than the primary channel is in its own strongest 1% of steps.  The previous
# value (0.10) sat above the observed maximum (0.0869), i.e. it never bound and had no
# measurement behind it.  Re-derive with `scratch_band_rho_rig.py` if that distribution moves.
_BAND_RHO_MAX = 0.0507
# The ramp's SPAN is expressed as a multiple of the detection excess, not as an absolute
# excess, and that is a correction to the first version of this fix.  An absolute top
# (`_BAND_EXCESS_MAX = 1.0`, "2x the baseline") looks natural but is not always above the
# zero point: the detection sits at excess `0.25 + 0.03/_ref`, which EXCEEDS 1.0 once
# `_ref < 0.04`.  For those baselines the two anchors cross and the ramp becomes ill-defined
# -- caught by `scratch_band_rho_rig.py` check 1f.  A relative span cannot cross, because
# `_e_det > 0.25` always, so it needs no second constant and no degenerate-case rule:
# 0 at the detection threshold, full at twice the detection excess.  Both ends are then
# derived from the probe-measured baseline alone.
_BAND_RAMP_SPAN = 2.0   # full strength at this multiple of the detection excess

def _lf_v_step(v, vmul):
    # Reachability for the Batch-2 sweep (scratch_constants_batch2_rig.py).  A call
    # count is NOT enough here: this site only MATTERS when the product exceeds the
    # ceiling, so `lf.v_step_bind` records the pre-clamp value.  Without that, an
    # "insensitive" verdict could not be told apart from "the branch ran but never
    # reached the ceiling" -- the Batch-0 failure mode.
    _r = float(v) * float(vmul)
    _ab_hit("lf.v_step_calls")
    if _r > _LF_V_MAX:
        _ab_hit("lf.v_step_bind", round(_r, 6))
    return min(_r, _LF_V_MAX)

def _lf_dir_weight(cos, db=None):
    cos = cos if torch.is_tensor(cos) else torch.as_tensor(cos)
    if db is None:
        db = 0.25
    db = min(max(float(db), 0.0), 0.9)
    return (1.0 + 0.6 * ((-cos - db) / max(1.0 - db, 1e-6)).clamp(0.0, 1.0)
            - 0.2 * ((cos - 0.7) / 0.3).clamp(0.0, 1.0)).clamp(0.8, 1.6)

def _lf_map_median(m):
    flat = m.reshape(m.shape[0], -1).float()
    med = flat.median(dim=1).values.view(-1, *([1] * (m.dim() - 1)))
    return med.clamp_min(1e-6)

def _lf_detail_map(x0):
    sh = x0.shape
    x4 = x0.reshape(-1, sh[1], *sh[-2:]).float()
    # REPLICATE-PAD, not zero-pad.  `avg_pool2d(padding=1)` divides by 9 even at
    # the border, so a constant field reads as detail there (and the outer latent
    # ring is manufactured detail on every image).  `_gaussian_blur_separable`
    # already replicates; this matches it.
    x4p = F.pad(x4, (1, 1, 1, 1), mode="replicate")
    hp = x4 - F.avg_pool2d(x4p, 3, stride=1)
    d = hp.pow(2).mean(dim=1).sqrt()
    med = d.reshape(d.shape[0], -1).median(dim=1).values.view(-1, 1, 1).clamp_min(1e-6)
    d = (d / med).clamp(1e-4, 1e4).log().clamp(-4.0, 4.0)
    return d.reshape(sh[0:1] + (1,) + sh[2:])

def _sstep(t):
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

def _lf_env_fade(kn, eg_n=0.0):
    _fade_floor = 0.15 + 0.85 * float(eg_n)  # 0.15 at eg_n=0, 1.0 at eg_n=1
    return _fade_floor + (1.0 - _fade_floor) * (1.0 - _sstep((float(kn) - 0.6) / 0.3))

def _lf_block_scale(jmap, block=4):
    sh = jmap.shape
    j4 = (jmap.reshape(-1, 1, *sh[-2:]).float() + 1e-12).log()
    blk = F.avg_pool2d(j4, int(block))
    full = F.interpolate(blk, size=sh[-2:], mode="bilinear", align_corners=False)
    return blk, full.reshape(sh)

def _lf_block_delta_bands(jmap, block=4):
    sh = jmap.shape
    j4 = jmap.reshape(-1, 1, *sh[-2:]).float()
    j4p = F.pad(j4, (1, 1, 1, 1), mode="replicate")
    b1 = F.avg_pool2d(j4p, 3, stride=1)
    b1p = F.pad(b1, (1, 1, 1, 1), mode="replicate")
    b2 = F.avg_pool2d(b1p, 3, stride=1)
    p = lambda t: F.avg_pool2d(t.pow(2), int(block))
    e_hi, e_mid, e_lo = p(j4 - b1), p(b1 - b2), p(b2)
    tot = (e_hi + e_mid + e_lo).clamp_min(1e-20)
    return e_lo / tot, tot

def _endgame_eg_err(endgame):
    """The endgame error scalar, from the endgame dict directly.

    `_prof_eg` reads exactly this out of a finished profile; it is hoisted here
    (and `_prof_eg` now delegates) so the PROBE can compute the same number while
    the profile is still being assembled -- the coverage curve needs `eg_n`, and
    `eg_n` needs this.  One implementation, two callers, for the same reason the
    rescue measurement was unified.
    """
    if not isinstance(endgame, dict):
        return None
    vals = []
    for lv in (endgame.get("levels") or []):
        for key in ("rec_q", "rec2_q", "rec_p50"):
            try:
                v = lv.get(key)
                if isinstance(v, (list, tuple)):
                    vals.append(float(v[1]))
                else:
                    vals.append(float(v))
            except (TypeError, ValueError, IndexError, KeyError):
                pass
    return max(vals) if vals else None


def _eg_n_from_err(eg_err):
    """`eg_n`, the same map the gains use (aflops.py:2361), so a probe-side
    quantity and its run-side consumer cannot disagree."""
    if eg_err is None:
        return 0.0
    try:
        return 1.0 - math.exp(-float(eg_err) / 0.06)
    except Exception:
        return 0.0


def _lf_rescue_cov_meta(x0s_ref, eg_err, floor_frac=None, gate_start=None):
    """Self-describing provenance for `cov_curve`, so a later reader knows which
    block size, latent size and `eg_n` produced it -- without which the curve is
    another number of unknown provenance."""
    try:
        lat = [int(x0s_ref.shape[-2]), int(x0s_ref.shape[-1])]
        blk = _lf_block_for(x0s_ref)
    except Exception:
        lat, blk = None, None
    out = {"blk": blk, "latent": lat,
           "eg_n": round(_eg_n_from_err(eg_err), 4),
           "src": "probe; aflops._lf_rescue_measure (shared with the run)"}
    # THE FLOOR IS PER STEP on BOTH SIDES.  The run computes
    # `floor_mag = rescue.floor * s` every step (`_lf_step_field`), so the probe must
    # too; this record used to hardwire 0.0 and a comment claiming the run's was 0,
    # which made the probe's `t_floor` conjunct vacuous and the claim false (D3).
    if floor_frac is not None:
        out["floor_frac"] = round(float(floor_frac), 6)
        out["floor_per_step"] = "floor_frac * sigma, as the run's rescue.floor * rescue.s"
    # WHICH STEP THE PROBE'S REFERENCE STARTS AT.  The run's gate is fed only from
    # `kn >= rescue_start`, so the probe's running reference starts there too (U08).
    if gate_start is not None:
        out["gate_start"] = round(float(gate_start), 6)
    return out


def _lf_rescue_window(kn, start=0.55, full=0.72):
    t_in = _sstep((float(kn) - float(start)) / max(float(full) - float(start), 1e-6))
    t_out = 1.0 - 0.5 * _sstep((float(kn) - 0.92) / 0.08)
    return t_in * t_out


def _lf_block_for(jmap, ref=32):
    """The rescue block size as a FRACTION of the latent, not a pixel count.

    WHY THIS IS DERIVED RATHER THAN CONSTANT.  `rescue.block` was a hardcoded 4
    PIXELS, so under one name it measured three different things depending on the
    latent: 4/48 of the image at the model probe, 4/16 at the cond probe, 4/128 in
    a 1024px run.  A block statistic computed at one of those cannot predict the
    same statistic at another, which is what blocked using the probe to see this
    pitfall coming.  Measured consequence elsewhere: the rescue's coverage gate
    sits at a CLIFF (`g_cov`, aflops.py:1724) and a checkpoint swap moved `frac`
    across it, turning a whole mechanism on and off (see NOTES.md).

    THE ANCHOR IS NOT ARBITRARY.  Keeping the number of blocks per side constant
    at 32 reproduces block = 4 EXACTLY at 128x128, the resolution the original
    constant was tuned at, so the operator's existing runs are bit-identical.  The
    rule only changes what happens at OTHER resolutions, which is the bug.

    Clamped to [2, 8]: below 2 a block stops averaging anything, above 8 the
    statistic goes blind to the fine structure the rescue is meant to catch.
    """
    try:
        m = min(int(jmap.shape[-2]), int(jmap.shape[-1]))
    except Exception:
        return 4
    return max(2, min(8, int(round(m / float(max(int(ref), 1))))))


def _lf_rescue_measure(jmap, d, blk, eg_n, floor_mag, ref_prev=None, diag=None,
                       ref_drop=0.2231):
    """The rescue gate's block statistics -- ONE implementation, two callers.

    Hoisted verbatim out of `_lf_step_field` so the RUN-side field and the
    PROBE-side prediction cannot drift apart.  This project already paid for that
    lesson once: `oversmooth` had a stored-vs-recomputed path split whose two
    consumers disagreed for months (DEAD_FEATURES.md section 3).  There is exactly
    one copy here, and `scratch_rescue_covcurve_rig.py` asserts it reproduces the
    original inline formula BIT-EXACTLY at three resolutions and two parameter
    settings -- a near-miss is a failure, not a rounding note.

    `ref_prev` carries the running reference across steps; pass None on the first.
    `ref_drop` is the reference's decay allowance for THIS step.  The default is
    the shipped per-step constant, so the run path is bit-identical; the probe
    additionally computes a variant with k*du (see `_lf_rescue_cov_curve`) by
    passing that value here.  One implementation, one parameter -- a second copy
    of the formula would recreate the stored-vs-recomputed split.
    `diag`, when a dict is passed, is filled with the pass fraction of EACH
    conjunct plus the median block log-jump.  The gate is a 4-way AND; on the
    HEALTHY probes the sole binding conjunct is t_ref at 0.0017-0.0035 pass, so
    the per-term split is what makes that locatable.

    Returns (hit, frac, ref_new, lf_share).
    """
    lf_share, tot_blk = _lf_block_delta_bands(jmap, blk)
    blk_log, _ = _lf_block_scale(jmap, blk)
    d_blk = F.avg_pool2d(d.reshape(-1, 1, *d.shape[-2:]), blk)
    n_bt = blk_log.shape[0]
    ref_tot = torch.quantile(tot_blk.reshape(n_bt, -1).float(), 0.10,
                             dim=1).view(-1, 1, 1, 1)
    ref = ref_prev
    if ref is None or ref.shape != blk_log.shape:
        ref = blk_log.clone()
    else:
        ref = torch.maximum(blk_log, ref - float(ref_drop))
    t_mag = tot_blk > 1.5 * ref_tot
    t_flr = tot_blk.sqrt() > floor_mag
    t_ref = (blk_log - ref) > -0.6931   # >= half the late peak
    t_dtl = d_blk > (0.1 * (1.0 + 0.7 * eg_n))
    hit = t_mag & t_flr & t_ref & t_dtl
    frac = hit.float().mean(dim=(2, 3), keepdim=True)
    if isinstance(diag, dict):
        try:
            # TENSORS, never floats, and cheap means/medians only.  `float(tensor)`
            # forces a GPU->CPU sync and `.quantile()` SORTs the whole block map;
            # measured, that made the probe curve 4x more expensive
            # (scratch_probe_cost_rig.py).  The caller converts once.
            diag["t_mag"] = t_mag.float().mean()
            diag["t_floor"] = t_flr.float().mean()
            diag["t_ref"] = t_ref.float().mean()
            diag["t_detail"] = t_dtl.float().mean()
            diag["d_blk_mean"] = d_blk.float().mean()
            diag["blk_log_med"] = blk_log.median()
        except Exception:
            pass
    return hit, frac, ref, lf_share


def _lf_rescue_cov_curve(xs, x0s, grid, eg_n, floor_frac=0.02, start=0.55,
                         ref=32):
    """Per-step rescue coverage along a PROBE trajectory.

    The probe is the only instrument that measures the model AND the prompt
    BEFORE the run, so it is where a pitfall like the coverage cliff has to be
    seen coming.  The inputs are already there: `jpx` in `_run_probe`
    (aflops.py:2972) is the SAME formula as the run's `jmap`, and `d` is
    `(x - x0)/sigma`, which the probe also holds.

    Indexing follows the run: step i pairs the jump INTO state i with that state's
    own `d`, i.e. jmap_i = |x0[i+1] - x0[i]| and d_i = (x[i+1] - x0[i+1])/s[i+1].

    BOTH OF THE RUN'S INPUTS ARE MIRRORED, because the curve is a PREDICTION of the
    run's gate and may only predict what the run evaluates:

      * `floor_frac` enters as `floor_frac * sigma` PER STEP, which is exactly the
        run's `floor_mag = rescue.floor * rescue.s` (`_lf_step_field`).  It used to be
        a single 0.0, so the probe's `t_floor` conjunct was always true (D3).
      * the running reference starts only where the run starts FEEDING the gate,
        `kn = (i+1)/n_steps >= start` (the run's `rescue_start`, 0.55).  Before that
        the run's `_st` is empty; advancing the probe's reference through those steps
        gave the first in-window step a reference the run never had (U08).

    Returns a dict:
      "curve"      per-step `frac` under the SHIPPED per-step decay (0.2231/step)
      "curve_kdu"  the same under `ref_drop = k*du`, the DERIVED form, or None
      "terms"      per-conjunct medians for "curve"
      "k"          the trajectory's own measured decay rate, nats per unit
                   log-sigma, or None if the trajectory does not decay
      "n_steps"    number of steps the curve covers

    WHY TWO CURVES.  Measured on the real corpus, the healthy probe's sole binding
    conjunct is `t_ref` (0.0017-0.0035 pass) while `t_mag` passes 0.71-0.78 and
    `t_detail` 0.44 -- and t_ref's allowance is a PER-STEP constant of 0.2231 while
    the probe's own median block-jump decays ~0.5 nats per step.  The jump map
    falls like sigma^p, so the allowance should be a RATE per unit log-sigma:
    k*du.  Two independent estimates of k agree at ~2 (the run's mean du is 0.117,
    and 0.2231/0.117 = 1.9; the probe's measured total decay is ~4 nats over
    sum(du) = 1.75, giving 2.3) -- i.e. the shipped constant was k*du_run all
    along, correct only at that one step count.

    That derivation could NOT be validated on synthetic fields:
    scratch_ref_decay_rig.py tried two and refuted both -- NonlinearFlow passes
    t_ref 100% at every spacing with a NEGATIVE k (its jump map grows), and
    PowerFlow(p=2) gives k ~ 0 with hit ~ 0.03.  Neither reproduces the real
    models' combination of hit 0.27-0.43 AND a decaying median block-jump, because
    the phenomenon depends on the model's actual spatial prediction structure.
    So the engine now emits BOTH curves and the next real runs decide.
    """
    try:
        n = min(len(xs), len(x0s), len(grid))
        if n < 3:
            return None
        # The run's step count, for the same `kn` the gate windows on.
        n_steps = max(1, len(grid) - 1)
        # Accumulate the per-step fractions ON DEVICE and convert once at the
        # end.  `float(tensor)` forces a GPU->CPU sync and the timings in
        # scratch_probe_cost_rig.py are flat across resolutions, i.e. dominated by
        # per-step sync/launch overhead rather than by tensor size.
        steps = []
        fr = []
        term_rows = []
        _TKEYS = ("t_mag", "t_floor", "t_ref", "t_detail", "d_blk_mean")
        med_seq = []
        du_seq = []
        u_seq = []          # the SAMPLE positions u = log(sigma), for the k span
        ref_prev = None
        for i in range(n - 1):
            x0a, x0b = x0s[i].float(), x0s[i + 1].float()
            jmap = (x0b - x0a).pow(2).mean(dim=1, keepdim=True).sqrt()
            s = float(grid[i + 1])
            if s <= 1e-8:
                continue
            # THE RUN'S GATE IS NOT FED BEFORE ITS WINDOW OPENS.  `_lf_step_field`
            # measures the rescue gate only when `kn >= rescue["start"]`, i.e. from
            # step k = i+1 with kn = k / n_steps; its reference is fresh at that
            # first step.  Advancing the probe's reference through the earlier steps
            # tested `t_ref` against a baseline the run never holds.
            kn = (i + 1) / float(n_steps)
            if kn < float(start):
                continue
            # The RUN's rescue gate passes `_lf_detail_map(x0)`; the probe used
            # to pass this sigma-normalised residual, a different quantity with a
            # different sign/shape/units, so its t_detail term could not predict
            # the gate.
            dd = _lf_detail_map(x0b)
            blk = _lf_block_for(jmap, ref)
            if min(int(jmap.shape[-2]), int(jmap.shape[-1])) < blk:
                continue
            du = math.log(max(float(grid[i]), 1e-8)) - math.log(max(s, 1e-8))
            # THE RUN'S FLOOR, PER STEP: `rescue.floor * rescue.s` (D3).
            floor_mag = float(floor_frac or 0.0) * s
            _dg = {}
            _hit, frac, ref_prev, _sh = _lf_rescue_measure(
                jmap, dd, blk, float(eg_n or 0.0), floor_mag,
                ref_prev=ref_prev, diag=_dg)
            if _dg:
                term_rows.append([_dg[k].detach().reshape(()) for k in _TKEYS
                                  if k in _dg])
                if "blk_log_med" in _dg:
                    med_seq.append(_dg["blk_log_med"].detach().reshape(()))
                    du_seq.append(du)
                    u_seq.append(math.log(max(s, 1e-8)))
            fr.append(frac.mean().detach().reshape(()))
            steps.append((jmap, dd, blk, du, floor_mag))
        if not fr:
            return None
        _curve = [round(float(v), 6) for v in torch.stack(fr).tolist()]
        # Per-term MEDIANS over the probe's steps, computed once for the whole
        # curve: one host transfer for every term together, instead of one sync
        # per step.
        _terms = None
        try:
            if term_rows and all(len(r) == len(_TKEYS) for r in term_rows):
                _m = torch.stack([torch.stack(r) for r in term_rows]).tolist()
                _terms = {}
                for j, k in enumerate(_TKEYS):
                    col = sorted(row[j] for row in _m)
                    _terms[k] = round(col[len(col) // 2], 6)
        except Exception:
            _terms = None

        # ---- the DERIVED variant: ref_drop = k*du -------------------------
        # k is the trajectory's OWN median decay in nats per unit log-sigma.
        # Nothing here is chosen; if the trajectory does not decay, k is None and
        # no variant is emitted (there is no decay to allow for).
        _k = None
        _kdu = None
        try:
            # A FROZEN TRAJECTORY MUST NOT YIELD A CONFIDENT k.  Measured: a
            # frozen profile (jump_curve all zeros, med_local_err_real 0.0)
            # produced k = 5.457457 from pure float noise, because the median
            # log-jump still wobbles by a few nats around the log(1e-12) floor and
            # the span is small.  So k and the k*du curve are emitted only when the
            # trajectory actually MOVES, tested RELATIVE to its own scale:
            #   rel = ||x0[i+1] - x0[i]|| / ||x0[i]||
            # which is the same relative measure `probe_diag.x0_minus_x_rel` already
            # reports.  A frozen call gives ~1e-8; a healthy one ~0.03-0.9.  The
            # floor of 1e-5 is two orders above the float32 noise floor (1.2e-7),
            # i.e. anchored to the arithmetic rather than tuned.
            _rel = []
            for _i in range(max(0, len(x0s) - 2)):
                _a, _b = x0s[_i].float(), x0s[_i + 1].float()
                _nb = float(_b.norm())
                if _nb > 1e-8:
                    _rel.append(float((_b - _a).norm()) / _nb)
            _rel_med = (sorted(_rel)[len(_rel) // 2] if _rel else 0.0)
            _moving = _rel_med > 1e-5
            if not _moving:
                # Logged ONCE per process: at the measured prevalence (139 of 224
                # profiles frozen) this would otherwise flood the console.
                global _FROZEN_PROBE_LOGGED
                if not _FROZEN_PROBE_LOGGED:
                    _FROZEN_PROBE_LOGGED = True
                    logging.warning(
                        "[PD-AOS-probe] trajectory FROZEN (median "
                        "||dx0||/||x0|| = %.3g): the model call is returning its "
                        "input, so no k and no k*du curve. This is a PROBE BUG, "
                        "not a model property -- see NOTES.md.", _rel_med)
            if _moving and len(med_seq) >= 2 and len(u_seq) >= 2:
                _span = u_seq[0] - u_seq[-1]
                if _span > 1e-9:
                    _mm = torch.stack(med_seq).tolist()
                    _kk = (_mm[0] - _mm[-1]) / _span
                    if _kk > 1e-6:
                        _k = round(float(_kk), 6)
        except Exception:
            _k = None
        if _k is not None:
            try:
                fr2 = []
                ref2 = None
                for (jmap, dd, blk, du, floor_mag) in steps:
                    _h2, fr_i, ref2, _s2 = _lf_rescue_measure(
                        jmap, dd, blk, float(eg_n or 0.0),
                        floor_mag, ref_prev=ref2,
                        ref_drop=_k * du)
                    fr2.append(fr_i.mean().detach().reshape(()))
                if fr2:
                    _kdu = [round(float(v), 6)
                            for v in torch.stack(fr2).tolist()]
            except Exception:
                _kdu = None
        return {"curve": _curve, "terms": _terms, "curve_kdu": _kdu,
                "k": _k, "n_steps": len(_curve)}
    except Exception:
        return None

def _lf_band_fractions(jmap):
    sh = jmap.shape
    j4 = jmap.reshape(-1, 1, *sh[-2:]).float()
    b1 = _gaussian_blur_separable(j4, 5)
    b2 = _gaussian_blur_separable(b1, 5)
    hi = j4 - b1
    mid = b1 - b2
    lo = b2
    e_hi = hi.pow(2).mean(dim=(1, 2, 3))
    e_mid = mid.pow(2).mean(dim=(1, 2, 3))
    e_lo = lo.pow(2).mean(dim=(1, 2, 3))
    tot = (e_hi + e_mid + e_lo).clamp_min(1e-12)
    n_bt = e_hi.shape[0]
    b = max(1, sh[0])
    if n_bt != b:
        e_hi = e_hi.reshape(b, -1).mean(dim=1)
        e_mid = e_mid.reshape(b, -1).mean(dim=1)
        e_lo = e_lo.reshape(b, -1).mean(dim=1)
        tot = (e_hi + e_mid + e_lo).clamp_min(1e-12)
    return ((e_hi / tot).mean().item(), (e_mid / tot).mean().item(),
            (e_lo / tot).mean().item())

def _lf_band_split(c):
    sh = c.shape
    c4 = c.reshape(-1, sh[1], *sh[-2:]).float()
    b1 = _gaussian_blur_separable(c4, 5)
    b2 = _gaussian_blur_separable(b1, 5)
    hi = (c4 - b1).to(c.dtype)
    mid = (b1 - b2).to(c.dtype)
    lo = b2.to(c.dtype)
    return hi.reshape(sh), mid.reshape(sh), lo.reshape(sh)

def _lf_window(kn):
    t_in = min(max((kn - 0.15) / 0.25, 0.0), 1.0)
    t_out = 1.0 - min(max((kn - 0.60) / 0.25, 0.0), 1.0)
    return (t_in * t_in * (3.0 - 2.0 * t_in)) * (t_out * t_out * (3.0 - 2.0 * t_out))

def _lf_vol_window(kn):
    t_out = 1.0 - min(max((kn - 0.55) / 0.25, 0.0), 1.0)
    return t_out * t_out * (3.0 - 2.0 * t_out)

def _lf_step_field(x0, jmap, vol_prev, med_ref, damp, w, w_vol, gains, kn=1.0,
                   trust=None, w_dir=None, prior=None, spatial_health=False,
                   block=4, rescue=None, resc_state=None, step_d=None,
                   gdir=None, eg_n=0.0, cum=None, gate=None, yield_map=None,
                   fragility=None):
    with torch.no_grad():
        d = _lf_detail_map(x0)
        raw = d.new_zeros(d.shape)
        rho_raw = d.new_zeros(d.shape)
        vol_new = None
        env = _lf_env_fade(kn, eg_n) if gains.get("env_fade", False) else 1.0
        amp_eff = gains["amp"] * env
        ramp_eff = gains["ramp"] * env
        if jmap is not None:
            j = jmap.float().unsqueeze(1)
            med = _lf_map_median(jmap).unsqueeze(1)
            health = None
            if spatial_health and min(jmap.shape[-2:]) >= int(block):
                blk_log, blk_full = _lf_block_scale(jmap, block)
                if med_ref is None or med_ref.shape != blk_log.shape:
                    med_ref = blk_log.clone()
                else:
                    med_ref = torch.maximum(blk_log, med_ref + math.log(0.8))
                health_blk = (0.5 * (blk_log - med_ref) + 1.0).clamp(0.0, 1.0)
                hb = F.interpolate(health_blk, size=jmap.shape[-2:],
                                   mode="bilinear", align_corners=False)
                if hb.shape[0] != jmap.shape[0]:
                    health = hb.reshape(jmap.shape[0], jmap.shape[1],
                                        *jmap.shape[-2:]).unsqueeze(1)
                else:
                    health = hb  # (B,1,H,W), already channel-aligned
                while health.dim() < raw.dim():
                    health = health.unsqueeze(2)
                scale_eff = torch.maximum(med, blk_full.exp().unsqueeze(1))
            else:
                if med_ref is None:
                    med_ref = med.clone()
                else:
                    med_ref = torch.maximum(med, 0.8 * med_ref)
                health = ((med / med_ref.clamp_min(1e-12)).clamp_min(1e-8).log()
                          * 0.5 + 1.0).clamp(0.0, 1.0)
                scale_eff = med
            lj = (j / scale_eff).clamp(1e-4, 1e4).log().clamp(-4.0, 4.0)
            vol_new = lj if vol_prev is None else 0.5 * vol_prev + 0.5 * lj
            dgate = (1.0 - 0.5 * gains["d"] * torch.sigmoid(d / 0.7)).clamp(0.0, 1.0)
            vol_w = w_dir if w_dir is not None else 1.0
            raw = raw + (gains["v"] * torch.tanh(vol_new / 0.6) * dgate
                         * w_vol * health * vol_w)
            stag = (scale_eff / j.float().clamp_min(1e-6)).log().clamp(0.0, 4.0)
            bold = (torch.tanh(stag / 0.8) * torch.sigmoid(-d / 0.5)
                    * w * health)
            raw = raw - gains["s"] * bold
            rho_raw = rho_raw + gains["s"] * bold
        comm = torch.tanh(d / 0.7)
        if prior is not None:
            pw = (0.75 + 0.5 * prior.float()).clamp(0.5, 1.5)
        else:
            pw = 1.0
        raw = raw - gains["d"] * comm * w * pw
        if fragility is not None:
            _fw = fragility.float()
            while _fw.dim() < raw.dim():
                _fw = _fw.unsqueeze(2)
            # fragile regions (probe-measured high local error) take a gentler
            # step: positive raw -> positive m_off -> retain more noise, i.e.
            # trust the model less exactly where it is least reliable.  The
            # authority scales with the model's measured local error.
            _frag_gain = float(gains.get("frag_gain", _LF_FRAG_GAIN))
            raw = raw + _frag_gain * _fw.to(raw.dtype)
        raw = raw * gains["k"]
        if damp is not None:
            keep = 1.0 - damp.float()
            raw = raw * keep
            rho_raw = rho_raw * keep
        trust_t = None
        if trust is not None:
            trust_t = trust.float().unsqueeze(1).clamp(0.2, 1.0)
            if gate is not None:
                if torch.is_tensor(gate):
                    trust_t = trust_t * gate.to(trust_t.dtype)
                else:
                    trust_t = trust_t * trust_t.new_full((), float(gate))
            raw = raw * trust_t
            rho_raw = rho_raw * trust_t
        if yield_map is not None:
            _y = yield_map.float().clamp(0.0, 1.0)
            while _y.dim() < raw.dim():
                _y = _y.unsqueeze(2)
            _keep = (1.0 - _y).to(raw.dtype)
            raw = raw * _keep
            rho_raw = rho_raw * _keep
        rescue_term = None
        resc_rho = None
        hi_boost = None
        resc_out = None
        _bound = amp_eff
        if (rescue is not None and rescue.get("on")
                and min(x0.shape[-2:]) >= int(rescue.get("block", 4))):
            _armed = kn >= float(rescue["start"])
            _wr = (_lf_rescue_window(kn, float(rescue["start"]),
                                     float(rescue["full"])) if _armed else 0.0)
            _st = resc_state if isinstance(resc_state, dict) else None
            if _armed and jmap is not None:
                # Scale-free block (aflops.py:_lf_block_for).  Equals 4 at
                # 128x128, i.e. the old hardcoded value, so runs at the
                # resolution the constant was tuned at are unchanged; every other
                # resolution now measures the same FRACTION of the image instead
                # of the same pixel count.
                _blk = _lf_block_for(jmap)
                floor_mag = (float(rescue.get("floor", 0.02))
                             * float(rescue.get("s", 0.0) or 0.0))
                # ONE implementation of the gate's block test, shared with the
                # probe (aflops.py:_lf_rescue_measure).  The probe can therefore
                # compute the identical statistic before the run.
                hit, frac, ref, lf_share = _lf_rescue_measure(
                    jmap, d, _blk, eg_n, floor_mag,
                    ref_prev=(_st.get("ref") if _st else None))
                g_cov = ((float(rescue.get("max_cov", 0.35)) - frac)
                         / 0.10).clamp(0.0, 1.0)
                need = 2.0 if rescue.get("fast") else 3.0
                credit = _st.get("credit") if _st else None
                if credit is None or credit.shape != hit.shape:
                    credit = torch.zeros_like(hit, dtype=torch.float32)
                credit = (credit + torch.where(hit, 1.0 / need,
                                               -0.5).float()).clamp(0.0, 1.0)
                _st_out = {"ref": ref, "credit": credit}
                if _wr > 0.0:
                    rw_blk = (((credit - 0.5) / 0.25).clamp(0.0, 1.0) * g_cov)
                    rw4 = F.interpolate(rw_blk, size=jmap.shape[-2:],
                                        mode="bilinear", align_corners=False)
                    if rw4.shape[0] != jmap.shape[0]:
                        rw = rw4.reshape(jmap.shape[0], jmap.shape[1],
                                         *jmap.shape[-2:]).unsqueeze(1)
                    else:
                        rw = rw4
                    while rw.dim() < raw.dim():
                        rw = rw.unsqueeze(2)
                    rw = rw.float()
                    if damp is not None:
                        rw = rw * (1.0 - damp.float())
                    if trust_t is not None:
                        rw = rw * ((trust_t - 0.4) / 0.6).clamp(0.0, 1.0)
                    rw = rw.clamp(0.0, 1.0)
                    _cov = float((rw > 0.05).float().mean())
                    resc_out = {"state": _st_out, "rw": None, "cov": _cov,
                                "hit": round(float(frac.mean()), 4),
                                "lf_share": round(float(lf_share.mean()), 4)}
                    if float(rw.max()) > 1e-4:
                        rescue_term = -(float(rescue["amp"]) * _wr) * rw * comm * pw
                        resc_rho = (float(rescue.get("ramp", 0.0)) * _wr) * rw
                        hi_boost = (_wr * rw).clamp(0.0, 1.0)
                        resc_out["rw"] = hi_boost
                else:
                    resc_out = {"state": _st_out, "rw": None, "cov": 0.0,
                                "hit": round(float(frac.mean()), 4),
                                "lf_share": round(float(lf_share.mean()), 4)}
        m_off = amp_eff * torch.tanh(raw)
        if rescue_term is not None:
            m_off = m_off + rescue_term
            _bound = amp_eff + float(rescue["amp"])
        dims = tuple(range(1, m_off.dim()))
        if step_d is not None:
            dlt = step_d.to(m_off.dtype)
            wgt = dlt.float().pow(2).mean(dim=1, keepdim=True).to(m_off.dtype)
            if _LF_CENTERING == "v4_proj":
                wsum = (wgt * wgt).sum(dim=dims, keepdim=True).clamp_min(1e-12)
                m_off = m_off - ((m_off * wgt).sum(dim=dims, keepdim=True)
                                / wsum) * wgt
            else:
                _wbar = ((m_off * wgt).sum(dim=dims, keepdim=True)
                         / wgt.sum(dim=dims, keepdim=True).clamp_min(1e-12))
                m_off = m_off - _wbar
            if gdir is not None:
                g_map = (dlt.float() * gdir.float()) \
                    .sum(dim=1, keepdim=True).to(m_off.dtype)
                gsum = (g_map * g_map).sum(dim=dims,
                                           keepdim=True).clamp_min(1e-12)
                m_off = m_off - ((m_off * g_map).sum(dim=dims, keepdim=True)
                                 / gsum) * g_map
        else:
            m_off = m_off - m_off.mean(dim=dims, keepdim=True)
        _mx_t = m_off.abs().max()
        _scale = (_bound / _mx_t.clamp_min(1e-8)).clamp(max=1.0)
        m_off = m_off * _scale
        cum_out = None
        _drift_B = float(gains.get("drift", 0.0) or 0.0)
        if _drift_B > 1e-6:
            _c4 = (cum.float().unsqueeze(1) if cum is not None
                   else torch.zeros_like(m_off))
            # Reachability for the Batch-2 sweep.  `_LF_DRIFT_RELAX` scales `_c4`,
            # and `_c4` is identically ZERO when `cum is None` -- so the two cases
            # must be countable separately, or "insensitive" could mean "the branch
            # ran with a zero multiplicand" instead of "the constant is dead".
            # `cum is None` is a Python-level test: no device sync.
            if cum is not None:
                _ab_hit("lf.drift_relax_cum")
            else:
                _ab_hit("lf.drift_relax_zerocum")
            if _c4.shape == m_off.shape:
                _B_soft = 0.5 * _drift_B
                _room = ((_drift_B - _c4.abs())
                         / max(_drift_B - _B_soft, 1e-6)).clamp(0.0, 1.0)
                _same = (m_off * _c4) > 0
                m_off = m_off * torch.where(_same, _room, 1.0)
                _ab_hit("lf.drift_relax_ran")
                m_off = m_off - _LF_DRIFT_RELAX * _c4
                m_off = (_c4 + m_off).clamp(-_drift_B, _drift_B) - _c4
                cum_out = (_c4 + m_off).squeeze(1)
        rho = ramp_eff * torch.tanh(rho_raw * gains["k"])
        if resc_rho is not None:
            rho = (rho + resc_rho).clamp(0.0, 0.95)
    return m_off, rho, vol_new, med_ref, resc_out, cum_out

def _lf_emap_from_step(corr_used, emap_pre_cut, x, x1):
    """The per-pixel error the closed-loop trust update sees, for the step that was taken.

    A STEP THE EULER CUT REPLACED HAS NO DEVIATION LEFT TO MEASURE: `x` IS `x1`, so
    `x - x1` is exactly 0, and reading that as "no error" raised trust on precisely the
    pixels whose correction had just been rejected (U09/3933).  The honest reading is
    the deviation that TRIGGERED the cut, captured before the replacement.  Every other
    step keeps the direct measurement, so this is bit-identical where no cut fired.
    """
    if corr_used == "raw" and emap_pre_cut is not None:
        return emap_pre_cut
    return (x - x1).float().pow(2).mean(dim=1).sqrt()


def _lf_trust_update(emap, emap_prev, m_prev, amp_prev, trust):
    with torch.no_grad():
        eff = None
        n_st = None
        if emap_prev is not None and m_prev is not None:
            lr = ((emap + 1e-8).log() - (emap_prev + 1e-8).log())
            lr = torch.nan_to_num(lr, nan=0.0, posinf=4.0, neginf=-4.0)
            _b = lr.shape[0]
            _med = lr.reshape(_b, -1).median(dim=1).values.view(
                _b, *([1] * (lr.dim() - 1)))
            lr_c = lr - _med
            steered = m_prev.abs() > (0.3 * amp_prev)
            bad = steered & (lr_c > 0.35)
            good = steered & (lr_c < -0.15)
            t = trust if trust is not None else torch.ones_like(lr_c)
            t = torch.where(bad, t * 0.55,
                            torch.where(good, (t * 1.15).clamp(max=1.0), t))
            trust = (t + 0.02).clamp(0.2, 1.0)
            n_f = steered.float().sum()
            eff = (good.float().sum() - bad.float().sum()) / n_f.clamp_min(1.0)
            n_st = n_f.detach().clone()
    return trust, emap.detach(), eff, n_st

def _lf_refresh(c, rho, surgical=False, ramp_cap=1.0, hi_boost=None, gen=None):
    with torch.no_grad():
        _rho_max = rho.max()
        _rho_active = _rho_max > 1e-4
        if not bool(_rho_active.item()):
            return c
        _hb = None
        if hi_boost is not None:
            _hb = hi_boost.to(c.dtype)
            if float(_hb.max()) <= 1e-4:
                _hb = None
        if surgical or _hb is not None:
            hi, mid, lo = _lf_band_split(c)

            def _one(band, f, scale):
                r2 = (f * f).clamp(0.0, 0.99)
                if gen is not None:
                    _noise = torch.randn(band.shape, device=band.device,
                                         dtype=band.dtype, generator=gen)
                else:
                    _fb_gen = torch.Generator(device=band.device)
                    _fb_gen.manual_seed(20260915)
                    _noise = torch.randn(band.shape, device=band.device,
                                         dtype=band.dtype, generator=_fb_gen)
                return torch.sqrt(1.0 - r2) * band + f * scale * _noise

            def _rms(band):
                dims = tuple(range(1, band.dim()))
                return band.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)

            def _pmag(band):
                return band.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)

            base_hi = 1.5 if surgical else 1.0
            base_lo = 0.5 if surgical else 1.0
            if _hb is not None:
                f_hi = (rho * base_hi * (1.0 + _hb)).clamp(0.0, 0.75)
                f_lo = (rho * base_lo * (1.0 - 0.5 * _hb)).clamp(0.0, 0.75)
                return (_one(hi, f_hi, _pmag(hi)) + _one(mid, rho, _rms(mid))
                        + _one(lo, f_lo, _pmag(lo))).to(c.dtype)
            _cap = float(ramp_cap)
            return (_one(hi, (rho * base_hi).clamp(max=_cap), _rms(hi))
                    + _one(mid, rho, _rms(mid))
                    + _one(lo, (rho * base_lo).clamp(max=_cap), _rms(lo))).to(c.dtype)
        dims = tuple(range(1, c.dim()))
        rms = c.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
        r2 = (rho * rho).clamp_max(0.99)
        if gen is not None:
            _noise = torch.randn(c.shape, device=c.device,
                                 dtype=c.dtype, generator=gen)
        else:
            _fb_gen = torch.Generator(device=c.device)
            _fb_gen.manual_seed(20260915)
            _noise = torch.randn(c.shape, device=c.device,
                                 dtype=c.dtype, generator=_fb_gen)
        return torch.sqrt(1.0 - r2) * c + rho * rms * _noise


def _pchip_slopes(xs, ys):
    n = len(xs)
    if n == 1:
        return [0.0]
    if n == 2:
        d = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-12)
        return [d, d]
    hs = [xs[i + 1] - xs[i] for i in range(n - 1)]
    ds = [(ys[i + 1] - ys[i]) / max(hs[i], 1e-12) for i in range(n - 1)]
    ms = [0.0] * n
    for i in range(1, n - 1):
        if ds[i - 1] * ds[i] <= 0.0:
            ms[i] = 0.0
        else:
            w_left  = 2.0 * hs[i] + hs[i - 1]   # weight for ds[i-1] (LEFT slope)
            w_right = hs[i] + 2.0 * hs[i - 1]   # weight for ds[i]   (RIGHT slope)
            ms[i] = (w_left + w_right) / (w_left / ds[i - 1] + w_right / ds[i])
    ms[0] = ((2.0 * hs[0] + hs[1]) * ds[0] - hs[0] * ds[1]) / (hs[0] + hs[1])
    if ms[0] * ds[0] <= 0.0:
        ms[0] = 0.0
    elif ds[0] * ds[1] < 0.0 and abs(ms[0]) > abs(3.0 * ds[0]):
        ms[0] = 3.0 * ds[0]
    ms[-1] = ((2.0 * hs[-1] + hs[-2]) * ds[-1] - hs[-1] * ds[-2]) / (hs[-1] + hs[-2])
    if ms[-1] * ds[-1] <= 0.0:
        ms[-1] = 0.0
    elif ds[-2] * ds[-1] < 0.0 and abs(ms[-1]) > abs(3.0 * ds[-1]):
        ms[-1] = 3.0 * ds[-1]
    return ms

def _pchip_eval(x0, x1, y0, y1, m0, m1, t):
    h = x1 - x0
    if h < 1e-12:
        return y0
    s = (t - x0) / h
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return (h00 * y0 + h10 * m0 * h + h01 * y1 + h11 * m1 * h)

def _pchip_interp(curve, xs, grid):
    n = len(curve)
    if n == 0:
        return []
    if n == 1:
        return [float(curve[0]) for _ in grid]
    if n == 2:
        ys = [float(curve[0]), float(curve[-1])]
        x_lo, x_hi = xs[0], xs[-1]
        out = []
        for t in grid:
            t = min(max(float(t), 0.0), 1.0)
            if t <= x_lo:
                out.append(ys[0]); continue
            if t >= x_hi:
                out.append(ys[1]); continue
            a = (t - x_lo) / max(x_hi - x_lo, 1e-9)
            out.append(ys[0] * (1.0 - a) + ys[1] * a)
        return out
    ys = [float(v) for v in curve]
    ms = _pchip_slopes(xs, ys)
    out = []
    x_lo, x_hi = xs[0], xs[-1]
    y_lo, y_hi = ys[0], ys[-1]
    j = 0
    for t in grid:
        t = min(max(float(t), 0.0), 1.0)
        if t <= x_lo:
            out.append(y_lo); continue
        if t >= x_hi:
            out.append(y_hi); continue
        while j + 1 < n and xs[j + 1] < t:
            j += 1
        out.append(_pchip_eval(xs[j], xs[j + 1], ys[j], ys[j + 1],
                              ms[j], ms[j + 1], t))
    return out

def _interp_curve_x(curve, xs, grid):
    return _pchip_interp(curve, xs, grid)


def _sigma_span(sigmas):
    """(hi, lo) log-sigma span of a grid, or None.  THE shared frame (U10/4100)."""
    try:
        raw = [float(v) for v in (sigmas if sigmas is not None else [])]
        rv = [v for v in raw if v > 1e-6]
    except Exception:
        return None
    if len(rv) < 2:
        return None
    lo = math.log(max(rv[-1], 1e-8))
    hi = math.log(max(rv[0], 1e-8))
    if hi - lo < 1e-9:
        return None
    return hi, lo


def _real_step_positions(sigmas):
    span = _sigma_span(sigmas)
    try:
        raw = [float(v) for v in (sigmas if sigmas is not None else [])]
    except Exception:
        return None
    if span is None:
        return None
    hi, lo = span
    out = []
    for v in raw[1:]:
        if v <= 1e-6:
            out.append(1.0)
        else:
            out.append(min(max((hi - math.log(v)) / (hi - lo), 0.0), 1.0))
    return out


def _curve_positions(anchors, sigmas):
    """Curve-sample positions in the RUN's log-sigma frame, in-frame samples only.

    TWO DEFECTS, ONE FRAME.  (U10/4100) The probe's own range and the run's differ
    whenever the probe reached lower (`probe_smin_frac`) or the run is truncated, so
    normalising a curve against ITS OWN span stretched it against the steps it was
    compared to.  (U11/4323) `jump_curve` carries m-1 samples and `curv_curve` m-2, so
    the old `len(sigma_curve) == len(curve) + 1` test silently sent every m-2 curve to
    an invented uniform grid -- the anchors were in `sigma_curve` (and `curv_sigma`)
    all along.

    Out-of-frame anchors are DROPPED, not clamped: `_pchip_interp` needs strictly
    increasing x, and clamping several samples to 0.0 or 1.0 would duplicate them.
    Returns (xs, keep_index) or (None, None).
    """
    span = _sigma_span(sigmas)
    if span is None:
        return None, None
    hi, lo = span
    xs, keep = [], []
    for i, a in enumerate(anchors):
        try:
            la = math.log(max(float(a), 1e-8))
        except (TypeError, ValueError):
            continue
        if not (lo <= la <= hi):
            continue
        p = (hi - la) / (hi - lo)
        if xs and p <= xs[-1]:
            continue
        xs.append(p)
        keep.append(i)
    if len(xs) < 2:
        return None, None
    return xs, keep


def _prof_step_curve(prof, slope, band):
    if not isinstance(prof, dict):
        return None, None
    if band:
        c = prof.get("band_curve") or []
    else:
        c = prof.get("slope_curve") or []
        if not c:
            jc = prof.get("jump_curve") or []
            gc = prof.get("gap_curve") or []
            if jc and len(gc) == len(jc):
                c = [j / max(g, 1e-9) for j, g in zip(jc, gc)]
            else:
                c = jc
    if not c:
        return None, None
    sc = prof.get("sigma_curve") or []
    anchors = None
    if len(sc) >= len(c) + 1:
        # `curve[k]` IS THE VALUE AT `sigma_curve[k+1]` -- the producer's own convention
        # (`curv_curve[k]` is the curvature at `sigma_curve[k+1]`, curves carry m-1 or
        # m-2 entries against the grid's m).  One slice covers both: k+1 .. k+1+len(c).
        anchors = [float(v) for v in sc[1:len(c) + 1]]
    return c, anchors


def _build_step_curve(prof_m, prof_c, w, n_steps, sigmas, slope=False,
                      band=False):
    cm, pm = _prof_step_curve(prof_m, slope, band)
    cc, pc = _prof_step_curve(prof_c, slope, band)
    if not cm and not cc:
        return None
    grid = _real_step_positions(sigmas)
    if not grid or len(grid) < 3:
        n_out = max(8, int(n_steps))
        grid = [(i + 1.0) / n_out for i in range(n_out)]
    n_out = len(grid)

    def _resample(curve, anchors):
        if not curve:
            return None
        if anchors and len(anchors) == len(curve):
            xs, keep = _curve_positions(anchors, sigmas)
            if xs:
                sub = [curve[i] for i in keep]
                if isinstance(sub[0], (list, tuple)):
                    k = len(sub[0])
                    return [_interp_curve_x([float(r[i]) for r in sub], xs, grid)
                            for i in range(k)]
                return _interp_curve_x(sub, xs, grid)
        # LAST RESORT, for a profile without `sigma_curve` (older than the stamp): the
        # curve's own index, centred.  No anchor information exists to do better.
        n = len(curve)
        xs = [(i + 0.5) / n for i in range(n)]
        if isinstance(curve[0], (list, tuple)):
            k = len(curve[0])
            return [_interp_curve_x([float(r[i]) for r in curve], xs, grid)
                    for i in range(k)]
        return _interp_curve_x(curve, xs, grid)

    rm = _resample(cm, pm)
    rc = _resample(cc, pc)
    if rm and rc:
        if isinstance(rm[0], (list, tuple)):
            out = [[(1.0 - w) * a + w * b for a, b in zip(mr, rr)]
                   for mr, rr in zip(rm, rc)]
        else:
            out = [(1.0 - w) * a + w * b for a, b in zip(rm, rc)]
    else:
        out = rm if rm else rc
    if not out:
        return None
    if band:
        if isinstance(out[0], (list, tuple)):
            return [round(float(x), 4) for x in out[0]]
        return [round(float(x), 4) for x in out]
    if len(out) < 4:
        return None
    sm3 = [sum(out[max(0, i - 1):i + 2]) / len(out[max(0, i - 1):i + 2])
           for i in range(len(out))]
    if any(x <= 1e-12 for x in sm3):
        return None
    gm = math.exp(sum(math.log(x) for x in sm3) / len(sm3))
    raw = [math.sqrt(x / gm) for x in sm3]
    qs = sorted(raw)
    q_lo = qs[min(len(qs) - 1, int(0.10 * len(qs)))]
    q_hi = qs[min(len(qs) - 1, int(0.90 * len(qs)))]
    spread = max(q_hi / max(q_lo, 1e-6), 1.0)
    rail_lo = max(0.45, 1.0 / spread)
    rail_hi = min(2.20, spread)
    sched = [min(max(r, min(rail_lo, rail_hi)), max(rail_lo, rail_hi))
             for r in raw]
    return [round(sched[i], 3) for i in range(len(sched))]


def _lf_drift_budget(is_distilled):
    return 0.06 if bool(is_distilled) else 0.10


def _probe_degenerate(prof):
    """True when a probe profile carries no usable dynamics evidence: a
    few-step/distilled model can converge in ~1 probe step, so its measured
    jump/curvature collapse to ~0 and averaging them into the blend only
    dilutes the meaningful per-prompt evidence."""
    if not isinstance(prof, dict):
        return True
    mj = float(prof.get("med_jump", 0.0) or 0.0)
    mc = float(prof.get("med_curv", 0.0) or 0.0)
    return mj < 1e-4 and mc < 1e-4


def _max_order_for_cap(cap, kmax=6):
    """Highest order whose max|Lagrange weight| this cap admits.  Uses the engine's OWN
    geometric weights, so the comparison cannot drift from the consumer."""
    if cap is None:
        return None
    best = 1
    for o in range(2, int(kmax) + 1):
        if _uniform_halfstep_wmax(o) <= float(cap):
            best = o
    return best


def _fuse_probe_prior(prof_m, prof_c, cfg):
    """The reliability prior over the two probes, from fields they ALREADY stamp.  0 NFE.

    Every term is a measurement, not a taste:
      * usable fraction  -- `n_probe_steps` against the steps this run asked for: a probe that
        measured fewer points than requested is a weaker estimate (the grid is built from the
        schedule, so this is normal, not a defect -- it is simply less evidence);
      * grid match       -- `clamp((log_gap_real/log_gap_probe)**2, 0.04, 1)`, the SAME scale the
        rtol derivation already trusts: the probe measured on its own grid, the run runs on the
        real one, and this is how far apart those are;
      * freeze           -- `probe_call.ret_rel == 0.0` is the pass-through detector (the probe
        returned its own input); a frozen probe gets no weight at all;
      * call path        -- `probe_call.path != "primary"` means a fallback invocation, which is
        a different measurement and is discounted rather than trusted.
    Returns None when NEITHER side has usable evidence, so the fusion stands down instead of
    averaging two profiles that measured nothing.
    """
    def _q(prof):
        if not isinstance(prof, dict) or _probe_degenerate(prof):
            return 0.0
        try:
            n = int(prof.get("n_probe_steps") or 0)
            want = max(4, int(cfg.get("probe_steps", 10) or 10))
            usable = min(1.0, n / float(max(want, 1)))
        except (TypeError, ValueError):
            usable = 0.5
        match = 1.0
        try:
            gp = float(prof.get("log_gap_probe") or 0.0)
            gr = float(prof.get("log_gap_real") or 0.0)
            if gp > 1e-9 and gr > 1e-9:
                match = min(max((gr / gp) ** 2, 0.04), 1.0)
        except (TypeError, ValueError):
            pass
        pc = prof.get("probe_call") or {}
        healthy = 1.0
        try:
            rr = pc.get("ret_rel")
            if rr is not None and float(rr) <= 0.0:
                healthy = 0.0
        except (TypeError, ValueError):
            pass
        prim = 1.0 if pc.get("path") in (None, "primary") else 0.5
        return max(0.0, usable * match * healthy * prim)

    qm, qc = _q(prof_m), _q(prof_c)
    if qm <= 0.0 and qc <= 0.0:
        return None
    if qm <= 0.0:
        return 1.0
    if qc <= 0.0:
        return 0.0
    return qc / (qm + qc)


def _fuse_probe_state(prof_m, prof_c, sigmas, cfg):
    """Per-step fusion state for the two probes, or None when it should stand down.  0 NFE.

    WHAT IS FUSED, and in what space: the PREDICTIONS, not the knobs.  Both probes store a
    per-step curvature curve and a per-step jump quantile curve; those are resampled onto this
    run's own sigma positions by the machinery the local-field curves already use, blended by a
    weight the loop updates from the run's OWN measured jump, and only THEN handed to the
    existing derivations.  Averaging the derived KNOBS instead is what made a cap of 2.15 and a
    cap of 2.5 average to 2.325 and silently admit order 4, which is the order the 2.5 arm asked
    for and the 2.15 arm refused -- a mean of two thresholds is not a compromise.
    """
    try:
        pos = _real_step_positions(sigmas)
        if not pos or len(pos) < 3:
            return None
        w0 = _fuse_probe_prior(prof_m, prof_c, cfg)
        if w0 is None:
            return None

        def _curve(prof, key, qidx=None):
            """Resample one stored per-step curve (or one quantile of a quantile curve) onto
            the run's step positions.  Returns a list the length of `pos`, or None."""
            if not isinstance(prof, dict):
                return None
            c = prof.get(key)
            if not c:
                return None
            try:
                if qidx is not None:
                    ys = [float(r[qidx]) for r in c if isinstance(r, (list, tuple))]
                else:
                    ys = [float(v) for v in c]
            except (TypeError, ValueError, IndexError):
                return None
            if len(ys) < 2:
                return None
            xs = [(i + 0.5) / len(ys) for i in range(len(ys))]
            out = _interp_curve_x(ys, xs, pos)
            return [max(0.0, float(v)) for v in out] if out else None

        cm = _curve(prof_m, "curv_curve")
        cc = _curve(prof_c, "curv_curve")
        if cm is None and cc is None:
            return None
        # ONE PREDICTION SERIES PER PROBE, EACH WITH ITS OWN MATCHED RUN-SIDE ESTIMATOR.
        # `jump_q` is the preferred curve (it carries p10/p90, so the residual can be divided by
        # the probe's OWN measured spread), but it is only stored when the distribution pass ran:
        # MEASURED on mklan, `cond_jump: 0` -- the cond profile had no `jump_q` at all, which left
        # the chi-square with nothing to score and froze the weight at its prior.  So a side falls
        # back to `jump_curve`, which every probe stores:
        #     jump_curve[i] = ||x0_{i+1} - x0_i|| / max(||x0_i||, ||x0_{i+1}||)
        # Both norms are over the same tensor, so the sqrt(N) cancels and this IS the ratio of RMS
        # values -- which is why the run-side counterpart is `RMS(dx0)/max(RMS x0_k, RMS x0_k+1)`
        # (`x0_jump_norm_rel`) and not the median-based `x0_jump_p50_rel` that matches `jump_q`.
        # Scoring a curve against the wrong estimator would measure the estimator, not the model.
        def _side(prof, y_q, y_c):
            J = _curve(prof, "jump_q", 1)
            if J is not None:
                p10c, p90c = _curve(prof, "jump_q", 0), _curve(prof, "jump_q", 2)
                if p10c and p90c:
                    sig = [max((b - a) / _NORM_SPAN_10_90, 1e-3 * abs(v))
                           for a, b, v in zip(p10c, p90c, J)]
                else:
                    sig = None
                if sig:
                    return {"J": J, "sig": sig, "src": "jump_q", "y_key": y_q}
            J = _curve(prof, "jump_curve")
            if J is None:
                return None
            # FALLBACK SCALE, and it is NOT an uncertainty: it is the spread of this probe's OWN
            # per-step predictions across the trajectory (10th-90th percentile of its own curve,
            # divided by the 10-90 span of a unit normal -- NOT by 1.349, which is the IQR
            # constant and inflated the spread 1.90x, U11/4347).  It makes the residual
            # dimensionless and comparable between the two sides; it does not say how well the
            # probe predicts, which is exactly what the update is trying to find out.
            qs = sorted(J)
            lo = qs[min(len(qs) - 1, len(qs) // 10)]
            hi = qs[min(len(qs) - 1, (9 * len(qs)) // 10)]
            s = max((hi - lo) / _NORM_SPAN_10_90, 1e-3 * max(abs(max(J)), 1e-9))
            return {"J": J, "sig": [max(s, 1e-3 * abs(v)) for v in J],
                    "src": "jump_curve", "y_key": y_c}

        sides = [_side(prof_m, "x0_jump_p50_rel", "x0_jump_norm_rel"),
                 _side(prof_c, "x0_jump_p50_rel", "x0_jump_norm_rel")]
        if sides[0] is None and sides[1] is None:
            return None
        # THE DERIVATION'S OWN INPUTS travel with the state, read from the same two profiles:
        # the per-order derivative scale (`dk_scale`, model side preferred exactly as the scalar
        # path prefers it), the probe's own grid gap, and the real-grid gap the cap is consumed
        # on (`h_real`), so `_fuse_step_cap` calls `_derived_wmax_cap` the way the scalar
        # derivation does instead of with invented arguments.
        _dk = None
        for _p in (prof_m, prof_c):
            if isinstance(_p, dict) and isinstance(_p.get("dk_scale"), dict) and _p["dk_scale"]:
                _dk = _p["dk_scale"]
                break
        _lg = _hr = None
        for _p in (prof_m, prof_c):
            if not isinstance(_p, dict):
                continue
            if _lg is None and _p.get("log_gap_probe") is not None:
                _lg = _p.get("log_gap_probe")
            if _hr is None and _p.get("log_gap_real") is not None:
                _hr = _p.get("log_gap_real")
        return {
            "w0": float(w0), "w": float(w0), "n": len(pos), "k": 0,
            "cm": cm, "cc": cc, "sides": sides,
            "loss": [0.0, 0.0], "n_ev": [0, 0], "unevaluated": 0,
            "dk": _dk, "log_gap": _lg, "h_real": _hr,
        }
    except Exception:
        return None


def _shipped_cap(med_curv):
    """The cap the SCALAR calibration would use when the derived bound is not enabled: the same
    3.0 -> 1.8 interpolation on log(med_curv + 1) that `_derive_calibrations` ships, clamped to
    [1.5, 2.5].  Mirrored here so the fusion can make the SAME choice the scalar path makes.
    """
    try:
        c = float(med_curv or 0.0)
        if c <= 0:
            return None
        lc = math.log(max(c, 1e-8) + 1.0)
        if lc < math.log(1.5):
            wb = 3.0
        elif lc > math.log(5.0):
            wb = 1.8
        else:
            t = (lc - math.log(1.5)) / (math.log(5.0) - math.log(1.5))
            wb = 3.0 - t * (3.0 - 1.8)
        return max(1.5, min(2.5, wb))
    except Exception:
        return None


def _fuse_step_cap(st, k, cfg, rtol, flags, cap_now=None, allow_raise=False):
    """The order cap for step `k` from the FUSED curvature, conservative on disagreement.

    The two profiles' curvatures are turned into caps by the SAME helper the scalar calibration
    uses (`_derived_wmax_cap`, same `dk_scale`, same real-grid spacing, same budget).  When those
    two caps imply a different maximum ORDER, the lower cap wins -- that is the operator's rule
    ("ask less of the model when the measurements disagree"), and it is applied on the DISCRETE
    decision rather than on the raw number, so the threshold is respected instead of averaged.
    Returns None when no cap is available, in which case the caller keeps today's value.

    `cap_now` IS THE CAP THE RUN WOULD USE WITHOUT THIS ARM, and `allow_raise` decides whether the
    fused cap may exceed it.  With `allow_raise=False` (the shipped default, and its own switch)
    the fused cap can only LOWER the step's cap, so switching the fusion on is a strict refinement
    of today's behaviour -- which is what makes the two arms comparable.  Allowing it to raise is a
    SECOND change: it also admits the probe-derived bound in the cases where that bound's own flag
    leaves it off, so anything measured under it cannot be attributed to the fusion alone.
    The clamp lives HERE rather than in the caller so a rig can test it without a sampling run.
    """
    try:
        if not st or k < 0 or k >= int(st["n"]):
            return None
        w = min(max(float(st.get("w", st.get("w0", 0.5))), 0.0), 1.0)
        cm, cc = st.get("cm"), st.get("cc")
        if cm is None and cc is None:
            return None
        if cm is None:
            c_f, c_m, c_c = cc[k], None, cc[k]
        elif cc is None:
            c_f, c_m, c_c = cm[k], cm[k], None
        else:
            c_f = (1.0 - w) * cm[k] + w * cc[k]
            c_m, c_c = cm[k], cc[k]
        _dk = st.get("dk")
        _h = float(st.get("h_real") or 0.0) or None
        _bud = float(flags.get("wmax_budget", 1.0) or 1.0)
        # WHICH DERIVATION?  THE SAME ONE THE SCALAR PATH USES.  This is the fix for the krea
        # collapse: the arm called `_derived_wmax_cap` UNCONDITIONALLY, while the scalar path only
        # uses that derivation when the profile's `wmax_bound` flag enables it -- so on a
        # shipped-cap model the arm was silently switching on a derivation the engine had left off.
        # MEASURED: krea2TurboOfficialComfy OFF `ord_max [4,5,5,5,4,4,4]` -> ON `[1,1,1,1,1]`, cap in
        # force 1.50 against a pre-arm 5.22 (order 2 instead of 5), while anima's cap moved
        # 5.38 -> 3.28 -- both of which admit order 5, which is why anima looked unaffected.
        _bound_on = bool(flags.get("wmax_bound"))
        st["derivation"] = "derived" if _bound_on else "shipped"
        cap_f = None
        if _bound_on:
            cap_f, _inf = _derived_wmax_cap(c_f, st.get("log_gap"), rtol, dk=_dk,
                                            h_real=_h, budget=_bud)
        else:
            cap_f = _shipped_cap(c_f)
        if cap_f is None and _bound_on and c_m is not None and c_c is not None:
            return None
        caps = []
        if _bound_on:
            for c_i in (c_m, c_c):
                if c_i is None:
                    continue
                _c, _i = _derived_wmax_cap(c_i, st.get("log_gap"), rtol, dk=_dk,
                                           h_real=_h, budget=_bud)
                if _c is not None:
                    caps.append(float(_c))
        out = None
        if len(caps) == 2:
            o_a = _max_order_for_cap(caps[0])
            o_b = _max_order_for_cap(caps[1])
            if o_a is not None and o_b is not None and o_a != o_b:
                out = min(caps)               # conservative: the LOWER order cap
        if out is None:
            out = float(cap_f) if cap_f is not None else (min(caps) if caps else None)
        # THE ARM MAY TIGHTEN, NEVER DEMOTE -- and the reference is the cap the run would ACTUALLY
        # have used (`cap_now`), NOT the scalar `wmax_base`.  MEASURED on the operator's own runs
        # (build aed6d1e233ed): ON admitted order 2-3 on krea where OFF admitted 4-5, and order 3
        # where OFF admitted 4-5 on mklan, changing the admitted order on 15 of every 16 steps
        # (`inert` 0.06).  The scalar floor did not prevent that, and the reason is the whole point:
        # OFF's per-step cap is the scalar PLUS the endgame POLICY raise (pre-arm 5.2-5.4), so
        # flooring at the scalar still stripped that raise.  Flooring at the POLICY value's own
        # order instead makes the cap effect ORDER-NEUTRAL BY CONSTRUCTION -- with raising
        # disallowed the fused cap can only move WITHIN the order band the run already had.
        # It also settles what ON means: it does not change the integrator's order, it changes how
        # the two probes are combined.
        if out is not None and not allow_raise and cap_now is not None:
            try:
                _o = _max_order_for_cap(cap_now)
                if _o is not None:
                    _floor = _uniform_halfstep_wmax(_o)
                    if float(out) < float(_floor):
                        out = float(_floor)
                        _ab_hit("fuse.cap_floored_policy", round(float(_floor), 4))
            except Exception:
                pass
            out = min(float(out), float(cap_now))
        return out
    except Exception:
        return None


def _fuse_step_update(st, k, measured):
    """One step of the weight update: an exponentially-weighted forecaster over the two probes.

        r_i(k) = |y(k) - J_i(k)| / sigma_i(k)      sigma_i = (p90-p10)/1.349 of the probe's own
                                                   per-step quantiles (robust sigma, MEASURED)
        L_i   += r_i(k)^2
        w(k+1) = w0 * exp(-eta * L_i / 2) / normaliser,  eta = 1/n_steps, clipped to [0.1, 0.9]

    `measured` is a DICT of the run's own per-step measurements, because each probe's curve is
    scored against the estimator that MATCHES it: `x0_jump_p50_rel` (per-pixel RMS jump over
    max(rms x0_k, rms x0_k+1), p50 over pixels) matches `jump_q`, and `x0_jump_norm_rel`
    (RMS(dx0)/max(RMS x0_k, RMS x0_k+1)) matches `jump_curve`.  Each side carries its own `y_key`.
    0 NFE.  The update is necessarily LAGGED one step: the measurement only exists after the step
    it describes, so `w(k)` is what the previous steps' evidence earned.  The clip is the
    hysteresis: a source may be doubted, not deleted, once the run is under way.
    """
    try:
        if not st or measured is None:
            return
        k = int(k)
        if k < 0 or k >= int(st["n"]):
            return
        # `setdefault`, not indexing: a state built by an older caller (or by a rig) then updates
        # instead of raising into the bare `except` below, which would have silently turned the
        # whole update into a no-op -- a failure mode that looks exactly like "the probes agreed".
        st.setdefault("n_ev", [0, 0])
        st.setdefault("loss", [0.0, 0.0])
        _sides = st.get("sides")
        if not _sides or len(_sides) < 2:
            st["unevaluated"] = int(st.get("unevaluated", 0)) + 1
            return
        # ONLY CREDIT A STEP WHERE BOTH SIDES WERE SCORED.  A side whose curve is missing keeps a
        # loss of exactly 0.0, and `exp(0) = 1` reads that as a PERFECT prediction -- so the
        # weight drifts toward the probe that was never evaluated.  MEASURED on the first live ON
        # runs (mklan log_00012/14/15): `loss: [15.9079, 0.0]` then `[0.0, 0.0]` with
        # `cond_jump: 0`, i.e. the cond side was never scored.  That is a statistic that is not a
        # measurement, which is the defect this project keeps paying for, so an unscored step now
        # leaves the weight exactly where it was and says so (`unevaluated`).
        _r = []
        for _sd in _sides:
            if not _sd:
                _r.append(None)
                continue
            J = _sd.get("J")
            if not J or k >= len(J):
                _r.append(None)
                continue
            y = measured.get(_sd.get("y_key")) if isinstance(measured, dict) else None
            if y is None:
                _r.append(None)
                continue
            _sig = _sd.get("sig") or []
            _s = float(_sig[k]) if k < len(_sig) else 0.0
            _s = max(_s, 1e-3 * max(abs(float(J[k])), 1e-9))
            _r.append((float(y) - float(J[k])) / max(_s, 1e-9))
        if _r[0] is None or _r[1] is None:
            st["unevaluated"] = int(st.get("unevaluated", 0)) + 1
            return
        for i, r in enumerate(_r):
            st["loss"][i] += r * r
            st["n_ev"][i] = int(st["n_ev"][i]) + 1
        eta = 1.0 / max(int(st["n"]), 1)
        w0 = float(st.get("w0", 0.5))
        try:
            aw = w0 * math.exp(-0.5 * eta * st["loss"][1])
            bw = (1.0 - w0) * math.exp(-0.5 * eta * st["loss"][0])
        except OverflowError:
            aw = bw = 1.0
        tot = aw + bw
        st["w"] = min(max(aw / tot, 0.1), 0.9) if tot > 0 else w0
        st["k"] = k + 1
    except Exception:
        pass


def _lf_fuse_evidence(sig_m, sig_c, prof_m, prof_c, cond_w):
    """Fuse the model-probe and per-prompt cond-probe evidence.

    Takes the ALREADY-EXTRACTED sigs (the caller owns `_one`, which depends on
    three other nested helpers) plus the two profiles, which are needed only for
    the degeneracy test.  Returns (sig_or_None, w_or_None, provenance_string).

    THE GUARD WAS ONE-SIDED, and measured corpus numbers say which way it should
    point.  Over 198 logs the freeze rate by probe type is:

        ENGINE-side probe     7 frozen / 297 healthy     2%
        NODE-side probe     122 frozen /  52 healthy    70%
        COND probe          275 frozen / 118 healthy    70%

    (`ret_rel`, the pass-through detector, separates them perfectly: every 0.0 is
    frozen, every non-zero healthy.)  Yet the fusion guarded only the MODEL side --
    `if _probe_degenerate(prof_m): w = 1.0` -- which hands 100% of the evidence to
    the cond probe, the one 3x MORE likely to be frozen.  And a frozen cond
    profile was not rejected anywhere: the selection path checks only the key and
    `version >= 7`, with no degeneracy test.

    So the guard is now symmetric:
      * only the model probe is degenerate -> w = 1.0, unchanged, with the same
        "model+cond(w=1.00)" provenance string, so runs that already took this path
        are bit-identical;
      * only the cond probe is degenerate  -> w = 0.0 (NEW: the model probe wins);
      * BOTH degenerate -> no evidence at all, so the local field stands down
        rather than averaging two profiles that measured nothing.
    """
    _dm = bool(isinstance(prof_m, dict) and _probe_degenerate(prof_m))
    _dc = bool(isinstance(prof_c, dict) and _probe_degenerate(prof_c))
    if _dm and _dc:
        return None, None, "none"
    if sig_m is not None and sig_c is not None:
        w = min(max(float(cond_w), 0.0), 1.0)
        if _dm:
            w = 1.0
        elif _dc:
            w = 0.0
        sig = {key: (1.0 - w) * sig_m[key] + w * sig_c[key]
               for key in ("ov", "osm", "err", "spike")}
        sig["dist"] = sig_m["dist"] or sig_c["dist"]
        sig["n"] = (1.0 - w) * sig_m["n"] + w * sig_c["n"]
        eg_m, eg_c = sig_m["eg"], sig_c["eg"]
        if eg_m is not None and eg_c is not None:
            sig["eg"] = (1.0 - w) * eg_m + w * eg_c
        else:
            sig["eg"] = eg_m if eg_m is not None else eg_c
        dir_m, dir_c = sig_m["dir"], sig_c["dir"]
        if dir_m is not None and dir_c is not None:
            sig["dir"] = (1.0 - w) * dir_m + w * dir_c
        else:
            sig["dir"] = dir_m if dir_m is not None else dir_c
        # The model-degenerate case keeps its EXACT existing string
        # ("model+cond(w=1.00)"), so runs that already took that path stay
        # comparable in the corpus.  The cond-degenerate case is new and has no
        # legacy to preserve, so it says WHY -- a log reader should not have to
        # remember that w=0.00 means "the cond probe measured nothing".
        _sfx = " cond-degenerate" if (_dc and not _dm) else ""
        return sig, w, "model+cond(w=%.2f)%s" % (w, _sfx)
    if sig_m is not None:
        return sig_m, None, ("model (cond degenerate)" if _dc else "model")
    if sig_c is not None:
        return sig_c, None, ("cond (model degenerate)" if _dm else "cond")
    return None, None, "none"


def _lf_auto_gains(prof_m, prof_c, cond_w, n_steps, sigmas=None, cfg=None):
    def _interp_curve(curve, grid):
        n = len(curve)
        if n == 0:
            return []
        if n == 1:
            return [float(curve[0]) for _ in grid]
        xs = [(i + 0.5) / n for i in range(n)]
        return _pchip_interp(curve, xs, grid)

    def _prof_n(prof):
        jc = prof.get("jump_curve") or []
        n_jump = int(prof.get("n_jump") or len(jc) or 0)
        n_err = int(prof.get("n_err") or max(0, len(jc) - 1))
        n_guard = int(prof.get("n_guard") or len(jc) or 0)
        ns = [x for x in (n_jump, n_err, n_guard) if x > 0]
        return min(ns) if ns else 0

    def _prof_eg(prof):
        # Delegates to the module-level extraction so the PROBE can compute the
        # same scalar before the profile exists (see _endgame_eg_err).
        return _endgame_eg_err(prof.get("endgame"))

    def _prof_dir(prof):
        dc = prof.get("dir_cons_valid") or []
        if not dc:
            dc = prof.get("dir_cons") or []
        if not dc:
            return None
        late = dc[len(dc) // 3:] or dc
        return sum(late) / float(len(late))

    def _one(prof):
        ts = prof.get("trajectory_state")
        if ts is None and isinstance(prof, dict) and prof.get("jump_curve"):
            try:
                ts = _compute_trajectory_state(prof)
            except Exception:
                ts = None
        ts = ts or {}
        try:
            err = float(ts.get("med_local_err_real"))
        except (TypeError, ValueError):
            err = -1.0
        if not (err >= 0.0):
            err = float(prof.get("med_local_err", 0.0) or 0.0)
            gp = float(prof.get("log_gap_probe", 0.0) or 0.0)
            gr = float(prof.get("log_gap_real", 0.0) or 0.0)
            if err > 0 and gp > 1e-9 and gr > 1e-9:
                err = err * min(max((gr / gp) ** 2, 0.04), 1.0)
        sm = float(prof.get("guard_sig_med", 0.0) or 0.0)
        sp = float(prof.get("guard_sig_p90", 0.0) or 0.0)
        spike = (sp / sm) if sm > 1e-9 else 1.0
        return {"ov": min(max(float(ts.get("oversteer", 0.0) or 0.0), 0.0), 1.0),
                "osm": min(max(float(ts.get("oversmooth", 0.0) or 0.0), 0.0), 1.0),
                "err": max(err, 0.0),
                "spike": max(spike, 1.0),
                "dist": bool(prof.get("is_distilled_eff", False)),
                "n": _prof_n(prof),
                "eg": _prof_eg(prof),
                "dir": _prof_dir(prof)}

    sig_m = _one(prof_m) if isinstance(prof_m, dict) else None
    sig_c = _one(prof_c) if isinstance(prof_c, dict) else None
    # The fusion (and its now-SYMMETRIC degeneracy guard) lives in
    # `_lf_fuse_evidence`, so it can be rigged directly against the frozen
    # signatures measured in the corpus.  See its docstring for the numbers.
    sig, _w_used, src = _lf_fuse_evidence(sig_m, sig_c, prof_m, prof_c, cond_w)
    if sig is None:
        ev = {"source": "none", "ov": 0.0, "osm": 0.0, "err": 0.0,
              "spike": 1.0, "dist": False, "n_steps": int(n_steps)}
        return None, ev
    ov, osm = sig["ov"], sig["osm"]
    err, spike = sig["err"], sig["spike"]
    dir_late = sig.get("dir")
    dir_real = None
    if cfg is None or bool(cfg.get("lf_dir_real_feedback", False)):
        if isinstance(prof_c, dict):
            _dr = prof_c.get("dir_real")
            if isinstance(_dr, list) and _dr:
                dir_real = _dr
        if dir_real is None and isinstance(prof_m, dict):
            _dr = prof_m.get("dir_real")
            if isinstance(_dr, list) and _dr:
                dir_real = _dr
    if dir_real is not None:
        _use = dir_real[-3:]
        dir_late = sum(float(x) for x in _use) / len(_use)
        dir_src = "run"
    else:
        dir_src = "probe"
    a4_bump = 0.0
    if dir_late is not None:
        _ratio = min(1.0, max(0.0, (0.2 - dir_late) / 0.2))
        if _ratio > 0.0:
            # Direction consistency (a mean cosine) is scale-invariant: it
            # directly flags a late-stage x0 oscillation and should always
            # raise the anti-artifact authority.  The old gap-ratio gate here
            # conflated the magnitude gap with the direction and zeroed this
            # signal for probes whose log_gap_probe/log_gap_real > 2.
            a4_bump = 0.15 * _ratio
            ov = min(1.0, ov + a4_bump)
    ev = max(ov, osm) if _EV_AB_MODE == "engine" else ov
    few = min(max((9.0 - float(n_steps)) / 6.0, 0.0), 1.0)
    err_n = 1.0 - math.exp(-err / 0.03)
    # same log mapping as s_ovs_spike: the spike ratio spans orders of
    # magnitude, so an exponential saturates and loses all resolution.
    spk_n = min(1.0, max(0.0, math.log(max(spike, 1.0)) / math.log(30.0)))
    k = min(max(0.55 + 0.95 * ev, 0.50), 1.50)
    v = min(max(0.40 + 1.30 * ov, 0.30), 1.70)
    s = min(max(0.25 + 1.25 * osm, 0.20), 1.50)
    d = min(max((0.75 + 0.75 * few) * (1.0 - 0.45 * ov), 0.30), 1.60)
    clamp = min(max(0.38 * (1.0 - 0.45 * err_n) * (1.0 - 0.30 * spk_n)
                    + 0.10 * ev, 0.10), 0.48)
    eg_err = sig.get("eg")
    eg_n = 0.0
    if eg_err is not None and eg_err > 0.0:
        eg_n = 1.0 - math.exp(-eg_err / 0.06)
        clamp = clamp * (1.0 - 0.35 * eg_n)
        v = v * min(1.3, 1.0 + 0.30 * eg_n)
    # Reachability for the Batch-2 sweep.  `v` was ALREADY clamped to a hardcoded
    # 1.70 earlier in this function, so this site can only bind when the `eg_n`
    # branch just above has raised it (up to 1.3x).  Recorded separately from
    # `_lf_v_step`, which is the constant's other use site: two sites, one
    # constant, two measurements.
    if v > _LF_V_MAX:
        _ab_hit("lf.vmax_auto_bind", round(float(v), 6))
    v = min(v, _LF_V_MAX)
    n_eff = float(sig.get("n") or 0.0)
    conf = math.sqrt(n_eff / (n_eff + 2.0)) if n_eff > 0 else 1.0
    k = k * conf
    clamp = clamp * conf
    k = min(max(k, 0.50), 1.50)
    clamp = min(max(clamp, 0.10), 0.48)
    v_sched = None
    # The SCALAR gains use the degeneracy-gated weight from `_lf_fuse_evidence`;
    # the curves must use the same one, or the log's provenance says "w=0.00
    # cond-degenerate" while the curves were blended at cond_probe_weight.
    _curve_w = (min(max(float(_w_used), 0.0), 1.0)
                if _w_used is not None else
                (min(max(float(cond_w), 0.0), 1.0)
                 if (sig_m is not None and sig_c is not None) else 0.0))
    try:
        v_sched = _build_step_curve(
            prof_m, prof_c,
            _curve_w,
            n_steps, sigmas, slope=True)
    except Exception:
        v_sched = None
    hi_ref = None
    try:
        hi_ref = _build_step_curve(
            prof_m, prof_c,
            _curve_w,
            n_steps, sigmas, slope=False, band=True)
    except Exception:
        hi_ref = None
    gains = {"k": round(k, 3), "v": round(v, 3), "d": round(d, 3),
             "s": round(s, 3), "clamp": round(clamp, 3),
             "v_sched": v_sched, "hi_ref": hi_ref, "env_fade": True,
             "drift": round(_lf_drift_budget(sig["dist"]), 3)}
    evidence = {"source": src, "ov": round(ov, 3), "osm": round(osm, 3),
                "err": round(err, 5), "spike": round(spike, 2),
                "dist": sig["dist"], "n_steps": int(n_steps),
                "n": (round(n_eff, 1) if n_eff > 0 else None),
                "conf": round(conf, 3),
                "eg": (round(float(eg_err), 5) if eg_err is not None else None),
                "eg_n": (round(eg_n, 3) if eg_err is not None else None),
                "dir": (round(float(dir_late), 3) if dir_late is not None
                        else None),
                "dir_src": dir_src,
                "bump": round(a4_bump, 3),
                "drift": gains["drift"],
                "v_sched": v_sched is not None,
                "band_ref": hi_ref is not None}
    return gains, evidence

def _compute_snr(s, space, sigma_data):
    sd = float(sigma_data) if sigma_data is not None else 1.0
    if space == "flow":
        return max((1.0 - float(s)) / max(float(s), 1e-6), 1e-6)
    return sd * sd / max(float(s) * float(s), 1e-8)

def _consistency_step(s, s2, space):
    """Step size (in the space the dynamics actually live in) used to normalize
    the per-step x0 change into a derivative for the consistency measure.

    Flow dynamics evolve in DDIM time t = s/(1-s) (that is the variable the
    ancestral split already works in); measuring the x0 change against a
    log-sigma step collapses the step to ~0 near s=1 on a shift-bunched flow
    schedule, which made the consistency signal inert.  VE dynamics evolve in
    log-sigma, so the plain log step is correct there.
    """
    s = float(s)
    s2 = float(s2)
    if space == "flow":
        tf = s / max(1.0 - s, 1e-6)
        tt = s2 / max(1.0 - s2, 1e-6)
        return math.log(max(tf, 1e-8)) - math.log(max(tt, 1e-8))
    return math.log(max(s, 1e-8)) - math.log(max(s2, 1e-8))


def _compute_consistency(derivative_norm, x0_norm, step_size):
    if x0_norm < 1e-8:
        return 0.0
    step = max(float(step_size), 1e-8)
    # derivative_norm is the raw per-step x0 change; divide by the step to get
    # a true derivative before comparing against the x0 magnitude.  The old
    # code *multiplied* by step_size, which drove the term to 0 (consistency
    # -> 1, i.e. "always stable") on flow models with tiny log-sigma steps.
    return 1.0 / (1.0 + float(derivative_norm) / (float(x0_norm) * step))

def _aflops_step(x, x0, hist, s, sn, order=2, lam_min=None, lam_max=None,
                 series_switch=None, order_mask=None, raw_slope=False):
    """Returns (x_new, lam_clamped, lam_raw), or (None, None, None).

    `order_mask` is optional and per-pixel over channels.  When it is not None the
    second-order slope term is multiplied by it, so a caller can give the term only to
    the pixels whose admitted order is >= 2 and leave the low-order/volatile pixels on
    the first-order exponential.  None is the shipped global term, byte-identical.

    `raw_slope` takes the SECOND-ORDER term's decomposition in the UNCLAMPED lambda while
    the first-order decay stays on the clamped one.  The clamp is active every step, so
    the clamped slope carries `(lambda_raw - lambda) * dx/du`, which does not shrink with
    N; taking the slope in the raw decomposition removes that residual.  Default False is
    the shipped clamped decomposition.

    THE LAMBDA IS ESTIMATED FROM ONE ESTIMATOR ON BOTH SIDES.  `v_n` is built from the
    `x0` this call is fed (`x0m`, the order ladder's estimate, on the corrector arm),
    so `v_prev` and `h_prev` read the estimator that SAME step fed, recorded in the
    history entry's fifth slot; a four-slot entry (or a None there) falls back to the
    raw measurement.  Before this, `v_n` used `x0m` and `v_prev` always used the raw
    `x0` (`hist[-1][2]`), so lambda was estimated from two different noisy estimators
    -- and because the difference between them is a SYSTEMATIC offset, the bias does
    not cancel: with a constant offset d the mixed estimate is
    `lambda + <d, dx>/||dx||^2`, while the consistent one cancels it.
    """
    if len(hist) < 1:
        return None, None, None
    du = math.log(max(float(s), 1e-8)) - math.log(max(float(sn), 1e-8))
    if du <= 0.0:
        return None, None, None
    x_prev = hist[-1][3]
    x0_prev = hist[-1][2]
    if len(hist[-1]) > 4 and hist[-1][4] is not None:
        x0_prev = hist[-1][4]
    s_prev_log = hist[-1][0]
    v_n = x0 - x
    v_prev = x0_prev - x_prev
    delta_v = (v_n - v_prev).flatten().float()
    delta_x = (x - x_prev).flatten().float()
    dx_norm_sq = float(delta_x.dot(delta_x))
    if dx_norm_sq < 1e-12:
        return None, None, None
    lam_raw = float(delta_v.dot(delta_x) / dx_norm_sq)
    _lo = _AFLOPS_LAM_MIN if lam_min is None else float(lam_min)
    _hi = _AFLOPS_LAM_MAX if lam_max is None else float(lam_max)
    lam = max(_lo, min(_hi, lam_raw))
    z = lam * du
    az = abs(z)
    if az > 10.0:
        return None, None, None
    h_n = x0 - (1.0 + lam) * x
    # A/B #1: the series/exp branch switch.  Shipped = the fixed 1e-4; derived =
    # sqrt(12*eps), the z at which the series' dominant relative error (z^2/12,
    # verified in scratch_step_constants_rig.py) meets the working precision.
    _sw = _AFLOPS_SERIES_SWITCH if series_switch is None else float(series_switch)
    _sw_is_custom = series_switch is not None
    if _sw_is_custom:
        _ab_hit("series.custom_switch_calls")
    if az < _sw:
        if _sw_is_custom:
            _ab_hit("series.custom_branch_series")
        exp_z    = 1.0 + z * (1.0 + z * (0.5 + z / 6.0))
        phi1_du  = du * (1.0 + z * (0.5 + z / 6.0))
        phi2_du2 = du * du * (0.5 + z / 6.0)
    else:
        if _sw_is_custom:
            _ab_hit("series.custom_branch_exp", round(az, 8))
        exp_z    = math.exp(z)
        em1      = math.expm1(z)
        phi1_du  = em1 / lam
        phi2_du2 = (em1 - z) / (lam * lam)
    x_new = exp_z * x + phi1_du * h_n
    if order >= 2:
        h_prev = x0_prev - (1.0 + lam) * x_prev
        u_log_cur = math.log(max(float(s), 1e-8))
        du_h = s_prev_log - u_log_cur
        if du_h > 1e-8:
            slope = (h_n - h_prev) / du_h
            _phi2 = phi2_du2
            if raw_slope and abs(lam_raw - lam) > 1e-9:
                _zr = lam_raw * du
                if abs(lam_raw) > 1e-9 and abs(_zr) <= 10.0:
                    _hnr = x0 - (1.0 + lam_raw) * x
                    _hpr = x0_prev - (1.0 + lam_raw) * x_prev
                    slope = (_hnr - _hpr) / du_h
                    if abs(_zr) < _sw:
                        _phi2 = du * du * (0.5 + _zr / 6.0)
                    else:
                        _phi2 = ((math.expm1(_zr) - _zr)
                                 / (lam_raw * lam_raw))
            _term = _phi2 * slope
            if order_mask is not None:
                _term = _term * order_mask.to(_term.dtype)
            x_new = x_new + _term
    return x_new, lam, lam_raw

# ---------------------------------------------------------------------------
# Model-internal cache isolation for probe calls
#
# Modern ComfyUI models increasingly keep per-run caches on the model
# instance (Qwen-Image-2.1's prefix/KV cache, Wan Animate pose-branch cache,
# ...).  Those caches are keyed by full-resolution sequence shapes: slots
# selected/created during a probe pass (tiny probe latents, synthetic
# trajectories) do not match -- and on some ComfyUI revisions cannot even be
# *compared* against -- the real run's sequences, which crashes or poisons
# the actual sampling that follows.  Probes therefore always run with the
# model's own caches switched OFF: {"device": "off"} is the explicit opt-out
# contract of ComfyUI's cache machinery itself (see e.g.
# QwenImage21Transformer2DModel.select_prefix_cache), cache entries that are
# not dicts are dropped outright, and the real sampling loop is untouched.
# The rewrite itself is inline in `_apply_probe_profile`-time cache isolation
# (see the `v2["device"] = "off"` block below): it inspects and rewrites the
# cache dicts it finds, so there is no single dict to name here.  A module
# constant `_CACHE_OPT_OUT` used to sit here holding that literal and NOTHING
# read it -- removed 2026-10-04 by the dead-code sweep, which is why the
# contract is described here and implemented at the use site.
# ---------------------------------------------------------------------------
_CACHE_ISOLATION_LOGGED = [False]
_FROZEN_PROBE_LOGGED = False   # one-shot: the frozen-probe warning floods otherwise
_PROBE_CALL_DIAG = {}          # which model call path the last probe step took


# Sampling cfg-function hooks a GRAPH can install through model_options.  Every
# ComfyUI sampling call returns through cfg_function, and comfy/samplers.py:592-596
# computes `cfg_result = x - model_options["sampler_cfg_function"](args)`.  A hook that
# yields zero for probe-shaped input therefore makes the call return EXACTLY its own
# input -- the frozen probe, bit-exact (ret_rel 0.0, not 1e-8).  Measured over the whole
# corpus: node 79 frozen / 0 healthy with the hook present and 0/20 without, cond
# 168/0 vs 0/52, engine healthy either way.  A probe must measure the MODEL, so these
# come off probe calls.  Rig: scratch_cfg_hook_freeze_rig.py.
_PROBE_CFG_HOOK_KEYS = ("sampler_cfg_function", "sampler_pre_cfg_function",
                        "sampler_post_cfg_function", "sampler_calc_cond_batch_function")
_CFG_HOOK_STRIP_LOGGED = [False]


def _probe_model_options(extra_args, strip_hooks=True):
    """Model options copy for probe calls: cfg-function hooks removed and
    model-internal caches disabled.  Returns None when neither applies (the
    caller then keeps using the run's own model_options, zero overhead).

    `strip_hooks=False` is the REAL-SAMPLING retry path: a failed sampling call
    may disable the model's internal caches, but it must not silently remove the
    graph's CFG hooks (RescaleCFG/PAG/SAG/...), which change the image.
    """
    mo = (extra_args or {}).get("model_options")
    if not isinstance(mo, dict):
        return None
    changed = False
    mo2 = None

    # ---- the graph's CFG-function hooks OFF (why: see above) --------------
    if strip_hooks:
        try:
            hook_keys = [k for k in _PROBE_CFG_HOOK_KEYS if k in mo]
            if hook_keys:
                mo2 = dict(mo)
                for k in hook_keys:
                    mo2.pop(k, None)
                changed = True
                if not _CFG_HOOK_STRIP_LOGGED[0]:
                    _CFG_HOOK_STRIP_LOGGED[0] = True
                    logging.info("[PD-AOS-probe] sampling cfg-function hook(s) "
                                 "disabled for probe calls (%s): a probe measures "
                                 "THE MODEL, not the graph's CFG wrapper -- "
                                 "comfy/samplers.py:596 returns x - hook(args), so "
                                 "a hook yielding zero makes the probe report its "
                                 "own input", ", ".join(sorted(hook_keys)))
        except Exception as _e:
            logging.debug("[PD-AOS] probe hook strip skipped: %s", _e)

    # ---- model-internal caches OFF (existing behaviour) -------------------
    try:
        to = mo.get("transformer_options")
        if isinstance(to, dict):
            cache_keys = [k for k in to.keys()
                          if isinstance(k, str) and "cache" in k.lower()]
            to2 = dict(to)
            off_keys = []
            for k in cache_keys:
                v = to2.get(k)
                if isinstance(v, dict):
                    if v.get("device") == "off":
                        continue
                    v2 = dict(v)
                    v2["device"] = "off"
                    to2[k] = v2
                else:
                    to2.pop(k, None)
                off_keys.append(k)
            if off_keys:
                if mo2 is None:
                    mo2 = dict(mo)
                mo2["transformer_options"] = to2
                changed = True
                if not _CACHE_ISOLATION_LOGGED[0]:
                    _CACHE_ISOLATION_LOGGED[0] = True
                    logging.info("[PD-AOS-probe] model-internal cache options "
                                 "disabled for probe calls (%s): probe-shaped "
                                 "sequences must never read or fill the model's "
                                 "own caches", ", ".join(sorted(off_keys)))
    except Exception as _e:
        logging.debug("[PD-AOS] probe cache isolation skipped: %s", _e)

    return mo2 if changed else None


# Instance-level cache isolation: several architectures auto-enable their
# per-run cache directly on the model instance at pre_run time (Qwen-Image-2.1
# arms its prefix cache whenever the sampler starts, no transformer_options
# entry required), so the options-level opt-out above is not sufficient for
# them.  Models of that family expose a `reset_prefix_cache(enabled)` switch
# on the diffusion model -- probes flip it off for the duration of a
# measurement and re-arm it afterwards.  The next real sampling run re-arms
# its own cache via the runtime's pre_run anyway, and a re-armed cache is
# transparent (it simply refills), so this can never change the run's
# numerical results -- only who is allowed to fill the slots.
_PROBE_CACHE_DISABLE_DEPTH = [0]


def _find_prefix_reset(model):
    """Locate the model instance's per-run cache switch (reset_prefix_cache),
    if the diffusion model exposes one.  Returns the callable or None."""
    try:
        patcher = _navigate_to_patcher(model)
        dm = None
        if patcher is not None:
            m = getattr(patcher, "model", None)
            dm = getattr(m, "diffusion_model", None) if m is not None else None
        reset = getattr(dm, "reset_prefix_cache", None) if dm is not None \
            else None
        return reset if callable(reset) else None
    except Exception:
        return None


def _probe_model_cache_guard(model, disable=True):
    """Flip the model instance's own per-run cache off (disable=True) around
    probe measurements, re-arming it on disable=False.  No-op for models
    without such a switch (guarded by hasattr, exceptions swallowed)."""
    try:
        reset = _find_prefix_reset(model)
        if reset is None:
            return
        if disable:
            _PROBE_CACHE_DISABLE_DEPTH[0] += 1
            if _PROBE_CACHE_DISABLE_DEPTH[0] == 1:
                reset(False)
                if not _CACHE_ISOLATION_LOGGED[0]:
                    _CACHE_ISOLATION_LOGGED[0] = True
                    logging.info("[PD-AOS-probe] model-internal per-run "
                                 "cache disabled for the probe measurement "
                                 "(instance switch), re-armed afterwards")
        else:
            if _PROBE_CACHE_DISABLE_DEPTH[0] > 0:
                _PROBE_CACHE_DISABLE_DEPTH[0] -= 1
            if _PROBE_CACHE_DISABLE_DEPTH[0] == 0:
                reset(True)
    except Exception:
        pass


# EVERY probe forward pass in the process.  The engine reports its own DELTA
# across a queue, which is the only number that matters for cost: a probe that
# runs node-side before the sampler and one that runs engine-side inside it cost
# the same, and a cache HIT costs nothing.  Exposed per run as
# `_phase_ms.probe_model_calls` so "how many denoiser calls did this queue pay?"
# is answerable from the log rather than inferred.
_PROBE_CALL_COUNT = [0]


def _probe_model_call(model, x, sigma, extra_args):
    """Revision-agnostic model call for every probe measurement.

    Two contracts are handled here so the probes stay compatible with the
    ComfyUI revision that happens to be running (this is REVISION
    compatibility, not a per-model handler -- the model itself is still
    invoked exactly as the chain invokes it):

    1. denoise_mask placement.  Current ComfyUI requires it as a positional
       argument of KSamplerX0Inpaint.__call__ (older revisions took it
       through extra_args).  When extra_args does not carry it, the first
       call would fail with "missing 1 required positional argument:
       'denoise_mask'" -- so the call is retried with an explicit None.
    2. Probe trajectories are synthetic and UNMASKED.  A real denoise_mask
       tensor belongs to the user's latent; on current ComfyUI it feeds the
       inpaint math inside KSamplerX0Inpaint.__call__ and would not
       broadcast against the probe latent.  It is therefore always dropped
       (logged) for probe calls.

    Both probe latents and cache state are shape-private: the call rides
    the chain with the model's internal caches disabled (see
    _probe_model_options) so probing can never read or pollute them,
    whatever the architecture.
    """
    ea = dict(extra_args or {})
    # Counted here and nowhere else.  `probe_model_calls` is written by the
    # engine's step loop from this counter, so a probe call made NODE-SIDE (when
    # no engine run is in flight) can never write into the previous run's record.
    try:
        _PROBE_CALL_COUNT[0] += 1
    except Exception:
        pass
    # Capture the RUN's cfg hooks BEFORE the sanitizer runs: _probe_model_options returns a
    # copy with them REMOVED, so reading ea["model_options"] further down would always
    # report empty.  This is the value that says whether the probe needed protecting.
    _hooks_in_run = sorted(
        str(_k) for _k in _PROBE_CFG_HOOK_KEYS
        if isinstance(ea.get("model_options"), dict) and _k in ea["model_options"])
    _mopt = _probe_model_options(ea)
    if _mopt is not None:
        ea["model_options"] = _mopt
    dropped = ea.get("denoise_mask")
    if dropped is not None:
        ea["denoise_mask"] = None
        try:
            _shape = tuple(dropped.shape)
        except Exception:
            _shape = None
        logging.info("[PD-AOS-probe] probing unmasked: the run's "
                     "denoise_mask does not apply to the synthetic probe "
                     "latent (%s dropped)",
                     "shape %s" % (_shape,) if _shape is not None
                     else type(dropped).__name__)
    # ---- WHICH CALL PATH ACTUALLY RAN, and with what -----------------------
    # SOLVED -- see "ROOT CAUSE FOUND" in NOTES.md and scratch_cfg_hook_freeze_rig.py.
    # The frozen probes were a GRAPH-INSTALLED `sampler_cfg_function`:
    # comfy/samplers.py:627 returns every sampling call through cfg_function, and
    # :592-596 computes `cfg_result = x - model_options["sampler_cfg_function"](args)`,
    # so a hook yielding zero returns EXACTLY the input -- bit-exact, which is why
    # ret_rel was 0.0 and not 1e-8.  The hooks are now stripped from probe calls by
    # _probe_model_options, and `cfg_hook_in_run` below records which ones the RUN
    # carried, so a log says whether this probe was protected.
    # (The text that stood here before blamed the bare `model(x, sigma)` fallback and
    # called the freeze INTERMITTENT.  Both were wrong; the freeze tracked the graph.)
    #
    # `cfg_hook_in_run` is written BELOW, after the clear() -- writing it here was a bug
    # that shipped in 55825a1 and showed up as cfg_hook_in_run=None in every log: the
    # assignment sat one line above `_PROBE_CALL_DIAG.clear()`, which wiped it before the
    # call that consumes it.  That is precisely the instrumentation trap described in the
    # sibling comment further down, committed anyway.  Read the order before adding a key.
    _PROBE_CALL_DIAG.clear()
    _mo = ea.get("model_options")
    _mo_keys = sorted(str(_k) for _k in _mo.keys()) if isinstance(_mo, dict) else []
    _to = _mo.get("transformer_options") if isinstance(_mo, dict) else None
    _to_keys = (sorted(str(_k) for _k in _to.keys())
                if isinstance(_to, dict) else [])
    _PROBE_CALL_DIAG.update({
        "extra_keys": ",".join(sorted(str(_k) for _k in ea.keys())) or "(none)",
        "model": type(model).__name__,
        "has_denoise_mask": bool("denoise_mask" in ea),
        # ---- WHICH DEVICE THE PROBE MEASURED ON (added 2026-10-08) ------------
        # `_probe_call_device` / `_ensure_model_gpu` can measure on a device the run does
        # not use: they decide from `loaded_size() > 0` and from whether a
        # `load_models_gpu` nudge succeeds, both of which depend on the runtime's eviction
        # state -- i.e. on what ran earlier in the session.  A CPU measurement and a CUDA
        # measurement of the same model are DIFFERENT measurements, and the corpus showed
        # the two producers of one slot at `loaded_mb` 4072.92 against 808.92.  Nothing
        # stamped the device, so that hazard could not be tested from a log; these three
        # keys are what make it testable.
        "x_device": str(getattr(x, "device", "?")),
        "x_dtype": str(getattr(x, "dtype", "?")),
        "model_options": bool(_mo),
        # ---- CONTENT, not just keys -------------------------------------
        # The frozen/healthy comparison left NO difference in path or key names:
        # both report path=primary, the same extra_keys and the same model class
        # (KSamplerX0Inpaint).  The remaining difference has to be in what those
        # arguments CARRY, and for a Flux-family model the conditioning rides in
        # model_options["transformer_options"] -- so record its keys and the size
        # of the conditioning it holds.  Without this the next round is another
        # inference instead of a reading.
        "mo_keys": ",".join(_mo_keys) or "(none)",
        "to_keys": ",".join(_to_keys) or "(none)",
        "to_n": (len(_to) if _to is not None else None),
        # ---- THE CFG HOOKS THE RUN CARRIED ---------------------------------
        # Written AFTER the clear() above (see the note there), and captured BEFORE
        # _probe_model_options stripped them (see _hooks_in_run).  Non-empty means the run
        # installed a sampling cfg hook -- the thing that used to make this probe report
        # its own input -- and that hook was removed for this call.
        "cfg_hook_in_run": _hooks_in_run,
        # ---- THE INPUT ITSELF --------------------------------------------
        # Reading the two paths end to end turned up exactly two structural
        # asymmetries, both confined to the node-invoked probes.  BOTH HAVE SINCE
        # BEEN SETTLED -- kept here so neither is re-derived:
        #   1. `_engine_cond_probe_run` used to run on `model_patcher.clone()`.
        #      REMOVED.  The clone shares the SAME model object
        #      (get_clone_model_override -> self.model, model_patcher.py:428-429;
        #      is_clone -> `self.model is other.model`, :592-595; and the module
        #      that actually runs is _prepare_sampling's `real_model = model.model`,
        #      sampler_helpers.py:202), so it was never a different model.
        #      MEASURED: every clone-path probe returned its own input for all
        #      seven non-anima models (node 56 frozen / 0 healthy, cond 114 / 0),
        #      while the SAME `_run_probe` on the real patcher is healthy for
        #      every model (engine 0 frozen / 60 healthy).
        #      The bookkeeping mechanism (patches_uuid copied unchanged at :450)
        #      is NOT the explanation -- parent and clone share ONE uuid, so it
        #      computes identically on both paths; rig scratch_uuid_identity_rig.py
        #      (20/20) settled that.  Mechanism still unknown.
        #      See NOTES.md, "THE FREEZE IS AN INTERACTION".
        #   2. the node-side path builds its latent with
        #      `_probe_latent_shape(ref_x=None, ...)`, i.e. with NO reference
        #      latent, so the probe is SQUARE.  REFUTED AS THE CAUSE: on the fix
        #      build the node probe and the engine probe pass the IDENTICAL shape
        #      [1, 16, 1, 48, 48] at the identical rank, and one is frozen while the
        #      other is healthy -- so the shape does not decide this.
        # What is still worth recording is the shape actually passed and the sigma it
        # was called at: they show what the probe was handed.
        "x_shape": (tuple(int(v) for v in x.shape) if hasattr(x, "shape")
                    else None),
        "sigma0": (round(float(sigma.reshape(-1)[0]), 6)
                   if hasattr(sigma, "reshape") else None),
    })
    # ---- PATCHER / WEIGHT-PATCH IDENTITY -----------------------------------
    # FROM THE COMFYUI SOURCE, which is the only place this could be settled:
    #   model_patcher.py:446-454  `clone()` builds a fresh patcher, copies no load
    #                             state, and sets `n.patches_uuid = self.patches_uuid`
    #                             -- the clone claims the PARENT's identity.
    #   model_patcher.py:1107      loading stamps `model.current_weight_patches_uuid
    #                             = patches_uuid` onto the model instance.
    #   model_patcher.py:1259      `unpatch_weights = current_weight_patches_uuid is
    #                             not None and (current_weight_patches_uuid !=
    #                             patches_uuid or force_patch_weights)`
    # So a patcher whose uuid MATCHES the stamped one is assumed already applied and
    # is NOT re-patched.  A clone sharing its parent's uuid can therefore believe its
    # weights are applied when they are not.
    #
    # THE CLONE WAS USED ONLY BY THE NODE-INVOKED PROBES (`_engine_cond_probe_run`) --
    # exactly the two paths that returned identity -- against the engine-side probe on
    # the real patcher, which never did.  That clone is now REMOVED (see the block
    # above).  The three values below are still recorded, because they show whether
    # the weights were applied at all, which is the thing the clone was getting wrong.
    try:
        _pp = _navigate_to_patcher(model)
        if _pp is not None:
            _pu = str(getattr(_pp, "patches_uuid", None))
            _mm_ = getattr(_pp, "model", None)
            _cu = str(getattr(_mm_, "current_weight_patches_uuid", None))
            _PROBE_CALL_DIAG["puuid"] = _pu[:8]
            _PROBE_CALL_DIAG["stamped_uuid"] = _cu[:8]
            _PROBE_CALL_DIAG["uuid_match"] = bool(_pu == _cu and _pu != "None")
            try:
                _PROBE_CALL_DIAG["loaded_mb"] = round(
                    float(getattr(_pp, "loaded_size", lambda: 0.0)()
                          or 0.0) / (1024 * 1024), 2)
            except Exception:
                _PROBE_CALL_DIAG["loaded_mb"] = None
            _PROBE_CALL_DIAG["patcher"] = type(_pp).__name__
            # WHERE THE RUNTIME WOULD PUT IT, beside where it actually ran (`x_device`):
            # `load_device` is what the runtime would use, `weights_device` is where the
            # parameters physically are, and a disagreement between the three is the
            # offload/streaming case.  The DTYPES are stamped beside them for a question the
            # log could not answer: "which precision did this run use?"  Measured 2026-10-08:
            # neither the model key (a digest), `loaded_mb` (the dynamic loader's resident byte
            # count) nor `x_dtype` (the LATENT's dtype) states it, and the directory a log sits
            # in is a NAME -- the `int8_model_adapter` transformer option appears in runs under
            # BOTH `anima-base-v1.0` and `animaprequantized_00001_` and is absent from most runs
            # in both, so a folder cannot be read as a precision.  `weights_storage_dtype` reads
            # a quantised parameter's `_qdata` (the payload `_param_digest` samples), i.e. the
            # RAW precision rather than whatever dtype a tensor subclass advertises.
            try:
                _PROBE_CALL_DIAG["load_device"] = str(
                    getattr(_pp, "load_device", None))
                _PROBE_CALL_DIAG["weights_device"] = str(
                    _model_weights_device(_pp))
                _wd, _ws = _model_weights_dtype(_pp)
                _PROBE_CALL_DIAG["weights_dtype"] = _wd
                _PROBE_CALL_DIAG["weights_storage_dtype"] = _ws
                try:
                    _PROBE_CALL_DIAG["model_dtype"] = str(_pp.model_dtype())
                except Exception:
                    _PROBE_CALL_DIAG["model_dtype"] = None
            except Exception:
                pass
    except Exception:
        pass
    # ---- PROBE RANK DECISION -----------------------------------------------
    # x_shape already shows the consequence (4-D vs 5-D).  `shape_src` records
    # the decision that BUILT this probe's latent -- which call site, whether
    # ref_x was a tensor and of what rank, and how the rank was resolved -- so
    # the reason is bound to the latent instead of to whatever ran last.
    try:
        if _LAST_SHAPE_DIAG:
            _PROBE_CALL_DIAG["shape_src"] = dict(_LAST_SHAPE_DIAG)
        if _PROBE_RANK_DIAG:
            _PROBE_CALL_DIAG["rank_why"] = dict(_PROBE_RANK_DIAG)
    except Exception:
        pass
    try:
        _out = model(x, sigma, **ea)
        _PROBE_CALL_DIAG["path"] = "primary"
        _PROBE_CALL_DIAG["out_device"] = str(getattr(_out, "device", "?"))
        # Did the model actually DO anything?  A pass-through returns x itself,
        # so record the relative displacement of the returned x0 from the input.
        # The frozen probe's ~1e-8 is float rounding on an unchanged trajectory;
        # a working call is 0.03-1.0.
        try:
            _nx = float(x.float().norm())
            _d = float((_out.float() - x.float()).norm())
            _PROBE_CALL_DIAG["ret_rel"] = round(_d / max(_nx, 1e-8), 8)
        except Exception:
            _PROBE_CALL_DIAG["ret_rel"] = None
        return _out
    except TypeError as e1:
        msg = str(e1)
        attempts = []
        if "denoise_mask" in msg:
            # Required positionally by this revision and absent from
            # extra_args (e.g. the node-side model probe, whose extra_args
            # start empty).
            attempts.append(("denoise_pos",
                             lambda: model(x, sigma, None, **ea)))
            attempts.append(("denoise_pos_min", lambda: model(
                x, sigma, None, model_options=ea.get("model_options", {}),
                seed=ea.get("seed"))))
        if "keyword" in msg:
            # The chain model only accepts the explicit kwargs of this
            # revision; retry with progressively less.
            # *** THIS IS THE SUSPECT *** -- `model(x, sigma)` carries NO
            # conditioning and NO model_options, and the node-side probe is the
            # documented case that reaches it.  If this path is what a FROZEN run
            # takes, the fix is to give the node-side probe the guider's real
            # extra_args rather than to touch any threshold.
            attempts.append(("BARE_NO_ARGS", lambda: model(x, sigma)))
        for _pname, att in attempts:
            try:
                _out = att()
                _PROBE_CALL_DIAG["path"] = _pname
                return _out
            except TypeError:
                continue
        _PROBE_CALL_DIAG["path"] = "FAILED"
        raise e1

# THE AMPLITUDE LADDER the noise-survival probe walks, as noise-variance
# FRACTIONS: amplitude = sqrt(f) * sigma.  Wide enough to cross the point where
# the model stops correcting the kick, which is the shape a derived noise scale
# needs and which the original (0.2, 0.35) ladder never reached -- measured on
# the first curves, `div` was still only 0.22 at the top of the old ladder.
_NOISE_SURVIVAL_LADDER = (0.2, 0.35, 0.5, 0.7)
# WHERE the ladder is evaluated, as quantiles of the RUN'S OWN sigma range.  The
# original code sampled `m//4` and `3m//4` of the PROBE GRID, which on a 16-step
# turbo run landed at sigma 0.66 and 0.29 -- so three separate measurement rounds
# never sampled above 0.66, leaving the composition phase (where a noise schedule
# matters most) unmeasured.
#
# HIGHEST QUANTILE FIRST, AND THAT ORDER IS LOAD-BEARING.  The cap is applied by
# taking targets in the order listed, so listing an endgame quantile first spends
# the budget before the high-sigma end can be reached -- which is exactly the gap
# this change exists to close.  Measured with the reverse order, the four points
# landed on 0.70..0.85 and never reached the endgame.
_NOISE_SURVIVAL_QUANTILES = (0.95, 0.62, 0.40, 0.18)


def _noise_survival_idxs(grid, sigmas, want):
    """Probe-grid indices to evaluate, chosen to span the RUN'S sigma range.

    `want` caps how many points are measured, since each costs a full amplitude
    ladder of model calls.  Indices are matched to the nearest grid point and
    de-duplicated.

    THE LAST GRID POINT IS ALWAYS INCLUDED, AND IS RESERVED BEFORE THE CAP IS
    APPLIED.  The lowest sigma the probe reaches is the endgame, where `div` is
    smallest and where the old sampling was weakest; an earlier version appended
    it and then truncated with `[:cap]`, which silently dropped it whenever the
    quantile choices already filled the cap -- so the endgame could go missing
    from a measurement whose whole purpose was to cover the range.
    """
    m = len(grid)
    if m < 2:
        return []
    floor_idx = m - 1
    # `sigmas` CAN BE A PLAIN LIST **OR** A TORCH TENSOR, and `float(tensor)` on a
    # multi-element tensor raises "Boolean value of Tensor with more than one value
    # is ambiguous".  The original sampler only ever indexed `grid`, so it never
    # met this; choosing indices from the RUN's sigma range does.  Normalised with
    # `float(...)` per element, which works for both, and any element that will not
    # convert is skipped rather than allowed to abort the measurement.
    pos = []
    for _v in _sigmas_as_floats(sigmas):
        if _v > 1e-6:
            pos.append(_v)
    if pos:
        hi, lo = max(pos), min(pos)
    else:
        hi, lo = float(grid[0]), float(grid[-1])
    cap = max(2, int(want))
    if hi <= lo:
        idxs = [m // 2, floor_idx]
        return sorted(set(idxs))[:cap]
    targets = [hi - q * (hi - lo) for q in _NOISE_SURVIVAL_QUANTILES]
    idxs = []
    for t in targets:
        if len(idxs) >= cap - 1:
            break
        i = min(range(m), key=lambda j: abs(float(grid[j]) - t))
        if i not in idxs and i != floor_idx:
            idxs.append(i)
    idxs.append(floor_idx)
    return sorted(idxs)


def _sigmas_as_floats(sigmas):
    """`sigmas` -> a flat list of python floats, whatever shape it arrives in.

    WHY THIS EXISTS RATHER THAN A CAST AT EACH SITE.  `float(v)` raises
    `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`
    when `v` is a CONTAINER of tensors rather than a tensor -- so
    `[float(v) for v in sigmas]` works for a 1-D tensor and fails for a 1-tuple or
    list WRAPPING one, which is exactly what the callers pass.  That failure is
    caught by the surrounding probe handlers and returned as a fallback constant,
    so it presents as "the measurement produced nothing" rather than as an error;
    two real restarts were spent on that presentation.

    Handles: a 1-D tensor, a list/tuple of floats, a list/tuple of 0-d tensors, a
    single-element container holding a 1-D tensor, and a 2-D tensor (flattened).
    Non-numeric entries are dropped rather than aborting the caller.
    """
    out = []
    def _push(v):
        if v is None:
            return
        if torch.is_tensor(v):
            for x in v.reshape(-1).tolist():
                try:
                    out.append(float(x))
                except (TypeError, ValueError):
                    pass
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _push(x)
            return
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    _push(sigmas)
    return out


_NOISE_SURVIVAL_LAST_ERROR = [None, None]


def _probe_noise_survival(model, extra_args, xs, x0s, grid, ones, gen=None,
                          curve=False, sigmas=None):
    """How much noise this model still corrects a step later, at a given sigma.

    Returns the SHIPPED answer -- one scalar, the largest amplitude fraction that
    kept `div` under 0.35 at every sampled point, clamped to [0.1, 0.35] -- and,
    when `curve` is set, ALSO the per-(sigma, amplitude) `div` series that the
    scalar is computed FROM.

    WHY THE SERIES IS WORTH KEEPING.  `div` is the whole measurement: how far an
    injected perturbation still shows in the model's next prediction, relative to
    an unperturbed reference.  It is the empirical form of "how much of a kick
    can this model absorb here", which is the quantity a stochastic sampler's
    noise scale would be derived from rather than hand-tuned.  Until now it was
    computed and immediately reduced by a comparison (`if div < 0.35`), so the
    only surviving record was a scalar -- and measured across 10,602 logged
    profiles that scalar is `None` in 69% of them and at the clamp or the
    skipped-probe fallback in most of the rest, i.e. it has rarely carried a
    usable per-model reading.

    THE SERIES DOES NOT `break` ON A FAILED AMPLITUDE (changed after the first
    measurement round).  The old ladder stopped at the first `div >= 0.35`, which
    truncated exactly the region a derivation needs -- where the model STOPS
    correcting.  Now the whole ladder is walked and every step is recorded,
    including the failures, while the SHIPPED scalar is still computed from the
    same passing-prefix rule so nothing downstream moves.

    The measurement costs one model call per (sigma, amplitude) pair, plus one
    reference call per sigma, and runs on the PROBE path only, once per model,
    cached -- it does not touch the per-step NFE budget.
    """
    series = []
    # Cleared per call so a stamp never reports the PREVIOUS measurement's error.
    _NOISE_SURVIVAL_LAST_ERROR[0] = None
    _NOISE_SURVIVAL_LAST_ERROR[1] = None
    try:
        m = len(x0s)
        # CLAMP THE CHOSEN INDICES TO THE TRAJECTORY, WHICH IS SHORTER THAN THE
        # GRID.  `_probe_reach_extend` appends points to `grid` that the probe
        # trajectory never visits (measured: `grid_pts: 8, sigmas_pts: 9` on a real
        # run), so an index chosen against the grid can be past the end of `xs` /
        # `x0s`.  The old code also indexed `x0s[i + 1]`, so the highest usable
        # index is `len-2`.
        #
        # THIS IS WHAT BROKE THE WIDENED LADDER, and it is worth naming precisely
        # because it failed SILENTLY: the old sampling used `m // 4` and `3m // 4`
        # of the TRAJECTORY, which were always in range; the widening switched to
        # indices chosen against the GRID's sigma range, which are not.  The
        # resulting IndexError was caught by this function's own `except`, which
        # returns the skipped-probe fallback -- so on 28 of 28 real profiles the
        # measurement looked like it had simply been switched off, with no error
        # anywhere.  Measured before the fix: `series_len: 0`, `safe_fresh_frac:
        # 0.15`, on a run whose switch was on and whose stamp said `ran: true`.
        m_eff = min(m, len(grid))
        idxs = [i for i in _noise_survival_idxs(
            grid, sigmas, len(_NOISE_SURVIVAL_LADDER)) if i <= m_eff - 2]
        if not idxs:
            _lo = min(_NOISE_SURVIVAL_LADDER)
            return (_lo, series) if curve else _lo
        # THE LADDER'S OWN TOP, not a literal.  The shipped code used a fixed
        # 0.35 -- the top of the OLD two-rung ladder -- for both the permissive
        # start and the final clamp, so widening the ladder could never report a
        # model that tolerates more: everything above 0.35 was masked by the
        # clamp.  Derived from the ladder, the scalar keeps its exact shipped
        # meaning (largest amplitude that passed everywhere) while being able to
        # say "this model survived the widest kick we tried".
        _top = max(_NOISE_SURVIVAL_LADDER)
        safe = _top  # start permissive; min() tightens below
        for i in idxs:
            if i + 1 >= m:
                i = max(0, m - 2)
            s_m = grid[i]
            sn_m = grid[i + 1]
            x_at = xs[i]
            x0_ref = x0s[i]
            x_next_clean = exact_step(x_at, x0_ref, s_m, sn_m)
            x0_next_clean = _probe_model_call(model, x_next_clean,
                                              sn_m * ones, extra_args)
            ref_norm = float(x0_next_clean.float().norm().clamp_min(1e-8))
            safe_i = 0.1
            broke = False
            for f in _NOISE_SURVIVAL_LADDER:
                noise = torch.randn(x_at.shape, device=x_at.device,
                                    dtype=torch.float32,
                                    generator=gen).to(x_at.dtype)
                amp = math.sqrt(f) * float(s_m)
                xpert = x_at + noise * amp
                x0_pert = _probe_model_call(model, xpert, s_m * ones,
                                            extra_args)
                x_next_pert = exact_step(xpert, x0_pert, s_m, sn_m)
                x0_next_pert = _probe_model_call(model, x_next_pert,
                                                 sn_m * ones, extra_args)
                div = float((x0_next_pert - x0_next_clean).float().norm()) / ref_norm
                ok = div < 0.35
                if curve:
                    series.append({
                        "sigma": round(float(s_m), 6),
                        "sigma_next": round(float(sn_m), 6),
                        "f": f,
                        "amp_over_sigma": round(math.sqrt(f), 6),
                        "div": round(div, 6),
                        # the magnitudes the ratio is built from, so a reader can
                        # see whether `div` is small because the reference is
                        # large rather than because the perturbation died
                        "ref_norm": round(ref_norm, 6),
                        "pert_norm": round(float((x0_next_pert - x0_next_clean)
                                                 .float().norm()), 6),
                        # WHY THE LADDER STOPPED, recorded on the entry that
                        # stopped it: without it a truncated ladder and a ladder
                        # that ran out are indistinguishable in the log.
                        "threshold_pass": bool(ok),
                        "ladder_stopped_here": bool(broke or not ok),
                    })
                if ok:
                    safe_i = f
                else:
                    broke = True
                    # RECORD, THEN STOP: the shipped scalar must keep its
                    # passing-prefix meaning, but the failing step is already in
                    # `series` above, so the truncation is visible rather than
                    # silent.
                    break
            safe = min(safe, safe_i)
        out = max(safe, min(_NOISE_SURVIVAL_LADDER))
        return (out, series) if curve else out
    except Exception as _exc:
        # THE REASON IS KEPT, NOT SWALLOWED.  This handler is why the widened
        # ladder failed invisibly: an IndexError against a trajectory shorter than
        # the grid returned the skipped-probe fallback, so every consumer saw a
        # clean "not measured" instead of an error, and it took a purpose-built
        # stamp to find it.  The RETURN still has to be the shipped 2-tuple, so the
        # reason goes to a module-level slot the caller stamps into the profile.
        # Bound as `_exc` rather than read from `sys`, which this module does not
        # import -- the first version of this handler referenced `sys` inside its
        # own try/except and therefore recorded nothing at all.
        try:
            _NOISE_SURVIVAL_LAST_ERROR[0] = "%s: %s" % (type(_exc).__name__, _exc)
        except Exception:
            pass
        return (0.15, series) if curve else 0.15


def _as_survival_pair(out):
    """Normalise the two documented return shapes of `_probe_noise_survival`.

    The function returns a float when `curve=False` and `(float, series)` when it
    is true.  Unpacking it unconditionally turned a skipped series into a
    TypeError that `_run_probe` reported as a failed whole probe.
    """
    return out if isinstance(out, tuple) else (out, [])


def _noise_survival_wanted(cfg):
    """Does this run have a consumer for the noise-survival measurement?

    `eta` is the ceiling the measurement budgets for; zero/None means there is no
    noise to place.  The explicit `probe_noise_survival` override wins, for A/B
    and inspection runs.
    """
    ov = cfg.get("probe_noise_survival")
    if ov is not None:
        return bool(ov)
    if bool(cfg.get("eta_from_probe", False)) or bool(cfg.get("eta_budget_from_probe", False)):
        return True
    try:
        return float(cfg.get("eta") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False

# ---------------------------------------------------------------------------
# DERIVED CALIBRATION CONSTANTS -- measured, rigged, and behind A/B TOGGLES.
#
# Two derived values were effectively CONSTANTS on real runs, for two unrelated reasons, and
# both are now derivable rather than hand-set.  Each has its own toggle so behaviour can be
# compared on one run, and BOTH the shipped and the derived value are logged every time
# regardless of the toggle, so a single run shows the whole delta.
#
# 1. THE PERCENTILE ESTIMATOR.  Both `_pct` bodies used `index = int(p*n)`, which for n = 2
#    gives int(0.5*2) == int(0.9*2) == 1 -- so `guard_sig_p90` IS `guard_sig_med`, r = 1.0 by
#    construction, and `guard_floor = 0.075*sqrt(r)` collapsed to its r = 1 value.  The cond
#    probe collects m-1 guard signals and Krea runs n_probe_steps = 3 (vs anima's 8), which
#    is exactly the observed split.  The estimator is ALSO biased at every n: at n = 10 it
#    returns the MAXIMUM for p = 0.9 and the 60th percentile for p = 0.5.  The corrected form
#    interpolates at position (n-1)*p.  Rig: scratch_percentile_collision_rig.py (12/12),
#    including the corpus prediction n_guard <= 2 <=> p90 == med with 0 violations.
#
# 2. THE WMAX CAP.  `wmax_base` is not a gain: `_mid_x0` returns `max(abs(w) for w in ws)`,
#    the largest LAGRANGE weight of the extrapolation to the interval midpoint, and the order
#    is REJECTED when that exceeds the cap.  Those weights are purely geometric (invariant to
#    shift+scale of the u axis) and, on a uniform grid extrapolating half a step, depend on
#    the ORDER ALONE: 1.5, 1.875, 2.1875, 3.28125, 5.41406 for orders 2..6.  So a cap of 2.5
#    is an ORDER SELECTOR -- admit <= 4, reject >= 5 -- chosen without ever reading the grid
#    spacing h, while the error bound carries h^k.  The derived cap reads h and the tolerance:
#
#        |err_k| <= geom_k / k! * |d^k x0/du^k|,   geom_k = (2k-1)!! / 2^k * h^k
#        =>  allow order k  iff  c_k * h^k * med_curv <= tol,  c_k = (2k-1)!! / (2^k * k!)
#
#    `med_curv` IS the measured derivative scale, not a proxy: the builders compute it as a
#    central second difference of x0 in u normalised by |x0| (aflops.py:3821-3823), i.e.
#    exactly |d^2 x0/du^2| / |x0|.  `h` is the measured mean log gap.  Both were verified
#    against an analytic non-degenerate ground truth: scratch_lagrange_geometric_rig.py
#    (6/6), including the closed form geom_k = h^k * (2k-1)!! / 2^k to 1.12e-15.
#
#    NOTE the c_k form uses the same derivative scale for every k -- an explicit smoothness
#    assumption, and a CONSERVATIVE one for k > 2 (higher derivatives of a smooth x0 decay).
#    It is stated here rather than buried: the bound is verified for whichever k is selected.
# ---------------------------------------------------------------------------
_CALIB_FLAGS_KEY = "calib_flags"
# How many calibrations took each arm, copied into the profile as `calib_ab_counts` at the end of every
# derivation (`_derive_calibrations`, `_derive_cond_calibrations`).  IT IS A PER-RUN COUNTER: zeroed at
# engine entry beside `_ab_reach_reset` and again when the run closes, so a profile's stamp describes
# the run that derived it.  Until 2026-10-07 it was never zeroed, so the stamp was the whole process's
# running total and grew monotonically across a session (census: scratch_fix_logging_1_3_out.txt,
# section 4).  Nothing in the sampler reads it -- it is written to the log and to the profile, and
# nowhere else.
_CALIB_AB = {"pct_derived": 0, "pct_shipped": 0,
             "wmax_derived": 0, "wmax_shipped": 0, "wmax_bound_unavailable": 0}


def _odd_double_factorial(k):
    """(2k-1)!! = 1*3*5*...*(2k-1), computed in code -- never quoted from memory."""
    p = 1
    for i in range(1, int(k) + 1):
        p *= (2 * i - 1)
    return p


def _ck(k):
    """(2k-1)!! / (2^k * k!) -- the geometric factor of a half-step uniform extrapolation,
    per unit h^k.  0.375, 0.3125, 0.2734, 0.2461 for k = 2..5."""
    return _odd_double_factorial(k) / float((2 ** int(k)) * math.factorial(int(k)))


def _uniform_halfstep_wmax(order):
    """max|Lagrange weight| for extrapolating half a step past `order` uniform nodes.

    Computed from the engine's OWN `_lagrange_weights`, so the cap and the consumer can never
    drift apart.  Verified geometric (shift/scale invariant) in the rig.
    """
    order = int(order)
    if order < 2:
        return 0.0
    us = [float(i) for i in range(order)]
    u_t = us[-1] + 0.5
    try:
        return float(max(abs(w) for w in _lagrange_weights(us, u_t)))
    except Exception:
        return 0.0


def _derived_wmax_cap(med_curv, log_gap, tol, dk=None, kmax=5, h_real=None,
                      budget=1.0):
    """The order the error bound permits, as a cap on max|Lagrange weight|.

    `dk` MUST supply the per-order derivative scale `|d^k x0/du^k| / |x0|`, measured for each
    k the caller wants certified.  An order is allowed only when it is BOTH measured and
    within tolerance -- we certify, we do not assume.

    WHY IT REFUSES WITHOUT `dk`: the first implementation substituted the k = 2 curvature for
    every k.  `scratch_derived_cap_rig.py` measured that as NOT a bound -- the true error
    exceeded it by a consistent 2.6-4.8x, which is the `c^(k-2)` factor the substitution drops
    (c = 4.9 at k = 5 for the rig's ground truth).  It also caught a units bug in that rig:
    comparing an ABSOLUTE error against a RELATIVE bound inflated the apparent failure to
    113x.  Both are recorded so neither is re-derived.  Passing dk = None therefore returns
    None, the caller keeps the shipped map, and `info["reason"]` says why.

    PROPAGATED INTO STEP ERROR before comparing with `tol`, which is the second correction and
    the one that made this arm refuse on every real run.  The step is
    `exact_step(x, x0_mid, s, s2) = r*x + (1-r)*x0_mid` with `r = s2/s = exp(-h)`, so an error
    EPS in the extrapolated x0_mid becomes `(1 - e^-h) * EPS` in the step.  `tol` bounds the
    STEP error, so the comparison is `(1 - e^-h) * bound <= tol`.  Comparing the x0 error
    against tol directly was ~70-100x too strict: measured bound(k=2) was 0.356 / 0.480 /
    0.770 against tol 0.005 / 0.0067 / 0.0056, so no order ever qualified and wmax_bound was
    inert on every run the operator made.

    Returns (cap_or_None, info).

    EVALUATED ON THE REAL GRID (`h_real`) when it is measured, because that is the grid the
    cap is consumed on.  `log_gap_probe` is the PROBE grid's spacing, 1.9x-4.9x coarser than
    the real one on the operator's own runs, and `h` enters as `h^k` -- so the comparison was
    ratio^k (14x..2900x over k = 2..5) too strict and refused every order on every logged
    run.  `scratch_extrap_remainder_rig.py` measures that directly: on the corpus' own
    dk/rtol/grids the probe-gap form admits an order at 0/39 (anima) and 0/15 (krea) steps,
    the real-gap form at 36/39 and 11/15.  Both rows are logged (`allowed` at the used h,
    `allowed_probe` at the probe h) so one run shows the whole delta.

    `budget` scales the tolerance THIS BOUND is compared against, and nothing else -- never
    `rtol` itself, which the corrector also consumes.  It is the operator's nudge for the
    one input that is not a measurement: `dk` is a trajectory statistic measured on the
    probe grid, so it is not a per-step upper bound, and the rig quantifies that residual
    (58 of 204 cases under-bound with the shipping median, up to 578-1807x at individual
    steps).  Default 1.0 = neutral; the arm is bit-identical to no multiplier at 1.0.
    """
    info = {"k": None, "bound": None, "ck": None, "h": log_gap, "med_curv": med_curv,
            "tol": tol, "cap": None, "allowed": [], "reason": None,
            "prop": None, "dk_present": sorted(int(k) for k in (dk or {}))}
    try:
        h_probe = float(log_gap or 0.0)
        c = float(med_curv or 0.0)
        t = float(tol or 0.0)
        budget = float(budget or 1.0)
        t_eff = t * max(budget, 1e-6)
        info["tol_rtol"] = t
        info["budget"] = budget
        info["tol_eff"] = t_eff
        info["h_probe"] = h_probe
        # WHICH GRID THE BOUND IS EVALUATED ON.  The cap is CONSUMED on the real schedule,
        # so the remainder belongs on the real schedule too: `log_gap_probe` is the probe
        # grid's spacing, which on the operator's own runs is 1.9x-4.9x COARSER than the
        # real grid, making the bound ratio^k too strict -- measured, that alone refused
        # every order on every logged run.  Both grids' bounds are reported so one run
        # shows the whole delta.
        h = h_probe
        h_src = "probe"
        try:
            if h_real is not None and float(h_real) > 0.0:
                h = float(h_real)
                h_src = "real"
        except (TypeError, ValueError):
            pass
        info["h_used"] = h
        info["h_src"] = h_src
        if not (h > 0.0 and c > 0.0 and t > 0.0):
            info["reason"] = "degenerate input (need h > 0, curv > 0, tol > 0)"
            return None, info
        if not dk:
            info["reason"] = ("per-order derivative scale not measured -- the k = 2 "
                              "curvature is NOT a valid stand-in for |d^k x0/du^k|")
            return None, info
        # (1 - r) with r = s2/s = exp(-h): how much of an x0 error survives into the step
        prop = 1.0 - math.exp(-h)
        info["prop"] = round(prop, 8)
        if not (prop > 0.0):
            info["reason"] = "step factor (1 - exp(-h)) is not positive"
            return None, info
        best = None
        for k in range(2, int(kmax) + 1):
            d_scale = dk.get(k)
            if d_scale is None:
                info["allowed"].append([k, None, "not measured"])
                continue
            bound = _ck(k) * (h ** k) * float(d_scale)
            step_err = prop * bound
            info["allowed"].append([k, round(bound, 8), round(float(d_scale), 8),
                                    round(step_err, 8)])
            if step_err <= t_eff:
                best = k
        # the same rows at the PROBE gap, for the audit: the delta is the h correction
        info["allowed_probe"] = [
            [k, round(_ck(k) * (h_probe ** k) * float(dk[k]), 8)] if dk.get(k) else
            [k, None] for k in range(2, int(kmax) + 1)]
        if best is None:
            info["reason"] = ("no measured order satisfies the bound at this tol, even "
                              "after propagating the x0 error into the step")
            return None, info
        info["k"] = best
        info["ck"] = round(_ck(best), 6)
        info["bound"] = round(_ck(best) * (h ** best) * float(dk[best]), 8)
        cap = _uniform_halfstep_wmax(best) * (1.0 + 1e-9)
        info["cap"] = round(cap, 6)
        return cap, info
    except Exception as e:
        info["reason"] = "exception: {}".format(e)
        return None, info


def _probe_pct(arr, p, derived=False):
    """Percentile for probe statistics, with the corrected estimator under the toggle.

    shipped : a[min(len(a)-1, int(p*len(a)))]  -- collides for n <= 2 and is biased at all n
    derived : linear interpolation at position (n-1)*p -- the standard estimator
    """
    if not arr:
        return 0.0
    a = sorted(arr)
    if not derived:
        return a[min(len(a) - 1, int(p * len(a)))]
    if len(a) == 1:
        return a[0]
    pos = (len(a) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (pos - lo) * (a[hi] - a[lo])


def _measure_dk_scale(x0s, dl, kmax=5, return_series=False):
    """Per-order derivative scale |d^k x0/du^k| / |x0|, measured from the probe trajectory.

    `return_series=True` also returns `{k: [per-step de-biased scale]}`.  The k-th
    difference at probe index i uses x0s[i-k..i], so the series entry i is anchored at
    the END of the stencil; `sigma_curve[k:]` is the matching sigma axis.  The scalar
    `out[k]` is the median of that series -- the same quantity as before.

    WHY THIS EXISTS: `_derived_wmax_cap` refuses without it, because substituting the k = 2
    curvature for every k was measured NOT to be a bound (the true error exceeded it by a
    consistent 2.6-4.8x -- the c^(k-2) factor).  This supplies the real thing.

    UNIFORM COEFFICIENTS ARE EXACT HERE: the probe grid is built log-uniform (`_probe_grid`),
    so the spacing in u = log sigma is constant by construction and the k-th difference with
    binomial coefficients is the right stencil.

    DE-BIASED, because a raw difference is NOT the derivative for a non-polynomial.  For
    x0 = e^{cu} the k-th difference over h gives exactly `e^{cu} * rho(ch)^k * c^k` with
    `rho(x) = (1 - e^{-x}) / x`, so the raw estimate is LOW by `rho(ch)^k`.  Measured on an
    analytic trajectory: raw/analytic = 0.75, 0.66, 0.58, 0.51 for k = 2..5 at ch = 0.255 --
    matching rho(ch)^k = 0.78, 0.69, 0.61, 0.54 -- and dividing by it recovers the analytic
    scale to a consistent 4.1 % at every order.  `ch` is measured, not assumed: c comes from
    the trajectory's own first-difference scale.  The 4.1 % residual is the estimator's
    accuracy and is reported rather than hidden behind a fudge factor.

    AN ORDER THAT CANNOT BE MEASURED IS OMITTED, not approximated: a k-th difference needs
    k + 1 consecutive samples.  On the 9-point engine grid k <= 5 is available; on a 3-point
    cond grid only k = 2 is, and the cap then legitimately falls back to the shipped map.

    MEDIAN OVER THE TRAJECTORY, NOT THE MAX.  The cap this feeds is ONE scalar applied at
    every step, so it has to describe the TYPICAL step, not the worst one.  Measured: the max
    is 19-27 for real runs, dominated by the tail where |x0| is small and relative curvature
    explodes, and it made the bound 70-100x larger than tol so NO order ever qualified and
    the arm was inert on every run.  The shipped map it replaces is also built from a MEDIAN
    (`med_curv`), so the median is the consistent statistic.
    """
    out = {}
    series = {}
    try:
        m = len(x0s)
        h = float(dl or 0.0)
        if m < 3 or not (h > 0.0):
            return (out, series) if return_series else out
        norms = [float(t.float().norm()) for t in x0s]

        def _diff(k, i):
            coeff = [((-1) ** j) * math.comb(k, j) for j in range(k + 1)]
            acc = None
            for j, c in enumerate(coeff):
                term = x0s[i - j].float() * float(c)
                acc = term if acc is None else acc + term
            return acc

        # the local exponential rate, from the measured first-difference scale
        d1 = 0.0
        for i in range(1, m):
            ref = max(norms[i - 1], norms[i])
            d1 = max(d1, float(_diff(1, i).norm()) / h / max(ref, 1e-8))
        ch = d1 * h
        rho = ((1.0 - math.exp(-ch)) / ch) if ch > 1e-9 else 1.0
        if not (rho > 0.0):
            rho = 1.0

        for k in range(2, int(kmax) + 1):
            if m < k + 1:
                break
            vals = []
            for i in range(k, m):
                d = _diff(k, i)
                if d is None:
                    continue
                ref = max(norms[i - k:i + 1]) if norms[i - k:i + 1] else 1.0
                vals.append(float(d.norm()) / (h ** k) / max(ref, 1e-8))
            if vals:
                deb = [v / (rho ** k) for v in vals]
                series[k] = [round(v, 8) for v in deb]
                s = sorted(deb)
                med = s[len(s) // 2] if len(s) % 2 else \
                    0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
                if med > 0.0:
                    out[k] = med
        return (out, series) if return_series else out
    except Exception:
        return (out, series) if return_series else out


def _calib_flags(profile):
    f = profile.get(_CALIB_FLAGS_KEY) if isinstance(profile, dict) else None
    if not isinstance(f, dict):
        return {}
    return f


def _calib_inputs(cfg):
    """The inputs that CHANGE a calibration, as one comparable record.

    These are stamped on the profile as `calib_flags` (the key it already had) and the
    calibration is re-derived from the profile's own MEASUREMENTS whenever the live cfg
    differs from the stamp.  The measurements -- `dk_scale`, `med_curv`, the gap fields --
    do not depend on any of these inputs, which is why re-deriving is exact and why a
    slider nudge must never cost a re-probe.  Without this the cached profile keeps the
    calibration it was built with: log_00159 reports `wmax_bound=False` in its cfg while
    its `calib_flags` say the arm was ON, i.e. the run did not do what the switch said.

    `pct_derived` IS DELIBERATELY ABSENT (U16/6906).  It is not a re-derivable input: it
    changes the percentile ESTIMATOR, so it changes what the profile's measurements ARE.
    Listing it here made `_recalibrated` re-derive the calibration from measurements taken
    with the other estimator and then stamp the live value, so the flag record no longer
    described its own numbers.  The estimator is a measurement input, and the probe key
    already carries it (`pd=` in `_schedule_sig` and the node key), so a toggle re-probes
    instead of relabelling.
    """
    try:
        budget = round(float(cfg.get("wmax_budget", 1.0) or 1.0), 6)
    except (TypeError, ValueError):
        budget = 1.0
    # THE LOOSENING MULTIPLIER BELONGS HERE TOO.  It changes the DERIVED `calib["rtol"]`, so a cached
    # profile must be re-derived when it changes -- the same rule-28 discipline as the budget above, and
    # the reason a knob nudge costs no re-probe.  Without it the operator's A/B would compare two runs
    # that both consumed the calibration the FIRST of them was built with, which is the "switch reads ON
    # while running OFF" defect this function exists to prevent.
    try:
        loosen = round(float(cfg.get("cond_rtol_loosen", 1.5) or 0.0), 6)
    except (TypeError, ValueError):
        loosen = 1.5
    return {"wmax_bound": bool(cfg.get("wmax_bound", False)),
            "wmax_budget": budget,
            "cond_rtol_loosen": max(0.0, min(_COND_RTOL_LOOSEN_MAX, loosen))}


def _derive_calibrations(profile):
    calib = {}
    med_curv = profile.get("med_curv", 0.0)
    # THREE states, and the middle one used to crash.  The key can be ABSENT (an older
    # profile) or present-but-None (the noise-survival probe ran and could not measure --
    # 125 of 398 corpus profiles carry None).  `dict.get(k, default)` covers only the
    # ABSENT case, so a None reached `math.sqrt(max(safe_f, 1e-4))` below and raised
    # `TypeError: '>' not supported between instances of 'float' and 'NoneType'` -- a latent
    # crash on a state the corpus demonstrably contains, found 2026-10-02 while re-deriving
    # these ranges (scratch_calib_ranges_rig.py).  Both unmeasured states take the documented
    # fallback, which is the same thing the skip path already substitutes when the probe is
    # gated off (aflops.py:4571).  A measured 0.0 is deliberately NOT mapped to the fallback:
    # that is a real reading meaning nothing survives, and it must keep its own meaning.
    safe_f = profile.get("safe_fresh_frac", 0.15)
    if safe_f is None:
        safe_f = 0.15
    med_err = float(profile.get("med_local_err", 0.0) or 0.0)
    gap_p = float(profile.get("log_gap_probe", 0.0) or 0.0)
    gap_r = float(profile.get("log_gap_real", 0.0) or 0.0)
    if med_err > 0:
        scale = 1.0
        if gap_p > 1e-9 and gap_r > 1e-9:
            scale = min(max((gap_r / gap_p) ** 2, 0.04), 1.0)
        calib["rtol"] = round(min(0.04, max(0.005, 3.0 * med_err * scale)), 4)
    # THE INPUTS, BESIDE THE VALUE THEY PRODUCED.  A calibration can only be reproduced -- or
    # compared against another run's -- if the reader can see what the probe was GIVEN, not just
    # what it produced.  Until now those inputs were spread across the profile's own fields with
    # nothing naming them as a set, so "did these two runs measure the same thing?" meant hunting
    # for the right fields in the right order.  Every profile field that feeds the derivation
    # above, or that identifies WHICH probe run produced it, is copied here as one dict.
    # NOTHING READS THIS: it is written for the log, so a corpus reader never has to guess.
    calib["probe_inputs"] = {k: profile.get(k) for k in (
        "n_probe_steps", "probe_cfg", "probe_patcher", "space", "version",
        "is_distilled_eff", "med_local_err", "med_local_err_real", "med_curv",
        "log_gap_probe", "log_gap_real", "guard_sig_med", "guard_sig_p90",
        "whitepoint", "dk_scale", "raw3", "scale", "safe_fresh_frac",
        "probe_rank", "probe_rank_tried", "probe_rank_ladder") if k in profile}
    sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
    sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
    if sig_med > 1e-9:
        r = sig_p90 / sig_med
        calib["guard_floor"] = round(
            min(0.5, max(0.05, 0.15 * math.sqrt(max(r, 0.25) / 4.0))), 4)
    if med_curv > 0:
        lc = math.log(max(med_curv, 1e-8) + 1.0)
        if lc < math.log(1.5):  # med_curv < 0.5
            wb = 3.0
        elif lc > math.log(5.0):  # med_curv > 4.0
            wb = 1.8
        else:
            t = (lc - math.log(1.5)) / (math.log(5.0) - math.log(1.5))
            wb = 3.0 - t * (3.0 - 1.8)
        _flags = _calib_flags(profile)
        _wb_shipped = round(max(1.5, min(2.5, wb)), 3)
        _wb_cap, _wb_info = _derived_wmax_cap(
            med_curv, profile.get("log_gap_probe"), calib.get("rtol"),
            dk=profile.get("dk_scale"),
            # the grid the cap is CONSUMED on, and the operator's nudge (see the helper)
            h_real=gap_r,
            budget=float(_flags.get("wmax_budget", 1.0) or 1.0))
        _use_derived_wb = bool(_flags.get("wmax_bound")) and _wb_cap is not None
        calib["wmax_base"] = (round(_wb_cap, 6) if _use_derived_wb else _wb_shipped)
        # BOTH values are logged whatever the toggle, so one run shows the whole delta and
        # the A/B needs no second run to be readable.
        calib["wmax_shipped"] = _wb_shipped
        calib["wmax_bound"] = (round(_wb_cap, 6) if _wb_cap is not None else None)
        calib["wmax_ab"] = _wb_info
        if _flags.get("wmax_bound"):
            if _wb_cap is not None:
                _CALIB_AB["wmax_derived"] += 1
            else:
                _CALIB_AB["wmax_bound_unavailable"] += 1
        else:
            _CALIB_AB["wmax_shipped"] += 1
    calib["safe_eta"] = round(min(1.0, max(0.1, math.sqrt(max(safe_f, 1e-4)))), 3)
    calib.update(_derive_anomaly_calibrations(profile))
    calib["calib_ab_counts"] = dict(_CALIB_AB)
    return calib

def _derive_anomaly_calibrations(profile):
    """Translate probe evidence into the anomaly-repair thresholds.

    The probe already measures everything the anomaly detector needs to know
    about the model's LEGITIMATE update statistics:

    - jump_q (per-step p10/p50/p90 of the normalized per-pixel jump; fallback
      guard_sig_med/p90) says how heavy-tailed the model's normal jump
      distribution is.  Heavy tails = large per-pixel jumps are structure,
      not anomalies, so the spatial/novelty z-thresholds must rise to avoid
      false flags; distributions tighter than a Gaussian reference can afford
      more sensitivity.
    - err_q (per-step p10/p50/p90 of the local error) tunes the value-error
      threshold the same way.
    - conv_step / n_jump say when the bulk of the trajectory work is done, so
      anomaly detection can arm relative to the measured convergence point
      instead of a fixed 15% of the run (0.15 default is reproduced exactly
      at conv_step = n_jump/3).
    - is_distilled_eff softens repair across the board (the promised
      behaviour of the probe_tune_anomaly toggle for distilled models).
    """
    calib = {}

    def _tail_ratio(quantile_curve):
        rs = []
        for q in (quantile_curve or []):
            try:
                p50, p90 = float(q[1]), float(q[2])
                if p50 > 1e-9:
                    rs.append(p90 / p50)
            except (TypeError, ValueError, IndexError):
                continue
        if not rs:
            return None
        return sorted(rs)[len(rs) // 2]

    spike = _tail_ratio(profile.get("jump_q"))
    if spike is None:
        sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
        sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
        if sig_med > 1e-9:
            spike = sig_p90 / sig_med
    if spike is not None and spike > 0:
        # Gaussian reference: p90/p50 = 1.815.  Thresholds sit at 1.0x for a
        # reference-shaped distribution, rise up to +60% for very heavy
        # tails, drop to -25% for unusually tight ones.
        mult = 1.0 + 0.6 * (spike - 1.815) / 1.815
        mult = min(max(mult, 0.75), 1.60)
        if bool(profile.get("is_distilled_eff", False)):
            mult = min(mult * 1.2, 1.9)  # soften repair for distilled models
        calib["an_spatial"] = round(5.0 * mult, 3)
        calib["an_novelty"] = round(4.0 * mult, 3)

    err_spike = _tail_ratio(profile.get("err_q"))
    if err_spike is not None and err_spike > 0:
        vmult = 1.0 + 0.5 * (err_spike - 1.815) / 1.815
        vmult = min(max(vmult, 0.80), 1.50)
        if bool(profile.get("is_distilled_eff", False)):
            vmult = min(vmult * 1.1, 1.6)
        calib["an_value"] = round(2.5 * vmult, 3)

    try:
        conv_step = int(profile.get("conv_step", -1))
        n_jump = int(profile.get("n_jump", 0))
    except (TypeError, ValueError):
        conv_step, n_jump = -1, 0
    if 0 <= conv_step < n_jump:
        frac = conv_step / float(n_jump)
        calib["an_from"] = round(min(max(0.05 + 0.30 * frac, 0.05), 0.35), 3)
    return calib

def _probe_gen(x):
    gen = torch.Generator(device=x.device)
    gen.manual_seed(20260915)
    return gen

def _lf_renoise(x0, s, space, gen):
    eps = torch.randn(x0.shape, device=x0.device, dtype=torch.float32,
                      generator=gen).to(x0.dtype)
    if space == "flow":
        sf = min(max(float(s), 0.0), 0.999)
        return (1.0 - sf) * x0 + sf * eps
    return x0 + float(s) * eps

def _probe_endgame(model, extra_args, x0_final, grid_last, pos_all, ones,
                   space, gen):
    try:
        cands = [float(v) for v in pos_all if 1e-6 < float(v) < float(grid_last) * 0.9]
        if len(cands) >= 2:
            s_a = cands[max(0, int(round(len(cands) * 0.45)))]
            s_b = cands[min(len(cands) - 1, int(round(len(cands) * 0.85)))]
        else:
            s_a = float(grid_last) * 0.30
            s_b = float(grid_last) * 0.10
        if not (s_b < s_a):
            s_b = s_a * 0.4
        ref_rms = float(x0_final.float().pow(2).mean().sqrt().clamp_min(1e-8))

        def _qs(map_t):
            flat = map_t.reshape(-1).float()
            q = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9],
                                                  device=flat.device))
            return [round(float(v), 6) for v in q]

        levels = []
        for s_t in (s_a, s_b):
            x_t = _lf_renoise(x0_final, s_t, space, gen)
            x0_a = _probe_model_call(model, x_t,
                                     x_t.new_full([x_t.shape[0]], s_t),
                                     extra_args)
            s2_t = s_t * 0.5
            x1_t = exact_step(x_t, x0_a, s_t, s2_t)
            x0_b = _probe_model_call(model, x1_t,
                                     x1_t.new_full([x1_t.shape[0]], s2_t),
                                     extra_args)
            a_rms = float(x0_a.float().pow(2).mean().sqrt().clamp_min(1e-8))
            rec_a = (x0_a.float() - x0_final.float()).pow(2).mean(dim=1).sqrt() / ref_rms
            rec_b = (x0_b.float() - x0_final.float()).pow(2).mean(dim=1).sqrt() / ref_rms
            jmp = (x0_b.float() - x0_a.float()).pow(2).mean(dim=1).sqrt() / a_rms
            levels.append({"s": round(float(s_t), 6),
                           "s2": round(float(s2_t), 6),
                           "rec_q": _qs(rec_a), "rec2_q": _qs(rec_b),
                           "jmp_q": _qs(jmp)})
        return {"levels": levels, "n": int(rec_a.numel())}
    except Exception as e:
        logging.warning("[PD-AOS-probe] endgame measurement skipped: %s", e)
        return None


def _probe_distilled_mode(cfg):
    """Few-step (distilled) detection for probe-grid shaping, available before
    the probe has produced its own is_distilled_eff: guidance-free operation
    (live cfg <= 1) or an explicit distilled model hint."""
    try:
        pc = cfg.get("_guider_live_cfg")
        if pc is not None and float(pc) <= 1.0 + 1e-6:
            return True
    except (TypeError, ValueError):
        pass
    try:
        mi = cfg.get("_model_info")
        if isinstance(mi, dict) and mi.get("is_distilled"):
            return True
    except Exception:
        pass
    return False


_PROBE_SMIN_FLOOR = 0.10    # flow: floor the probe grid at 10% of sigma_max


def _probe_reach_eff(cfg):
    """THE EFFECTIVE PROBE REACH, in one place, for the grid AND for every cache key.

    WHY ONE PLACE.  The reach was first added to the MODEL probe's key only, and the cond key was
    missed -- so on the operator's own reach sweep (anima-base-v1.0 logs 00059-00067, build
    e410d2c) every cond profile stayed stamped `reach_frac=0.1` while the requested reach went to
    0.002, and `cond_probe_state` read `cached` on seven of the eight runs after the first.  The
    cond profile is the Autotuner's PREFERRED source for the schedule and the lambda clamp's FIRST
    choice, so those consumers never saw the deeper measurement at all.  This is the same defect
    `_cond_probe_dims`'s own docstring records for `cond_probe_steps`/`cond_probe_resolution`,
    which is why the rule it states is repeated here rather than trusted to memory.

    Requests at or above the shipped floor are COLLAPSED to it, so a run asking for 0.1 and a run
    asking for 0.5 share one measurement (they measure the same grid) instead of fragmenting the
    cache; below the floor the value stands, because it changes the grid.
    """
    try:
        f = float(cfg.get("probe_smin_frac", _PROBE_SMIN_FLOOR))
    except (TypeError, ValueError):
        return _PROBE_SMIN_FLOOR
    if not (f > 0.0):
        return _PROBE_SMIN_FLOOR
    return f if f < _PROBE_SMIN_FLOOR else _PROBE_SMIN_FLOOR


def _probe_reach_extend(grid, sigmas, cfg):
    """Append probe points BELOW the shipped band, when `probe_smin_frac` asks for it.

    WHY THIS EXISTS.  `_probe_grid` stops at `max(smallest positive sigma of the schedule,
    _PROBE_SMIN_FLOOR * sigma_max)`.  Measured on 104 run-paired profiles (rig_schedule_derivation.py,
    section F): that floor is the schedule's own tail in 56 runs and the 0.1 constant in 48, and it
    is a measurement of where the trajectory settles in NONE of them -- `conv_sigma` equals the
    grid's own floor in 70 (67%).  So the low-sigma curvature was never measured, and "is this
    schedule's tail right?" could not be asked of the probe: the instrument's reach was defined by
    the number under test.

    WHAT IT DOES.  Appends points below the existing band at the BAND'S OWN lowest log spacing,
    down to `probe_smin_frac * sigma_max`.  Three properties make it safe to add:
      * the walk in `_run_probe` is DOWNWARD, so the model calls on the shipped points, and their
        measurements, are unchanged;
      * the result has the shipped grid as a strict PREFIX, which is what lets every scalar
        aggregate stay on the band it was calibrated on (`m - n_extra`);
      * the shipped default appends NOTHING, so the shipped configuration is bit-identical.
    Flow schedules only, matching `_probe_grid`'s own condition: on VE schedules that floor is the
    model's natural sigma_min, which its docstring says does not suffer the collapse.

    Returns `(grid, n_extra)`.  `n_extra` is 0 whenever the shipped reach is in force.
    """
    try:
        if not grid or len(grid) < 3:
            return grid, 0
        pos = [v for v in _sigmas_as_floats(sigmas) if v > 1e-6]
        if len(pos) < 2:
            return grid, 0
        smax = max(pos)
        if smax > 1.5:
            return grid, 0
        frac = _probe_reach_eff(cfg)
        if frac >= _PROBE_SMIN_FLOOR - 1e-12:
            return grid, 0
        lo_now = min(float(v) for v in grid)
        lo_new = smax * frac
        if lo_new >= lo_now * (1.0 - 1e-9) or lo_now <= 0.0:
            return grid, 0
        step = math.log(max(grid[-2], 1e-9)) - math.log(max(grid[-1], 1e-9))
        if step <= 1e-9:
            return grid, 0
        extra = []
        v = lo_now
        while v > lo_new * (1.0 + 1e-12):
            v = max(v * math.exp(-step), lo_new)
            extra.append(round(float(v), 8))
        if not extra:
            return grid, 0
        out = [round(float(v), 8) for v in grid] + extra
        if len(out) != len(grid) + len(extra) or any(
                out[i] <= out[i + 1] for i in range(len(out) - 1)):
            return grid, 0
        return out, len(extra)
    except Exception:
        return grid, 0


def _probe_calib_band(grid, sigmas, cfg):
    """`(n_cal, band_floor)` for a grid that may already carry a reach extension.

    The shipped band's floor is `max(smallest positive sigma of the schedule, _PROBE_SMIN_FLOOR *
    sigma_max)` -- the rule `_probe_grid` applies.  `_probe_reach_extend` only ever APPENDS below
    that floor, so the calibration band is a PREFIX of the result and its length is `n_cal`.  The
    scalar aggregates are computed on `[:n_cal]` for the reason given in `_run_probe`: a verdict
    that moved because the probe looked further would make any A/B uninterpretable.

    Needed where the grid ARRIVES from a caller rather than being built in place -- the focused cond
    probe is handed `focused_grid`, so it cannot use the `n_extra` the builder returned.  Returns
    `(len(grid), grid's own floor)` when no extension is possible, i.e. the shipped path.

    SAME RULE, ONE PLACE: `rig_probe_reach.py` C10 asserts this agrees with the `n_extra` that
    `_probe_reach_extend` returns, so the two cannot drift into disagreeing about the band.
    """
    try:
        if not grid:
            return 0, 0.0
        pos = [v for v in _sigmas_as_floats(sigmas) if v > 1e-6]
        if len(pos) < 2 or max(pos) > 1.5:
            return len(grid), min(float(v) for v in grid)
        floor = max(min(pos), max(pos) * _PROBE_SMIN_FLOOR)
        n = sum(1 for v in grid if float(v) >= floor * (1.0 - 1e-9))
        n = max(1, min(n, len(grid)))
        return n, floor
    except Exception:
        return len(grid or []), (min(grid) if grid else 0.0)


def _probe_grid(sigmas, K, distilled_mode=False):
    """Build the probe trajectory's sigma grid, GEOMETRIC (log-uniform) in
    sigma.  An index-uniform subsample of a high-shift schedule collapses most
    points into a tiny high-sigma band, producing degenerate (~0) jumps and
    curvature that fool the is_distilled_eff heuristic (a cfg>1 model with a
    bunched schedule reads as "converged in 2 steps").  Log-uniform spacing
    keeps the probe points meaningfully spread regardless of the shift.  For
    few-step (distilled) models the top half of log-sigma gets half the points,
    since that is where all the dynamics happen.

    The low bound is FLOORED for flow schedules: the schedule's smallest
    positive sigma is shift-dependent and, on a high-shift flow schedule, can
    collapse toward ~0 (the logged run gave min(g)=0.0023).  Anchoring the
    geometric grid at that min(g) spans a huge log-range, the jump curve
    decays abruptly, and is_distilled_eff falsely flags a cfg>1 model as
    distilled.  Clamping at 10% of sigma_max keeps the probe inside the
    model's active sigma regime and makes the measurement comparable across
    schedules."""
    g = [float(s) for s in sigmas if float(s) > 1e-6]
    if len(g) < 3:
        return None
    if len(g) <= K:
        return sorted(g, reverse=True)
    smax = max(g)
    smin = min(g)
    if smax <= 1.5:
        # flow: floor the low bound (VE schedules have a natural, sane min(g)
        # and do not suffer the high-shift collapse).
        smin = max(smin, smax * _PROBE_SMIN_FLOOR)
    lmax = math.log(max(smax, 1e-8))
    lmin = math.log(max(smin, 1e-8))
    if distilled_mode:
        n_hi = max(2, K // 2)
        lsplit = 0.5 * (lmax + lmin)
        hi = [math.exp(lmax + (lsplit - lmax) * (i / max(n_hi - 1, 1)))
              for i in range(n_hi)]
        lo = [math.exp(lsplit + (lmin - lsplit) * (i / max(K - n_hi - 1, 1)))
              for i in range(K - n_hi)]
        grid = hi + lo
    else:
        grid = [math.exp(lmax + (lmin - lmax) * (i / (K - 1))) for i in range(K)]
    grid = sorted(set(round(float(v), 8) for v in grid), reverse=True)
    return grid if len(grid) >= 3 else None


def _probe_grid_sig(sigmas, cfg):
    """A signature of the points the probe will WALK -- which is what it measures.

    A model probe is keyed on the schedule because its grid comes from the schedule.
    But `_probe_grid` derives that grid from the schedule's RANGE (max/min positive
    sigma), `probe_steps` and the distilled mode -- never from the interior placement.
    Hashing the raw sigma list therefore made every probe-derived placement a new cache
    slot: the Scheduler places steps FROM a probe (`mode: curvature`), the placement
    changes the sigmas, the sigmas were part of the probe key, so the engine re-probed on
    every prompt.  MEASURED on `moss_image_base_int8_convrot` (2026-10-18): 50-step runs
    with different placements but the same endpoints, each keyed differently, 12.6-17.3 s
    of `model_probe_ms` per generation.

    A SHORT schedule is different: when the schedule has no more than K positive sigmas,
    `_probe_grid` returns THOSE sigmas, so the grid signature is the schedule and the key
    still changes with placement -- correct, because the probe really does walk them.
    """
    try:
        K = max(4, int(cfg.get("probe_steps", 10)))
        g = _probe_grid(sigmas, K, _probe_distilled_mode(cfg))
    except Exception:
        g = None
    if not g:
        return "s:" + _sig_list_sig(sigmas)
    return "g:" + ",".join("%.6g" % float(v) for v in g)


def _fragility_map(frag_acc, size=8):
    """Downsample the accumulated per-pixel max local error into a coarse
    [0,1] fragility map (which regions the linear velocity model predicts
    least well -- the model's "weak spots").  Feeds the local field's spatial
    prior so it can steer volatility/dwell toward exactly those regions."""
    if frag_acc is None:
        return None
    try:
        fm = frag_acc.float().reshape(-1, 1, *frag_acc.shape[-2:])
        fm = F.adaptive_avg_pool2d(fm, (int(size), int(size)))
        fm = fm.mean(dim=0, keepdim=True)
        mx = float(fm.max().clamp_min(1e-8))
        fm = (fm / mx).clamp(0.0, 1.0)
        return {"h": int(size), "w": int(size),
                "v": [round(float(v), 4) for v in fm.reshape(-1)]}
    except Exception:
        return None


def _run_probe(model, x, sigmas, extra_args, cfg, callback=None):
    try:
        pres = max(16, int(cfg.get("probe_resolution", 48)))
        K = max(4, int(cfg.get("probe_steps", 10)))
        B = 1
        C = x.shape[1]
        if B < 1 or C < 1:
            return None
        shape = _probe_latent_shape(ref_x=x, pres=pres, batch=B,
                                    site="engine/_run_probe")
        px = torch.randn(shape, device=x.device, dtype=torch.float32,
                         generator=_probe_gen(x)).to(x.dtype)
        ones = px.new_ones([B])
        grid = _probe_grid(sigmas, K, _probe_distilled_mode(cfg))
        if grid is None:
            return None
        # PROBE REACH: `probe_smin_frac` may ask for points BELOW the shipped band.  The extension
        # is appended, so `grid[:m - _reach_extra]` is the shipped band verbatim -- and that prefix
        # is what every SCALAR aggregate below is computed on, deliberately: a verdict or a
        # threshold that moved because the probe looked further would make any A/B uninterpretable
        # (rule 25: one thing at a time).  The new information rides in the CURVES.
        _band_floor = min(float(v) for v in grid)
        grid, _reach_extra = _probe_reach_extend(grid, sigmas, cfg)
        px = px * float(grid[0])
        pos_all = [v for v in _sigmas_as_floats(sigmas) if v > 1e-6]
        gap_curve = [abs(math.log(max(grid[i], 1e-8))
                         - math.log(max(grid[i + 1], 1e-8)))
                     for i in range(len(grid) - 1)]
        real_gap_curve = []
        if len(pos_all) >= 2:
            pos_idx = []
            _j = 0
            for gv in grid:
                while _j < len(pos_all) and pos_all[_j] > gv * (1.0 + 1e-9):
                    _j += 1
                pos_idx.append(min(_j, len(pos_all) - 1))
            real_gaps = [abs(math.log(max(pos_all[t], 1e-8))
                             - math.log(max(pos_all[t + 1], 1e-8)))
                         for t in range(len(pos_all) - 1)]
            for i in range(len(grid) - 1):
                a, b = pos_idx[i], pos_idx[i + 1]
                if b <= a:
                    lo, hi = max(a - 1, 0), min(a + 1, len(pos_all) - 1)
                else:
                    lo, hi = a, b
                seg = real_gaps[lo:hi]
                real_gap_curve.append(sum(seg) / len(seg) if seg
                                      else (sum(real_gaps) / len(real_gaps)))
        else:
            real_gap_curve = list(gap_curve)
        space = _detect_space(torch.tensor(grid), cfg.get("noise_space", "auto"))
        xs = []
        x0s = []
        cur = px
        _dev_retry_done = False
        for i, s in enumerate(grid):
            _progress(callback, "probe", "progress", i, len(grid) + 4,
                      "PD-AOS model probe: step %d/%d" % (i + 1, len(grid)))
            x0 = None
            try:
                x0 = _probe_model_call(model, cur, s * ones, extra_args)
            except Exception as e:
                _pp = None if _dev_retry_done else _navigate_to_patcher(model)
                if (not _dev_retry_done and _pp is not None
                        and _is_device_error(e)):
                    _dev_retry_done = True
                    try:
                        _dev = _probe_call_device(_pp)
                        if cur.device != _dev:
                            cur = cur.to(_dev)
                        if ones.device != _dev:
                            ones = ones.to(_dev)
                        x0 = _probe_model_call(model, cur, s * ones, extra_args)
                        logging.info("[PD-AOS-probe] recovered from a "
                                     "device mismatch at s=%.4g (model "
                                     "re-secured on %s)", s, _dev)
                    except Exception:
                        x0 = None
                if x0 is None:
                    import traceback as _tb
                    logging.warning("[PD-AOS-probe] model call failed at s=%.4g: %s", s, e)
                    logging.warning("[PD-AOS-probe] failure traceback:\n%s", _tb.format_exc())
                    logging.warning("[PD-AOS-probe] failure context: %s",
                                    _probe_failure_diagnostics(model, cur, ones, extra_args, None))
                    break
            xs.append(cur)
            x0s.append(x0.detach())
            if i < len(grid) - 1:
                cur = exact_step(cur, x0, s, grid[i + 1])
        m = len(x0s)
        if m < 3:
            return None
        norms = [float(t.float().norm().clamp_min(1e-8)) for t in x0s]
        jumps = []
        for i in range(m - 1):
            d = float((x0s[i + 1] - x0s[i]).float().norm())
            jumps.append(d / max(max(norms[i], norms[i + 1]), 1e-8))
        dist_on = bool(cfg.get("probe_distributions", True))
        jump_q = []
        ojf_curve = []
        band_curve = []
        jrel_curve = []
        dabs_curve = []
        if dist_on:
            _qt = torch.tensor([0.1, 0.5, 0.9])
            for i in range(m - 1):
                jpx = (x0s[i + 1] - x0s[i]).float().pow(2).mean(dim=1).sqrt()
                sc = max(float(x0s[i].float().pow(2).mean().sqrt()),
                         float(x0s[i + 1].float().pow(2).mean().sqrt()))
                jpx_n = (jpx / max(sc, 1e-8)).reshape(-1)
                qs = torch.quantile(jpx_n, _qt.to(jpx_n.device))
                jump_q.append([round(float(v), 6) for v in qs])
                p50 = float(qs[1])
                if p50 > 1e-9:
                    ojf_curve.append(round(
                        float((jpx_n > 2.5 * p50).float().mean()), 4))
                try:
                    band_curve.append([round(v, 4) for v in
                                       _lf_band_fractions(jpx)])
                except Exception:
                    band_curve.append([0.33, 0.34, 0.33])
                # runtime-scale per-pixel relative jump: the SAME quantity the
                # escape's stagnation detector uses (median jump / median
                # per-pixel magnitude).  Stored so the escape threshold can be
                # derived from the model's own measured jump level.
                xm = x0s[i].float().pow(2).mean(dim=1).sqrt()
                jrel_curve.append(round(
                    float(jpx.median().clamp_min(0.0)
                          / xm.median().clamp_min(1e-8)), 6))
            for i in range(m):
                x4 = x0s[i].float().reshape(-1, x0s[i].shape[1],
                                            *x0s[i].shape[-2:])
                hp = x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)
                dabs_curve.append(round(
                    float(hp.norm() / x4.norm().clamp_min(1e-8)), 6))
        med_jump_rel = (sorted(jrel_curve)[len(jrel_curve) // 2]
                        if jrel_curve else None)
        med_detail_abs = (sorted(dabs_curve)[len(dabs_curve) // 2]
                          if dabs_curve else None)
        dir_cons = []
        if bool(cfg.get("probe_dir_cons", True)):
            for i in range(1, m - 1):
                d1 = (x0s[i] - x0s[i - 1]).float()
                d2 = (x0s[i + 1] - x0s[i]).float()
                num = (d1 * d2).sum(dim=1)
                den = d1.pow(2).sum(dim=1).sqrt() * d2.pow(2).sum(dim=1).sqrt()
                okm = den > 1e-8
                if bool(okm.any()):
                    dir_cons.append(round(float((num[okm] / den[okm]).mean()), 4))
        dir_cons_valid = []
        if dir_cons and len(gap_curve) >= 2:
            _med_g = sorted(gap_curve)[len(gap_curve) // 2]
            for _k in range(len(dir_cons)):
                _ga, _gb = gap_curve[_k], gap_curve[_k + 1]
                if (_med_g / 3.0) <= _ga <= 3.0 * _med_g and \
                        (_med_g / 3.0) <= _gb <= 3.0 * _med_g:
                    dir_cons_valid.append(dir_cons[_k])
            if len(dir_cons_valid) < 2:
                dir_cons_valid = []
        # oscillation onset: the sigma where the x0 trajectory first reverses
        # direction (dir_cons < 0) -- where the late noise fade should anchor
        # (the noise stops helping and starts oscillating the trajectory).
        osc_sigma = None
        for _k, _dc in enumerate(dir_cons):
            if _dc < 0 and _k + 1 < len(grid):
                osc_sigma = float(grid[_k + 1])
                break
        curvs = []
        for i in range(1, m - 1):
            dl_prev = math.log(max(grid[i - 1], 1e-8)) - math.log(max(grid[i], 1e-8))
            dl_next = math.log(max(grid[i], 1e-8)) - math.log(max(grid[i + 1], 1e-8))
            dl = 0.5 * (abs(dl_prev) + abs(dl_next))
            if dl < 1e-6:
                continue
            dd = (x0s[i + 1] - 2.0 * x0s[i] + x0s[i - 1]).float().norm()
            curvs.append(float(dd) / (dl * dl)
                         / max(max(norms[i - 1], norms[i], norms[i + 1]), 1e-8))
        local_errs = []
        err_q = []
        guard_sigs = []
        frag_acc = None
        for i in range(1, m - 1):
            s_i = float(grid[i])
            s_n = float(grid[i + 1])
            s_p = float(grid[i - 1])
            if s_n <= 1e-8 or s_i <= 1e-8:
                continue
            x1_i = exact_step(xs[i], x0s[i], s_i, s_n)
            ls = math.log(max(s_i, 1e-8))
            ls2 = math.log(max(s_n, 1e-8))
            lsp = math.log(max(s_p, 1e-8))
            lmid = 0.5 * (ls + ls2)
            t = (lmid - ls) / (lsp - ls)
            x0_mid = x0s[i] + t * (x0s[i - 1] - x0s[i])
            x_pred_i = exact_step(xs[i], x0_mid, s_i, s_n)
            local_errs.append(float((x_pred_i - x1_i).float().norm()
                                    / x1_i.float().norm().clamp_min(1e-8)))
            if dist_on:
                epx = ((x_pred_i - x1_i).float().pow(2).mean(dim=1).sqrt()
                       / float(x1_i.float().pow(2).mean().sqrt().clamp_min(1e-8)))
                eq = torch.quantile(epx.reshape(-1),
                                    torch.tensor([0.1, 0.5, 0.9],
                                                 device=epx.device))
                err_q.append([round(float(v), 6) for v in eq])
                # per-pixel fragility: max local error over the trajectory
                if frag_acc is None:
                    frag_acc = epx.detach().float().unsqueeze(1)
                else:
                    frag_acc = torch.maximum(frag_acc, epx.detach().float().unsqueeze(1))
        local_errs_real = []
        _err_probe_gaps = gap_curve[1:1 + len(local_errs)]
        _err_real_gaps = real_gap_curve[1:1 + len(local_errs)]
        for _e, _gp, _gr in zip(local_errs, _err_probe_gaps, _err_real_gaps):
            if _gp > 1e-9 and _gr > 1e-9:
                local_errs_real.append(
                    _e * min(max((_gr / _gp) ** 2, 0.04), 1.0))
            else:
                local_errs_real.append(_e)
        for i in range(1, m):
            guard_sigs.append(float(_jump_signal(x0s[i], x0s[i - 1], xs[i])))
        g_pos = [max(float(v), 1e-8) for v in grid]
        gap_probe = (sum(abs(math.log(g_pos[i]) - math.log(g_pos[i + 1]))
                         for i in range(len(g_pos) - 1))
                     / max(1, len(g_pos) - 1))
        pos_all = [v for v in _sigmas_as_floats(sigmas) if v > 1e-6]
        gap_real = 0.0
        if len(pos_all) > 1:
            gap_real = (sum(abs(math.log(max(pos_all[i], 1e-8))
                                - math.log(max(pos_all[i + 1], 1e-8)))
                            for i in range(len(pos_all) - 1))
                        / (len(pos_all) - 1))

        def _pct(arr, p):
            # The shipped body was `a[min(len(a)-1, int(p*len(a)))]`, which COLLIDES for
            # n <= 2 (int(0.5*2) == int(0.9*2) == 1, so p90 IS the median -- this is what
            # made guard_floor a constant 0.075) and is biased at every n (at n = 10 it
            # returns the maximum for p = 0.9).  Toggle: cfg["pct_derived"].  Counted for
            # reachability: a toggle wired to nothing looks exactly like one whose effect
            # is below the noise floor.
            _pd = bool(cfg.get("pct_derived", False)) if isinstance(cfg, dict) else False
            _CALIB_AB["pct_derived" if _pd else "pct_shipped"] += 1
            return _probe_pct(arr, p, derived=_pd)

        # THE CALIBRATION BAND, not the extended one.  `_reach_extra` points were appended below the
        # shipped floor, so the shipped band is the prefix of length `len(jumps) - _reach_extra`.
        # The distilled verdict is the sharpest example of why this matters: `early_frac` divides by
        # the TOTAL jump and widens its window with `len(jumps)//3`, so simply making the grid longer
        # inflates it and turns a cfg > 1 flow model into a false distilled positive -- the exact
        # failure `_PROBE_SMIN_FLOOR` was added for.  With `_reach_extra == 0` these are the shipped
        # expressions on the shipped arrays, unchanged.
        if _reach_extra > 0:
            _jcal = jumps[:max(0, len(jumps) - _reach_extra)] or jumps
            _ccal = curvs[:max(0, len(curvs) - _reach_extra)] or curvs
        else:
            _jcal, _ccal = jumps, curvs
        med_jump = _pct(_jcal, 0.5)
        med_curv = _pct(_ccal, 0.5)
        total_jump = sum(_jcal) + 1e-12
        early = _jcal[: max(1, len(_jcal) // 3)]
        early_frac = sum(early) / total_jump
        conv_step = len(_jcal)
        thr = 0.05 * (max(_jcal) if _jcal else 1.0)
        for i, jv in enumerate(_jcal):
            if jv < thr:
                conv_step = i
                break
        # Distilled = the x0 trajectory stops moving within the first few
        # steps AND most of its movement happens in that front window.  We do
        # NOT flag "last jump tiny relative to the max" as distilled: any
        # smooth fast-converging model (jump ~ sigma^p with p > 1) has a
        # small final jump once the grid reaches low sigma, so that clause
        # turned every cfg > 1 flow model into a false distilled positive.
        # The cfg override in _resolve_distilled_override already catches
        # true cfg <= 1 distilled models.
        # THE VERDICT'S OWN LENGTH MUST BE THE CALIBRATION BAND'S, not the full grid's: this
        # bound widens with the grid, so leaving len(jumps) here re-opened the false-distilled
        # path that pinning early_frac/conv_step alone had closed.  Found by a source census
        # (rig_probe_reach.py C11) written because the first version of this protection missed it.
        is_distilled_eff = bool(early_frac > 0.75 and
                                conv_step <= max(2, len(_jcal) // 3))
        is_distilled_eff = _resolve_distilled_override(
            cfg.get("probe_is_distilled", "auto"),
            cfg.get("_guider_live_cfg"),
            is_distilled_eff)
        # _PROBE_NOISE_SURVIVAL IS GATED ON ACTUAL NOISE, NOT DELETED.
        # Its answer reaches one consumer: `safe_fresh_frac` -> `calib["safe_eta"]` ->
        # the eta path in `_apply_probe_profile`.  With `eta` at 0 there is no noise
        # to budget, so the measurement is skipped; `probe_tune_eta` (default True)
        # does not by itself justify the model calls.  A positive `eta`, or an
        # explicit `probe_noise_survival`, turns them back on.
        #
        # KEPT FLEXIBLE ON PURPOSE.  `probe_noise_survival = False` forces the skip
        # for an A/B; `True` forces the measurement for an inspection run.
        _do_noise_survival = _noise_survival_wanted(cfg)
        # MEASUREMENT-ONLY switch: keeps the per-(sigma, amplitude) divergence
        # series instead of reducing it to the scalar `safe_fresh_frac`.  It
        # changes NOTHING the engine does -- the returned scalar is identical
        # either way -- and is deliberately NOT in the probe key, because it
        # changes what is RECORDED, not what is measured.  Shipping it costs one
        # boolean and buys the instrument any derived noise scale would have to
        # be built on.
        # `eta_from_probe` and `eta_budget_from_probe` CONSUME this curve; they
        # must record it even when the separate measurement widget is off, or the
        # feature silently has no input.
        _ns_curve = (bool(cfg.get("probe_noise_survival_curve", False))
                     or bool(cfg.get("eta_from_probe", False))
                     or bool(cfg.get("eta_budget_from_probe", False)))
        _progress(callback, "probe", "progress", len(grid), len(grid) + 4,
                  "PD-AOS model probe: noise survival" if _do_noise_survival else
                  "PD-AOS model probe: noise survival skipped (no eta consumer)")
        if _do_noise_survival:
            # The function returns a float when `curve=False` and a pair when it is
            # true.  The caller must accept both: unpacking unconditionally turned a
            # skipped series into a TypeError that the outer handler reported as a
            # failed whole probe.
            _ns_out = _probe_noise_survival(
                model, extra_args, xs, x0s, grid, ones,
                gen=_probe_gen(x), curve=_ns_curve, sigmas=sigmas)
            safe_f, _ns_series = _as_survival_pair(_ns_out)
            _ab_hit("probe.noise_survival_ran")
        else:
            # NOT MEASURED.  Keep it `None` so `_probe_budget_from_profile` cannot
            # report the skip fallback as a probe measurement; `_derive_calibrations`
            # substitutes 0.15 for its own `safe_eta` calculation.
            safe_f = None
            _ns_series = []
            _ab_hit("probe.noise_survival_skipped")
        # ---- THE DIAGNOSTIC STAMP (temporary, added to explain a regression) ----
        # The widened-ladder build recorded a POPULATED curve on 0 of 28 profiles
        # while the build before it recorded 19, and the function itself was shown
        # to work on both builds' real grids.  So the cause is on the path between
        # this call and the profile, and the two candidates -- the `curve` flag
        # arriving False, versus a populated series being discarded downstream --
        # are indistinguishable from the outside.
        #
        # Written BOTH onto the profile AND onto the cfg, because the whole
        # question is whether what this call returned is what the log ends up
        # carrying: if the profile disagrees with the cfg stamp, the loss is
        # downstream, and if they agree but both are empty, the flag was False.
        # Removed once the cause is known.
        _ns_diag = {
            "curve_flag_seen": bool(_ns_curve),
            "do_noise_survival": bool(_do_noise_survival),
            "ran": bool(_do_noise_survival),
            "series_len": len(_ns_series or []),
            "safe_fresh_frac": safe_f,
            "grid_pts": len(grid),
            "sigmas_pts": (len(sigmas) if sigmas is not None else None),
            "xs_pts": len(xs),
            "x0s_pts": len(x0s),
            # WHY it produced nothing, when it produced nothing.
            "error": _NOISE_SURVIVAL_LAST_ERROR[0],
        }
        cfg["_ns_diag"] = dict(_ns_diag)
        whitepoint = _whitepoint_from_x0s(x0s)
        endgame = None
        if bool(cfg.get("probe_endgame", True)):
            _progress(callback, "probe", "progress", len(grid) + 1, len(grid) + 4,
                      "PD-AOS model probe: endgame")
            pos_all_eg = [float(v) for v in sigmas if float(v) > 1e-6]
            endgame = _probe_endgame(model, extra_args, x0s[-1], grid[-1],
                                     pos_all_eg, ones, space, _probe_gen(x))
        # Computed ONCE here so the profile dict below can carry BOTH the curve
        # and its per-term breakdown without walking the trajectory twice.  The
        # breakdown matters: measured on the real corpus the probe curve lands
        # ~10x BELOW the run's (probe p50 0.0000 vs a run p50 of 0.27-0.41), and
        # the gate is a 4-way AND -- without the per-conjunct pass rates that
        # failure cannot be attributed to any one term.
        # THE RUN'S RESCUE INPUTS, NOT A PLACEHOLDER.  The run's floor is
        # `rescue.floor * sigma` per step and its gate is fed only from
        # `rescue_start`; both are passed so the probe predicts the gate the run
        # actually evaluates (D3, U08).
        _cov_floor = float(cfg.get("rescue_floor", 0.02))
        _cov_start = float(cfg.get("rescue_start", 0.55))
        _cov_r = _lf_rescue_cov_curve(
            xs, x0s, grid, _eg_n_from_err(_endgame_eg_err(endgame)),
            _cov_floor, _cov_start)
        profile = {
            "version": 7,
            # ---- WHAT PRODUCED THIS, AND AT WHAT GEOMETRY (added 2026-10-08) ----
            # The fields the adoption gate (`_profile_serves_model_probe`) reads, stamped by
            # the producer rather than inferred from optional ones: a cond profile carries
            # neither `gap` nor `focused` in every case, and the model slot is not
            # cond-free -- `_PROFILE_BY_MODEL` is written by the node-side COND probe and
            # `aflops_engine` promotes whatever it adopts into its own `|s:` slot.  That is
            # how a 16-step/48-px model measurement was replaced, in the same slot, by an
            # 8-step/16-px per-prompt one.  `n_probe_res` is new on THIS path (the cond
            # probes already stamped it), which is why the two producers were
            # indistinguishable in every log: the engine's profile read `n_probe_res: null`.
            # `probe_device` is what the probe actually measured ON; the runtime's residency
            # decisions (`_probe_call_device` / `_ensure_model_gpu`) can put it on a
            # different device from the run's, and nothing recorded that.
            "probe_src": "model",
            "n_probe_res": int(pres),
            "probe_device": str(getattr(x, "device", "?")),
            "space": space,
            "n_probe_steps": m,
            "sigma_curve": [round(v, 6) for v in grid],
            # THE REACH IN FORCE, stamped rather than inferred (rule 14).  `probe_band_floor` is
            # where the SHIPPED grid stopped and every scalar aggregate was computed;
            # `probe_reach_floor` is the lowest sigma actually measured.  A reader cannot tell a
            # 0.1-reach profile from a 0.01-reach one without both, and they are not comparable.
            "probe_reach_frac": round(float(cfg.get("probe_smin_frac",
                                                     _PROBE_SMIN_FLOOR)), 6),
            "probe_band_floor": round(float(_band_floor), 6),
            "probe_reach_floor": round(float(min(grid)), 6),
            "probe_reach_n_extra": int(_reach_extra),
            "jump_curve": [round(v, 6) for v in jumps],
            # Per-step probe diagnostics.  Why: a profile can report
            # jump_curve == 0 while the SAME model, in the real run, moves x0
            # strongly -- measured on Krea2 (probe jump 0,0,0 against the run's
            # own x0_jump_rms 1.17, 0.35, 0.28, ...), so the probe, not the
            # model, is what fails there.  The probe advances x with
            # exact_step(x, x0, s, s2) = r*x + (1-r)*x0, r = s2/s, so a frozen
            # trajectory has exactly two causes a bare jump_curve cannot
            # separate:
            #   x0_minus_x_rel ~ 0  -> the model call returned x0 == x (zero
            #                          velocity): the CALL is degenerate
            #   x0_minus_x_rel  > 0 -> the model answered and x still does not
            #                          move: the grid/step is degenerate
            # x_norm across the list shows whether x moved at all.
            "probe_diag": [
                {"s": round(float(grid[i]), 6),
                 "x_norm": round(float(xs[i].float().norm()), 6),
                 "x0_norm": round(float(x0s[i].float().norm()), 6),
                 "x0_minus_x_rel": round(
                     float((x0s[i].float() - xs[i].float()).norm())
                     / max(float(xs[i].float().norm()), 1e-8), 8)}
                for i in range(min(len(xs), len(x0s), len(grid)))
            ],
            "gap_curve": [round(v, 6) for v in gap_curve],
            "real_gap_curve": [round(v, 6) for v in real_gap_curve],
            "local_err_curve": [round(v, 6) for v in local_errs],
            "slope_curve": [round(j / max(gv, 1e-9), 8)
                            for j, gv in zip(jumps, gap_curve)],
            "med_local_err_real": (_pct(local_errs_real, 0.5)
                                   if local_errs_real else 0.0),
            # WHICH ESTIMATOR PRODUCED THESE NUMBERS (U16/6906).  `pct_derived` changes what
            # the percentiles ARE, so it is a MEASUREMENT input, not a calibration input:
            # stamped here so a reader can tell the two estimators apart without trusting a
            # key name, and left out of `calib_flags` for the same reason.
            "pct_derived": bool(cfg.get("pct_derived", False)),
            "curv_curve": [round(v, 6) for v in curvs],
            # THE SIGMA EACH CURVATURE POINT IS ANCHORED AT.  The convention is documented at
            # `aflops.py:7687` -- `curv_curve[k]` is the curvature at `sigma_curve[k+1]`, curves carry
            # m-2 entries against the grid's m -- and a reader who does not know it will mis-align every
            # derivation, as a rig of mine did this session.  Logging the anchors removes the need to
            # know it.
            "curv_sigma": [round(float(v), 6) for v in grid[1:1 + len(curvs)]],
            # PROBE-SIDE COVERAGE CURVE.  The same statistic the rescue gate
            # consumes in the run (`frac`), computed along THIS probe trajectory
            # by the SAME function, so a pitfall like the coverage cliff can be
            # seen BEFORE sampling rather than diagnosed afterwards.  The probe is
            # the only instrument that measures the model AND the prompt first.
            # See NOTES.md and scratch_rescue_cliff_rig.py.
            "cov_curve": (_cov_r or {}).get("curve"),
            "cov_curve_terms": (_cov_r or {}).get("terms"),
            # The DERIVED variant (ref_drop = k*du, k measured from this very
            # trajectory) alongside the shipped per-step curve.  Both are emitted
            # so the next real runs can decide which one actually tracks the run:
            # synthetic fields could not, see scratch_ref_decay_rig.py.
            "cov_curve_kdu": (_cov_r or {}).get("curve_kdu"),
            "cov_curve_k": (_cov_r or {}).get("k"),
            # WHICH probe model-call path ran, and with what arguments.  139 of
            # 224 profiles are frozen (model returns its input); this is what will
            # identify the cause on the next frozen run.
            "probe_call": dict(_PROBE_CALL_DIAG),
            "cov_curve_meta": _lf_rescue_cov_meta(
                x0s[-1] if x0s else None, _endgame_eg_err(endgame),
                _cov_floor, _cov_start),
            "med_jump": med_jump,
            "med_curv": med_curv,
            # THE PER-ORDER DERIVATIVE SCALE, measured from this very trajectory.  Without it
            # the derived wmax cap refuses (see _derived_wmax_cap), so this is what lets the
            # `wmax_bound` arm engage at all.  Omitted orders are ones this grid is too short
            # to certify, and the cap then falls back rather than guessing.
            "dk_scale": _measure_dk_scale(x0s, gap_probe, return_series=True)[0],
            # THE PER-STEP SERIES behind that median.  The scalar above is what the cap
            # consumes today; this is what a per-model order_max equation needs, because
            # a median hides the tail where the probe and the run disagree.  Anchor:
            # series[k][i] belongs to sigma_curve[k + i].
            "dk_series": _measure_dk_scale(x0s, gap_probe, return_series=True)[1],
            # THE A/B FLAGS THIS PROFILE WAS BUILT UNDER.  `_calib_flags` reads this, and
            # without it the `wmax_bound` arm was INERT whatever the widget said -- a toggle
            # wired to nothing, which is indistinguishable from one whose effect is below the
            # noise floor (the exact trap rule 9 exists for).  `cfg` is in scope here.
            _CALIB_FLAGS_KEY: _calib_inputs(cfg),
            "med_jump_rel": med_jump_rel,
            "med_detail_abs": med_detail_abs,
            "early_frac": early_frac,
            "conv_step": conv_step,
            "conv_sigma": (round(float(grid[int(conv_step)]), 6)
                           if 0 <= conv_step < len(grid) else None),
            "osc_sigma": (round(float(osc_sigma), 6)
                          if osc_sigma is not None else None),
            "is_distilled_eff": is_distilled_eff,
            "safe_fresh_frac": safe_f,
            # THE SERIES THE SCALAR WAS COMPUTED FROM, present only when
            # `probe_noise_survival_curve` asked for it (empty list otherwise).
            # Namespaced under its own key so no existing reader can mistake it
            # for part of the profile's contract: nothing consumes it, it is
            # written for the run log and for whoever derives a noise scale from
            # it later.
            "noise_survival_curve": _ns_series,
            # The diagnostic beside the thing it describes, so a reader can see
            # whether THIS profile agrees with the cfg stamp (`_ns_diag`).
            # Temporary; removed once the regression is explained.
            "_ns_curve_diag": dict(_ns_diag),
            "whitepoint": whitepoint,
            "med_local_err": (_pct(local_errs, 0.5) if local_errs else 0.0),
            "guard_sig_med": (_pct(guard_sigs, 0.5) if guard_sigs else 0.0),
            "guard_sig_p90": (_pct(guard_sigs, 0.9) if guard_sigs else 0.0),
            # BOTH estimators logged whatever the toggle, so one run shows the delta and,
            # with `n_guard`, whether the p90/med collision is in play for this profile.
            "guard_sig_med_shipped": (_probe_pct(guard_sigs, 0.5, False)
                                      if guard_sigs else 0.0),
            "guard_sig_p90_shipped": (_probe_pct(guard_sigs, 0.9, False)
                                      if guard_sigs else 0.0),
            "guard_sig_med_derived": (_probe_pct(guard_sigs, 0.5, True)
                                      if guard_sigs else 0.0),
            "guard_sig_p90_derived": (_probe_pct(guard_sigs, 0.9, True)
                                      if guard_sigs else 0.0),
            "log_gap_probe": gap_probe,
            "log_gap_real": gap_real,
            "n_jump": len(jumps),
            "n_err": len(local_errs),
            "n_guard": len(guard_sigs),
            "jump_q": jump_q,
            "err_q": err_q,
            "fragility": _fragility_map(frag_acc),
            "ojf_curve": ojf_curve,
            "band_curve": band_curve,
            "dir_cons": dir_cons,
            "dir_cons_valid": dir_cons_valid,
            "endgame": endgame,
            "probe_cfg": _probe_effective_cfg(model, extra_args),
        }
        profile["calib"] = _derive_calibrations(profile)
        profile["trajectory_state"] = _compute_trajectory_state(profile)
        return profile
    except Exception as e:
        logging.warning("[PD-AOS-probe] probe failed: %s", e)
        return None

def _recalibrated(profile, cfg, derive):
    """The profile with its calibration re-derived for THIS run's inputs.

    WHY THIS EXISTS: `profile["calib"]` is computed once, when the profile is built, and a
    profile is CACHED and reused across runs.  So a cached profile carries the calibration
    of whatever inputs were live when it was built -- log_00159 reports `wmax_bound=False`
    in its cfg while its own `calib_flags` say the arm was ON, i.e. the run did not do what
    the switch said.  Re-deriving here is exact (every measurement the derivation reads --
    `dk_scale`, `med_curv`, the gap fields -- is independent of the switches and of the
    budget) and free (pure arithmetic: no probe, no model call).  That is what makes a
    slider nudge take effect immediately instead of needing a re-probe.

    Returns the SAME object when the inputs already match, so the common path allocates
    nothing.  When they differ it updates the profile IN PLACE, deliberately: the report
    logs the cached profile objects themselves (`nodes.py` reads `_PROBE_PROFILES`), so
    copying would leave the log showing a calibration that did not run -- the same class of
    gap as a switch that reports ON while running OFF.
    """
    if not isinstance(profile, dict):
        return profile
    want = _calib_inputs(cfg)
    prev = _calib_flags(profile)
    if prev == want:
        return profile
    profile[_CALIB_FLAGS_KEY] = want
    try:
        # THE CONFIG THAT TRIGGERED THIS RE-DERIVATION IS THE ONE THE DERIVATION MUST READ.
        # `_derive_cond_calibrations` takes an optional cfg precisely so the value it uses comes from the
        # same cfg whose inputs justified re-deriving -- see its docstring for the divergence this closes.
        # The model-path derivation takes only the profile, so the two-argument call is attempted and the
        # one-argument form is the fallback rather than the reverse.
        try:
            profile["calib"] = derive(profile, cfg)
        except TypeError:
            profile["calib"] = derive(profile)
    except Exception as _e:
        # Do not leave the stamp claiming inputs whose calibration was never derived.
        profile[_CALIB_FLAGS_KEY] = prev
        logging.warning("[PD-AOS] calibration re-derivation failed (%s: %s); "
                        "keeping the previous calibration", type(_e).__name__, _e)
    return profile


def _apply_eta_cap(cfg, cap, source):
    """Cap a user-set `eta` at the probe-measured `safe_eta`.

    The old block only ran when `"eta" in auto_tunable`, which is the negation
    of `cfg["eta"] is not None`: the cap was unreachable on both branches.  A
    user-set eta above the measured safe value is now lowered, and the value in
    force, its source and the value it replaced are stamped for the log.
    """
    if cap is None or cfg.get("eta") is None:
        return
    try:
        before = float(cfg["eta"])
        cap_f = float(cap)
    except (TypeError, ValueError):
        return
    if cap_f < before:
        cfg["eta"] = cap_f
        cfg["_eta_cap_src"] = str(source)
        cfg["_eta_cap_value"] = round(cap_f, 6)
        cfg["_eta_cap_before"] = round(before, 6)


def _apply_probe_profile(cfg, profile, tune, auto_tunable=None):
    if not profile:
        return
    profile = _recalibrated(profile, cfg, _derive_calibrations)
    calib = profile.get("calib", {})
    # THE PER-ORDER DERIVATIVE SCALE, plumbed to the run so the remainder gate can consume
    # it.  It is a measurement from the engine probe; the cond probe only supplies it when
    # no model-site measurement exists (its grid is shorter, so it certifies fewer orders).
    try:
        _dk = profile.get("dk_scale")
        if isinstance(_dk, dict) and _dk:
            cfg["_dk_scale"] = {int(k): float(v) for k, v in _dk.items()}
    except Exception as _e:
        logging.debug("[PD-AOS] dk_scale not applied: %s", _e)
    if auto_tunable is None:
        auto_tunable = {k for k in ENGINE_DEFAULTS
                        if cfg.get(k) == ENGINE_DEFAULTS[k]}

    def _set(key, val):
        if key not in auto_tunable:
            return
        try:
            cur = float(cfg.get(key))
            val_f = float(val)
            if cur > 0 and val_f > 0 and key != "warmup":
                val_f = max(0.25 * cur, min(4.0 * cur, val_f))
                cfg[key] = val_f
            else:
                cfg[key] = val
        except (TypeError, ValueError):
            cfg[key] = val

    if tune.get("guard", True) and "guard_floor" in calib:
        _set("guard_floor", calib["guard_floor"])
    if tune.get("tol", True) and "rtol" in calib:
        _set("rtol", calib["rtol"])
    if tune.get("wmax", True) and "wmax_base" in calib:
        _set("wmax_base", calib["wmax_base"])
    if tune.get("anomaly", True):
        for _ak in ("an_from", "an_spatial", "an_novelty", "an_value"):
            if _ak in calib:
                _set(_ak, calib[_ak])
    if tune.get("eta", False):
        _apply_eta_cap(cfg, calib.get("safe_eta"), "model:calib.safe_eta")
    if profile.get("is_distilled_eff"):
        _set("warmup", 1)
    # The adaptive noise path ("auto") can only pick a beta once a jump map
    # exists -- the very first fresh-noise draw of a run had none and fell
    # back to plain white.  The probe already measured the model's spectral
    # structure (whitepoint); seed that first draw from it via the same
    # centroid->beta LUT the runtime uses, at half strength (evidence prior,
    # not a user override).
    wp = profile.get("whitepoint")
    if wp is not None and str(cfg.get("noise_color", "white")) == "auto":
        try:
            _bp = min(_centroid_to_beta(float(wp)), 0.0) * 0.5
            _cb_min = float(cfg.get("color_beta_min", -0.5) or 0.0)
            cfg["_noise_beta_prior"] = round(min(max(_bp, _cb_min), 0.0), 4)
        except (TypeError, ValueError):
            pass

def _model_weights_device(model_patcher):
    try:
        base_model = getattr(model_patcher, "model", None)
        if base_model is not None:
            try:
                return next(base_model.parameters()).device
            except (StopIteration, RuntimeError):
                pass
        ld = getattr(model_patcher, "load_device", None)
        if ld is not None:
            return torch.device(ld)
    except Exception:
        pass
    return torch.device("cpu")

def _model_weights_dtype(model_patcher):
    """`(logical, storage)` dtype of the model's first parameter, or `(None, None)`.

    WHY THIS EXISTS.  "Which precision did this run use?" was not answerable from a log: the
    model key is a digest, `loaded_mb` is the dynamic loader's resident byte count, and
    `probe_call.x_dtype` is the LATENT's dtype.  A run's directory is a NAME and does not answer
    it either -- measured 2026-10-08, the `int8_model_adapter` transformer option appears in runs
    under BOTH `anima-base-v1.0` and `animaprequantized_00001_` and is absent from most runs in
    both.  `storage` reads a quantised parameter's `_qdata` -- the same payload `_param_digest`
    samples -- so it reports the RAW precision instead of whatever dtype a tensor subclass
    advertises.  Nothing reads these yet: they are written so the next question about precision
    is a log read rather than an inference, exactly as `_seeds` was.
    """
    try:
        base = getattr(model_patcher, "model", None)
        p = None
        if base is not None:
            try:
                p = next(base.parameters())
            except (StopIteration, RuntimeError):
                p = None
        if p is None:
            return None, None
        logical = str(getattr(p, "dtype", None))
        qd = getattr(p, "_qdata", None)
        storage = str(getattr(qd, "dtype", None)) if qd is not None else logical
        return logical, storage
    except Exception:
        return None, None

def _ensure_model_gpu(model_patcher):
    try:
        ld = getattr(model_patcher, "load_device", None)
        if ld is None:
            return False
        ld = torch.device(ld)
        if ld.type == "cpu":
            return False
        try:
            if float(getattr(model_patcher, "loaded_size", lambda: 0.0)()
                     or 0.0) > 0.0:
                return True
        except (TypeError, ValueError, AttributeError):
            pass
        import comfy.model_management as _mm
        _load = getattr(_mm, "load_models_gpu", None)
        if _load is None:
            return False
        _load([model_patcher])
        logging.info("[PD-AOS-probe] model nudged onto %s before probing "
                     "(ComfyUI had left it offloaded)", ld)
        return True
    except Exception as e:
        logging.info("[PD-AOS-probe] could not nudge model onto the GPU "
                     "before probing (continuing on current devices): %s", e)
        return False

def _probe_call_device(model_patcher):
    try:
        ld = getattr(model_patcher, "load_device", None)
        if ld is not None:
            ld = torch.device(ld)
            if ld.type != "cpu":
                try:
                    resident = float(getattr(model_patcher, "loaded_size",
                                             lambda: 0.0)() or 0.0) > 0.0
                except (TypeError, ValueError, AttributeError):
                    resident = False
                if resident or _ensure_model_gpu(model_patcher):
                    return ld
    except Exception:
        pass
    return _model_weights_device(model_patcher)

def _is_device_error(e):
    try:
        msg = str(e).lower()
    except Exception:
        return False
    return ("device" in msg) or ("cuda" in msg) or ("cpu" in msg)

def _probe_failure_diagnostics(model, cur, ones, extra_args, gap_ctx=None):
    try:
        parts = []
        if torch.is_tensor(cur):
            parts.append("x=%s" % cur.device)
            parts.append("x_shape=%s" % (tuple(cur.shape),))
        if torch.is_tensor(ones):
            parts.append("t=%s" % ones.device)
        if isinstance(extra_args, dict):
            for k, v in list(extra_args.items())[:8]:
                if torch.is_tensor(v):
                    parts.append("%s=%s" % (k, v.device))
        if torch.cuda.is_available():
            parts.append("cur_dev=cuda:%d" % torch.cuda.current_device())
            parts.append("n_gpu=%d" % torch.cuda.device_count())
        patcher = None
        if isinstance(gap_ctx, dict):
            patcher = gap_ctx.get("patcher")
            if patcher is None:
                guider = gap_ctx.get("guider")
                if guider is not None:
                    patcher = getattr(guider, "model_patcher", None)
        if patcher is None:
            patcher = _navigate_to_patcher(model)
        if patcher is not None:
            ld = getattr(patcher, "load_device", None)
            try:
                ls = float(getattr(patcher, "loaded_size",
                                   lambda: 0.0)() or 0.0)
            except Exception:
                ls = -1.0
            parts.append("load_device=%s loaded_size=%.4g" % (ld, ls))
            base = getattr(patcher, "model", None)
            if base is not None:
                hist = {}
                try:
                    for _pn, _pp in base.named_parameters():
                        _d = str(_pp.device)
                        hist[_d] = hist.get(_d, 0) + 1
                except Exception:
                    pass
                if hist:
                    parts.append("weights=%s" % hist)
        return " ".join(parts) if parts else "(no device info)"
    except Exception:
        return "(diagnostics unavailable)"

def _guider_probe_cfg(guider):
    for attr in ("cfg1", "cfg"):
        v = _safe_getattr(guider, attr, None)
        if not isinstance(v, bool) and isinstance(v, (int, float)):
            return float(v)
    return None

def _guider_cond_signature(guider, probe_cfg=None):
    try:
        pos, neg = _guider_conds(guider)
        sig = _cond_signature(pos, neg)
        if probe_cfg is None:
            probe_cfg = _guider_probe_cfg(guider)
        if probe_cfg is not None:
            try:
                sig += "|cfg=%.6g" % float(probe_cfg)
            except (TypeError, ValueError):
                pass
        return sig
    except Exception:
        return None

def _generate_provisional_sigmas(model_patcher, steps=20):
    try:
        model_sampling = model_patcher.get_model_object("model_sampling")
        shift = getattr(model_sampling, "shift", None)
        if shift is None:
            shift = 1.0
        shift = float(shift)
        sigmas = aflops_scheduler_sigmas(model_sampling, int(steps),
                                          "log_uniform", shift=shift,
                                          denoise=1.0, tail_subdivide=1,
                                          tail_steps=1)
        return sigmas
    except Exception as e:
        logging.warning("[PD-AOS-probe] could not generate provisional "
                        "sigmas: %s", e)
        return None

def _set_probe_guider_conds(guider, positive, negative):
    def _list(c):
        if c is None:
            return []
        try:
            return list(c)
        except TypeError:
            return []

    def _is_converted(c):
        return bool(c) and all(isinstance(e, dict) for e in c)

    try:
        pos = _list(positive)
        neg = _list(negative)
        pos_conv = _is_converted(pos)
        neg_conv = _is_converted(neg)
        if pos_conv or neg_conv:
            try:
                import comfy.samplers as _cs
                try:
                    import comfy.sampler_helpers as _sh
                    _conv = getattr(_sh, "convert_cond", None)
                except Exception:
                    _conv = getattr(_cs, "convert_cond", None)
            except Exception:
                _conv = None

            def _norm(c, is_conv):
                if is_conv:
                    return [dict(e) if isinstance(e, dict) else e
                            for e in c]
                if not c:
                    return []
                if _conv is not None:
                    try:
                        r = _conv(list(c))
                        if r and all(isinstance(e, dict) for e in r):
                            return r
                    except Exception:
                        pass
                return list(c)

            pos_n = _norm(pos, pos_conv)
            neg_n = _norm(neg, neg_conv)
            guider.conds = {"positive": pos_n, "negative": neg_n}
            guider.original_conds = {
                "positive": list(pos_n), "negative": list(neg_n)}
            return True
        guider.set_conds(pos, neg)
        return True
    except Exception:
        return False

def _engine_cond_probe_run(model_patcher, positive, negative, cfg_scale,
                           cfg, sigmas, shape, callback=None, probe_fn=None,
                           probe_kind="cond"):
    """Run a probing pass as a REAL ComfyUI sampling run.

    This is the only way PD-AOS probes ever touch a model: the probe
    trajectory executes as the sampler function of a genuine
    CFGGuider + KSAMPLER run, so every model evaluation goes through
    ComfyUI's own sampling chain (prepare_sampling, process_conds,
    calc_cond_batch, cfg_function, model_function wrappers, additional
    models, guidance/concat/ref model_conds, ...) and the runtime prepares
    all device state for us (load_models_gpu, cuda_device_context,
    cast_to_load_options, pre_run).  That makes the probes MODEL-AGNOSTIC:
    SD/SDXL eps & v-prediction, Flux/Wan/Chroma-style flow models, video
    latents, guidance embeds, custom sampler_cfg_function hooks -- whatever
    ComfyUI can sample, the probe can measure, with no per-model handler
    code and nothing to re-implement when the next architecture lands.

    probe_fn(model, x, extra_args) -> profile overrides the default
    cond-probe call so the same machinery can also execute the MODEL probe
    node-side.  `model` is the KSamplerX0Inpaint wrapping the probe guider:
    calling it is exactly like calling the model during a real sampling
    step, and it returns the denoised (x0) prediction under the run's full
    CFG configuration.  `extra_args` is the run's own sampler kwargs
    (model_options, seed, denoise_mask) -- probes forward them so the chain
    model is invoked the way the revision requires (see _probe_model_call).
    """
    try:
        import comfy.samplers as _cs
        _Guider = getattr(_cs, "CFGGuider", None)
        _KS = getattr(_cs, "KSAMPLER", None)
        if _Guider is None or _KS is None:
            return None
        try:
            cfg_scale = float(cfg_scale)
        except (TypeError, ValueError):
            cfg_scale = 7.0
        prov_sigmas = sigmas
        try:
            s_max = float(prov_sigmas.reshape(-1)[0])
        except Exception:
            try:
                s_max = float(prov_sigmas[0])
            except Exception:
                return None
        if not (s_max > 1e-6):
            return None
        # ---- WHICH PATCHER: clone (restored) -------------------------------
        # The clone-removal experiment ran, was measured ineffective, and is
        # reverted below.  See "THE CLONE-REMOVAL EXPERIMENT" for the numbers.
        # WHY THIS CHANGED.  `clone()` re-uses the SAME model object, so it was
        # never a different model: get_clone_model_override returns self.model
        # (model_patcher.py:428-429), is_clone is `self.model is other.model`
        # (:592-595), and the module that actually runs is _prepare_sampling's
        # `real_model = model.model` (sampler_helpers.py:202).
        #
        # THE EVIDENCE FOR THIS CHANGE IS EMPIRICAL, NOT MECHANISTIC.  Measured
        # over every build (probe_call.ret_rel): every clone-path probe returned
        # its own input for all seven non-anima models (node 56 frozen / 0
        # healthy, cond 114 frozen / 0), while the SAME `_run_probe` on the real
        # patcher is healthy for every model (engine 0 frozen / 60 healthy);
        # anima is healthy on all three sites.  So
        #     frozen <=> (clone path) AND (model is not anima)
        # and the real patcher is the one variable that is healthy on every model.
        #
        # ---- THE CLONE-REMOVAL EXPERIMENT: RUN, MEASURED, REVERTED -----------
        # The clone was removed to test whether the frozen node/cond probes were
        # caused by it.  **THEY WERE NOT.**  On build b70ccf9fb59a -- the first
        # build where the removal was provably live in the running code (the log
        # carries `probe_patcher: "real"`, and the src_hash confirms the module
        # was re-imported) -- the NODE probe on cielbleuKrea2_v1-int8 STILL
        # returned its own input (ret_rel exactly 0.0, rank 5, [1,16,1,48,48])
        # while the GENUINE engine probe in the SAME run, on the SAME REAL
        # patcher, was healthy (ret_rel 0.00812805).  Same patcher, opposite
        # outcomes: the clone was never the discriminator.  The experiment is
        # CLOSED, and the clone is restored.
        #
        # The two earlier candidate mechanisms are ALSO settled -- do not
        # re-derive either:
        #   * patcher bookkeeping (patches_uuid copied unchanged, :450): the
        #     parent and the clone share ONE uuid, so it computes identically on
        #     both paths.  scratch_uuid_identity_rig.py (20/20) settled that.
        #   * the SQUARE synthetic latent: the node and engine probes pass the
        #     IDENTICAL shape at the identical rank, and only one is frozen.
        #
        # WHAT IS STILL OPEN: the node/cond paths freeze for a reason that
        # survives all three.  They differ from the healthy engine call in the
        # SYNTHETIC NESTED guider.sample RUN, not in the patcher and not in the
        # latent.  That is the remaining lead -- do NOT re-run the clone.
        #
        # Restoring the clone also restores deliberate isolation: the probing run
        # stays shape-private and away from the model's internal caches (see
        # _probe_model_options), and because clone() shares the
        # transformer_options dict with the original patcher, the sanitized dicts
        # REPLACE (never mutate) the clone's model_options.
        clone = model_patcher.clone()
        _probe_mopt = None
        try:
            _probe_mopt = _probe_model_options(
                {"model_options": getattr(clone, "model_options", None)})
            if _probe_mopt is not None:
                clone.model_options = _probe_mopt
        except Exception:
            _probe_mopt = None
        guider = _Guider(clone)
        if not _set_probe_guider_conds(guider, positive, negative):
            return None
        guider.set_cfg(float(cfg_scale))

        holder = {"profile": None}

        def _probe_sampler_fn(model, x, sigmas, extra_args=None,
                              callback=None, disable=False):
            try:
                _probe_model_cache_guard(model, True)
                try:
                    if probe_fn is not None:
                        holder["profile"] = probe_fn(model, x,
                                                     dict(extra_args or {}))
                    else:
                        holder["profile"] = _run_cond_probe(
                            model, x, prov_sigmas, dict(extra_args or {}), cfg,
                            gap_ctx={"guider": guider, "cfg": float(cfg_scale),
                                     "patcher": model_patcher},
                            callback=None)
                finally:
                    _probe_model_cache_guard(model, False)
            except Exception as e:
                import traceback as _tb
                logging.warning("[PD-AOS-%s-probe] probing run failed "
                                "inside the sampling window: %s\n%s",
                                probe_kind, e, _tb.format_exc())
            return x

        sampler = _KS(_probe_sampler_fn)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260915)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32)
        empty_latent = torch.zeros(shape, dtype=torch.float32)
        run_sigmas = torch.tensor([s_max, 0.0], dtype=torch.float32)
        # Guiders in the wild have drifted sample() signatures (custom
        # guiders, older ComfyUI revisions).  Try progressively simpler
        # invocations instead of giving up: there is no hand-rolled
        # model-call fallback anymore -- the chain IS the path.
        _sample_err = None
        _attempts = 0
        for _kwargs in ({"denoise_mask": None, "callback": None,
                         "disable_pbar": True, "seed": 20260915},
                        {"denoise_mask": None, "callback": None,
                         "disable_pbar": True},
                        {"callback": None, "disable_pbar": True},
                        {}):
            _attempts += 1
            try:
                guider.sample(noise, empty_latent, sampler, run_sigmas,
                              **_kwargs)
                _sample_err = None
                break
            except TypeError as _te:
                _sample_err = _te
                # A TypeError raised AFTER the sampler function returned is not a
                # signature mismatch: the probe has already run and its profiler is
                # in `holder`.  Retrying the call re-runs the ENTIRE probe -- up to
                # four times -- and the log shows one profile with no trace of the
                # extra attempts.  REPRODUCED synthetically: a fake guider whose
                # sample() runs the sampler and then raises TypeError makes
                # `probe_fn` run twice (rig_probe_cache.py N8a); after this guard,
                # once.  This is the random amplification that makes a small cond
                # probe occasionally cost several times its normal work.
                if holder["profile"] is not None:
                    logging.warning("[PD-AOS-%s-probe] the probing run completed "
                                    "but its wrapper raised TypeError afterwards "
                                    "(%s); keeping the measurement instead of "
                                    "re-running it", probe_kind, _te)
                    _sample_err = None
                    break
            except Exception as _se:
                _sample_err = _se
                break
        if _sample_err is not None:
            logging.warning("[PD-AOS-%s-probe] probing run could not start "
                            "through the ComfyUI sampling chain: %s",
                            probe_kind, _sample_err)
            return None
        if holder["profile"] is not None:
            logging.info("[PD-AOS-%s-probe] probing run executed as a "
                         "real ComfyUI sampling run (device state fully "
                         "prepared by the runtime: load_models_gpu + "
                         "additional models, cuda_device_context, "
                         "cast_to_load_options, pre_run)", probe_kind)
            # WHICH PATCHER THE PROBE RAN ON, stamped onto the PROFILE itself.
            # Deliberately NOT in _PROBE_CALL_DIAG: _probe_model_call clears that
            # dict at its start (aflops.py:3162), so anything written here would
            # be wiped before the call that consumes it -- the exact
            # instrumentation bug this project has already hit once.  On the
            # profile it cannot be clobbered, and it lands in the corpus (both
            # nodes.py whitelists now name it -- before that it was silently
            # dropped, and its ABSENCE was misread as "the change did not ship").
            # Current value is "clone"; a build that prints "real" is one running
            # the reverted clone-removal experiment.
            try:
                holder["profile"]["probe_patcher"] = "clone"
                holder["profile"]["probe_mopt_swapped"] = bool(
                    _probe_mopt is not None)
                # HOW MANY TIMES THE WRAPPER WAS ENTERED.  One is the normal case;
                # a higher value is the signature-retry path, and it is stamped so
                # an audit can see it without a terminal log.
                holder["profile"]["probe_attempts"] = _attempts
            except Exception:
                pass
        return holder["profile"]
    except Exception as e:
        import traceback as _tb
        logging.info("[PD-AOS-%s-probe] engine-run probe unavailable "
                     "(falling back to the direct path): %s\n%s",
                     probe_kind, e, _tb.format_exc())
        return None

def _sig_list_sig(lst):
    """Compact signature of a sigma list (for schedule-keyed probe caches)."""
    try:
        return hashlib.md5(
            ",".join("%.5f" % float(v) for v in lst).encode("utf-8")
        ).hexdigest()[:12]
    except Exception:
        return "?"


def _norm_sig_list(sigmas):
    """External sigma input -> descending float list, or None."""
    if sigmas is None:
        return None
    try:
        lst = _sigmas_as_floats(sigmas)
        if not lst:
            return None
    except Exception:
        return None
    if len(lst) < 4 or lst[0] <= lst[1] or lst[0] <= 0.0:
        return None
    return lst


def _run_node_side_cond_probe(model_patcher, positive, negative, cfg,
                              cfg_scale=7.0, sigmas=None):
    try:
        # Same reason as the model probe: this computes and caches a
        # `trajectory_state`, so it must use its OWN cfg's A/B modes rather than
        # a mode left behind by the last engine run.  See _apply_ab_modes.
        _apply_ab_modes(cfg)
        mk = _model_cache_key(model_patcher)
        cond_sig = cfg.get("cond_probe_sig")
        ext = _norm_sig_list(sigmas)
        # THE SAME DIMENSION SUFFIX AS `_cond_cache_key` -- one helper, so the node-side path and this
        # one cannot drift into writing and reading different keys (the failure `_cond_key_str` was
        # created to end).
        _key = (mk, str(cond_sig) + _cond_probe_dims(cfg)
                + ("|" + _probe_grid_sig(ext, cfg) if ext is not None else ""))
        if cond_sig is not None and not bool(cfg.get("cond_probe_force", False)):
            cached = _COND_PROBE_PROFILES.get(_key)
            if cached is not None:
                _PROFILE_BY_MODEL[mk] = cached
                # WHICH NODE-SIDE PROBE RAN gets its OWN phase key, and NOT `node_probe_state`.
                # The two probes both run in one queue -- the model probe per (model, config) and
                # this one per prompt -- so sharing one scalar means whichever runs second erases
                # the other's outcome.  The cost of that was visible in the corpus: a queue whose
                # only node-side work was THIS probe reported `probe_calls_node_side` > 0 with no
                # state beside it at all (10 runs), because `node_probe_state` belongs to the model
                # probe.  The vocabulary deliberately mirrors it: cache_hit / empty_deferred / ran.
                _nphase_set("node_cond_probe_state", "cache_hit")
                logging.info("[PD-AOS-cond-probe] node-side probe: model+prompt "
                             "unchanged, using cached profile (skip re-probe)")
                return cached, True
        # SAY SO WHEN THIS QUEUE PAYS FOR THE MEASUREMENT.  The node path used to be
        # silent on a run, so the only terminal line about a NEW prompt was the engine's
        # "using cached ..." above -- which made a fresh node-side measurement read as a
        # stale cache hit (moss log_00071, new prompt 941fa9ed3f77e510).  The work and
        # the report must name the same event.
        if ext is not None:
            sigmas = ext
        else:
            sigmas = _generate_provisional_sigmas(model_patcher, steps=20)
        if sigmas is None or len(sigmas) < 4:
            return None, False
        logging.info("[PD-AOS-cond-probe] node-side probe: measuring THIS prompt "
                     "(no cached profile)")
        pres = max(8, int(cfg.get("cond_probe_resolution", 16)))
        mi = _extract_model_info(model_patcher)
        B = 1
        def _fallback_latent_channels():
            try:
                lf = model_patcher.get_model_object("latent_format")
                v = _lf_get(lf, "latent_channels", 4)
                return int(v)
            except Exception:
                return 4
        if mi and mi.get("latent_channels"):
            C = int(mi["latent_channels"])
        else:
            C = _fallback_latent_channels()
        # ---- RANK LADDER, DRIVEN BY DEGENERACY (same reason as the model
        # probe above: ref_x is None, and a 4-D latent through a 5-D model is a
        # SILENT pass-through, so no exception is available to drive this).
        _ladder = _probe_rank_ladder(model_patcher, None, mi)

        def _run_at_rank(_rank):
            _shape = _probe_latent_shape(
                ref_x=None, pres=pres, batch=B, channels=C, rank=_rank,
                patcher=model_patcher, mi=mi,
                site="node/_run_node_side_cond_probe")
            return _engine_cond_probe_run(model_patcher, positive, negative,
                                          cfg_scale, cfg, sigmas, _shape)

        _t_nc = _t0()
        try:
            profile, _rank_used, _tried = _probe_rank_ladder_run(
                _run_at_rank, _ladder, _probe_degenerate)
        finally:
            # THE PER-PROMPT PROBE'S OWN CLOCK.  It had none, so "the small probe
            # is slow" could not be read from a log at all -- only the model
            # probe's `node_probe_ms` existed.  This lands in `_PHASE_MS` via the
            # node-phase copy at engine entry.
            _nphase_add("node_cond_probe_ms", _t_nc)
        if profile is not None:
            try:
                profile["probe_rank"] = _rank_used
                profile["probe_rank_tried"] = list(_tried)
                profile["probe_rank_ladder"] = list(_ladder)
            except Exception:
                pass
            # REMEMBER THE WORKING RANK so the ladder is paid ONCE PER MODEL,
            # not once per prompt.  The cond probe is per-prompt and NOT cached
            # across prompts, so without this an affected model would run a
            # doubled chain on every new prompt.  With it, _resolve_probe_rank
            # short-circuits to the working rank and the ladder settles first
            # try.  Correctness is unaffected: a remembered rank that still
            # produced a degenerate profile would simply advance again.
            if _rank_used in (4, 5):
                _remember_probe_rank(model_patcher, _rank_used)
        if profile is None:
            # No hand-rolled fallback: if the node-side chain run could not
            # execute, defer to the engine-side probe, which runs inside the
            # real sampling run through the very same ComfyUI chain.
            logging.info("[PD-AOS-cond-probe] node-side chain probe did "
                         "not execute; deferring to the engine-side probe "
                         "at sampling time")
            _nphase_set("node_cond_probe_state", "empty_deferred")
            return None, False

        if profile is not None and cond_sig is not None:
            _COND_PROBE_PROFILES[_key] = profile
            _plain = (mk, str(cond_sig))
            if _key != _plain:
                # Plain (schedule-agnostic) slot: the engine's per-prompt
                # fallback and any legacy plain-key lookup read it; the
                # schedule-keyed slot keeps freshness per schedule.
                _COND_PROBE_PROFILES[_plain] = profile
            # THE ENGINE'S OWN KEY, WHICH NEITHER SLOT ABOVE MATCHES.  The engine falls back to
            # `_COND_PROBE_PROFILES.get(_cond_cache_key(cfg, extra_args, mk))`, and that key is
            # `(mk, cond_sig + _cond_probe_dims(cfg))` -- so `_key` misses it by the `|g:` schedule
            # suffix and `_plain` misses it by the missing dimension suffix.  NEITHER COULD EVER
            # MATCH, so the engine re-ran the per-prompt probe at sampling time on every
            # generation.  Measured on the operator's runs (anima build 836ada7, logs 00129-00133,
            # `probe_schedule` ON): `cond_probe_state: ran` with `cond_probe_ms` 3.5-4.1 s per run,
            # ~25 engine probe model calls, on top of the node's own 39 -- the same measurement
            # taken twice, every generation.  Written here so the engine finds what already exists.
            _engine_key = (mk, str(cond_sig) + _cond_probe_dims(cfg))
            if _engine_key != _key and _engine_key != _plain:
                _COND_PROBE_PROFILES[_engine_key] = profile
            _PROFILE_BY_MODEL[mk] = profile
        _nphase_set("node_cond_probe_state", "ran")
        return profile, False
    except Exception as e:
        logging.warning("[PD-AOS-cond-probe] node-side probe failed "
                        "(falling back to engine-side): %s", e)
        return None, False

def _run_node_side_cond_probe_guider(guider, cfg, cfg_scale=None,
                                     sigmas=None):
    patcher = getattr(guider, "model_patcher", None)
    if patcher is None:
        patcher = _navigate_to_patcher(guider)
    if patcher is None:
        logging.warning("[PD-AOS-cond-probe] guider exposes no ModelPatcher; "
                        "node-side probe skipped")
        return None, False
    positive, negative = _guider_conds(guider)
    if positive is None and negative is None:
        logging.warning("[PD-AOS-cond-probe] guider carries no conditioning; "
                        "node-side probe skipped")
        return None, False
    eff_cfg = _guider_probe_cfg(guider)
    if eff_cfg is None:
        try:
            eff_cfg = float(cfg_scale if cfg_scale is not None
                            else cfg.get("cond_probe_cfg", 7.0))
        except (TypeError, ValueError):
            eff_cfg = 7.0
    if negative is None or (isinstance(negative, (list, tuple))
                            and len(negative) == 0):
        eff_cfg = 1.0
    return _run_node_side_cond_probe(patcher, positive, negative, cfg,
                                     eff_cfg, sigmas=sigmas)


def _node_probe_param_sig(cfg):
    """Fingerprint of the probe parameters that are baked into a model
    profile's measurement (widget changes must invalidate the cache).

    THE CALIB FLAGS BELONG HERE.  They change what the profile MEASURES -- which
    percentile estimator is used, and which wmax cap is derived -- so a profile built
    under one setting must not be reused after the switch is flipped.  They were missing,
    which produced exactly the symptom the operator reported twice: flipping the switch
    changed nothing, because `_PROBE_PROFILES.get(pkey)` hit the cached profile and the
    engine trusts an exact-key hit as-is.  The stale profile even carried the OLD
    `calib_flags` in its own log, which is how the reuse was visible.

    THE A/B MODES BELONG HERE FOR THE SAME REASON, and were missed when the calib
    flags were fixed.  They change `trajectory_state`, which is cached INSIDE the
    profile.  Measured consequence (scratch_oversmooth_engine_mode_rig.py): 40 of
    380 runs at the shipped default consumed a non-zero `oversmooth`, two of them
    in mode "off", because a node-side profile built under an arm was adopted by
    later default-mode runs.

    `wmax_budget` IS *NOT* HERE, by the same measured reasoning that keeps it out of
    `_schedule_sig`: the switch above changes what the profile MEASURES, whereas the
    budget reaches only the DERIVED calibration, and `_recalibrated` re-derives that
    from the profile's own budget-independent measurements before any consumer reads it.
    Adding it would fragment this cache (a fresh probe, ~10 model calls) for nothing.
    `scratch_wmax_budget_cover_rig.py` (9/9) carries the evidence, including the
    306-of-306 corpus check.

    THE DISTILLED GRID SHAPING BELONGS HERE, and it was the same kind of omission the two
    paragraphs above describe -- one round later.  It changes what the probe MEASURES
    (`_probe_key_env_part` carries the measurement), and this key is the node path's only line of
    defence: the engine's `_schedule_sig` is not consulted by a node-side probe.  Both keys call
    the SAME builder, so the two spellings cannot drift -- which is how the `_cond_key_str`
    defect happened.  (The CONDITIONING was keyed here for a few hours on 2026-10-08 and has been
    removed: a model measurement is shared across prompts by design, its store is model-keyed for
    that reason, and the evidence to the contrary pooled builds.  See `_probe_key_env_part`.)
    """
    return "|".join([
        str(int(cfg.get("probe_steps", 10))),
        str(int(cfg.get("probe_resolution", 48))),
        "d" if bool(cfg.get("probe_distributions", True)) else "-",
        "e" if bool(cfg.get("probe_endgame", True)) else "-",
        "c" if bool(cfg.get("probe_dir_cons", True)) else "-",
        "pd" if bool(cfg.get("pct_derived", False)) else "-",
        "wb" if bool(cfg.get("wmax_bound", False)) else "-",
        "osm=" + str(cfg.get("_osm_ab_mode", "engine") or "engine"),
        "ev=" + str(cfg.get("_ev_ab_mode", "engine") or "engine"),
        # `probe_smin_frac` changes the probe GRID, so it must invalidate the
        # node-side model slot.  The engine key and the cond key already carry
        # this term; the node key did not, so a reach change reused the old
        # measurement in the same session.
        "sf=%.4f" % float(_probe_reach_eff(cfg)),
        _probe_key_env_part(cfg),
    ])


def _node_model_probe_sigmas(patcher, cfg, sigmas=None, fallback=None):
    """The schedule the NODE-side MODEL probe measures on: CANONICAL, run-start preserving.

    WHY THIS EXISTS -- THE RE-PROBE LOOP (2026-10-18).  The node model probe used to be
    handed the run's own schedule and keyed on the points `_probe_grid` would walk.  For a
    long schedule that is harmless: `_probe_grid` derives the grid from the schedule's
    RANGE, so a different placement keys the same.  For a SHORT schedule (<= `probe_steps`
    positive sigmas) `_probe_grid` returns the schedule ITSELF, so the walked grid changes
    with the step count -- and, because the probe profile feeds the Scheduler's placement,
    the next run's schedule can change too.  A model-only measurement is then invalidated
    by the thing it just influenced.

    MEASURED on `moss_image_sft_int8_convrot`, one process, one model, one prompt
    (`cond_probe_sig 82fd429ce663faa5`), build 1a1e672d:
        log_00068   8 steps   node_probe_state ran   node_probe_ms 50 664.7
        log_00069  12 steps   node_probe_state ran   node_probe_ms 69 035.8
    while the two 40-step runs on either side (log_00066/67) were `cache_hit`.  The prompt
    did not change; the STEP COUNT did, and that was enough to re-run a ~50-70 s probe.

    THE NODE PROBE IS A MODEL MEASUREMENT, NOT A TRAJECTORY MEASUREMENT.  Its consumers
    resample the profile's curves onto the real sigmas (see the docstring of
    `_run_node_side_model_probe`), and the engine's own node-slot lookup
    (`_node_model_profile`) already keys on model + probe parameters + cfg, never on the
    grid.  Measuring this probe on a canonical grid makes the measurement and its identity
    agree with that consumer: same model, same probe parameters, same cfg, same RUN START
    -> same profile, whatever the step count or placement.

    WHAT IS PRESERVED FROM THE RUN: `smax`, the first positive sigma, because it is the
    start of the range the run actually visits.  It is invariant to the step count and
    placement, but moves with `denoise`, so an img2img run still measures from its own
    start instead of wasting points above it.  WHAT IS NOT PRESERVED: the interior and the
    low end.  The low end is the canonical `_PROBE_SMIN_FLOOR` of `smax` for flow (the
    engine's own floor), or the model's sigma_min for VE.

    The returned list carries `K+1` POSITIVE sigmas (plus a terminal 0.0), so `_probe_grid`
    has more than K and BUILDS its K-point geometric grid instead of returning a short
    schedule verbatim.  Measured on a flow sampling object at shift 5, K=16: 17 positive
    sigmas -> a 16-point grid 1.0 .. 0.1.

    `fallback` is the run's schedule, used only when no `smax` can be read.  In that case
    the old behaviour is kept, including its schedule-keyed identity; the alternative --
    treating a schedule-less call as canonical -- would hide the failure.
    """
    try:
        K = max(4, int(cfg.get("probe_steps", 10)))
    except (TypeError, ValueError):
        K = 10
    pos = []
    try:
        pos = [float(v) for v in _sigmas_as_floats(sigmas) if float(v) > 1e-6]
    except Exception:
        pos = []
    if pos:
        smax = max(pos)
        if smax > 1.5:
            try:
                smin = float(patcher.get_model_object(
                    "model_sampling").sigma_min())
            except Exception:
                smin = min(pos)
        else:
            smin = smax * _PROBE_SMIN_FLOOR
        try:
            smin = min(max(float(smin), 1e-8), smax * (1.0 - 1e-6))
            lmax = math.log(max(smax, 1e-8))
            lmin = math.log(max(smin, 1e-8))
            out = [math.exp(lmax + (lmin - lmax) * (i / float(K)))
                   for i in range(K + 1)] + [0.0]
            return torch.tensor(out, dtype=torch.float32)
        except Exception:
            pass
    return fallback


def _run_node_side_model_probe(guider, cfg, cfg_scale=None, sigmas=None):
    """Node-side MODEL probe: pre-compute the calibration profile in the
    Probe Options node BEFORE the KSampler starts.

    This is the model-probe counterpart of _run_node_side_cond_probe_guider.
    Without it the model probe only runs inside aflops_engine at SAMPLING
    time, so the profile arrives one generation too late for any node that
    executes before the KSampler -- most importantly the PD-AOS Scheduler,
    whose probe-based shift recommendation and distilled detection are blind
    on the first run.

    The probe borrows the guider's ModelPatcher, conditioning and live cfg
    scale, measures on a CANONICAL provisional log-uniform schedule (see
    `_node_model_probe_sigmas`; the consumers resample the profile's curves
    onto the real sigmas, exactly like the node-side cond probe), and caches
    the profile in _NODE_PROBE_PROFILES, _PROBE_PROFILES
    ("<mk>|s:node:<params>|<cfg>") and _PROFILE_BY_MODEL so the Scheduler can
    see it immediately and the engine can skip its own sampling-time probe.

    The schedule argument is a FALLBACK only.  A run's step count, placement
    and prompt must not invalidate a model measurement; keying this probe on
    the walked grid did exactly that and cost a fresh 50-70 s probe on every
    step-count change (see `_node_model_probe_sigmas` for the measurement).

    The measurement runs as a real ComfyUI sampling run through the chain
    (see _engine_cond_probe_run) -- model-agnostic, no per-model handlers.

    Returns (profile, from_cache); (None, False) means "fall back to the
    legacy engine-side probe".
    """
    try:
        # This probe computes (and caches) a `trajectory_state`, so it must
        # compute it under ITS OWN cfg's modes rather than inheriting whatever
        # the last engine run left in the module globals.  See _apply_ab_modes.
        _apply_ab_modes(cfg)
        patcher = getattr(guider, "model_patcher", None)
        if patcher is None:
            patcher = _navigate_to_patcher(guider)
        if patcher is None:
            logging.info("[PD-AOS-probe] guider exposes no ModelPatcher; "
                         "node-side model probe skipped")
            return None, False
        positive, negative = _guider_conds(guider)
        if positive is None and negative is None:
            logging.info("[PD-AOS-probe] guider carries no conditioning; "
                         "node-side model probe skipped")
            return None, False
        eff_cfg = _guider_probe_cfg(guider)
        if eff_cfg is None:
            eff_cfg = cfg_scale if cfg_scale is not None else 7.0
        try:
            eff_cfg = float(eff_cfg)
        except (TypeError, ValueError):
            eff_cfg = 7.0
        if negative is None or (isinstance(negative, (list, tuple))
                                and len(negative) == 0):
            eff_cfg = 1.0
        # The distilled 'auto' override keys off the live probe cfg; make it
        # visible to _run_probe (the engine recomputes it at sampling time).
        cfg.setdefault("_guider_live_cfg", eff_cfg)
        mk = _model_cache_key(patcher)
        ext = _norm_sig_list(sigmas)
        # THE SCHEDULE IS A FALLBACK, NOT AN INPUT (see `_node_model_probe_sigmas`).
        # Keying this on the run's walked grid re-ran the probe on every step-count or
        # placement change: measured, moss log_00068 (8 steps) 50.7 s and log_00069
        # (12 steps) 69.0 s on one model and one unchanged prompt.  The canonical grid
        # is model-derived, so the same model + probe parameters + cfg is the same key.
        prov = _node_model_probe_sigmas(patcher, cfg, ext, fallback=ext)
        nkey = (mk, _node_probe_param_sig(cfg), "%.2f" % eff_cfg,
                _probe_grid_sig(prov, cfg) if prov is not None else "provisional")
        if not bool(cfg.get("probe_force", False)):
            cached = _NODE_PROBE_PROFILES.get(nkey)
            if cached is not None and int(cached.get("version", 1) or 1) >= 7:
                _PROFILE_BY_MODEL[mk] = cached
                # A cache hit returns WITHOUT rewriting `_PROBE_PROFILES`, so a
                # node slot dropped by the old prune (or by a process that never
                # wrote it) stayed invisible to the engine's `_node_model_profile`.
                # Re-publish it under the same key the first run wrote, and the
                # engine can adopt the measurement instead of re-running it.
                try:
                    _PROBE_PROFILES["%s|s:node:%s|%s"
                                    % (mk, nkey[1], nkey[2])] = cached
                except Exception:
                    pass
                _nphase_set("node_probe_state", "cache_hit")
                logging.info("[PD-AOS-probe] node-side probe: model "
                             "unchanged, using cached profile (skip "
                             "re-probe)")
                return cached, True
        if prov is not None:
            sigmas = prov
        elif ext is not None:
            sigmas = ext
        else:
            sigmas = _generate_provisional_sigmas(patcher, steps=20)
        if sigmas is None or len(sigmas) < 4:
            return None, False
        pres = max(16, int(cfg.get("probe_resolution", 48)))
        # THE UI IS BLOCKED WHILE THIS RUNS, so say what it is before the first model
        # call.  Without this the node path was silent from the cache miss to the final
        # "probing run executed" line, which is how a finite 50-70 s measurement read as
        # a hang (the same reporting gap the cond path's "cached" line caused).
        logging.info("[PD-AOS-probe] node-side MODEL probe: measuring this model "
                     "(%d points @ %dx%d; the run schedule is not part of the key, "
                     "so step/prompt changes reuse it)",
                     max(4, int(cfg.get("probe_steps", 10))), pres, pres)
        mi = _extract_model_info(patcher)
        B = 1

        def _fallback_latent_channels():
            try:
                lf = patcher.get_model_object("latent_format")
                v = _lf_get(lf, "latent_channels", 4)
                return int(v)
            except Exception:
                return 4

        if mi and mi.get("latent_channels"):
            C = int(mi["latent_channels"])
        else:
            C = _fallback_latent_channels()
        # ---- RANK LADDER, DRIVEN BY DEGENERACY --------------------------
        # `ref_x` is None here, so the rank comes from detection and falls back
        # to 4.  A 4-D latent through a 5-D model runs SUCCESSFULLY and returns
        # its input, so nothing is raised and the ladder cannot be
        # exception-driven -- it must test the profile itself.  Measured on the
        # corpus: rank 4 -> 9 frozen / 0 healthy, rank 5 -> 0 frozen / 16 healthy.
        # See scratch_rank_ladder_rig.py.
        _ladder = _probe_rank_ladder(patcher, None, mi)

        def _probe_fn(model, x, extra_args=None):
            prof = _run_probe(model, x, sigmas, dict(extra_args or {}), cfg)
            if prof is not None:
                prof["probe_cfg"] = eff_cfg
            return prof

        def _run_at_rank(_rank):
            _shape = _probe_latent_shape(
                ref_x=None, pres=pres, batch=B, channels=C, rank=_rank,
                patcher=patcher, mi=mi,
                site="node/_run_node_side_model_probe")
            _t_np = _t0()
            try:
                return _engine_cond_probe_run(
                    patcher, positive, negative, eff_cfg, cfg, sigmas, _shape,
                    probe_fn=_probe_fn, probe_kind="model")
            finally:
                _nphase_add("node_probe_ms", _t_np)

        profile, _rank_used, _tried = _probe_rank_ladder_run(
            _run_at_rank, _ladder, _probe_degenerate)
        if profile is not None:
            try:
                profile["probe_rank"] = _rank_used
                profile["probe_rank_tried"] = list(_tried)
                profile["probe_rank_ladder"] = list(_ladder)
            except Exception:
                pass
            # Remember it (see the cond-probe twin above): pay the ladder once.
            if _rank_used in (4, 5):
                _remember_probe_rank(patcher, _rank_used)
        if profile is None:
            # No hand-rolled fallback: the engine-side probe (inside the
            # real sampling run, through the very same ComfyUI chain) takes
            # over at sampling time instead.
            logging.info("[PD-AOS-probe] node-side chain probe did not "
                         "execute; deferring to the engine-side probe at "
                         "sampling time")
            _nphase_set("node_probe_state", "empty_deferred")
            return None, False

        profile["node_side"] = True
        _nphase_set("node_probe_state", "ran")
        _NODE_PROBE_PROFILES[nkey] = profile
        try:
            _PROBE_PROFILES["%s|s:node:%s|%s" % (mk, nkey[1], nkey[2])] = \
                profile
            _prune_probe_cache(mk, keep=8)
        except Exception:
            pass
        _PROFILE_BY_MODEL[mk] = profile
        return profile, False
    except Exception as e:
        logging.warning("[PD-AOS-probe] node-side probe failed (falling "
                        "back to engine-side): %s", e)
        return None, False

def _compute_trajectory_state(profile):
    try:
        is_distilled = bool(profile.get("is_distilled_eff", False))
        jumps = profile.get("jump_curve", [])
        gaps_c = profile.get("gap_curve") or []
        if gaps_c and len(gaps_c) == len(jumps):
            slope_j = [j / max(g, 1e-9) for j, g in zip(jumps, gaps_c)]
        else:
            slope_j = jumps
        conv_ratio = 1.0  # neutral default
        if len(slope_j) >= 3:
            n3 = max(1, len(slope_j) // 3)
            early = sum(slope_j[:n3]) / n3
            late = sum(slope_j[-n3:]) / n3
            if early > 1e-8:
                conv_ratio = late / early
        trend_resid = None
        if len(slope_j) >= 6 and all(s > 1e-12 for s in slope_j):
            _k = max(2, min(3, len(slope_j) // 3))
            _tail = slope_j[-_k:]
            _mid = slope_j[-2 * _k:-_k]
            _t_med = sorted(_tail)[len(_tail) // 2]
            _m_med = sorted(_mid)[len(_mid) // 2]
            if _t_med > 1e-12 and _m_med > 1e-12:
                trend_resid = math.log(_t_med / _m_med)
        med_err_raw = float(profile.get("med_local_err", 0.0) or 0.0)
        gap_p = float(profile.get("log_gap_probe", 0.0) or 0.0)
        gap_r = float(profile.get("log_gap_real", 0.0) or 0.0)
        lec = profile.get("local_err_curve") or []
        gpc = profile.get("gap_curve") or []
        grc = profile.get("real_gap_curve") or []
        med_err = None
        if (lec and gpc and grc and len(lec) == len(gpc)
                and len(lec) == len(grc)):
            per = []
            for _e, _gp, _gr in zip(lec, gpc, grc):
                if _gp > 1e-9 and _gr > 1e-9:
                    per.append(_e * min(max((_gr / _gp) ** 2, 0.04), 1.0))
                else:
                    per.append(_e)
            if per:
                med_err = sorted(per)[len(per) // 2]
        if med_err is None:
            stored_real = profile.get("med_local_err_real")
            if stored_real is not None and med_err_raw > 0:
                med_err = float(stored_real)
            else:
                med_err = med_err_raw
                if med_err_raw > 0 and gap_p > 1e-9 and gap_r > 1e-9:
                    scale = min(max((gap_r / gap_p) ** 2, 0.04), 1.0)
                    med_err = med_err_raw * scale
        if trend_resid is not None:
            s_ovs_conv = max(
                max(0.0, min(1.0, (trend_resid - 0.3) / 0.7)),
                max(0.0, min(1.0, (conv_ratio - 1.0) / 1.0)))
        else:
            s_ovs_conv = max(0.0, min(1.0, (conv_ratio - 1.0) / 1.0))
        curvs = profile.get("curv_curve", [])
        med_curv = float(profile.get("med_curv", 0.0) or 0.0)
        late_curv_ratio = 1.0
        if len(curvs) >= 2 and med_curv > 1e-8:
            late_curv = sum(curvs[len(curvs) // 2:]) / max(1, len(curvs) - len(curvs) // 2)
            late_curv_ratio = late_curv / med_curv
        s_ovs_curv = max(0.0, min(1.0, (late_curv_ratio - 1.0) / 2.0))
        s_ovs_err = 1.0 - math.exp(-med_err / 0.03)
        gap = profile.get("gap")
        probe_cfg = None
        try:
            probe_cfg = float(profile.get("probe_cfg"))
        except (TypeError, ValueError):
            probe_cfg = None
        s_ovs_ov = 0.0
        if gap is not None and probe_cfg is not None and probe_cfg > 1.0 + 1e-6:
            rel_ov = float(gap.get("overshoot", 0.0)) / (probe_cfg - 1.0)
            s_ovs_ov = 1.0 - math.exp(-rel_ov / 1.2)
        sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
        sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
        spike_ratio = (sig_p90 / max(sig_med, 1e-8)
                       if sig_med > 1e-8 else 1.0)
        # The guard-signal p90/median ratio is a ratio that spans orders of
        # magnitude (a hot-pixel model measures ~17, a smooth one ~1.5).  A
        # log mapping keeps it in dynamic range instead of the exponential
        # saturating to ~1 for anything above ~5 and drowning the other
        # oversteer signals.
        s_ovs_spike = min(1.0, max(0.0,
                                   math.log(max(spike_ratio, 1.0)) / math.log(30.0)))
        s_ovs_bimod = 0.0
        ojf = profile.get("ojf_curve") or []
        if ojf:
            ojf_mean = sum(float(x) for x in ojf) / len(ojf)
            s_ovs_bimod = max(0.0, min(1.0, (ojf_mean - 0.06) / 0.24))

        signals = [s_ovs_conv, s_ovs_curv, s_ovs_err, s_ovs_ov, s_ovs_spike]
        if ojf and s_ovs_bimod > 0.0:
            signals.append(s_ovs_bimod)
        not_oversteer = 1.0
        for sv in signals:
            not_oversteer *= (1.0 - sv)
        oversteer = max(0.0, min(1.0, 1.0 - not_oversteer ** (1.0 / len(signals))))
        # REACHABILITY, and it is load-bearing for the constant audit: the eight `_OSM_RA_*`
        # thresholds below exist ONLY inside the "reanchored" arm, and the default mode is
        # "engine" -- so a sensitivity rig that sweeps those constants WITHOUT setting this
        # mode is varying code that never runs, and "insensitive" from such a rig means
        # nothing.  These counters make that checkable instead of invisible.
        _ab_hit("osm.mode_" + str(_OSM_AB_MODE))
        if _OSM_AB_MODE == "off":
            oversmooth = 0.0
        elif _OSM_AB_MODE == "reanchored":
            if trend_resid is not None:
                s_osm_conv = max(0.0, min(1.0, (_OSM_RA_TREND_ZERO - trend_resid)
                                             / _OSM_RA_TREND_W))
            else:
                s_osm_conv = max(0.0, min(1.0, (_OSM_RA_CONV_ZERO - conv_ratio)
                                             / _OSM_RA_CONV_W))
            s_osm_curv = max(0.0, min(1.0, (_OSM_RA_CURV_ZERO - late_curv_ratio)
                                         / _OSM_RA_CURV_W))
            s_osm_err = max(0.0, min(1.0, (_OSM_RA_ERR_ZERO - med_err)
                                        / _OSM_RA_ERR_W))
            oversmooth = max(0.0, min(1.0,
                s_osm_conv * (0.7 + 0.3 * max(s_osm_curv, s_osm_err))))
        else:
            # "engine": original absolute thresholds.  The distilled early-out
            # that used to live here was removed at the operator's request; it
            # changes nothing on its own because these thresholds still gate
            # every distilled profile to zero.
            if trend_resid is not None:
                s_osm_conv = max(0.0, min(1.0, (-3.2 - trend_resid) / 0.8))
            else:
                s_osm_conv = max(0.0, min(1.0, (0.03 - conv_ratio) / 0.03))
            s_osm_curv = max(0.0, min(1.0, (0.3 - late_curv_ratio) / 0.3))
            s_osm_err = max(0.0, min(1.0, (0.003 - med_err) / 0.003))
            oversmooth = max(0.0, min(1.0,
                s_osm_conv * (0.7 + 0.3 * max(s_osm_curv, s_osm_err))))

        return {
            "oversteer": round(oversteer, 4),
            "oversmooth": round(oversmooth, 4),
            "med_local_err_raw": round(med_err_raw, 6),
            "med_local_err_real": round(med_err, 6),
        }
    except Exception:
        return {"oversteer": 0.0, "oversmooth": 0.0}

def _safe_getattr(o, name, default=None):
    try:
        for cls in type(o).__mro__:
            if name in cls.__dict__:
                return getattr(o, name)
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict) and name in d:
            return d[name]
    except Exception:
        pass
    return default

def _looks_like_guider(o):
    try:
        if o is None or isinstance(o, (int, float, bool, str, bytes)):
            return False
        if not callable(_safe_getattr(o, "set_cfg", None)):
            return False
        cfg = _safe_getattr(o, "cfg", None)
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)):
            return False
        conds = _safe_getattr(o, "conds", None)
        if isinstance(conds, dict) and ("positive" in conds
                                        or "negative" in conds):
            return True
        if _safe_getattr(o, "positive", None) is not None:
            return True
        orig = _safe_getattr(o, "original_conds", None)
        if isinstance(orig, dict) and ("positive" in orig
                                       or "negative" in orig):
            return True
        return False
    except Exception:
        return False

def _guider_conds(g):
    pos = neg = None
    conds = _safe_getattr(g, "conds", None)
    if isinstance(conds, dict):
        pos = conds.get("positive")
        neg = conds.get("negative")
    if pos is None:
        orig = _safe_getattr(g, "original_conds", None)
        if isinstance(orig, dict):
            pos = orig.get("positive")
            neg = orig.get("negative")
    if pos is None:
        pos = _safe_getattr(g, "positive", None)
        if neg is None:
            neg = _safe_getattr(g, "negative", None)
    return pos, neg

def _iter_obj_graph(seed, max_nodes=64):
    seen = set()
    stack = [seed]
    count = 0
    while stack and count < max_nodes:
        o = stack.pop(0)
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        count += 1
        yield o
        fn = o
        if isinstance(fn, functools.partial):
            stack.append(fn.func)
            stack.extend(fn.args)
            try:
                stack.extend(fn.keywords.values())
            except Exception:
                pass
            fn = fn.func
        if callable(fn):
            f = _safe_getattr(fn, "__func__", fn)
            for cell in (_safe_getattr(f, "__closure__", None) or ()):
                try:
                    stack.append(cell.cell_contents)
                except (ValueError, TypeError):
                    pass
        if callable(o):
            self_obj = _safe_getattr(o, "__self__", None)
            if self_obj is not None:
                stack.append(self_obj)
        d = _safe_getattr(o, "__dict__", None)
        if isinstance(d, dict):
            for v in list(d.values())[:24]:
                stack.append(v)

def _find_guider_in_stack(limit=24):
    try:
        import sys as _sys
        f = _sys._getframe(1)
        n = 0
        while f is not None and n < limit:
            for v in list(f.f_locals.values()):
                if _looks_like_guider(v):
                    return v
            f = f.f_back
            n += 1
    except Exception:
        pass
    return None

def _resolve_gap_source(model, extra_args):
    pos = neg = None
    cfg = None
    if isinstance(extra_args, dict):
        pos = extra_args.get("cond")
        neg = extra_args.get("uncond")
        raw = extra_args.get("cond_scale", extra_args.get("cfg"))
        try:
            cfg = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            cfg = None
    guider = None
    for cand in _iter_obj_graph(model):
        if _looks_like_guider(cand):
            guider = cand
            break
    if guider is None:
        guider = _find_guider_in_stack()
    if guider is not None:
        gpos, gneg = _guider_conds(guider)
        if pos is None:
            pos, neg = gpos, gneg
        gcfg = _safe_getattr(guider, "cfg", None)
        if not isinstance(gcfg, bool) and isinstance(gcfg, (int, float)):
            if cfg is None:
                cfg = float(gcfg)
    return pos, neg, cfg, guider

def _build_gap_ctx(model, extra_args, patcher=None, positive=None,
                   negative=None):
    if patcher is None:
        patcher = _navigate_to_patcher(model)
    r_pos, r_neg, r_cfg, guider = _resolve_gap_source(model, extra_args)
    if positive is None:
        positive = r_pos
    if negative is None:
        negative = r_neg
    return {"patcher": patcher, "positive": positive, "negative": negative,
            "cfg": r_cfg, "guider": guider}

def _probe_effective_cfg(model, extra_args, gap_ctx=None):
    if isinstance(gap_ctx, dict):
        v = gap_ctx.get("cfg")
        if not isinstance(v, bool) and isinstance(v, (int, float)):
            return float(v)
    try:
        v = _safe_getattr(model, "cfg", None)
        if v is not None:
            return float(v)
    except Exception:
        pass
    # THE ENGINE'S MODEL IS KSamplerX0Inpaint, WHICH CARRIES NO `.cfg` -- IT WRAPS THE
    # GUIDER (U18/8092).  Without this hop every ENGINE-produced profile recorded
    # `probe_cfg: null`, so the adoption gate had nothing to compare and a CFG change
    # could silently reuse a profile measured at another guidance scale (U05/1361).
    # One or two attribute hops; no model calls.
    try:
        _inner = _safe_getattr(model, "inner_model", None)
        for _g in (model, _inner,
                   _safe_getattr(_inner, "inner_model", None)):
            if _g is None:
                continue
            _v = _guider_probe_cfg(_g)
            if _v is not None:
                return float(_v)
    except Exception:
        pass
    if isinstance(extra_args, dict):
        for key in ("cond_scale", "cfg"):
            try:
                if extra_args.get(key) is not None:
                    return float(extra_args[key])
            except (TypeError, ValueError):
                pass
    return None


def _resolve_distilled_override(mode, probe_cfg=None, heuristic=False):
    if mode is None:
        mode = "auto"
    if isinstance(mode, bool):
        return mode
    m = str(mode).strip().lower()
    if m in ("yes", "true", "on", "force", "1"):
        return True
    if m in ("no", "false", "off", "0"):
        return False
    if probe_cfg is not None:
        try:
            if float(probe_cfg) <= 1.0 + 1e-6:
                return True
        except (TypeError, ValueError):
            pass
    return bool(heuristic)

def _measure_guidance_gap(model, extra_args, xs, x0s, grid, ones,
                          gap_ctx=None):
    """Cond/uncond decomposition of the guidance gap, performed WITH
    ComfyUI's own machinery: the guider's cfg is temporarily pinned to 1.0
    (the chain's sampling_function/cfg_function then return the cond
    prediction) and to 0.0 (they return the uncond prediction).  Custom
    sampler_cfg_function hooks and the model's model_conds are honoured
    because the full sampling chain evaluates both calls -- no hand-rolled
    CFG, works for every model ComfyUI can sample."""
    try:
        probe_cfg = _probe_effective_cfg(model, extra_args, gap_ctx)
        if probe_cfg is not None and probe_cfg <= 1.0 + 1e-6:
            return None
        m = len(x0s)
        i = m // 2
        if i >= m:
            return None
        s_m = grid[i]
        x_at = xs[i]
        x0_cfg = x0s[i]
        guider = gap_ctx.get("guider") if isinstance(gap_ctx, dict) else None
        if guider is None:
            # No guider in reach -> skip the measurement.  Decomposing cond
            # vs uncond without ComfyUI's own guider would need hand-rolled
            # per-model CFG code, which PD-AOS no longer carries; the
            # consumers treat a missing gap as "unknown guidance strength".
            return None

        saved_cfg = getattr(guider, "cfg", None)
        x0_cond = x0_uncond = None
        try:
            guider.set_cfg(1.0)
            x0_cond = _probe_model_call(model, x_at, s_m * ones, extra_args)
            guider.set_cfg(0.0)
            x0_uncond = _probe_model_call(model, x_at, s_m * ones, extra_args)
        except Exception:
            x0_cond = x0_uncond = None
        finally:
            try:
                if saved_cfg is not None:
                    guider.set_cfg(saved_cfg)
            except Exception:
                pass

        if x0_cond is None or x0_uncond is None:
            return None
        return _gap_outputs(x0_cfg, x0_cond, x0_uncond)
    except Exception:
        return None

def _gap_outputs(x0_cfg, x0_cond, x0_uncond):
    uncond_norm = float(x0_uncond.float().norm().clamp_min(1e-8))
    strength = float((x0_cond - x0_uncond).float().norm()) / uncond_norm
    cond_norm = float(x0_cond.float().norm().clamp_min(1e-8))
    overshoot = float((x0_cfg - x0_cond).float().norm()) / cond_norm
    out = {"strength": round(strength, 4), "overshoot": round(overshoot, 4)}
    try:
        gd = (x0_cond.float() - x0_uncond.float())
        if gd.dim() == 5:
            gd = gd.mean(dim=2)          # collapse frames (broadcast back)
        if gd.dim() == 4 and gd.shape[0] == 1:
            c_n = int(gd.shape[1])
            h, w = gd.shape[-2:]
            if min(h, w) > 8:
                gd = F.adaptive_avg_pool2d(gd, (8, 8))
                h, w = gd.shape[-2:]
            nrm = float(gd.norm())
            if nrm > 1e-8:
                gdn = (gd / nrm)[0]
                out["prior_dir"] = {
                    "c": c_n, "h": int(h), "w": int(w),
                    "v": [round(float(t), 4) for t in gdn.reshape(-1)]}
        pm = gd.pow(2).mean(dim=1).sqrt()[0] if gd.dim() == 4 else \
            (x0_cond.float() - x0_uncond.float()).pow(2).mean(dim=1).sqrt()
        p90 = float(torch.quantile(pm.reshape(-1), 0.9))
        if p90 > 1e-8:
            pm = (pm / p90).clamp(0.0, 1.0)
            h, w = pm.shape[-2:]
            if min(h, w) > 16:
                pm = F.adaptive_avg_pool2d(pm.reshape(1, 1, h, w), (16, 16))[0, 0]
                h, w = pm.shape[-2:]
            out["prior"] = {"h": int(h), "w": int(w),
                            "v": [round(float(x), 3) for x in pm.reshape(-1)]}
    except Exception:
        pass
    return out

def _loosen_from_cfg(cfg):
    """The loosening multiplier FOR A GIVEN CFG, clamped to the derived range `[0, 8]`.

    WHY THIS EXISTS, AND WHY IT IS NOT THE MODULE GLOBAL ANY MORE (2026-10-04).
    This value used to be read from `_COND_RTOL_LOOSEN`, a module global set by `_apply_ab_modes`.  A
    provenance rig (`scratch_cond_loosen_provenance_rig.py`) then found the two functions that actually
    DERIVE a cond calibration -- `_run_cond_probe` and `_run_focused_cond_probe` -- never call that
    setter, while three OTHER entry points do.  Solving for the multiplier the recorded values imply
    showed `m = 1.0` where the run's cfg and the profile's own `calib_flags` stamp both said 0.0 and 5.0:
    **the stamp recorded one number and the arithmetic used another**, which is the defect class this
    project treats as unacceptable (a switch that reports a value while the code runs something else).
    Reading the cfg that is right there removes the class instead of patching one instance: there is no
    stale value to inherit because nothing is inherited.
    """
    try:
        v = float((cfg or {}).get("cond_rtol_loosen", ENGINE_DEFAULTS.get("cond_rtol_loosen", 1.5)))
    except (TypeError, ValueError):
        v = 1.5
    # The lower guard is CORRECTNESS, not taste: the multiply below clamps only its top, and the
    # downstream blend's clamp is skipped when the blend is not positive, so a negative multiplier would
    # reach cfg as a negative tolerance.
    return max(0.0, min(_COND_RTOL_LOOSEN_MAX, v))

def _derive_cond_calibrations(profile, cfg=None):
    calib = {}
    med_curv = profile.get("med_curv", 0.0)
    med_err_raw = float(profile.get("med_local_err", 0.0) or 0.0)
    gap_p = float(profile.get("log_gap_probe", 0.0) or 0.0)
    gap_r = float(profile.get("log_gap_real", 0.0) or 0.0)
    med_err = med_err_raw
    if med_err_raw > 0 and gap_p > 1e-9 and gap_r > 1e-9:
        med_err = med_err_raw * min(max((gap_r / gap_p) ** 2, 0.04), 1.0)
    if med_err > 0:
        calib["rtol"] = round(min(0.04, max(0.005, 3.0 * med_err)), 4)
    # THE INPUTS, BESIDE THE VALUE THEY PRODUCED -- see the model-path twin for why.  This path also
    # records the two settings that decide WHICH probe grid is measured, because they are read from
    # the run's own configuration here and are the values a reader would otherwise have to infer.
    calib["probe_inputs"] = {k: profile.get(k) for k in (
        "n_probe_steps", "probe_cfg", "probe_patcher", "space", "version",
        "is_distilled_eff", "med_local_err", "med_local_err_real", "med_curv",
        "log_gap_probe", "log_gap_real", "guard_sig_med", "guard_sig_p90",
        "whitepoint", "dk_scale", "raw3", "scale", "safe_fresh_frac",
        "probe_rank", "probe_rank_tried", "probe_rank_ladder") if k in profile}
    calib["probe_inputs"]["cfg_probe_steps"] = int((cfg or {}).get("probe_steps", 10))
    calib["probe_inputs"]["cfg_probe_resolution"] = int((cfg or {}).get("probe_resolution", 48))
    sig_med = float(profile.get("guard_sig_med", 0.0) or 0.0)
    sig_p90 = float(profile.get("guard_sig_p90", 0.0) or 0.0)
    if sig_med > 1e-9:
        r = sig_p90 / sig_med
        calib["guard_floor"] = round(
            min(0.5, max(0.05, 0.15 * math.sqrt(max(r, 0.25) / 4.0))), 4)
    # ORDER MATTERS: the gap-strength adjustment is applied BEFORE the wmax cap is derived,
    # because the cap is compared against `calib["rtol"]` -- the tolerance the run will
    # actually use.  Deriving the cap first and adjusting rtol afterwards (the shipped
    # order) made the logged `rtol` and the tol the cap consumed differ by the 1.5x factor:
    # anima log_00159 records `rtol` 0.0075 with `wmax_ab.tol` 0.005.
    gap = profile.get("gap")
    # THE GATE IS COUNTED SEPARATELY FROM THE ARMS, and that is a fix rather than decoration: the first
    # version counted only INSIDE `if strength < 0.5 and "rtol" in calib:`, so a run where the loosening
    # did not apply left the log COMPLETELY SILENT about why -- indistinguishable from a run where the
    # whole code path was never reached.  The operator's first A/B (12 runs, knob swept 0.0 .. 8.0) showed
    # exactly that: identical tolerances across every arm and an empty `ab_reach`, with nothing in the
    # record to say whether the profile had no `gap`, no derivable `rtol`, or was never re-derived.
    if gap is None:
        _ab_hit("cond.loosen.no_gap")
    else:
        _ab_hit("cond.loosen.entered")
        # ---------------------------------------------------------------------------------------
        # PATH A, APPLIED 2026-10-04 -- THE THRESHOLDS ARE GONE (register row 23, closed by the
        # operator: "we can close it and put it on record").  What used to be here:
        #
        #     if strength < 0.5 and "rtol" in calib:      rtol *= m      # TRUE on 214 of 214
        #     elif strength > 2.0 and "guard_floor" ...   gf  *= 0.7      # TRUE on 0 of 214
        #
        # `strength` is `||x0_cond - x0_uncond|| / ||x0_uncond||` and spans 0.0046..0.0861 over every
        # cond profile that carries one, so the first test never chose anything -- a CONSTANT wearing a
        # condition -- and the second was UNREACHABLE, with its target `guard_floor` not even a tunable
        # key (it is absent from `ENGINE_DEFAULTS`, so `_blend_set("guard_floor")` returns early on every
        # run: the arm could not have reached cfg even if it had fired).  The loosening is now gated on
        # the MEASUREMENT existing, which is what the branch was always for, and its SIZE is the knob.
        #
        # BEHAVIOUR-IDENTICAL, MEASURED rather than argued: `scratch_gap_strength_fix_rig.py` F3 shows
        # the change moves `rtol` on 0 of 891 cond profiles, and F2b shows the deleted arm was true on 0
        # of 214.  Nothing to A/B.
        # ---------------------------------------------------------------------------------------
        if "rtol" not in calib:
            _ab_hit("cond.loosen.no_rtol")
        _m = _loosen_from_cfg(cfg)
        if "rtol" in calib:
            _ab_hit("cond.rtol_loosen.x1.5_arm")
            if _m != 1.0:
                calib["rtol"] = round(min(0.04, calib["rtol"] * _m), 4)
                _ab_hit("cond.rtol_loosen.applied", round(_m, 4))
            else:
                _ab_hit("cond.rtol_loosen.off")
    if med_curv > 0:
        lc = math.log(max(med_curv, 1e-8) + 1.0)
        if lc < math.log(1.5):
            wb = 3.0
        elif lc > math.log(5.0):
            wb = 1.8
        else:
            t = (lc - math.log(1.5)) / (math.log(5.0) - math.log(1.5))
            wb = 3.0 - t * (3.0 - 1.8)
        _flags = _calib_flags(profile)
        _wb_shipped = round(max(1.5, min(2.5, wb)), 3)
        _wb_cap, _wb_info = _derived_wmax_cap(
            med_curv, profile.get("log_gap_probe"), calib.get("rtol"),
            dk=profile.get("dk_scale"),
            h_real=gap_r,
            budget=float(_flags.get("wmax_budget", 1.0) or 1.0))
        _use_derived_wb = bool(_flags.get("wmax_bound")) and _wb_cap is not None
        calib["wmax_base"] = (round(_wb_cap, 6) if _use_derived_wb else _wb_shipped)
        calib["wmax_shipped"] = _wb_shipped
        calib["wmax_bound"] = (round(_wb_cap, 6) if _wb_cap is not None else None)
        calib["wmax_ab"] = _wb_info
        if _flags.get("wmax_bound"):
            if _wb_cap is not None:
                _CALIB_AB["wmax_derived"] += 1
            else:
                _CALIB_AB["wmax_bound_unavailable"] += 1
        else:
            _CALIB_AB["wmax_shipped"] += 1
    if med_err > 0:
        calib["safe_eta"] = round(
            max(0.1, min(1.0, 0.7 / (med_err * 20.0 + 1.0))), 3)
    calib.update(_derive_anomaly_calibrations(profile))
    calib["calib_ab_counts"] = dict(_CALIB_AB)
    return calib

def _inherit_base_guard_stats(focused, base):
    if not isinstance(focused, dict) or not isinstance(base, dict):
        return focused
    bm = base.get("guard_sig_med")
    bp = base.get("guard_sig_p90")
    bf = base.get("fragility")
    if bm is None and bp is None and bf is None:
        return focused
    if (focused.get("guard_sig_med") == bm
            and focused.get("guard_sig_p90") == bp
            and focused.get("fragility") == bf):
        return focused
    out = dict(focused)
    if bm is not None:
        out["guard_sig_med"] = bm
    if bp is not None:
        out["guard_sig_p90"] = bp
    if bf is not None and out.get("fragility") is None:
        # the focused probe inherits the base probe's spatial fragility map
        # (the focused trajectory is too short to re-measure it reliably)
        out["fragility"] = bf
    try:
        out["trajectory_state"] = _compute_trajectory_state(out)
    except Exception:
        pass
    return out


def _carry_run_feedback(new_prof, old_prof):
    if not isinstance(new_prof, dict) or not isinstance(old_prof, dict):
        return new_prof
    fb = old_prof.get("dir_real")
    if isinstance(fb, list) and fb and "dir_real" not in new_prof:
        new_prof["dir_real"] = list(fb)
    return new_prof


def _detect_fragile_ranges(prof_m, prof_c):
    ranges = []
    for prof in (prof_m, prof_c):
        if not isinstance(prof, dict):
            continue
        eg = prof.get("endgame")
        if not isinstance(eg, dict):
            continue
        for lv in (eg.get("levels") or []):
            try:
                s = float(lv.get("s", 0))
                if s < 1e-6:
                    continue
                rec_q = lv.get("rec_q") or lv.get("rec_p50")
                if isinstance(rec_q, (list, tuple)) and len(rec_q) >= 2:
                    rec_p50 = float(rec_q[1])
                elif isinstance(rec_q, (int, float)):
                    rec_p50 = float(rec_q)
                else:
                    continue
                rec2_q = lv.get("rec2_q")
                s_attr = s
                if isinstance(rec2_q, (list, tuple)) and len(rec2_q) >= 2:
                    rec2_p50 = float(rec2_q[1])
                    if rec2_p50 > rec_p50:
                        rec_p50 = rec2_p50
                        # `rec2_q` IS MEASURED AT s/2 (U19/8402).  Attributing it to s
                        # moved the fragile band by half a step -- at 50 steps that is
                        # 1 % of the range, and the band is used to place the rescue
                        # window, so the shift is not cosmetic.  The value is kept as
                        # the max; only the sigma it is credited to changes.
                        s_attr = s * 0.5
                frag = min(1.0, max(0.0, rec_p50 / 0.06))
                if frag > 0.15:  # threshold for "worth focusing on"
                    ranges.append((s_attr, frag))
            except (TypeError, ValueError, IndexError):
                continue
    ranges.sort(key=lambda r: -r[1])
    if len(ranges) >= 2:
        sigmas_sorted = sorted([r[0] for r in ranges])
        return [(sigmas_sorted[0], sigmas_sorted[-1],
                 max(r[1] for r in ranges))]
    elif len(ranges) == 1:
        s, f = ranges[0]
        return [(s * 0.7, s * 1.3, f)]
    return []

def _rescue_window_from_fragile(sigmas, fragile_ranges):
    """Map measured endgame-fragile sigma ranges onto the detail-rescue
    window (kn space).  Returns (start, full) or (None, None).

    The rescue envelope ramps in between `start` and `full` (fractions of the
    run); by default those are fixed at 0.55/0.72 no matter WHERE the model
    actually misinterprets structured state.  The endgame probe measured
    exactly that (recovery error after re-noising the final x0 to lower
    sigmas), so place the ramp over the measured fragile band instead.
    """
    try:
        sv = [float(v) for v in (sigmas if sigmas is not None else [])
              if float(v) > 1e-6]
        if len(sv) < 4 or not fragile_ranges:
            return None, None
        n_steps = len(sv)  # engine kn = (step_index + 1) / n_steps

        def _kn_of(sigma):
            for i, v in enumerate(sv):
                if v <= sigma:
                    return (i + 1) / float(n_steps)
            return 1.0

        best = None
        for s_lo, s_hi, frag in fragile_ranges:
            try:
                k0 = _kn_of(float(s_hi))   # larger sigma = earlier step
                k1 = _kn_of(float(s_lo))   # smaller sigma = later step
            except (TypeError, ValueError):
                continue
            if k1 - k0 < 0.02:
                continue
            if best is None or float(frag) > best[2]:
                best = (k0, k1, float(frag))
        if best is None:
            return None, None
        start = min(max(best[0], 0.25), 0.80)
        full = min(max(best[1], start + 0.05), 0.90)
        return round(start, 3), round(full, 3)
    except Exception:
        return None, None

def _build_focused_grid(sigmas, fragile_ranges, K_base=4, K_max=8,
                        distilled_mode=False, cfg=None):
    base = _probe_grid(sigmas, K_base, distilled_mode) or []
    if not base:
        return []
    extra = []
    for s_lo, s_hi, frag in fragile_ranges:
        n_extra = min(3, max(1, int(frag * 3)))
        s_lo = max(s_lo, 1e-6)
        s_hi = max(s_hi, s_lo * 1.1)
        for i in range(n_extra):
            t = (i + 0.5) / n_extra
            s_extra = s_hi * (s_lo / s_hi) ** t
            extra.append(s_extra)
    grid = sorted(set(base + extra), reverse=True)
    if len(grid) > K_max:
        grid = grid[:K_max]
    # PROBE REACH IS APPLIED **AFTER** THE CAP, AND THAT ORDER IS THE WHOLE POINT.  `grid[:K_max]`
    # keeps the TOP points and drops the lowest, so extending the base grid before the cap would
    # have had its new low-sigma points truncated away -- the switch would read ON in the cfg and
    # do nothing, which is the silent-inert failure rule 14 exists for.  Appending here also leaves
    # the focused grid's fragile-range extras exactly where they were.
    if isinstance(cfg, dict):
        grid, _ = _probe_reach_extend(grid, sigmas, cfg)
    return grid

def _midpoint_local_errors(xs, x0s, grid, gap_curve, real_gap_curve):
    """The PROBE's local-error definition, shared by the focused path.

    `_run_probe` and `_run_cond_probe` compute the error of a midpoint prediction
    against the exact step.  The focused path used to store the consecutive x0
    jump here, which is a different quantity and roughly an order larger; this
    helper is the one implementation so a future edit cannot fork it again.
    """
    local_errs = []
    local_errs_real = []
    m = min(len(xs), len(x0s), len(grid))
    for i in range(1, m - 1):
        s_i = float(grid[i])
        s_n = float(grid[i + 1])
        s_p = float(grid[i - 1])
        if s_n <= 1e-8 or s_i <= 1e-8:
            continue
        x1_i = exact_step(xs[i], x0s[i], s_i, s_n)
        ls = math.log(max(s_i, 1e-8))
        ls2 = math.log(max(s_n, 1e-8))
        lsp = math.log(max(s_p, 1e-8))
        lmid = 0.5 * (ls + ls2)
        t = (lmid - ls) / max(lsp - ls, 1e-12)
        x0_mid = x0s[i] + t * (x0s[i - 1] - x0s[i])
        x_pred_i = exact_step(xs[i], x0_mid, s_i, s_n)
        local_errs.append(float((x_pred_i - x1_i).float().norm()
                                / x1_i.float().norm().clamp_min(1e-8)))
    _probe_gaps = gap_curve[1:1 + len(local_errs)]
    _real_gaps = real_gap_curve[1:1 + len(local_errs)]
    for _e, _gp, _gr in zip(local_errs, _probe_gaps, _real_gaps):
        if _gp > 1e-9 and _gr > 1e-9:
            scale = min(max((_gr / _gp) ** 2, 0.04), 1.0)
            local_errs_real.append(_e * scale)
        else:
            local_errs_real.append(_e)
    return local_errs, local_errs_real


def _run_focused_cond_probe(model, x, sigmas, extra_args, cfg, gap_ctx=None,
                            callback=None, fragile_ranges=None,
                            focused_grid=None, base_profile=None):
    try:
        _pf_patcher = gap_ctx.get("patcher") if isinstance(gap_ctx, dict) else None
        if _pf_patcher is not None:
            _pf_dev = _probe_call_device(_pf_patcher)
            if x.device != _pf_dev:
                x = x.to(_pf_dev)
        pres = max(8, int(cfg.get("cond_probe_resolution", 16)))
        B = 1
        C = x.shape[1]
        if B < 1 or C < 1:
            return None
        shape = _probe_latent_shape(ref_x=x, pres=pres, batch=B,
                                    site="engine/_run_focused_cond_probe")
        px = torch.randn(shape, device=x.device, dtype=torch.float32,
                         generator=_probe_gen(x)).to(x.dtype)
        ones = px.new_ones([B])
        grid = focused_grid
        if not grid or len(grid) < 3:
            return None
        grid = sorted(grid, reverse=True)
        # THE GRID ARRIVES ALREADY EXTENDED (`_build_focused_grid` applies the reach after its own
        # K_max cap), so the calibration band is DERIVED here rather than returned by the builder --
        # `_probe_calib_band` is the one place that rule lives, and rig_probe_reach.py checks it
        # agrees with the `n_extra` the builder reports.
        _n_cal, _band_floor = _probe_calib_band(grid, sigmas, cfg)
        _reach_extra = max(0, len(grid) - _n_cal)
        px = px * float(grid[0])
        pos_all = [v for v in _sigmas_as_floats(sigmas) if v > 1e-6]
        gap_curve = [abs(math.log(max(grid[i], 1e-8))
                         - math.log(max(grid[i + 1], 1e-8)))
                     for i in range(len(grid) - 1)]
        real_gap_curve = []
        if len(pos_all) >= 2:
            pos_idx = []
            _j = 0
            for gv in grid:
                while _j < len(pos_all) and pos_all[_j] > gv * (1.0 + 1e-9):
                    _j += 1
                pos_idx.append(min(_j, len(pos_all) - 1))
            real_gaps = [abs(math.log(max(pos_all[t], 1e-8))
                             - math.log(max(pos_all[t + 1], 1e-8)))
                         for t in range(len(pos_all) - 1)]
            for i in range(len(grid) - 1):
                a, b = pos_idx[i], pos_idx[i + 1]
                if b <= a:
                    lo, hi = max(a - 1, 0), min(a + 1, len(pos_all) - 1)
                else:
                    lo, hi = a, b
                seg = real_gaps[lo:hi]
                real_gap_curve.append(sum(seg) / len(seg) if seg
                                      else (sum(real_gaps) / len(real_gaps)))
        else:
            real_gap_curve = list(gap_curve)
        space = _detect_space(torch.tensor(grid), cfg.get("noise_space", "auto"))
        xs = []
        x0s = []
        cur = px
        _dev_retry_done = False
        for i, s in enumerate(grid):
            _progress(callback, "cond_probe", "progress", i, len(grid) + 4,
                      "PD-AOS focused probe: step %d/%d" % (i + 1, len(grid)))
            x0 = None
            try:
                x0 = _probe_model_call(model, cur, s * ones, extra_args)
            except Exception as e:
                if (not _dev_retry_done and _pf_patcher is not None
                        and _is_device_error(e)):
                    _dev_retry_done = True
                    try:
                        _dev = _probe_call_device(_pf_patcher)
                        if cur.device != _dev:
                            cur = cur.to(_dev)
                        if ones.device != _dev:
                            ones = ones.to(_dev)
                        x0 = _probe_model_call(model, cur, s * ones, extra_args)
                        logging.info("[PD-AOS-focused] recovered from a "
                                     "device mismatch at s=%.4g (model "
                                     "re-secured on %s)", s, _dev)
                    except Exception:
                        x0 = None
                if x0 is None:
                    import traceback as _tb
                    logging.warning("[PD-AOS-focused] model call failed at s=%.4g: %s", s, e)
                    logging.warning("[PD-AOS-focused] failure traceback:\n%s", _tb.format_exc())
                    logging.warning("[PD-AOS-focused] failure context: %s",
                                    _probe_failure_diagnostics(model, cur, ones, extra_args, gap_ctx))
                    break
            xs.append(cur)
            x0s.append(x0.detach())
            if i < len(grid) - 1:
                cur = exact_step(cur, x0, s, grid[i + 1])
        m = len(x0s)
        if m < 3:
            return None
        norms = [float(t.float().norm().clamp_min(1e-8)) for t in x0s]
        jumps = []
        for i in range(m - 1):
            d = float((x0s[i + 1] - x0s[i]).float().norm())
            jumps.append(d / max(max(norms[i], norms[i + 1]), 1e-8))
        curvs = []
        for i in range(1, m - 1):
            d1 = (x0s[i + 1] - x0s[i]).float().norm()
            d2 = (x0s[i] - x0s[i - 1]).float().norm()
            curvs.append(abs(float(d1) - float(d2)) / max(float(d1), float(d2), 1e-8))
        local_errs, local_errs_real = _midpoint_local_errors(
            xs, x0s, grid, gap_curve, real_gap_curve)
        guard_sigs = []
        for i in range(m - 1):
            sig = _jump_signal(x0s[i + 1], x0s[i], xs[i])
            guard_sigs.append(sig)
        gap = None
        probe_cfg = None
        # THE CALIBRATION BAND, as in `_run_probe`: `med_jump`/`med_curv` below are what the
        # `calib` derivations read, so they must not move because the probe looked further.  The
        # curves keep the full extent and the scalars stay on the shipped band.
        if _reach_extra > 0:
            _jcal = jumps[:max(0, len(jumps) - _reach_extra)] or jumps
            _ccal = curvs[:max(0, len(curvs) - _reach_extra)] or curvs
        else:
            _jcal, _ccal = jumps, curvs
        if gap is None and gap_ctx is not None and bool(cfg.get("cond_probe_gap", True)):
            try:
                probe_cfg = float(cfg.get("cond_probe_cfg"))
            except (TypeError, ValueError):
                probe_cfg = None
            if probe_cfg is not None and probe_cfg > 1.0 + 1e-6:
                gap = _measure_guidance_gap(model, extra_args, xs, x0s,
                                            grid, ones, gap_ctx=gap_ctx)
        _fp_cfg = _probe_effective_cfg(model, extra_args, gap_ctx)
        if _fp_cfg is None:
            _fp_cfg = probe_cfg
        is_distilled_eff = _resolve_distilled_override(
            cfg.get("cond_probe_is_distilled", "auto"), _fp_cfg, False)
        endgame = None
        if isinstance(base_profile, dict):
            endgame = base_profile.get("endgame")
        if endgame is None and bool(cfg.get("probe_endgame", True)):
            _progress(callback, "cond_probe", "progress", len(grid), len(grid) + 4,
                      "PD-AOS focused probe: endgame")
            endgame = _probe_endgame(model, extra_args, x0s[-1], grid[-1],
                                     pos_all, ones, space, _probe_gen(x))
        # Computed ONCE here so the profile dict below can carry BOTH the curve
        # and its per-term breakdown without walking the trajectory twice.  The
        # breakdown matters: measured on the real corpus the probe curve lands
        # ~10x BELOW the run's (probe p50 0.0000 vs a run p50 of 0.27-0.41), and
        # the gate is a 4-way AND -- without the per-conjunct pass rates that
        # failure cannot be attributed to any one term.
        # THE RUN'S RESCUE INPUTS, NOT A PLACEHOLDER.  The run's floor is
        # `rescue.floor * sigma` per step and its gate is fed only from
        # `rescue_start`; both are passed so the probe predicts the gate the run
        # actually evaluates (D3, U08).
        _cov_floor = float(cfg.get("rescue_floor", 0.02))
        _cov_start = float(cfg.get("rescue_start", 0.55))
        _cov_r = _lf_rescue_cov_curve(
            xs, x0s, grid, _eg_n_from_err(_endgame_eg_err(endgame)),
            _cov_floor, _cov_start)
        profile = {
            "version": 8,
            "cond_sig": cfg.get("cond_probe_sig"),
            "space": space,
            "n_probe_steps": m,
            # WHAT THIS PROFILE WAS MEASURED AT, so the artefact describes itself.  `n_probe_steps` was
            # already here; the RESOLUTION was not, which is why a run asking for 16 px consuming a 32 px
            # measurement could not be seen in the log at all.  `pres` is the EFFECTIVE value.
            "n_probe_res": int(pres),
            # A COND MEASUREMENT, SAID SO.  `_profile_serves_model_probe` refuses anything
            # that is not `probe_src == "model"`, because this profile is legitimately
            # written into `_PROFILE_BY_MODEL` and can be promoted into a model `|s:` slot;
            # declaring itself here is what makes that refusal possible.  `probe_device` is
            # what the probe measured ON (the cond path may have moved its latent to the
            # device `_probe_call_device` chose).
            "probe_src": "cond",
            "probe_device": str(getattr(x, "device", "?")),
            "sigma_curve": [round(v, 6) for v in grid],
            # THE REACH IN FORCE, stamped rather than inferred (rule 14).  `probe_band_floor` is
            # where the SHIPPED grid stopped and every scalar aggregate was computed;
            # `probe_reach_floor` is the lowest sigma actually measured.  A reader cannot tell a
            # 0.1-reach profile from a 0.01-reach one without both, and they are not comparable.
            "probe_reach_frac": round(float(cfg.get("probe_smin_frac",
                                                     _PROBE_SMIN_FLOOR)), 6),
            "probe_band_floor": round(float(_band_floor), 6),
            "probe_reach_floor": round(float(min(grid)), 6),
            "probe_reach_n_extra": int(_reach_extra),
            "jump_curve": [round(v, 6) for v in jumps],
            "gap_curve": [round(v, 6) for v in gap_curve],
            "real_gap_curve": [round(v, 6) for v in real_gap_curve],
            "local_err_curve": [round(v, 6) for v in local_errs],
            "med_local_err_real": (sorted(local_errs_real)[len(local_errs_real) // 2]
                                   if local_errs_real else 0.0),
            # WHICH ESTIMATOR PRODUCED THESE NUMBERS (U16/6906).  `pct_derived` changes what
            # the percentiles ARE, so it is a MEASUREMENT input, not a calibration input:
            # stamped here so a reader can tell the two estimators apart without trusting a
            # key name, and left out of `calib_flags` for the same reason.
            "pct_derived": bool(cfg.get("pct_derived", False)),
            "curv_curve": [round(v, 6) for v in curvs],
            # THE SIGMA EACH CURVATURE POINT IS ANCHORED AT.  The convention is documented at
            # `aflops.py:7687` -- `curv_curve[k]` is the curvature at `sigma_curve[k+1]`, curves carry
            # m-2 entries against the grid's m -- and a reader who does not know it will mis-align every
            # derivation, as a rig of mine did this session.  Logging the anchors removes the need to
            # know it.
            "curv_sigma": [round(float(v), 6) for v in grid[1:1 + len(curvs)]],
            # PROBE-SIDE COVERAGE CURVE -- see the note on the model-probe copy.
            # This is the FOCUSED cond probe, i.e. the per-prompt one that runs
            # over the fragile band, so its curve is the most directly comparable
            # to what the run's rescue gate will see late in sampling.
            "cov_curve": (_cov_r or {}).get("curve"),
            "cov_curve_terms": (_cov_r or {}).get("terms"),
            # The DERIVED variant (ref_drop = k*du, k measured from this very
            # trajectory) alongside the shipped per-step curve.  Both are emitted
            # so the next real runs can decide which one actually tracks the run:
            # synthetic fields could not, see scratch_ref_decay_rig.py.
            "cov_curve_kdu": (_cov_r or {}).get("curve_kdu"),
            "cov_curve_k": (_cov_r or {}).get("k"),
            # WHICH probe model-call path ran, and with what arguments.  139 of
            # 224 profiles are frozen (model returns its input); this is what will
            # identify the cause on the next frozen run.
            "probe_call": dict(_PROBE_CALL_DIAG),
            "cov_curve_meta": _lf_rescue_cov_meta(
                x0s[-1] if x0s else None, _endgame_eg_err(endgame),
                _cov_floor, _cov_start),
            # THIRD PERCENTILE IMPLEMENTATION, and the one that actually produced the frozen
            # guard_floor: this focused/cond builder used raw indexing, so the `_pct` toggle
            # did NOT reach it.  At n_guard = 2, `len//2` and `int(n*0.9)` are both 1 -- the
            # same element -- which is the collision.  Now it honours the same toggle, and
            # logs both estimators so one run shows the delta.
            "med_jump": _probe_pct(_jcal, 0.5,
                                   derived=bool(cfg.get("pct_derived", False))),
            "med_curv": _probe_pct(_ccal, 0.5,
                                   derived=bool(cfg.get("pct_derived", False))),
            # the per-order derivative scale this trajectory can certify (see the model
            # builder for why it exists); a 3-point grid certifies only k = 2.
            "dk_scale": _measure_dk_scale(
                x0s, (sum(gap_curve) / len(gap_curve) if gap_curve else 0.0),
                return_series=True)[0],
            "dk_series": _measure_dk_scale(
                x0s, (sum(gap_curve) / len(gap_curve) if gap_curve else 0.0),
                return_series=True)[1],
            # the A/B flags this profile was built under -- see the model builder for why
            # this must be here or the `wmax_bound` arm stays inert.
            _CALIB_FLAGS_KEY: _calib_inputs(cfg),
            "is_distilled_eff": is_distilled_eff,
            "med_local_err": _probe_pct(local_errs, 0.5,
                                        derived=bool(cfg.get("pct_derived", False))),
            "guard_sig_med": _probe_pct(guard_sigs, 0.5,
                                        derived=bool(cfg.get("pct_derived", False))),
            "guard_sig_p90": _probe_pct(guard_sigs, 0.9,
                                        derived=bool(cfg.get("pct_derived", False))),
            "guard_sig_med_shipped": _probe_pct(guard_sigs, 0.5, False),
            "guard_sig_p90_shipped": _probe_pct(guard_sigs, 0.9, False),
            "guard_sig_med_derived": _probe_pct(guard_sigs, 0.5, True),
            "guard_sig_p90_derived": _probe_pct(guard_sigs, 0.9, True),
            "log_gap_probe": (sum(gap_curve) / len(gap_curve) if gap_curve else 0.0),
            "log_gap_real": (sum(real_gap_curve) / len(real_gap_curve)
                             if real_gap_curve else 0.0),
            "gap": gap,
            "probe_cfg": probe_cfg if gap_ctx is not None else None,
            "n_jump": len(jumps),
            "n_err": len(local_errs),
            "n_guard": len(guard_sigs),
            "endgame": endgame,
            "whitepoint": _whitepoint_from_x0s(x0s),
            "focused": True,
            "fragile_ranges": [(round(s_lo, 6), round(s_hi, 6), round(f, 3))
                               for s_lo, s_hi, f in (fragile_ranges or [])],
        }
        profile["trajectory_state"] = _compute_trajectory_state(profile)
        # THE CFG GOES IN, so the loosening multiplier used here is the one THIS run's cfg names -- see
        # `_loosen_from_cfg` for the stamp-vs-arithmetic divergence this closes.
        profile["calib"] = _derive_cond_calibrations(profile, cfg)
        return profile
    except Exception as e:
        logging.warning("[PD-AOS-focused] probe failed: %s", e)
        return None

def _run_cond_probe(model, x, sigmas, extra_args, cfg, gap_ctx=None,
                    callback=None):
    try:
        _pf_patcher = gap_ctx.get("patcher") if isinstance(gap_ctx, dict) else None
        if _pf_patcher is not None:
            _pf_dev = _probe_call_device(_pf_patcher)
            if x.device != _pf_dev:
                x = x.to(_pf_dev)
        pres = max(8, int(cfg.get("cond_probe_resolution", 16)))
        K = max(3, int(cfg.get("cond_probe_steps", 4)))
        B = 1
        C = x.shape[1]
        if B < 1 or C < 1:
            return None
        grid = _probe_grid(sigmas, K, _probe_distilled_mode(cfg))
        if grid is None:
            return None
        # PROBE REACH, cond path -- same contract as the model probe: the extension is APPENDED, and
        # every scalar aggregate below stays on the shipped band (`m - _reach_extra`).
        _band_floor = min(float(v) for v in grid)
        grid, _reach_extra = _probe_reach_extend(grid, sigmas, cfg)
        pos_all = [v for v in _sigmas_as_floats(sigmas) if v > 1e-6]
        gap_curve = [abs(math.log(max(grid[i], 1e-8))
                         - math.log(max(grid[i + 1], 1e-8)))
                     for i in range(len(grid) - 1)]
        real_gap_curve = []
        if len(pos_all) >= 2:
            pos_idx = []
            _j = 0
            for gv in grid:
                while _j < len(pos_all) and pos_all[_j] > gv * (1.0 + 1e-9):
                    _j += 1
                pos_idx.append(min(_j, len(pos_all) - 1))
            real_gaps = [abs(math.log(max(pos_all[t], 1e-8))
                             - math.log(max(pos_all[t + 1], 1e-8)))
                         for t in range(len(pos_all) - 1)]
            for i in range(len(grid) - 1):
                a, b = pos_idx[i], pos_idx[i + 1]
                if b <= a:
                    lo, hi = max(a - 1, 0), min(a + 1, len(pos_all) - 1)
                else:
                    lo, hi = a, b
                seg = real_gaps[lo:hi]
                real_gap_curve.append(sum(seg) / len(seg) if seg
                                      else (sum(real_gaps) / len(real_gaps)))
        else:
            real_gap_curve = list(gap_curve)
        space = _detect_space(torch.tensor(grid), cfg.get("noise_space", "auto"))
        _rank_ladder = _probe_rank_ladder(_pf_patcher, x)
        _rank_used = None
        _ladder_i = 0
        while True:
            _rank = _rank_ladder[min(_ladder_i, len(_rank_ladder) - 1)]
            shape = _probe_latent_shape(ref_x=x, pres=pres, batch=B,
                                        rank=_rank,
                                        site="engine/_run_cond_probe")
            px = torch.randn(shape, device=x.device, dtype=torch.float32,
                             generator=_probe_gen(x)).to(x.dtype)
            ones = px.new_ones([B])
            px = px * float(grid[0])
            xs = []
            x0s = []
            cur = px
            _dev_retry_done = False
            _shape_retry = False
            for i, s in enumerate(grid):
                _progress(callback, "cond_probe", "progress", i, len(grid) + 4,
                          "PD-AOS cond probe: step %d/%d" % (i + 1, len(grid)))
                x0 = None
                try:
                    x0 = _probe_model_call(model, cur, s * ones, extra_args)
                except Exception as e:
                    if (not _dev_retry_done and _pf_patcher is not None
                            and _is_device_error(e)):
                        _dev_retry_done = True
                        try:
                            _dev = _probe_call_device(_pf_patcher)
                            if cur.device != _dev:
                                cur = cur.to(_dev)
                            if ones.device != _dev:
                                ones = ones.to(_dev)
                            x0 = _probe_model_call(model, cur, s * ones, extra_args)
                            logging.info("[PD-AOS-cond-probe] recovered from a "
                                         "device mismatch at s=%.4g (model "
                                         "re-secured on %s)", s, _dev)
                        except Exception:
                            x0 = None
                    if x0 is None:
                        if (_ladder_i + 1 < len(_rank_ladder)
                                and _is_shape_error(e)):
                            _shape_retry = True
                            logging.info(
                                "[PD-AOS-cond-probe] model rejected a %d-D "
                                "probe latent at s=%.4g (%s); re-running the "
                                "trajectory at %d-D",
                                cur.dim(), s, e, _rank_ladder[_ladder_i + 1])
                            break
                        import traceback as _tb
                        logging.warning("[PD-AOS-cond-probe] model call failed at s=%.4g: %s", s, e)
                        logging.warning("[PD-AOS-cond-probe] failure traceback:\n%s", _tb.format_exc())
                        logging.warning("[PD-AOS-cond-probe] failure context: %s",
                                        _probe_failure_diagnostics(model, cur, ones, extra_args, gap_ctx))
                        break
                xs.append(cur)
                x0s.append(x0.detach())
                if i < len(grid) - 1:
                    cur = exact_step(cur, x0, s, grid[i + 1])
            if not _shape_retry:
                _rank_used = _rank
                break
            _ladder_i += 1
        m = len(x0s)
        if m < 3:
            return None
        if _rank_used is not None:
            _remember_probe_rank(_pf_patcher, _rank_used)
        norms = [float(t.float().norm().clamp_min(1e-8)) for t in x0s]
        jumps = []
        for i in range(m - 1):
            d = float((x0s[i + 1] - x0s[i]).float().norm())
            jumps.append(d / max(max(norms[i], norms[i + 1]), 1e-8))
        dist_on = bool(cfg.get("probe_distributions", True))
        jump_q = []
        ojf_curve = []
        band_curve = []
        jrel_curve = []
        dabs_curve = []
        if dist_on:
            _qt = torch.tensor([0.1, 0.5, 0.9])
            for i in range(m - 1):
                jpx = (x0s[i + 1] - x0s[i]).float().pow(2).mean(dim=1).sqrt()
                sc = max(float(x0s[i].float().pow(2).mean().sqrt()),
                         float(x0s[i + 1].float().pow(2).mean().sqrt()))
                jpx_n = (jpx / max(sc, 1e-8)).reshape(-1)
                qs = torch.quantile(jpx_n, _qt.to(jpx_n.device))
                jump_q.append([round(float(v), 6) for v in qs])
                p50 = float(qs[1])
                if p50 > 1e-9:
                    ojf_curve.append(round(
                        float((jpx_n > 2.5 * p50).float().mean()), 4))
                try:
                    band_curve.append([round(v, 4) for v in
                                       _lf_band_fractions(jpx)])
                except Exception:
                    band_curve.append([0.33, 0.34, 0.33])
                # runtime-scale per-pixel relative jump: the SAME quantity the
                # escape's stagnation detector uses (median jump / median
                # per-pixel magnitude).  Stored so the escape threshold can be
                # derived from the model's own measured jump level.
                xm = x0s[i].float().pow(2).mean(dim=1).sqrt()
                jrel_curve.append(round(
                    float(jpx.median().clamp_min(0.0)
                          / xm.median().clamp_min(1e-8)), 6))
            for i in range(m):
                x4 = x0s[i].float().reshape(-1, x0s[i].shape[1],
                                            *x0s[i].shape[-2:])
                hp = x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)
                dabs_curve.append(round(
                    float(hp.norm() / x4.norm().clamp_min(1e-8)), 6))
        med_jump_rel = (sorted(jrel_curve)[len(jrel_curve) // 2]
                        if jrel_curve else None)
        med_detail_abs = (sorted(dabs_curve)[len(dabs_curve) // 2]
                          if dabs_curve else None)
        dir_cons = []
        if bool(cfg.get("probe_dir_cons", True)):
            for i in range(1, m - 1):
                d1 = (x0s[i] - x0s[i - 1]).float()
                d2 = (x0s[i + 1] - x0s[i]).float()
                num = (d1 * d2).sum(dim=1)
                den = d1.pow(2).sum(dim=1).sqrt() * d2.pow(2).sum(dim=1).sqrt()
                okm = den > 1e-8
                if bool(okm.any()):
                    dir_cons.append(round(float((num[okm] / den[okm]).mean()), 4))
        dir_cons_valid = []
        if dir_cons and len(gap_curve) >= 2:
            _med_g = sorted(gap_curve)[len(gap_curve) // 2]
            for _k in range(len(dir_cons)):
                _ga, _gb = gap_curve[_k], gap_curve[_k + 1]
                if (_med_g / 3.0) <= _ga <= 3.0 * _med_g and \
                        (_med_g / 3.0) <= _gb <= 3.0 * _med_g:
                    dir_cons_valid.append(dir_cons[_k])
            if len(dir_cons_valid) < 2:
                dir_cons_valid = []
        # oscillation onset: the sigma where the x0 trajectory first reverses
        # direction (dir_cons < 0) -- where the late noise fade should anchor
        # (the noise stops helping and starts oscillating the trajectory).
        osc_sigma = None
        for _k, _dc in enumerate(dir_cons):
            if _dc < 0 and _k + 1 < len(grid):
                osc_sigma = float(grid[_k + 1])
                break
        curvs = []
        for i in range(1, m - 1):
            dl_prev = math.log(max(grid[i - 1], 1e-8)) - math.log(max(grid[i], 1e-8))
            dl_next = math.log(max(grid[i], 1e-8)) - math.log(max(grid[i + 1], 1e-8))
            dl = 0.5 * (abs(dl_prev) + abs(dl_next))
            if dl < 1e-6:
                continue
            dd = (x0s[i + 1] - 2.0 * x0s[i] + x0s[i - 1]).float().norm()
            curvs.append(float(dd) / (dl * dl)
                         / max(max(norms[i - 1], norms[i], norms[i + 1]), 1e-8))
        local_errs = []
        err_q = []
        guard_sigs = []
        frag_acc = None
        for i in range(1, m - 1):
            s_i = float(grid[i])
            s_n = float(grid[i + 1])
            s_p = float(grid[i - 1])
            if s_n <= 1e-8 or s_i <= 1e-8:
                continue
            x1_i = exact_step(xs[i], x0s[i], s_i, s_n)
            ls = math.log(max(s_i, 1e-8))
            ls2 = math.log(max(s_n, 1e-8))
            lsp = math.log(max(s_p, 1e-8))
            lmid = 0.5 * (ls + ls2)
            t = (lmid - ls) / (lsp - ls)
            x0_mid = x0s[i] + t * (x0s[i - 1] - x0s[i])
            x_pred_i = exact_step(xs[i], x0_mid, s_i, s_n)
            local_errs.append(float((x_pred_i - x1_i).float().norm()
                                    / x1_i.float().norm().clamp_min(1e-8)))
            if dist_on:
                epx = ((x_pred_i - x1_i).float().pow(2).mean(dim=1).sqrt()
                       / float(x1_i.float().pow(2).mean().sqrt().clamp_min(1e-8)))
                eq = torch.quantile(epx.reshape(-1),
                                    torch.tensor([0.1, 0.5, 0.9],
                                                 device=epx.device))
                err_q.append([round(float(v), 6) for v in eq])
                # per-pixel fragility: max local error over the trajectory
                if frag_acc is None:
                    frag_acc = epx.detach().float().unsqueeze(1)
                else:
                    frag_acc = torch.maximum(frag_acc, epx.detach().float().unsqueeze(1))
        local_errs_real = []
        _err_probe_gaps = gap_curve[1:1 + len(local_errs)]
        _err_real_gaps = real_gap_curve[1:1 + len(local_errs)]
        for _e, _gp, _gr in zip(local_errs, _err_probe_gaps, _err_real_gaps):
            if _gp > 1e-9 and _gr > 1e-9:
                local_errs_real.append(
                    _e * min(max((_gr / _gp) ** 2, 0.04), 1.0))
            else:
                local_errs_real.append(_e)
        for i in range(1, m):
            guard_sigs.append(float(_jump_signal(x0s[i], x0s[i - 1], xs[i])))
        g_pos = [max(float(v), 1e-8) for v in grid]
        gap_probe = (sum(abs(math.log(g_pos[i]) - math.log(g_pos[i + 1]))
                         for i in range(len(g_pos) - 1))
                     / max(1, len(g_pos) - 1))
        pos_all = [v for v in _sigmas_as_floats(sigmas) if v > 1e-6]
        gap_real = 0.0
        if len(pos_all) > 1:
            gap_real = (sum(abs(math.log(max(pos_all[i], 1e-8))
                                - math.log(max(pos_all[i + 1], 1e-8)))
                            for i in range(len(pos_all) - 1))
                        / (len(pos_all) - 1))

        def _pct(arr, p):
            # The shipped body was `a[min(len(a)-1, int(p*len(a)))]`, which COLLIDES for
            # n <= 2 (int(0.5*2) == int(0.9*2) == 1, so p90 IS the median -- this is what
            # made guard_floor a constant 0.075) and is biased at every n (at n = 10 it
            # returns the maximum for p = 0.9).  Toggle: cfg["pct_derived"].  Counted for
            # reachability: a toggle wired to nothing looks exactly like one whose effect
            # is below the noise floor.
            _pd = bool(cfg.get("pct_derived", False)) if isinstance(cfg, dict) else False
            _CALIB_AB["pct_derived" if _pd else "pct_shipped"] += 1
            return _probe_pct(arr, p, derived=_pd)

        # SAME CALIBRATION-BAND RULE AS THE MODEL PROBE (`_run_probe`): the reach extension is a
        # suffix of `grid`/`jumps`/`curvs`, so the scalars stay on the shipped band and only the
        # curves carry the new low-sigma points.  See `_probe_reach_extend`.
        if _reach_extra > 0:
            _jcal = jumps[:max(0, len(jumps) - _reach_extra)] or jumps
            _ccal = curvs[:max(0, len(curvs) - _reach_extra)] or curvs
        else:
            _jcal, _ccal = jumps, curvs
        med_jump = _pct(_jcal, 0.5)
        med_curv = _pct(_ccal, 0.5)
        early = _jcal[: max(1, len(_jcal) // 3)]
        early_frac = sum(early) / (sum(_jcal) + 1e-12)
        conv_step = len(_jcal)
        thr = 0.05 * (max(_jcal) if _jcal else 1.0)
        for i, jv in enumerate(_jcal):
            if jv < thr:
                conv_step = i
                break
        # Distilled = the x0 trajectory stops moving within the first few
        # steps AND most of its movement happens in that front window.  We do
        # NOT flag "last jump tiny relative to the max" as distilled: any
        # smooth fast-converging model (jump ~ sigma^p with p > 1) has a
        # small final jump once the grid reaches low sigma, so that clause
        # turned every cfg > 1 flow model into a false distilled positive.
        # The cfg override in _resolve_distilled_override already catches
        # true cfg <= 1 distilled models.
        # THE VERDICT'S OWN LENGTH MUST BE THE CALIBRATION BAND'S, not the full grid's: this
        # bound widens with the grid, so leaving len(jumps) here re-opened the false-distilled
        # path that pinning early_frac/conv_step alone had closed.  Found by a source census
        # (rig_probe_reach.py C11) written because the first version of this protection missed it.
        is_distilled_eff = bool(early_frac > 0.75 and
                                conv_step <= max(2, len(_jcal) // 3))
        probe_cfg = _probe_effective_cfg(model, extra_args, gap_ctx)
        if probe_cfg is None:
            try:
                probe_cfg = float(cfg.get("cond_probe_cfg"))
            except (TypeError, ValueError):
                probe_cfg = None
        is_distilled_eff = _resolve_distilled_override(
            cfg.get("cond_probe_is_distilled", "auto"),
            probe_cfg,
            is_distilled_eff)
        gap = None
        if bool(cfg.get("cond_probe_gap", True)):
            if probe_cfg is not None and probe_cfg <= 1.0 + 1e-6:
                gap = None
            else:
                gap = _measure_guidance_gap(model, extra_args, xs, x0s,
                                            grid, ones, gap_ctx=gap_ctx)
        endgame = None
        if bool(cfg.get("probe_endgame", True)):
            _progress(callback, "cond_probe", "progress", len(grid), len(grid) + 4,
                      "PD-AOS cond probe: endgame")
            endgame = _probe_endgame(model, extra_args, x0s[-1], grid[-1],
                                     pos_all, ones, space, _probe_gen(x))
        # Computed ONCE here so the profile dict below can carry BOTH the curve
        # and its per-term breakdown without walking the trajectory twice.  The
        # breakdown matters: measured on the real corpus the probe curve lands
        # ~10x BELOW the run's (probe p50 0.0000 vs a run p50 of 0.27-0.41), and
        # the gate is a 4-way AND -- without the per-conjunct pass rates that
        # failure cannot be attributed to any one term.
        # THE RUN'S RESCUE INPUTS, NOT A PLACEHOLDER.  The run's floor is
        # `rescue.floor * sigma` per step and its gate is fed only from
        # `rescue_start`; both are passed so the probe predicts the gate the run
        # actually evaluates (D3, U08).
        _cov_floor = float(cfg.get("rescue_floor", 0.02))
        _cov_start = float(cfg.get("rescue_start", 0.55))
        _cov_r = _lf_rescue_cov_curve(
            xs, x0s, grid, _eg_n_from_err(_endgame_eg_err(endgame)),
            _cov_floor, _cov_start)
        profile = {
            "version": 8,
            "cond_sig": cfg.get("cond_probe_sig"),
            "space": space,
            "n_probe_steps": m,
            # WHAT THIS PROFILE WAS MEASURED AT, so the artefact describes itself.  `n_probe_steps` was
            # already here; the RESOLUTION was not, which is why a run asking for 16 px consuming a 32 px
            # measurement could not be seen in the log at all.  `pres` is the EFFECTIVE value.
            "n_probe_res": int(pres),
            # A COND MEASUREMENT, SAID SO -- see the focused twin above for why.
            "probe_src": "cond",
            "probe_device": str(getattr(x, "device", "?")),
            "sigma_curve": [round(v, 6) for v in grid],
            # THE REACH IN FORCE, stamped rather than inferred (rule 14).  `probe_band_floor` is
            # where the SHIPPED grid stopped and every scalar aggregate was computed;
            # `probe_reach_floor` is the lowest sigma actually measured.  A reader cannot tell a
            # 0.1-reach profile from a 0.01-reach one without both, and they are not comparable.
            "probe_reach_frac": round(float(cfg.get("probe_smin_frac",
                                                     _PROBE_SMIN_FLOOR)), 6),
            "probe_band_floor": round(float(_band_floor), 6),
            "probe_reach_floor": round(float(min(grid)), 6),
            "probe_reach_n_extra": int(_reach_extra),
            "jump_curve": [round(v, 6) for v in jumps],
            "gap_curve": [round(v, 6) for v in gap_curve],
            "real_gap_curve": [round(v, 6) for v in real_gap_curve],
            "local_err_curve": [round(v, 6) for v in local_errs],
            "slope_curve": [round(j / max(gv, 1e-9), 8)
                            for j, gv in zip(jumps, gap_curve)],
            "med_local_err_real": (_pct(local_errs_real, 0.5)
                                   if local_errs_real else 0.0),
            # WHICH ESTIMATOR PRODUCED THESE NUMBERS (U16/6906).  `pct_derived` changes what
            # the percentiles ARE, so it is a MEASUREMENT input, not a calibration input:
            # stamped here so a reader can tell the two estimators apart without trusting a
            # key name, and left out of `calib_flags` for the same reason.
            "pct_derived": bool(cfg.get("pct_derived", False)),
            "curv_curve": [round(v, 6) for v in curvs],
            # THE SIGMA EACH CURVATURE POINT IS ANCHORED AT.  The convention is documented at
            # `aflops.py:7687` -- `curv_curve[k]` is the curvature at `sigma_curve[k+1]`, curves carry
            # m-2 entries against the grid's m -- and a reader who does not know it will mis-align every
            # derivation, as a rig of mine did this session.  Logging the anchors removes the need to
            # know it.
            "curv_sigma": [round(float(v), 6) for v in grid[1:1 + len(curvs)]],
            # PROBE-SIDE COVERAGE CURVE.  The same statistic the rescue gate
            # consumes in the run (`frac`), computed along THIS probe trajectory
            # by the SAME function, so a pitfall like the coverage cliff can be
            # seen BEFORE sampling rather than diagnosed afterwards.  The probe is
            # the only instrument that measures the model AND the prompt first.
            # See NOTES.md and scratch_rescue_cliff_rig.py.
            "cov_curve": (_cov_r or {}).get("curve"),
            "cov_curve_terms": (_cov_r or {}).get("terms"),
            # The DERIVED variant (ref_drop = k*du, k measured from this very
            # trajectory) alongside the shipped per-step curve.  Both are emitted
            # so the next real runs can decide which one actually tracks the run:
            # synthetic fields could not, see scratch_ref_decay_rig.py.
            "cov_curve_kdu": (_cov_r or {}).get("curve_kdu"),
            "cov_curve_k": (_cov_r or {}).get("k"),
            # WHICH probe model-call path ran, and with what arguments.  139 of
            # 224 profiles are frozen (model returns its input); this is what will
            # identify the cause on the next frozen run.
            "probe_call": dict(_PROBE_CALL_DIAG),
            "cov_curve_meta": _lf_rescue_cov_meta(
                x0s[-1] if x0s else None, _endgame_eg_err(endgame),
                _cov_floor, _cov_start),
            "med_jump": med_jump,
            "med_curv": med_curv,
            # THE PER-ORDER DERIVATIVE SCALE, measured from this very trajectory.  Without it
            # the derived wmax cap refuses (see _derived_wmax_cap), so this is what lets the
            # `wmax_bound` arm engage at all.  Omitted orders are ones this grid is too short
            # to certify, and the cap then falls back rather than guessing.
            "dk_scale": _measure_dk_scale(x0s, gap_probe, return_series=True)[0],
            # THE PER-STEP SERIES behind that median.  The scalar above is what the cap
            # consumes today; this is what a per-model order_max equation needs, because
            # a median hides the tail where the probe and the run disagree.  Anchor:
            # series[k][i] belongs to sigma_curve[k + i].
            "dk_series": _measure_dk_scale(x0s, gap_probe, return_series=True)[1],
            # THE A/B FLAGS THIS PROFILE WAS BUILT UNDER.  `_calib_flags` reads this, and
            # without it the `wmax_bound` arm was INERT whatever the widget said -- a toggle
            # wired to nothing, which is indistinguishable from one whose effect is below the
            # noise floor (the exact trap rule 9 exists for).  `cfg` is in scope here.
            _CALIB_FLAGS_KEY: _calib_inputs(cfg),
            "med_jump_rel": med_jump_rel,
            "med_detail_abs": med_detail_abs,
            "early_frac": early_frac,
            "conv_step": conv_step,
            "conv_sigma": (round(float(grid[int(conv_step)]), 6)
                           if 0 <= conv_step < len(grid) else None),
            "osc_sigma": (round(float(osc_sigma), 6)
                          if osc_sigma is not None else None),
            "is_distilled_eff": is_distilled_eff,
            "med_local_err": (_pct(local_errs, 0.5) if local_errs else 0.0),
            "guard_sig_med": (_pct(guard_sigs, 0.5) if guard_sigs else 0.0),
            "guard_sig_p90": (_pct(guard_sigs, 0.9) if guard_sigs else 0.0),
            # BOTH estimators logged whatever the toggle, so one run shows the delta and,
            # with `n_guard`, whether the p90/med collision is in play for this profile.
            "guard_sig_med_shipped": (_probe_pct(guard_sigs, 0.5, False)
                                      if guard_sigs else 0.0),
            "guard_sig_p90_shipped": (_probe_pct(guard_sigs, 0.9, False)
                                      if guard_sigs else 0.0),
            "guard_sig_med_derived": (_probe_pct(guard_sigs, 0.5, True)
                                      if guard_sigs else 0.0),
            "guard_sig_p90_derived": (_probe_pct(guard_sigs, 0.9, True)
                                      if guard_sigs else 0.0),
            "log_gap_probe": gap_probe,
            "log_gap_real": gap_real,
            "gap": gap,
            "probe_cfg": probe_cfg,
            "n_jump": len(jumps),
            "n_err": len(local_errs),
            "n_guard": len(guard_sigs),
            "jump_q": jump_q,
            "err_q": err_q,
            "fragility": _fragility_map(frag_acc),
            "ojf_curve": ojf_curve,
            "band_curve": band_curve,
            "dir_cons": dir_cons,
            "dir_cons_valid": dir_cons_valid,
            "endgame": endgame,
            "whitepoint": _whitepoint_from_x0s(x0s),
        }
        profile["trajectory_state"] = _compute_trajectory_state(profile)
        # THE CFG GOES IN: the loosening multiplier must be the one THIS run's cfg names, not a value an
        # earlier entry point left in a module global (see `_loosen_from_cfg`).
        profile["calib"] = _derive_cond_calibrations(profile, cfg)
        return profile
    except Exception as e:
        logging.warning("[PD-AOS-cond-probe] probe failed: %s", e)
        return None

def _apply_cond_probe_profile(cfg, cond_profile, tune, weight, auto_tunable):
    if not cond_profile:
        return
    cond_profile = _recalibrated(cond_profile, cfg, _derive_cond_calibrations)
    cond_calib = cond_profile.get("calib", {})
    if cfg.get("_dk_scale") is None:
        try:
            _dk = cond_profile.get("dk_scale")
            if isinstance(_dk, dict) and _dk:
                cfg["_dk_scale"] = {int(k): float(v) for k, v in _dk.items()}
        except Exception:
            pass
    w = max(0.0, min(float(weight), 1.0))
    if w <= 0.0:
        return

    def _blend_set(key):
        # EVERY EARLY EXIT IS COUNTED, for the same reason the loosening's gate is: this function has
        # three silent returns, and any of them makes the cond path's contribution to `key` invisible in
        # the log.  That is what left the operator's first A/B with a null it could not explain.
        if key not in auto_tunable:
            _ab_hit("cond.blend.not_tunable." + str(key))
            return
        cond_val = cond_calib.get(key)
        if cond_val is None:
            _ab_hit("cond.blend.no_value." + str(key))
            return
        try:
            cur = float(cfg.get(key))
            cond_f = float(cond_val)
            blended = (1.0 - w) * cur + w * cond_f
            if cur > 0 and blended > 0 and key != "warmup":
                blended = max(0.25 * cur, min(4.0 * cur, blended))
            cfg[key] = blended
            _ab_hit("cond.blend.applied." + str(key))
        except (TypeError, ValueError):
            _ab_hit("cond.blend.unparseable." + str(key))
            pass

    if tune.get("guard", True):
        _blend_set("guard_floor")
    if tune.get("tol", True):
        _blend_set("rtol")
    if tune.get("wmax", True):
        _blend_set("wmax_base")
    if tune.get("eta", True):
        _apply_eta_cap(cfg, cond_calib.get("safe_eta"), "cond:calib.safe_eta")
    if cond_profile.get("is_distilled_eff"):
        _blend_set("warmup")
    if tune.get("anomaly", True):
        for _ak in ("an_from", "an_spatial", "an_novelty", "an_value"):
            _blend_set(_ak)



def _schedule_weight(curv_val, sig, smax, p=2.0, r=1.0):
    """Step-density weight of the fine-tune at one sigma.

    w = curv^(1/p) * r^(2*sigma/smax - 1)

    The bias enters as r = f(bias) = 4**(bias-1), i.e. r at the top of the
    schedule and 1/r at the bottom, so bias=1+d and bias=1-d give EXACT
    pointwise reciprocal weights (their product is curv^(2/p) at every
    sigma).  Extracted as a helper so the symmetry rig tests the shipping
    expression rather than a copy of it.
    """
    u = (sig / smax) if smax > 0.0 else sig
    return (max(float(curv_val), 1e-12) ** (1.0 / p)) * \
        (float(r) ** (2.0 * u - 1.0))


def _warp_frontier(profile, base_sigmas, bias=1.0, p=2.0, coarse=0.01):
    """Most aggressive SAFE warp for this profile + schedule (auto-warp).

    Criterion -- geometric, not an absolute error budget (rig
    scratch_safe_warp_rig.py showed that neither a fitted err~curv^a*du^p
    model nor the probe's gap-scaled local error predicts the engine's
    per-step error across runs: leave-one-run-out was off by 3-4 orders of
    magnitude).  What IS measurable, and therefore what gates the search:

      - COVERAGE: every step must stay inside the probe's measured sigma
        range.  Below the measured floor there is no curvature evidence at
        all, so that is exactly where "beyond measured" begins.  This is the
        binding constraint in practice.

    The step-difficulty ratio (worst curv*du^2 vs the user's own schedule) is
    computed and REPORTED but does not gate: the rig shows it stays within
    ~0.92-1.10 over the entire sweep, i.e. it carries almost no information.
    The terminal jump to sigma=0 is excluded (it is not an integrator step).

    Returns (warp_value, info); warp_value = 1.0 (neutral) when nothing below
    1.0 is safe, the profile is degenerate, or the search fails.
    """
    try:
        _base = [float(v) for v in base_sigmas]
        if not _base or len(_base) < 5:
            return 1.0, {"why": "no usable schedule"}
        curv = profile.get("curv_curve")
        sigc = profile.get("sigma_curve")
        if not curv or not sigc or len(curv) < 3 or len(sigc) < 4:
            return 1.0, {"why": "no curvature evidence"}
        n = min(len(curv), len(sigc) - 1)
        anchors = [(float(sigc[k + 1]), max(float(curv[k]), 1e-9))
                   for k in range(n)]
        _med_c = sorted(c for _, c in anchors)[len(anchors) // 2]
        if _med_c < 1e-4:
            return 1.0, {"why": "degenerate profile"}
        floor = anchors[-1][0]

        def _curv_at(s):
            if s >= anchors[0][0]:
                return anchors[0][1]
            if s <= anchors[-1][0]:
                return anchors[-1][1]
            for (s1, c1), (s2, c2) in zip(anchors, anchors[1:]):
                if s2 <= s <= s1:
                    l1, l2 = math.log(s1), math.log(s2)
                    l = math.log(s)
                    t = (l - l1) / (l2 - l1) if l2 != l1 else 0.0
                    return math.exp(math.log(c1)
                                    + t * (math.log(c2) - math.log(c1)))
            return anchors[-1][1]

        def _proxy(sig):
            worst = 0.0
            below = 0
            steps = len(sig) - 1
            for i in range(steps):
                s, sn = sig[i], sig[i + 1]
                if sn <= 0.0 or i == steps - 1:
                    continue
                du = math.log(s / sn)
                c = _curv_at(0.5 * (s + sn))
                worst = max(worst, c * du * du)
                if sn < floor * 0.999:
                    below += 1
            return worst, below

        _n_steps = len(_base) - 1
        # difficulty diagnostic: reference = the user's OWN schedule
        ref, below_base = _proxy(_base)
        best = 1.0
        info = {"worst_ratio": 1.0, "below": below_base, "below_base": below_base,
                "floor": round(floor, 6), "why": "nothing below 1.00 is safe"}
        w = 1.0
        while w > 0.0:
            w = round(w - coarse, 4)
            if w < 0.0:
                w = 0.0
            g = _build_curvature_schedule(profile, _n_steps, 1.0, p=p,
                                          bias=bias, warp=w, base_sigmas=_base)
            if g is None:
                info["why"] = "builder rejected the warped schedule"
                break
            sig = [float(v) for v in g.tolist()]
            worst, below = _proxy(sig)
            ratio = (worst / ref) if ref > 0.0 else 1.0
            # COVERAGE IS MEASURED AGAINST THE USER'S OWN SCHEDULE, NOT AGAINST ZERO.
            # WHY THIS CHANGED (2026-10-04).  The test used to be `below > 0`, i.e. the warp had to
            # leave EVERY step inside the probe's measured band -- but the band's floor is the probe
            # grid's LOWEST sigma while a real 50-step schedule runs on down to `sigma_min`, so the
            # user's own base already sits below it.  The search walks w DOWN from 1.0 and breaks at
            # the first refusal, so it broke on its first candidate, `best` stayed 1.0, and the
            # frontier reported `safe_range [1.0, 1.0]` on EVERY run it ever evaluated.
            # MEASURED on the corpus: auto_warp was ON in 439 runs and `warp_used` was 1.0 in all
            # 439, with `why` = "coverage: steps below the measured floor" -- an inert switch that
            # looked like a considered decision, which is DEAD_FEATURES Â§8 read as "unproven at n=1"
            # long after n had grown past 400.  The warp cannot be asked to repair a coverage
            # violation it did not cause; what it must not do is ADD one.
            if below > below_base:
                info["why"] = "coverage: the warp adds steps below the measured floor"
                info["below"] = below
                info["below_base"] = below_base
                break
            best = w
            info = {"worst_ratio": round(ratio, 4), "below": below, "below_base": below_base,
                    "floor": round(floor, 6), "why": "coverage-limited"}
        info["safe_range"] = [best, 1.0]
        return float(best), info
    except Exception:
        return 1.0, {"why": "search failed"}


def _build_curvature_schedule(profile, steps, denoise=1.0, p=2.0, bias=1.0,
                              warp=1.0, base_sigmas=None, engage=None):
    """Fine-tune the user's sigma list with the probe's curvature.

    `engage` decides WHETHER the derived placement is used at all, and it is an EXPLICIT argument
    rather than a module global (rule 14).  `None` means "decide from the knobs": engaged when the
    caller asked for it (`probe_schedule`) or when the tilt is off neutral (`bias != 1.0`), which is
    the staging the operator chose -- the shipped default (bias = 1.0, probe_schedule off) keeps the
    incoming schedule EXACTLY, and only a run that asks for the derived placement gets one.
    An explicit True/False overrides that.

    Base math (rig-fitted on CFG5log/CFG1log): the aflops integrator steps in
    u = log(sigma) (its du), so its consistency error is second order in du:
    err ~ C * curv * du^2  (rig: curvature exponent a ~= 0.9 pooled with
    p = 2 fixed by the integrator order).  Equal-error placement therefore
    uses density ~ curv^(1/p), p = 2 -> sqrt(curv).

    Two elementary knobs, both neutral at 1.0 on a 0.00-2.00 widget, both
    using the factor f(w) = 4**(w-1) so that w and (2 - w) are exact
    reciprocals -- equal power in either direction:

      warp  reshapes the SIGMA VALUES with the Mobius flow-shift map
            (u' = S*u/(1+(S-1)*u) on u = sigma/smax, S = f(warp)).  Endpoints
            are fixed, S and 1/S are exact inverses (the map is a
            one-parameter group: T_S . T_S' = T_(S S')), and the resulting
            step-density profile at w=1+d is the exact reciprocal of the
            profile at w=1-d.
      bias  tilts the step DENSITY between the top and the bottom of the
            schedule: weight = r**(2*sigma/smax - 1) with r = f(bias), i.e. r
            at the top and 1/r at the bottom -- pointwise exact reciprocals.

    Warp is applied first (it moves the sigma values), then bias re-allocates
    the steps on the warped axis.  bias = warp = 1.0 gives weight 1 everywhere
    and the identity map, so the user's list is reproduced exactly -- the
    probe only ever re-allocates steps WITHIN the schedule plugged in.

    Anchor semantics follow the probe source: curv_curve[k] is the curvature
    at sigma_curve[k+1] (not at the interval midpoint).

    Returns a descending sigma tensor [start ... 0.0], or None."""
    try:
        curv = profile.get("curv_curve")
        sigc = profile.get("sigma_curve")
        if not curv or not sigc or len(curv) < 3 or len(sigc) < 4:
            return None
        steps = int(steps)
        if steps < 4 or float(denoise) <= 0.0:
            return None
        n = min(len(curv), len(sigc) - 1)
        anchors = [(float(sigc[k + 1]), max(float(curv[k]), 1e-9))
                   for k in range(n)]
        if max(c for _, c in anchors) <= 1e-9:
            return None
        # Degeneracy guard: the probe's own bar (_probe_degenerate) says a
        # profile with med_curv < 1e-4 carries no usable dynamics evidence.
        # A numerically-flat trajectory (distilled model at cfg 1, measured
        # curv ~1e-6 with zeros clamped to 1e-9) still passes the max>1e-9
        # test above, and its 1e-6-vs-1e-9 noise contrast drives a 1000x
        # density swing: measured log (CFG1) -> one starved 0.39 sigma gap
        # (2.3x the input schedule's max gap) and a 109x corrector overshoot.
        # Reject such profiles: the schedule then passes through untouched.
        _med_c = sorted(c for _, c in anchors)[len(anchors) // 2]
        if _med_c < 1e-4:
            return None

        def curv_at(s):
            if s >= anchors[0][0]:
                return anchors[0][1]
            if s <= anchors[-1][0]:
                return anchors[-1][1]
            for (s1, c1), (s2, c2) in zip(anchors, anchors[1:]):
                if s2 <= s <= s1:
                    l1, l2 = math.log(s1), math.log(s2)
                    l = math.log(s)
                    t = (l - l1) / (l2 - l1) if l2 != l1 else 0.0
                    return math.exp(math.log(c1)
                                    + t * (math.log(c2) - math.log(c1)))
            return anchors[-1][1]

        try:
            _bs = [float(v) for v in base_sigmas]
        except Exception:
            _bs = None
        if not _bs or len(_bs) < 5:
            return None
        _bn = len(_bs) - 1
        # steps is derived from the external list; denoise is the user's
        # list as-is (they slice it with their own nodes).
        #
        # ---- knobs: bias + warp, both 1.0 = neutral -----------------------
        # Unit: f(w) = 4**(w-1), so the widget range [0, 2] spans a factor of
        # 4 each way and w / (2 - w) are EXACT reciprocals -> +0.5 and -0.5
        # carry the same amount of power in opposite directions (rig-verified
        # inverse pairing, see scratch_symmetry_rig.py).
        _r = 4.0 ** (float(bias) - 1.0)      # density tilt factor
        _s = 4.0 ** (float(warp) - 1.0)      # sigma-shift factor
        # WHETHER THE DERIVED PLACEMENT RUNS AT ALL.  `engage=None` means "ask the tilt": a run at
        # bias = 1.0 that did not switch the derived schedule on keeps its own list EXACTLY, which
        # is the staging chosen for this change.  Note this gates the DENSITY only -- the warp map
        # below is applied either way, because it is live in the shipped configuration (27 corpus
        # runs use it) and gating it here would silently disable a working knob.
        _engaged = (abs(float(bias) - 1.0) > 1e-9) if engage is None else bool(engage)
        # warp: Mobius flow-shift on the NORMALIZED schedule, which fixes both
        # endpoints (sigma = smax and 0 unchanged) and is a one-parameter
        # group (T_S . T_S' = T_{S S'}), so S and 1/S are exact inverses.
        if abs(_s - 1.0) > 1e-9:
            _smax_w = _bs[0] if _bs[0] > 0.0 else 1.0
            _w = []
            for _v in _bs:
                _u = _v / _smax_w
                _up = _s * _u / (1.0 + (_s - 1.0) * _u)
                _w.append(round(_up * _smax_w, 6))
            _w[-1] = _bs[-1]
            for _i in range(len(_w) - 1):
                if _w[_i] <= _w[_i + 1]:
                    return None
            _bs = _w
        _map = _bs[:-1]          # N boundary sigmas, descending
        _tail = _bs[-1]          # the user's own endpoint (usually 0)
        if _map[0] <= _map[-1] or _map[0] <= 0.0:
            return None
        _smax = _map[0] if _map[0] > 0.0 else 1.0

        if not _engaged:
            # Not engaged: hand the (possibly warped) base straight back, in the SAME form the
            # placement path would have returned it -- 6-decimal rounding, then float32 -- so a run
            # that does not ask for the derived schedule is bit-identical to the shipped behaviour
            # rather than merely close to it.
            _o = [round(float(v), 6) for v in _map] + [round(float(_tail), 6)]
            _o[0] = round(float(_map[0]), 6)
            return torch.tensor(_o, dtype=torch.float32)

        def sig_of_u(u):
            f = min(max(float(u), 0.0), float(_bn - 1))
            i = int(f)
            t = f - i
            if i >= _bn - 1:
                return _map[-1]
            return _map[i] + t * (_map[i + 1] - _map[i])

        M = 800
        us = [_bn * (i / M) for i in range(M + 1)]
        ws = [_schedule_weight(curv_at(sig_of_u(u)), sig_of_u(u), _smax,
                               p, _r) for u in us]
        # Resolution floor: the fine-tune may add steps where curvature needs them but must never
        # make any step COARSER than the user's own schedule.  Output gap = mean(w)/w(u) index
        # units, so enforce w(u) >= mean(w).  Without this the top-heavy curv^0.5 density starves
        # the low-sigma tail: measured CFG5 -- last positive sigma 0.132 vs simple's 0.0577, leaving
        # the image at a higher residual sigma ('not fully resolved').
        #
        # ---- THE FIXED POINT WAS DEGENERATE (fixed 2026-10-09) --------------------------------
        # This used to ITERATE the clamp to a fixed point of `mu = mean(max(w, mu))`.  That
        # equation is `q*A + (1-q)*mu = mu` (q the share above mu, A their mean), i.e. `A = mu`,
        # which has NO solution below `max(w)`: the mean of the values above mu is always strictly
        # greater than mu, so the bisection took its `_lo = mid` branch every time and terminated
        # at `mu = max(w)`.  The clamp then made EVERY weight identical, the cumulant became a
        # straight line, and the placement reproduced the base exactly -- for any curvature, any
        # `p`, any bias.  Measured on the corpus (`rig_schedule_derivation.py`): 832 builds over
        # 104 profiles matched their base to the builder's own 6-decimal rounding, `mu == max(w)`
        # on 104/104 runs, and 4056 swept (run, mu) points with ZERO interior solutions.  The
        # consequence was that the whole error-equidistribution -- the mechanism this function
        # exists for -- had never moved a single step in any run, and neither had `bias`.
        #
        # The rule the comment above STATES is a single clamp at the mean, which is what is
        # implemented now: after it, every realised gap is <= 1 index unit, which is exactly "no
        # step coarser than the base's".  It is not iterated, because iterating is what diverged to
        # max(w).  Measured effect of the one-shot form (floor ablation, same rig): the last
        # positive sigma lands at 1.06x the base's instead of 1.68x with no clamp at all, so the
        # tail protection the fixed point was reaching for is retained, while the density's SHAPE
        # survives and can re-place steps.
        _mu = sum(ws) / len(ws)
        if _mu > min(ws):
            ws = [max(v, _mu) for v in ws]
        cum = [0.0]
        for i in range(1, M + 1):
            cum.append(cum[-1] + 0.5 * (ws[i - 1] + ws[i])
                       * (us[i] - us[i - 1]))
        tot_w = cum[-1]
        if tot_w <= 1e-12:
            return None
        tgt = [tot_w * (j / _bn) for j in range(_bn)]
        out = []
        j = 0
        for i in range(1, M + 1):
            while j < len(tgt) and cum[i] >= tgt[j]:
                den = cum[i] - cum[i - 1]
                frac = (tgt[j] - cum[i - 1]) / den if den > 1e-12 \
                    else 0.0
                out.append(sig_of_u(us[i - 1]
                                    + frac * (us[i] - us[i - 1])))
                j += 1
        if len(out) < _bn:
            return None
        out = [round(float(v), 6) for v in out] + [round(_tail, 6)]
        out[0] = round(_map[0], 6)
        grid = torch.tensor(out, dtype=torch.float32)
        if grid.numel() != _bn + 1:
            return None
        return grid
    except Exception:
        return None


_DWELL_OV_GATE = 0.25     # below this measured oversteer the dwell never arms
# Oversteer at which the dwell reaches its cap.  Was 0.55, which no run ever
# reached: the strongest of 31 post-fix runs scored evidence.ov = 0.392, so the
# knob's top 29% did nothing.  Re-anchored at the measured p90 of the post-fix
# evidence.ov distribution (0.288), so the strongest decile of real runs attains
# the _DWELL_CAP ceiling while everything below it scales proportionally.
# Recorded in NOTES.md; scratch_dwell_calibration_rig.py derives it from the
# corpus rather than asserting it.  NOTE this is a stepping constant -- it scales
# how much of a step a pixel withholds -- so the anchor is measured but the
# visual effect still wants an A/B.
_DWELL_OV_REF = 0.288     # oversteer at which the dwell reaches its cap (p90)
_DWELL_CAP = 0.20         # strength ceiling (fraction of a step withheld)
_DWELL_R_HI = 2.0         # flux ratio (vs image median) at full dwell weight
_DWELL_EMA = 0.7          # decay of the per-pixel flux envelope
_DWELL_CURV_FLOOR = 1e-4  # max curvature below this = degenerate probe (no dwell)


def _dwell_env(kn, ramp_end, fade_start, fade_end):
    """Phase envelope of the per-pixel dwell: ramp in over the first steps,
    hold through the measured active zone, fade out before the endgame."""
    t_in = _sstep((float(kn) - 0.04) / max(float(ramp_end) - 0.04, 1e-6))
    t_out = 1.0 - _sstep((float(kn) - float(fade_start)) /
                         max(float(fade_end) - float(fade_start), 1e-6))
    return t_in * t_out


# ---------------------------------------------------------------------------
# Probe-driven lambda clamp.
#
# The clamp is a TRUST REGION on the lambda ESTIMATE, not a stability guard.
# lambda = -1 + <dx0, dx>/||dx||^2 is literally a measurement of how fast the
# denoiser's x0 is drifting along the step, and the probe measures exactly that
# quantity (jump_curve).  So where the measured jump is large the estimate is
# least trustworthy and the bound is pulled to the ANCHOR (lambda = -1, the
# exact constant-x0 step); where x0 has settled the estimate is informative and
# the bound relaxes to the headroom below the anchor (today's -3).
#
# Measured on the saved corpus: the clamp binds only in the first steps, and
# the probe's jump curve peaks in exactly that interval (23/27 non-degenerate
# profiles), decaying monotonically after it.
# ---------------------------------------------------------------------------
_LAM_ANCHOR = -1.0        # lambda = -1 is the exact constant-x0 step
_LAM_HEADROOM = 2.0       # -> the loose end is -3.0, today's default


def _recommend_lam_clamp_from_profile(prof_m, prof_c, sigmas):
    """Per-step lambda LOWER bound, from the probe's measured x0 jump.

    Returns one bound per sigma interval (len(sigmas) - 1 entries), or None
    when there is no usable evidence -- notably on DISTILLED models, whose
    measured jump is ~0 (they barely move x0) and so carries no signal; those
    keep the fixed bound.
    """
    try:
        prof = None
        for p in (prof_c, prof_m):
            if (isinstance(p, dict)
                    and len(p.get("jump_curve") or []) >= 1
                    and not _probe_degenerate(p)):
                prof = p
                break
        if prof is None:
            return None
        jump = [float(v) for v in (prof.get("jump_curve") or [])]
        sigc = [float(v) for v in (prof.get("sigma_curve") or [])]
        n = min(len(jump), max(0, len(sigc) - 1))
        if n < 1:
            return None
        jump = jump[:n]
        sigc = sigc[:n + 1]
        jmax = max(jump)
        jmin = min(jump)
        if jmax <= 0.0:
            return None                 # distilled: x0 does not move at all
        jspan = jmax - jmin
        if jspan < 1e-9:
            # flat jump curve: no gradient to grade the estimate by, so keep
            # the fixed bound rather than inventing a difference
            return None
        s = _sigmas_as_floats(sigmas)
        if len(s) < 2:
            return None
        out = []
        for i in range(len(s) - 1):
            sg = s[i]
            j = jump[-1]                # below the probed band: bottom interval
            for k in range(n):          # sigma_curve descends, so the first
                if sg >= sigc[k + 1]:   # k whose low edge sigma is under sg
                    j = jump[k]
                    break
            # normalise over the SPAN so the largest-jump interval is the anchor
            # and the smallest is the full headroom -- normalising by max alone
            # would leave the loose end unreachable unless a jump were exactly 0
            rel = min(max((j - jmin) / jspan, 0.0), 1.0)
            out.append(_LAM_ANCHOR - _LAM_HEADROOM * (1.0 - rel))
        return out
    except Exception:
        return None


def _run_lam_bound(jump, jump_max):
    """Causal per-step lambda lower bound from the RUN's own measured x0 jump.

    Preferred over the probe's because, measured against the saved corpus, the
    probe over-reports the jump by 2-4x on Anima (right shape, 0.93-0.97
    correlation, wrong magnitude) and reports a flat 0.0 on distilled Krea2
    where the run's real jump is 0.375 at the top.  This uses the previous
    step's already-computed x0, so it is available BEFORE the update.

    Normalised by the largest jump seen SO FAR: at the first measured step
    rel = 1 (the anchor), and because the jump decays monotonically down the
    schedule on both models the bound relaxes as the run proceeds.
    """
    try:
        if jump is None or jump_max is None or jump_max <= 0.0:
            return None
        rel = min(max(float(jump) / float(jump_max), 0.0), 1.0)
        return _LAM_ANCHOR - _LAM_HEADROOM * (1.0 - rel)
    except Exception:
        return None


def _recommend_dwell_from_profile(prof_m, prof_c, sigmas, lf_evid,
                                  fragile_ranges):
    """L1 per-pixel dwell: derive (strength, phase window) from probe evidence.

    Same evidence philosophy as the shift-schedule recommender and the local
    field's auto-gains, and it costs zero extra model calls: the probe
    decides HOW MUCH (the measured oversteer evidence scales the strength
    ceiling -- a model that does not overstep structure decisions never
    arms the dwell) and WHEN (the curvature curve's active zone places the
    fade-out; the endgame fragile band is a hard stop, so no new lag is
    ever introduced where the probe measured the model misreads structured
    state).  WHERE is decided at runtime by the per-pixel churn map --
    no probe can know which areas of THIS image will lag.  Everything here
    is derived from profile evidence; there are no architecture branches.

    Returns None (keep the exact classic full-step behavior) or
    {"dwell_max", "ramp_end", "fade_end", "ov"}.
    """
    try:
        if not isinstance(lf_evid, dict) or lf_evid.get("source") in (None, "none"):
            return None
        ov = min(max(float(lf_evid.get("ov", 0.0) or 0.0), 0.0), 1.0)
        if ov < _DWELL_OV_GATE:
            return None
        # finest curvature curve available drives the envelope (per-prompt
        # first, the same precedence as the shift schedule)
        prof = None
        for p in (prof_c, prof_m):
            if (isinstance(p, dict)
                    and len(p.get("curv_curve") or []) >= 3
                    and not _probe_degenerate(p)):
                prof = p
                break
        if prof is None:
            return None
        s = sigmas.detach().float()
        if s.numel() < 7:
            return None
        curv = [float(v) for v in (prof.get("curv_curve") or [])]
        sigc = prof.get("sigma_curve") or []
        n = min(len(curv), max(0, len(sigc) - 1))
        if n < 3:
            return None
        pts = sorted((math.log(max(float(sigc[i + 1]), 1e-8)), curv[i])
                     for i in range(n))
        cmax = max(c[1] for c in pts)
        if cmax <= _DWELL_CURV_FLOOR:
            return None
        # low-sigma edge of the measured active zone: scanning toward lower
        # sigma, structure decisions end where the curvature rises quiet ->
        # active; sig_q = the first (lowest) active sigma, so the dwell is
        # gone by the time the run passes the zone's edge (mapped onto this
        # run's grid as a fraction of steps)
        sv = [float(v) for v in s if float(v) > 1e-6]
        n_steps = len(sv)
        sig_q = None
        for i in range(1, len(pts)):
            if pts[i][1] >= 0.35 * cmax > pts[i - 1][1]:
                sig_q = math.exp(pts[i][0])
                break
        fade_end = 0.70
        if sig_q is not None and n_steps > 0:
            kn_q = next((i / float(n_steps) for i, v in enumerate(sv)
                         if v <= sig_q), 1.0)
            fade_end = kn_q
        fade_end = min(max(fade_end, 0.45), 0.75)
        # probe-measured endgame fragile band: hard stop before it
        try:
            rs, _rf = _rescue_window_from_fragile(s, fragile_ranges or [])
            if rs is not None:
                fade_end = min(fade_end, max(0.40, rs - 0.05))
        except Exception:
            pass
        strength = _DWELL_CAP * min(1.0, ov / _DWELL_OV_REF)
        return {"dwell_max": round(strength, 4),
                "ramp_end": round(min(0.30, 0.5 * fade_end), 4),
                "fade_end": round(fade_end, 4),
                "ov": round(ov, 4)}
    except Exception:
        return None


# THE `eg_n` GATE ON THE EXTRAPOLATION LIMIT, as a named constant instead of the bare
# `0.3` it used to be.  It is not a new number: it is the existing threshold, and naming
# it makes its MEANING checkable.  `eg_n` is defined at its producer as
#     eg_n = 1 - exp(-eg_err / 0.06)                                  (aflops.py:3238)
# so this gate is exactly `eg_err > 0.06 * ln(1 / (1 - 0.3))` = **eg_err > 0.0214**: the
# branch fires when the endgame error measurement exceeds 0.0214.  Whether that threshold
# is well placed is a separate question from the one under study (the DIRECTION of the
# response), and it is NOT settled by this constant being named.
_WMAX_EG_N_GATE = 0.3


ENGINE_DEFAULTS = {
    "order": 2, "extrap_space": "auto",
    # Noise defaults carry the tuned values the removed Options nodes used to
    # set (rig: old-vs-new log diff, CFG5/CFG1).  The old "massive
    # improvement" runs had eta_mode=adaptive + s_tmin=1.0 -> eta_eff was 0.0
    # for EVERY step (escape detection computed, injection fully gated); the
    # redesigned defaults fell back to cosine + s_tmin=0.0 -> 0.1 eta noise on
    # every step -> washed-out structure.  Restoring the tuned values as the
    # DEFAULT keeps the autotuner's "auto = good" contract.
    "eta_mode": "adaptive", "falloff": 2.0, "eta_min": 0.0,
    "eta_escape_jump": -1.0,      # <0 = auto: derived from the probe's med_jump_rel
    "eta_escape_detail": -1.0,    # <0 = auto: derived from the probe's med_detail_abs
    "eta_fade_sigma": -1.0,       # <0 = auto: derived from the probe's conv_sigma
    "eta_fade_floor": 0.25,       # fraction of the baseline eta retained after the late fade
    # NEUTRAL MULTIPLIER on the two DERIVED escape thresholds (jump and detail).
    # 1.0 = the probe's own measurement unchanged; >1 = the escape fires more
    # readily; 0 = it never fires.  It is NOT a tuned constant and NOT a
    # per-model table: the base it scales is the measurement taken on the model in
    # front of the engine, so sensitivity is one number instead of two hand-set
    # thresholds that would generalise only as far as the next checkpoint
    # resembles the one they were chosen on.
    "escape_gain": 1.0,
    # WHICH CONDITIONS MAY ARM THE STALL HALF OF THE ESCAPE.  A string rather
    # than a boolean because the flatness condition is not being replaced, it is
    # being JOINED -- "flatness" is the shipped behaviour and "momentum" adds a
    # second way to detect the same physical state.  The reversal (osc) half of
    # the detector is untouched by this key, so the A/B changes ONE thing.
    #
    # WHY A SECOND CONDITION IS NEEDED, measured on the first build that shipped
    # the Noise Options node (rig_eta_tune.py, 7 runs, 5 checkpoints): the two
    # inputs the stall term multiplies move in OPPOSITE directions as a run
    # converges.  Spearman against step position was -1.00 for `eta_jump_rel`
    # (the movement collapses) and +0.99 for `eta_detail_abs` (the image gains
    # structure), because the high-frequency term RISES as the picture forms
    # rather than falling flat.  A conjunction of "moving little" and "still
    # flat" is therefore unsatisfiable on those runs: `stall` read exactly 0.0000
    # on all 7 at the shipped gain of 1.0.  The gain DOES reach it -- the audit's
    # ten-run measurement found gains of 1.5 to 4.0 arming 1 to 39 steps -- so the
    # shipped sensitivity sits below that population's floor rather than the term
    # being zero by construction (2 of the 7 have min(jump) above the derived
    # threshold, which is dead by construction, and centring both thresholds on the
    # run's OWN medians still armed 1 step in 7).  The default gain is left at 1.0
    # because a gain high enough to arm this also noises steps that did not need it.
    #
    # So the flatness gate is kept AND a movement-collapse test is added: the
    # per-step movement has fallen well below this run's own recent level AND the
    # image is not still gaining detail.  Both halves of that test are relative to
    # the run in front of the engine, which is what makes it a measurement rather
    # than a constant.
    "escape_mode": "flatness",
    # The momentum branch's window, in steps: how much recent history "this run's
    # own recent level" is measured over.  Short enough that a genuine endgame
    # collapse is visible as a drop, long enough that one noisy step cannot arm it.
    "escape_momentum_window": 4,
    # How far below that recent level the movement must fall to count as a
    # collapse.  0.7 = the per-step movement is under 70% of the recent median.
    # Read it as the sensitivity of the NEW branch, exactly as `escape_gain` is
    # for the old one: lower arms more readily.  It is a ratio of the run against
    # itself, so it does not need to be re-chosen per checkpoint.
    "escape_momentum_collapse": 0.7,
    # How much detail is allowed to keep growing before the trajectory is called
    # stalled: at or below 0.15 the image is treated as no longer developing, and
    # above it the escape stands down even if the movement collapsed -- which is
    # what keeps the branch from firing at ordinary convergence, where the
    # movement always collapses but the picture is still being built.
    "escape_momentum_growth": 0.15,
    # WHETHER THE MOMENTUM ARM ALSO REQUIRES THE IMAGE TO BE FLAT, using the same
    # derived threshold (`eta_escape_detail`) the shipped flatness branch uses.
    # ON, because the branch was measured arming on a picture that was full of
    # intentional structure: 16-bit pixel art, where `detail_abs` ran 1.4x to
    # 3.3x the flatness threshold at every step that armed, and the escape then
    # mushed the pixel grid the prompt had asked for.  Clearing the derived
    # threshold cannot be satisfied by "the grid stopped getting busier", which is
    # what the growth term alone was reading.  OFF restores the ungated branch,
    # for anyone who found it useful on a picture that is not flat.
    "escape_momentum_flatness": True,
    # WHETHER THE MOMENTUM ARM'S GROWTH BUDGET IS DERIVED FROM THE PROBE instead
    # of used as the constant above.  OFF by default: it is a test arm, and the
    # default job is to not break.
    #
    # When ON, the budget becomes
    #     escape_momentum_growth + escape_growth_probe_open * (1 - commit)
    # where `commit` is how consistently the probe measured this model's
    # trajectory moving in ONE direction over the phase where it starts
    # reversing (`dir_cons`, clamped to [0, 1]).  A model that commits gets the
    # shipped constant and behaves exactly as it does today; a model whose
    # trajectory reverses throughout gets the budget opened up, because that
    # reversal is this engine's own stated signature of being stuck ("every pixel
    # moves, just back and forth" -- see `_global_stuck_signal`).
    #
    # WHY A COMMITMENT MEASUREMENT RATHER THAN A FIXED BUDGET, measured on 31 runs
    # of the build under study: the ratio that separates the runs where the arm
    # fired from the runs where it did not is the probe's `dir_cons` across its
    # FULL range -- firing runs peak at -0.32 or below, silent runs at +0.60 or
    # above -- and that curve is already measured per model AND per prompt.  A
    # constant cannot express that difference; a per-checkpoint table would be a
    # guess wearing a measurement's clothes.
    "escape_growth_probe": False,
    # How much extra growth budget a FULLY reversing model gets, on top of
    # `escape_momentum_growth`.  SAFE FALLBACK: fixed constant, chosen so the
    # opening clears the measured spread rather than to be optimal, because a
    # user cannot verify optimality on their own model.  Measured on the 31 runs
    # of build e95247d8a13e in the turbo/adapter regime: the detail growth at the
    # moment the movement collapses is -0.12 on the 50-step checkpoint and
    # +0.17..+0.35 on the 16-step turbo checkpoints, so 0.15 (the shipped budget)
    # sits inside that spread and 0.60 clears its top.  A POINTER to the
    # measurement, not the measurement: `rig_momentum_gate.py` carries it.
    "escape_growth_probe_open": 0.60,
    # ---- THE DERIVED NOISE SCHEDULE (measurement-only until switched on) ----
    # WHETHER eta COMES FROM THE PROBE'S OWN NOISE-ABSORPTION MEASUREMENT instead
    # of the escape policy.  OFF, so the shipped behaviour is untouched until it
    # is asked for.
    #
    # When ON, eta(sigma) = min(1, (budget / k(sigma))**2), with `k` the per-sigma
    # gain the probe measured (see `_eta_from_probe`).  It REPLACES the escape for
    # that run rather than adding to it: the point of deriving "how much noise can
    # this model absorb here" is that the question "is this trajectory stuck"
    # stops being needed.
    #
    # MEASURED BASIS for the switch existing at all: across five checkpoints, k
    # runs ~1.35 at sigma 0.66 and ~0.31 at sigma 0.28 on the same checkpoint, so
    # a budget of 0.10 asks for eta 0.0055 at the start of a run and 0.072 in the
    # middle -- against a shipped fixed eta of 1.0.  The escape layer was tuned to
    # approximate this from the wrong quantity (whether the image looked stuck),
    # which is why it needed a gain, a mode, a growth budget and a flatness gate.
    "eta_from_probe": False,
    # THE RESIDUAL BUDGET: how much of an injected perturbation may still show in
    # the model's next prediction, as a fraction of an unperturbed reference.
    # SAFE FALLBACK: fixed constant, chosen to be SAFE rather than optimal, because
    # a user cannot verify optimality on their own model.  Measured on the five
    # checkpoints of the ETA-options builds in the 8-50 step turbo regime, where
    # the probe's own pass/fail threshold sits at 0.35 -- so 0.10 is well inside
    # what that experiment already treated as corrected, and the schedule it
    # produces is 6x to 180x below the shipped fixed eta.  A POINTER to the
    # measurement, not the measurement: `rig_eta_response.py` carries it.
    "eta_probe_budget": 0.10,
    # WHETHER THAT BUDGET COMES FROM THE PROBE INSTEAD OF THE WIDGET.  OFF, so the
    # widget governs by default.
    #
    # ON uses `safe_fresh_frac` -- the profile's own measurement of how big a kick
    # this model still corrects, at a fixed 0.35 divergence threshold.  That is the
    # SAME quantity and the same units as `eta_probe_budget`, and it is already
    # measured on every probed run for `calib["safe_eta"]`; the widget's 0.10 is an
    # invented default for a number the engine measures anyway.
    #
    # It is still a SAFE FALLBACK rather than a derivation of what looks good: the
    # measurement says what the model CORRECTS, not what a viewer accepts, and no
    # measurement can settle the second.  When no profile carries a usable value
    # the widget's number is kept, so switching this on can never leave the
    # schedule without a budget.
    "eta_budget_from_probe": False,
    "wmax_base": 2.0,
    # SIGNED RESPONSE OF THE EXTRAPOLATION AUTHORITY TO THE `eg_n` ENDGAME EVIDENCE.
    # +1.0 = the shipped behaviour (pin `wmax_limit` to its ceiling -- see the use site);
    #  0.0 = stand the response down, leaving the `wmax_adapt` feedback alone;
    # -1.0 = INVERTED, i.e. a high endgame error LOWERS the authority to `wmax_base`.
    # It is a parameter rather than a decision because the sign is not decidable by reading
    # the code: it is decided by which sign moves the trajectory closer to the true ODE
    # solution.  scratch_wmax_authority_dir_rig.py measures that against an analytic limit.
    # MEASURED 2026-10-03, and the answer is to LEAVE THIS AT +1.0: inverting is worse on all
    # three analytic flows at N=16 (Linear +47.2%, TwoScale +0.16%, Nonlinear +1.67%) while
    # moving `wmax_limit` on 14 of 16 steps, so it is a real effect and not a silent arm.
    #
    # WHAT THE SAME RIG SHOWED, and it is why the analytic score is NOT the whole story: against
    # standing the branch DOWN, +1.0 changes the analytic distance-to-limit by only
    # +0.15% / 0.000% / -0.14%.  **That is NOT the same as "the switch does nothing"** -- an earlier
    # version of this comment said "the limit moves and the trajectory barely does", and the
    # operator's own images refute it.  Measured on five runs of one batch (mklanKrea2_v10, build
    # fccd7e2a8a18): +1 and +0.5 differ by 15.81 RMS in the DECODED IMAGE while their final latent
    # energies are 0.30% apart, because `wmax_limit` moves the PER-PIXEL order admission (63.62%
    # against 62.68% of pixels at step 4) -- a spatially distributed change that a global scalar or
    # an energy spectrum cannot see.  **The lesson: a single-field analytic flow scores an aggregate,
    # and this switch acts on a spatial pattern.**  See NOTES `## THE OPERATOR'S EYE IS RIGHT`.
    #
    # THE SIGN AT SHIP TIME IS ALSO THE OPERATOR'S PREFERENCE, on many runs and two model families:
    # "again, my preference is the 1 setting. Much better image quality/less artifact forming too."
    # The lever for the LEVEL rather than the sign is `wmax_base` (rig section 4), and the response
    # SATURATES: everything above an effective limit of about 3.8 behaves the same, so `+0.5` and
    # `+1.0` are within 0.30% in the latent.
    #
    # DEFAULT CHANGED TO 0.0 ON THE OPERATOR'S RULING, 2026-10-03 (it was +1.0).  Their words:
    # *"Seems to be in a good spot though. lf_ab true, order by bound true, order floor 6,
    # wmax_og_dir 0."*  The APPEARANCE call is theirs and rule 14 is explicit that this metric
    # cannot make it, so the analytic result below is kept as what it is -- a statement about
    # distance to an exact ODE solution, which is not the same question as how an image looks.
    #
    # WHY +1.0 WAS SHIPPED, still true and still worth reading: `+1.0` changes the analytic
    # distance-to-limit by only +0.15% / 0.000% / -0.14% against standing the branch DOWN, and
    # an earlier comment here read that as "the limit moves and the trajectory barely does".
    # The operator's own images refute that reading -- +1 and +0.5 differ by 15.81 RMS in the
    # decoded image while their final latent energies are 0.30% apart -- because the switch moves
    # the PER-PIXEL order admission, a spatially distributed change that a global scalar cannot
    # see.  So the switch is NOT inert, and `0` is a different image, not the same one.
    #
    # NOT A CACHE-KEY MEMBER, deliberately -- rule 28 asks whether a switch feeds a value that
    # is CACHED, and this one derives nothing: it reads `lf_evid["eg_n"]` and writes a per-step
    # local.  The probe profile is identical under every value of it, so listing it in
    # `_schedule_sig` / `_node_probe_param_sig` would fragment the probe cache (a fresh probe,
    # 10 model calls) for no measurement reason.  It is read from `cfg` at the use site and is
    # deliberately NOT a module global, which is the leak this rule also exists to prevent.
    "wmax_eg_dir": 0.0,
    # DERIVED-CONSTANT A/B ARMS -- see the block above `_derive_calibrations` for the
    # derivations and the rigs.  Both the shipped and the derived value are logged either way
    # so a single run shows the delta.
    #
    # `pct_derived` DEFAULTS ON because it is a bug fix, not a preference: the shipped
    # estimator `a[int(p*n)]` makes p90 IDENTICAL to the median for n <= 2, which silently
    # collapsed guard_floor to its r = 1 value on every Krea cond profile.  Percentiles are
    # supposed to be percentiles.  Set it False to A/B back to the old behaviour.
    "pct_derived": True,
    # `wmax_bound` DEFAULTS OFF because it changes WHICH EXTRAPOLATION ORDERS the engine
    # accepts -- i.e. the stepping itself -- which is an A/B, not a fix.  It is now
    # functional: profile["dk_scale"] supplies the per-order derivative scale it requires.
    # IT IS ALSO A WIDGET ON THE DEBUG OPTIONS NODE since 2026-10-18 (audit C2): while the
    # key was frozen off, the per-step fusion's "lower cap when the probes disagree" rule
    # could not run, because that disagreement is judged on the caps THIS arm derives.
    "wmax_bound": False,
    # THE OPERATOR'S NUDGE for the one input to the order bound that is NOT a measurement.
    # `dk_scale` is measured on the PROBE grid as a trajectory statistic, so it is not a
    # per-step upper bound (scratch_extrap_remainder_rig.py: 58 of 204 cases under-bound
    # with the shipping median, up to 578-1807x at individual steps), and the transfer of
    # that scale to the real grid cannot be verified without spending NFE.  This multiplier
    # scales ONLY the tolerance the remainder gate compares against -- never `rtol`, which
    # the corrector also consumes.  1.0 is neutral and bit-identical to no multiplier;
    # >1 admits higher orders at the same measured error, <1 is more conservative.
    "wmax_budget": 1.0,
    # THE COND-PATH LOOSENING MULTIPLIER (widget on the Debug Options node).  1.5 = shipped behaviour,
    # bit-identical to the literal it replaced; 1.0 = the loosening OFF (nothing is multiplied).  The
    # shipped CONDITION (`gap.strength < 0.5`) is kept unchanged around it, and that condition never
    # discriminates (see `_COND_RTOL_LOOSEN` above), so in practice this knob is the whole question: does
    # loosening the cond-path tolerance help or not.  Range and its derivation: `_COND_RTOL_LOOSEN`.
    "cond_rtol_loosen": 1.5,
    # THE ORDER SYSTEM'S EMA WEIGHT (widget on the Debug Options node).  Shipped **0.37** since
    # 2026-10-04, which IS the measured AR(1) optimum (CONSTANTS_AUDIT section 15: the logged `ord_innov`
    # series, n=682, has lag-1 autocorrelation 0.630, and for an exponential smoother `alpha* = 1 - rho_1`
    # = 0.370).  It was 0.5 -- 0.13 stiffer than the measurement wants -- until the A/B that section asked
    # for: the operator's controlled triple, a bit-identical replicate pair, a 3.33e-1 latent difference on
    # the arm, and his eye preferring 0.37.  Range 0..1 is DEFINITIONAL: it is a convex blend weight
    # between the previous EMA and the new sample.
    "ord_ema": 0.37,
    # FORCED PER-PIXEL ORDER FLOOR (widget on the Sampler node).  1.0 = shipped behaviour and
    # bit-identical: the branch below is not entered at all.  Above 1 it RAISES the order the
    # ladder would otherwise use -- as a DEMAND, never an override: everything downstream
    # that can refuse an order (the weight cap, and with `wmax_bound` ON the remainder bound)
    # still applies, so the floor cannot smuggle in an order the evidence rejects; and a pixel
    # `reset_m` snapped to 1 stays at 1, because that reset is the ladder's own response to
    # novelty or an anomaly.
    #
    # WHY IT EXISTS: the ladder's criterion judges PREDICTION (`ord_innov`, aflops.py:8757),
    # not USEFULNESS, and on the operator's own model the higher orders predict x0 1.4x-46x
    # worse than order 1, so the ladder sits at ~1.03 and nothing downstream can move it.
    # Whether a higher order makes the actual STEP worse is a different question that no
    # logged field answers -- this knob is what answers it, by forcing the order up and
    # reading the per-step `err` the engine already records.
    #
    # DEFAULT CHANGED TO 6.0 ON THE OPERATOR'S RULING, 2026-10-03 (it was 1.0, i.e. off).
    # This is ROUND LOG item 3, the release-defaults decision, and this is that decision.
    # CHANGED TO 0.0 ON THE OPERATOR'S RULING, 2026-10-05: AUTOMATIC IS THE DEFAULT.  Basis: 26 judged
    # prompts across two batches -- the error-gated automatic mode won 61 % of head-to-heads against the fixed
    # demand and was NEVER his worst pick in the first eight, while the fixed demand is bimodal (best or worst,
    # never middle).  1..6 remain the fixed demand; 0 means automatic.
    "order_floor": 0.0,
    # THE AUTOMATIC MODE'S DEMAND MAGNITUDE, now its OWN key.  It used to be read from `order_floor`, which was
    # safe only while the default was 6 -- flipping the default to 0 would have set the demand to 0 and made
    # AUTOMATIC inert without a single error message.  One key, one meaning.
    "order_floor_adaptive_demand": 6.0,
    # ---- THE GRADED DEMAND'S PARAMETERS, DERIVED FROM THE CORPUS (2026-10-05), NOT CHOSEN ----------------
    # Measured on the gated runs (order_floor = 0, order_demand_q = 0): ABOVE the trigger the order the engine
    # actually uses is p25 4.17 / p50 4.53 / p75 4.88 (240 steps), while AT OR BELOW it the usage is p25 1.81 /
    # p50 2.41 / p75 2.85 (361 steps).  So (a) usage really does rise with the error, which is what makes a
    # graded demand coherent, and (b) a flat demand of 6 asks for MORE order than the engine uses anywhere in
    # the open region -- it over-asks at the bottom of that region and cannot differentiate at the top.
    # lo/hi are the p25/p75 of that usage; r_hi is the p90 of the ratio above the trigger.
    "order_floor_adaptive_graded": True,
    "order_floor_adaptive_lo": 4.0,
    "order_floor_adaptive_hi": 5.0,
    # CHANGED 0.3031 -> 0.12 ON THE OPERATOR'S CLOSE-OUT RULING, 2026-10-05 (the "known good settings").
    # Evidence: batch 6 ACCEPTED the r_hi = 0.12 cell and rejected the 0.3031 one ("your prediction was right,
    # r_hi seems good"); batch 8 rated both well, so 0.12 has the better record; and on the corpus the 0.3031
    # ramp reaches full strength on only ~10 % of its demanding steps against ~78 % for 0.12 -- it approaches a
    # correct target lazily, and the accepted cell approaches it promptly.
    "order_floor_adaptive_r_hi": 0.12,
    # THE ADAPTIVE MODE'S THRESHOLD -- the only new constant this feature adds.  `order_floor = 0` means
    # AUTOMATIC: demand the shipped order above while the run's own error is HIGH, and demand nothing once
    # it settles.  The signal is the dimensionless `err/tol` the engine already computes every step, carried
    # one step forward because step k's `err` is only known at the END of step k.
    # THE VALUE IS MEASURED, NOT CHOSEN: on the operator's 16-prompt order_floor batch (build af9e1b8839,
    # 768 step records) that ratio has median 0.0130, p75 0.0505, p90 0.2944, max 1.1582, and decays within
    # a run (early 0.1433 -> mid 0.0553 -> late 0.0457).  p75 therefore means "demand the high order in the
    # worst quarter of the steps", which acts early and releases -- the shape the operator's own read of the
    # batch predicts.  PROVISIONAL BY CONSTRUCTION: the value in force is stamped per step
    # (`ord_floor_rtol`), so a batch tests its neighbours instead of arguing about them.
    "order_floor_adaptive_rtol": 0.05,
    # THE BOUND-DERIVED REQUEST'S STRENGTH, and 0.0 IS OFF.  When `order_floor = 0` (AUTOMATIC) and this is
    # above 0, the per-step request becomes the q-QUANTILE OF THE PER-PIXEL VALIDITY BOUND instead of the error
    # gate's binary 6-or-1.  WHY THE BOUND, MEASURED: over 544 step records corr(`err/tol`, granted `ord_max`)
    # is only +0.059 and flat across ratio thirds (4.34 / 4.47 / 4.50, median 5 in all three), so the ratio can
    # say WHETHER to ask but not HOW MUCH -- while the bound IS the engine's own per-pixel answer to "what order
    # is valid at this pixel's smoothness" (`_pixel_order_bound`, `aflops.py:10306`).  q = 0.5 is the median and
    # the arm worth running first; q = 1.0 is the maximum, which is roughly what the fixed demand of 6 already
    # asks for.  0.0-1.0 is a quantile's whole range.
    "order_demand_q": 0.0,
    # USE THE BOUND'S OWN MAP AS THE REQUEST.  True (shipped) = on any step where the per-pixel
    # validity bound was computed, ask for NOTHING: the bound's map already is the per-step request,
    # so the error gate's ramp cannot push an order past what the run's own remainder criterion
    # justifies at that pixel.  On steps with no bound (no history yet, or `order_by_bound` off) the
    # ramp still applies, and `order_floor` must be 0 (AUTOMATIC) for either to happen.  False
    # restores the ramp everywhere and is the per-run revert.  A saved Debug Options node carries
    # its own value, so this default governs NEW nodes and runs that have no such node.  It changes
    # only what is DERIVED from measurements already held, so it is not part of any probe key.
    # SHIPPED ON THE OPERATOR'S RULING 2026-10-18, after the matched A/B pair: on the krea cell the
    # admitted order fell 3.19 -> 1.10 and the step error 7.45x tolerance -> 0.09x, with the safety
    # cut firing on 6 steps before and 0 after.  An explicit `order_demand_q` above 0 still wins --
    # a deliberate request must not be swallowed by a default.  Evidence: rig_order_demand_ab.py.
    "order_demand_bound": True,
    # WHICH CRITERION PICKS THE PER-PIXEL ORDER.  False (shipped) = the innovation argmin,
    # i.e. "the order whose extrapolation best PREDICTS x0".  True = the per-pixel Lagrange
    # remainder, i.e. "the HIGHEST order still valid at this pixel's own smoothness" -- the
    # Runge criterion.  They answer different questions: the first is dominated by pixel-level
    # noise that the higher orders amplify (weights 1.5 -> 5.41) and settles near order 1 on
    # real models; the second admits order 6 where the trajectory is smooth and order 1 where
    # it is not, which is what the operator's intent needs.  See `_pixel_order_bound`.
    #
    # DEFAULT CHANGED TO TRUE ON THE OPERATOR'S RULING, 2026-10-03 (it was False).  Their
    # LF x ORDER hypothesis was *"without LF order=1 generally is better, but with LF order
    # floor=6 seems better"*, and the ruling pairs it with `lf_enable=True` and
    # `order_floor=6.0` -- so the three ship as one decision, not three separate ones.
    "order_by_bound": True,
    # HOW STRONGLY THE RUNGE CRITERION IS APPLIED, as a blend between the two order maps
    # it can produce: 0.0 is the map the criterion does NOT touch (the `order_by_bound=OFF`
    # arm, exactly), 1.0 is the criterion's own map (the ON arm, exactly).  In between the
    # two are mixed in ORDER UNITS, so a pixel whose innovation order is 1 and whose bound
    # is 4 reads 2 at 0.5.  A TEST DIAL, not a tuned default: 1.0 is the only value with
    # measurements behind it, and the dial exists because the operator could SEE a
    # difference between the arms on a krea checkpoint but could not say which was better
    # -- a graded sweep is how that gets decided by eye rather than by a verdict imposed
    # on them.  It does nothing while `order_by_bound` is OFF, and it is bit-identical to
    # the shipped criterion at 1.0.
    "order_bound_blend": 1.0,
    # RUN THE EXPENSIVE NOISE-SURVIVAL PROBE?  None (default) = follow its consumer: the call
    # runs only while `eta` is set, because `safe_fresh_frac` -> `safe_eta` is consumed only
    # there. True/False force it. Kept as a key (not a deletion) so the planned
    # adaptive-local-ETA work can turn the measurement back on, and so an A/B can force it.
    "probe_noise_survival": None,
    # WHETHER THE NOISE-SURVIVAL PROBE KEEPS ITS PER-SIGMA SERIES.  Measurement
    # only: the scalar it has always returned is identical either way, and this
    # changes what the probe RECORDS, so it is deliberately not part of the probe
    # key.  OFF by default because it is an instrument, not a behaviour.
    "probe_noise_survival_curve": False,
    # THE ERROR FEEDBACK ON THE EXTRAPOLATION LIMIT -- DEFAULT CHANGED TO FALSE 2026-10-03.
    # With it True the limit is `wmax_base * (1 + k*smooth)` where `smooth = min(rtol/prev_err, 3)`, so
    # a GROWING error LOWERS the authority: error up -> authority down -> order down ->
    # under-resolution -> error up.  That is a positive feedback loop with the wrong sign, measured on
    # the operator's own runs -- their `wmax_eg_dir = 0` run is the ONLY arm where this feedback is live
    # and the only arm whose limit MOVED (4.500 -> 2.060, mean admitted order 3.82 -> 2.9 in the last
    # five steps).  See the use site for the full argument, the anchors verified bit-identical, and the
    # saturation that makes the knob's flat positive half harmless.
    # SET IT TRUE to restore the old adaptive arm -- that is how this change is A/B-ed.
    # NOTE ON A TRAP THIS ENTRY SPRUNG ONCE: `ENGINE_DEFAULTS` is a hand-written 247-line dict, and a
    # DUPLICATE KEY in a dict literal SILENTLY OVERWRITES the earlier one.  A first attempt to change
    # this default added a second `"wmax_adapt"` entry near `wmax_base` and the later `True` won, so the
    # edit did nothing at all.  `scratch_wmax_loop_rig.py` now asserts there are NO duplicate keys in
    # this dict, because the failure mode is invisible: the source reads as if the switch were set.
    "wmax_adapt": False,
    "wmax_adapt_k": 0.5,
    "wmax_mid_floor": 1.55,
    "s_tmin": 1.0, "s_tmax": 0.0,
    "rtol": 0.01,
    "err_mode": "rel_l2",
    # CORRECTOR A/B.  True = the current corrector.  False = the
    # archive-known-good corrector behaviour (see the switch in the step loop).
    # Default True so existing workflows are unaffected; flip it to compare the
    # two arms on the same build, same seed, same schedule.
    "corrector_ab": True,
    # ORDER A/B.  True = shipped per-pixel adaptive order ladder.  False = force
    # the pure order-1 exponential integrator (no extrapolation, no 2nd-order
    # slope).  No single variable could be shown to separate helpful from
    # harmful order-2 steps (scratch_histcos_gate_rig.py), so this A/Bs the
    # machinery itself on the real model.
    "order_ab": True,
    "euler_cut": True,
    # DERIVED CONSTANTS -- NOW THE DEFAULT (the A/B is closed).  It was an A/B
    # (`derived_ab`, default False = shipped hand-set values); the verdict is that the
    # derived forms replace the hand-set ones, so the switch is gone and this is what
    # ships.  Each derived form replaces a handwave with a closed form or a measured
    # crossing: the series switch -> sqrt(12*eps) of the working dtype, the x0m
    # envelope -> sum|w_i| (the extrapolation's own amplification bound), and the
    # default rtol -> the measured 1.536e-2 crossing.  Measured consequence of the
    # switch being ON, so nobody expects more than it delivers: two of the three are
    # INERT on real runs (the series branches agree below the rounding floor; the
    # envelope guard only acts when the norm overshoots and it does not on real
    # trajectories), and only the rtol floor changes anything -- on the 19 of 20
    # corpus models whose probe rtol sits below the crossing.  Set False only to
    # reproduce historical (pre-verdict) behaviour.
    "derived_ab": True,
    # THE TOLERANCE MINIMUM'S A/B -- OFF IS WHAT THIS BUILD DOES TODAY (2026-10-07).
    #
    # WHAT IT SELECTS.  Every step is held against a tolerance.  Two different things can
    # supply that number: the value measured for the model in front of the engine (the
    # Autotuner's probe) and a fixed MINIMUM of 0.01536 that the engine applies on top of it.
    # ON = the measured value governs alone, however small it is.  OFF = the minimum is also
    # applied, so a measured tolerance below 0.01536 is raised to it.  Above the minimum the
    # two arms are identical, so nothing changes at all for a model whose measured tolerance
    # is already higher.
    #
    # WHICH WAY IT GOES, AND HOW FAR.  Below the minimum, turning this ON makes the step
    # tolerance SMALLER than this build uses: the step loop accepts less error, so the
    # 2nd-order correction is cut back to the plain step more often.  MEASURED on the good-era
    # corpus (build stamp 91d196146f3d, 439 runs, 7 models), so the size of the change is known
    # before any render: the minimum was doing the work on 289 of the 302 runs that HAD a
    # measured tolerance (298 of all 439), and on those 289 the median tolerance moves by
    # between 1.37x and 2.77x depending on the model.  That is a large change on almost every
    # model, not a trim.
    #
    # WHY IT SHIPS OFF.  Both numbers are measurements and they disagree; the corpus can say
    # which one is in force, never which one looks better, and the engine's own error metric
    # does not track appearance.  So the default stays with what this build already does and
    # the other arm waits for a human eye.  This switch changes no other default and leaves
    # `_DERIVED_RTOL` itself untouched.
    #
    # SAFE FALLBACK: fixed constant, chosen to be SAFE rather than optimal, because a user
    # cannot verify optimality on their own model.  Measured on the 439 runs of the good-era
    # corpus (stamp 91d196146f3d).  Evidence pointer: DEAD_FEATURES.md, "THE FOUR CALIBRATIONS
    # ARE ALREADY AUTHORITATIVE -- NO ENGINE EDIT".
    "use_measured_tolerance": False,
    # WHICH QUANTITY THE EULER-SAFETY CUT COMPARES -- ALSO CLOSED, and this one is
    # the tail fix.  "corrected" = the corrected step vs the plain exact step, which
    # mixes a term the blend already bounds with the ladder's own move, and where the
    # ladder contributes nothing evaluates `tol > tol` (a rounding lottery, measured
    # firing on 3 of 5 step counts).  "midpoint" = the ladder's midpoint vs the plain
    # step: only the term nothing else bounds.  The verdict is "midpoint", on the
    # T4 rig (scratch_cut_variable_rig.py, 8/8: 1.44x-1.93x better where the shipped
    # cut fired, every earlier step bit-identical) and on three real Krea2 pairs where
    # the shipped cut fired 1-3 times and the derived cut fired NOT ONCE.  Set
    # "corrected" only to reproduce historical behaviour.
    "cut_var": "midpoint",
    "grad_strength": 5.0,
    "aflops_lam_min": -3.0,
    "aflops_lam_max": 0.0,
    "x0m_env": 1.3,
    "outlier_guard": True, "guard_sensitivity": 2.5, "guard_hold": 2,
    "guard_window": 8, "guard_floor": 0.10, "guard_ratio": 3.0,
    "guard_quantile": 0.99,
    "guard_mad_floor": 0.1,
    "guard_phase_relax": 1.0,
    # THE ANOMALY DETECTOR'S ON/OFF.  `an_enable` is now a SAMPLER WIDGET (2026-10-03), which is
    # why it is the one key in this group that a user can reach: the others (`an_from`, `an_win`,
    # the four thresholds) are DERIVED per run from the probe at `aflops.py:4533` and `:5169`,
    # so exposing them as widgets would let a fixed number silently override a measurement.
    #
    # WHY IT BECAME A WIDGET, and it is the operator's own ruling: a switch they must TEST cannot
    # live in `ENGINE_DEFAULTS`, because a default lives in the source and every arm of an A/B
    # then costs a ComfyUI restart.  The prompt for it was a labelled artifact: see
    # `## THE LABELLED OFFENDER`, where the detector fired at step 9 of ONE run out of five,
    # `lf_damp` was set to 0.9 on the resulting mask, and that mask -- dilated by 2 -- damps
    # those pixels' local-field correction for SEVEN steps.  Turning it off is the one-run
    # experiment that separates "the detector is the cause" from "the prompt is".
    #
    # WHEN IT IS OFF the mask simply never fires: `amask` stays None, `lf_damp` stays None, and
    # the guard's separate `new_damp` path (which needs `guard_trip`) is the only remaining
    # source of damping.  The `lf.damp` report field then reads 0 for the whole run, which is
    # what makes the A/B legible in the corpus.
    "an_enable": True, "an_from": 0.15, "an_win": 9,
    "an_spatial": 5.0, "an_novelty": 4.0, "an_value": 2.5,
    "an_phase_relax": 0.5,
    "an_region_z": 8.0,
    "lf_enable": True,
    "lf_guard_feed": True,
    "lf_closed_loop": True,
    "lf_runtime_gate": True,
    "lf_env_fade": True,
    "lf_spatial_health": True,
    "lf_band_refresh": True,
    "lf_dir_vol": True,
    "lf_semantic_prior": True,
    "lf_detail_rescue": True,
    "lf_dir_real_feedback": False,
    "lf_drift_budget": 0.0,
    "rescue_cap": 0.35,      # rescue envelope = cap * clamp (<= ~5%/step)
    "rescue_start": 0.55,    # kn where detection arms
    "rescue_full": 0.72,     # kn where rescue authority reaches full
    "rescue_max_cov": 0.35,  # coverage fraction where the gate shuts off
    "rescue_floor": 0.02,    # min block delta magnitude, fraction of sigma
    "dwell_mode": "auto",   # per-pixel dwell: off | auto | manual (auto arms only with probe evidence)
    "dwell_max": 0.20,       # manual: max fraction of a step a pixel withholds
    "dwell_until": 0.65,     # manual: run fraction where the dwell window ends
    "probe_distributions": True,
    "probe_endgame": True,
    "probe_dir_cons": True,
    "eta": None, "s_noise": 1.0, "noise_space": "auto",
    "noise_color": "auto", "noise_beta": 1.0,
    "color_schedule": "constant",
    "color_beta_max": 1.5, "color_beta_min": -0.5,
    "color_temporal": False, "color_temporal_beta": 0.0,
    "color_decay_floor": 1e-3, "color_dc_zero": True,
    "color_strength": 1.0,
    "adaptive_tau_eta": 0.15,
    "adaptive_tau_err": 0.5,
    "adaptive_tau_snr": 2.0,
    "adaptive_cons_gamma": 1.5,
    "color_conc_gate": True,
    "color_conc_strength": 1.0,
    "_model_info": None,
    "probe_enabled": False,
    "probe_steps": 10,
    "probe_resolution": 48,
    "probe_force": False,
    "probe_tune_guard": True,
    "probe_tune_tol": True,
    "probe_tune_wmax": True,
    "probe_tune_anomaly": True,
    "probe_tune_eta": True,
    "probe_tune_lf": True,
    "cond_probe_enabled": False,
    "cond_probe_steps": 4,
    "cond_probe_resolution": 16,
    "cond_probe_gap": True,
    "cond_probe_weight": 0.5,
    # THE PER-STEP PROBE FUSION (2026-10-08).  OFF reproduces the shipped behaviour exactly:
    # the two probes' calibrations are combined ONCE into scalars (a fixed-weight mean of
    # `rtol`/`guard_floor`/`wmax_base`/`an_*`, cond applied over model).  ON fuses the two
    # probes' per-step PREDICTIONS instead -- resampled onto this run's sigmas, weighted by a
    # reliability prior updated at every step from the run's own measured jump -- and derives
    # the ORDER CAP per step from the fused curvature, taking the conservative (lower) cap when
    # the two probes' caps imply different orders.  It costs NO NFE: every input is a stored
    # profile field or a reduction of a tensor the run already has.
    # SCOPE, stated because a switch that claims more than it does is the defect this project
    # keeps paying for: ON currently reaches the ORDER CAP ONLY.  `guard_floor` and the `an_*`
    # thresholds stay the averaged scalars (their per-step evidence is not stored -- the probe's
    # per-step GUARD signal is computed and discarded), and the step's acceptance tolerance `tol`
    # is untouched in BOTH arms.  It is a TEST ARM, not established as better: nothing here can
    # say which looks better, so it ships OFF and the sampler node exposes it as a toggle.
    "probe_fuse_per_step": False,
    # WHETHER THE FUSED CAP MAY RAISE THE ORDER, and it is a separate switch because it is a
    # SEPARATE CHANGE.  OFF (the default): the fused cap may only LOWER a step's cap, never raise
    # it above the cap the run would have used anyway -- so switching the fusion on is a strict
    # refinement of today's behaviour, and the two arms can be compared without a second
    # difference riding along.  ON: the fused curvature may ADMIT A HIGHER ORDER than the cap in
    # force, which also turns on the probe-derived order bound in the cases where that bound's own
    # flag left it off -- so ON is two changes at once, and anything you see in it cannot be
    # attributed to the fusion alone.  TEST ARM, not a recommendation.
    # What it was set to is written into every run's log (`probe_fuse.cap_raise_allowed`).
    "probe_fuse_raise_cap": False,
    "cond_probe_tune_guard": True,
    "cond_probe_tune_tol": True,
    "cond_probe_tune_wmax": True,
    "cond_probe_tune_anomaly": True,
    "cond_probe_tune_eta": True,
    "cond_probe_tune_lf": True,
    "cond_probe_force": False,
    "cond_probe_cfg": 7.0,
    "cond_probe_focused": True,
    "history_consistency_gate": True,
    "history_consistency_threshold": 0.0,
    # PER-PIXEL HISTORY-CONSISTENCY GATE.  ON (shipped): the cosine is computed per pixel
    # over channels and only the locally reversing pixels go to order 1, so a volatile
    # corner no longer removes the ladder from a detailed area.  OFF is the old whole-image
    # gate: one negative cosine sets `eff_cap=1` for every pixel, which the saved runs show
    # switched the ladder off on 88.8% of the steps that were about to use order 2 or more.
    # DEFAULT ON (operator-verified): the paired corpus runs and the operator's eye agree on
    # anima, krea and moss (2026-10-19); the ON arm raises the admitted order on the gated
    # steps, at the cost of a higher `err` the operator accepted for the image.  A/B SWITCH:
    # set False to restore the old whole-image gate; the per-step log stamps
    # `hist_cos_px_neg` (the gated fraction) and the run stamp `hist_cos_per_pixel`.
    "hist_cos_per_pixel": True,
    # PER-PIXEL SECOND-ORDER TERM.  ON (shipped): the exponential integrator's slope term
    # is applied only where the admitted order is >= 2, so the volatile pixels the order map
    # keeps at 1 stay on the first-order exponential and the stale/detailed pixels keep the
    # second-order term.  OFF is the old global term (every pixel whenever `eff_cap >= 2`).
    # DEFAULT ON (operator-verified): the operator tested both arms and preferred ON; the
    # per-step log stamps `second_order_px` (the fraction that receives the term) and the run
    # stamp `second_order_per_pixel`.
    "second_order_per_pixel": True,
    # RAW-LAMBDA SECOND-ORDER SLOPE.  ON (shipped): the slope term's decomposition
    # (h_n, h_prev, slope, phi2) uses the UNCLAMPED lambda, while the first-order decay
    # stays clamped; OFF is the clamped decomposition.  DEFAULT ON is the operator's
    # choice after the paired runs.  The corpus pairs show the two arms are IDENTICAL
    # where the clamp does not bind (krea 16-step) and a small error reduction where it
    # binds (moss 20/50 steps, anima 3/50); the appearance call is the operator's, not
    # this comment's.  A/B SWITCH: OFF restores the clamped form; the per-step log stamps
    # `second_order_raw` and the run stamp `second_order_raw_lambda`.
    "second_order_raw_lambda": True,
    # NAME READS AS ONE THING, DOES ANOTHER: this blends the two candidate EXTRAPOLANTS
    # (order o and order o-1, per pixel, weighted by the last two innovations' cosine), not
    # the orders themselves -- the order is already a single integer per pixel by then.  See
    # the use site for the full note.  The key name is kept for saved graphs.
    "order_pixel_blend": True,
    "order_pixel_blend_sharpness": 1.0,
    "order_ladder_auto": True,
    "order_pixel_adaptive": True,
    # CEILING ON THE PER-PIXEL ORDER LADDER.  3 ships; 0 means the old ceiling of 10.
    # SAFE FALLBACK: fixed constant, chosen to be SAFE rather than optimal, because a user
    # cannot verify optimality on their own model.  Measured on a synthetic flow with a
    # slow, finely detailed half and a fast, volatile half (rig_detail_preservation_out.txt):
    # order 3 kept the fine detail 4.2x better than order 4, while 4 and 5 let the volatile
    # half overshoot its target envelope.  Not measured on a population of checkpoints.
    # A per-model value derived from the probe is not implemented yet; until it is, 3 is the
    # safe choice and 0 restores the old behaviour.
    "order_max": 3,
    "order_elev_gain": 0.10,
    "order_demote_gain": 0.25,
    "order_novelty_k": 4.0,
    "order_reversal_cos": 0.0,
    "order_reversal_mag": 1.0,
    "order_jema_decay": 0.7,
    "order_reset_on_anomaly": True,
    "probe_is_distilled": "auto",
    "cond_probe_is_distilled": "auto",
}
def _progress(callback, ptype, phase, i=0, n=1, msg="", extra=None):
    if callback is None:
        return
    try:
        evt = {"type": ptype, "phase": phase, "i": int(i), "n": int(n),
               "msg": msg}
        if extra:
            evt.update(extra)
        callback(evt)
    except Exception:
        pass


def _aflops_close_run():
    """Close this run's per-run records, on success OR on failure.

    `_ab_reach_end` leaves this run's counters in the carry and empties `_AB_REACH`,
    so the next run's reset carries only the counters its own pre-engine phase
    recorded; `_CALIB_AB` is zeroed for the same reason as the reset at the top of
    the engine.  Both are bookkeeping -- no sampling decision reads either.

    It is a separate function because the two exits need it: `_aflops_engine_body`
    calls it after its last step, and `aflops_engine` calls it from an exception
    handler (E7).  Before that, teardown ran only on success, so an aborted step loop
    left `_AB_REACH` populated and the NEXT run's log carried the aborted run's
    counters and calibration arms.
    """
    _ab_reach_end()
    _CALIB_AB.update({k: 0 for k in _CALIB_AB})


def _aflops_engine_body(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, cfg=None, log_errors=False):
    cfg = {**ENGINE_DEFAULTS, **(cfg or {})}
    global _LAST_CFG
    extra_args = {} if extra_args is None else extra_args
    # ---- THE MODEL'S OWN TYPE, WHEN THE GRAPH DID NOT SUPPLY IT (U02/435) ----
    # The node-side block only fills `_model_info` when the MODEL is linked to the
    # Autotuner or the Sampler, and the engine's space then fell back to
    # `_detect_space`: "flow if sigmas[0] <= 1.5".  A TRUNCATED VE SCHEDULE (img2img,
    # low denoise) starts below 1.5, so it read as FLOW and the probes measured in
    # different units from the run.  The engine already holds the model object, so the
    # facts are one `_extract_model_info` away -- no model calls, a handful of getattr.
    #
    # THE SPACE IS RESOLVED HERE, ONCE, BEFORE THE PROBES RUN.  The three probes call
    # `_detect_space` with `cfg["noise_space"]`; writing the resolved value into that
    # key is what makes the probes and the run agree.  The user's request is kept in
    # `_noise_space_requested`, so a log still says "auto" was asked for.
    if cfg.get("_model_info") is None:
        try:
            _mi_engine = _extract_model_info(model)
            if _mi_engine is not None:
                cfg["_model_info"] = _mi_engine
                cfg["_model_info_src"] = "engine"
        except Exception as _e_mi:
            logging.debug("[PD-AOS] engine model_info unavailable: %s", _e_mi)
    _mi0 = cfg.get("_model_info") or {}
    _req_space = str(cfg.get("noise_space", "auto"))
    if _mi0.get("is_flow") is not None and _req_space == "auto":
        cfg["_noise_space_requested"] = "auto"
        cfg["noise_space"] = "flow" if _mi0["is_flow"] else "ve_edm"
        cfg["_space_src"] = "model_info"
    elif _req_space != "auto":
        cfg["_space_src"] = "user"
    else:
        cfg["_space_src"] = "first_sigma"
    # ---- THE GUIDANCE IN FORCE (U05/1361) ------------------------------------
    # The probe is a real sampling run THROUGH the guider, so CFG changes what it
    # measures.  Read it once here, before the probe key is built and before any
    # probe runs, so `_schedule_sig` can key on it and `_profile_serves_model_probe`
    # can refuse a profile measured at another CFG.
    try:
        _cf_eff = _probe_effective_cfg(model, extra_args)
    except Exception:
        _cf_eff = None
    if _cf_eff is not None:
        cfg["_probe_cfg_eff"] = round(float(_cf_eff), 4)
    # ---- per-run phase timing (see _PHASE_MS for why this exists) -----------
    # A FRESH dict, not `.clear()`: the log keeps a reference to this object, so
    # clearing it in place would make the previous run's record describe THIS one.
    global _PHASE_MS
    _PHASE_MS = {}
    # SEED FROM THE NODE.  The Autotuner's node-side probe ran before this
    # sampler, so its time belongs to this queue's record -- and it is the largest
    # single block of what the operator sees as "53 seconds for the autotuner".
    # Taking a copy and clearing the source keeps each queue's node work in the
    # record of the run it preceded, exactly once.
    try:
        _PHASE_MS.update(_NODE_PHASE_MS)
        _NODE_PHASE_MS.clear()
    except Exception:
        pass
    _t_engine = _t0()
    # Cumulative probe-forward-pass counter at engine entry.  Its DIFF from the
    # previous entry is the NODE-side probe's NFE -- work done before this run
    # began, which a per-run delta structurally cannot see.  `probe_model_calls`
    # below is the engine's own share, written once per step so it is current at
    # the last step without needing an end-of-run hook.
    try:
        _PHASE_MS["probe_calls_node_side"] = (
            int(_PROBE_CALL_COUNT[0]) - int(_LAST_ENGINE_CALLS[0]))
        _LAST_ENGINE_CALLS[0] = int(_PROBE_CALL_COUNT[0])
    except Exception:
        pass
    _PHASE_MS["probe_calls_at_entry"] = int(_PROBE_CALL_COUNT[0])
    _PHASE_MS["probe_model_calls"] = 0
    # Each run's A/B reachability record describes THAT run only.  Reset must
    # happen HERE, at the very top: the rtol site is evaluated before the loop
    # (and before _LAST_ERRORS.clear below), so resetting later silently erased
    # its counter -- caught by scratch_ab_reach_rig.py, which is exactly the
    # class of mistake this instrumentation exists to make visible.
    _ab_reach_reset()
    # THE CALIBRATION COUNTERS ARE PER-RUN TOO.  `_CALIB_AB` counts how many calibrations took each arm
    # and is copied into the profile as `calib_ab_counts`, so a corpus reader can tell which arm a
    # profile's numbers came from.  It was never zeroed, so the stamp reported the whole PROCESS's
    # running total instead: over the good-era corpus one process's stamp climbs to `pct_derived` 804
    # (scratch_fix_logging_1_3_out.txt section 4), i.e. it counts every calibration derived since
    # ComfyUI started rather than this run's.  Zeroed here beside `_ab_reach_reset`, and again when the
    # run closes (see the end of this function), so the window a stamp describes is exactly one run --
    # the calibrations derived for it node-side before this entry, and the ones derived after it.
    _CALIB_AB.update({k: 0 for k in _CALIB_AB})
    # THE ORDER GATE'S COUNTERS BELONG TO ONE RUN TOO.  They were left cumulative when the
    # gate shipped, so every run after the first in a ComfyUI session reported the running
    # TOTAL: measured on the operator's own runs, a run with `wmax_bound=False` reported
    # `on=107` (carried over from the previous run) while the first run of the session
    # correctly reported `on=0`.  A counter that answers "did this toggle fire" has to be
    # per-run or it answers a different question.  scratch_order_gate_smoke_rig.py MISSED
    # this because the rig zeroed the counters itself before each run -- the harness was
    # doing the engine's job.  scratch_wmax_bound_onoff_rig.py now checks it from the
    # CORPUS instead, where no harness can hide it.
    _ORDER_GATE_AB.update({k: 0 for k in _ORDER_GATE_AB})
    # A/B experiment switches, applied FIRST -- before anything can compute or
    # cache a trajectory_state.  This must stay at the top of the function: the
    # probe block below caches `trajectory_state` (including oversmooth) into
    # _PROBE_PROFILES, so setting these later made every run use the PREVIOUS
    # run's mode -- a one-run lag that silently invalidated a 6-run A/B.  See
    # scratch_oversmooth_ab_rig.py, which asserts this ordering.
    #
    # NOTE the ordering fix alone was NOT enough, and the corpus proves it: on an
    # exact probe-cache hit the probe never runs, so ordering is irrelevant and
    # the run consumes the cached state directly.  The modes are therefore also
    # in the cache key itself (`_schedule_sig`).  See
    # scratch_oversmooth_engine_mode_rig.py for both mechanisms.
    _apply_ab_modes(cfg)
    _env_fp = os.environ.get("AFLOPS_FOCUSED_PROBE")
    if _env_fp == "0":
        cfg["cond_probe_focused"] = False
    elif _env_fp == "1":
        cfg["cond_probe_focused"] = True
    auto_tunable = {k for k in ENGINE_DEFAULTS
                    if cfg.get(k) == ENGINE_DEFAULTS[k]}

    # ---------------------------------------------------------------------
    # STANDALONE MODE -- the Autotuner is not connected.
    #
    # The Autotuner is what emits `options`, and `options` is what turns the
    # probes on.  With neither probe enabled there is NO source for the
    # probe-derived tuning levels, so every subsystem that REQUIRES them must be
    # switched off explicitly rather than left to fall back on an absolute
    # ENGINE_DEFAULTS constant.  Measured basis for that rule: absolute
    # thresholds on measured quantities are unreliable across models (the
    # documented 2-9x probe inflation), and the Krea2 family alone spans
    # hit p50 0.346..0.427 across 12 checkpoints while the PROMPT moves the same
    # quantity ~3x more than the checkpoint does.
    #
    # WHAT IS DISABLED, and why each one:
    #   lf_enable       -- its gains are derived FROM probe evidence; with
    #                      evidence.source == "none" the field already stood down,
    #                      so this only makes an implicit behaviour explicit.
    #   dwell_mode      -- "auto arms only with probe evidence"; same reasoning.
    #   eta_escape_jump -- "-1.0 = auto: derived from the probe's med_jump_rel".
    #   eta_fade_sigma  -- "-1.0 = auto: derived from the probe's conv_sigma".
    #                      Both autocues have NO source standalone.
    #   probe_tune_*    -- nothing to tune from; turning them off cannot change
    #                      behaviour but stops a future stale profile from being
    #                      applied as if it were this run's measurement.
    #
    # WHAT IS DELIBERATELY **NOT** DISABLED, because disabling it would be a guess
    # and this project does not ship guesses: the outlier guard and the anomaly
    # detector are run-side safety nets.  They stay ON and now run on
    # ENGINE_DEFAULTS -- which is reported so the operator can SEE that their
    # thresholds are untuned rather than believing them to be probe-derived.
    # Whether they misbehave standalone is UNMEASURED (the corpus holds exactly one
    # standalone run), so they are left alone until there is evidence.
    _probe_any = bool(cfg.get("probe_enabled")) or bool(cfg.get("cond_probe_enabled"))
    cfg["_standalone"] = not _probe_any
    if cfg["_standalone"]:
        cfg["lf_enable"] = False
        cfg["dwell_mode"] = "off"
        cfg["eta_escape_jump"] = 0.0
        cfg["eta_escape_detail"] = 0.0
        cfg["eta_fade_sigma"] = 0.0
        for _tk in ("probe_tune_guard", "probe_tune_tol", "probe_tune_wmax",
                    "probe_tune_anomaly", "probe_tune_eta", "probe_tune_lf",
                    "cond_probe_tune_guard", "cond_probe_tune_tol",
                    "cond_probe_tune_wmax", "cond_probe_tune_anomaly",
                    "cond_probe_tune_eta", "cond_probe_tune_lf"):
            cfg[_tk] = False
        # Recorded so the corpus can tell a standalone run from a probed one by a
        # POSITIVE field.  Today the only way to identify one is the ABSENCE of
        # probe_key/_ms_info, which is exactly the kind of implicit signal this
        # project has been burned by.
        cfg["_standalone_note"] = (
            "Autotuner not connected: no probe evidence. Disabled lf_enable, "
            "dwell_mode, eta_escape_*, eta_fade_sigma, probe_tune_*. The outlier "
            "guard and anomaly detector stay ON but run on ENGINE_DEFAULTS "
            "(untuned) thresholds.")
        if log_errors:
            logging.info("[PD-AOS] STANDALONE MODE: no Autotuner connected, so no "
                         "probe evidence exists. Disabled the layers that REQUIRE "
                         "it: local field (lf_enable), dwell (dwell_mode), the "
                         "eta escape/fade auto-thresholds, and probe_tune_*. The "
                         "outlier guard and anomaly detector remain ON, but their "
                         "thresholds are ENGINE_DEFAULTS, NOT probe-derived.")
    lf_prof_m = None
    lf_prof_c = None
    lf_prof_w = float(cfg.get("cond_probe_weight", 0.5))
    _gap_ctx = None
    _t_gap = _t0()
    try:
        _gap_ctx = _build_gap_ctx(model, extra_args)
    except Exception:
        _gap_ctx = None
    _phase_add("gap_ctx_ms", _t_gap)
    _t_eff = _t0()
    try:
        cfg["_guider_live_cfg"] = _probe_effective_cfg(model, extra_args,
                                                       _gap_ctx)
    except Exception:
        cfg["_guider_live_cfg"] = None
    _phase_add("guider_cfg_ms", _t_eff)

    def _dist_resolved(prof, key):
        if not isinstance(prof, dict):
            return prof
        want = _resolve_distilled_override(
            cfg.get(key, "auto"), cfg.get("_guider_live_cfg"),
            prof.get("is_distilled_eff", False))
        if want == bool(prof.get("is_distilled_eff", False)):
            return prof
        prof = dict(prof)
        prof["is_distilled_eff"] = want
        try:
            prof["trajectory_state"] = _compute_trajectory_state(prof)
        except Exception:
            pass
        return prof

    if bool(cfg.get("probe_enabled", False)):
        _t_mk = _t0()
        mk = _model_cache_key(model)
        _t_sch = _t0()
        pkey = mk + "|s:" + _schedule_sig(x, sigmas, cfg)
        _phase_add("model_key_ms", _t_mk)
        _phase_set("schedule_sig_ms", (time.perf_counter() - _t_sch) * 1e3
                   if _t_sch is not None else 0.0)
        profile = _PROBE_PROFILES.get(pkey)
        if profile is not None and int(profile.get("version", 1) or 1) < 7:
            profile = None
        # THE SLOT'S NAME SAYS `<model>|s:<schedule sig>`; THE PROFILE INSIDE MUST BE A MODEL
        # MEASUREMENT (added 2026-10-08).  An exact key cannot vouch for the PRODUCER: the
        # adoption path below writes a profile INTO this slot (`_PROBE_PROFILES[pkey] =
        # profile` on the promotion), and until this gate existed that promotion carried
        # per-prompt COND measurements here -- measured, `anima-base-v1.0` log_00019/00020,
        # where the model slot held the run's cond profile byte for byte at 8 steps / 16 px
        # instead of the engine's own 16 steps / 48 px.  Checking the artefact rather than
        # the order of operations is deliberate: an invariant that holds only because of
        # statement order is the defect class rule 28 names.
        if profile is not None and str(profile.get("probe_src") or "") != "model":
            profile = None
        # An EXACT-signature hit is THIS run's own measurement under THIS config,
        # so it is trusted as-is and suppresses our probe.
        # Regression fixed here: probe_from_node used to be assigned only on the
        # FALLBACK branch below, so an exact-key hit left it False and the engine
        # re-probed on every run.  That is why runs 69-73 of the 68-73 A/B set
        # came out healthy while run 68, which took the fallback, adopted a stale
        # DEGENERATE profile and ran the whole generation on zero-dynamics
        # evidence.
        probe_from_node = profile is not None
        if probe_from_node:
            _phase_set("model_probe_state", "engine_cache_exact")
        # Fall back to the most recent profile for this model when the
        # schedule-qualified key misses -- e.g. a profile pre-computed
        # node-side by the Probe Options node BEFORE the KSampler started
        # (measured on a provisional schedule; its curves are resampled onto
        # the real sigmas by the consumers, exactly like the node-side cond
        # probe).  Without this fallback the engine would re-probe here at
        # sampling time even though the values are already cached.
        if profile is None and not bool(cfg.get("probe_force", False)):
            # BEFORE THE `_latest_profile` FALLBACK, because that one is what goes wrong: it
            # returns `_PROFILE_BY_MODEL[mk]` on sight, and the node's COND probe -- which runs
            # LAST and re-runs every generation -- is what wrote that entry.  This asks for the
            # node's MODEL measurement by name instead of taking whatever happens to be newest.
            _t_nm = _t0()
            _nm = _node_model_profile(mk, cfg)
            _phase_add("node_model_slot_ms", _t_nm)
            if _nm is not None and _profile_serves_model_probe(_nm, cfg):
                profile = _nm
                probe_from_node = True
                _phase_set("model_probe_state", "node_model_slot")
        if profile is None and not bool(cfg.get("probe_force", False)):
            _t_lp = _t0()
            profile = _latest_profile(model)
            _phase_add("latest_profile_ms", _t_lp)
            if profile is not None and int(profile.get("version", 1) or 1) < 7:
                profile = None
            # FALLBACK provenance is unknown, so it only counts as "our probe
            # already ran" when it can serve the curvature consumers.  Adoption
            # still happens either way -- the whitepoint / adaptive-noise
            # consumers want it -- but a too-short OR DEGENERATE profile must
            # never suppress our own measurement.
            probe_from_node = (profile is not None
                               and _profile_serves_model_probe(profile, cfg))
            if profile is not None:
                _phase_set("model_probe_state",
                           "node_adopted" if probe_from_node
                           else "node_adopted_NOT_SERVING")
            if probe_from_node and log_errors:
                logging.info("[PD-AOS-probe] using pre-computed profile "
                             "(node-side probe ran before the KSampler; "
                             "skip sampling-time probe)")
        if (profile is None or not probe_from_node
                or bool(cfg.get("probe_force", False))):
            if log_errors:
                logging.info("[PD-AOS-probe] running pre-profile pass "
                             "(%d steps @ %dx%d latent)",
                             int(cfg.get("probe_steps", 10)),
                             int(cfg.get("probe_resolution", 48)),
                             int(cfg.get("probe_resolution", 48)))
            _progress(callback, "probe", "start", 0,
                      int(cfg.get("probe_steps", 10)) + 4,
                      "PD-AOS model probe")
            _probe_model_cache_guard(model, True)
            _t_mp = _t0()
            try:
                _adopted_prof = profile
                _fresh = _run_probe(model, x, sigmas, extra_args, cfg,
                                    callback=callback)
                # Never lose evidence a previous build kept: if our own probe
                # produced nothing, fall back to the adopted profile.
                profile = _fresh if _fresh is not None else _adopted_prof
            finally:
                _phase_add("model_probe_ms", _t_mp)
                _phase_set("model_probe_state",
                           "ran" if profile is not None else "ran_EMPTY")
                _probe_model_cache_guard(model, False)
            _progress(callback, "probe", "end",
                      int(cfg.get("probe_steps", 10)) + 4,
                      int(cfg.get("probe_steps", 10)) + 4,
                      "PD-AOS model probe done")
            if profile is not None:
                _carry_run_feedback(profile, _PROBE_PROFILES.get(pkey))
                _PROBE_PROFILES[pkey] = profile
                _PROFILE_BY_MODEL[mk] = profile
                _prune_probe_cache(mk, keep=8)
                if log_errors:
                    logging.info("[PD-AOS-probe] profile cached: distilled=%s "
                                 "safe_fresh_frac=%.3f med_jump=%.4f med_curv=%.4f",
                                 profile.get("is_distilled_eff"),
                                 float(profile.get("safe_fresh_frac") or -1.0),
                                 float(profile.get("med_jump") or -1.0),
                                 float(profile.get("med_curv") or -1.0))
        if profile is not None:
            if probe_from_node:
                # Promote the pre-computed (node-side) profile into the
                # current schedule-qualified cache slot so later runs and the
                # end-of-run direction feedback find it directly.
                try:
                    _PROBE_PROFILES[pkey] = profile
                    _prune_probe_cache(mk, keep=8)
                except Exception:
                    pass
            profile = _dist_resolved(profile, "probe_is_distilled")
            tune = {"guard": bool(cfg.get("probe_tune_guard", True)),
                    "tol": bool(cfg.get("probe_tune_tol", True)),
                    "wmax": bool(cfg.get("probe_tune_wmax", True)),
                    "anomaly": bool(cfg.get("probe_tune_anomaly", True)),
                    "eta": bool(cfg.get("probe_tune_eta", False))}
            _apply_probe_profile(cfg, profile, tune, auto_tunable)
            cfg["_probe_key"] = pkey
            lf_prof_m = profile
            if profile.get("is_distilled_eff"):
                cfg["_distilled_eff"] = True
    if bool(cfg.get("cond_probe_enabled", False)):
        mk = _model_cache_key(model)
        ckey = _cond_cache_key(cfg, extra_args, mk)
        if ckey is None and log_errors:
            logging.info("[PD-AOS-cond-probe] no conditioning available to "
                         "fingerprint; profile cannot be cached")
        skey = None
        if ckey is not None:
            try:
                skey = (ckey[0], ckey[1], _schedule_sig(x, sigmas, cfg))
            except Exception:
                skey = None
        cond_profile = None
        from_cache = False
        if skey is not None and not bool(cfg.get("cond_probe_force", False)):
            cond_profile = _COND_PROBE_PROFILES_SCHED.get(skey)
            from_cache = cond_profile is not None
            if (cond_profile is not None
                    and int(cond_profile.get("version", 1) or 1) < 7):
                cond_profile = None
                from_cache = False
        if (cond_profile is None and ckey is not None
                and not bool(cfg.get("cond_probe_force", False))):
            # Fall back to the latest per-prompt profile for this model+prompt
            # (pre-computed node-side by the autotuner on the incoming
            # schedule); its curves are resampled onto the real sigmas by the
            # consumers, exactly like the model probe's node-side fallback.
            cond_profile = _COND_PROBE_PROFILES.get(ckey)
            from_cache = cond_profile is not None
            if (cond_profile is not None
                    and int(cond_profile.get("version", 1) or 1) < 7):
                cond_profile = None
                from_cache = False
        if cond_profile is not None:
            # A PROFILE THE NODE MEASURED FOR THIS QUEUE IS NOT A CACHE HIT.  The node
            # writes its result under this exact key, so the lookup above finds it; the
            # report used to call that `cached` + "model+prompt+schedule unchanged" --
            # measured on moss log_00071, a NEW prompt (`941fa9ed3f77e510`) with
            # `node_cond_probe_state: ran`.  `_cond_profile_origin` reads the producer's
            # own state instead of inferring it from the store.
            _origin = _cond_profile_origin(
                from_cache, _PHASE_MS.get("node_cond_probe_state"))
            _phase_set("cond_probe_state", _origin)
            if log_errors:
                if _origin == "node_probe":
                    logging.info("[PD-AOS-cond-probe] adopting the node-side "
                                 "per-prompt profile measured for THIS queue "
                                 "(skip the engine-side probe)")
                elif _origin == "cached":
                    logging.info("[PD-AOS-cond-probe] using cached per-prompt "
                                 "profile (model+prompt+schedule unchanged; "
                                 "skip probe)")
                else:
                    logging.info("[PD-AOS-cond-probe] adopting a node-side "
                                 "per-prompt profile (skip the engine-side probe)")
        else:
            if log_errors:
                logging.info("[PD-AOS-cond-probe] running per-prompt probe "
                             "(%d steps @ %dx%d latent, weight=%.2f)",
                             int(cfg.get("cond_probe_steps", 4)),
                             int(cfg.get("cond_probe_resolution", 16)),
                             int(cfg.get("cond_probe_resolution", 16)),
                             float(cfg.get("cond_probe_weight", 0.5)))
            _cond_K = int(cfg.get("cond_probe_steps", 4))
            _progress(callback, "cond_probe", "start", 0, _cond_K + 4,
                      "PD-AOS cond probe")
            _probe_model_cache_guard(model, True)
            _t_cp = _t0()
            try:
                cond_profile = _run_cond_probe(
                    model, x, sigmas, extra_args, cfg,
                    callback=callback,
                    gap_ctx=_gap_ctx)
            finally:
                _phase_add("cond_probe_ms", _t_cp)
                _phase_set("cond_probe_state",
                           "ran" if cond_profile is not None else "ran_EMPTY")
                _probe_model_cache_guard(model, False)
            _progress(callback, "cond_probe", "end", _cond_K + 4, _cond_K + 4,
                      "PD-AOS cond probe done")
            if cond_profile is not None and cond_profile.get("gap") is None \
                    and ckey is not None:
                _prev_gap = (_COND_PROBE_PROFILES.get(ckey) or {}).get("gap")
                if isinstance(_prev_gap, dict) and _prev_gap:
                    cond_profile["gap"] = _prev_gap
        if cond_profile is not None:
            cond_profile = _dist_resolved(cond_profile,
                                          "cond_probe_is_distilled")
            if ckey is not None:
                if not from_cache:
                    if skey is not None:
                        _carry_run_feedback(
                            cond_profile,
                            _COND_PROBE_PROFILES_SCHED.get(skey))
                    _COND_PROBE_PROFILES[ckey] = cond_profile
                    if skey is not None:
                        _COND_PROBE_PROFILES_SCHED[skey] = cond_profile
                # THE SAME SPELLING THE LOG STORE USES, so the stamped key can actually find
                # the profile it names.  See `_cond_key_str` for the measured consequence of
                # the two spellings having diverged (0 of 443 logs matched).
                cfg["_cond_probe_key"] = _cond_key_str(ckey)
            cond_tune = {"guard": bool(cfg.get("cond_probe_tune_guard", True)),
                         "tol": bool(cfg.get("cond_probe_tune_tol", True)),
                         "wmax": bool(cfg.get("cond_probe_tune_wmax", True)),
                         "anomaly": bool(cfg.get("cond_probe_tune_anomaly", True)),
                         "eta": bool(cfg.get("cond_probe_tune_eta", True))}
            cond_weight = float(cfg.get("cond_probe_weight", 0.5))
            _apply_cond_probe_profile(cfg, cond_profile, cond_tune,
                                      cond_weight, auto_tunable)
            lf_prof_c = cond_profile
            if cond_profile.get("is_distilled_eff"):
                cfg["_distilled_eff"] = True
            if log_errors:
                gap = cond_profile.get("gap")
                ts = cond_profile.get("trajectory_state", {})
                logging.info("[PD-AOS-cond-probe] profile ready: "
                             "distilled=%s med_jump=%.4f med_curv=%.4f "
                             "med_err=%.4f gap=%s "
                             "oversteer=%.3f oversmooth=%.3f "
                             "weight=%.2f",
                             cond_profile.get("is_distilled_eff"),
                             cond_profile.get("med_jump", -1.0),
                             cond_profile.get("med_curv", -1.0),
                             cond_profile.get("med_local_err", -1.0),
                             "n/a" if gap is None else
                             "s=%.2f/o=%.2f" % (gap.get("strength", 0),
                                                gap.get("overshoot", 0)),
                             ts.get("oversteer", 0.0),
                             ts.get("oversmooth", 0.0),
                             cond_weight)
    # Derive the escape thresholds from the probe's measured per-pixel jump /
    # detail levels -- the SAME quantities the runtime detector uses -- unless
    # the user set them explicitly (a negative value means "auto").
    #
    # `escape_gain` scales the DERIVED values only.  An explicit positive
    # threshold is a measurement the user made about their own model and is left
    # exactly as given; the gain exists so sensitivity can be varied WITHOUT
    # hand-setting an absolute, which would be a value from some other model's
    # population wearing a measurement's clothes (rule 19).
    #
    # NOT part of the probe cache key, and it must not be: it changes what is
    # DERIVED from the probe's measurements, never what the probe MEASURES, and
    # this whole block re-runs from cfg on every engine entry.  Same reasoning as
    # `use_measured_tolerance` above, and the measured precedent is `wmax_budget`,
    # which `_node_probe_param_sig`'s own docstring keeps out of the key for this
    # reason: "adding it would fragment this cache (a fresh probe, ~10 model
    # calls) for nothing".  Kept out of `_probe_key_env_part` too; `rig_eta_node.py`
    # asserts both, with positive controls.
    try:
        def _prof_scalar(profs, key):
            for _p in profs:
                if isinstance(_p, dict) and _p.get(key) is not None:
                    return float(_p[key])
            return None
        _egain = max(0.0, float(cfg.get("escape_gain", 1.0) or 0.0))
        if float(cfg.get("eta_escape_jump", -1.0)) < 0:
            _mj = _prof_scalar((lf_prof_c, lf_prof_m), "med_jump_rel")
            if _mj is not None and _mj > 0:
                # stalled = the per-pixel jump collapsed to a quarter of its
                # measured normal level, scaled by the sensitivity gain.
                cfg["eta_escape_jump"] = round(max(0.25 * _mj * _egain, 1e-4), 6)
        if float(cfg.get("eta_escape_detail", -1.0)) < 0:
            _md = _prof_scalar((lf_prof_c, lf_prof_m), "med_detail_abs")
            if _md is not None and _md > 0:
                # flat = the high-frequency energy dropped to half of the
                # measured normal level, scaled by the sensitivity gain.
                cfg["eta_escape_detail"] = round(max(0.5 * _md * _egain, 0.02), 6)
        # THE MOMENTUM ARM'S GROWTH BUDGET, DERIVED PER MODEL/PROMPT.
        #
        # The budget is what decides whether the momentum arm may fire: the
        # movement has collapsed AND the image is not still gaining detail faster
        # than this.  Shipped as a constant (0.15), it is measured to be the
        # BINDING gate on turbo checkpoints and not on others -- on the build
        # under study it blocked 39 steps on mklanKrea2 and 12 on museByStableYogi
        # while blocking 2 in 490 on anima-base-v1.0 -- because the same model
        # sampled in 16 steps builds more detail per step than in 50.  A constant
        # cannot mean the same thing at both densities.
        #
        # Derived instead from what the probe measured, in two parts:
        #
        #   `commit`  -- how consistently the model's trajectory moves in ONE
        #                direction, over the phase where the probe saw it start
        #                reversing.  A model that commits never has the problem
        #                the escape exists to fix, so it gets no extra budget;
        #                one that reverses throughout gets the full opening.
        #                This is a MEASUREMENT of the model and the prompt in
        #                front of the engine, which is what lets it generalise
        #                where a per-checkpoint table would not.
        #   `open`    -- the extra budget granted at full reversal.  The MEASURED
        #                basis: the detail growth at the moment of collapse is
        #                -0.12 on the 50-step model and +0.17..+0.35 on the turbo
        #                checkpoints, so the shipped 0.15 sits inside that spread
        #                and 0.60 clears the top of it with the narrowest margin
        #                that still arms.  A SAFE FALLBACK rather than a
        #                derivation, and the value in force is stamped per run as
        #                `eta_growth_budget`.
        #
        # The two parameterisations tried -- `open * (1 - commit)` with a ceiling,
        # and `shipped + open * (1 - commit)` over the shipped floor -- arm 116 and
        # 118 steps of the same 681, so the reading is not sensitive to which is
        # used; the additive form is kept because it can only ever LOOSEN the
        # shipped budget, never silence an arm that already fired.
        if bool(cfg.get("escape_growth_probe", False)):
            try:
                _cm, _cm_raw, _cm_n = _probe_commitment(
                    (lf_prof_c, lf_prof_m), cfg)
                if _cm is not None:
                    _open = max(0.0, float(cfg.get("escape_growth_probe_open",
                                                   0.60) or 0.0))
                    _base = float(cfg.get("escape_momentum_growth", 0.15) or 0.15)
                    cfg["escape_momentum_growth"] = round(
                        _base + _open * (1.0 - _cm), 6)
                    cfg["_escape_commit"] = round(_cm_raw, 4)
                    cfg["_escape_commit_n"] = _cm_n
            except Exception:
                pass
        if float(cfg.get("eta_fade_sigma", -1.0)) < 0:
            # Anchor the late noise fade at the oscillation onset (where the
            # probe measured the x0 trajectory first reversing direction);
            # fall back to the jump-collapse sigma when the trajectory stays
            # monotone (no reversal -> fade where the model converged).
            _cs = (_prof_scalar((lf_prof_c, lf_prof_m), "osc_sigma")
                   or _prof_scalar((lf_prof_c, lf_prof_m), "conv_sigma"))
            if _cs is not None and _cs > 0:
                cfg["eta_fade_sigma"] = round(float(_cs), 6)
    except Exception:
        pass
    _fragile = []
    if bool(cfg.get("probe_endgame", True)) and (lf_prof_m is not None
                                                 or lf_prof_c is not None):
        try:
            _fragile = _detect_fragile_ranges(lf_prof_m, lf_prof_c)
        except Exception:
            _fragile = []
    _fp_on = bool(cfg.get("cond_probe_focused", True)) and bool(cfg.get("lf_enable", True))
    if (_fp_on
            and lf_prof_m is not None and lf_prof_c is not None
            and bool(cfg.get("cond_probe_enabled", False))
            and bool(cfg.get("probe_endgame", True))):
        try:
            if _fragile:
                _focused_key = None
                if skey is not None:
                    _focused_key = (skey[0], skey[1], skey[2], "focused")
                _focused_prof = None
                if _focused_key is not None and not bool(
                        cfg.get("cond_probe_force", False)):
                    _focused_prof = _COND_PROBE_PROFILES_SCHED.get(_focused_key)
                if _focused_prof is None:
                    _focused_grid = _build_focused_grid(
                        sigmas, _fragile,
                        K_base=max(3, int(cfg.get("cond_probe_steps", 4))),
                        K_max=max(6, int(cfg.get("cond_probe_steps", 4)) + 4),
                        distilled_mode=_probe_distilled_mode(cfg), cfg=cfg)
                    if _focused_grid and len(_focused_grid) >= 4:
                        if log_errors:
                            logging.info("[PD-AOS-focused] running focused "
                                         "cond probe (%d points, %d fragile "
                                         "ranges: %s)",
                                         len(_focused_grid), len(_fragile),
                                         [(round(sl, 4), round(sh, 4), round(f, 2))
                                          for sl, sh, f in _fragile])
                        _fp_total = len(_focused_grid)
                        _progress(callback, "cond_probe", "start", 0,
                                  _fp_total,
                                  "PD-AOS focused cond probe")
                        _probe_model_cache_guard(model, True)
                        _t_fp = _t0()
                        try:
                            _focused_prof = _run_focused_cond_probe(
                                model, x, sigmas, extra_args, cfg,
                                callback=callback,
                                gap_ctx=_gap_ctx,
                                fragile_ranges=_fragile,
                                focused_grid=_focused_grid,
                                base_profile=lf_prof_c)
                        finally:
                            # THE FOCUSED PASS HAD NO CLOCK.  It is a SECOND full
                            # per-prompt trajectory, triggered only when the base
                            # profile reports fragile ranges -- i.e. exactly the
                            # state-dependent extra work that reads as "the small
                            # probe got slow at random".  Timed so the log says
                            # whether it ran and what it cost.
                            _phase_add("focused_probe_ms", _t_fp)
                            _probe_model_cache_guard(model, False)
                        _progress(callback, "cond_probe", "end",
                                  _fp_total, _fp_total,
                                  "PD-AOS focused cond probe done")
                        if _focused_prof is not None and _focused_key is not None:
                            _carry_run_feedback(
                                _focused_prof,
                                _COND_PROBE_PROFILES_SCHED.get(_focused_key))
                            _COND_PROBE_PROFILES_SCHED[_focused_key] = _focused_prof
                if _focused_prof is not None:
                    _focused_prof = _dist_resolved(
                        _focused_prof, "cond_probe_is_distilled")
                    _focused_prof = _inherit_base_guard_stats(
                        _focused_prof, lf_prof_c)
                    lf_prof_c = _focused_prof
                    if log_errors:
                        ts = _focused_prof.get("trajectory_state", {})
                        logging.info("[PD-AOS-focused] profile ready: "
                                     "oversteer=%.3f oversmooth=%.3f "
                                     "med_err=%.4f n_steps=%d",
                                     ts.get("oversteer", 0.0),
                                     ts.get("oversmooth", 0.0),
                                     _focused_prof.get("med_local_err", -1.0),
                                     _focused_prof.get("n_probe_steps", 0))
        except Exception as e:
            if log_errors:
                logging.warning("[PD-AOS-focused] probe skipped: %s", e)
    # ---- THE DERIVED NOISE SCHEDULE, BUILT ONCE PER ENTRY -----------------
    # From the SAME profiles the escape thresholds above came from, so the two
    # arms cannot disagree about which measurement they are reading.
    _eta_probe_fn = None
    if bool(cfg.get("eta_from_probe", False)):
        try:
            # WHICH BUDGET: the widget's value, or the one the probe measured?
            # `_probe_budget_from_profile` returns None when nothing carries a
            # usable value, and the widget's number is kept -- so this toggle can
            # never leave the schedule without a budget.
            _budget = float(cfg.get("eta_probe_budget", 0.10) or 0.10)
            cfg["_eta_budget_src"] = "widget"
            if bool(cfg.get("eta_budget_from_probe", False)):
                _pb = _probe_budget_from_profile((lf_prof_c, lf_prof_m))
                if _pb is not None:
                    _budget = _pb
                    cfg["_eta_budget_src"] = "probe"
                    cfg["_eta_budget_measured"] = round(_pb, 6)
            _eta_probe_fn = _eta_from_probe((lf_prof_c, lf_prof_m), _budget)
            if _eta_probe_fn is not None:
                cfg["_eta_probe_gains"] = list(_eta_probe_fn.gains)
                cfg["_eta_probe_sigmas"] = list(_eta_probe_fn.measured_sigmas)
                cfg["_eta_probe_budget"] = _eta_probe_fn.budget
                if log_errors:
                    logging.info("[PD-AOS-eta] probe-derived schedule armed: "
                                 "%d sigma sample(s) %s, gains %s, budget %.3f",
                                 len(_eta_probe_fn.measured_sigmas),
                                 [round(v, 3) for v in _eta_probe_fn.measured_sigmas],
                                 [round(v, 3) for v in _eta_probe_fn.gains],
                                 _eta_probe_fn.budget)
            elif log_errors:
                logging.info("[PD-AOS-eta] eta_from_probe is ON but the probe "
                             "supplied no noise-survival curve; the shipped "
                             "escape/eta path is used unchanged.")
        except Exception as _e:
            _eta_probe_fn = None
            if log_errors:
                logging.warning("[PD-AOS-eta] derived schedule unavailable: %s", _e)
    eta_base = 0.0 if cfg["eta"] is None else float(cfg["eta"])
    space_mode = cfg["extrap_space"]
    adaptive_on = bool(cfg.get("order_pixel_adaptive", True))
    ladder_auto = adaptive_on and bool(cfg.get("order_ladder_auto", True))
    if adaptive_on:
        _om = int(cfg.get("order_max", 0) or 0)
        order_cap = max(2, min(16, _om if _om > 0 else 10))
    else:
        order_cap = max(1, min(4, int(cfg["order"])))
    order = order_cap
    hist_cap = max(8, order_cap) if adaptive_on else 8
    corrector = "aflops"
    # ---- CORRECTOR A/B ------------------------------------------------------
    # True (default)  = the current engine's corrector.
    # False           = the ARCHIVE-known-good corrector behaviour, from
    #                   `_antique/comfyui-research-samplers/old/8/aflops - cfg5
    #                   works properly.py`, whose `_aflops_step` body is the same
    #                   formula (only the return arity differs).  The three
    #                   differences this switch reverts:
    #                     1. the integrator is fed `x0`      instead of `x0m`
    #                        (archive line 6861  vs  current line 7811)
    #                     2. the per-step run-driven lambda bound
    #                        (`_run_lam_bound`) is NOT applied
    #                     3. the corrected step is ALWAYS taken -- no blend
    #                        toward the midpoint and no Euler-safety cut to x1
    #                   With it False, `corr` can never read "raw", which is
    #                   itself the fingerprint that the OLD arm ran.
    #
    #                   NOT a probe-cache key.  `_aflops_step` has exactly ONE
    #                   call site (the step loop), and no probe function calls
    #                   it -- probes advance with `exact_step` only.  So the
    #                   probe MEASUREMENT is arm-independent and a shared cached
    #                   probe is correct, not a leak.  The arm is recorded in
    #                   `_LAST_CFG` and in every step's `corr`, so a run's
    #                   label can never contradict its evidence.
    corrector_ab = bool(cfg.get("corrector_ab", True))
    # ---- ORDER / EXTRAPOLATION A/B -----------------------------------------
    # True (default) = the shipped per-pixel adaptive order ladder.
    # False          = force the PURE ORDER-1 exponential integrator: no midpoint
    #                  extrapolation (x0m stays x0), no 2nd-order slope term, and
    #                  x_pred = x1.  That is the paper's A-Euler step with the
    #                  adaptive lambda and nothing on top.
    #
    # WHY THIS TOGGLE EXISTS (measured, not assumed):
    # `scratch_histcos_gate_rig.py` measured, on a denoiser with sigma-scaled
    # estimation error, 58 step instances where the order-2 step landed CLOSER to
    # the ideal trajectory and 26 where it landed FARTHER -- and NO single
    # variable separates the two groups (du: helps up to 0.551, hurts from 0.028;
    # wmax: 1.98 vs 0.00; hist_cos: 0.023 vs 0.907).  So a threshold on any one
    # of them cannot be derived, and the honest move is to A/B the machinery
    # itself on the real model: if forcing order 1 changes the tail artifacts,
    # the extrapolation is the culprit; if not, it is not.
    #
    # In the real logs the gate releases the ladder for the first time at step 10
    # (hist_cos flips positive) at the LARGEST du, and that first-ever order-2
    # step overshoots: err/tol 2.0x then 6.8x, cut to 'raw' on the ON arm.
    order_ab = bool(cfg.get("order_ab", True))
    # ---- EULER-SAFETY CUT ARM (T2) -----------------------------------------
    # True (default) = shipped: when the corrected step departs from the plain
    #                  exact step by more than `tol`, replace it with the plain
    #                  step.  False = keep the corrected step unconditionally.
    # Added ONLY so the tail question can be measured as a controlled arm
    # (scratch_tail_three_arm_rig.py); the default is byte-identical to the
    # shipped behaviour, which the rig asserts before it reports anything.
    euler_cut = bool(cfg.get("euler_cut", True))
    # ---- WHICH VARIABLE THE EULER-SAFETY CUT TESTS (T4) ---------------------
    # "corrected" (default) = shipped: compares the CORRECTED step to the plain
    #                         step, which the blend's own bound is a component of.
    # "midpoint"            = the derived form: compares the LADDER's midpoint step
    #                         to the plain step, i.e. tests only the term nothing
    #                         else bounds.  See the long derivation at the cut.
    cut_var = str(cfg.get("cut_var", "midpoint") or "midpoint")
    if cut_var not in ("corrected", "midpoint"):
        cut_var = "midpoint"
    # ---- DERIVED-CONSTANTS A/B ---------------------------------------------
    # False (default) = the SHIPPED hand-set values, so nothing changes unless
    #                   this is switched on.
    # True            = the DERIVED forms, each of which replaced a handwave with
    #                   either a closed form or a measured crossing:
    #
    #   1. the `1e-4` series switch -> `sqrt(12*eps)` of the working dtype.
    #      DERIVED in scratch_step_constants_rig.py: the truncated Taylor branch's
    #      dominant relative error is the phi2 term z^2/12, verified numerically
    #      (law/series ratio 0.997..1.001 over z = 1e-5..1e-2), so the switch
    #      belongs where that error meets the precision: 1.196e-3 for float32,
    #      5.162e-8 for float64.  The shipped 1e-4 is 12x conservative for f32 and
    #      1937x too large for f64.  (Effect is tiny; included for correctness.)
    #
    #   2. the `_AFLOPS_X0M_ENV = 1.3` envelope -> `sum|w_i|`, the extrapolation's
    #      OWN exact amplification bound.  DERIVED in scratch_x0m_env_rig.py: x0m
    #      is a Lagrange combination with sum(w) = 1 (verified, max|sum(w)-1| =
    #      3.6e-15), so ||x0m|| <= (sum|w_i|)*max||x0_i||.  Over 2094 real
    #      (order, geometry) situations sum|w| has median 4.459 and exceeds 1.3 in
    #      100% of them, by 1.42x at order 2 rising to 17.6x at order 10.  So the
    #      fixed 1.3 is near-correct only at order 2 and drifts with the order for
    #      reasons unrelated to the data.  (Fires on 7.4% of extrapolations.)
    #
    #   3. `rtol` -> 1.536e-2, the crossing MEASURED in scratch_rtol_derive_rig.py:
    #      the value of |step2 - x1| at which the 2nd-order correction stops
    #      landing closer to the ideal than the plain step.  Below it keep the
    #      correction, above it cut.  The shipped standalone default 0.01 sits
    #      BELOW that crossing, so it cuts early across the band between them.
    #      CORRECTED 2026-10-07: this line used to read "Only applied when the
    #      probe has NOT supplied a tolerance", and that is the OPPOSITE of what
    #      the code does -- measured false on 298 of the 439 runs of the
    #      2026-10-06 corpus.  It is a MINIMUM on whatever tolerance is in force,
    #      INCLUDING a probe-measured one: on that corpus it raised the step
    #      tolerance on 289 of the 302 runs the probe HAD supplied a value to.
    #      The A/B that lets the measured value govern alone is
    #      `use_measured_tolerance` (see ENGINE_DEFAULTS).
    #
    # MEASUREMENT PROTOCOL, and it is mandatory (NOTES.md): hold the PROMPT FIXED
    # across arms.  Prompt choice moves err/tol by a median 41% against a 6%
    # run-to-run floor, so a prompt-varying A/B cannot resolve this.
    derived_ab = bool(cfg.get("derived_ab", True))
    _DERIVED_RTOL = 1.536e-2
    err_mode = cfg["err_mode"]
    grad_strength = float(cfg.get("grad_strength", 5.0))
    # THE ORDER-SYSTEM'S EMA WEIGHT, READ ONCE PER RUN FROM THIS RUN'S OWN CFG (2026-10-04).
    # It was the module constant `_ORD_EMA = 0.5`, which is why the operator could not reach it.  It is
    # read HERE, in the same function that consumes it, rather than from a module global set by some
    # other entry point -- that indirection is what let the cond-path loosening report one value and use
    # another for four rounds (`scratch_cond_loosen_provenance_rig.py`).
    # The range is DEFINITIONAL, not chosen: it is a convex blend weight between the previous EMA and the
    # new sample, so 0.0 = ignore history and 1.0 = ignore the new sample.  THE SHIPPED VALUE IS 0.37, the
    # measured AR(1) optimum (`CONSTANTS_AUDIT` section 15: the corpus's own logged `ord_innov` series,
    # n=682, has lag-1 autocorrelation 0.630, and `alpha* = 1 - rho_1 = 0.370`).  It was 0.5 -- 0.13 above
    # the optimum -- until 2026-10-04, when Â§ 15's "a value change would need its own A/B" was satisfied:
    # the operator's controlled triple (0.5 / 0.37 / 0.5, one session, one prompt, one seed; the 144-key
    # resolved cfg differs in exactly ONE key) has a BIT-IDENTICAL replicate pair and a 3.33e-1 latent /
    # 99.1 %-of-pixels difference on the arm, and he judged 0.37 better.  Closed form and eye agree.
    try:
        _ord_ema = float(cfg.get("ord_ema", ENGINE_DEFAULTS.get("ord_ema", 0.37)))
    except (TypeError, ValueError):
        _ord_ema = 0.37
    _ord_ema = max(0.0, min(1.0, _ord_ema))
    _ab_hit("order.ema_in_force", round(_ord_ema, 4))
    noise_space = cfg["noise_space"]
    s_noise = float(cfg["s_noise"])
    guard = (_OutlierGuard(window=cfg["guard_window"], z=cfg["guard_sensitivity"],
                           floor=cfg["guard_floor"],
                           ratio=cfg.get("guard_ratio", 3.0),
                           hold=cfg["guard_hold"],
                           warmup=min(4, max(1, (len(sigmas) - 1) // 4)),
                           mad_floor=cfg.get("guard_mad_floor", 0.1))
             if cfg["outlier_guard"] else None)
    # A quantile must be in [0, 1]; the widget's declared range used to allow
    # up to 4.0, and `_jump_signal` hands this straight to `torch.quantile`,
    # which raises.  Clamp at the read site so an API graph cannot bypass it.
    gquant = _guard_quantile_value(cfg)
    wmax_base = float(cfg.get("wmax_base", 2.0))
    # Default matches ENGINE_DEFAULTS: the feedback is OFF unless asked for (see the `else` branch at
    # the use site for why the loop is wrong-signed). A cfg that omits the key gets the same answer as
    # one that carries the shipped default -- a mismatch here is how a switch comes to mean two things.
    wmax_adapt = bool(cfg.get("wmax_adapt", False))
    wmax_mid_floor = float(cfg.get("wmax_mid_floor", 1.55))
    wmax_adapt_k = float(cfg.get("wmax_adapt_k", 0.5))
    wmax_limit = wmax_base
    # ---- PER-STEP PROBE FUSION: BUILT ONCE, UPDATED EVERY STEP --------------------------
    # `None` whenever the arm is off or the evidence cannot support it, so every use site below
    # is a single no-op test and the OFF arm is byte-identical to the shipped behaviour.
    _fuse_on = bool(cfg.get("probe_fuse_per_step", False))
    # Whether the fused cap may RAISE the order, its own switch (see ENGINE_DEFAULTS).  Read once
    # per run, like every other switch here.
    _fuse_raise = bool(cfg.get("probe_fuse_raise_cap", False))
    _fuse_st = (_fuse_probe_state(lf_prof_m, lf_prof_c, sigmas, cfg)
                if _fuse_on else None)
    _fuse_caps = []
    # The step's cap trace, carried in LOCALS until the raw-measurement block builds `_raw` --
    # writing into `_raw` here raised UnboundLocalError, because `_raw` is created later in the
    # step than the cap decision is taken (the rig caught it; it would have crashed every ON run).
    _cap_pre_arm = None
    _cap_in_force = None
    if _fuse_st is not None:
        _ab_hit("fuse.armed", round(float(_fuse_st.get("w0", 0.5)), 4))
    elif _fuse_on:
        _ab_hit("fuse.standdown")
    # Read ONCE per run from `cfg`, exactly like `wmax_base` above -- deliberately NOT a module
    # global, so it cannot leak across entry points the way `_osm_ab_mode` did (rule 28).
    try:
        _eg_dir = float(cfg.get("wmax_eg_dir", 1.0))
    except Exception:
        _eg_dir = 1.0
    if _eg_dir != _eg_dir:      # NaN guard: a NaN would silently disable the branch
        _eg_dir = 1.0
    _eg_dir = max(-1.0, min(1.0, _eg_dir))
    # the forced per-pixel order floor (see ENGINE_DEFAULTS): read once per run, not per step
    # `or 1.0` WOULD SWALLOW THE ADAPTIVE MODE: `0 or 1.0` is 1.0, so `order_floor = 0` would read back
    # as 1 and the widget would be wired to nothing.  Explicit None check instead (rule 12's class: the value
    # that reaches the decision, not the one that was asked for).
    # THE FALLBACK IS THE SHIPPED DEFAULT, not a hard-coded 1.0: a caller that omits the key must get the same
    # behaviour as the UI default, or the two silently disagree (one default, one source).
    _of_raw = cfg.get("order_floor", ENGINE_DEFAULTS.get("order_floor", 0.0))
    _order_floor = float(1.0 if _of_raw is None else _of_raw)
    # `order_floor = 0` = AUTOMATIC.  `_prev_rtol` is seeded here and carried at the end of every step next
    # to `prev_err`, so the gate reads the LAST MEASURED ratio and never a fabricated one.
    _order_floor_auto = bool(_order_floor <= 0.0)
    _order_floor_demand = float(ENGINE_DEFAULTS.get("order_floor_adaptive_demand", 6.0))
    # THE GRADED CURVE'S KNOBS.  `graded=False` restores the previous flat demand exactly, which is what makes
    # the two comparable in one A/B without a code change.
    _of_graded = bool(cfg.get("order_floor_adaptive_graded",
                              ENGINE_DEFAULTS.get("order_floor_adaptive_graded", True)))
    _of_lo = float(cfg.get("order_floor_adaptive_lo",
                           ENGINE_DEFAULTS.get("order_floor_adaptive_lo", 4.0)))
    _of_hi = float(cfg.get("order_floor_adaptive_hi",
                           ENGINE_DEFAULTS.get("order_floor_adaptive_hi", 5.0)))
    _of_r_hi = float(cfg.get("order_floor_adaptive_r_hi",
                             ENGINE_DEFAULTS.get("order_floor_adaptive_r_hi", 0.3031)))
    _order_floor_rtol = float(cfg.get("order_floor_adaptive_rtol",
                                      ENGINE_DEFAULTS.get("order_floor_adaptive_rtol", 0.05)))
    # `order_demand_q > 0` selects the BOUND-DERIVED request for the automatic mode (see ENGINE_DEFAULTS).
    _order_demand_q = float(cfg.get("order_demand_q",
                                    ENGINE_DEFAULTS.get("order_demand_q", 0.0)) or 0.0)
    _order_demand_q = max(0.0, min(1.0, _order_demand_q))
    # `order_demand_bound` makes the bound's OWN MAP the request on any step where it was computed --
    # no scalar and no quantile (see ENGINE_DEFAULTS).  A run-level arm, so it is read once here.
    _order_demand_bound = bool(cfg.get("order_demand_bound",
                                       ENGINE_DEFAULTS.get("order_demand_bound", True)))
    _prev_rtol = None
    # which criterion picks the per-pixel order, and the tolerance the remainder bound is
    # compared against -- the SAME tolerance the order gate uses, so the two agree
    _order_by_bound = bool(cfg.get("order_by_bound", False))
    # HOW MUCH OF THE RUNGE MAP REPLACES THE INNOVATION MAP (0 = none, 1 = all).  Read
    # once per run: it is a run-level dial, and `_order_bound_target` clamps it.
    _order_blend = float(cfg.get("order_bound_blend", 1.0) or 1.0)
    _order_tol = float(cfg.get("rtol", 0.01) or 0.01) * max(
        float(cfg.get("wmax_budget", 1.0) or 1.0), 1e-6)
    an_enable = bool(cfg["an_enable"])
    an_from = float(cfg["an_from"])
    an_win = int(cfg["an_win"])
    if an_win % 2 == 0:
        an_win += 1
    an_spatial = float(cfg["an_spatial"])
    an_novelty = float(cfg["an_novelty"])
    an_value = float(cfg["an_value"])
    an_phase_relax = float(cfg.get("an_phase_relax", 0.5))
    an_region_z = float(cfg.get("an_region_z", 12.0))
    an_sustain = float(cfg.get("an_sustain_mult", 3.0))
    lf_feed = bool(cfg.get("lf_guard_feed", True))
    n_cand = len(sigmas) - 1
    # ---- probe-driven lambda clamp ---------------------------------------
    # The clamp is a trust region on the lambda ESTIMATE (see the helper
    # above).  An explicit non-default TESTING choice wins, so the A/B toggle
    # still behaves exactly as labelled.
    # The clamp is always probe-driven now: the A/B mode widget was removed
    # once 'auto' proved to be the right setting, so nothing can write a fixed
    # aflops_lam_min/max any more.
    _lam_min_steps = None
    # largest x0 jump seen so far this run, for the causal run-driven bound
    _run_jump_max = 0.0
    if bool(cfg.get("lam_clamp_from_probe", True)):
        _lam_min_steps = _recommend_lam_clamp_from_profile(
            lf_prof_m, lf_prof_c, sigmas)
        if _lam_min_steps is not None:
            logging.info("[PD-AOS] probe-driven lam clamp: per-step lower "
                         "bound %.2f .. %.2f (anchor %.2f, headroom %.2f); "
                         "distilled/no-evidence models keep the fixed bound",
                         min(_lam_min_steps), max(_lam_min_steps),
                         _LAM_ANCHOR, _LAM_HEADROOM)
    lf_auto_g, lf_evid = _lf_auto_gains(lf_prof_m, lf_prof_c, lf_prof_w,
                                        n_cand, sigmas=sigmas, cfg=cfg)
    lf_gains = None
    _lf_clamp = 0.0
    _lf_drift = float(cfg.get("lf_drift_budget", 0.0) or 0.0)
    if _lf_drift <= 0.0:
        _lf_drift = float((lf_auto_g or {}).get("drift") or 0.10)
    if lf_auto_g is not None:
        _lf_clamp = min(max(float(lf_auto_g["clamp"]), 0.05), 1.0)
        # fragility gentle-step authority scales with the model's measured
        # (gap-corrected) local error: a more nonlinear velocity field needs
        # stronger gentle-steering in its fragile regions.
        _frag_gain = _LF_FRAG_GAIN
        try:
            _me = None
            for _p in (lf_prof_c, lf_prof_m):
                if isinstance(_p, dict):
                    _ts = _p.get("trajectory_state") or {}
                    _v = _ts.get("med_local_err_real")
                    if _v is None:
                        _v = _p.get("med_local_err")
                    if _v is not None:
                        _me = float(_v)
                        break
            if _me is not None and _me > 0:
                # Reachability for the Batch-2 sweep: `_LF_FRAG_GAIN` is the
                # FALLBACK and only acts when this branch is NOT taken;
                # `_FRAG_GAIN_MIN`/`_MAX` only act when the raw ratio crosses
                # them, so each gets its own counter instead of being assumed to
                # bind.  The arithmetic is unchanged: `_raw_g` is the same
                # expression that used to be inlined.
                _raw_g = _me / _FRAG_ERR_REF
                if _raw_g < _FRAG_GAIN_MIN:
                    _ab_hit("lf.frag_clamp_lo", round(_raw_g, 6))
                if _raw_g > _FRAG_GAIN_MAX:
                    _ab_hit("lf.frag_clamp_hi", round(_raw_g, 6))
                _frag_gain = min(max(_raw_g, _FRAG_GAIN_MIN), _FRAG_GAIN_MAX)
                _ab_hit("lf.frag_derived", round(_frag_gain, 6))
            else:
                _ab_hit("lf.frag_default", _LF_FRAG_GAIN)
        except Exception:
            pass
        lf_gains = {"k": float(lf_auto_g["k"]),
                    "v": float(lf_auto_g["v"]),
                    "d": float(lf_auto_g["d"]),
                    "s": float(lf_auto_g["s"]),
                    "amp": 0.15 * _lf_clamp,
                    "ramp": 0.35 * _lf_clamp,
                    "drift": _lf_drift,
                    "v_sched": lf_auto_g.get("v_sched"),
                    "hi_ref": lf_auto_g.get("hi_ref"),
                    "frag_gain": round(_frag_gain, 3),
                    "env_fade": bool(cfg.get("lf_env_fade", True))}
    lf_on = (bool(cfg.get("lf_enable", True))
             and lf_gains is not None and lf_gains["k"] > 1e-3)
    lf_vol = None
    lf_medref = None
    lf_damp = None
    lf_cum = None
    lf_cum_next = None
    lf_dir_acc = [0.0, 0]
    lf_closed_loop = lf_on and bool(cfg.get("lf_closed_loop", True))
    lf_trust = None
    lf_emap_prev = None
    lf_m_prev = None
    lf_amp_prev = 0.0
    lf_gate = None
    lf_rho_prev = None
    lf_applied_prev = False
    lf_suppress = None
    lf_spatial_health = lf_on and bool(cfg.get("lf_spatial_health", True))
    lf_prior = None
    if lf_on and bool(cfg.get("lf_semantic_prior", True)):
        try:
            _gap = (lf_prof_c or {}).get("gap") or {}
            _pr = _gap.get("prior")
            if isinstance(_pr, dict) and _pr.get("v"):
                _pg = torch.tensor([float(v) for v in _pr["v"]],
                                   device=x.device,
                                   dtype=torch.float32).reshape(
                                       1, 1, int(_pr["h"]), int(_pr["w"]))
                _pg = F.interpolate(_pg, size=x.shape[-2:], mode="bilinear",
                                    align_corners=False).clamp(0.0, 1.0)
                if x.dim() == 5:
                    _pg = _pg.unsqueeze(2)  # broadcast over frames
                lf_prior = _pg.to(x.dtype)
        except Exception:
            lf_prior = None
    lf_evid["prior"] = lf_prior is not None
    lf_frag = None
    if lf_on:
        # spatial fragility prior: per-prompt probe first, model probe fallback
        try:
            for _fp in (lf_prof_c, lf_prof_m):
                _fr = (_fp or {}).get("fragility") if isinstance(_fp, dict) else None
                if isinstance(_fr, dict) and _fr.get("v"):
                    _fg = torch.tensor([float(v) for v in _fr["v"]],
                                       device=x.device,
                                       dtype=torch.float32).reshape(
                                           1, 1, int(_fr["h"]), int(_fr["w"]))
                    _fg = F.interpolate(_fg, size=x.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False).clamp(0.0, 1.0)
                    if x.dim() == 5:
                        _fg = _fg.unsqueeze(2)
                    lf_frag = _fg.to(x.dtype)
                    break
        except Exception:
            lf_frag = None
    lf_evid["fragility"] = lf_frag is not None
    lf_gdir = None
    if lf_on:
        try:
            _gap = (lf_prof_c or {}).get("gap") or {}
            _pd = _gap.get("prior_dir")
            if isinstance(_pd, dict) and _pd.get("v"):
                _c = int(_pd.get("c", x.shape[1]))
                if _c == int(x.shape[1]):
                    _tg = torch.tensor([float(v) for v in _pd["v"]],
                                       device=x.device,
                                       dtype=torch.float32)
                    _tg = _tg.reshape(1, _c, int(_pd["h"]), int(_pd["w"]))
                    _tg = F.interpolate(_tg, size=x.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False)
                    if x.dim() == 5:
                        _tg = _tg.unsqueeze(2)  # broadcast over frames
                    lf_gdir = _tg.to(x.dtype)
        except Exception:
            lf_gdir = None
    lf_evid["prior_dir"] = lf_gdir is not None
    lf_rescue_on = lf_on and bool(cfg.get("lf_detail_rescue", True))
    _lf_rescue = None
    if lf_rescue_on:
        _lf_rescue = {"on": True, "block": 4,
                      "start": float(cfg.get("rescue_start", 0.55)),
                      "full": float(cfg.get("rescue_full", 0.72)),
                      "amp": (float(cfg.get("rescue_cap", 0.35)) * _lf_clamp),
                      "ramp": 0.5 * lf_gains["ramp"],
                      "max_cov": float(cfg.get("rescue_max_cov", 0.35)),
                      "floor": float(cfg.get("rescue_floor", 0.02)),
                      "fast": bool(n_cand <= 9
                                   or bool(cfg.get("_distilled_eff", False))),
                      "s": 0.0}
        # Endgame evidence -> rescue window placement: when the probe
        # measured WHERE structured-state misinterpretation lives (fragile
        # sigma band), ramp the rescue authority over that band instead of
        # the fixed 0.55/0.72 of the run.  Only when the user left
        # rescue_start/rescue_full untouched and either probe_tune_lf toggle
        # is on.
        if (_fragile and (bool(cfg.get("probe_tune_lf", True))
                          or bool(cfg.get("cond_probe_tune_lf", True)))
                and "rescue_start" in auto_tunable
                and "rescue_full" in auto_tunable):
            _rs, _rf = _rescue_window_from_fragile(sigmas, _fragile)
            if _rs is not None:
                if log_errors:
                    logging.info("[PD-AOS-lf] rescue window placed from "
                                 "endgame evidence: start=%.3f full=%.3f",
                                 _rs, _rf)
                _lf_rescue["start"] = _rs
                _lf_rescue["full"] = _rf
    lf_resc_state = None
    jhist = []
    prev_elev = None
    model_info = cfg.get("_model_info")
    dist_eff = bool(cfg.get("_distilled_eff", False))
    dist_hint = bool((model_info or {}).get("is_distilled")) or dist_eff
    sigmas = sigmas.detach().float().to(x.device)
    # ---- L1: per-pixel dwell fraction ------------------------------------
    # Pixels whose x0 prediction keeps churning (per-pixel jump envelope
    # above the image median) take a PARTIAL step toward the next sigma:
    # their effective noise level lags the grid, which is the per-pixel
    # version of the high global shift -- applied only where the artifact
    # forms instead of warping the whole schedule.  The probe decides HOW
    # MUCH (oversteer evidence) and WHEN (curvature active zone + endgame
    # fragile band); the run decides WHERE.  Off by default; auto arms only
    # on qualifying probe evidence (otherwise the exact classic behavior).
    _dwell_mode = str(cfg.get("dwell_mode", "off") or "off").strip().lower()
    if _dwell_mode not in ("auto", "manual"):
        _dwell_mode = "off"
    dwell_max = 0.0
    dwell_fade = 0.65
    if _dwell_mode == "manual":
        dwell_max = min(max(float(cfg.get("dwell_max", 0.20) or 0.20),
                            0.0), 0.45)
        dwell_fade = min(max(float(cfg.get("dwell_until", 0.65) or 0.65),
                             0.2), 0.85)
    elif _dwell_mode == "auto":
        try:
            _dw = _recommend_dwell_from_profile(
                lf_prof_m, lf_prof_c, sigmas, lf_evid, _fragile)
            if _dw is not None:
                dwell_max = float(_dw["dwell_max"])
                dwell_fade = float(_dw["fade_end"])
                if log_errors:
                    logging.info(
                        "[PD-AOS-dwell] adopted: max=%.3f ramp_end=%.2f "
                        "fade_end=%.2f (oversteer=%.3f, source=%s)",
                        dwell_max, float(_dw["ramp_end"]), dwell_fade,
                        float(_dw["ov"]), lf_evid.get("source", "?"))
            elif log_errors:
                logging.info("[PD-AOS-dwell] off: no qualifying probe "
                             "evidence (oversteer below gate or no usable "
                             "curvature curve)")
        except Exception as e:
            if log_errors:
                logging.warning("[PD-AOS-dwell] skipped: %s", e)
    dwell_on = dwell_max > 1e-3 and n_cand >= 6
    dwell_ramp = 0.30
    dwell_fade_start = 1.0
    if dwell_on:
        dwell_ramp = min(0.30, 0.5 * dwell_fade)
        dwell_fade_start = max(dwell_fade - 0.15, dwell_ramp + 0.02)
        cfg["_dwell"] = {"mode": _dwell_mode, "adopted": True,
                         "dwell_max": round(dwell_max, 4),
                         "ramp_end": round(dwell_ramp, 4),
                         "fade_end": round(dwell_fade, 4)}
        if log_errors and _dwell_mode == "manual":
            logging.info("[PD-AOS-dwell] manual: max=%.3f fade_end=%.2f",
                         dwell_max, dwell_fade)
    elif _dwell_mode != "off":
        cfg["_dwell"] = {"mode": _dwell_mode, "adopted": False}
    dw_ema = None
    dw_w = None
    sigma_data = 1.0
    if model_info is not None:
        if model_info.get("is_flow") is not None and noise_space == "auto":
            space_override = "flow" if model_info["is_flow"] else "ve_edm"
        else:
            space_override = None
        sigma_data = model_info.get("sigma_data", 1.0) or 1.0
        # `is_video` is a PROBE/SHAPE fact, not a noise-policy switch.  It used to
        # auto-enable `color_temporal` here, but `color_temporal_beta` has no node
        # emitter and stays 0.0, so the filter never ran; the auto-enable only made
        # the log claim a temporal policy that did not exist.  Removed 2026-10-17.
    else:
        space_override = None
    if space_override is not None:
        space = space_override
    else:
        space = _detect_space(sigmas, noise_space)
    # The sampler is needed whenever the effective eta can be > 0.  With the
    # v15 escape scheme eta can come from eta_min even when eta_base
    # (stochasticity) is 0, so gate on either.
    _eta_floor_cfg = float(cfg.get("eta_min", 0.0) or 0.0)
    noise_sampler = (_make_noise_sampler(x, extra_args, cfg)
                     if (eta_base > 0 or _eta_floor_cfg > 0) else None)
    _lf_seed_raw = extra_args.get("seed", None)
    if _lf_seed_raw is None:
        _lf_seed_raw = 0
    _lf_seed_bytes = hashlib.sha256(
        ("lf_noise:" + str(_lf_seed_raw)).encode()).digest()
    _lf_noise_seed = int.from_bytes(_lf_seed_bytes[:8], 'little')
    lf_noise_gen = torch.Generator(device=x.device)
    lf_noise_gen.manual_seed(_lf_noise_seed)
    is_adaptive_noise = (str(cfg.get("noise_color", "white")) == "auto" and
                         noise_sampler is not None)
    ones = x.new_ones([x.shape[0]])
    pos = sigmas[sigmas > 1e-8]
    s_max_n = float(sigmas[0])
    s_min_n = float(pos[-1]) if len(pos) > 0 else 1e-8
    calls = 0
    tol = max(float(cfg.get("rtol", 0.05)), 1e-3)
    # ---------------------------------------------------------------------------------------
    # THE TOLERANCE-FLOOR A/B.  `use_measured_tolerance` (ENGINE_DEFAULTS, OFF = shipped)
    # chooses between the two things that can govern `tol`: the tolerance measured for THIS
    # model by the probe, or that value with the 1.536e-2 minimum applied on top.
    #
    # OFF is what the engine does today, and with it the block below is what HEAD ran -- the
    # guard reduces to HEAD's, and the statements inside it are byte-identical to HEAD's, so
    # the default changes nothing.  ON skips the minimum entirely, so the measured value
    # governs however small it is.
    #
    # WHY THE ARM IS READ HERE rather than from a module global set by another entry point:
    # this is the site that consumes it, in the function that derives it from its own cfg, so
    # there is nothing to inherit and nothing that can arrive stale -- the class of defect that
    # `cond_rtol_loosen` shipped twice (see `_apply_ab_modes`).  It changes what is DERIVED from
    # cfg, never what the probe MEASURES, so it needs no cache key: no profile depends on it,
    # and this read happens again on every entry into this function.
    _use_measured_tol = bool(cfg.get("use_measured_tolerance", False))
    # The probe's own value BEFORE the minimum, kept so the log can name it (see the `_LAST_CFG`
    # stamp below): a corpus reader must never have to infer which arm ran from a difference
    # between two numbers.
    _rtol_measured = float(tol)
    # REACHABILITY (rule 9): the site counts its own arrival in EVERY arm, and each arm records
    # the value it left in force, so "this switch is connected" is a counter in
    # `aflops_cfg.ab_reach` rather than an assumption.  A toggle wired to nothing looks exactly
    # like one whose effect is below the noise floor, and the corpus cannot tell them apart.
    _ab_hit("rtol.site")
    if _use_measured_tol:
        _ab_hit("rtol.arm_probe", round(_rtol_measured, 8))
    # ---------------------------------------------------------------------------------------
    # A/B #3: a FLOOR at the measured crossing.
    #
    # MY FIRST VERSION OF THIS WAS A NO-OP ON EVERY PROBED RUN, and the corpus
    # proved it: it substituted the crossing only when `tol` still equalled
    # ENGINE_DEFAULTS["rtol"] = 0.01, but with the Autotuner connected `rtol` is
    # probe-derived and the observed values were 0.014 / 0.0148 / 0.0162 / 0.0169
    # / 0.0087 -- never 0.01.  Five derived_ab A/B pairs on five models came back
    # with ZERO differing per-step fields, which is how it was caught.
    #
    # The measurement says the shipped default is 1.536x too tight, so the honest
    # generalisation is a FLOOR at the crossing rather than a swap of one
    # hard-coded value for another: any tolerance below the crossing is raised to
    # it, and anything already above is left alone.
    #
    # "LEFT ALONE" MEANS THE VALUE, NOT THE RUN: this is a MINIMUM on whatever tolerance is in
    # force, INCLUDING a probe-measured one, not a fallback for runs the probe never reached.
    # On the good-era corpus (stamp 91d196146f3d) it raised the tolerance on 298 of 439 runs --
    # 289 of them runs the probe HAD supplied a value to.  `use_measured_tolerance` is the arm
    # that lets the measured value govern alone, and it is the ONLY thing the conjunct below adds.
    if derived_ab and not _use_measured_tol:
        _ab_hit("rtol.checked", float(tol))
        if tol < _DERIVED_RTOL:
            _ab_hit("rtol.fired", float(tol))
            tol = _DERIVED_RTOL
        else:
            _ab_hit("rtol.already_above", float(tol))
    # The number the step loop will actually compare against, stamped once per run in every
    # arm.  `_ab_hit` keeps the LAST value seen and this site fires once per run, so this is
    # that run's value, not a session total.
    _ab_hit("rtol.in_force", round(float(tol), 8))
    space_sum = {"log_sigma": 0.0, "sigma": 0.0}
    space_locked = None if space_mode == "auto" else space_mode
    hist = []
    # The two detector inputs, kept per step for the momentum branch
    # (`_momentum_signal`), which compares this step against the run's own recent
    # window.  Grown only in adaptive mode, where the detector runs at all.
    jrel_hist = []
    dabs_hist = []
    ord_px = None
    j_ema = None
    Iema = None
    prev_err = None
    prev_cons = None
    _LAST_ERRORS.clear()
    # Full-precision identity of the schedule this run actually consumed, plus
    # the seed.  Without both, two runs cannot be shown to be comparable at
    # all: the per-step sigma column is rounded to 6dp and _ms_info.sigmas_in
    # (the autotuner's INPUT) to 5dp, so schedules differing below those
    # precisions look identical in the log even though the model amplifies the
    # difference into a visibly different image within two steps.  Measured in
    # DEAD_FEATURES.md section 13; see _sigmas_digest for the specific case
    # that made this necessary.
    _sig_digest = _sigmas_digest(sigmas)
    try:
        _seed_val = int(extra_args.get("seed"))
    except (TypeError, ValueError):
        _seed_val = None
    _LAST_CFG = {"mode": "aflops",
                 "_build": _BUILD,
                 # THE RUN ID, so a saved latent can be named from the log and the two files carry
                 # a permanent shared key.  Top level rather than inside `cfg`, because that is the
                 # shortest path for a JSON-extract node: `aflops_cfg.run_id`.  See `_new_run_id`
                 # for why a per-RUN id is needed and why every existing digest is the wrong tool
                 # (they are config identities, and replicates share all of them).
                 "run_id": _new_run_id(),
                 # WHICH KIND OF MODEL KEY this run used.  `param_digest` or `model_hash` means the
                 # key distinguishes two checkpoints; a `_DEGRADED` value means it MIGHT NOT, and a
                 # reader should exclude the run from any cross-model comparison.  Stamped because
                 # the `wc:` collision was invisible in the log for exactly as long as it took to
                 # think of looking.
                 "_model_key_kind": _MODEL_KEY_KIND,
                 # THE PATCH IDENTITY IN THE KEY (LoRA / wrappers / hooks).  A LoRA run
                 # and a no-LoRA run have the same base digest, so without this stamp a
                 # cache miss caused by loading a LoRA would look like one caused by
                 # nothing.  Measured: log_00072/73 (LoRA) and log_00066-71 (no LoRA) all
                 # carried `pd:61457a9d6c516c11`; the tag is what separates them.  Read
                 # off THIS run's patcher rather than the module global, so a run with
                 # `probe_enabled` off still stamps its own model.
                 "_model_patch_tag": _patcher_patch_tag(
                     _navigate_to_patcher(model)),
                 # WHERE THIS RUN'S WALL TIME WENT.  PD-AOS logged no timing at
                 # all before this, so "the probes feel slow" could not be
                 # answered from the corpus.  Milliseconds per phase, plus the
                 # probe cache states and the number of probe forward passes the
                 # QUEUE paid (node-side probe included, since the counter is a
                 # process counter and the engine reports its delta).  See
                 # `_PHASE_MS` for the measured regression this exists to
                 # attribute.  Keys: `pre_sampling_ms`, `model_key_ms`,
                 # `model_probe_ms`/`_state`, `cond_probe_ms`/`_state`,
                 # `node_probe_ms`/`_state`, `gap_ctx_ms`, `guider_cfg_ms`,
                 # `schedule_sig_ms`, `latest_profile_ms`, `probe_model_calls`
                 # (the engine's own share of the probe forward passes) and
                 # `probe_calls_node_side` (the NODE-side probe's forward passes,
                 # the work done before this run began).
                 #
                 # MEASURED ON THE OPERATOR'S FIRST RUN OF THIS INSTRUMENTATION:
                 # `model_key_ms` was 6057 ms and 6386 ms per queue -- 32-51 % of
                 # the whole pre-sampling window.  PD-AOS had no timing at all
                 # before this, so a 6-second-per-queue cost in the one function
                 # whose docstring promised "bounded work" was invisible for as
                 # long as it took to build this record.
                 "_phase_ms": _PHASE_MS,
                 # STANDALONE PROVENANCE.  Without a positive field, a standalone
                 # run can only be identified by the ABSENCE of probe_key /
                 # _ms_info -- an implicit signal, which is the kind this project
                 # has repeatedly been burned by.  Stamped here so the corpus
                 # states it outright, and so it survives into the saved log.
                 "_standalone": bool(cfg.get("_standalone", False)),
                 "_standalone_note": cfg.get("_standalone_note"),
                 "seed": _seed_val,
                 # EVERY SEED THIS RUN USED, WRITTEN DOWN RATHER THAN INFERRED.  The run's seed is
                 # not the only one in play: the low-frequency renoise generator is seeded from a
                 # SHA-256 HASH of it, the eta noise sampler is seeded from it directly, and the
                 # probe's generator has a fixed seed of its own.  Without this block a reader
                 # comparing two runs has to REASON about which numbers were equal, and reasoning
                 # is how a difference gets missed -- the values are cheap to record and expensive
                 # to re-derive.  It also records WHETHER the noise sampler was built at all,
                 # because that is gated on the effective eta and is invisible in the output when
                 # the gate is closed.  Written for the log; nothing reads it.
                 "_seeds": {
                     "extra_args_seed": _seed_val,
                     "lf_seed_raw": _lf_seed_raw,
                     "lf_noise_seed": _lf_noise_seed,
                     "noise_sampler_built": noise_sampler is not None,
                     "eta_floor_cfg": _eta_floor_cfg,
                     "noise_color": str(cfg.get("noise_color", "white")),
                     "color_schedule": str(cfg.get("color_schedule", "constant")),
                     "is_adaptive_noise": bool(is_adaptive_noise),
                     "probe_gen_seed": 20260915,
                     "probe_resolution": int(cfg.get("probe_resolution", 48)),
                     "probe_steps": int(cfg.get("probe_steps", 10)),
                 },
                 # A/B provenance: which experiment arm this run was.
                 "_osm_ab_mode": _OSM_AB_MODE,
                 "_ev_ab_mode": _EV_AB_MODE,
                 # CORRECTOR ARM.  True = current corrector, False = the
                 # archive-known-good corrector.  Stamped explicitly so a run's
                 # label can never contradict its evidence; the second
                 # fingerprint is per-step `corr`, which can only read "raw" on
                 # the True arm.
                 "corrector_ab": bool(corrector_ab),
                 # ORDER ARM.  True = adaptive order ladder, False = forced
                 # order 1.  Per-step `eff_cap`/`order` in the log is the second
                 # fingerprint: forced order 1 leaves eff_cap == 1 everywhere.
                 "order_ab": bool(order_ab),
                 # HISTORY-CONSISTENCY GATE ARM.  OFF (the default) cuts the whole ladder
                 # when one global cosine is negative; ON gates only the locally negative
                 # pixels and leaves the stale/detailed pixels at their order.  Per-step
                 # `hist_cos_px_neg`/`hist_cos_px_on` are the fingerprints.
                 "hist_cos_per_pixel": bool(cfg.get("hist_cos_per_pixel", False)),
                 # SECOND-ORDER TERM ARM.  OFF (the default) applies the slope term to
                 # every pixel; ON applies it only where the admitted order is >= 2.
                 # Per-step `second_order_px` is the fraction that received it.
                 "second_order_per_pixel": bool(cfg.get("second_order_per_pixel", False)),
                 "second_order_raw_lambda": bool(cfg.get("second_order_raw_lambda", False)),
                 # EULER-SAFETY CUT ARM: True = shipped (cut when the corrected
                 # step departs from the plain step by more than tol), False =
                 # keep the corrected step.  Per-step fingerprint is `corr`,
                 # which can only read "raw" when this is True.
                 "euler_cut": bool(euler_cut),
                 # WHICH VARIABLE THE CUT TESTS.  "corrected" = shipped (corrected
                 # step vs plain step); "midpoint" = derived (ladder midpoint vs
                 # plain step, the term the blend does not already bound).
                 "cut_var": str(cut_var),
                 # DERIVED CONSTANTS ARM.  True = the derived forms of the series
                 # switch, the x0m envelope and the default rtol; False = shipped.
                 "derived_ab": bool(derived_ab),
                 # THE TOLERANCE-MINIMUM ARM IN FORCE, the value the probe supplied BEFORE the
                 # minimum, and the number the step loop actually compared against.  Three
                 # separate fields on purpose (rule 8): from `tol` alone a reader cannot tell
                 # "the minimum raised it" from "the model's own tolerance was already above
                 # it", and from the arm alone they cannot tell what it did.  `rtol_measured`
                 # is the value the probe's own measurement produced; `tol_in_force` is what
                 # every step of this run used, and it is also stamped per step as `tol`.
                 "use_measured_tolerance": bool(_use_measured_tol),
                 "rtol_measured": round(float(_rtol_measured), 8),
                 "tol_in_force": round(float(tol), 8),
                 # WHICH A/B PATHS ACTUALLY FIRED.  A flag wired to nothing is
                 # indistinguishable from a flag whose effect is below the noise
                 # floor, and that ambiguity already cost two rounds of dead A/B
                 # runs (five model pairs, zero differing fields).  Stamped as the
                 # final snapshot after the loop so the log states it outright.
                 # See `_ab_hit` / scratch_ab_reach_rig.py.
                 "ab_reach": _ab_reach_snapshot(),
                 "sigmas_digest": _sig_digest,
                 # The EFFECTIVE lambda clamp, so a log self-describes which
                 # clamp produced it.  Previously this had to be inferred from
                 # the lambda values, which is ambiguous: lambda > 0 is legal
                 # under [-1,1] but not under [-3,0].
                 "aflops_lam_min": (cfg.get("aflops_lam_min")
                                    if cfg.get("aflops_lam_min") is not None
                                    else _AFLOPS_LAM_MIN),
                 "aflops_lam_max": (cfg.get("aflops_lam_max")
                                    if cfg.get("aflops_lam_max") is not None
                                    else _AFLOPS_LAM_MAX),
                 "_lam_clamp_always_auto": True,
                 "lam_clamp_from_probe": bool(_lam_min_steps is not None),
                 "lam_clamp_probe_range": ([round(min(_lam_min_steps), 4),
                                            round(max(_lam_min_steps), 4)]
                                           if _lam_min_steps else None),
                 # the rule's actual per-step output, so a log shows what it
                 # chose rather than only the range it spanned
                 "lam_clamp_probe_bounds": ([round(v, 4) for v in _lam_min_steps]
                                            if _lam_min_steps else None),
                 # The autotuner's schedule decision and the bias/warp widgets
                 # that produced it.  Without this a warp/bias A/B cannot be
                 # attributed from the logs at all -- the sigma grid identifies
                 # the PROMPT (the schedule is curvature-driven), not the knob.
                 "_shift_schedule_node": cfg.get("_shift_schedule_node"),
                 "_ms_info": cfg.get("_ms_info"),
                 "order": order, "order_cap": order_cap,
                 "adaptive_order": adaptive_on,
                 "ladder_auto": ladder_auto,
                 "space": space_locked or "auto",
                 "corrector": corrector,
                 "eta_base": eta_base, "s_noise": s_noise,
                 "err_mode": err_mode,
                 "noise_color": cfg.get("noise_color", "white"),
                 "color_schedule": cfg.get("color_schedule", "constant"),
                 "color_beta_max": cfg.get("color_beta_max", 1.5),
                 "color_beta_min": cfg.get("color_beta_min", -0.5),
                 "model_info": (None if model_info is None else dict(model_info)),
                 "steps": n_cand, "distilled_hint": dist_hint,
                 "sigma_data": sigma_data,
                 "probe_key": cfg.get("_probe_key"),
                 "cond_probe_key": cfg.get("_cond_probe_key"),
                 "lf": {"enabled": lf_on,
                        "strength": (lf_gains["k"] if lf_gains else 0.0),
                        "detail": (lf_gains["d"] if lf_gains else 0.0),
                        "volatility": (lf_gains["v"] if lf_gains else 0.0),
                        "resharpen": (lf_gains["s"] if lf_gains else 0.0),
                        "clamp": _lf_clamp,
                        "drift": (lf_gains["drift"] if lf_gains else 0.0),
                        # THE FRAGILITY GAIN WAS NOT LOGGED UNTIL 2026-10-02, and that made
                        # the `_FRAG_ERR_REF` / `_FRAG_GAIN_MIN` grounding UNVERIFIABLE from
                        # the corpus: the value multiplies the gentle-step term in fragile
                        # regions, so a run could not be audited for the change it was
                        # supposed to carry (rule 8 -- log enough for the decision to be
                        # auditable).  Measured on the first post-change runs by
                        # RECOMPUTING it from the profile instead, which is exactly the
                        # workaround this field removes.  Stamped beside the other gains so
                        # one run shows the whole set.
                        "frag_gain": (lf_gains.get("frag_gain") if lf_gains else None),
                        "guard_feed": lf_feed,
                        "runtime_gate": (lf_on and lf_closed_loop
                                         and bool(cfg.get("lf_runtime_gate", True))),
                        "v3": {"closed_loop": lf_closed_loop,
                               "env_fade": bool(lf_gains and lf_gains.get("env_fade")),
                               "spatial_health": lf_spatial_health,
                               "band_refresh": (lf_on and bool(cfg.get("lf_band_refresh", True))
                                                and bool(lf_gains and lf_gains.get("hi_ref"))),
                               "dir_vol": (lf_on and bool(cfg.get("lf_dir_vol", True))),
                               "semantic_prior": lf_prior is not None,
                               "v_sched": bool(lf_gains and lf_gains.get("v_sched")),
                               "rescue": lf_rescue_on},
                        "evidence": lf_evid},
                 "cfg": {k: v for k, v in cfg.items() if k != "_model_info"}}
    if log_errors and lf_on:
        logging.info(
            "[PD-AOS] local schedule field auto-tuned from probe "
            "discovery (%s: oversteer=%.2f oversmooth=%.2f err=%.4f "
            "spike=%.2f distilled=%s -> strength=%.2f volatility=%.2f "
            "detail=%.2f resharpen=%.2f clamp=%.2f drift=%.2f | v3: n=%s "
            "conf=%s eg=%s dir=%s closed_loop=%s env_fade=%s spatial_health=%s "
            "v_sched=%s band_ref=%s prior=%s rescue=%s)",
            lf_evid.get("source", "?"), lf_evid.get("ov", 0.0),
            lf_evid.get("osm", 0.0), lf_evid.get("err", 0.0),
            lf_evid.get("spike", 1.0), lf_evid.get("dist", False),
            lf_gains["k"], lf_gains["v"], lf_gains["d"],
            lf_gains["s"], _lf_clamp, lf_gains["drift"],
            lf_evid.get("n"), lf_evid.get("conf"), lf_evid.get("eg"),
            lf_evid.get("dir"), lf_closed_loop,
            lf_gains.get("env_fade"), lf_spatial_health,
            bool(lf_gains.get("v_sched")), bool(lf_gains.get("hi_ref")),
            lf_prior is not None, lf_rescue_on)
    elif (log_errors and not lf_on and bool(cfg.get("lf_enable", True))
          and lf_evid is not None and lf_evid.get("source") == "none"):
        logging.info(
            "[PD-AOS] local schedule field OFF: no probe evidence "
            "(the field is automatic-only; enable the Model Probe or "
            "Cond Probe so its gains can be derived -- it never runs on "
            "guessed defaults)")
    # The engine's own sampling call.  If a model-internal cache (prefix/KV
    # caches keyed by full-resolution sequence shapes) blows up mid-run --
    # e.g. slots left behind by an older ComfyUI revision, an interrupted
    # earlier run, or any other source outside our probe isolation -- retry
    # once with the model's caches disabled and stay on that path for the
    # rest of the run.  Normal runs keep the model's own caching behaviour
    # completely untouched.
    _run_caches_off = [False]

    def _run_model_call(xx, ss):
        if _run_caches_off[0]:
            ea = dict(extra_args)
            _mopt = _probe_model_options(ea, strip_hooks=False)
            if _mopt is not None:
                ea["model_options"] = _mopt
            return model(xx, ss, **ea)
        try:
            return model(xx, ss, **extra_args)
        except Exception as e:
            ea2 = dict(extra_args)
            _mopt = _probe_model_options(ea2, strip_hooks=False)
            _fixed = _mopt is not None
            if not _fixed:
                # Last resort for models whose own per-run cache carries the
                # poisoned state without any options-level switch: flip the
                # instance switch off for the rest of this run.  A disabled
                # cache is transparent (it just recomputes), so this can
                # never change the run's results.
                _reset = _find_prefix_reset(model)
                if callable(_reset):
                    try:
                        _reset(False)
                        _fixed = True
                    except Exception:
                        _fixed = False
            if not _fixed:
                raise
            if _mopt is not None:
                ea2["model_options"] = _mopt
            logging.warning("[PD-AOS] model call failed (%s: %s); retrying "
                            "with the model's internal caches disabled for "
                            "the rest of this run", type(e).__name__, e)
            _run_caches_off[0] = True
            return model(xx, ss, **ea2)
    for i in range(n_cand):
        # The first step's arrival ends the PRE-SAMPLING window: everything the
        # engine spent before it -- model key, model probe, cond probe -- is
        # inside it.  This is the quantity ComfyUI's own log can measure from
        # outside (2.16 s on build fccd7e2a8a18, 13.2-15.6 s from bf9dbade0f8a);
        # stating it here lets a run's log be compared with that clock.
        if "pre_sampling_ms" not in _PHASE_MS:
            try:
                _phase_set("pre_sampling_ms",
                           (time.perf_counter() - _t_engine) * 1e3)
            except Exception:
                pass
        # The engine's own share of the probe forward passes.  Written here, at
        # the top of each step, because the run's report dict is built BEFORE the
        # loop and the log holds a reference to `_PHASE_MS`: a value set only at
        # stamp time would never appear, and every engine-side probe call has
        # already happened by step 1.
        try:
            _phase_set("probe_model_calls",
                       _PROBE_CALL_COUNT[0]
                       - int(_PHASE_MS.get("probe_calls_at_entry", 0)))
        except Exception:
            pass
        s = float(sigmas[i])
        sn = float(sigmas[i + 1])
        k = i + 1
        # THE x0 ESTIMATOR THIS STEP FED TO THE INTEGRATOR, recorded with the step so
        # that the NEXT step's `v_prev`/`h_prev` come from the same estimator (C4).
        # None when the integrator did not run this step, and then the next step falls
        # back to the raw measurement -- the only estimator that exists for that step.
        _x0_fed_hist = None
        x_input = x
        x0 = _run_model_call(x, sigmas[i] * ones)
        calls += 1
        u_log = math.log(max(s, 1e-8))
        u_sig = s
        frac = min(max((math.log(max(s_max_n, 1e-8)) - math.log(max(s, 1e-8))) /
                       max(math.log(max(s_max_n, 1e-8)) - math.log(max(s_min_n, 1e-8)), 1e-8), 0.0), 1.0)
        jmap = (x0 - hist[-1][2]).pow(2).mean(dim=1).sqrt() if len(hist) > 0 else None
        eta_stuck = eta_jump_rel = eta_detail_abs = None
        eta_probe = None
        eta_momentum = eta_momentum_ratio = eta_growth_rate = None
        if cfg["eta_mode"] == "adaptive":
            eta_jump_rel, eta_detail_abs = _stuck_components(x0, jmap)
            _dir_cos = None
            if jmap is not None and len(hist) >= 2:
                _d1 = (x0.detach() - hist[-1][2]).float()
                _d2 = (hist[-1][2] - hist[-2][2]).float()
                _den = _d1.pow(2).sum().sqrt() * _d2.pow(2).sum().sqrt()
                if float(_den) > 1e-8:
                    _dir_cos = float((_d1 * _d2).sum() / _den)
            # THE MOMENTUM BRANCH READS THE PREVIOUS STEPS, so it is fed the
            # series BEFORE this step is appended: a window that contained the
            # value under test could never see a drop against it.  Computed even
            # when `escape_mode` is "flatness" (and even when the escape is
            # switched off entirely) so the log carries the branch's own reading
            # on BOTH arms of an A/B -- a comparison needs the same field on both
            # sides, and the value is discarded by `_global_stuck_signal` unless
            # the mode selects it.
            if eta_jump_rel is not None:
                eta_momentum, eta_momentum_ratio, eta_growth_rate = _momentum_signal(
                    eta_jump_rel, eta_detail_abs, jrel_hist, dabs_hist, cfg)
            eta_stuck = (_global_stuck_signal(x0, jmap, cfg,
                                              dir_cos=_dir_cos, s=float(s),
                                              jhist=jrel_hist, dhist=dabs_hist)
                         if jmap is not None else 0.0)
            if eta_jump_rel is not None:
                jrel_hist.append(float(eta_jump_rel))
                dabs_hist.append(float(eta_detail_abs))
            eta_eff = _adaptive_eta(prev_err, prev_cons, s, sn, s_max_n, s_min_n,
                                    space, sigma_data, s_noise, model_info,
                                    eta_base, cfg,
                                    few_step=(n_cand <= 8 or dist_eff),
                                    stuck=eta_stuck)
        else:
            eta_eff = _eta_schedule(eta_base, cfg["eta_mode"], frac, s, s_max_n,
                                    cfg["falloff"], cfg["eta_min"])
        # ---- THE PROBE-DERIVED SCHEDULE --------------------------------
        # REPLACES the level this step would otherwise use, rather than adding to
        # it, and it applies to EVERY eta_mode: the derivation answers "how much
        # can this model absorb at this sigma", which is a question no mode-based
        # schedule asks.  `eta` stays the CEILING, so a user who sets it lower
        # still gets a lower level, and the run log carries both numbers
        # (`eta_eff` here, `eta_probe` beside it) so the two are comparable.
        #
        # Falls through silently when the probe supplied no curve -- the schedule
        # is only as good as its measurement.
        if _eta_probe_fn is not None:
            # `s` is the step's sigma, already a python float at this point.
            _ep = _eta_probe_fn(s)
            if _ep is not None:
                eta_probe = float(_ep)
                # REPLACE, DO NOT MINIMISE.  The first version of this line was
                # `eta_eff = min(eta_eff, eta_probe * min(1.0, eta_base))`, which
                # made the whole derivation INERT on every run it was switched on
                # for: in `adaptive` mode `eta_eff` is the ESCAPE's output, and
                # that is exactly 0.0 on every step where the detector does not
                # fire -- which was every step of every run measured -- so
                # `min(0.0, anything)` is 0.0.  Measured consequence: the widget
                # moved `eta_probe` exactly as designed (budget 1.0 -> 1.0,
                # 0.35 -> 0.196, 0.2 -> 0.021..0.544, 0.01 -> 0.00005) while
                # `eta_eff` stayed 0.0 on all 16 steps of every run, so the images
                # could not have changed.
                #
                # The derived schedule is an ALTERNATIVE to the escape, not a
                # ceiling on it: it answers "how much can this model absorb here",
                # which needs no stuck-detector.  So it sets the level, and `eta`
                # remains the user's ceiling on top.
                eta_eff = eta_probe * min(1.0, eta_base)
        if not (s >= float(cfg["s_tmin"]) and (float(cfg["s_tmax"]) <= 0.0 or s <= float(cfg["s_tmax"]))):
            eta_eff = 0.0
        if dist_eff and i == n_cand - 2:
            eta_eff = 0.0
        outlier_sig = None
        guard_trip = guard_active = False
        if guard is not None and len(hist) > 0 and sn > 1e-8:
            outlier_sig = _jump_signal(x0, hist[-1][2], x, q=gquant)
            z_relax = float(cfg.get("guard_phase_relax", 1.0))
            z_eff = float(cfg["guard_sensitivity"]) * (1.0 + z_relax * (1.0 - frac))
            guard_trip, guard_active = guard.step(outlier_sig, z=z_eff)
        if eta_eff < 1e-6:
            eta_eff = 0.0
        s2, s_up = _ancestral_split(s, sn, eta_eff, space)
        amask = None
        an_detected = False
        step_cons = None
        if jmap is not None and s2 > 1e-8:
            _xn = float(x0.float().norm().clamp_min(1e-8))
            _jn = float(jmap.float().norm().clamp_min(1e-8))
            _m0 = _consistency_step(s, s2, space)
            step_cons = _compute_consistency(_jn, _xn, _m0)
        j_prev_max = torch.stack(jhist, 0).max(dim=0).values if jhist else None
        elev = None
        if an_enable and jmap is not None:
            elev = _elevation_mask(jmap, an_win, an_value, x0, mult=an_sustain)
        if (an_enable and k >= int(an_from * n_cand)
                and len(jhist) >= (1 if n_cand <= 6 else 2)
                and j_prev_max is not None and s2 > 1e-8 and jmap is not None):
            an_phase_mult = 1.0 + an_phase_relax * (1.0 - frac)
            amask = _anomaly_mask(x0, hist[-1][2], j_prev_max,
                                  an_win,
                                  an_spatial * an_phase_mult,
                                  an_novelty * an_phase_mult,
                                  an_value,
                                  jmap=jmap,
                                  region_z=an_region_z)
            if prev_elev is not None and elev is not None:
                sust = elev & prev_elev
                if len(jhist) > 0:
                    sust = sust & (jmap > jhist[-1])
                sust = _clean_mask(sust, erode=1, dilate=2)
                if bool(sust.any()):
                    amask = amask | sust
            if bool(amask.any()):
                an_detected = True
                # REACHABILITY, because `an_enable` is an A/B switch and a toggle wired to nothing
                # is indistinguishable from one whose effect is below the noise floor (rule 9).
                # The per-step `an_detected` field already says WHICH step fired; this counter says
                # whether the arm fired AT ALL, which is the thing an off/on comparison needs.
                # Without it, "an_enable=False changed nothing" cannot be told apart from
                # "an_enable=False never reached the engine".
                _ab_hit("an.detected")
        if jmap is not None:
            jhist.append(jmap)
            if len(jhist) > 3:
                jhist.pop(0)
        prev_elev = elev
        if dwell_on and jmap is not None:
            # per-pixel churn envelope: max-decay EMA of the jump map,
            # normalized by the image median (>= median: keeping pace, no
            # dwell; ~2x median: full dwell weight)
            _jdw = jmap.float().unsqueeze(1)
            dw_ema = (_jdw if dw_ema is None
                      else torch.maximum(_jdw, _DWELL_EMA * dw_ema))
            _b_dw = dw_ema.shape[0]
            _dw_med = dw_ema.reshape(_b_dw, -1).median(dim=1).values\
                .reshape(-1, *([1] * (dw_ema.dim() - 1))).clamp_min(1e-12)
            dw_w = (((dw_ema / _dw_med) - 1.0) / (_DWELL_R_HI - 1.0))\
                .clamp(0.0, 1.0)
        lf_m = lf_rho = None
        lf_diag = None
        lf_cum_next = None
        # Measured convergence point of the trajectory (probe profile).  Used
        # to gate refresh-noise injection: below it the model has no movement
        # left, so injected noise is not integrated away -- it survives to the
        # output.  Rigged from CFG1 (distilled): the rescue applied a refresh
        # to 38% of pixels (rho 0.0155) on the penultimate step whose
        # sigma_next == conv_sigma == 0.173927, and the terminal step only
        # re-predicts x0, so the noise landed in the image -- a plain
        # simple+euler run injects nothing there.
        _conv_sigma = None
        for _pf in (lf_prof_c, lf_prof_m):
            if isinstance(_pf, dict):
                _cv = _pf.get("conv_sigma")
                try:
                    if _cv is not None and float(_cv) > 0.0:
                        _conv_sigma = float(_cv)
                        break
                except (TypeError, ValueError):
                    continue
        if lf_on:
            _kn = k / n_cand
            if lf_damp is not None:
                lf_damp = lf_damp * 0.65
                lf_damp = torch.where(lf_damp < 0.02,
                                      torch.zeros_like(lf_damp), lf_damp)
            if lf_feed:
                new_damp = None
                if guard_trip and jmap is not None:
                    flat_j = jmap.reshape(jmap.shape[0], -1).float()
                    thr = torch.quantile(
                        flat_j, min(max(gquant, 0.0), 0.999),
                        dim=1).view(-1, *([1] * (jmap.dim() - 1)))
                    new_damp = (jmap.float() >= thr).float().unsqueeze(1) * 0.9
                if amask is not None and bool(amask.any()):
                    msk = amask.float().unsqueeze(1)
                    new_damp = msk if new_damp is None else torch.maximum(new_damp, msk)
                if new_damp is not None:
                    lf_damp = (new_damp if lf_damp is None
                               else torch.maximum(lf_damp, new_damp))
            lf_wdir = None
            if bool(cfg.get("lf_dir_vol", True)) and len(hist) >= 2:
                _d1 = (x0.detach() - hist[-1][2]).float()
                _d2 = (hist[-1][2] - hist[-2][2]).float()
                _num = (_d1 * _d2).sum(dim=1)
                _den = (_d1.pow(2).sum(dim=1).sqrt()
                        * _d2.pow(2).sum(dim=1).sqrt())
                _cos = torch.where(_den > 1e-8, _num / _den.clamp_min(1e-8),
                                   torch.zeros_like(_num))
                if _kn >= 1.0 / 3.0:
                    _cos_mean = _cos.mean()
                    lf_dir_acc[0] += float(_cos_mean)
                    lf_dir_acc[1] += 1
                lf_wdir = _lf_dir_weight(
                    _cos, 1.0 / math.sqrt(max(1, x.shape[1]))).unsqueeze(1)
            g_step = lf_gains
            _vs = lf_gains.get("v_sched")
            if _vs and i < len(_vs) and float(_vs[i]) != 1.0:
                g_step = dict(lf_gains)
                g_step["v"] = _lf_v_step(lf_gains["v"], float(_vs[i]))
            if _lf_rescue is not None:
                _lf_rescue["s"] = float(s)
            lf_m, lf_rho, vol_new, med_new, lf_resc, lf_cum_next = _lf_step_field(
                x0, jmap, lf_vol, lf_medref, lf_damp,
                _lf_window(_kn), _lf_vol_window(_kn), g_step, kn=_kn,
                trust=(lf_trust if lf_closed_loop else None),
                gate=(lf_gate if lf_closed_loop else None),
                yield_map=lf_suppress,
                w_dir=lf_wdir, prior=lf_prior, fragility=lf_frag,
                spatial_health=lf_spatial_health,
                rescue=_lf_rescue, resc_state=lf_resc_state,
                eg_n=float(lf_evid.get("eg_n") or 0.0),
                step_d=(x_input - x0), gdir=lf_gdir,
                cum=lf_cum)
            lf_resc_rw = None
            if lf_resc is not None:
                lf_resc_state = lf_resc.get("state")
                lf_resc_rw = lf_resc.get("rw")
            if vol_new is not None:
                lf_vol = vol_new
            lf_medref = med_new
            # Per-step local-field telemetry is computed for the report
            # regardless of log_errors (the report collects per-step data
            # whenever the run is < 64 steps, so gating this on log_errors
            # silently blanked the field's diagnostics for the JSON report).
            lf_diag = {"m_min": round(float((1.0 + lf_m).min()), 4),
                       "m_max": round(float((1.0 + lf_m).max()), 4),
                       "m_abs": round(float(lf_m.abs().mean()), 4),
                       "rho_max": round(float(lf_rho.max()), 4),
                       "damp": (round(float((lf_damp > 0.05).float().mean()), 4)
                                if lf_damp is not None else 0.0),
                       "env": round(_lf_env_fade(_kn, float(lf_evid.get("eg_n") or 0.0))
                                    if lf_gains.get("env_fade") else 1.0, 3),
                       "vmul": (round(float(g_step["v"])
                                      / max(lf_gains["v"], 1e-9), 3)
                                if (_vs and i < len(_vs)) else None),
                       "trust": (round(float(lf_trust.mean()), 3)
                                 if lf_trust is not None else None),
                       "tmin": (round(float(lf_trust.min()), 3)
                                if lf_trust is not None else None),
                       "gate": (round(float(lf_gate), 3)
                                if lf_gate is not None else None),
                       "yield": (round(float(lf_suppress.mean()), 3)
                                 if lf_suppress is not None else None),
                       # LOCALIZATION OF THE LOCAL FIELD'S OWN CORRECTION.  `lf_m` and `lf_rho`
                       # ARE the deviation from the plain path (plain = 0 for both), so their
                       # tile extremes answer "did the field act in a SMALL REGION at THIS
                       # step?" -- a WITHIN-RUN question needing no second run and no arm
                       # pairing, and therefore NOT inheriting the measured run-to-run
                       # non-reproducibility that would confound a corrected-vs-plain A/B.
                       # A large SHARE late in the run on a few tiles is the artifact signature
                       # the whole-tensor channels average away.  Each is
                       # (max tile mean, share of the total, y, x) or None.
                       "m_tile": (_tile_extremes(lf_m) if lf_m is not None else None),
                       "rho_tile": (_tile_extremes(lf_rho) if lf_rho is not None else None),
                       "resc": ({"cov": round(float(lf_resc.get("cov", 0.0)), 4),
                                 "hit": lf_resc.get("hit", 0.0),
                                 "lf_share": lf_resc.get("lf_share", 0.0)}
                                if lf_resc is not None else 0.0),
                       # The block size the rescue actually measured with, and the
                       # latent it measured on.  `_lf_block_for` returns 4 at
                       # 128x128 -- the old hardcoded value -- so these two fields
                       # are how a later reader VERIFIES from the corpus that the
                       # scale-free rule changed nothing at the operator's usual
                       # resolution, instead of taking that on trust.
                       "blk": _lf_block_for(jmap) if jmap is not None else None,
                       "latent": ([int(x.shape[-2]), int(x.shape[-1])]
                                  if x is not None else None),
                       "drift_p90": (round(float(lf_cum_next.abs()
                                                  .quantile(0.9)), 4)
                                     if lf_cum_next is not None else 0.0),
                       "drift": round(float(lf_gains.get("drift") or 0.0), 3)}
        x1 = exact_step(x, x0, s, s2)
        use_space = space_locked or "log_sigma"
        # Order-k correction needs >= k prior velocity samples.  The old
        # len(hist)+1 let an order-2 midpoint + exponential-integrator slope
        # run on the FIRST correction step (1 history entry): a 2-point
        # interpolation across the sigma~1 commit, where x0 flips
        # discontinuously.  Log rig (CFG5log/CFG1log): the only >1x-tol step
        # errors at that point are exactly these first order-2 corrections
        # (2.79x / 6.29x tol; every order-1 step nearby <= 0.41x tol).
        eff_cap = max(1, min(order_cap, len(hist)))
        if guard_active:
            eff_cap = min(eff_cap, 1)
        # ORDER A/B: force the pure order-1 exponential integrator.  Placed after
        # every other eff_cap decision so it is unconditional when off.
        if not order_ab:
            _ab_hit("order.forced_to_1")
            eff_cap = 1
        _ab_hit_max("order.eff_cap_pre_gate", eff_cap)
        hist_cos_val = None
        hist_cos_px_neg = None
        _hist_cos_px_on = bool(cfg.get("hist_cos_per_pixel", False)) and adaptive_on
        if (bool(cfg.get("history_consistency_gate", True))
                and len(hist) >= 2 and eff_cap >= 2 and s2 > 1e-8):
            _hd1 = (x0.float() - hist[-1][2].float())
            _hd2 = (hist[-1][2].float() - hist[-2][2].float())
            _hn1 = _hd1.norm().clamp_min(1e-8)
            _hn2 = _hd2.norm().clamp_min(1e-8)
            hist_cos_val = float(((_hd1 * _hd2).sum() / (_hn1 * _hn2))
                                 .clamp(-1.0, 1.0))
            _thr = float(cfg.get("history_consistency_threshold", 0.0))
            if _hist_cos_px_on:
                # LOCAL cosines over channels.  The mask is applied to the admitted order
                # map below, after the ladder has built it.  The global `eff_cap` is NOT
                # cut here: a local reversal should gate that pixel, not the whole image.
                _pn1 = _hd1.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
                _pn2 = _hd2.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
                _cos_px = ((_hd1 * _hd2).sum(dim=1, keepdim=True)
                           / (_pn1 * _pn2)).clamp(-1.0, 1.0)
                hist_cos_px_neg = (_cos_px < -_thr)
                _ab_hit("histcos.px.applied",
                        float(hist_cos_px_neg.float().mean()))
            else:
                if hist_cos_val < -_thr:
                    eff_cap = 1
                elif (hist_cos_val < _thr and eff_cap > 2
                      and not (adaptive_on
                               or bool(cfg.get("order_pixel_blend", True)))):
                    eff_cap = 2
        if wmax_adapt and prev_err is not None:
            smooth = max(0.0, min(float(cfg["rtol"]) / max(prev_err, 1e-8), 3.0))
            wmax_limit = wmax_base * (1.0 + wmax_adapt_k * smooth)
        else:
            # THE FEEDBACKLESS LIMIT IS THE CEILING, NOT THE FLOOR -- changed 2026-10-03.
            #
            # WHAT WAS WRONG WITH THE OLD `else`.  It returned `wmax_base`, the FLOOR of the band, so
            # "no feedback" meant "minimum authority".  The operator's `wmax_eg_dir = 0` run exposed
            # the consequence: `0` is the ONLY arm where the feedback is live, and it was the only arm
            # whose limit MOVED -- decaying 4.500 -> 2.060 across the run, taking the mean admitted
            # order from 3.82 down to 2.9 in the last five steps.  The loop is
            #     error up -> authority down -> order down -> under-resolution -> error up,
            # a POSITIVE feedback loop with the WRONG SIGN, and `wmax_eg_dir = +1` was winning the
            # operator's preference only because it OVERRODE that loop (see the `_eg_n` branch below).
            # Measured: `0` beats `-1` (it starts at the ceiling) and loses to `+1` (it decays).
            #
            # WHAT CHANGED, and it is deliberately TWO things -- moving the constant alone would have
            # changed nothing visible, because `wmax_adapt` still defaulted to True:
            #   (i)  this branch now returns `wmax_base * (1 + wmax_adapt_k * 3)` -- the CEILING of the
            #        same band the adaptive arm spans -- so "no feedback" means "full authority";
            #   (ii) `wmax_adapt` now DEFAULTS TO FALSE, so the loop is off unless it is asked for.
            #
            # ANCHORS, verified in code rather than in my head (rule 7) by `scratch_wmax_loop_rig.py`:
            # `wmax_eg_dir = +1` and `-1` are BIT-IDENTICAL to before the change (4.5000 and 1.8000 for
            # krea2Turbo), because both ended at a band endpoint that did not move; and
            # `wmax_eg_dir = 0` now equals what `+1` used to be.  Setting `wmax_adapt = True` restores
            # the old ADAPTIVE arm exactly, so the change is reversible by config and A/B-able.
            #
            # AND THE SAME CLAIM RE-CHECKED ON REAL RUNS, because the unit rig fed the `eg_n` gate a
            # passing value and the gate is what decides whether this branch is reached at all
            # (`scratch_eg_gate_rig.py`, 56 corpus runs over the three builds that carry the knob):
            # the gate passes on 56/56, and ALL 41 runs whose `|dir| = 1` hit their endpoint EXACTLY --
            # `lo` on 23 `-1` runs, `hi` on 18 `+1` runs, ZERO step-limits moved.
            #
            # ONLY `|dir| = 1` IS AN ANCHOR, and that is the part the unit rig's summary overstated.
            # This branch interpolates FROM the incoming `wmax_limit`, so removing the feedback moved
            # the map's INPUT to `hi` and the intermediate arms moved with it: `+0.5` is now identical
            # to `+1` (both `hi`), and `-0.5` interpolates from `hi` rather than from the feedback
            # value.  Measured: 144 of 408 step-limits on the `0` and `+/-0.5` arms move, which is the
            # intended effect on the `0` arm and the unavoidable consequence on the others.
            #
            # ONE LATENT COUPLING, recorded because it is not obvious from here: the gate is
            # `_eg_n > _WMAX_EG_N_GATE`, `eg_n` is a PROBE-side quantity so it is constant for a whole
            # run (verified: constant within all 56 runs), and it passed on every observed run.  So
            # `-1`'s endpoint exactness is currently unconditional in practice but is not unconditional
            # in principle -- a run with `eg_n <= 0.3` would skip this branch entirely and `-1` would
            # do nothing.  No such run has been observed.
            #
            # A ROUTE THAT WAS TRIED AND REJECTED, recorded so it is not retried: folding the x2.5 into
            # the DERIVED `wmax_base` instead. `_eg_hi = wmax_base * (1 + 3k)`, so raising `wmax_base`
            # raises the CEILING with it -- `wmax_base` 4.5 would put `+1` at 11.25 and break the
            # anchor.  The band must not move; only which end the feedbackless case returns.
            #
            # CONSEQUENCE, accepted knowingly: the knob's positive half goes FLAT (`0` through `+1` all
            # give the ceiling), so the live range is `-1 .. 0`.  That is the measured SATURATION
            # showing up in the knob rather than a new dead zone: effective limits 2.179 / 2.373 /
            # 3.813 / 5.447 gave `x0_grad` 0.4000 / 0.3955 / 0.3861 / 0.3872, i.e. everything above
            # ~3.8 behaves the same and the ceiling is past it.
            #
            # AND THE PAYOFF: `wmax_limit` no longer depends on `prev_err`, so the `_eg_n` branch below
            # has nothing left to override.  It is kept for now -- removing it is a separate change --
            # but it is now a historical workaround rather than a load-bearing override.
            wmax_limit = wmax_base * (1.0 + wmax_adapt_k * 3.0)
        # ---- THE `eg_n` RESPONSE: ONE MONOTONE MAP, NOT A max/min PAIR ----------------------
        # WHAT THIS REPLACED, and why it was broken.  The first version of this parameter RAISED the
        # limit for a positive value (`max(wmax_limit, target)`) and LOWERED it for a negative one
        # (`min(...)`).  Two consequences, both measured:
        #   (i)  `-d` and `+d` were NOT two points on one line: `+0.5` was a FLOOR at the midpoint and
        #        `-0.5` a CEILING at the same midpoint, so they behaved completely differently while
        #        targeting the same number.  `+0.5` logged 3.8133 (it bound); `-0.5` logged 2.3734,
        #        which is the ADAPTIVE value, because its ceiling never bound at all.
        #   (ii) `_eg_dir = -0.5` and `_eg_dir = 0.0` therefore produced the SAME LIMIT -- two knob
        #        values with one behaviour, which is the defect the operator named: "Broken knobs that
        #        work partially are never good."
        # The operator also supplied the test a sane parameterisation has to pass, verbatim:
        # *"Obvously from a logic point of view -0.5 *should* be further away from a 1.0 value than a
        # +0.5 is."*
        #
        # THE MAP, chosen to satisfy that and to keep every anchor exact.  It is a ONE-SIDED
        # interpolation from the ADAPTIVE value toward the endpoint the knob points at:
        #     u >= 0 :  wmax_limit = adaptive + u * (hi - adaptive)
        #     u <  0 :  wmax_limit = adaptive + |u| * (lo - adaptive)
        # MONOTONE BY CONSTRUCTION, for every possible incoming value, because `lo <= wmax_limit <= hi`
        # always holds on BOTH paths into this branch: with the feedback on, `wmax_adapt` builds the
        # limit as `wmax_base * (1 + k*smooth)` with `smooth` clamped to [0, 3], so it lies in
        # `[lo, hi]`; with the feedback off (the default since 2026-10-03) it is returned as `hi`
        # directly.  Each side is therefore a linear interpolation between two fixed points on the
        # same side of the incoming value, so it cannot reverse direction.
        #
        # A FIRST ATTEMPT AT THIS IS WORTH RECORDING, because it looked reasonable and FAILED the
        # operator's own test: blending toward a geometric midpoint target,
        # `(1-|u|)*adaptive + |u|*tgt(u)`, is NOT monotone when the target lies on the opposite side
        # of the adaptive value from the direction of travel -- at adaptive = 2.179 it went
        # 2.4595 (u=-0.5) then DOWN to 2.4024 (u=-0.25), because the weight on the adaptive term moved
        # further than the target did.  Caught by checking monotonicity over the whole knob range at
        # three adaptive levels instead of trusting the anchors, which were all exact.
        #
        # ANCHORS, exact and checked rather than asserted:
        #   u =  0    -> `adaptive`, untouched (the branch-off behaviour)
        #   u = +1    -> `adaptive + 1*(hi - adaptive)` = `hi`, BIT-IDENTICAL to the shipped branch
        #   u = -1    -> `lo`
        _eg_n = float(lf_evid.get("eg_n") or 0.0)
        if _eg_n > _WMAX_EG_N_GATE and _eg_dir != 0.0:
            _eg_lo = wmax_base
            _eg_hi = wmax_base * (1.0 + wmax_adapt_k * 3.0)
            if _eg_dir > 0.0:
                wmax_limit = wmax_limit + _eg_dir * (_eg_hi - wmax_limit)
                _ab_hit("wmax.eg_raise")
            else:
                wmax_limit = wmax_limit + (-_eg_dir) * (_eg_lo - wmax_limit)
                _ab_hit("wmax.eg_lower")
        # ---- THE PER-STEP FUSED CAP (see ENGINE_DEFAULTS["probe_fuse_per_step"]) ------------
        # Placed AFTER the adaptive block above, so on the ON arm the fused cap is the step's
        # authority and the adaptive feedback cannot compound with it (the OFF arm never
        # reaches this test, so its accumulated `wmax_limit` is untouched).
        if _fuse_st is not None:
            _cap_now = wmax_limit
            # THE CAP IN FORCE IS RECORDED, because "did this arm move the cap the gate saw?" could
            # not be answered from a log and I got it wrong once by comparing the fused cap against
            # `cfg.wmax_base` -- the SCALAR -- when the gate reads `wmax_limit`, which the endgame
            # nudge has already moved by this point.  Both numbers reach the step record (see the
            # raw-measurements block, which is where `_raw` exists).
            _cap_pre_arm = float(_cap_now)
            _fc = _fuse_step_cap(_fuse_st, k - 1, cfg, cfg.get("rtol"),
                                 _calib_flags(lf_prof_m),
                                 cap_now=_cap_now, allow_raise=_fuse_raise)
            if _fc is not None:
                if not _fuse_raise and float(_fc) > float(_cap_now) + 1e-12:
                    _ab_hit("fuse.cap_clipped", round(float(_fc), 4))
                elif _fuse_raise and float(_fc) > float(_cap_now) + 1e-12:
                    _ab_hit("fuse.cap_raised", round(float(_fc), 4))
                wmax_limit = float(_fc)
                _fuse_caps.append(round(float(_fc), 6))
                _ab_hit("fuse.cap_applied", round(float(_fc), 4))
            else:
                _ab_hit("fuse.cap_unavailable")
            _cap_in_force = float(wmax_limit)
        eff = 1
        wmax = 0.0
        x0m = x0
        # The remainder gate's own record for this step (None when the arm is off, when
        # the order's scale was unmeasured, or when no order was admitted at all).
        _wgate_info = None
        # WHY THE ORDER SITS WHERE IT SITS.  `ord_mean`/`ord_max` are measured AFTER the
        # allowed-ladder maps the pixel's request down to the highest admitted order, so
        # they cannot distinguish "the criterion asked for 1" from "the criterion asked
        # higher and the cap clamped it" -- two completely different defects with two
        # completely different fixes.  These separate them and are TELEMETRY ONLY:
        # read-only reductions of tensors that already exist, assigned to locals that
        # nothing but the log record reads.
        #   ord_want   mean per-pixel order the CRITERION requested (before the clamp)
        #   ord_pstar  mean of p_star, the order whose extrapolation best predicts x0
        #   ord_innov  mean per-order innovation, orders 1..min(P,5) -- the curve whose
        #              argmin IS p_star, so it says which order the criterion prefers
        _ord_want = None
        _ord_ps = None
        _ord_innov = None
        # the Runge criterion's own record: for each order the per-pixel bound's mean and the
        # fraction of pixels it ADMITTED -- the spatial spread that is the whole point
        _ord_bound = None
        # how many pixels the forced order floor actually raised this step (rule 9: a knob
        # that cannot show it fired is indistinguishable from one wired to nothing)
        _ord_floor_n = None
        # THE DEMAND LEVEL THIS STEP PUT IN FORCE (1.0 = nothing raised).  Initialised here so the
        # record can stamp it on steps where the ladder block does not run; that block overwrites it.
        _fl_val = 1.0
        # THE ORDER MAP'S SPATIAL RECORD for this step (None unless the ladder ran, i.e. it is None
        # exactly when the map does not exist).  Initialised here so the log record always has the
        # field, which is what makes "the ladder did not run" distinguishable from "logging broke".
        _ord_spat = None
        if eff_cap >= 2 and s2 > 1e-8 and len(hist) >= 1:
            u2_log = math.log(max(s2, 1e-8))
            u2_sig = s2
            if space_locked is None:
                pts_log = [(h[0], h[2]) for h in hist] + [(u_log, x0)]
                pts_sig = [(h[1], h[2]) for h in hist] + [(u_sig, x0)]
                _lim2 = max(wmax_limit, wmax_mid_floor)
                for ce in range(eff_cap, 1, -1):
                    c_log = _mid_x0(pts_log, 0.5 * (u_log + u2_log), ce)
                    _ok_ce, _gi_ce = _order_gate(
                        pts_log, 0.5 * (u_log + u2_log), ce, c_log[1],
                        (_lim2 if ce == 2 else wmax_limit), cfg)
                    if not _ok_ce:
                        continue
                    if k >= 2:
                        c_sig = _mid_x0(pts_sig, 0.5 * (s + u2_sig), ce)
                        space_sum["log_sigma"] += _err_norm(
                            exact_step(x, c_log[0], s, s2), x1, err_mode,
                            x0_ref=x0, strength=grad_strength)
                        space_sum["sigma"] += _err_norm(
                            exact_step(x, c_sig[0], s, s2), x1, err_mode,
                            x0_ref=x0, strength=grad_strength)
                    x0m, wmax, eff = c_log[0], c_log[1], ce
                    _wgate_info = _gi_ce
                    break
                if eff >= 2 and k >= 5:
                    space_locked = min(space_sum, key=space_sum.get)
                    use_space = space_locked
            else:
                u_i = u_log if use_space == "log_sigma" else u_sig
                u_n = u2_log if use_space == "log_sigma" else s2
                pts = [((h[0] if use_space == "log_sigma" else h[1]), h[2])
                       for h in hist] + [(u_i, x0)]
                _lim2 = max(wmax_limit, wmax_mid_floor)
                for ce in range(eff_cap, 1, -1):
                    cand = _mid_x0(pts, 0.5 * (u_i + u_n), ce)
                    _ok_ce, _gi_ce = _order_gate(
                        pts, 0.5 * (u_i + u_n), ce, cand[1],
                        (_lim2 if ce == 2 else wmax_limit), cfg,
                        space=use_space)
                    if _ok_ce:
                        x0m, wmax, eff = cand[0], cand[1], ce
                        _wgate_info = _gi_ce
                        break
        blend_w_mean = None
        ord_tel = None
        dwell_tel = None
        ladder_ran = False
        lf_contam = None
        if lf_applied_prev and lf_m_prev is not None:
            # Reachability for the Batch-2 sweep: the amplitude mask always runs
            # when the branch does, but the rho term is a SECOND, optional operand
            # (`lf_rho_prev is not None`), so it needs its own counter -- otherwise
            # `_LF_ACT_RHO` could read "insensitive" because it was never read.
            _cm_amp = (lf_m_prev.abs() > (_LF_ACT_M * lf_amp_prev)).unsqueeze(1)
            _cm = _cm_amp
            _ab_hit("lf.contam_amp")
            _cm_rho = None
            if lf_rho_prev is not None:
                _cm_rho = (lf_rho_prev > _LF_ACT_RHO)
                _cm = _cm | _cm_rho
                _ab_hit("lf.contam_rho")
            lf_contam = _cm.to(torch.bool)
            # THE PER-PIXEL COVERAGE, WHICH THE CONSTANT AUDIT COULD NOT GET FROM THE LOGS.
            # `CONSTANTS_AUDIT` Â§11 left two rows unresolved for exactly one reason: the step record
            # carried `m_abs` (a MEAN) and `m_min`/`m_max` (extremes), and a mean cannot say where a
            # per-PIXEL threshold sits in the distribution it thresholds.  `_LF_ACT_M`'s placement was
            # therefore "correct in kind, exact placement needs per-pixel data", and `_LF_ACT_RHO`'s
            # inertness could only be inferred from `rho_max` (a per-step MAXIMUM, which is an upper
            # bound on the operand's reach rather than its effect).
            #
            # WHAT IS STAMPED, and why these and not quantiles: the mask IS a CDF test, so the honest
            # per-pixel record is the CDF itself at a few points -- `|m| / amp` against the shipped
            # threshold and against round numbers on either side.  Cheap (a compare + a mean per point,
            # no sort, no extra kernel class) and it places the constant exactly.  `rho_only` is the
            # operand's OWN contribution: pixels the rho term adds that the amplitude term did not
            # already flag, which is the number that says whether `_LF_ACT_RHO` does anything.
            try:
                _mr = lf_m_prev.abs() / lf_amp_prev.abs().clamp_min(1e-12)
                lf_diag["contam"] = {
                    "amp": round(float(_cm_amp.float().mean()), 4),
                    "rho": (round(float(_cm_rho.float().mean()), 4)
                            if _cm_rho is not None else None),
                    "rho_only": (round(float((_cm_rho & ~_cm_amp).float().mean()), 5)
                                 if _cm_rho is not None else None),
                    "cdf": [round(float((_mr > _t).float().mean()), 4)
                            for _t in (0.1, float(_LF_ACT_M), 0.5, 1.0)],
                    "cdf_at": [0.1, float(_LF_ACT_M), 0.5, 1.0],
                }
            except Exception:
                pass
        lf_applied_prev = False
        if (bool(cfg.get("order_pixel_adaptive", True))
                and eff_cap >= 2 and s2 > 1e-8 and len(hist) >= 1):
            bcos = None
            if len(hist) >= 2:
                _bd1 = (x0.float() - hist[-1][2].float())
                _bd2 = (hist[-1][2].float() - hist[-2][2].float())
                _bnum = (_bd1 * _bd2).sum(dim=1, keepdim=True)
                _bden = (_bd1.pow(2).sum(dim=1, keepdim=True).sqrt()
                         * _bd2.pow(2).sum(dim=1, keepdim=True).sqrt())
                bcos = torch.where(_bden > 1e-8,
                                   _bnum / _bden.clamp_min(1e-8),
                                   torch.zeros_like(_bnum))
            u_w = u_log if use_space == "log_sigma" else u_sig
            u2_w = u2_log if use_space == "log_sigma" else s2
            u_mid_w = 0.5 * (u_w + u2_w)
            pts_w = [((h[0] if use_space == "log_sigma" else h[1]), h[2])
                     for h in hist] + [(u_w, x0)]
            _lim2 = max(wmax_limit, wmax_mid_floor)
            cands = [x0]
            allowed = [True]
            wlevs = [0.0]
            winfo = [None]
            for ce in range(2, eff_cap + 1):
                c, wv = _mid_x0(pts_w, u_mid_w, ce)
                cands.append(c)
                wlevs.append(wv)
                _ok_ce, _gi_ce = _order_gate(pts_w, u_mid_w, ce, wv,
                                             (_lim2 if ce == 2 else wmax_limit), cfg,
                                             space=use_space)
                allowed.append(_ok_ce)
                winfo.append(_gi_ce)
            _max_ord = eff_cap
            while _max_ord > 1 and not allowed[_max_ord - 1]:
                _max_ord -= 1
            if _max_ord >= 2:
                _wgate_info = winfo[_max_ord - 1]
            wmax = max(wlevs[1:_max_ord]) if _max_ord >= 2 else 0.0
            P = min(len(hist), order_cap)
            I_list = _pixel_innovations(pts_w[:-1], u_w, x0, P)
            if I_list and I_list[0] is None and jmap is not None:
                I_list[0] = jmap.float().unsqueeze(1)
            _shape = next((e.shape for e in I_list if e is not None),
                          (x0.shape[0], 1, *x0.shape[2:]))
            dev = x0.device
            _fill = torch.full(_shape, _ORD_BIG, device=dev,
                               dtype=torch.float32)
            Im = torch.stack([_fill if e is None else e.float()
                              for e in I_list], 0)
            reset_m = None
            j = jmap.float().unsqueeze(1) if jmap is not None else None
            # REACHABILITY FOR THE ORDER SYSTEM'S OWN CONSTANTS.  `_ORD_Z` and `_ORD_R_FLOOR`
            # are read ONLY inside the `j is not None` + `ladder_auto` branch below, and the EMA weight
            # (`_ord_ema`, a per-run value from cfg since 2026-10-04 -- NOT the old `_ORD_EMA` constant,
            # which is gone) only in the innovation-EMA update: a sensitivity rig that sweeps them without
            # proving this branch ran is varying code that may never execute (the exact
            # vacuity that made the first Batch-0 sweep meaningless).  Counted, so a rig can
            # assert non-vacuity instead of assuming it.
            if j is not None:
                _ab_hit("order.jmap_present")
            if j is not None:
                if j_ema is None:
                    j_ema = j
                else:
                    if ladder_auto:
                        _r = (j / j_ema.clamp_min(1e-12)).clamp(1e-6, 1e6)
                        _lr = _r.log()
                        _lb = _lr.shape[0]
                        _flat = _lr.reshape(_lb, -1)
                        _med = _flat.median(dim=1).values.unsqueeze(1)
                        _mad = (_flat - _med).abs().median(dim=1).values
                        _mz = (0.6745 * (_flat - _med)
                               / _mad.clamp_min(1e-3).unsqueeze(1))
                        reset_m = ((_mz > _ORD_Z).reshape(_lr.shape)
                                   & (_r > _ORD_R_FLOOR))
                        _ab_hit("order.novelty_gate")
                    else:
                        reset_m = (j > (float(cfg.get("order_novelty_k", 4.0))
                                        * j_ema))
                    _rc = (0.0 if ladder_auto
                           else float(cfg.get("order_reversal_cos", 0.0)))
                    _rm = float(cfg.get("order_reversal_mag", 1.0))
                    if bcos is not None:
                        reset_m = reset_m | ((bcos < _rc)
                                             & (j > (_rm * j_ema)))
            if reset_m is None:
                reset_m = torch.zeros(_shape, dtype=torch.bool, device=dev)
            if (amask is not None
                    and bool(cfg.get("order_reset_on_anomaly", True))):
                reset_m = reset_m | amask.to(dev).bool().unsqueeze(1)
            o = ord_px
            if o is None:
                o = torch.full(_shape, 2.0, device=dev, dtype=torch.float32)
            o = o.clamp(1.0, float(max(P, 2)))
            if P >= 2:
                idxT = torch.arange(P, device=dev, dtype=o.dtype).view(
                    P, *([1] * o.dim()))
                if ladder_auto:
                    meas = Im < _ORD_BIG
                    if Iema is None:
                        Iema = Im.clone()
                    else:
                        if Iema.shape[0] < P:
                            _fills = torch.full(
                                (P - Iema.shape[0],) + _shape, _ORD_BIG,
                                device=dev, dtype=torch.float32)
                            Iema = torch.cat([Iema, _fills], 0)
                        Iema = Iema[:P]
                    Iema = torch.where(
                        meas & (Iema < _ORD_BIG),
                        _ord_ema * Iema + (1.0 - _ord_ema) * Im,
                        torch.where(meas, Im, Iema))
                    _ab_hit("order.ema_update")
                    _bias = (1e-4 * idxT)
                    p_star = torch.where(
                        (idxT < float(_max_ord)) & meas,
                        Iema + _bias,
                        torch.full_like(Iema, _ORD_BIG)).argmin(0).float() + 1.0
                    move = (p_star - o).clamp(-1.0, 1.0)
                    if lf_contam is not None:
                        move = torch.where(lf_contam & (move > 0.0),
                                           torch.zeros_like(move), move)
                    o_new = torch.where(
                        reset_m, torch.ones_like(o),
                        (o + move).clamp(1.0, float(max(P, 2))))
                    # THE RUNGE CRITERION (`order_by_bound`), when it is the arm in force:
                    # replace "the order that best predicts x0" with "the HIGHEST order whose
                    # remainder is still inside tolerance at THIS pixel's smoothness".  It
                    # composes with the safety path -- `allowed` is passed in, so an order this
                    # step refuses for everyone is refused here too -- and a pixel the ladder
                    # reset stays at 1, because a reset is a response to novelty/anomaly.
                    if _order_by_bound:
                        # THE STEP FACTOR, FROM THE SIGMAS THEMSELVES (U06/2372): a gap in
                        # the ladder's own space is not a log gap on the sigma branch.
                        _prop_bound = 1.0 - min(1.0, s2 / max(s, 1e-8))
                        _ob, _obi = _pixel_order_bound(
                            pts_w, u_mid_w, _prop_bound, allowed, _order_tol)
                        if _ob is not None:
                            o_new = torch.where(
                                reset_m, torch.ones_like(o_new),
                                _order_bound_target(o_new, _ob, _order_blend))
                            _ord_bound = _obi
                    try:
                        _ord_want = float(o_new.mean())
                        _ord_ps = float(p_star.mean())
                        _ord_innov = [round(float(Iema[q].mean()), 8)
                                      for q in range(min(int(P), 5))]
                    except Exception:
                        _ord_want = _ord_ps = _ord_innov = None
                else:
                    I_o = (Im * (idxT == (o - 1.0).unsqueeze(0))).sum(0)
                    I_lo = (Im * (idxT == (o - 2.0).clamp_min(0.0)
                                  .unsqueeze(0))).sum(0)
                    I_hi = (Im * (idxT == o.unsqueeze(0))).sum(0)
                    demote_m = I_o > ((1.0 + float(cfg.get("order_demote_gain",
                                                           0.25))) * I_lo)
                    elev_m = ((o < float(P))
                              & (I_hi < ((1.0 - float(cfg.get("order_elev_gain",
                                                              0.10))) * I_o)))
                    o_new = torch.where(reset_m, torch.ones_like(o),
                                        torch.where(demote_m,
                                                    (o - 1.0).clamp_min(1.0),
                                                    torch.where(elev_m,
                                                                o + 1.0, o)))
            else:
                o_new = torch.where(reset_m, torch.ones_like(o), o)
            # THE FORCED ORDER FLOOR.  Applied AFTER the ladder's own choice and in BOTH
            # branches, before the allowed-ladder maps anything down -- so it is a demand
            # that the safety path can still refuse, not an override.  A pixel `reset_m`
            # snapped to 1 stays at 1.
            # ONE DEMAND PATH FOR BOTH MODES.  `1..6` demand their own value on every step, exactly as before;
            # `0` (AUTOMATIC) demands the shipped order only while the run's last-measured `err/tol` is above
            # the threshold.  Applied in BOTH branches and before the allowed-ladder maps anything down, so it
            # stays a DEMAND the safety path can refuse -- the property `notes/038` established, inherited.
            _fl_val = _order_floor
            if _order_floor_auto:
                # WHICH REQUEST THIS STEP MAKES: one owner (`_demand_policy`), called identically by the
                # per-step stamp below.  The stamp used to test the predicate inline, which is how the
                # bound arm was named on runs that executed the graded one (`cond_rtol_loosen`).
                _demand_arm = _demand_policy(_ord_bound is not None, _order_demand_q,
                                             _order_by_bound, _order_demand_bound, _of_graded)
                if _demand_arm == "bound_map":
                    # THE BOUND IS THE REQUEST.  With `order_by_bound` in force `o_new` IS the per-pixel
                    # validity bound, so a scalar demand can only push an order the run's own remainder
                    # criterion has already refused AT THAT PIXEL.  Asking for nothing is therefore the
                    # derivation, not the absence of one -- the map is the per-step request, and it is
                    # already applied above.
                    _fl_val = 1.0
                    _ab_hit("order.demand_bound_map.applied", 1.0)
                elif _demand_arm == "bound":
                    # THE BOUND-DERIVED QUANTILE REQUEST.  With `order_by_bound` in force `o_new` IS the
                    # per-pixel bound, so this asks for the q-quantile of what the engine itself considers
                    # valid for THIS step -- the same question the fixed demand asks, at the step's own
                    # scale, and needing no new quantity and no extra pass.
                    try:
                        _ok = o_new[~reset_m] if reset_m is not None else o_new
                        _ok = _ok.reshape(-1)
                        _fl_val = (float(torch.quantile(_ok.float(), _order_demand_q))
                                   if _ok.numel() else _order_floor_demand)
                        _ab_hit("order.demand_bound.applied", round(float(_fl_val), 4))
                    except Exception:
                        _fl_val = _order_floor_demand
                else:
                    _gate_open = bool(_prev_rtol is not None
                                      and _prev_rtol > _order_floor_rtol)
                    # ONE OWNER (`_graded_demand`), called identically by the stamp below.
                    _fl_val = _graded_demand(_prev_rtol, _order_floor_rtol, _of_lo, _of_hi,
                                             _of_r_hi, _order_floor_demand, _of_graded)
                    # REACHABILITY (rule 9): the site counts its own invocations, so a batch can prove the
                    # gate RAN rather than infer it from an effect.
                    _ab_hit("order.floor_adaptive.open" if _gate_open
                            else "order.floor_adaptive.shut",
                            None if _prev_rtol is None else round(float(_prev_rtol), 6))
            if _fl_val > 1.0:
                try:
                    _fl = torch.full_like(o_new, min(float(_fl_val),
                                                     float(max(P, 2))))
                    _ord_floor_n = int(((o_new < _fl) & ~reset_m).sum())
                    o_new = torch.where(reset_m, torch.ones_like(o_new),
                                        torch.maximum(o_new, _fl))
                except Exception:
                    _ord_floor_n = None
            if _hist_cos_px_on and hist_cos_px_neg is not None:
                _neg = hist_cos_px_neg
                if _neg.shape != o_new.shape:
                    _neg = _neg.expand_as(o_new)
                _n_gated = int(_neg.sum())
                o_new = torch.where(_neg, torch.ones_like(o_new), o_new)
                _ab_hit("histcos.px.order_gated", _n_gated)
            ord_px = o_new
            ladder_ran = True
            arm = []
            _run = 1
            for ce in range(1, eff_cap + 1):
                if allowed[ce - 1]:
                    _run = ce
                arm.append(_run)
            arm_t = torch.tensor(arm, device=dev, dtype=torch.long)
            o_l = o_new.clamp(1.0, float(eff_cap)).long()
            o_map = arm_t[(o_l - 1)].float()
            # ---- THE SPATIAL INSTRUMENT: the order map's own ROUGHNESS -------------------
            # WHY THIS EXISTS.  The visible artifact is BLOTCHY, and `wmax_limit` was measured to act
            # on the PER-PIXEL order admission (63.62% against 62.68% of pixels admitted at one step),
            # which is spatially structured and therefore invisible to every global metric this
            # project had.  `ord_mean` is a mean; it cannot see a spatial pattern.  This logs the
            # pattern's ROUGHNESS, which is the quantity a spatial fix has to reduce.
            #
            # BOTH MAPS ARE LOGGED, and that is the point: `o_new` is what the CRITERION REQUESTED and
            # `o_map` is what the safety ladder ADMITTED.  Comparing their roughness separates "the
            # criterion asks for a rough map" from "the threshold admission makes it rough" -- two
            # different defects with two different fixes, and this is the measurement that tells them
            # apart rather than assuming.
            #
            #   tv        mean |order difference| over 4-neighbour pairs (discrete total variation):
            #             a smooth map scores near 0, a salt-and-pepper map scores near 1
            #   bnd       fraction of 4-neighbour pairs whose orders DIFFER
            #   tv_want   the same variation for the REQUESTED map
            #   ord_tiles 16x16 map of the mean admitted order -- the spatial pattern, so a blotch
            #             can be LOCATED rather than only counted
            #   rough_tiles 16x16 map of the per-pixel roughness -- WHERE the map is incoherent
            #   img_tiles 16x16 map of the image-side high-pass RMS -- the CONTENT axis, so the
            #             order map can be read against image detail instead of only its own TV
            #   jump_tiles 16x16 map of the per-step x0 jump RMS -- the TEMPORAL axis: high on
            #             volatile tiles, low on stale ones.  With img_tiles this separates
            #             "stale but detailed" (high img, low jump) from "volatile" (high jump),
            #             which is the distinction the per-pixel order is supposed to make.
            #
            # PAYLOAD: 2 scalars + 4 x 256 tiles = 1026 numbers per step, about 64 KB over a 16-step
            # run, so a log roughly doubles.  Bounded on purpose: a FULL 208x144 map per step would
            # be ~1 MB per run and unreadable in the corpus.
            _ord_spat = None
            if len(_LAST_ERRORS) < 64 or log_errors:
                try:
                    with torch.no_grad():
                        def _diffs(m):
                            d = m.float()
                            return ((d[..., 1:, :] - d[..., :-1, :]).abs(),
                                    (d[..., :, 1:] - d[..., :, :-1]).abs())

                        def _tiles(m, g=16):
                            d = m.float()
                            d = d.reshape(-1, d.shape[-2], d.shape[-1]).mean(0)
                            d = F.adaptive_avg_pool2d(d.unsqueeze(0).unsqueeze(0), g)[0, 0]
                            # ONE device sync, not 256.  `float(v)` on a CUDA
                            # scalar synchronises, so the per-element form cost
                            # 256 syncs per call and 512 per step (this function
                            # runs twice).  MEASURED on the operator's RTX 3090 by
                            # scratch_sync_cost_rig.py: `_tiles` 10.787 ms against
                            # 0.846 ms for this form (12.7x), and the whole
                            # `ord_spat` block 24.176 ms/step against 1.701 ms
                            # (14.2x) -- worth 0.36 s of a 16-step run and 0.90 s
                            # of a 40-step one.  `.tolist()` moves the tensor once;
                            # the VALUES are identical, which that rig asserts
                            # element-for-element rather than assuming.
                            return [round(v, 3) for v in d.flatten().tolist()]

                        _adh, _adw = _diffs(o_map)
                        _wth, _wtw = _diffs(o_new)
                        _r = torch.zeros_like(o_map)
                        _r[..., :-1, :] = _r[..., :-1, :] + _adh
                        _r[..., 1:, :] = _r[..., 1:, :] + _adh
                        _r[..., :, :-1] = _r[..., :, :-1] + _adw
                        _r[..., :, 1:] = _r[..., :, 1:] + _adw
                        _jump_tiles = None
                        if hist:
                            _jump_tiles = _tiles(
                                (x0.float() - hist[-1][2].float()).pow(2).mean(
                                    dim=1, keepdim=True).sqrt())
                        _ord_spat = {
                            "tv": round(float(0.5 * (_adh.mean() + _adw.mean())), 4),
                            "bnd": round(float(0.5 * ((_adh > 0.5).float().mean()
                                                      + (_adw > 0.5).float().mean())), 4),
                            "tv_want": round(float(0.5 * (_wth.mean() + _wtw.mean())), 4),
                            "ord_tiles": _tiles(o_map),
                            "rough_tiles": _tiles(_r),
                            # IMAGE-SIDE ACTIVITY at the same 16x16 grid, so the order
                            # map can be read against IMAGE content instead of only its
                            # own total variation.  High-pass RMS per tile; cheap, no
                            # model call, and only written on ladder-active steps.
                            # The latent can be 4-D or 5-D (the model's own
                            # `latent_dimensions`); `avg_pool2d` only takes 3/4-D, so a
                            # 5-D (B,C,T,H,W) is folded to (B*T,C,H,W) first.
                            "img_tiles": _tiles(
                                (lambda _im: (
                                    _im - F.avg_pool2d(
                                        F.pad(_im, (1, 1, 1, 1),
                                              mode="replicate"),
                                        3, stride=1)).pow(2).mean(
                                            dim=1, keepdim=True).sqrt())(
                                    (x0.float().permute(0, 2, 1, 3, 4).reshape(
                                        -1, x0.shape[1], x0.shape[-2], x0.shape[-1])
                                     if x0.dim() == 5
                                     else x0.float().reshape(
                                         -1, x0.shape[-3], x0.shape[-2], x0.shape[-1])))),
                            # TEMPORAL activity: per-step x0 jump RMS per tile.  High on
                            # volatile tiles, low on stale ones; paired with `img_tiles` it
                            # separates stale-but-detailed from volatile.
                            "jump_tiles": _jump_tiles,
                        }
                        _ab_hit("order.spatial_logged")
                except Exception:
                    _ord_spat = None
            C = torch.stack([c.float() for c in cands], 0)
            idxC = torch.arange(eff_cap, device=dev,
                                dtype=torch.long).view(
                eff_cap, *([1] * o_map.dim()))
            _sel_hi = (idxC == (o_map.long() - 1).unsqueeze(0)).to(C.dtype)
            _sel_lo = (idxC == (o_map.long() - 2).clamp_min(0)
                       .unsqueeze(0)).to(C.dtype)
            A_hi = (C * _sel_hi).sum(0)
            A_lo = (C * _sel_lo).sum(0)
            # NAMING, because the key reads as something it is not: `order_pixel_blend`
            # blends two EXTRAPOLANTS -- A_hi (the pixel's own admitted order o) and A_lo
            # (one order below it) -- weighted per pixel by the cosine of the last two x0
            # innovations.  It does NOT blend the orders themselves: `o_map` is already a
            # single integer order per pixel, chosen upstream.  A convex combination of two
            # polynomials of different degree is not itself an extrapolant of either degree,
            # which is exactly the point -- where the trajectory has just turned (bcos < 0)
            # the lower order is trusted more.  The key name is kept for saved graphs; read
            # it as "blend the two candidate extrapolants".
            _sharp = float(cfg.get("order_pixel_blend_sharpness",
                                   ENGINE_DEFAULTS.get("order_pixel_blend_sharpness", 1.0)))
            if bcos is not None:
                _w = torch.sigmoid(bcos * _sharp)
            else:
                _w = torch.ones_like(A_hi[:, :1])
            x0m = _w * A_hi + (1.0 - _w) * A_lo
            eff = 2 if _max_ord >= 2 else 1
            if len(_LAST_ERRORS) < 64 or log_errors:
                blend_w_mean = float(_w.mean())
                ord_tel = (float(o_map.mean()), int(o_map.max()),
                           float(reset_m.float().mean()))
            if ladder_auto and P >= 2:
                idxY = torch.arange(P, device=dev, dtype=torch.long).view(
                    P, *([1] * o_map.dim()))
                I1m = Im[0]
                I_ap = (Im * (idxY == (o_map.long() - 1)
                              .unsqueeze(0))).sum(0)
                _yv = (1.0 - I_ap / (I1m + 1e-6)).clamp(0.0, 1.0)
                _yv = torch.where((I1m < _ORD_BIG) & (I_ap < _ORD_BIG),
                                  _yv, torch.zeros_like(_yv))
                lf_suppress = (_yv if lf_suppress is None
                               else _ord_ema * lf_suppress
                               + (1.0 - _ord_ema) * _yv)
            if j is not None:
                j_ema = torch.maximum(
                    j, (_ORD_ENV_DECAY if ladder_auto
                        else float(cfg.get("order_jema_decay", 0.7)))
                    * j_ema)
                _ab_hit("order.env_decay")
        elif (bool(cfg.get("order_pixel_blend", True))
                and eff >= 3 and s2 > 1e-8 and len(hist) >= 2):
            if space_locked is None:
                if use_space == "log_sigma":
                    _bpts = [(h[0], h[2]) for h in hist] + [(u_log, x0)]
                    _bmid = 0.5 * (u_log + u2_log)
                else:
                    _bpts = [(h[1], h[2]) for h in hist] + [(u_sig, x0)]
                    _bmid = 0.5 * (s + u2_sig)
            else:
                _bu_i = u_log if use_space == "log_sigma" else u_sig
                _bu_n = u2_log if use_space == "log_sigma" else s2
                _bpts = [((h[0] if use_space == "log_sigma" else h[1]), h[2])
                         for h in hist] + [(_bu_i, x0)]
                _bmid = 0.5 * (_bu_i + _bu_n)
            _c_o2 = _mid_x0(_bpts, _bmid, 2)
            _ok_o2, _gi_o2 = _order_gate(_bpts, _bmid, 2, _c_o2[1],
                                         max(wmax_limit, wmax_mid_floor), cfg,
                                         space=use_space)
            if _ok_o2:
                _wgate_info = _gi_o2
                _x0m_o2 = _c_o2[0]
                _bd1 = (x0.float() - hist[-1][2].float())
                _bd2 = (hist[-1][2].float() - hist[-2][2].float())
                _bnum = (_bd1 * _bd2).sum(dim=1, keepdim=True)
                _bden = (_bd1.pow(2).sum(dim=1, keepdim=True).sqrt()
                         * _bd2.pow(2).sum(dim=1, keepdim=True).sqrt())
                _bcos = torch.where(_bden > 1e-8,
                                    _bnum / _bden.clamp_min(1e-8),
                                    torch.zeros_like(_bnum))
                _sharp = float(cfg.get("order_pixel_blend_sharpness",
                                   ENGINE_DEFAULTS.get("order_pixel_blend_sharpness", 1.0)))
                _w = torch.sigmoid(_bcos * _sharp)
                x0m = _w * x0m + (1.0 - _w) * _x0m_o2
                blend_w_mean = float(_w.mean())
        if guard_active and adaptive_on and ord_px is not None:
            ord_px = torch.ones_like(ord_px)
        if not ladder_ran and lf_suppress is not None:
            lf_suppress = _ord_ema * lf_suppress
        if eff >= 2:
            # A/B #2: the envelope multiplier.  Shipped = the fixed 1.3; derived =
            # `sum|w_i|`, the extrapolation's own exact amplification bound.
            _ab_hit("env.eff_ge_2")
            if derived_ab:
                _ab_hit("env.derived_arm")
                try:
                    _us = [h[0] for h in hist[-(eff - 1):]] + [u_log]
                    _u2e = math.log(max(s2, 1e-8))
                    _we = _lagrange_weights(_us, 0.5 * (u_log + _u2e))
                    _x0m_env = max(sum(abs(w) for w in _we), 1.0)
                except Exception:
                    _x0m_env = _AFLOPS_X0M_ENV
                _ab_hit("env.derived_value", round(float(_x0m_env), 4))
            else:
                _x0m_env = (_AFLOPS_X0M_ENV
                            if cfg.get("x0m_env") is None
                            else float(cfg["x0m_env"]))
            _env_list = [h[2] for h in hist[-(eff + 1):]] + [x0]
            _max_norm = torch.stack(
                [t.float().norm() for t in _env_list]).max()
            _env = _x0m_env * _max_norm
            _over = x0m.float().norm() > _env
            _n_over = int(_over.sum())
            _ab_hit("env.guard_evaluated")
            # The guard's own decision variable, normalised so it is comparable
            # across the two arms: the guard fires exactly when this ratio
            # exceeds the envelope multiplier in use (1.3 shipped, sum|w| derived).
            try:
                _ab_hit_max("env.ratio_vs_maxnorm",
                            round(float(x0m.float().norm()
                                        / _max_norm.clamp_min(1e-12)), 4))
            except Exception:
                pass
            if _n_over > 0:
                _ab_hit("env.guard_fired_pixels", _n_over)
            x0m = torch.where(_over, x0.float(), x0m.float())
        x_pred = exact_step(x, x0m, s, s2) if eff >= 2 else x1
        err = _err_norm(x_pred, x1, err_mode, x0_ref=x0,
                        strength=grad_strength) if (eff >= 2 and s2 > 1e-8) else None
        emap = None
        if lf_on and err is not None:
            emap = (x_pred - x1).float().pow(2).mean(dim=1).sqrt()
            if lf_closed_loop:
                lf_trust, lf_emap_prev, _eff, _nst = _lf_trust_update(
                    emap, lf_emap_prev, lf_m_prev, lf_amp_prev, lf_trust)
                if _eff is not None and bool(cfg.get("lf_runtime_gate", True)):
                    _base = (torch.ones_like(_eff) if lf_gate is None
                             else lf_gate)
                    _cand = (_base + _GATE_ETA * (_eff - _GATE_TARGET))\
                        .clamp(_GATE_FLOOR, 1.0)
                    lf_gate = torch.where(_nst > 0, _cand, _base)
        corr_used = None
        # THE PRE-CUT DEVIATION, when the Euler-safety cut fires (U09/3933).  After the
        # cut `x` IS `x1`, so a deviation computed later is exactly 0; this keeps the
        # per-pixel error the cut actually acted on.  None on a step without a cut.
        _emap_pre_cut = None
        # T2c diagnostics MUST be initialised BEFORE the corrector block below,
        # not after it: an earlier version of this instrumentation reset them in the
        # `cerr = None` block that FOLLOWS the corrector, which wiped the blend
        # weight before it could be logged and made `corr_w` read None on every
        # step -- a value that looked like evidence and was an artefact.
        _corr_w_used = None
        _e_pred_used = None
        _e_raw_used = None
        aflops_lam = None
        aflops_lam_raw = None
        _lam_lo = None
        _lam_src = None
        _so_mask = None
        _so_raw = False
        if s2 > 1e-8:
            # The residual integrator consumes the order-ladder midpoint (x0m),
            # not the raw x0 -- this is what feeds the per-pixel adaptive order
            # into the actual step.  When the ladder is gated (eff_cap == 1) x0m
            # is x0, so the plain 1st-order path is unchanged.
            # probe-driven per-step lower bound when the probe had usable
            # evidence, else the fixed cfg bound (incl. the TESTING toggle)
            _lam_lo = cfg.get("aflops_lam_min")
            if _lam_min_steps is not None and i < len(_lam_min_steps):
                _lam_lo = _lam_min_steps[i]
            _lam_src = "fixed"
            if _lam_min_steps is not None:
                _lam_src = "probe"
            # RUN-driven bound wins over both: it uses the step's OWN measured
            # x0 jump (already computed above, so it is causal), which is the
            # real quantity the probe only approximates.
            # A/B: the archive-known-good corrector (corrector_ab False) did not
            # have this run-driven per-step bound at all.
            if (corrector_ab and len(hist) >= 1
                    and bool(cfg.get("lam_clamp_from_run", True))):
                try:
                    _rj = float((x0.float()
                                 - hist[-1][2].float()).pow(2).mean().sqrt())
                    if _rj > _run_jump_max:
                        _run_jump_max = _rj
                    _rb = _run_lam_bound(_rj, _run_jump_max)
                    if _rb is not None:
                        _lam_lo = _rb
                        _lam_src = "run"
                except Exception:
                    pass
            # A/B #1: what the integrator is FED.  The archive-known-good
            # corrector used the raw x0 (`_aflops_step(x, x0, ...)`, archive
            # line 6861) and used the extrapolated x0m only for the predictor
            # x_pred.  Measured on non-degenerate ground truth
            # (scratch_corrector_budget_rig.py section E) feeding x0 beats
            # feeding x0m at 4/4 extrapolation orders.
            _x0_fed = x0m if corrector_ab else x0
            _x0_fed_hist = _x0_fed.detach()
            _ab_hit("corr.x0_fed_x0m" if corrector_ab else "corr.x0_fed_raw")
            if corrector_ab and _lam_src == "run":
                _ab_hit("corr.run_lam_bound")
            _so_mask = None
            if bool(cfg.get("second_order_per_pixel", False)) and ord_px is not None:
                # Give the second-order slope term only to the pixels whose ADMITTED order
                # is >= 2.  The order map already protects volatile pixels; applying the
                # term globally overrides that protection (D/B: the term is net harmful in
                # the volatile half and is not the detail carrier).  None is the shipped
                # global term, byte-identical.
                _so_mask = (ord_px.float() >= 2.0).to(x.dtype)
                _ab_hit("secondorder.px.mask", float(_so_mask.mean()))
            _so_raw = bool(cfg.get("second_order_raw_lambda", False))
            if _so_raw:
                _ab_hit("secondorder.raw")
            aflops_out, aflops_lam, aflops_lam_raw = _aflops_step(
                x, _x0_fed, hist, s, s2, order=eff_cap,
                lam_min=_lam_lo,
                lam_max=cfg.get("aflops_lam_max"),
                series_switch=(_derived_series_switch(x.dtype)
                               if derived_ab else None),
                order_mask=_so_mask,
                raw_slope=_so_raw)
            if aflops_out is not None:
                # the integrator's deviation is measured against the *midpoint*
                # step (x_pred), which is the 1st-order version of the same
                # clean estimate the integrator uses.
                err = _err_norm(aflops_out, x_pred, err_mode, x0_ref=x0m,
                                strength=grad_strength)
                # The 2nd-order exponential integrator is only trusted while its
                # deviation from the midpoint step stays inside rtol.  At high
                # latent resolution (and low sigma) the velocity field is more
                # nonlinear, so the correction can overshoot -- which reads as
                # burned / glitchy patches.  Blend back toward the midpoint step
                # as the error grows past rtol (err == the already-computed step
                # error, so this is zero extra model calls).
                # A/B #3a: the archive-known-good corrector took the corrected
                # step UNCONDITIONALLY (archive line 6868: `x = aflops_out`).
                if (corrector_ab and err is not None and err > tol):
                    _ab_hit("corr.blend_fired", round(float(err), 8))
                    _w = max(tol / max(err, 1e-12), 0.0)
                    x = x_pred + _w * (aflops_out - x_pred)
                    # T2c INSTRUMENTATION.  The weight actually used here, under a
                    # name that cannot be confused with the PIXEL-blend weight that
                    # the log's `blend_w` field carries (:7903).  An earlier attempt
                    # to explain T2c tested `blend_w` against `tol/err` and concluded
                    # the blend was innocent -- but it was reading the pixel-blend
                    # field, so that refutation was itself invalid.  Rule 12: read
                    # the use site, then measure.
                    _corr_w_used = float(_w)
                else:
                    if corrector_ab:
                        _ab_hit("corr.blend_skipped")
                    x = aflops_out
                    _corr_w_used = None
                corr_used = "aflops"
        if corr_used is None:
            x = x_pred
            corr_used = "mid"
        cerr = None
        # The blend's OWN normalized distance from x_pred, in the same units as
        # `err`/`tol` (same x0_ref), computed at the cut site.  T2c: if the blend
        # fires it sets this to `tol` IDENTICALLY -- w = tol/err and the norm is
        # linear in the difference -- so `_e_raw ~ tol` reduces to a statement about
        # how far the plain step x1 is from the midpoint step x_pred.  Measuring it
        # here is what turns that from a guess into a number.
        # (NOT re-initialised here -- see the note at `corr_used = None` above.)
        # Euler-safety fallback: the order-ladder midpoint + exponential
        # integrator are OUR extra over what ComfyUI Euler does (the plain
        # exact step x1 = exact_step(x, x0, s, s2)).  When the corrected step
        # deviates from x1 by more than the tolerance, the correction is over
        # budget -- the measured CFG1 'not fully resolved' downgrade, tail
        # steps 12-15 at err/tol 1.2/1.78/2.73/5.23 with midpoint Lagrange
        # weight wmax 3.3-3.8.  Use the plain step there (never worse than
        # Euler), and record the deviation that triggered it.
        # A/B #3b: the archive-known-good corrector had NO Euler-safety cut, so
        # `corr` can never read "raw" with corrector_ab False.
        #
        # CORRECTNESS QUESTION BEING MEASURED (T2, scratch_tail_three_arm_rig.py):
        # this cut compares `_e_raw` -- the deviation of the corrected step from
        # the plain exact step -- against `tol`, an ABSOLUTE tolerance.  But the
        # size of a legitimate correction SCALES WITH du: at the tail du is
        # largest (0.581 against 0.105 mid-run, measured over 242 runs), so the
        # corrected step legitimately departs furthest from the plain step
        # exactly where the correction has the most work to do, and the cut
        # fires there (corr='raw' on 64% of runs at the second-to-last step).
        # If that reading is right the cut discards the BEST correction, and
        # comparing a du-scaled quantity to a fixed tolerance is a units error
        # rather than a tuning problem.  `euler_cut=False` is the arm that
        # measures it; the default preserves the shipped behaviour exactly.
        if corrector_ab and s2 > 1e-8 and corr_used in ("aflops", "mid"):
            if not euler_cut:
                _ab_hit("cut.disabled")
            else:
                _ab_hit("cut.armed")
                # WHAT THIS CUT COMPARES -- and the derived alternative (T4).
                #
                # The final step is C, the blend output.  Write P = x_pred (the
                # ladder's midpoint step) and E = x1 (the plain exact step).  The
                # blend has ALREADY bounded the corrector's excursion:
                #     ||C - P||_w = min(err, tol) * ||x0m||_w          (algebra: w = tol/err)
                # so the correction is inside tolerance of the midpoint by
                # construction.  The shipped cut then tests
                #     ||C - E||_w / ||x0||_w  >  tol
                # which by the triangle inequality is NOT the same quantity:
                #     ||C - E||  <=  ||C - P||  +  ||P - E||
                #                   ^ bounded    ^ UNBOUNDED: the ladder's own move,
                #                     (blend)      which nothing else tests
                # So the shipped variable mixes a term the blend owns with a term
                # nothing owns.  Measured consequence (T2c, NOTES.md): where the
                # ladder contributed nothing, P == E identically, so `||C - E||`
                # collapses onto the blend's own bound and the cut evaluates
                # `tol > tol` -- a float-rounding lottery (3 of 5 step counts fire).
                #
                # THE DERIVED FORM: test the term nothing else tests, and only that.
                #     keep the ladder's midpoint  iff  ||P - E||_w / ||x0||_w <= tol
                # i.e. fall back to the plain step exactly when the LADDER moved the
                # step further than tolerance allows.  Properties, both checkable:
                #   - where P == E (no ladder contribution) the test is `0 > tol`,
                #     so it can never fire and the lottery is gone;
                #   - where the ladder DID move the step it still fires, because the
                #     ladder's move is a component of the shipped variable too.
                # So the derived fire set must be a strict SUBSET of the shipped one,
                # excluding precisely the boundary collisions.
                #
                # Default remains "corrected" = the shipped behaviour, byte-identical;
                # "midpoint" is the derived arm, measured in
                # scratch_cut_variable_rig.py.
                if cut_var == "midpoint":
                    _e_raw = _err_norm(x_pred, x1, err_mode, x0_ref=x0,
                                       strength=grad_strength)
                else:
                    _e_raw = _err_norm(x, x1, err_mode, x0_ref=x0,
                                       strength=grad_strength)
                _e_raw_used = _e_raw
                # T2c: the same distance measured to x_pred instead of x1, in the
                # same units.  Since x_pred - x1 = (1-r)(x0m - x0), the gap between
                # these two numbers IS the quantity that decides whether `_e_raw`
                # lands on `tol` by construction.
                try:
                    _e_pred_used = _err_norm(x, x_pred, err_mode, x0_ref=x0m,
                                             strength=grad_strength)
                except Exception:
                    _e_pred_used = None
                if _e_raw is not None and _e_raw > tol:
                    _ab_hit("cut.fired", round(float(_e_raw), 8))
                    # CAPTURED BEFORE THE REPLACEMENT: this is the per-pixel error that
                    # triggered the cut, and the only honest reading of the step.
                    _emap_pre_cut = (x - x1).float().pow(2).mean(dim=1).sqrt()
                    x = x1
                    corr_used = "raw"
                    cerr = _e_raw
                # UNITS NOTE (T2b).  `_e_raw` is the distance between the corrected
                # step and the plain exact step.  `tol` is ONE number for the whole
                # run.  That distance scales as a power of the interval `du`
                # (MEASURED on the quantity this cut actually tests: `_e_raw` ~
                # du^2.158, r2 0.909, n=82, on a bounded synthetic flow), while `tol`
                # does not move -- so the criterion is scale-mismatched and must fire
                # hardest at the largest du, which is the tail.  Whether to rescale
                # `tol` by (du/du_ref)^p, or to compare a du-independent quantity
                # instead, is decided in T3/T4 on the strength of that exponent.
                # RETRACTED HERE: this comment previously quoted `err ~ du^2.34` and
                # `du^0.61`.  Both were computed on `err` (corrected-vs-MIDPOINT)
                # rather than on `_e_raw` (corrected-vs-PLAIN), which is a different
                # quantity, so both were void.  See scratch_cut_scaling_rig.py.
                elif corrector_ab:
                    _ab_hit("cut.not_fired")
        if s2 > 1e-8 and eff < 2:
            if err is None:
                cerr = _err_norm(x, x1, err_mode, x0_ref=x0,
                                 strength=grad_strength)
            if lf_on and emap is None and corr_used not in (None, "mid"):
                emap = _lf_emap_from_step(corr_used, _emap_pre_cut, x, x1)
                if lf_closed_loop:
                    lf_trust, lf_emap_prev, _eff, _nst = _lf_trust_update(
                        emap, lf_emap_prev, lf_m_prev, lf_amp_prev,
                        lf_trust)
                    if (_eff is not None
                            and bool(cfg.get("lf_runtime_gate", True))):
                        _base = (torch.ones_like(_eff) if lf_gate is None
                                 else lf_gate)
                        _cand = (_base + _GATE_ETA * (_eff - _GATE_TARGET))\
                            .clamp(_GATE_FLOOR, 1.0)
                        lf_gate = torch.where(_nst > 0, _cand, _base)
        if lf_on and lf_m is not None and s2 > 1e-8:
            x = x0 + (1.0 + lf_m) * (s2 / max(s, 1e-8)) * (x_input - x0) + (x - x1)
            _surgical = False
            # 0.0, not None: every branch that sets `_surgical` True also sets this (the
            # no-baseline branch clears `_surgical` instead of defaulting), so a sentinel
            # would only invite a default to be re-introduced.
            _rho_band = 0.0
            # The band-surgical (checkerboard) refresh targets late-stage
            # banding/checkerboard artifacts on cfg>1 flow models.  On a
            # distilled model it fires spuriously mid-run: the trajectory is
            # flat (no band_curve baseline of its own), so hi_ref is either
            # absent or borrowed from a foreign profile, and the detector
            # injects refresh noise (rho up to 0.0044, steps 9-13) that a
            # converged model cannot integrate away.  Gate it off for
            # distilled runs (consistent with the rest of the distilled LF
            # treatment: reduced drift budget, no semantic prior, eta tail
            # cut).  Audited from the CFG1 distilled log: band_ref=true with
            # zero band_curve in any dumped profile for this model.
            # THE BASELINE'S SPATIAL SCALE MUST MATCH THE RUN'S (U08).  Both sides
            # measure the hi band with the same 5-tap blur, so a 48px probe and a
            # 128px run are not measuring the same scales: the same content reads
            # hi 0.0497 at 128 and 0.0166 at 48 (3.0x; white noise 8.7x), against a
            # threshold of 1.25*ref + 0.03 -- so the detector is either always blind
            # or always firing depending on which way the resolution runs.  A probe
            # cannot see the run's finest band at all, so the baseline is used only
            # when the probe measured AT the run's latent size; otherwise the
            # detector declines and counts it.  `hi_ref` itself stays in the log, so
            # the mismatch is still visible.
            _band_res = None
            for _p in (lf_prof_m, lf_prof_c):
                if isinstance(_p, dict) and _p.get("n_probe_res") is not None:
                    try:
                        _band_res = int(_p["n_probe_res"])
                    except (TypeError, ValueError):
                        _band_res = None
                    break
            _band_scaled = (jmap is not None and _band_res is not None
                            and _band_res == min(int(jmap.shape[-2]),
                                                 int(jmap.shape[-1])))
            if (not dist_eff
                    and bool(cfg.get("lf_band_refresh", True))
                    and lf_gains.get("hi_ref") and jmap is not None
                    and i < len(lf_gains["hi_ref"])
                    and _band_scaled):
                try:
                    _hi = _lf_band_fractions(jmap)[0]
                    _ref = float(lf_gains["hi_ref"][i])
                    # The detection threshold, kept as a value so the ramp's zero point can
                    # BE it instead of approximating it (see _BAND_DET_A / _BAND_DET_B).
                    _det = _BAND_DET_A * _ref + _BAND_DET_B
                    _surgical = _hi > _det
                    if _surgical:
                        _ab_hit("lf.band_surgical")
                        if _ref > 1e-9:
                            # refresh magnitude scales with the measured excess of the
                            # hi-band over the model's own probe-measured baseline: exactly
                            # 0 at the detection threshold, full at `_BAND_RAMP_SPAN` times
                            # that excess.  BOTH ends are derived from `_det`, so there is
                            # one source of truth for where the refresh starts, and the span
                            # cannot invert however small the baseline gets.
                            _excess = (_hi - _ref) / _ref
                            _e_det = (_det - _ref) / _ref
                            _ramp = min(max((_excess - _e_det)
                                            / max(_e_det * (_BAND_RAMP_SPAN - 1.0), 1e-9),
                                            0.0), 1.0)
                            # Reachability for the Batch-2 sweep: record the ramp itself --
                            # pinned at 0 (the detection is only just satisfied) or at 1
                            # (saturated) are two DIFFERENT reasons for an "insensitive"
                            # verdict, and neither is dead code.
                            _ab_hit("lf.band_formula", round(_ramp, 6))
                            if _ramp <= 0.0:
                                _ab_hit("lf.band_ramp_zero", round(_excess, 6))
                            elif _ramp >= 1.0:
                                _ab_hit("lf.band_ramp_sat", round(_excess, 6))
                            _rho_band = _BAND_RHO_MAX * _ramp
                        else:
                            # No measured hi-band baseline.  The excess is a RATIO to `_ref`,
                            # so with `_ref ~ 0` it is undefined and the detection cannot be
                            # quantified -- decline rather than substitute a number.  The old
                            # code used the hardcoded `_LF_BAND_RHO = 0.05` here, a branch
                            # that never fired in 8444 corpus step records.
                            _surgical = False
                            _rho_band = 0.0
                            _ab_hit("lf.band_noref")
                except Exception:
                    _surgical = False
            elif (not dist_eff
                  and bool(cfg.get("lf_band_refresh", True))
                  and lf_gains.get("hi_ref") and jmap is not None
                  and i < len(lf_gains["hi_ref"])):
                # Eligible, but the probe baseline was measured at another spatial
                # scale (or carries no resolution stamp), so it is not a prediction
                # of this run's hi band.  Counted, so "the detector never fires" can
                # be attributed to the scale rather than to the model.
                _ab_hit("lf.band_refused_scale")
            if dist_eff and lf_diag is not None:
                lf_diag["band_off"] = "distilled"
            _rho_eff = lf_rho
            if _surgical and (_rho_eff is None or float(_rho_eff.max()) <= 1e-4):
                # THE decisive reachability counter for this block.  `_rho_band` is
                # computed on the `_surgical` path but only USED here, so a sensitivity
                # sweep of `_BAND_RHO_MAX` means nothing unless this fires: without it the
                # value is computed and thrown away, which is a reachability fact and not a
                # sensitivity one.  Measured 2026-10-02: fires on 2 of 8444 corpus step
                # records (both anima, step 32), because the detection tends to co-occur
                # with an ALREADY-ACTIVE primary rho channel, which is what this guard
                # stands down for.  That is the design (a fallback for the silent case),
                # not a wiring fault -- but it is why the constants here cannot be
                # calibrated from data.
                _ab_hit("lf.band_inject")
                _ab_hit("lf.band_rho_val", round(float(_rho_band), 6))
                # Checkerboard/banding signature detected, but the refresh
                # magnitude is gated on the stagnation+low-detail channel,
                # which stays silent when a fast few-step model converges
                # globally (all jumps collapse together -> median-relative
                # stagnation reads 0).  Inject a band-surgical refresh scaled
                # by the measured hi-band excess so the late-stage background
                # artifact can still be broken up.
                _shp = (lf_rho.shape if lf_rho is not None
                        else (x.shape[0], 1, *x.shape[2:]))
                # `_rho_band` is always assigned when `_surgical` is True -- every branch
                # above sets it, including the no-baseline branch, which also clears
                # `_surgical` -- so there is no default to substitute here.  The old
                # `is not None` dance against `_LF_BAND_RHO` was unreachable, and the
                # constant is gone.  A zero ramp needs no special case either: the
                # application guard below requires `_rho_eff.max() > 1e-4`, so a zero
                # refresh simply does not reach the step.
                _rho_val = _rho_band
                _rho_eff = torch.full(_shp, _rho_val, device=x.device,
                                      dtype=x.dtype)
            # Distilled gate: a distilled model converges globally by design,
            # so its jump field collapses everywhere at once.  The stagnation
            # channel (rho = s * tanh(log(med/j)) * sigmoid(-detail)) reads
            # that global convergence as per-pixel stagnation and re-noises the
            # whole image -- the flat 'inject=applied' path that still fired at
            # steps 3-13 (rho up to 0.0049) after the band-surgical and
            # conv_sigma gates.  There is nothing for the refresh to fix on a
            # distilled model: kill the whole refresh application.
            if not dist_eff and _rho_eff is not None \
                    and float(_rho_eff.max()) > 1e-4:
                # No-integration gate: skip the refresh once the measured
                # trajectory has converged (sigma_next at or below the probe's
                # conv_sigma).  Injected noise there cannot be integrated away
                # by the remaining steps, so it is pure damage -- and it is
                # exactly what pulled a distilled/cfg1 run below plain
                # simple+euler.
                if (_conv_sigma is not None
                        and float(s2) <= float(_conv_sigma)):
                    if lf_diag is not None:
                        lf_diag["inject"] = "skipped:converged"
                    _rho_eff = None
            if not dist_eff and _rho_eff is not None \
                    and float(_rho_eff.max()) > 1e-4:
                x = x0 + _lf_refresh(x - x0, _rho_eff, surgical=_surgical,
                                     ramp_cap=lf_gains["ramp"],
                                     hi_boost=lf_resc_rw, gen=lf_noise_gen)
                if lf_diag is not None:
                    lf_diag["band"] = ("surgical" if _surgical else "flat")
                    lf_diag["inject"] = "applied"
            elif dist_eff and lf_diag is not None:
                lf_diag["inject"] = "off:distilled"
            lf_m_prev = lf_m.detach().squeeze(1)
            lf_amp_prev = (lf_gains["amp"]
                           * (_lf_env_fade(k / n_cand,
                                           float(lf_evid.get("eg_n") or 0.0))
                              if lf_gains.get("env_fade") else 1.0))
            lf_rho_prev = lf_rho.detach()
            lf_applied_prev = True
            if lf_cum_next is not None:
                lf_cum = lf_cum_next
        if (dwell_on and dw_w is not None and s2 > 1e-8 and sn > 1e-8
                and not guard_active):
            # L1 per-pixel dwell: partial deterministic step for churning
            # pixels (x <- x_input + alpha * (x - x_input)); the final step
            # and the guard's protective pass-through always run at full
            # grid speed.
            _dw_env = _dwell_env(k / n_cand, dwell_ramp, dwell_fade_start,
                                 dwell_fade)
            _dw_w = dw_w
            if lf_on and lf_m is not None:
                # one controller per pixel: where the local field already
                # steers at a significant fraction of its amplitude, the
                # dwell stands down so the combined displacement deviation
                # stays inside the small-dose budget
                try:
                    _la = (lf_m.detach().float().abs()
                           / max(float(lf_gains["amp"]), 1e-6)).clamp(0.0, 1.0)
                    _dw_w = _dw_w * (1.0 - _la).to(_dw_w.dtype)
                except Exception:
                    pass
            _dw_alpha = (1.0 - dwell_max * _dw_env * _dw_w).clamp(0.35, 1.0)
            if float(_dw_alpha.min()) < 0.9999:
                try:
                    x = x_input + _dw_alpha.to(x.dtype) * (x - x_input)
                    if len(_LAST_ERRORS) < 64 or log_errors:
                        dwell_tel = {"w": round(float(_dw_w.mean()), 4),
                                     "wmax": round(float(_dw_w.max()), 4),
                                     "amin": round(float(_dw_alpha.min()), 4),
                                     "env": round(_dw_env, 4)}
                except Exception as e:
                    # never let the dwell break a run: one failure disables
                    # it for the rest of this sampling run
                    dwell_on = False
                    dw_w = None
                    if log_errors:
                        logging.warning("[PD-AOS-dwell] disabled for this "
                                        "run: %s", e)
        if s_up > 0:
            override_beta = None
            if is_adaptive_noise:
                if jmap is not None:
                    fresh_frac = (s_up * s_noise) ** 2 / max(sn * sn, 1e-12)
                    x0_norm = float(x0.float().norm().clamp_min(1e-8))
                    jmap_norm = float(jmap.float().norm().clamp_min(1e-8))
                    m0_step = _consistency_step(s, s2, space)
                    consistency = _compute_consistency(jmap_norm, x0_norm, m0_step)
                    override_beta = _predict_adaptive_beta(
                        jmap, err, eta_eff, s, space, sigma_data,
                        consistency, cfg, corr_used=corr_used,
                        fresh_frac=fresh_frac)
                if override_beta is None and cfg.get("_noise_beta_prior") is not None:
                    # No jump map yet (the very first fresh-noise draw) or a
                    # degenerate centroid: fall back to the probe's spectral
                    # whitepoint prior instead of plain white noise.
                    override_beta = float(cfg["_noise_beta_prior"])
            if is_adaptive_noise:
                fresh = noise_sampler(sigmas[i], sigmas[i + 1],
                                      override_beta=override_beta) * (s_up * s_noise)
            else:
                fresh = noise_sampler(sigmas[i], sigmas[i + 1]) * (s_up * s_noise)
            if cfg.get("color_conc_gate", True) and hasattr(noise_sampler, "last_beta"):
                _cb = float(noise_sampler.last_beta)
                if abs(_cb) > 1e-6:
                    _conc = _spectral_concentration(_cb)
                    _g = 1.0 / math.sqrt(max(_conc, 1.0))
                    _g = 1.0 + (_g - 1.0) * float(cfg.get("color_conc_strength", 1.0))
                    fresh = fresh * _g
            if space == "flow":
                x = x * ((1.0 - sn) / max(1.0 - s2, 1e-6)) + fresh
            else:
                x = x + fresh
        hist.append((u_log, u_sig, x0.detach(), x_input.detach(),
                     _x0_fed_hist))
        if len(hist) > hist_cap:
            hist.pop(0)
        prev_err = err
        prev_cons = step_cons
        # ---- raw per-step measurements (cheap; logged whether used or not) ---
        # Deliberately record RAW quantities, not just decisions, so signals we
        # do not consume yet can still be mined out of a saved log later.
        # x0_jump_rms is the RUN's own version of the probe's jump_curve, at
        # full step resolution rather than the probe's coarse grid -- which is
        # what the clamp analysis currently has to approximate.  RMS (not norm)
        # so the numbers do not scale with latent size.
        _raw = {}
        try:
            _xi = x_input.float()
            _x0f = x0.float()
            _x0_rms = float(_x0f.pow(2).mean().sqrt())
            _raw["x_rms"] = round(float(_xi.pow(2).mean().sqrt()), 6)
            _raw["x0_rms"] = round(_x0_rms, 6)
            _raw["step_rms"] = round(float((x.float() - _xi).pow(2).mean().sqrt()), 6)
            if len(hist) >= 2:
                _jp = hist[-2][2].float()
                _djump = _x0f - _jp
                _jr = float(_djump.pow(2).mean().sqrt())
                _raw["x0_jump_rms"] = round(_jr, 6)
                _raw["x0_jump_rel"] = round(_jr / max(_x0_rms, 1e-8), 6)
                # THE PROBE'S OWN ESTIMATOR, so the fusion's residual compares like with like:
                # per-pixel RMS jump over max(rms x0_k, rms x0_k+1), reduced by p50 over pixels
                # -- exactly the construction `jump_q` stores on the probe grid (see `_run_probe`).
                # Zero NFE: a different reduction of the same `_djump`.  MEASURED on the newest
                # runs, the global-RMS `x0_jump_rel` above reads 1.9-3.1x larger than this, so a
                # residual taken against THAT would have measured the estimator, not the model.
                # COMPUTED ONLY ON THE FUSION ARM, so the OFF arm's records are byte-identical to
                # the shipped ones -- a switch that adds a log field when it is off has already
                # changed the thing an A/B is supposed to hold fixed (found by
                # scratch_fix_logging_1_3.py, which asserts the per-step record's field set).
                if _fuse_on:
                    try:
                        _pjpx = _djump.pow(2).mean(dim=1).sqrt()
                        _sc = max(_x0_rms, float(_jp.pow(2).mean().sqrt()))
                        _raw["x0_jump_p50_rel"] = round(
                            float((_pjpx / max(_sc, 1e-8)).median()), 6)
                        # AND THE MATCH FOR `jump_curve`, which is the curve the probes ALWAYS store
                        # (`jump_q` is only written when the distribution pass runs, and on mklan
                        # the cond profile had none, which froze the fusion).  `jump_curve[i] =
                        # ||x0_{i+1} - x0_i|| / max(||x0_i||, ||x0_{i+1}||)`; both norms are over
                        # the same tensor, so the sqrt(N) cancels and this IS RMS(dx0)/max(RMS).
                        _raw["x0_jump_norm_rel"] = round(
                            _jr / max(_x0_rms, float(_jp.pow(2).mean().sqrt()), 1e-8), 6)
                    except Exception:
                        pass
                # spatial structure of the JUMP: whether the movement is
                # broad (composition) or detail-level
                _sj = _spatial_stats(_djump)
                _raw["dx0_std"] = round(_sj[1], 6)
                _raw["dx0_grad"] = round(_sj[2], 6)
                _raw["dx0_hf"] = round(_sj[3], 6)
            # spatial/spectral structure of the clean estimate itself
            _s0 = _spatial_stats(_x0f)
            _raw["x0_std"] = round(_s0[1], 6)
            _raw["x0_grad"] = round(_s0[2], 6)
            _raw["x0_hf"] = round(_s0[3], 6)
            # WHERE that high-frequency energy sits.  `x0_hf` is a whole-tensor reduction and
            # cannot tell a uniformly detailed image from one with a single damaged patch -- the
            # distinction NOTES' artifact entry says nothing in the logs could make.  This is
            # the field form of the same quantity, reduced to its tile extremes
            # (max tile mean, share of the total, y, x).  One extra blur pair per step, the same
            # cost class as `_spatial_stats`, which is already permanent.
            _ht = _tile_extremes(_hf_field(_x0f))
            if _ht is not None:
                _raw["x0_hf_tile"] = [round(_ht[0], 6), round(_ht[1], 4), _ht[2], _ht[3]]
        except Exception:
            pass
        _raw["lam_min_used"] = (round(_lam_lo, 4) if _lam_lo is not None else None)
        # which source produced the bound: "run" (the step's own measured jump),
        # "probe" (the probe's jump curve), or "fixed" (cfg / default)
        _raw["lam_bound_src"] = _lam_src
        _raw["lam_max_used"] = (round(float(cfg.get("aflops_lam_max", _AFLOPS_LAM_MAX)), 4)
                                if _lam_lo is not None else None)
        # THE REMAINDER GATE'S OWN DECISION, logged every step it made one.  Without this
        # the criterion cannot be audited: `wmax` is the weight the SHIPPED test reads,
        # while this arm decides on `step_err` against `tol_eff`, and those are different
        # quantities.  Enough digits to reproduce the comparison (rule 8).
        _raw["wmax_budget"] = round(float(cfg.get("wmax_budget", 1.0) or 1.0), 4)
        # The cap trace lands HERE, not at the decision site: `_raw` is built after the step, and
        # the cap is chosen before it (writing there raised UnboundLocalError on every ON run).
        if _fuse_st is not None and _cap_pre_arm is not None:
            _raw["wmax_cap_pre_arm"] = round(float(_cap_pre_arm), 6)
            _raw["wmax_cap_in_force"] = round(
                float(_cap_in_force if _cap_in_force is not None else _cap_pre_arm), 6)
        # ---- THE FUSION'S OWN STEP --------------------------------------------------------
        # Record the weight that was IN FORCE for this step, then advance it from THIS step's
        # measurement.  The measurement only exists now, so it cannot inform the step it
        # describes: `w(k+1)` is what steps 0..k earned, which is the one-step lag a forecaster
        # necessarily has.  Zero NFE -- the measurement is a reduction of tensors already held.
        if _fuse_st is not None:
            _raw["fuse_w_in_force"] = round(float(_fuse_st.get("w", 0.0)), 4)
            # THE MEASUREMENT IS PASSED WHOLE: each probe's curve is scored against the estimator
            # that matches it, and the side carries its own `y_key` (see `_fuse_probe_state`).
            _fuse_step_update(_fuse_st, k - 1, _raw)
            _raw["fuse_w_next"] = round(float(_fuse_st.get("w", 0.0)), 4)
            _ab_hit("fuse.w_used", round(float(_fuse_st.get("w", 0.0)), 4))
        # WHAT THE ORDER SYSTEM WAS ASKING FOR, next to what it got (`order`/`ord_mean`).
        # If `ord_want` is high while `ord_mean` is ~1, the LIMITER is clamping and the fix
        # is in the limiter; if `ord_want` is ~1, the CRITERION prefers order 1 and no cap
        # change can help.  `ord_innov` is the curve whose argmin chooses p_star.
        if _ord_want is not None:
            _raw["ord_want"] = round(float(_ord_want), 4)
        if _ord_ps is not None:
            _raw["ord_pstar"] = round(float(_ord_ps), 4)
        if _ord_innov is not None:
            _raw["ord_innov"] = _ord_innov
        # The Runge criterion's record: per order, the mean remainder bound and the fraction
        # of pixels admitted.  This is where the spatial adaptivity is visible -- if the
        # criterion is doing what it claims, order 5-6 is admitted on SOME pixels while order
        # 2 is refused on others in the same step.
        if _ord_bound is not None:
            _raw["order_by_bound"] = True
            _raw["ord_bmax"] = _ord_bound.get("kmax")
            _raw["ord_btol"] = _ord_bound.get("tol_x0")
            _raw["ord_bper_k"] = _ord_bound.get("per_k")
            # THE RUN-SIDE per-order derivative scale, median over pixels, for the
            # same orders `per_k` judged.  NOT the probe's `dk_scale`: this one is
            # normalised by the PER-PIXEL |x0| on the RUN's grid; the probe divides by
            # the WHOLE-TENSOR norm on the PROBE grid.  Measured on 3d843e2 they differ
            # by 10^2-10^5 and the gap grows with k, so the two must not be fed to the
            # same fit.  This is the bound's own scale; the probe-scale series for the
            # order_max equation is `dk_series` (profile side).
            _raw["ord_bdk"] = _ord_bound.get("dk_run")
            # WHICH ARM OF THE BOUND A/B PRODUCED THIS STEP.  The switch is in the cfg dump, but a
            # per-step record is what shows it ACTED, and `per_k`'s own 4th element already carries
            # "applied" / "no_history" per order -- so a run can be checked for having corrected the
            # orders it could, rather than for merely having been switched on (rule 9).
            # WHAT THE ORDER SYSTEM WAS ASKING FOR, next to what it got.  The remainder's scale is
            # now evaluated at the FAR END for every order that has the history for it (see
            # `_pixel_order_bound`), and `ord_bfix` records how many it could correct this step plus
            # the per-order `applied` / `no_history` marker inside `ord_bper_k`.
            _raw["ord_bfix"] = _ord_bound.get("bound_fix")
        # The forced floor's own record: the value in force and how many pixels it raised.
        # `order_floor` is in the cfg dump, but a per-step count is what shows it ACTED.
        if _order_floor_auto:
            # WHICH POLICY SET THE LEVEL, and its strength -- so a corpus reader never infers the request
            # from a config dump (the `cond_rtol_loosen` lesson).
            #
            # THE STAMP CALLS THE SAME FUNCTION THE DEMAND SITE CALLED (`_demand_policy`), so it cannot
            # name a policy the maths did not run.  Testing the predicate inline is how the bound arm was
            # once named on every run with `order_by_bound` off (115 of 418 good-era runs).
            _raw["ord_demand_q"] = _order_demand_q
            _raw["ord_demand_policy"] = _demand_policy(_ord_bound is not None, _order_demand_q,
                                                       _order_by_bound, _order_demand_bound,
                                                       _of_graded)
            # WHAT THE GATE DECIDED THIS STEP, recomputed from the same two inputs the demand used, so the
            # record cannot disagree with the maths even on a step where the ladder never ran.  The fixed
            # modes stamp EXACTLY what they stamped before: this branch is new, that one is untouched.
            _gate_rec = bool(_prev_rtol is not None and _prev_rtol > _order_floor_rtol)
            # THE GRADED RAMP'S VALUE, on every arm -- it is a pure function of the carried error ratio,
            # so on the "bound" and "bound_map" arms it does NOT report the demand that ran (audit
            # U04/U25 `aflops.py:13449`, still open).  `ord_demand` below is the value in force.
            _raw["ord_floor"] = _graded_demand(_prev_rtol, _order_floor_rtol, _of_lo, _of_hi,
                                               _of_r_hi, _order_floor_demand, _of_graded)
            _raw["ord_floor_px"] = _ord_floor_n
            # THE DEMAND LEVEL ACTUALLY PUT IN FORCE THIS STEP (1.0 = nothing raised), on every arm, so
            # the request can be audited instead of inferred from a pixel count.
            _raw["ord_demand"] = round(float(_fl_val), 4)
            _raw["ord_floor_gate"] = _gate_rec
            _raw["ord_floor_rtol"] = _order_floor_rtol
            _raw["ord_floor_prev_ratio"] = (None if _prev_rtol is None
                                            else round(float(_prev_rtol), 6))
        elif _order_floor > 1.0:
            _raw["ord_floor"] = _order_floor
            _raw["ord_floor_px"] = _ord_floor_n
        if _wgate_info is not None:
            _raw["wbound"] = float(_wgate_info["step_err"])
            _raw["wtol"] = round(float(_wgate_info["tol_eff"]), 9)
            _raw["wgate_k"] = int(_wgate_info["k"])
            _raw["wgeom"] = float(_wgate_info["geom"])
        # THE ADAPTIVE ORDER GATE'S INPUT, computed where BOTH err and tol are live -- they are the same
        # two values the step record logs, so the gate's input is exactly the number a corpus reader
        # sees.  err/tol is dimensionless, so ONE threshold transfers across models and schedules; it is
        # carried one step forward because this step's err only exists here.  NO try/except: a silent
        # guard at the first site hid a NameError and held the gate shut on every step, even at 1e-9.
        # THE NAME MUST MATCH THE GATE'S (`_prev_rtol`): the first version wrote `prev_rtol` here while the
        # gate read `_prev_rtol`, so the gate read a variable nothing assigned and never opened.
        _prev_rtol = (None if (err is None or not tol) else float(err) / float(tol))
        if len(_LAST_ERRORS) < 64 or log_errors:
            _LAST_ERRORS.append({"mode": "engine", "step": k,
                                 "sigma": round(s, 6), "sigma_next": round(sn, 6),
                                 "err": None if err is None or math.isinf(err) else round(err, 6),
                                 "cerr": (None if cerr is None
                                          else round(cerr, 6)),
                                 # THE CUT'S OWN DECISION VARIABLE, logged every
                                 # step the cut is armed -- not only when it fires.
                                 # Without this the criterion cannot be audited:
                                 # `err` is the corrected-vs-MIDPOINT distance,
                                 # while the cut tests `_e_raw` = corrected-vs-PLAIN,
                                 # and those are different quantities.  Measured in
                                 # scratch_cut_scaling_rig.py, which initially
                                 # analysed `err` and drew the wrong power law.
                                 #
                                 # NINE decimals, not six: the decision is `_e_raw > tol`
                                 # and `_e_raw` lands within ~1e-6 of tol at the
                                 # largest-du step, so a 6dp log cannot reproduce the
                                 # engine's own fire/no-fire decision.  The rig's
                                 # self-validation check caught exactly that.
                                 "e_raw": (None if _e_raw_used is None
                                           else round(_e_raw_used, 9)),
                                 # T2c diagnostics: the blend weight ACTUALLY used
                                 # (distinct from `blend_w`, which is the
                                 # pixel-blend mean), and the same deviation
                                 # measured to x_pred instead of x1.
                                 "corr_w": (None if _corr_w_used is None
                                            else round(_corr_w_used, 9)),
                                 "e_pred": (None if _e_pred_used is None
                                            else round(_e_pred_used, 9)),
                                 "tol": round(tol, 6), "order": eff,
                                 "eff_cap": eff_cap,
                                 "hist_cos": (round(hist_cos_val, 4)
                                              if hist_cos_val is not None else None),
                                 # THE PER-PIXEL GATE'S OWN RECORD.  `hist_cos_px_neg` is the
                                 # fraction of pixels whose LOCAL cosine is negative and are
                                 # therefore set to order 1; `hist_cos_px_on` is the effective
                                 # arm (OFF when the per-pixel machinery is not adaptive).
                                 "hist_cos_px_neg": (round(float(hist_cos_px_neg.float().mean()), 4)
                                                     if hist_cos_px_neg is not None else None),
                                 "hist_cos_px_on": bool(_hist_cos_px_on),
                                 # THE PER-PIXEL SECOND-ORDER TERM'S RECORD.  `second_order_px`
                                 # is the fraction of pixels that receive the slope term when the
                                 # arm is ON; None means the arm was OFF or the ladder did not
                                 # produce an admitted map.
                                 "second_order_px": (round(float(_so_mask.float().mean()), 4)
                                                     if _so_mask is not None else None),
                                 "second_order_px_on": bool(cfg.get("second_order_per_pixel", False)),
                                 "second_order_raw": bool(_so_raw),
                                 "blend_w": (round(blend_w_mean, 4)
                                             if blend_w_mean is not None else None),
                                 "ord_mean": (round(ord_tel[0], 3)
                                              if ord_tel is not None else None),
                                 "ord_max": (ord_tel[1]
                                             if ord_tel is not None else None),
                                 "ord_reset": (round(ord_tel[2], 4)
                                               if ord_tel is not None else None),
                                 # THE ORDER MAP'S SPATIAL STRUCTURE (see the block where it is
                                 # computed for what each field means and why BOTH the requested and
                                 # the admitted map are recorded).  None when the ladder did not run.
                                 "ord_spat": _ord_spat,
                                 "wmax": round(wmax, 4),
                                 "wmax_limit": round(wmax_limit, 4),
                                 # The ARMs must be readable from the run, not just from the cfg
                                 # (rule 8): `wmax_limit` alone cannot distinguish "the `eg_n`
                                 # branch pinned this" from "the adaptive feedback produced it",
                                 # and those are the two hypotheses under test.
                                 "wmax_eg_dir": round(_eg_dir, 4),
                                 "wmax_eg_n": round(_eg_n, 4),
                                 "outlier": None if outlier_sig is None else round(outlier_sig, 6),
                                 "guard_sev": round(getattr(guard, "last_sev", 1.0), 4) if guard is not None else None,
                                 "space": use_space if len(hist) >= 1 else "log_sigma",
                                 "guard": guard_trip,
                                 "guard_active": guard_active,
                                 "an_detected": an_detected,
                                 "corr": corr_used,
                                 "eta_eff": round(eta_eff, 6),
                                 "eta_stuck": (round(eta_stuck, 4)
                                               if eta_stuck is not None else None),
                                 "eta_jump_rel": (round(eta_jump_rel, 6)
                                                  if eta_jump_rel is not None else None),
                                 "eta_detail_abs": (round(eta_detail_abs, 4)
                                                    if eta_detail_abs is not None else None),
                                 # THE MOMENTUM BRANCH'S OWN READING, stamped even
                                 # when `escape_mode` is "flatness" so an A/B can
                                 # be read from the artefacts themselves: the
                                 # score says whether it WOULD have armed this
                                 # step, and the two ratios say by how much.  A
                                 # branch that only writes its result when it is
                                 # on cannot be compared against the arm where it
                                 # is off (rule 8).
                                 "eta_momentum": (round(eta_momentum, 4)
                                                  if eta_momentum is not None else None),
                                 "eta_momentum_ratio": (round(eta_momentum_ratio, 4)
                                                        if eta_momentum_ratio is not None else None),
                                 "eta_growth_rate": (round(eta_growth_rate, 4)
                                                     if eta_growth_rate is not None else None),
                                 # THE BUDGET THIS RUN ACTUALLY USED.  When the
                                 # probe-derived form is armed the budget is no
                                 # longer the constant in the config, and a
                                 # reader must never have to infer it (rule 14):
                                 # stamp the value in force.
                                 "eta_growth_budget": (
                                     round(float(cfg.get("escape_momentum_growth", 0.15)), 6)
                                     if str(cfg.get("escape_mode", "")) == "momentum"
                                     else None),
                                 # THE DERIVED SCHEDULE'S OWN LEVEL, beside the
                                 # level actually used, so the two arms of the A/B
                                 # can be compared from the artefact (rule 8).
                                 "eta_probe": (round(float(eta_probe), 8)
                                               if eta_probe is not None else None),
                                 "eta_escape_commit": cfg.get("_escape_commit"),
                                 "noise_beta": (round(noise_sampler.last_beta, 4)
                                                if noise_sampler is not None and
                                                hasattr(noise_sampler, "last_beta") else None),
                                 "aflops_lam": (round(aflops_lam, 4)
                                                if aflops_lam is not None else None),
                                 # The UNCLAMPED estimate.  Reporting only the
                                 # clamped one made it impossible to tell from a
                                 # log whether the clamp was binding or what
                                 # the raw estimate even looked like.
                                 "aflops_lam_raw": (round(aflops_lam_raw, 4)
                                                    if aflops_lam_raw is not None
                                                    else None),
                                 "lf": lf_diag,
                                 "dwell": dwell_tel,
                                 "calls": calls,
                                 **_raw})
            if len(_LAST_ERRORS) > 64:
                del _LAST_ERRORS[:-64]
        if log_errors:
            _nb = (noise_sampler.last_beta if noise_sampler is not None and
                   hasattr(noise_sampler, "last_beta") else None)
            logging.info("[PD-AOS] k=%d sigma=%.4f err=%s tol=%.4g ord=%d "
                         "guard=%s an=%s corr=%s eta=%.3f nb=%s lam=%s calls=%d"
                         " hcos=%s ec=%d bw=%s",
                         k, s, "n/a" if err is None else f"{err:.4g}", tol, eff,
                         guard_trip, an_detected,
                         corr_used, eta_eff,
                         "n/a" if _nb is None else f"{_nb:.2f}",
                         "n/a" if aflops_lam is None else f"{aflops_lam:.3f}",
                         calls,
                         "n/a" if hist_cos_val is None else f"{hist_cos_val:.3f}",
                         eff_cap,
                         "n/a" if blend_w_mean is None else f"{blend_w_mean:.3f}")
        if callback is not None and not disable:
            callback({"x": x, "denoised": x0, "i": k, "sigma": sigmas[i], "sigma_hat": sigmas[i]})
            _progress(callback, "step", "progress", k, n_cand,
                      "PD-AOS sampling: step %d/%d" % (k, n_cand),
                      extra={"value": k, "total": n_cand})
    try:
        if (lf_dir_acc[1] > 0
                and bool(cfg.get("lf_dir_real_feedback", False))):
            _dir_val = round(lf_dir_acc[0] / lf_dir_acc[1], 4)
            _profs = []
            if bool(cfg.get("cond_probe_enabled", False)) and skey is not None:
                _p = _COND_PROBE_PROFILES_SCHED.get(skey)
                if _p is not None:
                    _profs.append(_p)
            if bool(cfg.get("probe_enabled", False)) and pkey is not None:
                _p = _PROBE_PROFILES.get(pkey)
                if _p is not None:
                    _profs.append(_p)
            for _p in _profs:
                _lst = _p.setdefault("dir_real", [])
                _lst.append(_dir_val)
                del _lst[:-5]
            if log_errors and _profs:
                logging.info("[PD-AOS-lf] run direction feedback: "
                             "dir_real=%.3f (%d steps measured)",
                             _dir_val, lf_dir_acc[1])
    except Exception:
        pass
    # Final A/B reachability snapshot, taken AFTER the loop so it reflects what
    # actually ran rather than the empty record built before it.  This is what
    # makes a saved log self-describing about its own A/B validity.
    try:
        _LAST_CFG["ab_reach"] = _ab_reach_snapshot()
        # THE VALUE IN FORCE, STAMPED RATHER THAN INFERRED FROM THE GATE'S COUNTERS.
        # `wmax_bound` became a widget on the Debug Options node (audit C2), so a log
        # has to say which arm ran; `wmax_gate` below counts off/on, which implies it
        # only when the gate was reached at all.
        _LAST_CFG["wmax_bound"] = bool(cfg.get("wmax_bound", False))
        # THE ORDER CRITERION AND THE MODEL SPACE, STAMPED SO AN A/B ARM IS
        # IDENTIFIABLE FROM THE ARTEFACT (rule 33).  `order_by_bound` selects the
        # per-pixel Runge criterion over the innovation argmin, and it is a Debug
        # Options widget -- without this stamp the two arms of that A/B look
        # identical in the log, which is exactly the trap a label is supposed to
        # close.  `space_model` is the resolved flow/VE answer (the ladder's own
        # `space` field is a different thing and is stamped per step).
        _LAST_CFG["order_by_bound"] = bool(cfg.get("order_by_bound", True))
        _LAST_CFG["order_pixel_adaptive"] = bool(
            cfg.get("order_pixel_adaptive", True))
        # THE DIAL IN FORCE, so a sweep is attributable from the log alone (the order
        # map itself only says what it ended up being, not what was asked for).
        _LAST_CFG["order_bound_blend"] = round(
            min(max(float(cfg.get("order_bound_blend", 1.0) or 1.0), 0.0), 1.0), 3)
        _LAST_CFG["_space_src"] = cfg.get("_space_src")
        _LAST_CFG["space_model"] = space
        # Rule 9 for the remainder gate: a switch wired to nothing is indistinguishable
        # from one whose effect is below the noise floor.  Every verdict is counted, and
        # the run's own record carries the count.
        _LAST_CFG["wmax_gate"] = dict(_ORDER_GATE_AB)
        # THE FUSION'S OWN RECORD, so an A/B needs no second run to be readable: whether the arm
        # was armed, the reliability PRIOR it started from, the weight it ended on, and the caps
        # it actually put in force (first few, rounded).  The per-step trace is in each step
        # record (`fuse_w_in_force`, `fuse_w_next`), which is what "firing per step" has to mean
        # if a switch is not to be a claim.
        if _fuse_on:
            _LAST_CFG["probe_fuse"] = {
                "armed": _fuse_st is not None,
                "w_prior": (round(float(_fuse_st.get("w0", 0.0)), 4)
                            if _fuse_st is not None else None),
                "w_final": (round(float(_fuse_st.get("w", 0.0)), 4)
                            if _fuse_st is not None else None),
                "n_positions": (int(_fuse_st.get("n", 0)) if _fuse_st is not None else 0),
                "caps_in_force": list(_fuse_caps[:8]),
                "loss": ([round(float(v), 4) for v in _fuse_st.get("loss", [])]
                         if _fuse_st is not None else None),
                # WHAT THE WEIGHT UPDATE ACTUALLY SAW.  `n_ev` counts the steps each side was
                # scored on and `unevaluated` the steps skipped because one side's curve was
                # missing -- without these, a loss of 0.0 is indistinguishable from a probe that
                # predicted perfectly, which is how the first live ON run's weight came to move
                # toward a probe that was never evaluated.
                "n_ev": ([int(v) for v in _fuse_st.get("n_ev", [])]
                         if _fuse_st is not None else None),
                "unevaluated": (int(_fuse_st.get("unevaluated", 0))
                                if _fuse_st is not None else None),
                "curve_lens": ({"model_curv": len(_fuse_st.get("cm") or []),
                                "cond_curv": len(_fuse_st.get("cc") or []),
                                "model_jump": len((_fuse_st.get("sides") or [{}])[0].get("J") or [])
                                if (_fuse_st.get("sides") or [None])[0] else 0,
                                "cond_jump": len((_fuse_st.get("sides") or [{}, {}])[1].get("J") or [])
                                if len(_fuse_st.get("sides") or []) > 1
                                and (_fuse_st.get("sides") or [None, None])[1] else 0}
                               if _fuse_st is not None else None),
                # WHICH ESTIMATOR EACH SIDE WAS SCORED AGAINST: `jump_q` (with the probe's own
                # measured p10/p90 spread) or the `jump_curve` fallback (with its own trajectory
                # spread, which is NOT an uncertainty).  Without this the two are indistinguishable
                # in a log, and a side scored against the wrong estimator would look like a model
                # that predicts badly.
                "estimator": ([(_sd or {}).get("src") for _sd in (_fuse_st.get("sides") or [])]
                              if _fuse_st is not None else None),
                "y_key": ([(_sd or {}).get("y_key") for _sd in (_fuse_st.get("sides") or [])]
                          if _fuse_st is not None else None),
                "cap_raise_allowed": bool(_fuse_raise),
                # WHICH DERIVATION THE CAP CAME FROM, and the scalar it was floored against: the
                # arm must make the SAME derivation choice the scalar path makes, and may tighten
                # only within the order policy the run already had.
                "cap_derivation": ((_fuse_st or {}).get("derivation")
                                   if _fuse_st is not None else None),
                "cap_scalar": cfg.get("wmax_base"),
                "cap_scalar_order": _max_order_for_cap(cfg.get("wmax_base")),
            }
    except Exception:
        pass
    _aflops_close_run()
    return x


def aflops_engine(model, x, sigmas, extra_args=None, callback=None, disable=None,
                  cfg=None, log_errors=False):
    """The sampling entry point: the step loop, with the run's records closed on ANY exit.

    A successful run stamps its own record and closes it inside the body.  If the body
    raises -- including ComfyUI's `InterruptProcessingException`, which is a
    `BaseException` and so would slip past an `except Exception` -- this wrapper stamps
    the failure in `_LAST_CFG` (with the A/B counters and reach record that caused it)
    and closes the per-run records before re-raising, so the aborted run cannot lend
    its counters to the next run's log.
    """
    try:
        return _aflops_engine_body(model, x, sigmas, extra_args=extra_args,
                                   callback=callback, disable=disable,
                                   cfg=cfg, log_errors=log_errors)
    except BaseException:
        try:
            _LAST_CFG["run_aborted"] = {
                "calib_ab": dict(_CALIB_AB),
                "ab_reach": _ab_reach_snapshot(),
                "wmax_bound": bool((cfg or {}).get("wmax_bound", False)),
                "order_by_bound": bool((cfg or {}).get("order_by_bound", True)),
                "phase": "engine",
            }
        except Exception:
            pass
        _aflops_close_run()
        raise


def _subdivide_tail(sig, tail_steps, factor):
    if factor <= 1 or len(sig) < 4:
        return sig
    tail_steps = max(1, min(int(tail_steps), len(sig) - 2))
    split = len(sig) - tail_steps - 1
    parts = [sig[:split + 1]]
    tail = sig[split:]
    for a, b in zip(tail[:-1], tail[1:]):
        lo, hi = float(a), float(b)
        if hi <= 0.0:
            grid = torch.exp(torch.linspace(math.log(max(lo, 1e-8)), math.log(1e-8),
                                            int(factor) + 1, device=sig.device, dtype=sig.dtype))
            grid[-1] = 0.0
        else:
            grid = torch.exp(torch.linspace(math.log(max(lo, 1e-8)), math.log(hi),
                                            int(factor) + 1, device=sig.device, dtype=sig.dtype))
        parts.append(grid[1:])
    return torch.cat(parts)

_SCHEDULER_MAP = {
    "log_uniform": "exponential",
    "sigma_uniform": "simple",
    "timestep_uniform": "normal",
    "karras": "karras",
    "beta": "beta",
}

def aflops_scheduler_sigmas(model_sampling, steps, spacing="log_uniform",
                            shift=1.0, denoise=1.0, tail_subdivide=1, tail_steps=4):
    steps = int(steps)
    if steps < 1 or float(denoise) <= 0.0:
        return torch.FloatTensor([])
    total = steps if float(denoise) >= 1.0 else int(steps / float(denoise))
    sig = comfy.samplers.calculate_sigmas(model_sampling, _SCHEDULER_MAP[spacing], total)
    sig = sig[-(steps + 1):].detach().float()
    shift = float(shift)
    if shift != 1.0:
        s0 = float(sig[0])
        if s0 > 1.5:
            u = sig / max(s0, 1e-8)
            sig = s0 * (shift * u / (1.0 + (shift - 1.0) * u))
        else:
            sig = shift * sig / (1.0 + (shift - 1.0) * sig)
    sig[-1] = 0.0
    return _subdivide_tail(sig, tail_steps, tail_subdivide)