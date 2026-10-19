# -*- coding: utf-8 -*-
"""Faithful A-FloPS "A-Euler" for native flow models (Algorithm 2 of
Jin et al., AAAI 2026, arXiv:2509.00036), as a ComfyUI KSampler engine.

This reproduces the AUTHORS' reference implementation
(github.com/jinc7461/AFloPs, class AEuler), i.e. the native-flow sampler
that generated the paper's SD3.5 ablation:

    flow time    t = 1 - sigma            (LINEAR in sigma, noise -> clean)
    velocity     v = -d,  d = (x - denoised)/sigma
    first step   x + d*(sigma_next - sigma)        <- the sigma-Euler, exact
    later steps  adaptive exponential integrator:
        c      = <v - v_prev, x - x_prev> / (||x - x_prev||^2 + 1e-8)  (Eq.12)
        h_now  = v - c*x ;  h_prev = v_prev - c*x_prev
        dt     = t_next - t_now
        x'     = x*e^{c*dt} + A*h_now + B*(h_now - h_prev)/(t_now - t_prev)
        A      = (e^{c*dt} - 1)/c ,  B = (e^{c*dt} - 1 - c*dt)/c^2   (Eq.13/15)

Why t = 1 - sigma and not the model's internal time: ComfyUI's CONST flow
sampling interpolates x = (1-sigma)*x0 + sigma*noise, so sigma IS the
interpolation parameter and the flow time is linear in it.  Two consequences:

  * the first (flow-time Euler) step is algebraically IDENTICAL to ComfyUI's
    own sigma-Euler step, so there is no truncation error at the start;
  * the velocity v = -d stays bounded, and because dt is an actual sigma
    difference, ANY sigma schedule integrates correctly.

`c` is per-batch-element (reduced over the trailing dims), matching the
reference.  The reference does NOT clamp `c` for this sampler; pass
lam_min/lam_max to clamp it (testing / A-B only).

Note: the sigma<->flow-time map helpers at the top of this file are for the
companion scheduling node (uniform-flow-time grids for non-flow models); the
sampler itself needs no map.
"""
import math

import torch


# `_FLUX_CLASS` WAS REMOVED 2026-10-04, one step after the function that read it: deleting
# `_flow_time_of_sigma` orphaned this constant, and the sweep's own census flagged it on the next run.
# Flux goes through the generic `ms.timestep` branch of `_sigma_of_flow_time` below.
_COSMOS_CLASS = "ModelSamplingCosmosRFlow"


# ---------------------------------------------------------------------------
# sigma <- model-flow-time map, used by the companion scheduling node through
# `_sigma_of_flow_time` (called below at `_uniform_flow_time_sigmas`).
# THE INVERSE, `_flow_time_of_sigma`, WAS REMOVED 2026-10-04 by the dead-code
# sweep: nothing called it -- no schedule, no node, no rig -- while its own
# docstring claimed the scheduling node used it (rule 26: a stale comment reads
# as current truth).
# ---------------------------------------------------------------------------
def _sigma_of_flow_time(ms, tau):
    """sigma at flow time tau (1 = pure noise, 0 = clean) from the model's own map.

    THE TIMESTEP SCALE IS NOT FLOW TIME ON EVERY CLASS.  `ModelSamplingDiscreteFlow`
    (SD3/3.5, Wan, LTXV, and `ModelSamplingAV`, which inherits it) defines
    `sigma(timestep) = time_snr_shift(shift, timestep / multiplier)` with
    `multiplier = 1000` (`comfy/model_sampling.py:328-329`), i.e. it wants a NATIVE
    TIMESTEP in [0, multiplier], not tau in [0, 1].  Passing tau there returned
    `time_snr_shift(shift, tau/1000)` and the emitted top sigma came out 334x too
    small on SD3, 126x on Wan and 423x on LTXV (measured; "1000x" is the
    argument-domain error, not the emitted ratio).  Those classes are the ones that
    carry `multiplier`, so the map is `sigma(tau * multiplier)` and it is exact at
    any shift.  Every other class takes flow time (Flux) or its own fraction
    (Cosmos, handled above) directly.
    """
    cls = type(ms).__name__
    tau = float(tau)
    if cls == _COSMOS_CLASS:
        smax = float(ms.sigma_max)
        if tau >= smax / (smax + 1.0):
            return smax
        return tau / (1.0 - tau)
    mult = getattr(ms, "multiplier", None)
    try:  # the model's own forward map, in the time base that map expects
        if mult is not None:
            return float(ms.sigma(torch.tensor(tau * float(mult))))
        return float(ms.sigma(torch.tensor(tau)))
    except Exception:
        return tau / max(1.0 - tau, 1e-6)


def _uniform_flow_time_sigmas(ms, steps):
    """The paper's OWN grid (Algorithm 2): t = n/N (noise -> clean), uniform
    flow time over [0, 1]; sigma = sigma(tau = 1 - t).  The scheduler node emits
    this and that is its only producer -- the sampler does NOT rebuild it, it
    consumes whatever sigma list it is handed."""
    out = []
    for n in range(steps + 1):
        t = n / steps
        out.append(_sigma_of_flow_time(ms, 1.0 - t))
    return torch.FloatTensor(out)


# ---------------------------------------------------------------------------
# exponential-integrator phi-functions (tensor form: c is per batch element)
# ---------------------------------------------------------------------------
def _phi1(z):
    """phi1(z) = (e^z - 1)/z, with a Taylor branch at z ~ 0."""
    small = z.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(z), z)
    return torch.where(small,
                       1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)),
                       torch.expm1(safe) / safe)


def _phi2(z):
    """phi2(z) = (e^z - 1 - z)/z^2, with a Taylor branch at z ~ 0."""
    small = z.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(z), z)
    return torch.where(small,
                       0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z / 120.0)),
                       (torch.expm1(safe) - safe) / (safe * safe))


def _lambda_per_sample(dv, dx, eps=1e-8):
    """Per-batch-element least-squares lambda (paper Eq.12 / reference)."""
    fv = dv.reshape(dv.shape[0], -1)
    fx = dx.reshape(dx.shape[0], -1)
    num = (fv * fx).sum(dim=1)
    den = (fx * fx).sum(dim=1) + eps
    return (num / den).view(-1, *([1] * (dv.ndim - 1)))


def _aeuler_step(x, v, x_prev, v_prev, t_now, t_prev, t_next,
                 lam_min=None, lam_max=None):
    """One adaptive step (paper Eq.12-16, reference AEuler.step)."""
    c = _lambda_per_sample(v - v_prev, x - x_prev)
    if lam_min is not None and lam_max is not None:
        c = c.clamp(lam_min, lam_max)
    t_s = t_next - t_now
    h_now = v - c * x
    h_prev = v_prev - c * x_prev
    d2 = (h_now - h_prev) / max(t_now - t_prev, 1e-8)
    z = c * t_s
    A = _phi1(z) * t_s
    B = _phi2(z) * t_s * t_s
    return x * torch.exp(z) + A * h_now + B * d2, c


@torch.no_grad()
def sample_paper_aeuler(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, lam_min=None, lam_max=None, **kwargs):
    """A-FloPS A-Euler on the caller's sigma list (any schedule).

    lam_min/lam_max are TESTING knobs: the reference does not clamp lambda, so
    the default is no clamp.  Set them (e.g. -1.0/1.0) to A/B the paper's
    stated stability clamp.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    n_steps = len(sigmas) - 1
    if n_steps < 1:
        return x

    x_prev = v_prev = None
    t_prev = None

    for i in range(n_steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        # dx/dsigma (the Karras derivative); t-velocity is v = -d since
        # t = 1 - sigma.
        if sigma > 0.0:
            d = (x - denoised) / sigma
        else:
            d = torch.zeros_like(x)
        v = -d

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})

        t_now = 1.0 - sigma
        t_next = 1.0 - sigma_next
        x_before = x

        have_prev = x_prev is not None
        valid_prev = have_prev and (t_now - t_prev) > 0.0 and t_next > t_now

        if not valid_prev:
            # First step (reference AEuler): plain Euler in sigma -- which IS
            # the flow-time Euler step because t = 1 - sigma is linear.
            x = x + d * (sigma_next - sigma)
        else:
            x, _c = _aeuler_step(x, v, x_prev, v_prev, t_now, t_prev, t_next,
                                 lam_min, lam_max)

        x_prev = x_before
        v_prev = v
        t_prev = t_now

    return x
