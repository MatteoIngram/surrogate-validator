"""
Surrogate-data validation for financial return series and trading strategies.

Question answered: does a series (or a strategy's returns) contain structure
that noise with the same amplitude distribution and linear autocorrelation
would not produce?

Methodology refactored from the dissertation pipeline (BDS.py,
surrogatedata.py, CODE/LS.py):
  - IAAFT surrogates, 100 iterations, stop when ranks stop changing
  - GARCH(1,1), constant mean, Student-t innovations; standardised residuals
  - BDS with epsilon = 1.5 standard deviations
  - delay tau = first local minimum of average mutual information (64 bins)
  - embedding dim m = first m with < 1% false nearest neighbours
    (Kennel et al. tests, R_tol = 15, A_tol = 2)
  - Rosenstein largest Lyapunov exponent via nolds.lyap_r with
    Theiler window 20 and fit length 10
  - rank-based surrogate p-value with minimum 1 / (1 + N)

Changes from the dissertation code:
  - GARCH is estimated by maximum likelihood with scipy.optimize, not `arch`
    (unavailable in Pyodide). Estimates match arch to four decimals.
  - BDS is computed by a port of the statsmodels algorithm with 64-bit
    counts; statsmodels' own version overflows int32 in Pyodide.
  - Lyapunov line fit uses nolds' fit="poly" (least squares) instead of the
    RANSAC default, which needs scikit-learn and is not deterministic.
  - The Lyapunov surrogate test is two-sided, not one-sided upper. On the
    Lorenz positive control the observed exponent lies BELOW every IAAFT
    surrogate (true neighbours diverge more slowly than the spurious
    neighbours of linear noise), so a one-sided upper test cannot detect it.

Pure functions, fixed seeds, no plotting. Runs natively and in Pyodide.
"""

import math
import warnings

import numpy as np
from scipy import optimize, signal, special, stats
from scipy.spatial import cKDTree

__version__ = "1.0.0"

DEFAULT_SURROGATES = 199
MAX_OBS = 5000
MIN_OBS = 500
ALPHA = 0.05

BDS_EPS = 1.5          # epsilon as a multiple of the standard deviation
LYAP_TSEP = 20         # Theiler window (samples)
LYAP_FIT = 10          # trajectory length for the divergence-curve fit
AMI_BINS = 64
AMI_MAX_LAG = 30
FNN_MAX_M = 10
FNN_THRESHOLD = 1.0    # percent


def _noop(*_args, **_kwargs):
    pass


# ======================================================================
#  Surrogates
# ======================================================================
def iaaft(x, rng, n_iter=100):
    """Iterative amplitude-adjusted Fourier transform surrogate
    (Schreiber & Schmitz 1996). Preserves the amplitude distribution exactly
    and the power spectrum approximately."""
    x = np.asarray(x, float)
    n = len(x)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    prev = None
    for _ in range(n_iter):
        S = np.fft.rfft(s)
        s = np.fft.irfft(target_amp * np.exp(1j * np.angle(S)), n=n)
        ranks = np.argsort(np.argsort(s))
        s = sorted_x[ranks]
        if prev is not None and np.array_equal(ranks, prev):
            break
        prev = ranks
    return s


def make_surrogates(x, n, method="iaaft", seed=0):
    """Return an (n, len(x)) array of surrogates.

    method="iaaft"   keeps distribution and linear autocorrelation
    method="shuffle" keeps distribution only (i.i.d. null)
    """
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(x)))
    for i in range(n):
        if method == "iaaft":
            out[i] = iaaft(x, rng)
        elif method == "shuffle":
            out[i] = rng.permutation(x)
        else:
            raise ValueError(f"unknown surrogate method: {method}")
    return out


def rank_p_value(observed, surrogates, tail="two-sided"):
    """Rank-based Monte Carlo p-value (Theiler et al. 1992).

    upper:     (1 + #{s >= obs}) / (1 + N)
    two-sided: the same count on |value - median|, with the median taken over
               the pooled observed and surrogate values so that the observed
               value and the surrogates stay exchangeable under the null.
               Smallest attainable p is 1 / (N + 1) in both cases.
    """
    s = np.asarray(surrogates, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if tail == "upper":
        return float((1 + np.sum(s >= observed)) / (1 + n))
    if tail == "lower":
        return float((1 + np.sum(s <= observed)) / (1 + n))
    med = np.median(np.append(s, observed))
    return float((1 + np.sum(np.abs(s - med) >= abs(observed - med))) / (1 + n))


# ======================================================================
#  GARCH(1,1) by maximum likelihood
# ======================================================================
def _garch_variance(eps, omega, alpha, beta, backcast):
    """sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1},
    sigma2_0 = backcast. Linear recursion, evaluated with lfilter."""
    u = np.empty_like(eps)
    u[0] = backcast
    u[1:] = omega + alpha * eps[:-1] ** 2
    return signal.lfilter([1.0], [1.0, -beta], u)


def _garch_nll(params, y, backcast):
    mu, omega, alpha, beta, nu = params
    eps = y - mu
    s2 = _garch_variance(eps, omega, alpha, beta, backcast)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    ll = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
          - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(s2)
          - (nu + 1) / 2 * np.log1p(eps ** 2 / (s2 * (nu - 2))))
    total = -np.sum(ll)
    return total if np.isfinite(total) else 1e10


def garch_filter(x):
    """Fit a constant-mean GARCH(1,1) with Student-t innovations by maximum
    likelihood and return standardised residuals (eps_t / sigma_t).

    The series is rescaled to unit variance before fitting (the dissertation
    rescaled to percent for the same reason: optimiser conditioning).
    Parameters are reported on the original scale.
    """
    x = np.asarray(x, float)
    scale = x.std()
    y = (x - x.mean()) / scale
    backcast = 1.0

    bounds = [(-1.0, 1.0), (1e-6, 10.0), (0.0, 1.0), (0.0, 1.0), (2.05, 200.0)]
    cons = [{"type": "ineq", "fun": lambda p: 0.99999 - p[2] - p[3]}]

    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.80), (0.05, 0.94), (0.15, 0.60)]:
        start = np.array([0.0, max(1 - a0 - b0, 0.01), a0, b0, 8.0])
        with warnings.catch_warnings():
            # SLSQP clips steps that overshoot the bounds; harmless
            warnings.simplefilter("ignore", RuntimeWarning)
            res = optimize.minimize(_garch_nll, start, args=(y, backcast),
                                    method="SLSQP", bounds=bounds,
                                    constraints=cons,
                                    options={"maxiter": 500, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    mu, omega, alpha, beta, nu = best.x
    eps = y - mu
    s2 = _garch_variance(eps, omega, alpha, beta, backcast)
    z = eps / np.sqrt(s2)
    return {
        "std_resid": z,
        "cond_vol": np.sqrt(s2) * scale,
        "params": {
            "mu": float(mu * scale + x.mean()),
            "omega": float(omega * scale ** 2),
            "alpha": float(alpha),
            "beta": float(beta),
            "nu": float(nu),
            "persistence": float(alpha + beta),
        },
        "loglik": float(-best.fun - len(y) * np.log(scale)),
        "converged": bool(best.success),
    }


# ======================================================================
#  BDS test (Brock, Dechert, Scheinkman & LeBaron 1996)
# ======================================================================
def bds_test(x, m_max=5, eps=BDS_EPS):
    """BDS statistics and asymptotic two-sided p-values for m = 2..m_max,
    with epsilon = eps * std(x).

    Same algorithm and conventions as statsmodels.tsa.stattools.bds
    (Kanzler 1999 variance, conditioning on the first m - 1 values), which
    it reproduces to machine precision. statsmodels itself is not used
    because it sums indicator counts in the platform integer type: in
    Pyodide (wasm32) that is int32, and the variance term overflows for
    series longer than about 1,400 observations.
    """
    x = np.asarray(x, float)
    n = len(x)
    if m_max < 2 or m_max >= n:
        raise ValueError("m_max must be in [2, len(x) - 1]")
    epsilon = eps * x.std(ddof=1)
    ind = np.abs(x[:, None] - x[None, :]) < epsilon

    def corrsum(J):
        k = len(J)   # mean over i < j; the diagonal is all True
        return (J.sum(dtype=np.int64) - k) / (k * (k - 1))

    # variance terms, from the full indicator matrix
    c1_full = corrsum(ind)
    row = ind.sum(axis=1, dtype=np.int64).astype(float)
    kk = ((row ** 2).sum() - 3 * row.sum() + 2 * n) / (n * (n - 1) * (n - 2))

    stats_, pvals = [], []
    joint = ind
    for m in range(2, m_max + 1):
        joint = joint[1:, 1:] & joint[:-1, :-1]
        c_m = corrsum(joint)
        c_1 = corrsum(ind[m - 1:, m - 1:])
        tmp = sum(kk ** (m - j) * c1_full ** (2 * j) for j in range(1, m))
        var = 4 * (kk ** m + 2 * tmp + (m - 1) ** 2 * c1_full ** (2 * m)
                   - m ** 2 * kk * c1_full ** (2 * m - 2))
        w = math.sqrt(n - m + 1) * (c_m - c_1 ** m) / math.sqrt(var)
        stats_.append(w)
        pvals.append(2 * stats.norm.sf(abs(w)))
    return {"m": list(range(2, m_max + 1)), "stat": np.array(stats_),
            "p": np.array(pvals)}


# ======================================================================
#  Phase-space reconstruction and Lyapunov exponent
# ======================================================================
def embed(x, m, tau):
    """Rows = [x_i, x_{i+tau}, ..., x_{i+(m-1)tau}]."""
    x = np.asarray(x, float)
    N = len(x) - (m - 1) * tau
    if N <= 0:
        raise ValueError("series too short for these m, tau")
    return np.column_stack([x[i * tau: i * tau + N] for i in range(m)])


def ami(x, max_lag=AMI_MAX_LAG, bins=AMI_BINS):
    """Average mutual information I(x_t; x_{t+tau}) for tau = 1..max_lag."""
    x = np.asarray(x, float)
    out = np.empty(max_lag)
    for tau in range(1, max_lag + 1):
        a, b = x[:-tau], x[tau:]
        c_ab = np.histogram2d(a, b, bins=bins)[0]
        p_ab = c_ab / c_ab.sum()
        p_a, p_b = p_ab.sum(axis=1), p_ab.sum(axis=0)
        nz = p_ab > 0
        out[tau - 1] = np.sum(p_ab[nz] * np.log(p_ab[nz] / np.outer(p_a, p_b)[nz]))
    return out


def first_local_min(v):
    """1-based lag of the first local minimum (argmin if none)."""
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] < v[i + 1]:
            return i + 1
    return int(np.argmin(v)) + 1


def fnn(x, tau, max_m=FNN_MAX_M, R_tol=15.0, A_tol=2.0):
    """Percentage of false nearest neighbours for m = 1..max_m
    (Kennel, Brown & Abarbanel 1992)."""
    x = np.asarray(x, float)
    sigma = np.std(x)
    pct = []
    for m in range(1, max_m + 1):
        Y, Y1 = embed(x, m, tau), embed(x, m + 1, tau)
        M = len(Y1)
        Ym = Y[:M]
        dist, idx = cKDTree(Ym).query(Ym, k=2)
        nnd, nn = dist[:, 1], idx[:, 1]
        valid = nnd > 0
        extra = np.abs(Y1[:, -1] - Y1[nn, -1])
        d1 = np.linalg.norm(Y1 - Y1[nn], axis=1)
        crit1 = np.zeros(M, bool)
        crit1[valid] = (extra[valid] / nnd[valid]) > R_tol
        crit2 = (d1 / sigma) > A_tol
        false = crit1 | crit2
        pct.append(100.0 * np.mean(false[valid]))
    return np.array(pct)


def choose_tau(x, max_lag=AMI_MAX_LAG):
    return first_local_min(ami(x, max_lag))


def choose_m(x, tau, max_m=FNN_MAX_M, threshold=FNN_THRESHOLD):
    f = fnn(x, tau, max_m)
    below = np.where(f < threshold)[0]
    return int(below[0] + 1) if below.size else int(np.argmin(f)) + 1


def lyapunov(x, tau=None, m=None, min_tsep=LYAP_TSEP,
             trajectory_len=LYAP_FIT, curve=False):
    """Largest Lyapunov exponent, Rosenstein et al. (1993), per sample.

    tau defaults to the first AMI minimum and m to the first dimension with
    < 1% false nearest neighbours; pass either to override.
    Returns a dict with the exponent, the parameters used, and optionally the
    mean log-divergence curve (the diagnostic: chaos shows a linear scaling
    region, noise saturates immediately).
    """
    import nolds
    x = np.asarray(x, float)
    tau_auto, m_auto = tau is None, m is None
    if tau is None:
        tau = choose_tau(x)
    if m is None:
        m = choose_m(x, tau)
    tau, m = int(tau), max(int(m), 2)
    out = {"tau": tau, "m": m, "tau_auto": tau_auto, "m_auto": m_auto,
           "min_tsep": min_tsep, "trajectory_len": trajectory_len}
    if curve:
        le, (ks, div, _poly) = nolds.lyap_r(
            x, emb_dim=m, lag=tau, min_tsep=min_tsep, tau=1,
            trajectory_len=trajectory_len, fit="poly", debug_data=True)
        out["k"] = np.asarray(ks, float)
        out["divergence"] = np.asarray(div, float)
    else:
        le = nolds.lyap_r(x, emb_dim=m, lag=tau, min_tsep=min_tsep, tau=1,
                          trajectory_len=trajectory_len, fit="poly")
    out["lambda"] = float(le)
    return out


# ======================================================================
#  Multiple testing
# ======================================================================
def holm(pvalues):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    k = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        adj[i] = running
    return adj


def _fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ======================================================================
#  Series test
# ======================================================================
def _bds_summary(stats_by_m):
    """Single pre-specified statistic for the surrogate comparison:
    the mean BDS statistic over m = 2..m_max (avoids testing each m)."""
    return float(np.mean(stats_by_m))


def _series_block(x, label, key, tau, m, n_surrogates, m_max, seed,
                  progress, step0, total):
    """BDS and Lyapunov on one series against its IAAFT surrogates."""
    bds_obs = bds_test(x, m_max)
    ly_obs = lyapunov(x, tau=tau, m=m, curve=True)
    bds_sur = np.empty(n_surrogates)
    ly_sur = np.empty(n_surrogates)
    curves = []
    rng = np.random.default_rng(seed)
    for i in range(n_surrogates):
        s = iaaft(x, rng)
        bds_sur[i] = _bds_summary(bds_test(s, m_max)["stat"])
        ly = lyapunov(s, tau=ly_obs["tau"], m=ly_obs["m"], curve=True)
        ly_sur[i] = ly["lambda"]
        curves.append(ly["divergence"])
        progress("surrogates", step0 + i + 1, total, label)

    bds_stat = _bds_summary(bds_obs["stat"])
    curves = np.vstack(curves)
    return {
        "bds": {
            "id": f"bds_{key}",
            "series": key,
            "name": f"BDS, {label}",
            "statistic": bds_stat,
            "stat_label": f"mean BDS statistic, m = 2..{m_max}",
            "p_value": rank_p_value(bds_stat, bds_sur, tail="two-sided"),
            "tail": "two-sided",
            "per_m": [{"m": mm, "stat": float(s), "p_asymptotic": float(p)}
                      for mm, s, p in zip(bds_obs["m"], bds_obs["stat"], bds_obs["p"])],
            "surrogate": bds_sur,
        },
        "lyap": {
            "id": f"lyap_{key}",
            "series": key,
            "name": f"Lyapunov, {label}",
            "statistic": ly_obs["lambda"],
            "stat_label": "largest Lyapunov exponent, per observation",
            "p_value": rank_p_value(ly_obs["lambda"], ly_sur, tail="two-sided"),
            "tail": "two-sided",
            "tau": ly_obs["tau"],
            "m": ly_obs["m"],
            "surrogate": ly_sur,
            "curve": {
                "k": ly_obs["k"],
                "observed": ly_obs["divergence"],
                "surrogate_mean": curves.mean(axis=0),
                "surrogate_lo": np.quantile(curves, 0.025, axis=0),
                "surrogate_hi": np.quantile(curves, 0.975, axis=0),
            },
        },
        "tau": ly_obs["tau"],
        "m": ly_obs["m"],
        "tau_auto": ly_obs["tau_auto"],
        "m_auto": ly_obs["m_auto"],
    }


def _direction(t):
    sur = np.asarray(t["surrogate"], float)
    return "above" if t["statistic"] > np.median(sur) else "below"


def _series_verdicts(tests, alpha):
    text = {
        "bds_raw": {
            "above": "The returns are more dependent than noise with the same "
                     "distribution and autocorrelation. In daily returns this is "
                     "usually volatility clustering.",
            "below": "The returns differ from the surrogates in their nonlinear "
                     "dependence, with a lower BDS statistic than matched noise.",
            "none": "BDS finds no dependence beyond what the surrogates already "
                    "contain.",
        },
        "bds_resid": {
            "above": "Dependence remains after filtering GARCH(1,1) volatility. "
                     "Something other than volatility clustering is present "
                     "(regime changes, asymmetric volatility, or a misspecified "
                     "variance model are the usual candidates).",
            "below": "After GARCH(1,1) filtering the residuals still differ from "
                     "their surrogates, with a lower BDS statistic than matched "
                     "noise.",
            "none": "After GARCH(1,1) filtering, BDS finds nothing the surrogates "
                    "do not also show. Any dependence in the raw series is "
                    "accounted for by conditional variance.",
        },
        "lyap_raw": {
            "above": "Nearby trajectories diverge faster than in the surrogates. "
                     "This indicates nonlinear structure but is not by itself "
                     "evidence of chaos: inspect the divergence curve.",
            "below": "Nearby trajectories diverge more slowly than in the "
                     "surrogates. Deterministic dynamics do this, but so does "
                     "volatility clustering, which places calm-period "
                     "observations close together; the residual test separates "
                     "the two.",
            "none": "The exponent sits inside the surrogate distribution. "
                    "Sensitivity to initial conditions is no different from "
                    "matched linear noise.",
        },
        "lyap_resid": {
            "above": "After GARCH filtering, divergence is still faster than in "
                     "the surrogates.",
            "below": "After GARCH filtering, divergence is still slower than in "
                     "the surrogates. Volatility clustering no longer explains "
                     "this; it is consistent with deterministic structure, or "
                     "with dependence the GARCH(1,1) model does not capture.",
            "none": "After GARCH filtering, the exponent is indistinguishable "
                    "from the surrogates. No evidence of low-dimensional "
                    "deterministic dynamics.",
        },
    }
    adj = holm([t["p_value"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = float(a)
        t["reject"] = bool(t["p_value"] < alpha)
        t["direction"] = _direction(t)
        p = _fmt_p(t["p_value"])
        if t["reject"]:
            t["verdict"] = (f"Rejects the null at {alpha:.0%} (p = {p}); observed "
                            f"value is {t['direction']} the surrogates. "
                            + text[t["id"]][t["direction"]])
        else:
            t["verdict"] = (f"Does not reject the null at {alpha:.0%} (p = {p}). "
                            + text[t["id"]]["none"])
    return adj


def series_test(returns, n_surrogates=DEFAULT_SURROGATES, tau=None, m=None,
                m_max=5, seed=0, alpha=ALPHA, progress=None):
    """BDS on raw returns and on GARCH(1,1) standardised residuals, and the
    Rosenstein Lyapunov exponent of each, all against IAAFT surrogates.
    Each test gets a rank-based p-value; the overall verdict uses
    Holm-adjusted p-values across the four tests."""
    progress = progress or _noop
    x = np.asarray(returns, float)
    total = 2 * n_surrogates

    progress("garch", 0, total, "fitting GARCH(1,1)")
    g = garch_filter(x)
    z = g["std_resid"]

    raw = _series_block(x, "raw returns", "raw", tau, m, n_surrogates,
                        m_max, seed, progress, 0, total)
    # residuals use the embedding chosen on the raw series, so the two
    # Lyapunov tests are comparable
    resid = _series_block(z, "GARCH residuals", "resid", raw["tau"], raw["m"],
                          n_surrogates, m_max, seed + 1, progress,
                          n_surrogates, total)

    tests = [raw["bds"], resid["bds"], raw["lyap"], resid["lyap"]]
    adj = _series_verdicts(tests, alpha)
    any_reject = bool(np.any(adj < alpha))
    by_id = {t["id"]: t for t in tests}
    if not any_reject:
        overall = ("No test rejects after Holm correction. The series is "
                   "statistically indistinguishable from noise with the same "
                   "distribution and linear autocorrelation.")
    elif by_id["bds_resid"]["p_holm"] < alpha or by_id["lyap_resid"]["p_holm"] < alpha:
        overall = ("Structure survives GARCH(1,1) filtering. The series contains "
                   "nonlinear dependence that volatility clustering alone does "
                   "not explain.")
    else:
        overall = ("Nonlinear dependence is present in the raw series but "
                   "disappears after GARCH(1,1) filtering: the structure is "
                   "volatility clustering, not predictable dynamics in the mean.")

    return {
        "tests": tests,
        "overall": {"reject": any_reject, "method": "Holm", "alpha": alpha,
                    "verdict": overall},
        "garch": {k: v for k, v in g.items() if k not in ("std_resid", "cond_vol")},
        "params": {"tau": raw["tau"], "m": raw["m"], "tau_auto": raw["tau_auto"],
                   "m_auto": raw["m_auto"], "bds_eps": BDS_EPS, "m_max": m_max,
                   "min_tsep": LYAP_TSEP, "trajectory_len": LYAP_FIT},
    }


# ======================================================================
#  Strategy test
# ======================================================================
EULER_GAMMA = 0.5772156649015329


def sharpe(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def deflated_sharpe(sr, T, skew, kurt, n_trials=1, sr_var=None):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014).

    sr       per-period (not annualised) Sharpe ratio
    kurt     Pearson kurtosis (3 for a normal)
    n_trials number of strategy variants tried
    sr_var   variance of the Sharpe ratio across those trials
    Returns (DSR, SR0). With n_trials = 1, SR0 = 0 and DSR equals the
    probabilistic Sharpe ratio.
    """
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(sr_var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    else:
        sr0 = 0.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    denom = max(denom, 1e-12)
    dsr = stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom))
    return float(dsr), float(sr0)


def strategy_test(returns, positions, n_surrogates=DEFAULT_SURROGATES,
                  n_trials=1, periods_per_year=252, seed=0, alpha=ALPHA,
                  progress=None):
    """Compare the strategy's Sharpe ratio with two nulls and deflate it.

    Strategy return on row t = position_t * return_t (the position is the
    exposure held over the period whose return is on the same row).

    surrogate null    the same positions applied to IAAFT surrogates of the
                      returns: markets with the same distribution and
                      autocorrelation but no other structure
    permutation null  the positions shuffled in time against the real
                      returns: the same exposures with no timing information
    """
    progress = progress or _noop
    r = np.asarray(returns, float)
    pos = np.asarray(positions, float)
    strat = pos * r
    T = len(strat)
    ann = math.sqrt(periods_per_year)
    sr_obs = sharpe(strat)

    rng_s = np.random.default_rng(seed)
    rng_p = np.random.default_rng(seed + 1)
    sr_sur = np.empty(n_surrogates)
    sr_perm = np.empty(n_surrogates)
    total = 2 * n_surrogates
    for i in range(n_surrogates):
        sr_sur[i] = sharpe(pos * iaaft(r, rng_s))
        progress("surrogates", i + 1, total, "surrogate null")
    for i in range(n_surrogates):
        sr_perm[i] = sharpe(rng_p.permutation(pos) * r)
        progress("surrogates", n_surrogates + i + 1, total, "permutation null")

    skew = float(stats.skew(strat))
    kurt = float(stats.kurtosis(strat, fisher=False))
    # cross-trial Sharpe variance: estimated from the permutation null, i.e.
    # the spread of Sharpe ratios that skill-free variants would produce
    sr_var = float(np.var(sr_perm, ddof=1))
    dsr, sr0 = deflated_sharpe(sr_obs, T, skew, kurt, n_trials, sr_var)

    def verdict(p, yes, no):
        rej = p < alpha
        head = (f"Rejects the null at {alpha:.0%} (p = {_fmt_p(p)}). " if rej else
                f"Does not reject the null at {alpha:.0%} (p = {_fmt_p(p)}). ")
        return rej, head + (yes if rej else no)

    p_sur = rank_p_value(sr_obs * ann, sr_sur * ann, "upper")
    p_perm = rank_p_value(sr_obs * ann, sr_perm * ann, "upper")
    rej_s, v_s = verdict(
        p_sur,
        "The positions earn more on the real returns than on surrogate markets "
        "with the same distribution and autocorrelation.",
        "Surrogate markets with the same distribution and autocorrelation give a "
        "Sharpe ratio this high often enough that the result is not "
        "distinguishable from luck.")
    rej_p, v_p = verdict(
        p_perm,
        "The timing of the positions matters: shuffled positions rarely do as well.",
        "Shuffling the positions in time does about as well. The result comes "
        "from average exposure, not timing.")
    p_dsr = 1 - dsr
    rej_d = p_dsr < alpha
    v_d = (f"Deflated Sharpe ratio {dsr:.3f} "
           + (f"exceeds {1 - alpha:.2f}: " if rej_d else f"is below {1 - alpha:.2f}: ")
           + (f"the Sharpe ratio is significant after adjusting for "
              f"{n_trials} trial{'s' if n_trials != 1 else ''}, skewness and "
              f"kurtosis." if rej_d else
              f"after adjusting for {n_trials} trial{'s' if n_trials != 1 else ''}, "
              f"skewness and kurtosis, the Sharpe ratio is not significant."))

    tests = [
        {"id": "sharpe_surrogate", "name": "Sharpe vs surrogate null",
         "statistic": sr_obs * ann, "stat_label": "annualised Sharpe ratio",
         "p_value": p_sur, "tail": "upper", "surrogate": sr_sur * ann,
         "reject": bool(rej_s), "verdict": v_s},
        {"id": "sharpe_permutation", "name": "Sharpe vs permutation null",
         "statistic": sr_obs * ann, "stat_label": "annualised Sharpe ratio",
         "p_value": p_perm, "tail": "upper", "surrogate": sr_perm * ann,
         "reject": bool(rej_p), "verdict": v_p},
        {"id": "deflated_sharpe", "name": "Deflated Sharpe ratio",
         "statistic": dsr, "stat_label": "DSR, probability true Sharpe > SR0",
         "p_value": float(p_dsr), "tail": "upper", "surrogate": None,
         "reject": bool(rej_d), "verdict": v_d,
         "sr0_annualised": sr0 * ann, "n_trials": int(n_trials)},
    ]
    all_reject = all(t["reject"] for t in tests)
    if all_reject:
        overall = ("All three tests reject. The Sharpe ratio is unlikely under "
                   "either null and survives deflation.")
    elif not any(t["reject"] for t in tests):
        overall = ("No test rejects. The Sharpe ratio is consistent with luck "
                   "under both nulls.")
    else:
        overall = ("Mixed result. At least one null explains the Sharpe ratio; "
                   "treat the strategy as unproven.")
    return {
        "tests": tests,
        "overall": {"reject": all_reject, "method": "all three must reject",
                    "alpha": alpha, "verdict": overall},
        "summary": {
            "sharpe_annualised": sr_obs * ann,
            "mean_annualised": float(strat.mean() * periods_per_year),
            "vol_annualised": float(strat.std(ddof=1) * ann),
            "skew": skew, "kurtosis": kurt, "T": T,
            "periods_per_year": periods_per_year,
            "mean_abs_position": float(np.mean(np.abs(pos))),
        },
    }


# ======================================================================
#  Entry point
# ======================================================================
class ValidationError(ValueError):
    pass


def _clean(obj):
    """Make the output JSON-serialisable (NaN/inf become None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def run_validation(df, mode="series", n_surrogates=DEFAULT_SURROGATES,
                   tau=None, m=None, n_trials=1, seed=0, progress=None):
    """Validate a DataFrame with columns date, return (and position for
    mode="strategy") and return a JSON-serialisable result dict."""
    import pandas as pd

    cols = {c.strip().lower(): c for c in df.columns}
    need = ["date", "return"] + (["position"] if mode == "strategy" else [])
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValidationError(f"missing column(s): {', '.join(missing)}")
    data = pd.DataFrame({c: df[cols[c]] for c in need})
    for c in need[1:]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        if data[c].isna().any():
            row = int(np.where(data[c].isna())[0][0]) + 2
            raise ValidationError(f"non-numeric or missing value in '{c}' at row {row}")
    if len(data) < MIN_OBS:
        raise ValidationError(f"{len(data)} rows; at least {MIN_OBS} are required")
    if n_surrogates < 19:
        raise ValidationError("use at least 19 surrogates (minimum p-value 0.05)")

    notes = []
    if len(data) > MAX_OBS:
        notes.append(f"Series has {len(data)} observations; only the most "
                        f"recent {MAX_OBS} were used to keep run time bounded.")
        data = data.iloc[-MAX_OBS:]
    if mode == "series" and 4 / (n_surrogates + 1) >= ALPHA:
        notes.append(f"With {n_surrogates} surrogates the smallest Holm-adjusted "
                     f"p-value is {4 / (n_surrogates + 1):.3f}, so the overall "
                     f"verdict cannot reject at 5%. Use at least 80.")

    x = data["return"].to_numpy(float)
    if x.std() == 0:
        raise ValidationError("returns are constant")

    if mode == "series":
        res = series_test(x, n_surrogates=n_surrogates, tau=tau, m=m,
                          seed=seed, progress=progress)
    elif mode == "strategy":
        res = strategy_test(x, data["position"].to_numpy(float),
                            n_surrogates=n_surrogates, n_trials=n_trials,
                            seed=seed, progress=progress)
    else:
        raise ValidationError(f"unknown mode: {mode}")

    dates = data["date"].astype(str)
    res.update({
        "mode": mode,
        "n_obs": int(len(x)),
        "start": dates.iloc[0],
        "end": dates.iloc[-1],
        "n_surrogates": int(n_surrogates),
        "seed": int(seed),
        "warnings": notes,
        "version": __version__,
    })
    (progress or _noop)("done", 1, 1, "")
    return _clean(res)
