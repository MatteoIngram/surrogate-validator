"""Native tests for validate.py.  Run from the repository root:

    python -m pytest tests -s

-s prints the measured rejection rates quoted in the README.
The Monte Carlo tests run in parallel processes; the full suite takes a few
minutes on a laptop.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import validate as v  # noqa: E402

ALPHA = 0.05


# ----------------------------------------------------------------------
# Data generators
# ----------------------------------------------------------------------
def lorenz_x(n=1000, dt=0.02, burn=1000):
    def f(_t, u):
        x, y, z = u
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]
    t = np.arange(n + burn) * dt
    sol = solve_ivp(f, [0, t[-1]], [1, 1, 1], t_eval=t, rtol=1e-9, atol=1e-9)
    return sol.y[0][burn:]


def garch_sim(n, seed, omega=0.05, alpha=0.10, beta=0.85, burn=500):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    e = np.empty(n + burn)
    s2 = omega / (1 - alpha - beta)
    for t in range(n + burn):
        e[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * e[t] ** 2 + beta * s2
    return e[burn:]


def _p_values(res):
    return {t["id"]: t["p_value"] for t in res["tests"]}


def _noise_run(seed):
    x = np.random.default_rng(1000 + seed).standard_normal(500)
    res = v.series_test(x, n_surrogates=199, seed=seed)
    return res["overall"]["reject"], _p_values(res)


def _garch_run(seed):
    res = v.series_test(garch_sim(1000, seed), n_surrogates=199, seed=seed)
    return _p_values(res)


def _pool_map(fn, seeds):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        return list(ex.map(fn, seeds))


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def test_iaaft_preserves_distribution_and_spectrum():
    x = garch_sim(1000, 0)
    s = v.make_surrogates(x, 5, seed=1)
    for row in s:
        assert np.allclose(np.sort(row), np.sort(x))
        ax, as_ = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(row))
        assert np.corrcoef(ax, as_)[0, 1] > 0.95
    # fixed seed is reproducible; shuffle keeps the distribution too
    assert np.array_equal(s, v.make_surrogates(x, 5, seed=1))
    sh = v.make_surrogates(x, 2, method="shuffle")
    assert np.allclose(np.sort(sh[0]), np.sort(x))


def test_rank_p_value_bounds():
    sur = np.arange(199, dtype=float)
    assert v.rank_p_value(1e6, sur, "upper") == pytest.approx(1 / 200)
    assert v.rank_p_value(1e6, sur, "two-sided") == pytest.approx(1 / 200)
    assert v.rank_p_value(-1e6, sur, "two-sided") == pytest.approx(1 / 200)
    assert v.rank_p_value(99.0, sur, "two-sided") == pytest.approx(1.0)


def test_garch_recovers_parameters():
    x = garch_sim(5000, 7, omega=0.05, alpha=0.10, beta=0.85)
    g = v.garch_filter(x)
    p = g["params"]
    assert g["converged"]
    assert abs(p["alpha"] - 0.10) < 0.04
    assert abs(p["beta"] - 0.85) < 0.06
    assert abs(np.std(g["std_resid"]) - 1) < 0.05


def test_bds_shapes():
    out = v.bds_test(np.random.default_rng(0).standard_normal(600), m_max=5)
    assert out["m"] == [2, 3, 4, 5]
    assert len(out["stat"]) == len(out["p"]) == 4


def test_bds_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    for x in (garch_sim(1500, 2), np.random.default_rng(4).standard_normal(700)):
        ours = v.bds_test(x, m_max=5)
        stat, p = sm.bds(x, max_dim=5, distance=1.5)
        assert np.allclose(ours["stat"], stat, rtol=1e-10, atol=1e-12)
        assert np.allclose(ours["p"], p, rtol=1e-10, atol=1e-12)


def test_bds_counts_do_not_overflow_int32():
    # regression: statsmodels sums indicator counts in the platform int,
    # which is int32 in Pyodide; this term exceeds 2**31 here
    x = garch_sim(2000, 2)
    ind = np.abs(x[:, None] - x) < 1.5 * x.std(ddof=1)
    rows = ind.sum(1).astype(np.int64)
    assert (rows ** 2).sum() > 2 ** 31
    assert np.all(np.abs(v.bds_test(x)["stat"]) > 2)   # GARCH: strongly dependent


def test_holm():
    adj = v.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_deflated_sharpe_single_trial_is_psr():
    dsr, sr0 = v.deflated_sharpe(0.1, 1000, 0.0, 3.0, n_trials=1)
    assert sr0 == 0.0
    # normal returns: z = SR sqrt(T-1) / sqrt(1 + SR^2 / 2)
    from scipy.stats import norm
    assert dsr == pytest.approx(norm.cdf(0.1 * np.sqrt(999) / np.sqrt(1.005)))
    dsr_many, sr0_many = v.deflated_sharpe(0.1, 1000, 0.0, 3.0, 100, 0.001)
    assert sr0_many > 0 and dsr_many < dsr


# ----------------------------------------------------------------------
# Required behaviour
# ----------------------------------------------------------------------
def test_lorenz_positive_exponent_and_rejects():
    x = lorenz_x()
    ly = v.lyapunov(x)
    assert ly["lambda"] > 0
    res = v.series_test(x, n_surrogates=199)
    p = _p_values(res)
    print(f"\nLorenz: lambda = {ly['lambda']:+.4f} per sample "
          f"({ly['lambda'] / 0.02:+.3f} per time unit), p = {p}")
    assert res["overall"]["reject"]
    assert p["lyap_raw"] < ALPHA


def test_white_noise_does_not_reject():
    runs = _pool_map(_noise_run, range(50))
    overall = np.array([r[0] for r in runs])
    per_test = {k: np.mean([r[1][k] < ALPHA for r in runs]) for k in runs[0][1]}
    print(f"\nWhite noise, 50 runs: overall false-positive rate "
          f"{overall.mean():.2f}; per test {per_test}")
    assert (~overall).mean() >= 0.90


def test_garch_rejects_raw_not_residuals():
    runs = _pool_map(_garch_run, range(20))
    raw = np.mean([r["bds_raw"] < ALPHA for r in runs])
    resid = np.mean([r["bds_resid"] < ALPHA for r in runs])
    print(f"\nGARCH(1,1), 20 runs: raw BDS rejection rate {raw:.2f}, "
          f"residual BDS rejection rate {resid:.2f}")
    assert raw >= 0.90
    assert resid <= 0.10


# ----------------------------------------------------------------------
# Strategy mode
# ----------------------------------------------------------------------
def test_strategy_detects_timing_and_ignores_noise():
    rng = np.random.default_rng(3)
    r = garch_sim(1000, 3) / 100
    # informative: a noisy signal correlated with the next return
    signal = r + rng.standard_normal(len(r)) * r.std() * 2
    good = v.strategy_test(r, np.sign(signal), n_surrogates=99)
    assert all(t["reject"] for t in good["tests"])
    # uninformative: random positions
    bad = v.strategy_test(r, rng.choice([-1.0, 1.0], len(r)), n_surrogates=99)
    assert not bad["tests"][0]["reject"]
    assert not bad["tests"][1]["reject"]


# ----------------------------------------------------------------------
# Entry point and input validation
# ----------------------------------------------------------------------
def _df(n, strategy=False):
    d = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n).astype(str),
                      "return": garch_sim(n, 1) / 100})
    if strategy:
        d["position"] = 1.0
    return d


def test_run_validation_is_json_serialisable():
    out = v.run_validation(_df(600), "series", n_surrogates=19)
    json.dumps(out)
    assert out["n_obs"] == 600 and len(out["tests"]) == 4
    assert any("Holm" in w for w in out["warnings"])
    s = v.run_validation(_df(600, True), "strategy", n_surrogates=19)
    json.dumps(s)
    assert len(s["tests"]) == 3


@pytest.mark.parametrize("df,mode,msg", [
    (_df(600).drop(columns="return"), "series", "missing column"),
    (_df(600), "strategy", "missing column"),
    (_df(499), "series", "at least 500"),
    (_df(600).assign(**{"return": ["x"] + [0.01] * 599}), "series", "non-numeric"),
])
def test_run_validation_rejects_bad_input(df, mode, msg):
    with pytest.raises(v.ValidationError, match=msg):
        v.run_validation(df, mode, n_surrogates=19)


def test_long_series_is_capped(monkeypatch):
    monkeypatch.setattr(v, "MAX_OBS", 700)
    out = v.run_validation(_df(800, True), "strategy", n_surrogates=19)
    assert out["n_obs"] == 700
    assert any("most recent 700" in w for w in out["warnings"])


def test_sample_data_file():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == ["date", "return", "position"]
    assert 1400 <= len(df) <= 1600
    assert df.notna().all().all()
