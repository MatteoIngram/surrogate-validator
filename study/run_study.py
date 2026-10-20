"""
Real-data study: run the series test on major asset classes.

Downloads daily prices with yfinance (not used by the website), computes log
returns of the adjusted close, takes the most recent 1,500 observations
before END, and runs validate.run_validation in series mode with 199 IAAFT
surrogates.

END defaults to today (UTC), exclusive, so only completed daily closes are
used. Set STUDY_END=YYYY-MM-DD to reproduce an earlier run. Runs weekly via
.github/workflows/weekly-study.yml.

Only test statistics are written to study/results.json. Raw prices are not
stored or published. If any asset fails to download or validate, the script
exits with an error and leaves the existing results.json untouched.

    pip install -r study/requirements.txt
    python study/run_study.py
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import validate as v  # noqa: E402

END = os.environ.get("STUDY_END") or time.strftime("%Y-%m-%d", time.gmtime())  # exclusive
N_OBS = 1500
N_SURROGATES = 199

ASSETS = [
    # (asset class, name, yfinance ticker, note)
    ("Equity indices", "S&P 500", "^GSPC", ""),
    ("Equity indices", "FTSE 100", "^FTSE", ""),
    ("Equity indices", "Euro Stoxx 50", "^STOXX50E", ""),
    ("Equity indices", "Nikkei 225", "^N225", ""),
    ("Equity indices", "MSCI Emerging Markets", "EEM", "iShares ETF used as proxy for the index"),
    ("FX", "EUR/USD", "EURUSD=X", ""),
    ("FX", "GBP/USD", "GBPUSD=X", ""),
    ("FX", "USD/JPY", "JPY=X", ""),
    ("Crypto", "Bitcoin", "BTC-USD", "trades seven days a week"),
    ("Crypto", "Ethereum", "ETH-USD", "trades seven days a week"),
    ("Commodities and rates", "Gold", "GC=F", "front-month future"),
    ("Commodities and rates", "Brent crude", "BZ=F", "front-month future; WTI excluded (negative price in April 2020)"),
    ("Commodities and rates", "US 10-year Treasury", "ZN=F", "front-month T-note future"),
    ("Private markets", "Listed private equity", "PSP",
     "Invesco Global Listed Private Equity ETF; a daily proxy, since private "
     "fund returns are quarterly and appraisal-smoothed"),
]


def load_returns(ticker, attempts=3):
    import yfinance as yf
    for k in range(attempts):
        try:
            # start well before the window, so period/end interplay cannot
            # silently shorten the series
            px = yf.download(ticker, start="2015-01-01", end=END,
                             auto_adjust=True, progress=False)["Close"]
            px = px.squeeze("columns").dropna()
            px = px[px > 0]
            r = np.log(px).diff().dropna()
            if len(r) >= N_OBS:
                return r.iloc[-N_OBS:]
            raise ValueError(f"{ticker}: only {len(r)} returns")
        except Exception as e:  # network errors, empty frames, rate limits
            if k == attempts - 1:
                raise RuntimeError(f"{ticker}: download failed ({e})") from e
            time.sleep(10 * (k + 1))


def run_one(args):
    spec, r = args
    cls, name, ticker, note = spec
    df = pd.DataFrame({"date": r.index.strftime("%Y-%m-%d"), "return": r.values})
    t0 = time.time()
    res = v.run_validation(df, "series", n_surrogates=N_SURROGATES)
    keep = ("id", "name", "statistic", "p_value", "p_holm", "reject", "direction")
    tests = []
    for t in res["tests"]:
        row = {k: t[k] for k in keep}
        s = np.asarray(t["surrogate"], float)
        row["surrogate_median"] = float(np.median(s))
        row["surrogate_q025"] = float(np.quantile(s, 0.025))
        row["surrogate_q975"] = float(np.quantile(s, 0.975))
        tests.append(row)
    return {
        "asset_class": cls, "name": name, "ticker": ticker, "note": note,
        "start": res["start"], "end": res["end"], "n_obs": res["n_obs"],
        "zero_return_share": float(np.mean(r.values == 0)),
        "params": {k: res["params"][k] for k in ("tau", "m")},
        "garch": res["garch"]["params"],
        "overall": res["overall"]["verdict"],
        "overall_reject": res["overall"]["reject"],
        "tests": tests,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    # download serially (gentler on Yahoo), test in parallel
    data = {a[2]: load_returns(a[2]) for a in ASSETS}
    stale = [t for t, r in data.items()
             if (pd.Timestamp(END) - r.index[-1]).days > 10]
    if stale:
        raise SystemExit(f"stale data, not writing results: {stale}")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rows = list(ex.map(run_one, [(a, data[a[2]]) for a in ASSETS]))
    out = {
        "generated": time.strftime("%Y-%m-%d", time.gmtime()),
        "end": max(a["end"] for a in rows),
        "n_obs": N_OBS,
        "n_surrogates": N_SURROGATES,
        "validate_version": v.__version__,
        "source": "Yahoo Finance via yfinance; daily adjusted closes, log returns",
        "assets": rows,
    }
    path = os.path.join(ROOT, "study", "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for a in rows:
        p = {t["id"]: t["p_value"] for t in a["tests"]}
        print(f"{a['name']:24s} {a['start']}..{a['end']} n={a['n_obs']} "
              f"tau={a['params']['tau']} m={a['params']['m']} "
              f"bds_raw={p['bds_raw']:.3f} bds_res={p['bds_resid']:.3f} "
              f"ly_raw={p['lyap_raw']:.3f} ly_res={p['lyap_resid']:.3f} "
              f"zeros={a['zero_return_share']:.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
