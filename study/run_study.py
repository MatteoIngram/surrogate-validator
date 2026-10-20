"""
Real-data study: run the series test on major asset classes.

Downloads daily prices with yfinance (run locally; not used by the website),
computes log returns of the adjusted close, takes the most recent 1,500
observations up to END, and runs validate.run_validation in series mode
with 199 IAAFT surrogates.

Only test statistics are written to study/results.json. Raw prices are not
stored or published.

    pip install yfinance
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

END = "2026-10-17"          # exclusive: last close used is Friday 16 October 2026
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


def load_returns(ticker):
    import yfinance as yf
    px = yf.download(ticker, end=END, period="max", auto_adjust=True,
                     progress=False)["Close"]
    px = px.squeeze("columns").dropna()
    px = px[px > 0]
    r = np.log(px).diff().dropna()
    return r.iloc[-N_OBS:]


def run_one(spec):
    cls, name, ticker, note = spec
    r = load_returns(ticker)
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
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rows = list(ex.map(run_one, ASSETS))
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "end": "2026-10-16",
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
