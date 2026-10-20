# Surrogate data test for return series

A static web page that tests whether a financial return series, or a trading strategy's returns, contains structure that noise with the same distribution and linear autocorrelation would not produce.

All computation runs in the browser with [Pyodide](https://pyodide.org). There is no backend and the data is never uploaded.

## Files

| File | Purpose |
|---|---|
| `index.html` | The page. HTML, CSS and JS inline; Python runs in a Web Worker. |
| `validate.py` | The validation library. Fetched by the page at run time; also runs natively. |
| `sample_data.csv` | 1,500 business days of synthetic GARCH(1,1) returns with a 20-day momentum position. Offered for download on the page as `example_strategy.csv`. |
| `sample_series.csv` | The same returns without the position column, offered for download as `example_series.csv`. |
| `tests/test_validate.py` | Native pytest suite. |
| `study/run_study.py` | Runs the series test on 14 market series (needs `yfinance`; run locally, not by the website). |
| `study/results.json` | The study's test statistics, shown on the page. No prices are stored. |

## Input

A CSV (comma, semicolon, tab or pipe separated) or Excel workbook (.xlsx, .xls, .ods) with a date column and either a return column or a price column, plus a position column for the strategy test. The reader:
- finds the header below any title or preamble lines, and the first sheet with data in a workbook;
- maps columns by name (for example `Date`/`Trade Date`, `Return`/`Change %`, `Adj Close`/`Close`/`PX_LAST`, `Position`/`Weight`/`Signal`), or by content when the date column has an unusual name;
- converts prices to log returns;
- reads decimal commas, percentages, thousands separators and accounting negatives such as `(0.5)`;
- recognises ISO, day-first and month-first dates and Excel date numbers;
- sorts newest-first files.

It reports everything it inferred. Hard requirements: at least 500 returns, no missing values or repeated dates, and no date gaps longer than ten typical intervals. Series longer than 5,000 observations are truncated to the most recent 5,000, with a warning. Excel files are parsed with SheetJS, loaded only when a workbook is chosen.

## Method

**Series test.** Four tests, each against 199 IAAFT surrogates:

| Test | Statistic |
|---|---|
| BDS, returns | mean BDS statistic over m = 2..5, ε = 1.5σ, on AR(p)-prewhitened returns |
| BDS, GARCH residuals | as above, on AR(p)-GARCH(1,1) standardised residuals |
| Lyapunov, raw returns | Rosenstein largest exponent of the unfiltered returns |
| Lyapunov, GARCH residuals | as above, on the residuals |

Linear dependence is part of the null, so it is removed before BDS and GARCH with an AR(p) model, p chosen by BIC up to 10 (Hsieh, 1991).

Each test reports a two-sided rank p-value. The overall verdict applies a Holm correction across the four.

**Strategy test.** The annualised Sharpe ratio of `position × return` is compared with two nulls:
- the same positions applied to IAAFT surrogates of the returns;
- the positions permuted in time.

The test also reports the deflated Sharpe ratio (Bailey & López de Prado, 2014) for a user-supplied number of strategy variants tried.

### Relation to the dissertation pipeline

`validate.py` is refactored from the dissertation code: `BDS.py`, `surrogatedata.py` and `CODE/LS.py`. The following are kept:
- IAAFT, at 100 iterations with a rank-convergence stop;
- GARCH(1,1) with Student-t innovations (now on AR(p) residuals; `surrogatedata.py` also used an AR mean);
- BDS with ε = 1.5σ;
- τ from the first minimum of average mutual information (64 bins);
- m from false nearest neighbours (R_tol = 15, A_tol = 2, 1% threshold);
- the Rosenstein exponent via `nolds.lyap_r`, with Theiler window 20 and fit length 10;
- the rank p-value (1 + #)/(1 + N).

Deliberate changes:

1. **GARCH without `arch`.** `arch` is not available in Pyodide, so GARCH is estimated by maximum likelihood with `scipy.optimize` (SLSQP). On the S&P 500 series used in the dissertation, the estimates match `arch` to four decimals (α = 0.1025, β = 0.8922, ν = 6.10). The standardised residuals correlate with `arch`'s at 0.9999996.
2. **BDS reimplemented.** `statsmodels.tsa.stattools.bds` loads in Pyodide but gives wrong answers there. It sums indicator counts in the platform integer, which is int32 under WebAssembly, and its variance term overflows above about 1,400 observations. On the sample data the browser returned BDS ≈ 0.3–0.8 instead of 8–13. `validate.py` ports the same algorithm with 64-bit counts; natively it reproduces statsmodels exactly (tested) and runs about 7× faster.
3. **Lyapunov fit.** The line fit uses `fit="poly"` (least squares) rather than nolds' default RANSAC, which needs scikit-learn and is not deterministic.
4. **Two-sided Lyapunov test.** The dissertation used a one-sided test, on the hypothesis that chaos gives a *larger* exponent than the linear null. On the Lorenz positive control the opposite holds. The observed exponent (0.042 per sample) lies *below* every IAAFT surrogate (about 0.15). True neighbours on an attractor separate more slowly than the spurious neighbours of matched linear noise. A one-sided upper test returns p = 1.0 on Lorenz, so it cannot detect chaos. The test here is two-sided, using distances from the pooled median, which keeps the minimum p at 1/(N+1). The same applies to the raw S&P 500 result in `surrogatedata.py`: an exponent below the surrogates would have been read as "no evidence" under the one-sided test.

5. **AR prewhitening (added in 1.1.0).** Stress testing showed that without it the series test rejected a purely linear AR(1) process (φ = 0.3) in 35% of runs, mostly through BDS on the GARCH residuals: a constant-mean GARCH passes linear dependence straight through, and BDS is very sensitive to it. With prewhitening the rate is 10% (2 of 20 runs, consistent with 5% at this run count).

## Validation results

Measured with `python -m pytest tests -s` (199 surrogates, seeded).

| Process | Runs | Result |
|---|---|---|
| Gaussian white noise, n = 500 | 50 | Overall false-positive rate (Holm, 5%): **2%**. Per test: BDS raw 4%, BDS residuals 2%, Lyapunov raw 4%, Lyapunov residuals 6%. |
| GARCH(1,1), ω = 0.05, α = 0.10, β = 0.85, n = 1,000 | 20 | Raw BDS rejects in **100%**. Residual BDS rejects in **0%**. |
| Lorenz x-component, dt = 0.02, n = 1,000 | 1 | λ = +0.042 per sample. All four tests reject with p = 0.005, the minimum possible. |

The per-test false-positive rates are all within sampling error of the nominal 5%. With 50 runs, a 95% interval around 5% is roughly 0–11%.

The Lorenz exponent per time unit (2.1) overstates the true value (0.906): the series is short and m = 3 is chosen automatically. The test requires only a positive exponent and a rejection.

Additional tests cover:
- IAAFT preserving the amplitude distribution and spectrum;
- recovery of GARCH parameters;
- BDS agreeing with statsmodels;
- Holm adjustment;
- the deflated Sharpe ratio reducing to the probabilistic Sharpe ratio for one trial;
- strategy power and size;
- input validation;
- truncation at 5,000 observations.

## Results on market data

`study/run_study.py` runs the series test on 14 daily series to 16 October 2026 (most recent 1,500 log returns each, 199 surrogates): five equity indices, three FX pairs, Bitcoin and Ethereum, gold, Brent, US 10-year Treasury futures, and listed private equity (Invesco PSP, a daily stand-in for private markets).

- BDS on prewhitened returns rejects for **14 of 14** series (p = 0.005, the minimum).
- BDS on AR-GARCH(1,1) residuals rejects for **0 of 14** (smallest p: Bitcoin, 0.075).
- The raw Lyapunov test rejects only for the S&P 500 (p = 0.03), about what chance predicts across 14 tests, and not after Holm correction. No residual Lyapunov test rejects.

**Conclusion.** In every asset class tested, the nonlinear dependence in daily returns is volatility clustering. There is no evidence of low-dimensional deterministic chaos, and nothing remains for these tests to find once linear autocorrelation and GARCH volatility are modelled. A structural break would produce the same signature, so the study cannot separate smooth volatility clustering from regime shifts.

## Stress tests

Additional hard cases, 20 runs each at n = 1,000 (overall Holm rejection rate at 5%):

| Process | Overall rejection | Reading |
|---|---|---|
| i.i.d. Student-t(3), very fat tails | 5% | Correct size |
| AR(1), φ = 0.3, linear only | 35% before prewhitening, 10% after | Fixed in 1.1.0 |
| 40% zero returns (illiquid asset), 40 runs | 0% | Correct; ties do not distort the surrogates |
| i.i.d. noise whose volatility doubles halfway | 100%, all through raw BDS; residual BDS 0% | Known limitation: a structural break looks like volatility clustering |

## Known weaknesses

- **Non-stationarity.** A single variance break produces the same signature as volatility clustering (raw BDS rejects, residuals do not). The tool cannot distinguish a regime change from GARCH dynamics.
- **Fixed embedding for surrogates.** τ and m are chosen on the observed series and reused for its surrogates; the Lyapunov test inherits any instability in that choice.
- **IAAFT is approximate.** The spectrum is matched only approximately. Prewhitening removes most of the effect on BDS, but strongly autocorrelated series can still be slightly oversized.
- **Strategy test takes positions as given.** It cannot re-run the strategy rule on surrogate markets, so a rule that adapts to the data is tested only through its realised positions; the deflated Sharpe ratio depends on an honest count of variants tried.
- **Cross-trial Sharpe variance** for the deflated Sharpe ratio is estimated from the permutation null, not from the variants actually tried.
- **Run time** grows roughly as n^1.7; 5,000 observations take about 10 minutes in a browser, longer on phones.
- **AI explanations** are generated text and can be wrong; the numbers on the page are authoritative. Only a summary of statistics is sent, but it does leave the browser when the feature is used.

## Running locally

```bash
pip install numpy scipy pandas nolds pytest statsmodels   # statsmodels only for the cross-check test
python -m pytest tests -s
python -m http.server 8000                                  # then open http://localhost:8000
```

The page must be served over HTTP. Opened as a `file://` URL, the browser blocks the worker's fetch of `validate.py`.

## AI explanation (optional)

After a run, visitors can ask Claude (Anthropic) to explain the results. They paste their own Anthropic API key; it is kept in memory (or in the browser's local storage if they tick "remember") and sent only to `api.anthropic.com`. The request contains a summary of test statistics, p-values and settings, which the page shows before sending, and never the returns. The model is Claude Opus 5.5 at medium effort, streamed, with Anthropic's server-side refusal fallback enabled. Follow-up questions are supported, and the explanation is included in the downloaded report. The Anthropic SDK is loaded from jsDelivr only when the feature is used.

## Runtime

For 1,500 observations and 199 surrogates:

| Mode | Desktop Chrome | Native |
|---|---|---|
| Series test | about 70 s | about 25 s |
| Strategy test | a few seconds | a few seconds |

Cost grows roughly as n^1.7, so 5,000 observations take about 10 minutes in the browser. The first visit also downloads about 30 MB from the Pyodide CDN (cached afterwards).

Statistics are identical natively and in the browser. Surrogate p-values can differ by a rank or two, because floating-point differences in the FFT shift individual IAAFT surrogates.

## Sample data

`sample_data.csv` was generated with seed 20240101:
- **Returns:** GARCH(1,1) with ω = 2e-6, α = 0.08, β = 0.90 and Student-t(6) innovations scaled to unit variance, plus a constant mean of 3e-4.
- **Position:** the sign of the previous 20 days' summed return.

The returns have no predictable mean, so the strategy test should not reject. The series test should reject on raw BDS but not on the residuals.

## Dependencies

- In the browser: Pyodide 0.27.7 (numpy, scipy, pandas), plus nolds installed from PyPI with micropip.
- Loaded only when needed: SheetJS 0.20.3 (Excel files, from cdn.sheetjs.com) and the Anthropic TypeScript SDK 0.131.0 (AI explanations, from jsDelivr).
- Study only, on your machine: yfinance.
- Fonts: Source Serif 4, IBM Plex Sans and IBM Plex Mono, from Google Fonts.
- No chart or UI libraries.

## References

- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Brock, W. A., Dechert, W. D., Scheinkman, J. A. & LeBaron, B. (1996). A test for independence based on the correlation dimension. *Econometric Reviews*, 15(3), 197–235.
- Hsieh, D. A. (1991). Chaos and nonlinear dynamics: application to financial markets. *Journal of Finance*, 46(5), 1839–1877.
- Kennel, M. B., Brown, R. & Abarbanel, H. D. I. (1992). Determining embedding dimension for phase-space reconstruction. *Physical Review A*, 45, 3403–3411.
- Rosenstein, M. T., Collins, J. J. & De Luca, C. J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D*, 65, 117–134.
- Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Physical Review Letters*, 77(4), 635–638.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77–94.

Statistical diagnostics only. Not investment advice.
