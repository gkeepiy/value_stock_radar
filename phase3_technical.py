from __future__ import annotations

"""AI Value Stock Radar — Phase 3.

Run: python phase3_technical.py
Next: python phase4_signal.py (uses the same weighted scoring function)

Fundamental scores are preserved. Technical scoring retains the original
30/20/15/15/10 allocation (90 raw points, normalized to 100).
Williams %R is informational only. Combined Score requires valid fundamental
and technical scores, at a 5:3 ratio.
"""

import argparse
import re
import shutil
import typing as t
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PHASE2_PATTERN = "sp500_fundamental_scores_*.csv"
PHASE3_PATTERN = "sp500_technical_scores_*.csv"
PHASE4_PATTERN = "sp500_radar_signals_*.csv"
PRICE_PERIOD = "2y"
PRICE_INTERVAL = "1d"
RSI_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
WILLIAMS_PERIOD = 14
MA_SHORT, MA_LONG = 50, 200
MOMENTUM_3M_DAYS, MOMENTUM_6M_DAYS = 63, 126
HIGH_52W_DAYS = 252
SIDEWAYS_SESSIONS = 31  # More than 30 completed trading sessions.
VOLUME_PERIOD = 20
PRICE_MAX_AGE_DAYS = 7
SCORE_MODEL = "F5_T3_v2"
WEIGHTS = {"Fundamental Score": 5 / 8, "Technical Score": 3 / 8}
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def finite(value: t.Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def valid_score(value: t.Any) -> bool:
    return finite(value) and 0 <= float(value) <= 100


def utc_day(value: t.Any = None) -> pd.Timestamp:
    stamp = pd.Timestamp(value) if value is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def latest_file(pattern: str) -> Path:
    paths = list(DATA_DIR.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {DATA_DIR / pattern}")
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, p.name))


def find_latest_phase2_file() -> Path:
    return latest_file(PHASE2_PATTERN)


def normalize_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if "Ticker" not in df.columns:
        raise ValueError("Ticker 컬럼이 없습니다.")
    result = df.copy()
    result["Ticker"] = result["Ticker"].astype("string").str.strip().str.upper()
    result = result[result["Ticker"].notna() & result["Ticker"].ne("")].copy()
    if result["Ticker"].duplicated().any():
        raise ValueError("중복 Ticker가 있습니다. 중복 행을 정리하세요.")
    return result


def load_phase2_data() -> tuple[pd.DataFrame, Path]:
    path = find_latest_phase2_file()
    df = normalize_tickers(pd.read_csv(path))
    if "Fundamental Score" not in df.columns:
        raise ValueError("Phase 2에 Fundamental Score 컬럼이 없습니다.")
    if df.empty:
        raise ValueError("분석할 종목이 없습니다.")
    print(f"📂 Phase 2 데이터 로드: {path}")
    return df, path


def download_price_data(tickers: list[str]) -> pd.DataFrame:
    # Import lazily so the numerical checks can run without network/yfinance.
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("먼저 pip install numpy pandas yfinance 를 실행하세요.") from exc
    print(f"📡 {len(tickers)}개 종목: {PRICE_PERIOD} 일별 가격 다운로드")
    # end is exclusive. Exclude today's possibly unfinished daily bar.
    prices = yf.download(tickers=tickers, period=PRICE_PERIOD,
                         end=utc_day().strftime("%Y-%m-%d"), interval=PRICE_INTERVAL,
                         auto_adjust=True, group_by="ticker", threads=True,
                         progress=True)
    if prices is None or prices.empty:
        raise RuntimeError("가격 데이터를 다운로드하지 못했습니다.")
    if not isinstance(prices.columns, pd.MultiIndex) and len(tickers) != 1:
        raise ValueError("여러 종목의 가격 데이터에 종목별 열 구분이 없습니다.")
    if len(tickers) == 1:
        prices.attrs["single_ticker"] = tickers[0]
    return prices


def extract_ticker_frame(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(prices.columns, pd.MultiIndex):
        if ticker in prices.columns.get_level_values(0):
            df = prices[ticker].copy()
        elif ticker in prices.columns.get_level_values(1):
            df = prices.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if prices.attrs.get("single_ticker", ticker) != ticker:
            return pd.DataFrame()
        df = prices.copy()
    if not set(OHLCV).issubset(df.columns):
        return pd.DataFrame()
    df = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if df.index.tz is not None:
        # Preserve the exchange-local calendar date of a daily bar.
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df.loc[~df.index.duplicated(keep="last")].dropna(subset=["Close"]).sort_index()


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    # A completely flat series is neutral, not oversold.
    return rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


def calculate_stochastic(high: pd.Series, low: pd.Series,
                         close: pd.Series) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(STOCH_K_PERIOD).min()
    highest = high.rolling(STOCH_K_PERIOD).max()
    width = (highest - lowest).where(highest > lowest)
    k = 100 * (close - lowest) / width
    return k, k.rolling(STOCH_D_PERIOD).mean()


def calculate_williams_r(high: pd.Series, low: pd.Series,
                         close: pd.Series) -> pd.Series:
    highest = high.rolling(WILLIAMS_PERIOD).max()
    lowest = low.rolling(WILLIAMS_PERIOD).min()
    width = (highest - lowest).where(highest > lowest)
    return -100 * (highest - close) / width


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["Close"]
    result["MA50"] = close.rolling(MA_SHORT).mean()
    result["MA200"] = close.rolling(MA_LONG).mean()
    result["RSI14"] = calculate_rsi(close)
    result["Stoch K"], result["Stoch D"] = calculate_stochastic(
        result["High"], result["Low"], close)
    result["Williams %R"] = calculate_williams_r(result["High"], result["Low"], close)
    result["Momentum 3M"] = close / close.shift(MOMENTUM_3M_DAYS) - 1
    result["Momentum 6M"] = close / close.shift(MOMENTUM_6M_DAYS) - 1
    # Do not label a 100-day high as an actual 52-week high.
    result["52W High"] = result["High"].rolling(HIGH_52W_DAYS).max()
    result["52W Drawdown"] = close / result["52W High"] - 1
    # Preserve the supplied implementation: the 20-day average includes today.
    result["Volume 20D Avg"] = result["Volume"].rolling(VOLUME_PERIOD).mean()
    result["Volume Ratio"] = result["Volume"] / result["Volume 20D Avg"].replace(0, np.nan)
    result["Daily Return"] = close.pct_change(fill_method=None)
    return result


def crossed_above(a: pd.Series, b: pd.Series) -> bool:
    if len(a) < 2 or len(b) < 2 or not all(finite(x) for x in
                                           [a.iloc[-2], a.iloc[-1], b.iloc[-2], b.iloc[-1]]):
        return False
    return bool(a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1])


def crossed_below(a: pd.Series, b: pd.Series) -> bool:
    return crossed_above(b, a)


def classify_trend(price: float, ma50: float, ma200: float, momentum_6m: float) -> str:
    if not all(finite(v) for v in [price, ma50, ma200]):
        return "UNKNOWN"
    if price > ma50 > ma200 and (not finite(momentum_6m) or momentum_6m > 0):
        return "STRONG_UPTREND"
    if price > ma200 and ma50 > ma200:
        return "UPTREND"
    if price < ma50 < ma200:
        return "STRONG_DOWNTREND"
    return "DOWNTREND" if price < ma200 else "NEUTRAL"


def classify_rsi(rsi: float) -> str:
    if not finite(rsi):
        return "UNKNOWN"
    if rsi >= 75:
        return "EXTREME_OVERBOUGHT"
    if rsi >= 70:
        return "OVERBOUGHT"
    if rsi <= 25:
        return "EXTREME_OVERSOLD"
    if rsi <= 30:
        return "OVERSOLD"
    if rsi >= 55:
        return "BULLISH"
    return "BEARISH" if rsi <= 45 else "NEUTRAL"


def classify_stochastic(k: float, d: float) -> str:
    if not all(finite(v) for v in [k, d]):
        return "UNKNOWN"
    if k <= 20 and d <= 20:
        return "OVERSOLD"
    if k >= 80 and d >= 80:
        return "OVERBOUGHT"
    return "BULLISH" if k > d else "BEARISH" if k < d else "NEUTRAL"


def classify_williams_r(wr: float) -> str:
    if not finite(wr):
        return "UNKNOWN"
    if wr <= -80:
        return "OVERSOLD"
    if wr >= -20:
        return "OVERBOUGHT"
    return "BULLISH" if wr > -50 else "BEARISH" if wr < -50 else "NEUTRAL"


def classify_drawdown(drawdown: float) -> str:
    if not finite(drawdown):
        return "UNKNOWN"
    for boundary, state in [(-.40, "DEEP_CRASH"), (-.30, "DEEP_DRAWDOWN"),
                            (-.20, "MAJOR_DRAWDOWN"), (-.10, "CORRECTION")]:
        if drawdown <= boundary:
            return state
    return "NEAR_52W_HIGH" if drawdown >= -.03 else "NORMAL"


# Original scoring rules, including all thresholds, retained for valid data.
# Missing indicators now return NaN instead of an artificial zero score.
def trend_score(price: float, ma50: float, ma200: float) -> float:
    if not all(finite(v) for v in [price, ma50, ma200]):
        return np.nan
    return float(12 * (price > ma200) + 8 * (price > ma50) + 10 * (ma50 > ma200))


def momentum_score(momentum_3m: float, momentum_6m: float) -> float:
    if not all(finite(v) for v in [momentum_3m, momentum_6m]):
        return np.nan
    def points(value: float, upper: float, middle: float, bottom: float) -> float:
        if value >= upper:
            return 10.0
        if value >= middle:
            return 8.0
        if value >= 0:
            return 6.0
        return 3.0 if value >= bottom else 0.0
    return points(momentum_3m, .15, .05, -.10) + points(momentum_6m, .25, .10, -.15)


def rsi_score(rsi: float) -> float:
    if not finite(rsi):
        return np.nan
    if 50 <= rsi <= 65:
        return 15.0
    if 40 <= rsi < 50:
        return 12.0
    if 30 <= rsi < 40 or 65 < rsi <= 70:
        return 10.0
    if rsi < 30:
        return 8.0
    return 6.0 if 70 < rsi <= 75 else 3.0


def stochastic_score(k: float, d: float, bullish_cross: bool, bearish_cross: bool) -> float:
    if not all(finite(v) for v in [k, d]):
        return np.nan
    score = 10.0 if bullish_cross else 0.0 if bearish_cross else 7.0 if k > d else 4.0
    score += 5.0 if k <= 30 and k > d else 0.0 if k >= 80 and k < d else 3.0
    return min(score, 15.0)


def volume_score(volume_ratio: float, daily_return: float) -> float:
    if not all(finite(v) for v in [volume_ratio, daily_return]) or volume_ratio < 0:
        return np.nan
    if volume_ratio >= 1.5 and daily_return > 0:
        return 10.0
    if volume_ratio >= 1.2 and daily_return > 0:
        return 8.0
    return 6.0 if volume_ratio >= .8 else 4.0


def calculate_total_technical_score(trend_points: float, momentum_points: float,
                                    rsi_points: float, stochastic_points: float,
                                    volume_points: float) -> float:
    values = [trend_points, momentum_points, rsi_points, stochastic_points, volume_points]
    if not all(finite(v) for v in values):
        return np.nan
    return float(np.clip(sum(values) / 90 * 100, 0, 100))


def technical_grade(score: float) -> str:
    if not finite(score):
        return "N/A"
    for threshold, grade in [(85, "A"), (75, "B+"), (65, "B"), (55, "C+"), (45, "C"), (35, "D")]:
        if score >= threshold:
            return grade
    return "F"


def technical_state(price: float, ma50: float, ma200: float, rsi: float,
                    momentum_6m: float, drawdown: float, bullish_cross: bool,
                    bearish_cross: bool) -> str:
    if finite(rsi) and rsi >= 75 and finite(drawdown) and drawdown >= -.05:
        return "OVERHEATED"
    if bullish_cross and finite(drawdown) and drawdown <= -.10:
        return "RECOVERY"
    if finite(rsi) and rsi <= 30:
        return "OVERSOLD"
    if all(finite(v) for v in [price, ma50, ma200, momentum_6m]) and price > ma50 > ma200 and momentum_6m > 0:
        return "STRONG_UPTREND"
    if finite(price) and finite(ma200) and price > ma200:
        return "UPTREND"
    if bearish_cross and finite(price) and finite(ma50) and price < ma50:
        return "WEAKENING"
    if all(finite(v) for v in [price, ma50, ma200]) and price < ma50 < ma200:
        return "STRONG_DOWNTREND"
    return "DOWNTREND" if finite(price) and finite(ma200) and price < ma200 else "NEUTRAL"


def add_combined_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    missing = pd.Series("", index=result.index, dtype="string")
    valid = pd.Series(True, index=result.index)
    total = pd.Series(0.0, index=result.index)
    for column, weight in WEIGHTS.items():
        values = pd.to_numeric(result.get(column, pd.Series(np.nan, index=result.index)), errors="coerce")
        ok = values.map(valid_score)
        if column == "Fundamental Score" and "Data Quality" in result:
            # Phase 2 can score a failed Phase 1 download as zero; it is missing data.
            ok &= ~result["Data Quality"].astype("string").str.upper().eq("FAILED").fillna(False)
        quality = {"Technical Score": "Technical Data Quality"}.get(column)
        if quality is not None:
            ok &= result.get(quality, pd.Series("MISSING", index=result.index)).eq("OK")
        valid &= ok
        missing = missing.mask(~ok, missing + column + "; ")
        result[column.replace(" Score", " Contribution")] = (values * weight).where(ok)
        total += values.fillna(0) * weight
    result["Combined Score"] = total.where(valid).round(2)
    result["Combined Data Quality"] = np.where(valid, "OK", "INCOMPLETE")
    result["Combined Missing Inputs"] = missing.str.rstrip("; ")
    result["Combined Rank"] = result["Combined Score"].rank(ascending=False, method="min").astype("Int64")
    result["Score Model"] = SCORE_MODEL
    return result


def empty_technical_output(ticker: str) -> dict[str, t.Any]:
    numeric = ["Price", "Daily Return", "MA50", "MA200", "Price vs MA50", "Price vs MA200", "RSI14",
               "Stoch K", "Stoch D", "Williams %R", "Oscillator Identity Error", "Momentum 3M", "Momentum 6M",
               "52W High", "52W Drawdown", "Sideways 31D Range", "Volume", "Volume 20D Avg", "Volume Ratio", "Trend Score",
               "Momentum Score", "RSI Score", "Stochastic Score", "Volume Score", "Technical Score"]
    output: dict[str, t.Any] = dict.fromkeys(numeric, np.nan)
    output.update({"Ticker": ticker, "Technical Data Quality": "FAILED", "Technical Grade": "N/A",
                   "Technical Failed Reason": "", "Technical As Of": "", "Stoch Bullish Cross": False,
                   "Stoch Bearish Cross": False, "Williams Oversold Exit": False, "Williams Overbought Exit": False,
                   "Oscillator Confirmation": "UNKNOWN", "Technical State": "UNKNOWN", "Trend State": "UNKNOWN",
                   "RSI State": "UNKNOWN", "Stochastic State": "UNKNOWN", "Williams State": "UNKNOWN",
                   "Drawdown State": "UNKNOWN"})
    return output


def analyze_stock(ticker: str, prices: pd.DataFrame, as_of: t.Any = None) -> dict[str, t.Any]:
    output = empty_technical_output(ticker)
    try:
        cutoff = utc_day(as_of)
        df = extract_ticker_frame(prices, ticker)
        if df.empty:
            raise ValueError("유효한 가격 데이터가 없습니다.")
        df = df.loc[df.index < cutoff]  # Reproducible as-of cutoff; no unfinished bar.
        if df.empty:
            raise ValueError("유효한 가격 데이터가 없습니다.")
        output["Technical As Of"] = df.index[-1].strftime("%Y-%m-%d")
        if len(df) < MA_LONG:
            output.update({"Technical Data Quality": "PARTIAL", "Technical Failed Reason": "MA200 계산에 200거래일이 필요합니다."})
            return output
        if (cutoff - df.index[-1]).days > PRICE_MAX_AGE_DAYS:
            output.update({"Technical Data Quality": "STALE", "Technical Failed Reason": "최근 가격이 7일 이상 오래되었습니다."})
            return output
        recent = df.tail(HIGH_52W_DAYS)
        tolerance = recent["Close"].abs() * 1e-8
        bad = ((recent["Close"] <= 0) | (recent["Volume"] < 0)
               | (recent["High"] < recent["Low"])
               | (recent["Close"] > recent["High"] + tolerance)
               | (recent["Close"] < recent["Low"] - tolerance))
        if bad.any():
            output.update({"Technical Data Quality": "INVALID", "Technical Failed Reason": "최근 가격/거래량 데이터가 유효하지 않습니다."})
            return output
        # The entire trailing 31 completed sessions must fit inside a 10% close range.
        if len(df) >= SIDEWAYS_SESSIONS:
            window = df["Close"].tail(SIDEWAYS_SESSIONS)
            output["Sideways 31D Range"] = float(window.max() / window.min() - 1)
        df = add_indicators(df)
        last = df.iloc[-1]
        price, ma50, ma200 = last["Close"], last["MA50"], last["MA200"]
        rsi, k, d, wr = last["RSI14"], last["Stoch K"], last["Stoch D"], last["Williams %R"]
        m3, m6, drawdown = last["Momentum 3M"], last["Momentum 6M"], last["52W Drawdown"]
        bull = crossed_above(df["Stoch K"], df["Stoch D"]) and finite(k) and k <= 30
        bear = crossed_below(df["Stoch K"], df["Stoch D"]) and finite(k) and k >= 70
        scores = [trend_score(price, ma50, ma200), momentum_score(m3, m6), rsi_score(rsi),
                  stochastic_score(k, d, bull, bear), volume_score(last["Volume Ratio"], last["Daily Return"])]
        total = calculate_total_technical_score(*scores)
        for column in output.keys() & set(last.index):
            output[column] = last[column]
        output.update(dict(zip(["Trend Score", "Momentum Score", "RSI Score", "Stochastic Score", "Volume Score"], scores)))
        output.update({"Price": price, "Price vs MA50": price / ma50 - 1 if ma50 != 0 else np.nan,
                       "Price vs MA200": price / ma200 - 1 if ma200 != 0 else np.nan,
                       "Stoch Bullish Cross": bool(bull), "Stoch Bearish Cross": bool(bear),
                       "Williams Oversold Exit": crossed_above(df["Williams %R"], pd.Series(-80., index=df.index)),
                       "Williams Overbought Exit": crossed_below(df["Williams %R"], pd.Series(-20., index=df.index)),
                       "Trend State": classify_trend(price, ma50, ma200, m6), "RSI State": classify_rsi(rsi),
                       "Stochastic State": classify_stochastic(k, d), "Williams State": classify_williams_r(wr),
                       "Drawdown State": classify_drawdown(drawdown), "Technical Score": round(total, 2),
                       "Technical Grade": technical_grade(total),
                       "Technical State": technical_state(price, ma50, ma200, rsi, m6, drawdown, bull, bear),
                       "Technical Data Quality": "OK" if finite(total) else "PARTIAL",
                       "Technical Failed Reason": "" if finite(total) else "기술지표 계산에 필요한 데이터 일부가 없습니다."})
        if finite(k) and finite(wr):
            if STOCH_K_PERIOD == WILLIAMS_PERIOD:
                error = abs(wr - (k - 100))
                output["Oscillator Identity Error"] = error
                output["Oscillator Confirmation"] = "REDUNDANT_SAME_WINDOW" if error < 1e-8 else "CHECK_DATA"
            else:
                output["Oscillator Confirmation"] = "DIFFERENT_WINDOWS_NOT_INDEPENDENT"
    except Exception as exc:
        output["Technical Failed Reason"] = f"{type(exc).__name__}: {exc}"
    return output


def run_technical_engine(fundamentals: pd.DataFrame, prices: pd.DataFrame,
                         as_of: t.Any = None) -> pd.DataFrame:
    fundamentals = normalize_tickers(fundamentals)
    if not isinstance(prices.columns, pd.MultiIndex) and len(fundamentals) > 1:
        raise ValueError("복수 종목 분석에는 ticker가 포함된 MultiIndex 가격 데이터가 필요합니다.")
    rows = []
    for index, (_, fundamental) in enumerate(fundamentals.iterrows(), start=1):
        technical = analyze_stock(str(fundamental["Ticker"]), prices, as_of=as_of)
        row = fundamental.to_dict()
        row.update(technical)
        rows.append(row)
        if index % 25 == 0 or index == len(fundamentals):
            print(f"Technical 분석: {index}/{len(fundamentals)}")
    result = add_technical_rank(pd.DataFrame(rows))
    return add_combined_scores(result)


def add_technical_rank(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Technical Rank"] = result["Technical Score"].rank(ascending=False, method="min").astype("Int64")
    return result


def save_phase3(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"sp500_technical_scores_{datetime.now():%Y-%m-%d}.csv"
    df.sort_values(["Combined Score", "Fundamental Score", "Technical Score"],
                   ascending=False, na_position="last").to_csv(path, index=False, encoding="utf-8-sig")
    return path


def reweight_phase4(phase4_path: Path, phase3_path: Path) -> Path:
    """Explicit post-Phase-4 step. Preserve signals and fundamentals; back up CSV."""
    p4 = normalize_tickers(pd.read_csv(phase4_path))
    p3 = normalize_tickers(pd.read_csv(phase3_path))
    date3 = re.search(r"_(\d{4}-\d{2}-\d{2})\.csv$", phase3_path.name)
    date4 = re.search(r"_(\d{4}-\d{2}-\d{2})\.csv$", phase4_path.name)
    if not date3 or not date4 or date3[1] != date4[1]:
        raise ValueError("같은 YYYY-MM-DD 날짜의 Phase 3/4 결과 파일이 필요합니다.")
    if p3.empty or "Score Model" not in p3 or not p3["Score Model"].eq(SCORE_MODEL).all():
        raise ValueError("새 버전 Phase 3 결과 파일이 필요합니다.")
    missing = set(p4["Ticker"]) - set(p3["Ticker"])
    if missing:
        raise ValueError(f"Phase 3에 없는 종목이 있습니다: {sorted(missing)[:5]}")
    indexed = p3.set_index("Ticker").reindex(p4["Ticker"])
    old_f = pd.to_numeric(p4["Fundamental Score"], errors="coerce").to_numpy(dtype=float)
    new_f = pd.to_numeric(indexed["Fundamental Score"], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(old_f, new_f, equal_nan=True, atol=1e-8, rtol=0):
        raise ValueError("두 파일의 펀더멘털 점수가 다릅니다. 같은 입력으로 Phase 3와 Phase 4를 다시 실행하세요.")
    generated = set(empty_technical_output("_schema")) - {"Ticker"}
    generated |= {"Technical Rank"}
    for column in generated & set(indexed.columns):
        p4[column] = indexed[column].to_numpy()
    p4 = add_combined_scores(p4)
    p4 = p4.sort_values(["Combined Score", "Fundamental Score", "Technical Score"], ascending=False, na_position="last")
    if "Display Rank" in p4:
        p4["Display Rank"] = p4["Combined Rank"].to_numpy()
    if "Recommendation Rank" in p4:
        p4["Recommendation Rank"] = p4["Combined Rank"].to_numpy()
    backup = phase4_path.with_name(phase4_path.name + f".{uuid4().hex[:8]}.bak")
    temporary = phase4_path.with_name(phase4_path.name + ".tmp")
    shutil.copy2(phase4_path, backup)
    try:
        p4.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(phase4_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"✅ Phase 4 점수 재적용: {phase4_path}\n원본 백업: {backup}")
    print("참고: 기존 Signal/Confidence는 유지되며 새 가중치로 재판정하지 않습니다.")
    return phase4_path


def print_summary(df: pd.DataFrame) -> None:
    print(f"총 종목: {len(df)} | 평균 Technical: {df['Technical Score'].mean():.2f}")
    complete = df["Combined Data Quality"].eq("OK")
    print(f"종합점수 계산 가능: {int(complete.sum())}/{len(df)} (펀더멘털 62.5% / 기술 37.5%)")
    if not complete.all():
        print("자료 부족으로 종합점수가 비어 있는 항목:")
        print(df.loc[~complete, "Combined Missing Inputs"].value_counts().to_string())


def print_top_stocks(df: pd.DataFrame, top_n: int = 20) -> None:
    ready = df[df["Combined Data Quality"].eq("OK")]
    columns = ["Ticker", "Fundamental Score", "Technical Score", "Combined Score"]
    if ready.empty:
        print("종합 추천 순위 없음: 펀더멘털·기술 점수를 확인하세요.")
        return
    print(ready.sort_values("Combined Score", ascending=False).head(top_n)[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reweight-phase4", metavar="CSV_OR_latest",
                        help="Phase 4 실행 뒤 종합점수만 재적용합니다. 원본 CSV를 백업 후 교체합니다.")
    parser.add_argument("--phase3-file", type=Path, help="재가중할 때 사용할 Phase 3 CSV (생략: 최신 파일)")
    args = parser.parse_args()
    if args.reweight_phase4:
        path = latest_file(PHASE4_PATTERN) if args.reweight_phase4 == "latest" else Path(args.reweight_phase4)
        reweight_phase4(path, args.phase3_file or latest_file(PHASE3_PATTERN))
        return
    fundamentals, source = load_phase2_data()
    prices = download_price_data(fundamentals["Ticker"].astype(str).tolist())
    result = run_technical_engine(fundamentals, prices)
    path = save_phase3(result)
    print_summary(result)
    print_top_stocks(result)
    print(f"📂 Source: {source}\n💾 저장 완료: {path}")
    print("다음 실행: python phase4_signal.py (별도 재가중 명령 불필요)")


if __name__ == "__main__":
    main()
