from __future__ import annotations

import typing as t
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from phase3_technical import add_combined_scores, normalize_tickers


# ============================================================
# 0) CONFIG
# ============================================================

DATA_DIR = Path("data")

PHASE3_PATTERN = (
    "sp500_technical_scores_*.csv"
)


# ============================================================
# The only signal: quality company, deep drawdown, and a 31-session base.
BASE_FUNDAMENTAL_MIN = 70.0
BASE_DRAWDOWN_MAX = -0.30
BASE_RANGE_MAX = 0.10


# 1) LOAD PHASE 3
# ============================================================

def find_latest_phase3_file() -> Path:

    files = list(
        DATA_DIR.glob(
            PHASE3_PATTERN
        )
    )

    if not files:

        raise FileNotFoundError(
            "Phase 3 결과 파일을 찾을 수 없습니다.\n"
            "data/sp500_technical_scores_YYYY-MM-DD.csv "
            "파일이 있는지 확인하세요."
        )

    return max(
        files,
        key=lambda p: p.stat().st_mtime,
    )


def load_phase3_data() -> tuple[
    pd.DataFrame,
    Path,
]:

    path = (
        find_latest_phase3_file()
    )

    print(
        f"📂 Phase 3 데이터 로드: {path}"
    )

    df = pd.read_csv(
        path
    )

    if "Ticker" not in df.columns:

        raise ValueError(
            "Phase 3 데이터에 Ticker 컬럼이 없습니다."
        )

    df[
        "Ticker"
    ] = (
        df[
            "Ticker"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return (
        df,
        path,
    )


# ============================================================
# 2) SAFE HELPERS
# ============================================================

def to_float(
    value: t.Any,
) -> float | None:

    if value is None:
        return None

    try:

        x = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if not np.isfinite(
        x
    ):

        return None

    return x


def to_bool(
    value: t.Any,
) -> bool:

    if isinstance(
        value,
        bool,
    ):

        return value

    if value is None:

        return False

    try:

        if pd.isna(
            value
        ):

            return False

    except Exception:

        pass

    if isinstance(
        value,
        str,
    ):

        return (
            value.strip().lower()
            in {
                "true",
                "1",
                "yes",
                "y",
            }
        )

    return bool(
        value
    )


# ============================================================
# 3) COMBINED SCORE
# ============================================================

def calculate_combined_score(row: pd.Series) -> float:
    """Use the shared F50/T30/C15/S5 calculation."""
    return float(add_combined_scores(pd.DataFrame([row]))["Combined Score"].iloc[0])


# ============================================================
# 4) DRAWDOWN BASE

def signal_drawdown_base(row: pd.Series) -> bool:
    """Fundamental >=70, >=30% below 52-week high, 31 sessions within 10%."""
    fundamental = to_float(row.get("Fundamental Score"))
    drawdown = to_float(row.get("52W Drawdown"))
    sideways_range = to_float(row.get("Sideways 31D Range"))
    return (
        row.get("Technical Data Quality") == "OK"
        and fundamental is not None and fundamental >= BASE_FUNDAMENTAL_MIN
        and drawdown is not None and drawdown <= BASE_DRAWDOWN_MAX
        and sideways_range is not None and 0 <= sideways_range <= BASE_RANGE_MAX
    )


# 9) RISK FLAGS
# ============================================================

def detect_risk_flags(
    row: pd.Series,
) -> list[str]:

    flags = []

    price = to_float(
        row.get(
            "Price"
        )
    )

    ma200 = to_float(
        row.get(
            "MA200"
        )
    )

    rsi = to_float(
        row.get(
            "RSI14"
        )
    )

    momentum_6m = to_float(
        row.get(
            "Momentum 6M"
        )
    )

    drawdown = to_float(
        row.get(
            "52W Drawdown"
        )
    )

    bearish_cross = to_bool(
        row.get(
            "Stoch Bearish Cross"
        )
    )

    if (
        price is not None
        and ma200 is not None
        and price < ma200
    ):

        flags.append(
            "BELOW_MA200"
        )

    if (
        rsi is not None
        and rsi >= 75
    ):

        flags.append(
            "RSI_OVERHEATED"
        )

    if (
        momentum_6m is not None
        and momentum_6m <= -0.20
    ):

        flags.append(
            "WEAK_6M_MOMENTUM"
        )

    if (
        drawdown is not None
        and drawdown <= -0.40
    ):

        flags.append(
            "DEEP_PRICE_COLLAPSE"
        )

    if bearish_cross:

        flags.append(
            "STOCH_BEARISH_CROSS"
        )

    return flags


# ============================================================
# 11) SIGNAL CONFIDENCE

def determine_confidence(row: pd.Series, signals: list[str]) -> str:
    if not signals:
        return "NONE"
    if row.get("Combined Data Quality") != "OK":
        return "UNAVAILABLE"
    fundamental = to_float(row.get("Fundamental Score"))
    technical = to_float(row.get("Technical Score"))
    combined = to_float(row.get("Combined Score"))
    if fundamental is None or technical is None or combined is None:
        return "UNAVAILABLE"
    if fundamental >= 85 and technical >= 70 and combined >= 80:
        return "HIGH"
    if fundamental >= 75:
        return "MEDIUM"
    return "LOW"


# 12) SIGNAL ENGINE

def evaluate_signals(row: pd.Series) -> dict[str, t.Any]:
    drawdown_base = signal_drawdown_base(row)
    signals = ["DRAWDOWN_BASE"] if drawdown_base else []
    risk_flags = detect_risk_flags(row)
    return {
        "Signal Drawdown Base": drawdown_base,
        "Signals": "|".join(signals),
        "Signal Count": len(signals),
        "Primary Signal": "DRAWDOWN_BASE" if drawdown_base else "NONE",
        "Signal Confidence": determine_confidence(row, signals),
        "Risk Flags": "|".join(risk_flags),
        "Risk Flag Count": len(risk_flags),
    }


# 13) RUN SIGNAL ENGINE
# ============================================================

def run_signal_engine(df: pd.DataFrame) -> pd.DataFrame:
    result = add_combined_scores(normalize_tickers(df))
    print("🚦 Signal Engine 실행...")
    columns = list(evaluate_signals(pd.Series(dtype=object)))
    signal_rows = [evaluate_signals(row) for _, row in result.iterrows()]
    signal_df = pd.DataFrame(signal_rows, index=result.index, columns=columns)
    result = result.drop(columns=columns, errors="ignore")
    return pd.concat([result, signal_df], axis=1)


# ============================================================
# 14) DISPLAY RANK
# ============================================================

def add_display_rank(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Display Rank"] = result["Combined Rank"].astype("Int64")
    active = result["Primary Signal"].ne("NONE") & result["Combined Data Quality"].eq("OK")
    result["Signal Rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[active, "Signal Rank"] = result.loc[active, "Combined Score"].rank(
        ascending=False, method="min").astype("Int64")
    return result


# ============================================================
# 15) RADAR CATEGORY
# ============================================================

def radar_category(row: pd.Series) -> str:
    return "DRAWDOWN_BASE" if row.get("Primary Signal") == "DRAWDOWN_BASE" else "NONE"


def add_radar_category(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "Radar Category"
    ] = result.apply(
        radar_category,
        axis=1,
    )

    return result


# ============================================================
# 16) COLUMN ORDER
# ============================================================

def reorder_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    priority = [
        "Display Rank", "Combined Rank",
        "Combined Data Quality", "Combined Missing Inputs", "Score Model",
        "Williams %R", "Williams State", "Oscillator Confirmation",

        "Signal Rank",

        "Ticker",
        "Short Name",
        "Sector",
        "Industry",

        "Combined Score",

        "Fundamental Score",
        "Technical Score",

        "Fundamental Grade",
        "Technical Grade",

        "Primary Signal",
        "Signals",

        "Radar Category",

        "Signal Confidence",
        "Signal Count",

        "Risk Flags",
        "Risk Flag Count",

        "Signal Drawdown Base",

        "Value Score",
        "Quality Score",
        "Growth Score",
        "Stability Score",

        "Price",
        "MA50",
        "MA200",

        "Price vs MA50",
        "Price vs MA200",

        "RSI14",

        "Stoch K",
        "Stoch D",

        "Stoch Bullish Cross",
        "Stoch Bearish Cross",

        "Momentum 3M",
        "Momentum 6M",

        "52W High",
        "52W Drawdown",
        "Sideways 31D Range",

        "Volume",
        "Volume 20D Avg",
        "Volume Ratio",

        "Technical State",
        "Trend State",
        "RSI State",
        "Stochastic State",
        "Drawdown State",

        "Forward PER",
        "PBR",
        "PSR",
        "EV/EBITDA",
        "FCF Yield",

        "ROE",
        "ROA",

        "Operating Margin",
        "Profit Margin",

        "Revenue CAGR",
        "Revenue Growth",
        "Earnings Growth",

        "Debt/Equity",

        "Free Cash Flow",
        "Operating Cash Flow",

        "Data Quality",
        "Technical Data Quality",

        "Failed Reason",
        "Technical Failed Reason",
    ]

    existing = [

        column

        for column in priority

        if column in df.columns

    ]

    remaining = [

        column

        for column in df.columns

        if column not in existing

    ]

    return df[
        existing
        + remaining
    ].copy()


# ============================================================
# 17) SAVE PHASE 4
# ============================================================

def save_phase4(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"sp500_radar_signals_{datetime.now():%Y-%m-%d}.csv"
    # Score ordering is independent of signal presence.
    output = df.sort_values(["Combined Score", "Fundamental Score", "Technical Score"],
                            ascending=False, na_position="last")
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ============================================================
# 18) PRINT SIGNAL SUMMARY
# ============================================================

def print_signal_summary(
    df: pd.DataFrame,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "SIGNAL ENGINE SUMMARY"
    )

    print(
        "=" * 80
    )

    active = df[
        df[
            "Primary Signal"
        ]
        != "NONE"
    ]

    print(
        f"전체 종목:"
        f" {len(df)}"
    )

    print(
        f"Signal 발생:"
        f" {len(active)}"
    )

    print()

    signal_columns = {"DRAWDOWN BASE": "Signal Drawdown Base"}

    for label, column in (
        signal_columns.items()
    ):

        if column in df.columns:

            count = (
                df[column]
                .fillna(False)
                .astype(bool)
                .sum()
            )

            print(
                f"{label:<25}"
                f"{count}"
            )

    print()


# ============================================================
# 19) PRINT TOP SIGNALS
# ============================================================

def print_top_signals(
    df: pd.DataFrame,
    top_n: int = 30,
) -> None:

    active = df[
        df[
            "Primary Signal"
        ]
        != "NONE"
    ].copy()

    active = active[active["Combined Data Quality"].eq("OK")]

    if active.empty:

        print(
            "현재 Signal 발생 종목이 없습니다."
        )

        return

    active = active.sort_values(

        by=[
            "Combined Score",
            "Fundamental Score",
            "Technical Score",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )

    columns = [

        "Signal Rank",

        "Ticker",

        "Short Name",

        "Combined Score",

        "Fundamental Score",

        "Technical Score",

        "Primary Signal",

        "Signal Confidence",

        "52W Drawdown",

        "RSI14",

        "Momentum 6M",

        "Risk Flags",
    ]

    columns = [

        column

        for column in columns

        if column in active.columns

    ]

    print()
    print(
        "=" * 150
    )

    print(
        f"TOP {top_n} RADAR SIGNALS"
    )

    print(
        "=" * 150
    )

    print(

        active[
            columns
        ]

        .head(
            top_n
        )

        .to_string(
            index=False
        )

    )

    print()


# ============================================================
# 20) MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=============================================="
    )

    print(
        "        AI VALUE STOCK RADAR"
    )

    print(
        "        PHASE 4 - SIGNAL ENGINE"
    )

    print(
        "=============================================="
    )

    print()

    # --------------------------------------------------------
    # Load Phase 3
    # --------------------------------------------------------

    df, source_path = (
        load_phase3_data()
    )

    print(
        f"✅ 분석 대상:"
        f" {len(df)}개"
    )

    # --------------------------------------------------------
    # Signal Engine
    # --------------------------------------------------------

    result = (
        run_signal_engine(
            df
        )
    )

    # --------------------------------------------------------
    # Signal ranking
    # --------------------------------------------------------

    result = (
        add_display_rank(
            result
        )
    )

    # --------------------------------------------------------
    # Radar category
    # --------------------------------------------------------

    result = (
        add_radar_category(
            result
        )
    )

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    result = (
        reorder_columns(
            result
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        save_phase4(
            result
        )
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_signal_summary(
        result
    )

    print_top_signals(
        result,
        top_n=30,
    )

    print(
        f"📂 Source:"
        f" {source_path}"
    )

    print(
        f"💾 저장 완료:"
        f" {output_path}"
    )

    print()

    print(
        "✅ PHASE 4 COMPLETE"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
