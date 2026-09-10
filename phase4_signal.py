from __future__ import annotations

import typing as t
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0) CONFIG
# ============================================================

DATA_DIR = Path("data")

PHASE3_PATTERN = (
    "sp500_technical_scores_*.csv"
)


# ============================================================
# SIGNAL THRESHOLDS
# ============================================================

# Fundamental
STRONG_FUNDAMENTAL = 80.0
GOOD_FUNDAMENTAL = 75.0

# Phase 2 component maximums:
# Value     35
# Quality   30
# Growth    20
# Stability 15

STRONG_VALUE = 27.0
STRONG_QUALITY = 23.0

# Technical
STRONG_TECHNICAL = 70.0
RECOVERY_TECHNICAL = 55.0

# Drawdown
FALLEN_ANGEL_DRAWDOWN = -0.25
DEEP_VALUE_DRAWDOWN = -0.20

# RSI recovery zone
RECOVERY_RSI_MIN = 35.0
RECOVERY_RSI_MAX = 55.0


# ============================================================
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

def calculate_combined_score(
    row: pd.Series,
) -> float:

    fundamental = to_float(
        row.get(
            "Fundamental Score"
        )
    )

    technical = to_float(
        row.get(
            "Technical Score"
        )
    )

    if (
        fundamental is None
        or technical is None
    ):

        return 0.0

    score = (
        fundamental
        + technical
    ) / 2

    return round(
        score,
        2,
    )


# ============================================================
# 4) QUALITY VALUE
# ============================================================

def signal_quality_value(
    row: pd.Series,
) -> bool:

    fundamental = to_float(
        row.get(
            "Fundamental Score"
        )
    )

    value = to_float(
        row.get(
            "Value Score"
        )
    )

    quality = to_float(
        row.get(
            "Quality Score"
        )
    )

    if any(
        x is None
        for x in [
            fundamental,
            value,
            quality,
        ]
    ):

        return False

    return (
        fundamental
        >= STRONG_FUNDAMENTAL

        and

        value
        >= STRONG_VALUE

        and

        quality
        >= STRONG_QUALITY
    )


# ============================================================
# 5) FALLEN ANGEL
# ============================================================

def signal_fallen_angel(
    row: pd.Series,
) -> bool:

    fundamental = to_float(
        row.get(
            "Fundamental Score"
        )
    )

    drawdown = to_float(
        row.get(
            "52W Drawdown"
        )
    )

    if (
        fundamental is None
        or drawdown is None
    ):

        return False

    return (
        fundamental
        >= STRONG_FUNDAMENTAL

        and

        drawdown
        <= FALLEN_ANGEL_DRAWDOWN
    )


# ============================================================
# 6) RECOVERY
# ============================================================

def signal_recovery(
    row: pd.Series,
) -> bool:

    fundamental = to_float(
        row.get(
            "Fundamental Score"
        )
    )

    technical = to_float(
        row.get(
            "Technical Score"
        )
    )

    price = to_float(
        row.get(
            "Price"
        )
    )

    ma50 = to_float(
        row.get(
            "MA50"
        )
    )

    rsi = to_float(
        row.get(
            "RSI14"
        )
    )

    drawdown = to_float(
        row.get(
            "52W Drawdown"
        )
    )

    bullish_cross = to_bool(
        row.get(
            "Stoch Bullish Cross"
        )
    )

    technical_state = str(
        row.get(
            "Technical State",
            "",
        )
    ).upper()

    if any(
        x is None
        for x in [
            fundamental,
            technical,
            price,
            ma50,
        ]
    ):

        return False

    # 반드시 어느 정도 좋은 기업
    if (
        fundamental
        < GOOD_FUNDAMENTAL
    ):

        return False

    # 너무 약한 기술 상태 제외
    if (
        technical
        < RECOVERY_TECHNICAL
    ):

        return False

    # 조정을 받은 적 없는 종목은
    # recovery로 보지 않음
    if (
        drawdown is not None
        and drawdown > -0.10
    ):

        return False

    price_recovery = (
        price > ma50
    )

    rsi_recovery = (
        rsi is not None
        and
        RECOVERY_RSI_MIN
        <= rsi
        <= RECOVERY_RSI_MAX
    )

    trigger = (
        bullish_cross
        or
        rsi_recovery
        or
        technical_state == "RECOVERY"
    )

    return (
        price_recovery
        and trigger
    )


# ============================================================
# 7) DEEP VALUE REVERSAL
# ============================================================

def signal_deep_value_reversal(
    row: pd.Series,
) -> bool:

    fundamental = to_float(
        row.get(
            "Fundamental Score"
        )
    )

    value = to_float(
        row.get(
            "Value Score"
        )
    )

    quality = to_float(
        row.get(
            "Quality Score"
        )
    )

    technical = to_float(
        row.get(
            "Technical Score"
        )
    )

    drawdown = to_float(
        row.get(
            "52W Drawdown"
        )
    )

    rsi = to_float(
        row.get(
            "RSI14"
        )
    )

    bullish_cross = to_bool(
        row.get(
            "Stoch Bullish Cross"
        )
    )

    technical_state = str(
        row.get(
            "Technical State",
            "",
        )
    ).upper()

    if any(
        x is None
        for x in [
            fundamental,
            value,
            quality,
            technical,
            drawdown,
        ]
    ):

        return False

    fundamentally_strong = (

        fundamental
        >= STRONG_FUNDAMENTAL

        and

        value
        >= STRONG_VALUE

        and

        quality
        >= STRONG_QUALITY
    )

    deep_discount = (

        drawdown
        <= DEEP_VALUE_DRAWDOWN
    )

    reversal_confirmation = (

        bullish_cross

        or

        technical_state
        == "RECOVERY"

        or

        (
            rsi is not None
            and
            35 <= rsi <= 55
            and
            technical >= 60
        )
    )

    return (

        fundamentally_strong

        and

        deep_discount

        and

        reversal_confirmation
    )


# ============================================================
# 8) QUALITY MOMENTUM
# ============================================================

def signal_quality_momentum(
    row: pd.Series,
) -> bool:

    fundamental = to_float(
        row.get(
            "Fundamental Score"
        )
    )

    quality = to_float(
        row.get(
            "Quality Score"
        )
    )

    technical = to_float(
        row.get(
            "Technical Score"
        )
    )

    price = to_float(
        row.get(
            "Price"
        )
    )

    ma50 = to_float(
        row.get(
            "MA50"
        )
    )

    ma200 = to_float(
        row.get(
            "MA200"
        )
    )

    momentum_6m = to_float(
        row.get(
            "Momentum 6M"
        )
    )

    if any(
        x is None
        for x in [
            fundamental,
            quality,
            technical,
            price,
            ma50,
            ma200,
            momentum_6m,
        ]
    ):

        return False

    return (

        fundamental
        >= STRONG_FUNDAMENTAL

        and

        quality
        >= STRONG_QUALITY

        and

        technical
        >= STRONG_TECHNICAL

        and

        price
        > ma50
        > ma200

        and

        momentum_6m
        > 0
    )


# ============================================================
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
# 10) PRIMARY SIGNAL
# ============================================================

SIGNAL_PRIORITY = {

    "DEEP_VALUE_REVERSAL":
        5,

    "QUALITY_VALUE":
        4,

    "RECOVERY":
        3,

    "QUALITY_MOMENTUM":
        2,

    "FALLEN_ANGEL":
        1,
}


def determine_primary_signal(
    signals: list[str],
) -> str:

    if not signals:

        return "NONE"

    return max(
        signals,
        key=lambda signal:
        SIGNAL_PRIORITY.get(
            signal,
            0,
        ),
    )


# ============================================================
# 11) SIGNAL CONFIDENCE
# ============================================================

def determine_confidence(
    row: pd.Series,
    signals: list[str],
) -> str:

    if not signals:

        return "NONE"

    fundamental = to_float(
        row.get(
            "Fundamental Score"
        )
    )

    technical = to_float(
        row.get(
            "Technical Score"
        )
    )

    combined = to_float(
        row.get(
            "Combined Score"
        )
    )

    data_quality = str(
        row.get(
            "Data Quality",
            "",
        )
    ).upper()

    technical_quality = str(
        row.get(
            "Technical Data Quality",
            "",
        )
    ).upper()

    if (
        data_quality == "FAILED"
        or
        technical_quality == "FAILED"
    ):

        return "LOW"

    if (
        fundamental is not None
        and technical is not None
        and combined is not None
        and fundamental >= 85
        and technical >= 70
        and combined >= 80
    ):

        return "HIGH"

    if (
        fundamental is not None
        and fundamental >= 75
    ):

        return "MEDIUM"

    return "LOW"


# ============================================================
# 12) SIGNAL ENGINE
# ============================================================

def evaluate_signals(
    row: pd.Series,
) -> dict[str, t.Any]:

    signals = []

    quality_value = (
        signal_quality_value(
            row
        )
    )

    fallen_angel = (
        signal_fallen_angel(
            row
        )
    )

    recovery = (
        signal_recovery(
            row
        )
    )

    deep_value_reversal = (
        signal_deep_value_reversal(
            row
        )
    )

    quality_momentum = (
        signal_quality_momentum(
            row
        )
    )

    if quality_value:

        signals.append(
            "QUALITY_VALUE"
        )

    if fallen_angel:

        signals.append(
            "FALLEN_ANGEL"
        )

    if recovery:

        signals.append(
            "RECOVERY"
        )

    if deep_value_reversal:

        signals.append(
            "DEEP_VALUE_REVERSAL"
        )

    if quality_momentum:

        signals.append(
            "QUALITY_MOMENTUM"
        )

    risk_flags = (
        detect_risk_flags(
            row
        )
    )

    primary_signal = (
        determine_primary_signal(
            signals
        )
    )

    confidence = (
        determine_confidence(
            row,
            signals,
        )
    )

    return {

        "Signal Quality Value":
            quality_value,

        "Signal Fallen Angel":
            fallen_angel,

        "Signal Recovery":
            recovery,

        "Signal Deep Value Reversal":
            deep_value_reversal,

        "Signal Quality Momentum":
            quality_momentum,

        "Signals":
            "|".join(
                signals
            ),

        "Signal Count":
            len(
                signals
            ),

        "Primary Signal":
            primary_signal,

        "Signal Confidence":
            confidence,

        "Risk Flags":
            "|".join(
                risk_flags
            ),

        "Risk Flag Count":
            len(
                risk_flags
            ),
    }


# ============================================================
# 13) RUN SIGNAL ENGINE
# ============================================================

def run_signal_engine(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    print()
    print(
        "🚦 Signal Engine 실행..."
    )

    # --------------------------------------------------------
    # Combined Score 먼저 생성
    # --------------------------------------------------------

    result[
        "Combined Score"
    ] = result.apply(
        calculate_combined_score,
        axis=1,
    )

    signal_rows = []

    total = len(
        result
    )

    for index, (_, row) in enumerate(
        result.iterrows(),
        start=1,
    ):

        evaluation = (
            evaluate_signals(
                row
            )
        )

        signal_rows.append(
            evaluation
        )

        if (
            index % 50 == 0
            or index == total
        ):

            print(
                f"Signal 분석: "
                f"{index}/{total}"
            )

    signal_df = pd.DataFrame(
        signal_rows,
        index=result.index,
    )

    result = pd.concat(
        [
            result,
            signal_df,
        ],
        axis=1,
    )

    return result


# ============================================================
# 14) DISPLAY RANK
# ============================================================

def add_display_rank(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # Signal 발생 종목만 대상으로 Rank
    active = (
        result[
            "Primary Signal"
        ]
        != "NONE"
    )

    result[
        "Signal Rank"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    if active.any():

        ranked = (
            result.loc[
                active,
                "Combined Score",
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(
                "Int64"
            )
        )

        result.loc[
            active,
            "Signal Rank"
        ] = ranked

    return result


# ============================================================
# 15) RADAR CATEGORY
# ============================================================

def radar_category(
    row: pd.Series,
) -> str:

    primary = str(
        row.get(
            "Primary Signal",
            "NONE",
        )
    )

    if primary == "DEEP_VALUE_REVERSAL":

        return "DEEP_VALUE"

    if primary == "QUALITY_VALUE":

        return "VALUE"

    if primary == "RECOVERY":

        return "RECOVERY"

    if primary == "QUALITY_MOMENTUM":

        return "MOMENTUM"

    if primary == "FALLEN_ANGEL":

        return "FALLEN_ANGEL"

    return "NONE"


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

        "Signal Quality Value",
        "Signal Fallen Angel",
        "Signal Recovery",
        "Signal Deep Value Reversal",
        "Signal Quality Momentum",

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

def save_phase4(
    df: pd.DataFrame,
) -> Path:

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    today = (
        datetime.now()
        .strftime(
            "%Y-%m-%d"
        )
    )

    path = (
        DATA_DIR
        / (
            "sp500_radar_signals_"
            f"{today}.csv"
        )
    )

    # --------------------------------------------------------
    # Signal 종목 우선
    # 그 안에서는 Combined Score 순
    # --------------------------------------------------------

    output = df.copy()

    output[
        "_has_signal"
    ] = (
        output[
            "Primary Signal"
        ]
        != "NONE"
    ).astype(
        int
    )

    output = output.sort_values(

        by=[
            "_has_signal",
            "Combined Score",
            "Fundamental Score",
            "Technical Score",
        ],

        ascending=[
            False,
            False,
            False,
            False,
        ],

        na_position="last",
    )

    output = output.drop(
        columns=[
            "_has_signal"
        ]
    )

    output.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

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

    signal_columns = {

        "QUALITY VALUE":
            "Signal Quality Value",

        "FALLEN ANGEL":
            "Signal Fallen Angel",

        "RECOVERY":
            "Signal Recovery",

        "DEEP VALUE REVERSAL":
            "Signal Deep Value Reversal",

        "QUALITY MOMENTUM":
            "Signal Quality Momentum",
    }

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