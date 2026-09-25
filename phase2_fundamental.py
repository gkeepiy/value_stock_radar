from __future__ import annotations

from datetime import datetime
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd


# ============================================================
# 0) CONFIG
# ============================================================

DATA_DIR = Path("data")

RAW_PATTERN = "sp500_raw_*.csv"

VALUE_MAX_SCORE = 35.0
QUALITY_MAX_SCORE = 30.0
GROWTH_MAX_SCORE = 20.0
STABILITY_MAX_SCORE = 15.0

TOTAL_MAX_SCORE = 100.0


# ============================================================
# 1) LOAD LATEST PHASE 1 FILE
# ============================================================

def find_latest_phase1_file() -> Path:

    files = list(
        DATA_DIR.glob(
            RAW_PATTERN
        )
    )

    if not files:
        raise FileNotFoundError(
            "Phase 1 결과 파일을 찾을 수 없습니다.\n"
            "data/sp500_raw_YYYY-MM-DD.csv 파일이 "
            "있는지 확인하세요."
        )

    # 실제 수정시간 기준 최신 파일
    latest = max(
        files,
        key=lambda p: p.stat().st_mtime,
    )

    return latest


def load_phase1_data() -> tuple[pd.DataFrame, Path]:

    path = find_latest_phase1_file()

    print(
        f"📂 Phase 1 데이터 로드: {path}"
    )

    df = pd.read_csv(
        path
    )

    if "Ticker" not in df.columns:
        raise ValueError(
            "Phase 1 파일에 Ticker 컬럼이 없습니다."
        )

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df, path


# ============================================================
# 2) SAFE NUMBER
# ============================================================

def to_float(
    value: t.Any,
) -> float | None:

    if value is None:
        return None

    try:
        x = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(x):
        return None

    return x


# ============================================================
# 3) PREPARE NUMERIC DATA
# ============================================================

def prepare_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    numeric_columns = [
        "Market Cap",
        "Revenue CAGR",
        "Revenue CAGR Years Used",
        "Revenue Growth",
        "Earnings Growth",
        "Forward PER",
        "PBR",
        "PSR",
        "EV/EBITDA",
        "FCF Yield",
        "ROE",
        "ROA",
        "Debt/Equity",
        "Operating Margin",
        "Profit Margin",
        "Free Cash Flow",
        "Operating Cash Flow",
    ]

    for column in numeric_columns:

        if column not in result.columns:
            result[column] = np.nan

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    if "Sector" not in result.columns:
        result["Sector"] = "Unknown"

    result["Sector"] = (
        result["Sector"]
        .fillna("Unknown")
        .astype(str)
    )

    return result


# ============================================================
# 4) FCF YIELD FALLBACK
# ============================================================

def ensure_fcf_yield(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    missing = (
        result["FCF Yield"].isna()
        &
        result["Free Cash Flow"].notna()
        &
        result["Market Cap"].notna()
        &
        (result["Market Cap"] > 0)
    )

    result.loc[
        missing,
        "FCF Yield"
    ] = (
        result.loc[
            missing,
            "Free Cash Flow"
        ]
        /
        result.loc[
            missing,
            "Market Cap"
        ]
    )

    return result


# ============================================================
# 5) SECTOR PERCENTILE
# ============================================================

def sector_percentile(
    df: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool,
    positive_only: bool = False,
) -> pd.Series:

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    valid = values.notna()

    if positive_only:
        valid &= values > 0

    output = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    if not valid.any():
        return output

    temp = pd.DataFrame(
        {
            "Sector": df.loc[
                valid,
                "Sector",
            ],
            "Value": values.loc[
                valid
            ],
        }
    )

    # higher_is_better:
    # 큰 값일수록 percentile 높음
    #
    # lower_is_better:
    # 작은 값일수록 percentile 높음

    percentile = (
        temp
        .groupby(
            "Sector"
        )["Value"]
        .rank(
            pct=True,
            ascending=(
                True
                if higher_is_better
                else False
            ),
            method="average",
        )
    )

    output.loc[
        percentile.index
    ] = percentile.clip(
        0,
        1,
    )

    return output


# ============================================================
# 6) NORMALIZED COMPONENT SCORE
# ============================================================

def normalized_weighted_score(
    row: pd.Series,
    components: dict[
        str,
        tuple[str, float]
    ],
    maximum_score: float,
) -> float:

    raw_score = 0.0
    available_weight = 0.0

    for _, (
        percentile_column,
        weight,
    ) in components.items():

        value = to_float(
            row.get(
                percentile_column
            )
        )

        if value is None:
            continue

        raw_score += (
            value
            * weight
        )

        available_weight += weight

    if available_weight == 0:
        return 0.0

    return (
        raw_score
        / available_weight
        * maximum_score
    )


# ============================================================
# 7) VALUE SCORE
# ============================================================

def calculate_value_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Sector Relative Percentiles
    # --------------------------------------------------------

    result[
        "PBR Percentile"
    ] = sector_percentile(
        result,
        "PBR",
        higher_is_better=False,
        positive_only=True,
    )

    result[
        "Forward PER Percentile"
    ] = sector_percentile(
        result,
        "Forward PER",
        higher_is_better=False,
        positive_only=True,
    )

    result[
        "PSR Percentile"
    ] = sector_percentile(
        result,
        "PSR",
        higher_is_better=False,
        positive_only=True,
    )

    result[
        "EV/EBITDA Percentile"
    ] = sector_percentile(
        result,
        "EV/EBITDA",
        higher_is_better=False,
        positive_only=True,
    )

    result[
        "FCF Yield Percentile"
    ] = sector_percentile(
        result,
        "FCF Yield",
        higher_is_better=True,
        positive_only=False,
    )

    # --------------------------------------------------------
    # Weight
    #
    # PBR           10
    # Forward PER    8
    # EV/EBITDA      8
    # FCF Yield      6
    # PSR            3
    #
    # Total         35
    # --------------------------------------------------------

    components = {
        "PBR": (
            "PBR Percentile",
            10.0,
        ),
        "PER": (
            "Forward PER Percentile",
            8.0,
        ),
        "EVEBITDA": (
            "EV/EBITDA Percentile",
            8.0,
        ),
        "FCF": (
            "FCF Yield Percentile",
            6.0,
        ),
        "PSR": (
            "PSR Percentile",
            3.0,
        ),
    }

    result[
        "Value Score"
    ] = result.apply(
        lambda row:
        normalized_weighted_score(
            row,
            components,
            VALUE_MAX_SCORE,
        ),
        axis=1,
    )

    return result


# ============================================================
# 8) QUALITY SCORE
# ============================================================

def calculate_quality_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "ROE Percentile"
    ] = sector_percentile(
        result,
        "ROE",
        higher_is_better=True,
    )

    result[
        "ROA Percentile"
    ] = sector_percentile(
        result,
        "ROA",
        higher_is_better=True,
    )

    result[
        "Operating Margin Percentile"
    ] = sector_percentile(
        result,
        "Operating Margin",
        higher_is_better=True,
    )

    result[
        "Profit Margin Percentile"
    ] = sector_percentile(
        result,
        "Profit Margin",
        higher_is_better=True,
    )

    # --------------------------------------------------------
    # ROE              10
    # ROA               7
    # Operating Margin  7
    # Profit Margin     4
    # Positive FCF      2
    #
    # Total            30
    # --------------------------------------------------------

    def calculate_row(
        row: pd.Series,
    ) -> float:

        raw_score = 0.0
        available_weight = 0.0

        components = [
            (
                "ROE Percentile",
                10.0,
            ),
            (
                "ROA Percentile",
                7.0,
            ),
            (
                "Operating Margin Percentile",
                7.0,
            ),
            (
                "Profit Margin Percentile",
                4.0,
            ),
        ]

        for column, weight in components:

            value = to_float(
                row.get(
                    column
                )
            )

            if value is None:
                continue

            raw_score += (
                value
                * weight
            )

            available_weight += weight

        fcf = to_float(
            row.get(
                "Free Cash Flow"
            )
        )

        if fcf is not None:

            available_weight += 2.0

            if fcf > 0:
                raw_score += 2.0

        if available_weight == 0:
            return 0.0

        return (
            raw_score
            / available_weight
            * QUALITY_MAX_SCORE
        )

    result[
        "Quality Score"
    ] = result.apply(
        calculate_row,
        axis=1,
    )

    return result


# ============================================================
# 9) GROWTH SCORE
# ============================================================

def revenue_cagr_score(
    value: t.Any,
    years: t.Any,
) -> float:

    cagr = to_float(
        value
    )

    years_used = to_float(
        years
    )

    if cagr is None:
        return 0.0

    if cagr >= 0.20:
        base = 10.0

    elif cagr >= 0.15:
        base = 8.5

    elif cagr >= 0.10:
        base = 7.0

    elif cagr >= 0.05:
        base = 5.0

    elif cagr > 0:
        base = 3.0

    else:
        base = 0.0

    if years_used is None:
        confidence = 0.50

    elif years_used >= 4:
        confidence = 1.00

    elif years_used >= 3:
        confidence = 0.90

    elif years_used >= 2:
        confidence = 0.75

    else:
        confidence = 0.50

    return (
        base
        * confidence
    )


def earnings_growth_score(
    value: t.Any,
) -> float:

    x = to_float(
        value
    )

    if x is None:
        return 0.0

    if x >= 0.25:
        return 6.0

    if x >= 0.15:
        return 5.0

    if x >= 0.10:
        return 4.0

    if x >= 0.05:
        return 3.0

    if x >= 0:
        return 1.5

    return 0.0


def revenue_growth_score(
    value: t.Any,
) -> float:

    x = to_float(
        value
    )

    if x is None:
        return 0.0

    if x >= 0.20:
        return 4.0

    if x >= 0.15:
        return 3.5

    if x >= 0.10:
        return 3.0

    if x >= 0.05:
        return 2.0

    if x >= 0:
        return 1.0

    return 0.0


def calculate_growth_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "Growth CAGR Raw"
    ] = result.apply(
        lambda row:
        revenue_cagr_score(
            row.get(
                "Revenue CAGR"
            ),
            row.get(
                "Revenue CAGR Years Used"
            ),
        ),
        axis=1,
    )

    result[
        "Growth Earnings Raw"
    ] = result[
        "Earnings Growth"
    ].apply(
        earnings_growth_score
    )

    result[
        "Growth Revenue Raw"
    ] = result[
        "Revenue Growth"
    ].apply(
        revenue_growth_score
    )

    # --------------------------------------------------------
    # Revenue CAGR      10
    # Earnings Growth    6
    # Revenue Growth     4
    #
    # Total             20
    # --------------------------------------------------------

    def calculate_row(
        row: pd.Series,
    ) -> float:

        raw_score = 0.0
        available_weight = 0.0

        if pd.notna(
            row.get(
                "Revenue CAGR"
            )
        ):

            raw_score += row[
                "Growth CAGR Raw"
            ]

            available_weight += 10.0

        if pd.notna(
            row.get(
                "Earnings Growth"
            )
        ):

            raw_score += row[
                "Growth Earnings Raw"
            ]

            available_weight += 6.0

        if pd.notna(
            row.get(
                "Revenue Growth"
            )
        ):

            raw_score += row[
                "Growth Revenue Raw"
            ]

            available_weight += 4.0

        if available_weight == 0:
            return 0.0

        return (
            raw_score
            / available_weight
            * GROWTH_MAX_SCORE
        )

    result[
        "Growth Score"
    ] = result.apply(
        calculate_row,
        axis=1,
    )

    return result


# ============================================================
# 10) STABILITY SCORE
# ============================================================

def debt_score(
    value: t.Any,
) -> float:

    x = to_float(
        value
    )

    if x is None:
        return 0.0

    if x <= 0.25:
        return 7.0

    if x <= 0.50:
        return 6.0

    if x <= 1.00:
        return 5.0

    if x <= 1.50:
        return 3.0

    if x <= 2.00:
        return 1.5

    return 0.0


def history_score(
    value: t.Any,
) -> float:

    years = to_float(
        value
    )

    if years is None:
        return 0.0

    if years >= 4:
        return 2.0

    if years >= 3:
        return 1.5

    if years >= 2:
        return 1.0

    if years >= 1:
        return 0.5

    return 0.0


def calculate_stability_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "Stability Debt Raw"
    ] = result[
        "Debt/Equity"
    ].apply(
        debt_score
    )

    result[
        "Stability History Raw"
    ] = result[
        "Revenue CAGR Years Used"
    ].apply(
        history_score
    )

    # --------------------------------------------------------
    # Debt / Equity       7
    # Operating CF        3
    # Free Cash Flow      3
    # Revenue History     2
    #
    # Total              15
    # --------------------------------------------------------

    def calculate_row(
        row: pd.Series,
    ) -> float:

        raw_score = 0.0
        available_weight = 0.0

        debt = to_float(
            row.get(
                "Debt/Equity"
            )
        )

        if debt is not None:

            available_weight += 7.0

            raw_score += row[
                "Stability Debt Raw"
            ]

        operating_cf = to_float(
            row.get(
                "Operating Cash Flow"
            )
        )

        if operating_cf is not None:

            available_weight += 3.0

            if operating_cf > 0:
                raw_score += 3.0

        fcf = to_float(
            row.get(
                "Free Cash Flow"
            )
        )

        if fcf is not None:

            available_weight += 3.0

            if fcf > 0:
                raw_score += 3.0

        years = to_float(
            row.get(
                "Revenue CAGR Years Used"
            )
        )

        if years is not None:

            available_weight += 2.0

            raw_score += row[
                "Stability History Raw"
            ]

        if available_weight == 0:
            return 0.0

        return (
            raw_score
            / available_weight
            * STABILITY_MAX_SCORE
        )

    result[
        "Stability Score"
    ] = result.apply(
        calculate_row,
        axis=1,
    )

    return result


# ============================================================
# 11) FUNDAMENTAL SCORE
# ============================================================

def fundamental_grade(
    score: t.Any,
) -> str:

    value = to_float(
        score
    )

    if value is None:
        return "N/A"

    if value >= 90:
        return "A+"

    if value >= 85:
        return "A"

    if value >= 80:
        return "A-"

    if value >= 75:
        return "B+"

    if value >= 70:
        return "B"

    if value >= 65:
        return "B-"

    if value >= 60:
        return "C+"

    if value >= 50:
        return "C"

    return "D"


def calculate_fundamental_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = prepare_numeric_columns(
        df
    )

    result = ensure_fcf_yield(
        result
    )

    print(
        "   → Value Score 계산"
    )

    result = calculate_value_scores(
        result
    )

    print(
        "   → Quality Score 계산"
    )

    result = calculate_quality_scores(
        result
    )

    print(
        "   → Growth Score 계산"
    )

    result = calculate_growth_scores(
        result
    )

    print(
        "   → Stability Score 계산"
    )

    result = calculate_stability_scores(
        result
    )

    result[
        "Fundamental Score"
    ] = (
        result[
            "Value Score"
        ]
        +
        result[
            "Quality Score"
        ]
        +
        result[
            "Growth Score"
        ]
        +
        result[
            "Stability Score"
        ]
    )

    score_columns = [
        "Value Score",
        "Quality Score",
        "Growth Score",
        "Stability Score",
        "Fundamental Score",
    ]

    for column in score_columns:

        result[column] = (
            result[column]
            .clip(
                lower=0,
                upper=100,
            )
            .round(2)
        )

    result[
        "Fundamental Grade"
    ] = result[
        "Fundamental Score"
    ].apply(
        fundamental_grade
    )

    return result


# ============================================================
# 12) RANKINGS
# ============================================================

def add_rankings(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "Fundamental Rank"
    ] = (
        result[
            "Fundamental Score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(
            "Int64"
        )
    )

    result[
        "Sector Fundamental Rank"
    ] = (
        result
        .groupby(
            "Sector"
        )[
            "Fundamental Score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(
            "Int64"
        )
    )

    valid_count = (
        result[
            "Fundamental Score"
        ].notna().sum()
    )

    if valid_count > 0:

        result[
            "Fundamental Percentile"
        ] = (
            result[
                "Fundamental Score"
            ]
            .rank(
                pct=True,
                ascending=True,
            )
            * 100
        ).round(
            2
        )

    else:

        result[
            "Fundamental Percentile"
        ] = np.nan

    return result


# ============================================================
# 14) COLUMN ORDER
# ============================================================

def reorder_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    priority = [
        "Fundamental Rank",
        "Ticker",
        "Short Name",
        "Sector",
        "Industry",
        "Market Cap",

        "Fundamental Score",
        "Fundamental Grade",
        "Fundamental Percentile",

        "Value Score",
        "Quality Score",
        "Growth Score",
        "Stability Score",

        "Sector Fundamental Rank",

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
        "Revenue CAGR Years Used",
        "Revenue CAGR Method",
        "Revenue Growth",
        "Earnings Growth",

        "Debt/Equity",
        "Free Cash Flow",
        "Operating Cash Flow",

        "PBR Percentile",
        "Forward PER Percentile",
        "PSR Percentile",
        "EV/EBITDA Percentile",
        "FCF Yield Percentile",

        "ROE Percentile",
        "ROA Percentile",
        "Operating Margin Percentile",
        "Profit Margin Percentile",

        "Data Quality",
        "Failed Reason",
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

    return result_or_df(
        df,
        existing + remaining,
    )


def result_or_df(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:

    return df[
        columns
    ].copy()


# ============================================================
# 15) SAVE
# ============================================================

def save_phase2(
    df: pd.DataFrame,
) -> Path:

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    path = (
        DATA_DIR
        / (
            "sp500_fundamental_scores_"
            f"{today}.csv"
        )
    )

    output = df.sort_values(
        by=[
            "Fundamental Score",
            "Value Score",
            "Quality Score",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        na_position="last",
    )

    output.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


# ============================================================
# 16) PRINT SUMMARY
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "FUNDAMENTAL SCORE SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"총 종목:"
        f" {len(df)}"
    )

    print(
        f"평균 점수:"
        f" {df['Fundamental Score'].mean():.2f}"
    )

    print(
        f"중앙값:"
        f" {df['Fundamental Score'].median():.2f}"
    )

    print(
        f"최고 점수:"
        f" {df['Fundamental Score'].max():.2f}"
    )

    print(
        f"80점 이상:"
        f" {(df['Fundamental Score'] >= 80).sum()}"
    )

    print(
        f"70점 이상:"
        f" {(df['Fundamental Score'] >= 70).sum()}"
    )


    print()


# ============================================================
# 17) PRINT TOP 20
# ============================================================

def print_top_stocks(
    df: pd.DataFrame,
    top_n: int = 20,
) -> None:

    columns = [
        "Fundamental Rank",
        "Ticker",
        "Short Name",
        "Sector",
        "Fundamental Score",
        "Fundamental Grade",
        "Value Score",
        "Quality Score",
        "Growth Score",
        "Stability Score",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    top = (
        df
        .sort_values(
            "Fundamental Score",
            ascending=False,
        )
        .head(
            top_n
        )
    )

    print(
        "=" * 130
    )

    print(
        f"TOP {top_n} FUNDAMENTAL STOCKS"
    )

    print(
        "=" * 130
    )

    print(
        top[
            columns
        ].to_string(
            index=False
        )
    )

    print()


# ============================================================
# 18) DATA QUALITY SUMMARY
# ============================================================

def print_data_quality(
    df: pd.DataFrame,
) -> None:

    if "Data Quality" not in df.columns:
        return

    print(
        "DATA QUALITY"
    )

    print(
        "-" * 50
    )

    print(
        df[
            "Data Quality"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()


# ============================================================
# 19) MAIN
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
        "        PHASE 2 - FUNDAMENTAL ENGINE"
    )

    print(
        "=============================================="
    )

    print()

    # --------------------------------------------------------
    # Phase 1 Load
    # --------------------------------------------------------

    df, source_path = (
        load_phase1_data()
    )

    print(
        f"✅ 종목 수:"
        f" {len(df)}"
    )

    print()

    # --------------------------------------------------------
    # Fundamental Engine
    # --------------------------------------------------------

    print(
        "🧮 Fundamental Score 계산 시작..."
    )

    result = (
        calculate_fundamental_scores(
            df
        )
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    result = add_rankings(
        result
    )

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    result = reorder_columns(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        save_phase2(
            result
        )
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()

    print_data_quality(
        result
    )

    print_summary(
        result
    )

    print_top_stocks(
        result,
        top_n=20,
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
        "✅ PHASE 2 COMPLETE"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()