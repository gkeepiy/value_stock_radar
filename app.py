from __future__ import annotations

"""주식추천 전용 화면. 실행: streamlit run app.py"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from phase3_technical import WEIGHTS, add_combined_scores, finite, normalize_tickers

DATA_DIR = Path("data")
PHASE3_PATTERN = "sp500_technical_scores_*.csv"
INDEX_OPTIONS = ["전체", "S&P 500", "Russell 2000"]
SCORE_LABELS = {"Fundamental Score": "펀더멘털", "Technical Score": "기술적 분석"}
LABELS = {"Display Rank": "순위", "Ticker": "종목", "Short Name": "기업명", "Index": "지수",
          "Sector": "업종", "Combined Score": "종합점수", "Price": "주가",
          "52W Drawdown": "52주 고점 대비", "Sideways 31D Range": "최근 31거래일 종가 범위",
          "Currency": "통화", "Technical As Of": "가격 기준일",
          "Combined Missing Inputs": "부족한 평가 항목", **SCORE_LABELS}
STATE_LABELS = {"STRONG_UPTREND": "강한 상승 추세", "UPTREND": "상승 추세",
                "STRONG_DOWNTREND": "강한 하락 추세", "DOWNTREND": "하락 추세",
                "NEUTRAL": "중립", "BULLISH": "상승 우세", "BEARISH": "하락 우세",
                "OVERSOLD": "과매도", "OVERBOUGHT": "과매수", "EXTREME_OVERSOLD": "극단적 과매도",
                "EXTREME_OVERBOUGHT": "극단적 과매수", "OVERHEATED": "과열", "RECOVERY": "회복",
                "WEAKENING": "약화", "UNKNOWN": "자료 부족"}
def safe_text(value, default="—") -> str:
    if value is None or pd.isna(value) or str(value).strip() in {"", "NONE", "nan"}:
        return default
    return str(value)


def number_text(value, digits=1) -> str:
    return f"{float(value):,.{digits}f}" if finite(value) else "—"


def percent_text(value) -> str:
    return f"{float(value):.1%}" if finite(value) else "—"


def state_text(value) -> str:
    return STATE_LABELS.get(safe_text(value), safe_text(value))


def missing_text(value) -> str:
    text = safe_text(value)
    for column, label in SCORE_LABELS.items():
        text = text.replace(column, label)
    return text.replace("; ", ", ")


def find_latest_phase3_file() -> Path:
    files = list(DATA_DIR.glob(PHASE3_PATTERN))
    if not files:
        raise FileNotFoundError("분석 결과 파일이 없습니다. Phase 1~3을 실행해 주세요.")
    return max(files, key=lambda p: (p.stat().st_mtime_ns, p.name))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    result = normalize_tickers(df)
    numeric = list(WEIGHTS) + ["Price", "RSI14", "Stoch K", "Stoch D", "Williams %R", "Momentum 3M",
              "Sideways 31D Range",
              "Momentum 6M", "52W Drawdown", "MA50", "MA200", "Volume Ratio", "Value Score", "Quality Score",
              "Growth Score", "Stability Score", "PBR", "Forward PER", "PSR", "EV/EBITDA", "FCF Yield",
              "ROE", "ROA", "Debt/Equity", "Operating Margin", "Profit Margin", "Trend Score",
              "Momentum Score", "RSI Score", "Stochastic Score", "Volume Score"]
    for column in numeric:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Do not trust a previous 50:50 total from an older CSV.
    return add_combined_scores(result)


@st.cache_data(ttl=300)
def read_data(path: str, mtime_ns: int) -> pd.DataFrame:
    # File modification time is part of the cache key.
    return prepare_data(pd.read_csv(path))


def load_data() -> tuple[pd.DataFrame, Path]:
    path = find_latest_phase3_file()
    return read_data(str(path), path.stat().st_mtime_ns), path


def filter_rows(df: pd.DataFrame, index_name: str = "전체", sectors: list[str] | None = None,
                search: str = "") -> pd.DataFrame:
    result = df.copy()
    if index_name != "전체" and "Index" in result:
        result = result[
            result["Index"].fillna("").astype(str).str.contains(index_name, regex=False)
        ]
    if sectors is not None and "Sector" in result:
        result = result[result["Sector"].fillna("미분류").astype(str).isin(sectors)]
    if search.strip():
        query = search.strip().casefold()
        company = result.get("Short Name", pd.Series("", index=result.index)).fillna("").astype(str)
        mask = result["Ticker"].str.casefold().str.contains(query, regex=False)
        mask |= company.str.casefold().str.contains(query, regex=False)
        result = result[mask]
    return result


def sort_all_stocks(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(
        ["Combined Score", "Fundamental Score", "Technical Score", "Ticker"],
        ascending=[False, False, False, True],
        na_position="last",
    ).copy()
    scored = result["Combined Score"].notna()
    result["Display Rank"] = pd.NA
    result.loc[scored, "Display Rank"] = np.arange(1, int(scored.sum()) + 1)
    return result


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Display Rank", "Ticker", "Short Name", "Index", "Combined Score", *WEIGHTS,
               "52W Drawdown", "Sideways 31D Range", "Sector", "Price", "Currency",
               "Technical As Of", "Combined Missing Inputs"]
    table = df[[c for c in columns if c in df]].copy()
    if "Combined Missing Inputs" in table:
        table["Combined Missing Inputs"] = table["Combined Missing Inputs"].map(missing_text)
    for column in ["Combined Score", *WEIGHTS, "Price"]:
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    return table.rename(columns=LABELS)


def metrics(row: pd.Series, entries: list[tuple[str, str, str]]) -> None:
    for start in range(0, len(entries), 4):
        group = entries[start:start + 4]
        for box, (column, label, fmt) in zip(st.columns(len(group)), group):
            value = percent_text(row.get(column)) if fmt == "percent" else number_text(row.get(column))
            box.metric(label, value)


def show_fundamental(row: pd.Series) -> None:
    metrics(row, [("Value Score", "가치평가 /35", "number"), ("Quality Score", "수익성·품질 /30", "number"),
                  ("Growth Score", "성장 /20", "number"), ("Stability Score", "안정성 /15", "number")])
    metrics(row, [("PBR", "PBR", "number"), ("Forward PER", "예상 PER", "number"),
                  ("EV/EBITDA", "EV/EBITDA", "number"), ("FCF Yield", "잉여현금흐름 수익률", "percent"),
                  ("ROE", "ROE", "percent"), ("ROA", "ROA", "percent"),
                  ("Operating Margin", "영업이익률", "percent"), ("Debt/Equity", "Debt/Equity (원자료)", "number")])


def show_technical(row: pd.Series) -> None:
    metrics(row, [("RSI14", "RSI 14", "number"), ("Momentum 3M", "3개월 모멘텀", "percent"),
                  ("Momentum 6M", "6개월 모멘텀", "percent"), ("52W Drawdown", "52주 고점 대비", "percent"),
                  ("MA50", "50일 이동평균", "number"), ("MA200", "200일 이동평균", "number"),
                  ("Volume Ratio", "평균 대비 거래량 배수", "number")])
    components = [("Trend Score", "추세", 30), ("Momentum Score", "모멘텀", 20),
                  ("RSI Score", "RSI", 15), ("Stochastic Score", "스토캐스틱", 15), ("Volume Score", "거래량", 10)]
    st.dataframe(pd.DataFrame([{"평가 항목": label, "원점수": row.get(key), "최대 배점": maximum}
                              for key, label, maximum in components]), hide_index=True, width="stretch")
    st.caption("기술적 점수 = 원점수 합계 ÷ 90 × 100")
    st.write(f"추세: {state_text(row.get('Trend State'))} · RSI: {state_text(row.get('RSI State'))}")
    metrics(row, [("Stoch K", "스토캐스틱 %K", "number"), ("Stoch D", "스토캐스틱 %D", "number"),
                  ("Williams %R", "Williams %R", "number")])
    st.write(f"스토캐스틱: {state_text(row.get('Stochastic State'))} · Williams %R: {state_text(row.get('Williams State'))}")
    st.caption("Williams %R은 보조 확인용입니다. 같은 14기간 Fast %K와 중복되는 정보이므로 추가 가점이 없습니다.")
    metrics(row, [("Sideways 31D Range", "최근 31거래일 종가 범위", "percent")])
    if row.get("Oscillator Confirmation") == "CHECK_DATA":
        st.warning("두 오실레이터의 계산값이 일치하지 않아 원자료 확인이 필요합니다.")


def show_stock_detail(row: pd.Series) -> None:
    st.subheader(f"{row['Ticker']} · {safe_text(row.get('Short Name'))}")
    st.caption(f"{safe_text(row.get('Sector'))} · {safe_text(row.get('Industry'))} · 가격 기준일 {safe_text(row.get('Technical As Of'))}")
    entries = [("Combined Score", "종합점수 /100", "number")]
    entries += [(key, label + " /100", "number") for key, label in SCORE_LABELS.items()]
    metrics(row, entries)
    contribution = [{"평가 항목": SCORE_LABELS[key], "점수": row.get(key), "비중": f"{weight:.0%}",
                     "종합점수 기여": row.get(key.replace(" Score", " Contribution"))} for key, weight in WEIGHTS.items()]
    st.dataframe(pd.DataFrame(contribution), hide_index=True, width="stretch")
    if row.get("Combined Data Quality") != "OK":
        st.info("종합 평가 대기: " + missing_text(row.get("Combined Missing Inputs")))
    reason = safe_text(row.get("Technical Failed Reason"))
    if reason != "—":
        st.warning(reason)
    risks = safe_text(row.get("Risk Flags"))
    if risks != "—":
        translations = {"BELOW_MA200": "200일 이동평균 아래", "RSI_OVERHEATED": "RSI 과열",
                        "WEAK_6M_MOMENTUM": "6개월 모멘텀 약화", "DEEP_PRICE_COLLAPSE": "고점 대비 큰 하락",
                        "STOCH_BEARISH_CROSS": "스토캐스틱 하향 교차"}
        st.warning("참고할 위험 요인: " + ", ".join(translations.get(v, v) for v in risks.split("|")))
    tabs = st.tabs(["펀더멘털", "기술적 분석"])
    with tabs[0]:
        show_fundamental(row)
    with tabs[1]:
        show_technical(row)


def choose_stock(df: pd.DataFrame, key: str) -> pd.Series | None:
    if df.empty:
        return None
    indexed = df.set_index("Ticker", drop=False)
    ticker = st.selectbox("상세 종목 선택", indexed.index.tolist(), key=key,
                          format_func=lambda x: f"{x} · {safe_text(indexed.loc[x].get('Short Name'))}")
    return indexed.loc[ticker]


def show_stock_recommendations(raw_df: pd.DataFrame, source_path: Path) -> None:
    st.title("전체 종목 종합점수")
    st.caption("종합점수 내림차순 · 펀더멘털 62.5% · 기술적 분석 37.5%")
    st.sidebar.header("종목 필터")
    index_name = st.sidebar.radio("지수", INDEX_OPTIONS, horizontal=True)
    if index_name != "전체" and "Index" not in raw_df.columns:
        st.warning("현재 데이터에 지수 구분(Index)이 없습니다. 최신 Phase 1 데이터로 다시 수집한 뒤 Phase 2~3을 실행해 주세요.")
        return
    sectors = None
    if "Sector" in raw_df:
        choices = sorted(raw_df["Sector"].fillna("미분류").astype(str).unique())
        sectors = st.sidebar.multiselect("업종", choices, default=choices)
    search = st.sidebar.text_input("종목 코드 / 기업명")
    limit = st.sidebar.selectbox("표시 종목 수", [30, 50, 100, "전체"], index=3)
    filtered = filter_rows(raw_df, index_name, sectors, search)
    ranked = sort_all_stocks(filtered)
    boxes = st.columns(3)
    scored = ranked["Combined Score"].notna()
    boxes[0].metric("표시 종목", len(ranked))
    boxes[1].metric("종합점수 있음", int(scored.sum()))
    boxes[2].metric("평균 종합점수", number_text(ranked.loc[scored, "Combined Score"].mean()))
    if ranked.empty:
        st.info("선택한 조건에 해당하는 종목이 없습니다.")
    else:
        shown = ranked if limit == "전체" else ranked.head(int(limit))
        st.dataframe(display_table(shown), hide_index=True, width="stretch", height=520)
        st.caption(f"전체 {len(ranked)}개 중 {len(shown)}개 표시 · 점수가 없는 종목은 목록 아래쪽에 표시")
        row = choose_stock(shown, "recommendation_stock")
        if row is not None:
            show_stock_detail(row)
    st.caption("점수는 설정한 평가 기준의 충족도이며 미래 수익률이나 상승 확률이 아닙니다.")
    st.sidebar.caption(f"결과 파일: {source_path.name}")


def main() -> None:
    st.set_page_config(page_title="AI Value Stock Radar", layout="wide")
    try:
        raw_df, source_path = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()
        return
    show_stock_recommendations(raw_df, source_path)


if __name__ == "__main__":
    main()
