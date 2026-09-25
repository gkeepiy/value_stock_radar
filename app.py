from __future__ import annotations

"""주식추천 전용 화면. 실행: streamlit run app.py"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from phase3_technical import WEIGHTS, add_combined_scores, finite, normalize_tickers

DATA_DIR = Path("data")
PHASE4_PATTERN = "sp500_radar_signals_*.csv"
SCORE_LABELS = {"Fundamental Score": "펀더멘털", "Technical Score": "기술적 분석",
                "Cycle Score": "경기·산업 순환", "Seasonality Score": "계절성"}
LABELS = {"Display Rank": "순위", "Ticker": "종목", "Short Name": "기업명",
          "Sector": "업종", "Combined Score": "종합점수", "Price": "주가",
          "Currency": "통화", "Technical As Of": "가격 기준일",
          "Combined Missing Inputs": "부족한 평가 항목", **SCORE_LABELS}
STATE_LABELS = {"STRONG_UPTREND": "강한 상승 추세", "UPTREND": "상승 추세",
                "STRONG_DOWNTREND": "강한 하락 추세", "DOWNTREND": "하락 추세",
                "NEUTRAL": "중립", "BULLISH": "상승 우세", "BEARISH": "하락 우세",
                "OVERSOLD": "과매도", "OVERBOUGHT": "과매수", "EXTREME_OVERSOLD": "극단적 과매도",
                "EXTREME_OVERBOUGHT": "극단적 과매수", "OVERHEATED": "과열", "RECOVERY": "회복",
                "WEAKENING": "약화", "UNKNOWN": "자료 부족"}
QUALITY_LABELS = {"OK": "정상", "MISSING": "자료 없음", "PARTIAL": "일부 자료 부족",
                  "FAILED": "자료 조회 실패", "STALE": "자료가 오래됨", "INVALID": "유효하지 않은 자료",
                  "INSUFFICIENT_HISTORY": "과거 이력 부족", "FUTURE_OR_INVALID_DATE": "기준일 확인 필요",
                  "DUPLICATE_RECORDS": "중복 자료", "INVALID_SCORE_OR_SOURCE": "점수 또는 출처 확인 필요"}


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


def find_latest_phase4_file() -> Path:
    files = list(DATA_DIR.glob(PHASE4_PATTERN))
    if not files:
        raise FileNotFoundError("분석 결과 파일이 없습니다. Phase 3와 Phase 4를 실행해 주세요.")
    return max(files, key=lambda p: (p.stat().st_mtime_ns, p.name))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    result = normalize_tickers(df)
    numeric = list(WEIGHTS) + ["Price", "RSI14", "Stoch K", "Stoch D", "Williams %R", "Momentum 3M",
              "Momentum 6M", "52W Drawdown", "MA50", "MA200", "Volume Ratio", "Value Score", "Quality Score",
              "Growth Score", "Stability Score", "PBR", "Forward PER", "PSR", "EV/EBITDA", "FCF Yield",
              "ROE", "ROA", "Debt/Equity", "Operating Margin", "Profit Margin", "Seasonality Samples",
              "Seasonality Month", "Seasonality Mean Return", "Seasonality Win Rate", "Trend Score",
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
    path = find_latest_phase4_file()
    return read_data(str(path), path.stat().st_mtime_ns), path


def filter_rows(df: pd.DataFrame, sectors: list[str] | None = None, search: str = "",
                min_fundamental: float = 0, min_technical: float = 0) -> pd.DataFrame:
    result = df.copy()
    if sectors is not None and "Sector" in result:
        result = result[result["Sector"].fillna("미분류").astype(str).isin(sectors)]
    if search.strip():
        query = search.strip().casefold()
        company = result.get("Short Name", pd.Series("", index=result.index)).fillna("").astype(str)
        mask = result["Ticker"].str.casefold().str.contains(query, regex=False)
        mask |= company.str.casefold().str.contains(query, regex=False)
        result = result[mask]
    for column, threshold in [("Fundamental Score", min_fundamental), ("Technical Score", min_technical)]:
        if threshold > 0:
            result = result[result[column].ge(threshold)]
    return result


def split_recommendations(df: pd.DataFrame, min_combined: float = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready = df["Combined Data Quality"].eq("OK") & df["Combined Score"].notna()
    complete = df.loc[ready & df["Combined Score"].ge(min_combined)].copy()
    complete = complete.sort_values(["Combined Score", "Fundamental Score", "Technical Score", "Ticker"],
                                    ascending=[False, False, False, True])
    complete["Display Rank"] = np.arange(1, len(complete) + 1)
    pending = df.loc[~ready].copy()
    pending["Display Rank"] = pd.NA
    return complete, pending


def display_table(df: pd.DataFrame, pending=False) -> pd.DataFrame:
    columns = (["Ticker", "Short Name"] if pending else ["Display Rank", "Ticker", "Short Name"])
    columns += ["Combined Score", *WEIGHTS, "Sector", "Price", "Currency", "Technical As Of"]
    if pending:
        columns += ["Combined Missing Inputs"]
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
    if row.get("Oscillator Confirmation") == "CHECK_DATA":
        st.warning("두 오실레이터의 계산값이 일치하지 않아 원자료 확인이 필요합니다.")


def show_cycle_seasonality(row: pd.Series) -> None:
    metrics(row, [("Cycle Score", "경기·산업 순환 /100", "number"), ("Seasonality Score", "계절성 /100", "number")])
    st.write("순환 자료 상태:", QUALITY_LABELS.get(safe_text(row.get("Cycle Data Quality")), "자료 확인 필요"))
    st.write("기준일:", safe_text(row.get("Cycle As Of")))
    st.write("출처:", safe_text(row.get("Cycle Source")))
    if safe_text(row.get("Cycle Notes")) != "—":
        st.write("설명:", safe_text(row.get("Cycle Notes")))
    st.divider()
    st.write("계절성 자료 상태:", QUALITY_LABELS.get(safe_text(row.get("Seasonality Data Quality")), "자료 확인 필요"))
    metrics(row, [("Seasonality Month", "대상 월", "number"), ("Seasonality Samples", "과거 동월 표본 수", "number"),
                  ("Seasonality Mean Return", "과거 동월 평균 수익률", "percent"),
                  ("Seasonality Win Rate", "과거 동월 상승 비율", "percent")])
    st.caption("완료된 과거 월별 데이터만 사용합니다. 동월 상승 비율은 미래 적중률이 아닙니다.")


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
    tabs = st.tabs(["펀더멘털", "기술적 분석", "순환·계절성"])
    with tabs[0]:
        show_fundamental(row)
    with tabs[1]:
        show_technical(row)
    with tabs[2]:
        show_cycle_seasonality(row)


def choose_stock(df: pd.DataFrame, key: str) -> pd.Series | None:
    if df.empty:
        return None
    indexed = df.set_index("Ticker", drop=False)
    ticker = st.selectbox("상세 종목 선택", indexed.index.tolist(), key=key,
                          format_func=lambda x: f"{x} · {safe_text(indexed.loc[x].get('Short Name'))}")
    return indexed.loc[ticker]


def show_stock_recommendations(raw_df: pd.DataFrame, source_path: Path) -> None:
    st.title("주식추천 레이더")
    st.caption("펀더멘털 50% · 기술적 분석 30% · 경기·산업 순환 15% · 계절성 5%")
    st.sidebar.header("종목 필터")
    sectors = None
    if "Sector" in raw_df:
        choices = sorted(raw_df["Sector"].fillna("미분류").astype(str).unique())
        sectors = st.sidebar.multiselect("업종", choices, default=choices)
    search = st.sidebar.text_input("종목 코드 / 기업명")
    min_f = st.sidebar.slider("최소 펀더멘털 점수", 0, 100, 0)
    min_t = st.sidebar.slider("최소 기술적 점수", 0, 100, 0)
    min_c = st.sidebar.slider("최소 종합점수", 0, 100, 0)
    limit = st.sidebar.selectbox("표시 종목 수", [30, 50, 100, "전체"])
    filtered = filter_rows(raw_df, sectors, search, min_f, min_t)
    ready, pending = split_recommendations(filtered, min_c)
    boxes = st.columns(3)
    boxes[0].metric("추천 조건 충족", len(ready))
    boxes[1].metric("평균 종합점수", number_text(ready["Combined Score"].mean()))
    boxes[2].metric("자료 부족", len(pending))
    if ready.empty:
        st.info("현재 필터에서 네 가지 평가 자료를 모두 갖춘 종목이 없습니다.")
    else:
        shown = ready if limit == "전체" else ready.head(int(limit))
        st.dataframe(display_table(shown), hide_index=True, width="stretch", height=520)
        st.caption(f"조건에 맞는 {len(ready)}개 중 {len(shown)}개 표시 · 주가는 원자료 통화 기준")
        row = choose_stock(shown, "recommendation_stock")
        if row is not None:
            show_stock_detail(row)
    if not pending.empty:
        with st.expander(f"자료 부족 종목 확인 · {len(pending)}개"):
            st.caption("필요한 데이터가 채워질 때까지 추천 순위에서 제외합니다.")
            st.dataframe(display_table(pending, pending=True), hide_index=True, width="stretch")
            row = choose_stock(pending, "pending_stock")
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
