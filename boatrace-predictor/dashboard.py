"""
BOAT LAB ダッシュボード
Streamlit製のローカルブラウザUI
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

# ─── ページ設定 ───
st.set_page_config(
    page_title="BOAT LAB",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
<style>
    /* ── ヘッダー ── */
    .boat-header {
        background: linear-gradient(135deg, #0a3d62 0%, #1a6da0 50%, #0a3d62 100%);
        color: white;
        padding: 2rem 2rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(10,61,98,0.3);
    }
    .boat-header h1 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2.8rem;
        letter-spacing: 8px;
        margin: 0;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(88,166,255,0.8), 0 0 40px rgba(88,166,255,0.4);
    }
    .boat-header p {
        font-family: 'Noto Sans JP', sans-serif;
        margin: 0.5rem 0 0;
        font-size: 0.85rem;
        letter-spacing: 3px;
        color: rgba(255,255,255,0.65);
    }

    /* ── メトリクスカード ── */
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1a6da0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.78rem; color: #64748b; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.9rem; font-weight: bold; color: #0a3d62; }
    .metric-sub   { font-size: 0.74rem; color: #94a3b8; margin-top: 0.2rem; }

    /* ── 日付ボタン ── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 20px !important;
        font-size: 0.82rem !important;
        padding: 0.3rem 0.8rem !important;
        border: 1px solid #cbd5e1 !important;
        background: white !important;
        color: #334155 !important;
        font-weight: 500 !important;
        transition: all 0.15s;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #1a6da0 !important;
        color: white !important;
        border-color: #1a6da0 !important;
    }

    /* ── サイドバー ── */
    .date-btn-active > button {
        background: #1a6da0 !important;
        color: white !important;
        border-color: #1a6da0 !important;
        font-weight: 700 !important;
    }

    hr { border-color: #e2e8f0; }
    .footer { text-align: center; color: #94a3b8; font-size: 0.78rem; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── 会場カラー（ライト版） ───
_VENUE_COLORS = {
    "桐生":   "#dbeafe", "戸田":   "#dbeafe", "江戸川": "#bfdbfe",
    "平和島": "#bfdbfe", "多摩川": "#bfdbfe",
    "浜名湖": "#dcfce7", "蒲郡":   "#dcfce7", "常滑":   "#bbf7d0",
    "津":     "#bbf7d0", "三国":   "#bbf7d0",
    "びわこ": "#ede9fe", "住之江": "#ede9fe", "尼崎":   "#ddd6fe",
    "鳴門":   "#fef9c3", "丸亀":   "#fef9c3", "児島":   "#fef08a",
    "宮島":   "#fef08a",
    "徳山":   "#ffedd5", "下関":   "#ffedd5", "若松":   "#fed7aa",
    "芦屋":   "#fed7aa", "福岡":   "#fed7aa", "唐津":   "#fdba74",
    "大村":   "#fdba74",
}
_DEFAULT_BG = "#f8fafc"


# ─── Google Sheets 接続 ───
@st.cache_resource(show_spinner=False)
def get_spreadsheet():
    cred_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "./credentials.json")
    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    if not Path(cred_path).exists() or not spreadsheet_id:
        return None
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(cred_path, scopes=scopes)
    return gspread.authorize(creds).open_by_key(spreadsheet_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_predictions(sheet_name: str) -> pd.DataFrame:
    try:
        ss = get_spreadsheet()
        if ss is None:
            return pd.DataFrame()
        return pd.DataFrame(ss.worksheet(sheet_name).get_all_records())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_results() -> pd.DataFrame:
    try:
        ss = get_spreadsheet()
        if ss is None:
            return pd.DataFrame()
        return pd.DataFrame(ss.worksheet("成績").get_all_records())
    except Exception:
        return pd.DataFrame()


def calc_stats(df: pd.DataFrame) -> dict:
    empty = {"total": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0}
    if df.empty or "予想買い目" not in df.columns:
        return empty
    df = df[~df["予想買い目"].isin(["", "（予想なし）", "見送り", "-"])].copy()
    total = len(df)
    if total == 0:
        return empty
    hits = (df["的中"] == "○").sum()
    total_bet = total * 100
    total_return = df[df["的中"] == "○"]["実際の払戻"].apply(
        lambda x: int(str(x).replace(",", "")) if str(x).replace(",", "").isdigit() else 0
    ).sum()
    profit = total_return - total_bet
    return {
        "total": total, "hits": int(hits),
        "hit_rate": hits / total * 100,
        "roi": total_return / total_bet * 100 if total_bet > 0 else 0.0,
        "profit": profit,
    }


# ════════════════════════════════
# ヘッダー
# ════════════════════════════════
st.markdown("""
<div class="boat-header">
    <h1>🚤 BOAT LAB</h1>
    <p>競艇 AI 予想ダッシュボード</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════
# サイドバー
# ════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ 設定")
    if st.button("🔄 データを更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("**📊 表示設定**")
    show_all_confidence = st.checkbox("★☆☆も表示する", value=True)


# ════════════════════════════════
# 日付ワンクリック切替
# ════════════════════════════════
today = datetime.now().date()

# セッションステートで選択日付を管理
if "selected_date" not in st.session_state:
    st.session_state.selected_date = today.strftime("%Y-%m-%d")

# 直近7日分のボタンを横並び
st.markdown("**📅 日付を選択**")
date_cols = st.columns(7)
for i in range(7):
    d = today - timedelta(days=i)
    d_str = d.strftime("%Y-%m-%d")
    label = "今日" if i == 0 else f"{d.month}/{d.day}"
    with date_cols[i]:
        if st.button(label, key=f"date_{i}", use_container_width=True):
            st.session_state.selected_date = d_str

date_str = st.session_state.selected_date
st.caption(f"表示中: {date_str}")
st.markdown("---")


# ════════════════════════════════
# データ読み込み
# ════════════════════════════════
with st.spinner("データ読み込み中..."):
    df_pred = load_predictions(date_str)
    df_results = load_results()

stats = calc_stats(df_results)


# ════════════════════════════════
# メトリクス行
# ════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">📋 累計予想点数</div>
        <div class="metric-value">{stats['total']}</div>
        <div class="metric-sub">的中 {stats['hits']} 回</div>
    </div>""", unsafe_allow_html=True)
with c2:
    hr = stats['hit_rate']
    c = "#16a34a" if hr >= 10 else "#dc2626"
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">🎯 的中率</div>
        <div class="metric-value" style="color:{c}">{hr:.1f}%</div>
        <div class="metric-sub">ランダム比較: 0.8%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    roi = stats['roi']
    c = "#16a34a" if roi >= 100 else "#dc2626"
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">💹 回収率</div>
        <div class="metric-value" style="color:{c}">{roi:.1f}%</div>
        <div class="metric-sub">100%超えで黒字</div>
    </div>""", unsafe_allow_html=True)
with c4:
    profit = stats['profit']
    c = "#16a34a" if profit >= 0 else "#dc2626"
    sign = "+" if profit >= 0 else ""
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">💰 累計収支</div>
        <div class="metric-value" style="color:{c}">{sign}¥{profit:,}</div>
        <div class="metric-sub">100円/点で計算</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ════════════════════════════════
# タブ
# ════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎯 本日の予想", "📋 成績一覧", "📈 分析グラフ"])


# ── TAB1: 本日の予想 ──
with tab1:
    if df_pred.empty:
        st.info(f"📭 {date_str} の予想データがまだありません。\n`python main.py --mode auto` を実行してください。")
    else:
        df_show = df_pred.copy()
        if not show_all_confidence and "信頼度" in df_show.columns:
            df_show = df_show[df_show["信頼度"] != "★☆☆"]

        disp_cols = [c for c in ["競艇場", "レース", "区分", "買い目（3連単）", "信頼度", "的中確率", "オッズ"] if c in df_show.columns]
        df_show = df_show[disp_cols].reset_index(drop=True)

        def style_pred_row(row):
            bg = _VENUE_COLORS.get(str(row.get("競艇場", "")), _DEFAULT_BG)
            conf = str(row.get("信頼度", ""))
            tier = str(row.get("区分", ""))
            if conf == "★★★":
                bg = "#fffbeb"
            styles = [f"background-color: {bg};" for _ in row]
            # 区分列: 本命=青、中穴=オレンジ
            if "区分" in row.index:
                idx = list(row.index).index("区分")
                if tier == "本命":
                    styles[idx] = "background-color: #dbeafe; color: #1d4ed8; font-weight: bold;"
                elif tier == "中穴":
                    styles[idx] = "background-color: #ffedd5; color: #c2410c; font-weight: bold;"
            # 信頼度列
            if "信頼度" in row.index:
                idx = list(row.index).index("信頼度")
                if conf == "★★★":
                    styles[idx] = "background-color: #fffbeb; color: #d97706; font-weight: bold;"
                elif conf == "★★☆":
                    styles[idx] = f"background-color: {bg}; color: #64748b;"
                else:
                    styles[idx] = f"background-color: {bg}; color: #cbd5e1;"
            return styles

        st.dataframe(
            df_show.style.apply(style_pred_row, axis=1),
            use_container_width=True, height=600
        )


# ── TAB2: 成績一覧 ──
with tab2:
    if df_results.empty:
        st.info("📭 成績データがまだありません。")
    else:
        date_options = ["全期間"] + sorted(df_results["日付"].unique().tolist(), reverse=True)
        filter_date = st.selectbox("絞り込み", date_options)
        df_disp = df_results[df_results["日付"] == filter_date].reset_index(drop=True) \
            if filter_date != "全期間" else df_results.reset_index(drop=True)

        def style_result_row(row):
            bg = _VENUE_COLORS.get(str(row.get("競艇場", "")), _DEFAULT_BG)
            styles = [f"background-color: {bg};" for _ in row]
            if "的中" in row.index:
                idx = list(row.index).index("的中")
                hit = row.get("的中", "")
                if hit == "○":
                    styles[idx] = "background-color: #dcfce7; color: #16a34a; font-weight: bold;"
                elif hit == "×":
                    styles[idx] = "background-color: #fee2e2; color: #dc2626;"
            if "収支（円）" in row.index:
                idx = list(row.index).index("収支（円）")
                try:
                    v = int(str(row.get("収支（円）", 0)).replace(",", ""))
                    col = "#16a34a" if v > 0 else "#dc2626" if v < 0 else ""
                    if col:
                        styles[idx] = f"background-color: {bg}; color: {col}; font-weight: bold;"
                except Exception:
                    pass
            return styles

        st.dataframe(
            df_disp.style.apply(style_result_row, axis=1),
            use_container_width=True, height=500
        )


# ── TAB3: 分析グラフ ──
with tab3:
    if df_results.empty or stats["total"] == 0:
        st.info("📭 分析に必要なデータがまだ足りません。")
    else:
        df_r = df_results[~df_results["予想買い目"].isin(["", "（予想なし）", "見送り", "-"])].copy()
        base_layout = dict(
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            font=dict(color="#334155"),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📈 日別収支推移")
            daily = df_r.groupby("日付").apply(lambda g: pd.Series({
                "予想数": len(g),
                "払戻合計": g[g["的中"] == "○"]["実際の払戻"].apply(
                    lambda x: int(str(x).replace(",", "")) if str(x).replace(",", "").isdigit() else 0
                ).sum(),
            })).reset_index()
            daily["収支"] = daily["払戻合計"] - daily["予想数"] * 100
            fig1 = go.Figure()
            fig1.add_bar(x=daily["日付"], y=daily["収支"],
                         marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in daily["収支"]])
            fig1.update_layout(height=270, **base_layout)
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            st.markdown("#### 🏟️ 会場別的中率")
            vs = df_r.groupby("競艇場").apply(lambda g: pd.Series({
                "予想数": len(g), "的中数": (g["的中"] == "○").sum(),
            })).reset_index()
            vs["的中率"] = vs["的中数"] / vs["予想数"] * 100
            vs = vs.sort_values("的中率", ascending=True)
            fig2 = px.bar(vs, x="的中率", y="競艇場", orientation="h",
                          color="的中率", color_continuous_scale=["#fca5a5", "#86efac"],
                          text=vs["的中率"].apply(lambda x: f"{x:.1f}%"))
            fig2.update_layout(height=270, coloraxis_showscale=False, **base_layout)
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### ⭐ 信頼度別的中率")
            if "信頼度" in df_r.columns:
                cs = df_r.groupby("信頼度").apply(lambda g: pd.Series({
                    "予想数": len(g), "的中数": (g["的中"] == "○").sum(),
                })).reset_index()
                cs["的中率"] = cs["的中数"] / cs["予想数"] * 100
                fig3 = px.bar(cs, x="信頼度", y="的中率", color="信頼度",
                              color_discrete_map={"★★★": "#f59e0b", "★★☆": "#60a5fa", "★☆☆": "#cbd5e1"},
                              text=cs["的中率"].apply(lambda x: f"{x:.1f}%"))
                fig3.update_layout(height=260, showlegend=False, **base_layout)
                st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.markdown("#### 💹 累計回収率の推移")
            df_s = df_r.sort_values("日付").reset_index(drop=True)
            df_s["払戻"] = df_s.apply(
                lambda row: int(str(row["実際の払戻"]).replace(",", ""))
                if row["的中"] == "○" and str(row["実際の払戻"]).replace(",", "").isdigit() else 0, axis=1
            )
            df_s["累計回収率"] = df_s["払戻"].cumsum() / ((df_s.index + 1) * 100) * 100
            fig4 = go.Figure()
            fig4.add_scatter(x=list(range(1, len(df_s)+1)), y=df_s["累計回収率"],
                             mode="lines", line=dict(color="#1a6da0", width=2.5))
            fig4.add_hline(y=100, line_dash="dash", line_color="#dc2626",
                           annotation_text="損益分岐(100%)", annotation_font_color="#dc2626")
            fig4.update_layout(height=260, yaxis_title="回収率（%）",
                               xaxis_title="予想点数", **base_layout)
            st.plotly_chart(fig4, use_container_width=True)


st.markdown("---")
st.markdown('<div class="footer">🚤 BOAT LAB — 競艇 AI 予想システム ｜ データは60秒ごとに自動更新</div>',
            unsafe_allow_html=True)
