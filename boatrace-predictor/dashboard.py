"""
BOAT LAB ダッシュボード
Streamlit製のローカルブラウザUI（ダークモード）
"""

import os
from datetime import datetime
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

# ─── ダークテーマ CSS ───
st.markdown("""
<style>
    /* ── 全体背景 ── */
    .stApp, .stApp > div {
        background-color: #0e1117 !important;
        color: #e0e0e0 !important;
    }
    /* サイドバー */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* タブ */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e !important;
        border-radius: 6px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }

    /* セレクトボックス・入力 */
    .stSelectbox > div > div, .stDateInput > div > div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }

    /* チェックボックス */
    .stCheckbox label { color: #c9d1d9 !important; }

    /* スライダー */
    .stSlider { color: #c9d1d9 !important; }

    /* ボタン */
    .stButton > button {
        background-color: #1f6feb !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background-color: #388bfd !important; }

    /* 区切り線 */
    hr { border-color: #30363d !important; }

    /* info/warning ボックス */
    .stAlert { background-color: #161b22 !important; border-color: #30363d !important; color: #c9d1d9 !important; }

    /* ── ヘッダー ── */
    .boat-header {
        background: linear-gradient(135deg, #0d1117, #1a3a5c);
        border: 1px solid #1f6feb;
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(31,111,235,0.2);
    }
    .boat-header h1 { margin: 0; font-size: 2.6rem; letter-spacing: 6px; color: #58a6ff; }
    .boat-header p  { margin: 0.4rem 0 0; font-size: 0.9rem; color: #8b949e; letter-spacing: 2px; }

    /* ── メトリクスカード ── */
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #1f6feb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.9rem; font-weight: bold; color: #58a6ff; }
    .metric-sub   { font-size: 0.75rem; color: #6e7681; margin-top: 0.2rem; }

    /* フッター */
    .footer { text-align: center; color: #6e7681; font-size: 0.8rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─── 会場カラー（ダーク版） ───
_VENUE_COLORS = {
    # 関東 - 青系
    "桐生":   "#0d2b45", "戸田": "#0a2540", "江戸川": "#07203b",
    "平和島": "#051b36", "多摩川": "#041631",
    # 中部 - 緑系
    "浜名湖": "#0d2b0d", "蒲郡": "#0a2509", "常滑": "#082006",
    "津": "#061b04", "三国": "#051602",
    # 近畿 - 紫系
    "びわこ": "#1e0b3b", "住之江": "#1a0935", "尼崎": "#160730",
    # 中国・四国 - 黄系
    "鳴門": "#2e2600", "丸亀": "#292100", "児島": "#241c00",
    "宮島": "#1f1700",
    # 九州 - オレンジ系
    "徳山": "#2e1500", "下関": "#291100", "若松": "#240e00",
    "芦屋": "#1f0b00", "福岡": "#1a0800", "唐津": "#150500",
    "大村": "#100200",
}
_DEFAULT_DARK = "#161b22"


def _venue_row_style(row):
    """行全体に会場カラーを適用"""
    venue = row.get("競艇場", "") if hasattr(row, "get") else ""
    bg = _VENUE_COLORS.get(str(venue), _DEFAULT_DARK)
    return [f"background-color: {bg}; color: #e0e0e0;" for _ in row]


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
    client = gspread.authorize(creds)
    return client.open_by_key(spreadsheet_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_predictions(sheet_name: str) -> pd.DataFrame:
    try:
        ss = get_spreadsheet()
        if ss is None:
            return pd.DataFrame()
        ws = ss.worksheet(sheet_name)
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_results() -> pd.DataFrame:
    try:
        ss = get_spreadsheet()
        if ss is None:
            return pd.DataFrame()
        ws = ss.worksheet("成績")
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()


def calc_stats(df: pd.DataFrame) -> dict:
    if df.empty or "予想買い目" not in df.columns:
        return {"total": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0}
    df = df[~df["予想買い目"].isin(["", "（予想なし）", "見送り", "-"])].copy()
    total = len(df)
    if total == 0:
        return {"total": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0}
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
    st.markdown("### 📅 日付選択")
    today = datetime.now().date()
    selected_date = st.date_input("日付", value=today, max_value=today, label_visibility="collapsed")
    date_str = selected_date.strftime("%Y-%m-%d")

    st.markdown("---")
    if st.button("🔄 データを更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 表示設定")
    show_all_confidence = st.checkbox("★☆☆も表示する", value=True)


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
    c = "#3fb950" if hr >= 10 else "#f85149"
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">🎯 的中率</div>
        <div class="metric-value" style="color:{c}">{hr:.1f}%</div>
        <div class="metric-sub">ランダム比較: 0.8%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    roi = stats['roi']
    c = "#3fb950" if roi >= 100 else "#f85149"
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">💹 回収率</div>
        <div class="metric-value" style="color:{c}">{roi:.1f}%</div>
        <div class="metric-sub">100%超えで黒字</div>
    </div>""", unsafe_allow_html=True)
with c4:
    profit = stats['profit']
    c = "#3fb950" if profit >= 0 else "#f85149"
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

        disp_cols = [c for c in ["競艇場", "レース", "買い目（3連単）", "信頼度", "的中確率", "オッズ"] if c in df_show.columns]
        df_show = df_show[disp_cols].reset_index(drop=True)

        def style_pred_row(row):
            bg = _VENUE_COLORS.get(str(row.get("競艇場", "")), _DEFAULT_DARK)
            # 信頼度で上書き
            conf = row.get("信頼度", "")
            if conf == "★★★":
                bg = "#2d2000"
            styles = [f"background-color: {bg}; color: #e0e0e0;" for _ in row]
            # 信頼度列だけ色を変える
            if "信頼度" in row.index:
                idx = list(row.index).index("信頼度")
                if conf == "★★★":
                    styles[idx] = "background-color: #2d2000; color: #f0a500; font-weight: bold;"
                elif conf == "★★☆":
                    styles[idx] = f"background-color: {bg}; color: #8b949e;"
                else:
                    styles[idx] = f"background-color: {bg}; color: #484f58;"
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

        df_disp = df_results.copy()
        if filter_date != "全期間":
            df_disp = df_disp[df_disp["日付"] == filter_date]
        df_disp = df_disp.reset_index(drop=True)

        def style_result_row(row):
            bg = _VENUE_COLORS.get(str(row.get("競艇場", "")), _DEFAULT_DARK)
            styles = [f"background-color: {bg}; color: #e0e0e0;" for _ in row]
            # 的中列
            if "的中" in row.index:
                idx = list(row.index).index("的中")
                hit = row.get("的中", "")
                if hit == "○":
                    styles[idx] = "background-color: #0d3b1a; color: #3fb950; font-weight: bold;"
                elif hit == "×":
                    styles[idx] = "background-color: #3b0d0d; color: #f85149;"
            # 収支列
            if "収支（円）" in row.index:
                idx = list(row.index).index("収支（円）")
                try:
                    v = int(str(row.get("収支（円）", 0)).replace(",", ""))
                    if v > 0:
                        styles[idx] = f"background-color: {bg}; color: #3fb950; font-weight: bold;"
                    elif v < 0:
                        styles[idx] = f"background-color: {bg}; color: #f85149;"
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

        DARK_BG  = "#0e1117"
        DARK_PAPER = "#161b22"
        GRID_COLOR = "#30363d"
        TEXT_COLOR = "#c9d1d9"

        base_layout = dict(
            paper_bgcolor=DARK_PAPER,
            plot_bgcolor=DARK_BG,
            font=dict(color=TEXT_COLOR),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        )

        col_a, col_b = st.columns(2)

        # 日別収支推移
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
            fig1.add_bar(
                x=daily["日付"], y=daily["収支"],
                marker_color=["#3fb950" if v >= 0 else "#f85149" for v in daily["収支"]],
            )
            fig1.update_layout(height=270, **base_layout)
            st.plotly_chart(fig1, use_container_width=True)

        # 会場別的中率
        with col_b:
            st.markdown("#### 🏟️ 会場別的中率")
            vs = df_r.groupby("競艇場").apply(lambda g: pd.Series({
                "予想数": len(g),
                "的中数": (g["的中"] == "○").sum(),
            })).reset_index()
            vs["的中率"] = vs["的中数"] / vs["予想数"] * 100
            vs = vs.sort_values("的中率", ascending=True)
            fig2 = px.bar(vs, x="的中率", y="競艇場", orientation="h",
                          color="的中率", color_continuous_scale=["#f85149", "#3fb950"],
                          text=vs["的中率"].apply(lambda x: f"{x:.1f}%"))
            fig2.update_layout(height=270, coloraxis_showscale=False, **base_layout)
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)

        # 信頼度別的中率
        with col_c:
            st.markdown("#### ⭐ 信頼度別的中率")
            if "信頼度" in df_r.columns:
                cs = df_r.groupby("信頼度").apply(lambda g: pd.Series({
                    "予想数": len(g), "的中数": (g["的中"] == "○").sum(),
                })).reset_index()
                cs["的中率"] = cs["的中数"] / cs["予想数"] * 100
                fig3 = px.bar(cs, x="信頼度", y="的中率",
                              color="信頼度",
                              color_discrete_map={"★★★": "#f0a500", "★★☆": "#58a6ff", "★☆☆": "#484f58"},
                              text=cs["的中率"].apply(lambda x: f"{x:.1f}%"))
                fig3.update_layout(height=260, showlegend=False, **base_layout)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("信頼度データがありません")

        # 累計回収率の推移
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
                             mode="lines", line=dict(color="#58a6ff", width=2))
            fig4.add_hline(y=100, line_dash="dash", line_color="#f85149",
                           annotation_text="損益分岐(100%)", annotation_font_color="#f85149")
            fig4.update_layout(height=260, yaxis_title="回収率（%）",
                               xaxis_title="予想点数", **base_layout)
            st.plotly_chart(fig4, use_container_width=True)


# フッター
st.markdown("---")
st.markdown(
    '<div class="footer">🚤 BOAT LAB - 競艇AI予想システム ｜ データは60秒ごとに自動更新</div>',
    unsafe_allow_html=True,
)
