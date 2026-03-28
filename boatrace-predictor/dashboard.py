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

# ─── カスタムCSS ───
st.markdown("""
<style>
    /* ヘッダー */
    .boat-header {
        background: linear-gradient(135deg, #0a3d62, #1a6da0);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .boat-header h1 { margin: 0; font-size: 2.4rem; letter-spacing: 4px; }
    .boat-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.8; }

    /* メトリクスカード */
    .metric-box {
        background: #f8f9fa;
        border-left: 4px solid #1a6da0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.8rem; color: #666; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #0a3d62; }
    .metric-sub   { font-size: 0.8rem; color: #888; }

    /* 予想カード */
    .race-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .race-title { font-size: 1rem; font-weight: bold; color: #0a3d62; margin-bottom: 0.5rem; }
    .combo-row  { display: flex; align-items: center; gap: 0.8rem; padding: 0.3rem 0; border-bottom: 1px solid #f0f0f0; }
    .combo-row:last-child { border-bottom: none; }
    .combo      { font-size: 1.1rem; font-weight: bold; font-family: monospace; min-width: 80px; }
    .star-3     { color: #f39c12; }
    .star-2     { color: #95a5a6; }
    .star-1     { color: #bdc3c7; }
    .prob-badge { background: #eaf4ff; color: #1a6da0; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
    .odds-badge { background: #fff3e0; color: #e67e22; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }

    /* 的中バッジ */
    .hit-badge  { background: #d4efdf; color: #1e8449; border-radius: 4px; padding: 2px 10px; font-weight: bold; }
    .miss-badge { background: #fadbd8; color: #c0392b; border-radius: 4px; padding: 2px 10px; }

    /* サイドバー */
    .sidebar-title { font-size: 1.1rem; font-weight: bold; color: #0a3d62; margin-bottom: 1rem; }

    /* プラス収支 */
    .profit-plus { color: #1e8449; font-weight: bold; }
    .profit-minus { color: #c0392b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


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
    """予想シートを読み込む"""
    try:
        ss = get_spreadsheet()
        if ss is None:
            return pd.DataFrame()
        ws = ss.worksheet(sheet_name)
        records = ws.get_all_records()
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_results() -> pd.DataFrame:
    """成績シートを読み込む"""
    try:
        ss = get_spreadsheet()
        if ss is None:
            return pd.DataFrame()
        ws = ss.worksheet("成績")
        records = ws.get_all_records()
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def calc_stats(df_results: pd.DataFrame) -> dict:
    """成績から統計を計算する"""
    if df_results.empty:
        return {"total": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0}

    df = df_results[~df_results["予想買い目"].isin(["", "（予想なし）", "見送り", "-"])].copy()
    total = len(df)
    if total == 0:
        return {"total": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0}

    hits = (df["的中"] == "○").sum()
    total_bet = total * 100
    total_return = df[df["的中"] == "○"]["実際の払戻"].apply(
        lambda x: int(str(x).replace(",", "")) if str(x).replace(",", "").isdigit() else 0
    ).sum()
    profit = total_return - total_bet
    roi = total_return / total_bet * 100 if total_bet > 0 else 0.0

    return {
        "total": total,
        "hits": int(hits),
        "hit_rate": hits / total * 100,
        "roi": roi,
        "profit": profit,
    }


# ─── ヘッダー ───
st.markdown("""
<div class="boat-header">
    <h1>🚤 BOAT LAB</h1>
    <p>競艇AI予想ダッシュボード</p>
</div>
""", unsafe_allow_html=True)


# ─── サイドバー ───
with st.sidebar:
    st.markdown('<div class="sidebar-title">📅 日付選択</div>', unsafe_allow_html=True)
    today = datetime.now().date()
    selected_date = st.date_input("日付", value=today, max_value=today)
    date_str = selected_date.strftime("%Y-%m-%d")

    st.markdown("---")
    if st.button("🔄 データを更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**📊 表示設定**")
    show_all_confidence = st.checkbox("★☆☆も表示する", value=True)
    min_prob = st.slider("最低確率フィルター", 0, 20, 0, step=1)


# ─── データ読み込み ───
with st.spinner("データ読み込み中..."):
    df_pred = load_predictions(date_str)
    df_results = load_results()

stats = calc_stats(df_results)


# ─── メトリクス行 ───
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">📋 累計予想点数</div>
        <div class="metric-value">{stats['total']}</div>
        <div class="metric-sub">的中 {stats['hits']} 回</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    hr = stats['hit_rate']
    hr_color = "#1e8449" if hr >= 10 else "#c0392b"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">🎯 的中率</div>
        <div class="metric-value" style="color:{hr_color}">{hr:.1f}%</div>
        <div class="metric-sub">3連単ランダム比較: 0.8%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    roi = stats['roi']
    roi_color = "#1e8449" if roi >= 100 else "#c0392b"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">💹 回収率</div>
        <div class="metric-value" style="color:{roi_color}">{roi:.1f}%</div>
        <div class="metric-sub">100%超えで黒字</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    profit = stats['profit']
    p_color = "#1e8449" if profit >= 0 else "#c0392b"
    p_sign = "+" if profit >= 0 else ""
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">💰 累計収支</div>
        <div class="metric-value" style="color:{p_color}">{p_sign}¥{profit:,}</div>
        <div class="metric-sub">100円/点で計算</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── タブ ───
tab1, tab2, tab3 = st.tabs(["🎯 本日の予想", "📋 成績一覧", "📈 分析グラフ"])


# ════════════════════════════════
# TAB1: 本日の予想
# ════════════════════════════════
with tab1:
    if df_pred.empty:
        st.info(f"📭 {date_str} の予想データがまだありません。\n\n`python main.py --mode auto` を実行してください。")
    else:
        # フィルター適用
        df_show = df_pred.copy()
        if not show_all_confidence:
            df_show = df_show[df_show.get("信頼度", "★☆☆") != "★☆☆"]
        if min_prob > 0 and "的中確率" in df_show.columns:
            df_show = df_show[
                df_show["的中確率"].apply(
                    lambda x: float(str(x).replace("%", "")) >= min_prob
                    if str(x).replace("%", "").replace(".", "").isdigit() else True
                )
            ]

        # 会場・レースごとにグループ化
        if "競艇場" in df_show.columns and "レース" in df_show.columns:
            groups = df_show.groupby(["競艇場", "レース"], sort=False)
            cols = st.columns(2)
            col_idx = 0
            for (venue, race_no), group in groups:
                with cols[col_idx % 2]:
                    rows_html = ""
                    for _, row in group.iterrows():
                        combo = row.get("買い目（3連単）", "-")
                        prob  = row.get("的中確率", "-")
                        odds  = row.get("オッズ", "-")
                        conf  = row.get("信頼度", "★☆☆")
                        star_class = "star-3" if conf == "★★★" else "star-2" if conf == "★★☆" else "star-1"
                        rows_html += f"""
                        <div class="combo-row">
                            <span class="{star_class}">{conf}</span>
                            <span class="combo">{combo}</span>
                            <span class="prob-badge">確率 {prob}</span>
                            <span class="odds-badge">{odds}</span>
                        </div>
                        """
                    st.markdown(f"""
                    <div class="race-card">
                        <div class="race-title">🏁 {venue} {race_no}R</div>
                        {rows_html}
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1
        else:
            st.dataframe(df_show, use_container_width=True)


# ════════════════════════════════
# TAB2: 成績一覧
# ════════════════════════════════
with tab2:
    if df_results.empty:
        st.info("📭 成績データがまだありません。")
    else:
        # 日付フィルター
        date_options = ["全期間"] + sorted(df_results["日付"].unique().tolist(), reverse=True)
        filter_date = st.selectbox("絞り込み", date_options)

        df_disp = df_results.copy()
        if filter_date != "全期間":
            df_disp = df_disp[df_disp["日付"] == filter_date]

        # スタイル適用
        def style_hit(val):
            if val == "○":
                return "background-color: #d4efdf; color: #1e8449; font-weight: bold;"
            elif val == "×":
                return "background-color: #fadbd8; color: #c0392b;"
            return ""

        def style_profit(val):
            try:
                v = int(str(val).replace(",", "").replace("¥", ""))
                if v > 0:
                    return "color: #1e8449; font-weight: bold;"
                elif v < 0:
                    return "color: #c0392b;"
            except Exception:
                pass
            return ""

        styled = df_disp.style.applymap(style_hit, subset=["的中"]).applymap(
            style_profit, subset=["収支（円）"]
        )
        st.dataframe(styled, use_container_width=True, height=500)


# ════════════════════════════════
# TAB3: 分析グラフ
# ════════════════════════════════
with tab3:
    if df_results.empty or stats["total"] == 0:
        st.info("📭 分析に必要なデータがまだ足りません。")
    else:
        df_r = df_results[~df_results["予想買い目"].isin(["", "（予想なし）", "見送り", "-"])].copy()

        col_a, col_b = st.columns(2)

        # ── 日別収支推移 ──
        with col_a:
            st.markdown("#### 📈 日別収支推移")
            daily = df_r.groupby("日付").apply(lambda g: pd.Series({
                "予想数": len(g),
                "的中数": (g["的中"] == "○").sum(),
                "払戻合計": g[g["的中"] == "○"]["実際の払戻"].apply(
                    lambda x: int(str(x).replace(",", "")) if str(x).replace(",", "").isdigit() else 0
                ).sum(),
            })).reset_index()
            daily["収支"] = daily["払戻合計"] - daily["予想数"] * 100
            daily["回収率"] = daily["払戻合計"] / (daily["予想数"] * 100) * 100

            fig1 = go.Figure()
            fig1.add_bar(
                x=daily["日付"], y=daily["収支"],
                marker_color=["#1e8449" if v >= 0 else "#c0392b" for v in daily["収支"]],
                name="収支"
            )
            fig1.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                height=280,
                yaxis_title="収支（円）",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig1, use_container_width=True)

        # ── 会場別的中率 ──
        with col_b:
            st.markdown("#### 🏟️ 会場別的中率")
            venue_stats = df_r.groupby("競艇場").apply(lambda g: pd.Series({
                "予想数": len(g),
                "的中数": (g["的中"] == "○").sum(),
            })).reset_index()
            venue_stats["的中率"] = venue_stats["的中数"] / venue_stats["予想数"] * 100
            venue_stats = venue_stats.sort_values("的中率", ascending=True)

            fig2 = px.bar(
                venue_stats, x="的中率", y="競艇場", orientation="h",
                color="的中率", color_continuous_scale=["#fadbd8", "#d4efdf"],
                text=venue_stats["的中率"].apply(lambda x: f"{x:.1f}%"),
            )
            fig2.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                height=280,
                plot_bgcolor="white",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)

        # ── 信頼度別的中率 ──
        with col_c:
            st.markdown("#### ⭐ 信頼度別的中率")
            if "信頼度" in df_r.columns:
                conf_stats = df_r.groupby("信頼度").apply(lambda g: pd.Series({
                    "予想数": len(g),
                    "的中数": (g["的中"] == "○").sum(),
                })).reset_index()
                conf_stats["的中率"] = conf_stats["的中数"] / conf_stats["予想数"] * 100

                fig3 = px.bar(
                    conf_stats, x="信頼度", y="的中率",
                    color="信頼度",
                    color_discrete_map={"★★★": "#f39c12", "★★☆": "#95a5a6", "★☆☆": "#dfe6e9"},
                    text=conf_stats["的中率"].apply(lambda x: f"{x:.1f}%"),
                )
                fig3.update_layout(
                    margin=dict(l=0, r=0, t=20, b=0),
                    height=260,
                    plot_bgcolor="white",
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("信頼度データがありません")

        # ── 回収率推移（累計）──
        with col_d:
            st.markdown("#### 💹 累計回収率の推移")
            df_sorted = df_r.sort_values("日付").reset_index(drop=True)
            df_sorted["累計賭け金"] = (df_sorted.index + 1) * 100
            df_sorted["払戻"] = df_sorted.apply(
                lambda row: int(str(row["実際の払戻"]).replace(",", ""))
                if row["的中"] == "○" and str(row["実際の払戻"]).replace(",", "").isdigit()
                else 0, axis=1
            )
            df_sorted["累計払戻"] = df_sorted["払戻"].cumsum()
            df_sorted["累計回収率"] = df_sorted["累計払戻"] / df_sorted["累計賭け金"] * 100

            fig4 = go.Figure()
            fig4.add_scatter(
                x=list(range(1, len(df_sorted) + 1)),
                y=df_sorted["累計回収率"],
                mode="lines",
                line=dict(color="#1a6da0", width=2),
                name="回収率",
            )
            fig4.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="損益分岐(100%)")
            fig4.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                height=260,
                yaxis_title="回収率（%）",
                xaxis_title="予想点数",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig4, use_container_width=True)

# ─── フッター ───
st.markdown("---")
st.markdown(
    "<center><small>🚤 BOAT LAB - 競艇AI予想システム ｜ データは60秒ごとに自動更新</small></center>",
    unsafe_allow_html=True,
)
