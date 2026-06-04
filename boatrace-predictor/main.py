"""
競艇3連単予想ツール - メインスクリプト
使い方:
  python main.py --mode download   # 今日のデータをダウンロード
  python main.py --mode train      # モデルを学習（初回・再学習時）
  python main.py --mode predict    # 本日の予想を生成してスプレッドシートに出力
  python main.py --mode backtest   # 過去データで回収率を検証
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"


def cmd_download(days_back: int = 0):
    """データダウンロード"""
    from src.collector.downloader import download_file, extract_lzh

    today = datetime.now()
    target = today - timedelta(days=days_back)

    print(f"=== データダウンロード: {target.strftime('%Y-%m-%d')} ===")

    # 今日の番組表
    b_path = download_file("B", target)
    if b_path:
        extract_lzh(b_path)

    # 昨日以前の競走成績
    if days_back > 0:
        k_path = download_file("K", target)
        if k_path:
            extract_lzh(k_path)
    else:
        # 昨日の競走成績
        yesterday = today - timedelta(days=1)
        k_path = download_file("K", yesterday)
        if k_path:
            extract_lzh(k_path)


def cmd_download_history(months: int = 3):
    """過去データの一括ダウンロード（学習用）"""
    from src.collector.downloader import download_range, extract_all

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30 * months)

    print(f"=== 過去データ一括ダウンロード: {start_date.strftime('%Y-%m-%d')} 〜 {end_date.strftime('%Y-%m-%d')} ===")
    print("※ サーバー負荷軽減のため3秒間隔でアクセスします。時間がかかります。")

    download_range("B", start_date, end_date, interval=3.0)
    download_range("K", start_date, end_date, interval=3.0)

    print("=== 解凍中 ===")
    extract_all("B")
    extract_all("K")


def cmd_train():
    """モデル学習"""
    import pandas as pd
    from src.collector.parser import parse_all_programs, parse_all_results
    from src.features.builder import build_features, compute_recent_form_lookup, compute_border_lookup
    from src.model.trainer import train_model, load_model
    from src.model.predictor import build_payout_lookup

    print("=== データ読み込み中 ===")
    df_program = parse_all_programs(RAW_DIR / "B")
    df_rank, df_payout = parse_all_results(RAW_DIR / "K")

    if df_program.empty or df_rank.empty:
        print("[ERROR] データがありません。先に python main.py --mode download_history を実行してください")
        return

    print(f"番組表: {len(df_program)}行, 競走成績: {len(df_rank)}行")

    # 直近調子・昇降級ボーダー判定の事前計算
    print("=== 直近調子・ボーダー判定を計算中 ===")
    recent_form_lookup = compute_recent_form_lookup(df_rank, lookback=10)
    # 学習データの最新日を基準に評価期間を決定
    latest_date = df_rank["date"].max() if "date" in df_rank.columns else ""
    border_lookup = compute_border_lookup(df_rank, latest_date) if latest_date else {}
    print(f"  直近調子: {len(recent_form_lookup)}選手 / ボーダー判定: {len(border_lookup)}選手")

    print("=== 特徴量生成中 ===")
    df_features = build_features(df_program, df_rank, df_payout,
                                 recent_form_lookup=recent_form_lookup,
                                 border_lookup=border_lookup)
    print(f"特徴量: {len(df_features)}レース分")

    print("=== モデル学習中 ===")
    train_model(df_features)

    print("=== 払戻ルックアップ生成中 ===")
    model = load_model()
    if model:
        build_payout_lookup(df_payout, model, df_features)
    else:
        print("[WARN] モデル読み込みに失敗したため払戻ルックアップをスキップしました")

    print("=== 学習完了 ===")


def _convert_beforeinfo(raw: dict) -> "pd.DataFrame":
    """
    fetch_beforeinfo_for_races の出力を DataFrame に変換する
    raw: {(venue_code, race_no): {1: {"exhibition_time": 6.82, "exhibition_st": 0.12}, ...}}
    """
    import pandas as pd
    from datetime import datetime
    records = []
    today = datetime.now().strftime("%Y-%m-%d")
    for (venue_code, race_no), boats in raw.items():
        for boat_no, info in boats.items():
            records.append({
                "date": today,
                "venue_code": venue_code,
                "race_no": race_no,
                "boat_no": boat_no,
                "exhibition_time": info.get("exhibition_time"),
                "exhibition_st": info.get("exhibition_st"),
            })
    return pd.DataFrame(records) if records else pd.DataFrame()


def _filter_upcoming_races(df_program: "pd.DataFrame", df_features: "pd.DataFrame", now: datetime) -> "pd.DataFrame":
    """
    発走時刻が現在時刻より後のレースのみを残す。
    発走時刻が取得できないレースは除外しない（全件残す）。
    """
    import pandas as pd

    now_str = now.strftime("%H:%M")

    if "scheduled_time" not in df_program.columns:
        return df_features

    # venue_code + race_no → scheduled_time のマッピング
    time_map = (
        df_program[["venue_code", "race_no", "scheduled_time"]]
        .dropna(subset=["scheduled_time"])
        .drop_duplicates(subset=["venue_code", "race_no"])
        .set_index(["venue_code", "race_no"])["scheduled_time"]
        .to_dict()
    )

    if not time_map:
        print("[INFO] 発走時刻データなし: 全レースを対象とします")
        return df_features

    def is_upcoming(row):
        key = (str(row["venue_code"]).zfill(2), int(row["race_no"]))
        t = time_map.get(key)
        if t is None:
            return True  # 時刻不明なレースは残す
        return t > now_str

    mask = df_features.apply(is_upcoming, axis=1)
    filtered = df_features[mask]

    skipped = len(df_features) - len(filtered)
    print(f"[INFO] 現在時刻: {now_str} / 発走済み除外: {skipped}レース / 残り: {len(filtered)}レース")
    return filtered


def cmd_predict(venue: str = None, race_no: int = None):
    """本日の予想生成"""
    from src.collector.downloader import download_file, extract_lzh, download_range, extract_all
    from src.collector.parser import parse_program, parse_all_results
    from src.collector.odds import fetch_odds_for_races
    from src.collector.beforeinfo import fetch_beforeinfo_for_races
    from src.features.builder import (build_features, add_course_advantage, get_feature_columns,
                                       compute_recent_form_lookup, compute_border_lookup)
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations
    from src.output.sheets import write_predictions

    today = datetime.now()

    # 今日の番組表をダウンロード
    b_path = download_file("B", today)
    if not b_path:
        print("[ERROR] 番組表のダウンロードに失敗しました")
        return

    txt_path = extract_lzh(b_path)
    if not txt_path:
        print("[ERROR] 番組表の解凍に失敗しました")
        return

    # パース
    df_program = parse_program(txt_path)
    if df_program.empty:
        print("[ERROR] 番組表のパースに失敗しました")
        return

    # 直近KファイルをDLして直近調子・ボーダー判定を計算
    import pandas as pd
    print("=== 直近成績データ取得・計算中 ===")
    k_start = today - timedelta(days=60)
    k_end = today - timedelta(days=1)
    download_range("K", k_start, k_end, interval=1.0)
    extract_all("K")
    k_raw_dir = RAW_DIR / "K"
    df_rank_hist, _ = parse_all_results(k_raw_dir) if k_raw_dir.exists() else (pd.DataFrame(), pd.DataFrame())
    if not df_rank_hist.empty:
        recent_form_lookup = compute_recent_form_lookup(df_rank_hist)
        border_lookup = compute_border_lookup(df_rank_hist, today.strftime("%Y-%m-%d"))
        print(f"  直近調子: {len(recent_form_lookup)}選手 / ボーダー判定: {len(border_lookup)}選手")
    else:
        recent_form_lookup, border_lookup = {}, {}

    # 直前情報（展示タイム）取得
    print("=== 直前情報（展示タイム）取得中 ===")
    df_features_tmp = build_features(df_program, pd.DataFrame(), pd.DataFrame(),
                                     recent_form_lookup=recent_form_lookup,
                                     border_lookup=border_lookup)
    df_beforeinfo_raw = fetch_beforeinfo_for_races(df_features_tmp, today)

    # 直前情報をDataFrame形式に変換
    df_beforeinfo = _convert_beforeinfo(df_beforeinfo_raw)

    # 特徴量生成（展示タイム込み）
    df_features = build_features(df_program, pd.DataFrame(), pd.DataFrame(), df_beforeinfo,
                                 recent_form_lookup=recent_form_lookup,
                                 border_lookup=border_lookup)

    # 未発走レースのみに絞り込む（競艇場・レース番号の指定がない場合）
    if venue is None and race_no is None:
        df_features = _filter_upcoming_races(df_program, df_features, today)

    # 競艇場・レース番号で絞り込む
    if venue is not None or race_no is not None:
        # 場名または場コードで検索
        from src.collector.parser import VENUE_CODES
        target_codes = []
        if venue is not None:
            # 場名（例: "福岡"）または場コード（例: "22"）で検索
            for code, name in VENUE_CODES.items():
                if venue in (code, name):
                    target_codes.append(code)
            if not target_codes:
                print(f"[ERROR] 競艇場が見つかりません: '{venue}'")
                print(f"  使用可能: {', '.join(VENUE_CODES.values())}")
                return
            mask = df_features["venue_code"].isin(target_codes)
            df_features = df_features[mask]

        if race_no is not None:
            df_features = df_features[df_features["race_no"] == race_no]

        if df_features.empty:
            print(f"[ERROR] 指定のレースが見つかりません（競艇場: {venue or '全場'} / レース: {race_no or '全レース'}R）")
            return
        print(f"[INFO] 絞り込み: {venue or '全場'} {race_no or '全'}R → {len(df_features)}行")

    # モデル読み込み
    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません。先に python main.py --mode train を実行してください")
        return

    # リアルタイムオッズ取得
    print("=== リアルタイムオッズ取得中 ===")
    all_live_odds = fetch_odds_for_races(df_features, today)

    # 予想生成
    print("=== 予想生成中 ===")
    recommendations = get_recommendations(model, df_features, all_live_odds=all_live_odds)

    # 結果表示
    print("\n【本日の推奨買い目】")
    print(recommendations[recommendations["combination"] != "見送り"].to_string(index=False))

    # スプレッドシートに出力
    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")
    if spreadsheet_id and spreadsheet_id != "your_spreadsheet_id_here" and Path(credentials_path).exists():
        write_predictions(spreadsheet_id, recommendations)
    else:
        print("\n[INFO] スプレッドシート出力をスキップしました（未設定）")
        print("      Google連携を設定する場合は .env ファイルを編集してください")


def cmd_auto():
    """自動予想モード（発走10分前に自動予想・結果記録）"""
    from src.scheduler.auto_runner import run_auto

    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")

    if not spreadsheet_id or spreadsheet_id == "your_spreadsheet_id_here":
        print("[ERROR] SPREADSHEET_ID が設定されていません（.env ファイルを確認してください）")
        return
    if not Path(credentials_path).exists():
        print("[ERROR] Google認証ファイルが見つかりません（GOOGLE_CREDENTIALS_PATH を確認してください）")
        return

    run_auto(spreadsheet_id, credentials_path)


def cmd_results(date_str: str = None):
    """今日の予想と実際の結果を照合して成績シートに書き込む"""
    import time
    from src.collector.parser import VENUE_CODES
    from src.collector.result_scraper import fetch_race_result
    from src.output.sheets import update_result_row, update_summary_sheet, apply_colors_to_results_sheet

    spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
    credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")
    if not spreadsheet_id or spreadsheet_id == "your_spreadsheet_id_here":
        print("[ERROR] SPREADSHEET_ID が設定されていません")
        return
    if not Path(credentials_path).exists():
        print("[ERROR] Google認証ファイルが見つかりません")
        return

    import gspread
    from google.oauth2.service_account import Credentials
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(spreadsheet_id)

    today = datetime.now()
    target_date = date_str or today.strftime("%Y-%m-%d")

    # 予想シートを読み込む
    try:
        sheet = spreadsheet.worksheet(target_date)
        rows = sheet.get_all_records()
    except Exception:
        print(f"[ERROR] 予想シート「{target_date}」が見つかりません")
        return

    # venue_name → venue_code の逆引きマップ
    name_to_code = {v: k for k, v in VENUE_CODES.items()}

    # 有効な予想行（見送り・ヘッダー以外）を抽出してレース単位でまとめる
    races = {}
    for row in rows:
        combo = str(row.get("買い目（3連単）", ""))
        venue_name = str(row.get("競艇場", ""))
        race_no_raw = row.get("レース", "")
        if combo in ("", "見送り", "-", "買い目（3連単）"):
            continue
        try:
            race_no = int(race_no_raw)
        except (ValueError, TypeError):
            continue
        key = (venue_name, race_no)
        races[key] = venue_name  # 重複排除用（結果取得は1回）

    if not races:
        print(f"[INFO] {target_date} の予想データが見つかりません")
        return

    print(f"=== 結果照合開始: {target_date} / {len(races)}レース ===")
    hit = 0
    total = 0

    for (venue_name, race_no), _ in sorted(races.items(), key=lambda x: x[0][1]):
        venue_code = name_to_code.get(venue_name)
        if not venue_code:
            print(f"  [SKIP] 場コード不明: {venue_name}")
            continue

        result = fetch_race_result(today, venue_code, race_no)
        if not result.get("available"):
            print(f"  [--] {venue_name} {race_no}R: 結果未確定（レース未終了の可能性）")
            continue

        combination = result["combination"]
        payout = result["payout"]
        print(f"  [OK] {venue_name} {race_no}R: 結果={combination} 払戻={payout:,}円")

        update_result_row(
            spreadsheet_id,
            date=target_date,
            venue_name=venue_name,
            race_no=race_no,
            actual_combination=combination,
            actual_payout=payout,
            credentials_path=credentials_path,
        )

        # 的中チェック（このレースの予想と照合）
        race_preds = [r for r in rows
                      if str(r.get("競艇場", "")) == venue_name
                      and str(r.get("レース", "")) == str(race_no)
                      and str(r.get("買い目（3連単）", "")) not in ("", "見送り", "-")]
        for pred in race_preds:
            total += 1
            if pred.get("買い目（3連単）") == combination:
                hit += 1

        time.sleep(1.0)

    print(f"\n=== 照合完了 ===")
    if total > 0:
        print(f"  予想数: {total}点 / 的中: {hit}点 / 的中率: {hit/total*100:.1f}%")
    apply_colors_to_results_sheet(spreadsheet_id, credentials_path)
    update_summary_sheet(spreadsheet_id, credentials_path)
    print(f"  → スプレッドシートの「成績」「サマリー」シートを更新しました")


def cmd_build_racer_stats():
    """選手の戦術スタイル統計（積極性スコア）を K-file から構築する"""
    from src.collector.racer_stats_builder import build_racer_style_lookup, save_racer_style_stats

    k_dir = RAW_DIR / "K"
    if not k_dir.exists() or not list(k_dir.glob("k*.txt")):
        print("[ERROR] K-fileがありません。先に python main.py --mode download_history を実行してください")
        return

    print("=== 選手戦術スタイル統計を構築中 ===")
    stats = build_racer_style_lookup(k_dir)
    if not stats:
        print("[ERROR] 統計の構築に失敗しました（データ不足の可能性）")
        return

    save_racer_style_stats(stats)
    print(f"=== 完了: {len(stats)}選手の戦術スタイルを保存しました ===")
    print("次回の予想から自動で反映されます（git pull → python main.py --mode auto）")


def cmd_backtest():
    """バックテスト（過去データで回収率検証）"""
    import pandas as pd
    from src.collector.parser import parse_all_programs, parse_all_results
    from src.features.builder import build_features
    from src.model.trainer import load_model
    from src.model.predictor import get_recommendations

    print("=== バックテスト開始 ===")
    df_program = parse_all_programs(RAW_DIR / "B")
    df_rank, df_payout = parse_all_results(RAW_DIR / "K")
    df_features = build_features(df_program, df_rank, df_payout)

    model = load_model()
    if not model:
        print("[ERROR] モデルが見つかりません")
        return

    # 直近60日分でバックテスト
    recent_dates = sorted(df_features["date"].unique())[-60:]
    df_recent = df_features[df_features["date"].isin(recent_dates)]
    print(f"対象期間: {recent_dates[0]} 〜 {recent_dates[-1]}（{len(recent_dates)}日間）")

    recommendations = get_recommendations(model, df_recent)
    if recommendations.empty:
        print("[WARN] 推奨買い目が生成されませんでした")
        return

    # 払戻データをキー付きdictに変換（高速化）
    payout_dict = {}
    for _, row in df_payout[df_payout["bet_type"] == "３連単"].iterrows():
        key = (row["date"], row.get("venue_name", ""), int(row["race_no"]), row["combination"])
        payout_dict[key] = int(row["payout"])

    tiers = ["本命穴", "超穴", "激穴"]
    stats = {t: {"bets": 0, "hits": 0, "ret": 0, "hit_combos": []} for t in tiers}
    stats["合計"] = {"bets": 0, "hits": 0, "ret": 0, "hit_combos": []}

    for _, rec in recommendations.iterrows():
        combo = rec.get("combination", "")
        if not combo or combo in ("見送り", "-", ""):
            continue
        tier = rec.get("tier", "")
        if tier not in tiers:
            continue

        key = (rec["date"], rec["venue_name"], int(rec["race_no"]), combo)
        payout = payout_dict.get(key, 0)

        stats[tier]["bets"] += 100
        stats["合計"]["bets"] += 100
        if payout > 0:
            stats[tier]["hits"] += 1
            stats[tier]["ret"] += payout
            stats[tier]["hit_combos"].append(
                f"  {rec['date']} {rec['venue_name']} {rec['race_no']}R "
                f"{combo} → ¥{payout:,}"
            )
            stats["合計"]["hits"] += 1
            stats["合計"]["ret"] += payout

    print(f"\n{'='*55}")
    print(f"{'tier':<8} {'点数':>5} {'的中':>5} {'的中率':>7} {'払戻合計':>10} {'回収率':>7}")
    print(f"{'-'*55}")
    for t in tiers + ["合計"]:
        s = stats[t]
        b, h, r = s["bets"], s["hits"], s["ret"]
        hr = f"{h/b*100*100:.2f}%" if b > 0 else "  -  "  # 100円×点数
        roi = f"{r/b*100:.1f}%" if b > 0 else "  -  "
        marker = " ◀" if t == "合計" else ""
        print(f"{t:<8} {b//100:>5} {h:>5} {hr:>7} {r:>10,} {roi:>7}{marker}")

    print(f"{'='*55}")

    for t in tiers:
        if stats[t]["hit_combos"]:
            print(f"\n【{t} 的中内訳】")
            for line in stats[t]["hit_combos"]:
                print(line)

    # モデル精度チェック: 神熱ラベルの的中率
    hot_recs = recommendations[recommendations["bet_label"] == "神熱"]
    if not hot_recs.empty:
        hot_bets, hot_hits, hot_ret = 0, 0, 0
        for _, rec in hot_recs.iterrows():
            combo = rec.get("combination", "")
            if not combo or combo in ("見送り", "-", ""):
                continue
            key = (rec["date"], rec["venue_name"], int(rec["race_no"]), combo)
            payout = payout_dict.get(key, 0)
            hot_bets += 100
            if payout > 0:
                hot_hits += 1
                hot_ret += payout
        hot_roi = f"{hot_ret/hot_bets*100:.1f}%" if hot_bets > 0 else "-"
        print(f"\n【神熱ラベルのみ】 {hot_bets//100}点 / {hot_hits}的中 / ROI {hot_roi}")

    # ── 荒れ条件分析: 条件クリア時に実際に100倍以上が出るか ──
    print(f"\n{'='*60}")
    print("【荒れ条件分析】荒れ条件クリア時の実際の払戻分布")
    print(f"{'='*60}")

    from src.model.predictor import _calc_arare_score, ARARE_MIN_SCORE

    # 払戻を (date, venue, race_no) → 払戻額 に事前変換（高速化）
    race_payout_map = {}
    for (d, vn, rn, _), v in payout_dict.items():
        rk = (d, vn, rn)
        if rk not in race_payout_map or v > race_payout_map[rk]:
            race_payout_map[rk] = v

    # レースごとに1回だけ処理（arare分析 + PT別を同時に集計）
    race_keys_seen = set()
    arare_races     = []
    non_arare_races = []
    pt_buckets_bt   = {pt: [] for pt in range(0, 7)}
    pt_buckets_bt["7以上"] = []

    for _, row in df_recent.iterrows():
        date_    = row.get("date", "")
        venue_   = row.get("venue_name", "")
        race_no_ = int(row.get("race_no", 0))
        key_ = (date_, venue_, race_no_)
        if key_ in race_keys_seen:
            continue
        race_keys_seen.add(key_)

        score, _ = _calc_arare_score(row, None)
        winning_payout = race_payout_map.get(key_, 0)

        entry = {"score": score, "payout": winning_payout}
        if score >= ARARE_MIN_SCORE:
            arare_races.append(entry)
        else:
            non_arare_races.append(entry)

        bucket_key = score if score <= 6 else "7以上"
        if bucket_key in pt_buckets_bt:
            pt_buckets_bt[bucket_key].append(winning_payout)

    def _payout_stats(races, label):
        total = len(races)
        if total == 0:
            print(f"\n{label}: データなし")
            return
        over100  = sum(1 for r in races if r["payout"] >= 10_000)
        over150  = sum(1 for r in races if r["payout"] >= 15_000)
        over200  = sum(1 for r in races if r["payout"] >= 20_000)
        over300  = sum(1 for r in races if r["payout"] >= 30_000)
        under100 = total - over100
        avg_pay  = sum(r["payout"] for r in races) / total
        print(f"\n{label}（{total}レース）")
        print(f"  払戻 100倍以上(≥¥10,000): {over100}R / {over100/total*100:.1f}%")
        print(f"  払戻 150倍以上(≥¥15,000): {over150}R / {over150/total*100:.1f}%")
        print(f"  払戻 200倍以上(≥¥20,000): {over200}R / {over200/total*100:.1f}%")
        print(f"  払戻 300倍以上(≥¥30,000): {over300}R / {over300/total*100:.1f}%")
        print(f"  払戻 100倍未満(本命決着) : {under100}R / {under100/total*100:.1f}%")
        print(f"  平均払戻              : ¥{avg_pay:,.0f}")

    _payout_stats(arare_races,     f"■ 荒れ条件クリア(スコア≥{ARARE_MIN_SCORE})")
    _payout_stats(non_arare_races, f"■ 荒れ条件未達 (スコア<{ARARE_MIN_SCORE})")

    if arare_races and non_arare_races:
        a_rate = sum(1 for r in arare_races     if r["payout"] >= 10_000) / len(arare_races)
        n_rate = sum(1 for r in non_arare_races if r["payout"] >= 10_000) / len(non_arare_races)
        print(f"\n→ 荒れ条件クリア時の100倍以上率: {a_rate*100:.1f}%")
        print(f"→ 荒れ条件未達時の100倍以上率 : {n_rate*100:.1f}%")
        if n_rate > 0:
            print(f"→ 荒れ条件クリアの優位性     : {a_rate/n_rate:.2f}倍")

    # ── 荒れPT別の万舟出現率 ──
    print(f"\n{'='*75}")
    print("【荒れPT別 払戻分布】")
    print(f"{'='*75}")

    print(f"{'PT':<8} {'レース数':>8} {'中穴率20-99倍':>13} {'万舟率100倍+':>12} {'150倍+率':>9} {'300倍+率':>9} {'平均払戻':>10}")
    print(f"{'-'*75}")
    for pt_key in list(range(0, 7)) + ["7以上"]:
        races_in_bucket = pt_buckets_bt[pt_key]
        total = len(races_in_bucket)
        if total == 0:
            label = f"{pt_key}PT" if isinstance(pt_key, int) else f"{pt_key}（神熱）"
            print(f"{label:<8} {'0':>8} {'-':>13} {'-':>12} {'-':>9} {'-':>9} {'-':>10}")
            continue
        chuana  = sum(1 for p in races_in_bucket if 2_000 <= p < 10_000)
        over100 = sum(1 for p in races_in_bucket if p >= 10_000)
        over150 = sum(1 for p in races_in_bucket if p >= 15_000)
        over300 = sum(1 for p in races_in_bucket if p >= 30_000)
        avg_pay = sum(races_in_bucket) / total
        label = f"{pt_key}PT" if isinstance(pt_key, int) else f"{pt_key}（神熱）"
        print(
            f"{label:<8} {total:>8} "
            f"{chuana/total*100:>12.1f}% "
            f"{over100/total*100:>11.1f}% "
            f"{over150/total*100:>8.1f}% "
            f"{over300/total*100:>8.1f}% "
            f"{avg_pay:>10,.0f}"
        )
    print(f"{'='*75}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競艇3連単予想ツール")
    parser.add_argument(
        "--mode",
        choices=["download", "download_history", "train", "predict", "backtest", "auto", "results", "nirentan", "build_racer_stats"],
        required=True,
        help="実行モード"
    )
    parser.add_argument("--months", type=int, default=3, help="過去データ取得月数（download_historyモード用）")
    parser.add_argument("--days-back", type=int, default=0, help="何日前のデータを取得するか（downloadモード用）")
    parser.add_argument("--venue", type=str, default=None, help="競艇場名または場コード（例: 福岡 または 22）")
    parser.add_argument("--race", type=int, default=None, help="レース番号（例: 6）")
    parser.add_argument("--date", type=str, default=None, help="分析対象日（例: 2026-04-18）nirentanモード用")
    args = parser.parse_args()

    if args.mode == "download":
        cmd_download(args.days_back)
    elif args.mode == "download_history":
        cmd_download_history(args.months)
    elif args.mode == "train":
        cmd_train()
    elif args.mode == "predict":
        cmd_predict(venue=args.venue, race_no=args.race)
    elif args.mode == "backtest":
        cmd_backtest()
    elif args.mode == "auto":
        cmd_auto()
    elif args.mode == "results":
        cmd_results()
    elif args.mode == "build_racer_stats":
        cmd_build_racer_stats()
    elif args.mode == "nirentan":
        from src.output.sheets import analyze_nirentan
        target_date = args.date or datetime.now().strftime("%Y-%m-%d")
        spreadsheet_id = os.environ.get("SPREADSHEET_ID", "")
        credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH")
        analyze_nirentan(spreadsheet_id, target_date, credentials_path)
