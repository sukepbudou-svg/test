"""
万舟パターン分析スクリプト
成績18シートから「万舟が出たレース」の傾向を分析し、
勝負所ロジック改善のヒントを抽出する。

実行方法:
  python analyze_manza.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MANZA_THRESHOLD = 10000  # 万舟の基準（払戻 ≥ 10,000円 = 100倍以上）
BIG_THRESHOLD   = 30000  # 大穴の基準（300倍以上）


def get_sheet_data():
    import gspread
    from google.oauth2.service_account import Credentials

    creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH")
    spreadsheet_id = os.environ.get("SPREADSHEET_ID")

    if not creds_path or not spreadsheet_id:
        print("[ERROR] .env に GOOGLE_CREDENTIALS_PATH と SPREADSHEET_ID が必要です")
        sys.exit(1)

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(spreadsheet_id)

    # 成績18シート
    try:
        sheet = spreadsheet.worksheet("成績18")
    except Exception:
        print("[ERROR] 成績18シートが見つかりません")
        sys.exit(1)

    records = sheet.get_all_records()
    print(f"[OK] 成績18 から {len(records)} 行読み込み")
    return records


def parse_records(records):
    """
    成績18の列:
    日付 | 競艇場 | レース | 狙い | 予想買い目 |
    実際の結果 | イン逃げ率 | 実際の払戻 | 的中 | 収支 |
    本日レース数 | 勝負推奨 | 荒れPT
    """
    races = {}  # key: (日付, 競艇場, レース) → race info

    for row in records:
        date    = str(row.get("日付", "")).strip()
        venue   = str(row.get("競艇場", "")).strip()
        race_no = str(row.get("レース", "")).strip()

        if not date or not venue or not race_no:
            continue

        key = (date, venue, race_no)

        try:
            payout = int(str(row.get("実際の払戻", 0)).replace(",", "").replace("¥", "") or 0)
        except ValueError:
            payout = 0

        try:
            arare_pt = int(str(row.get("荒れPT", 0)) or 0)
        except ValueError:
            arare_pt = 0

        actual_combo = str(row.get("実際の結果", "")).strip()
        bet_label    = str(row.get("勝負推奨", "")).strip()
        nigerate     = str(row.get("イン逃げ率", "")).strip()

        # 同じレースに複数の予想行がある場合、払戻・実際の結果は共通なので上書きでOK
        if key not in races:
            races[key] = {
                "date":        date,
                "venue":       venue,
                "race_no":     race_no,
                "payout":      payout,
                "arare_pt":    arare_pt,
                "actual_combo": actual_combo,
                "bet_label":   bet_label,
                "nigerate":    nigerate,
                "hit":         False,
                "preds":       [],
            }
        else:
            if payout > 0:
                races[key]["payout"] = payout
            if actual_combo:
                races[key]["actual_combo"] = actual_combo
            if arare_pt > 0:
                races[key]["arare_pt"] = arare_pt
            if bet_label:
                races[key]["bet_label"] = bet_label

        pred_combo = str(row.get("予想買い目", "")).strip()
        hit        = str(row.get("的中", "")).strip()
        if pred_combo and pred_combo not in ("-", "（予想なし）", ""):
            races[key]["preds"].append(pred_combo)
            if hit == "○":
                races[key]["hit"] = True

    return list(races.values())


def analyze(races):
    total = len(races)
    if total == 0:
        print("[WARN] 有効なレースデータがありません")
        return

    manza_races = [r for r in races if r["payout"] >= MANZA_THRESHOLD]
    big_races   = [r for r in races if r["payout"] >= BIG_THRESHOLD]

    print("\n" + "="*60)
    print(f"  分析対象レース: {total} 件")
    print(f"  万舟が出たレース (払戻≥100倍): {len(manza_races)} 件 ({len(manza_races)/total*100:.1f}%)")
    print(f"  大穴が出たレース (払戻≥300倍): {len(big_races)} 件 ({len(big_races)/total*100:.1f}%)")
    print("="*60)

    # ── 1. 荒れPT別の万舟出現率 ──
    print("\n【1】荒れPT別 万舟出現率")
    print(f"  {'PT':>4}  {'レース数':>6}  {'万舟数':>6}  {'万舟率':>8}  {'平均払戻':>10}")
    pt_groups = defaultdict(list)
    for r in races:
        pt = r["arare_pt"]
        pt_key = f"{pt}以上" if pt >= 7 else str(pt)
        pt_groups[pt].append(r)
    for pt in sorted(pt_groups.keys()):
        grp = pt_groups[pt]
        mz  = [r for r in grp if r["payout"] >= MANZA_THRESHOLD]
        avg_pay = sum(r["payout"] for r in grp) / len(grp) if grp else 0
        label = f"{pt}以上" if pt >= 7 else f"PT{pt}"
        print(f"  {label:>6}  {len(grp):>6}  {len(mz):>6}  {len(mz)/len(grp)*100:>7.1f}%  {avg_pay:>10,.0f}円")

    # ── 2. 万舟が出たレースのPT分布 ──
    print("\n【2】万舟が出たレースの荒れPT分布（万舟の何%が各PTから？）")
    for pt in sorted(pt_groups.keys()):
        grp_mz = [r for r in pt_groups[pt] if r["payout"] >= MANZA_THRESHOLD]
        if grp_mz:
            pct = len(grp_mz) / len(manza_races) * 100 if manza_races else 0
            label = f"{pt}以上" if pt >= 7 else f"PT{pt}"
            print(f"  {label:>6}: {len(grp_mz):>4} 件 ({pct:>5.1f}%)")

    # ── 3. 会場別 万舟出現率 ──
    print("\n【3】会場別 万舟出現率（レース数 ≥ 5 の会場）")
    venue_groups = defaultdict(list)
    for r in races:
        v = r["venue"].replace("▲", "").replace("◎", "").strip()
        venue_groups[v].append(r)
    venue_data = []
    for v, grp in venue_groups.items():
        if len(grp) < 5:
            continue
        mz  = [r for r in grp if r["payout"] >= MANZA_THRESHOLD]
        avg = sum(r["payout"] for r in grp) / len(grp)
        venue_data.append((v, len(grp), len(mz), len(mz)/len(grp)*100, avg))
    venue_data.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'会場':>8}  {'レース数':>6}  {'万舟数':>6}  {'万舟率':>8}  {'平均払戻':>10}")
    for v, rc, mzc, mzr, avg in venue_data:
        print(f"  {v:>8}  {rc:>6}  {mzc:>6}  {mzr:>7.1f}%  {avg:>10,.0f}円")

    # ── 4. レース番号別 万舟出現率 ──
    print("\n【4】レース番号別 万舟出現率")
    rno_groups = defaultdict(list)
    for r in races:
        try:
            rno = int(r["race_no"])
        except ValueError:
            continue
        rno_groups[rno].append(r)
    print(f"  {'R':>4}  {'レース数':>6}  {'万舟数':>6}  {'万舟率':>8}")
    for rno in sorted(rno_groups.keys()):
        grp = rno_groups[rno]
        mz  = [r for r in grp if r["payout"] >= MANZA_THRESHOLD]
        print(f"  {rno:>3}R  {len(grp):>6}  {len(mz):>6}  {len(mz)/len(grp)*100:>7.1f}%")

    # ── 5. 万舟が出たレースで我々は予想していたか ──
    print("\n【5】万舟レースにおける予想カバー状況")
    predicted_manza     = [r for r in manza_races if r["preds"]]
    hit_manza           = [r for r in manza_races if r["hit"]]
    unpredicted_manza   = [r for r in manza_races if not r["preds"]]
    print(f"  万舟が出たレースのうち:")
    print(f"    予想を出していた: {len(predicted_manza)} 件 ({len(predicted_manza)/max(len(manza_races),1)*100:.1f}%)")
    print(f"    そのうち的中   : {len(hit_manza)} 件 ({len(hit_manza)/max(len(predicted_manza),1)*100:.1f}%)")
    print(f"    予想なし(見逃し): {len(unpredicted_manza)} 件 ({len(unpredicted_manza)/max(len(manza_races),1)*100:.1f}%)")

    # ── 6. 暴れ熊ラベル別 万舟出現率 ──
    print("\n【6】暴れ熊ラベル別 万舟出現率")
    label_groups = defaultdict(list)
    for r in races:
        lbl = r["bet_label"] if r["bet_label"] else "（なし）"
        label_groups[lbl].append(r)
    for lbl, grp in sorted(label_groups.items()):
        mz = [r for r in grp if r["payout"] >= MANZA_THRESHOLD]
        print(f"  {lbl:>14}: {len(mz):>4}/{len(grp):>4} 件 ({len(mz)/len(grp)*100:>5.1f}%)")

    # ── 7. 高払戻 上位20レース ──
    print("\n【7】高払戻 上位20レース（我々が見逃した万舟含む）")
    top_races = sorted(races, key=lambda x: x["payout"], reverse=True)[:20]
    print(f"  {'日付':>12}  {'会場':>8}  {'R':>3}  {'払戻':>10}  {'PT':>4}  {'的中':>4}  {'実際の結果':>12}")
    for r in top_races:
        hit_mark = "○" if r["hit"] else ("△予" if r["preds"] else "×見逃")
        print(f"  {r['date']:>12}  {r['venue'].replace('▲','').replace('◎',''):>8}  "
              f"{r['race_no']:>3}R  {r['payout']:>10,}円  PT{r['arare_pt']:>2}  "
              f"{hit_mark:>4}  {r['actual_combo']:>12}")

    # ── 8. 結論サマリー ──
    print("\n" + "="*60)
    print("  【結論】万舟出現のPT依存性")
    low_pt  = [r for r in races if r["arare_pt"] <= 3]
    high_pt = [r for r in races if r["arare_pt"] >= 6]
    low_mz  = [r for r in low_pt  if r["payout"] >= MANZA_THRESHOLD]
    high_mz = [r for r in high_pt if r["payout"] >= MANZA_THRESHOLD]
    if low_pt:
        print(f"  PT≤3 (低荒れ): 万舟率 {len(low_mz)/len(low_pt)*100:.1f}%  ({len(low_mz)}/{len(low_pt)}件)")
    if high_pt:
        print(f"  PT≥6 (高荒れ): 万舟率 {len(high_mz)/len(high_pt)*100:.1f}%  ({len(high_mz)}/{len(high_pt)}件)")
    print("="*60)
    print()


def main():
    print("万舟パターン分析を開始します...")
    records = get_sheet_data()
    races   = parse_records(records)
    analyze(races)


if __name__ == "__main__":
    main()
