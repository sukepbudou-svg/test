"""
大穴レース傾向分析スクリプト
実結果が大穴帯（actual_payout >= 8000 = 80倍超）になったレースの特徴を調べる
"""
import sqlite3
import re
from collections import Counter, defaultdict

conn = sqlite3.connect("data/perry.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 結果が記録されているレースを1レース1行に集約
# actual_payoutはレース内最大値を使用（同一レースで同じはず）
cur.execute("""
    SELECT
        date, venue_name, race_no, arare_score,
        MAX(race_grade) as race_grade,
        MAX(arare_reasons) as arare_reasons,
        MAX(bet_label) as bet_label,
        MAX(nigerate_str) as nigerate_str,
        MAX(actual_payout) as actual_payout,
        MAX(actual_combination) as actual_combination
    FROM predictions
    WHERE result_recorded_at IS NOT NULL
      AND actual_payout > 0
    GROUP BY date, venue_name, race_no
    ORDER BY actual_payout DESC
""")
all_rows = [dict(r) for r in cur.fetchall()]

total = len(all_rows)
okuma = [r for r in all_rows if r["actual_payout"] >= 8000]   # 80倍超
chuuma = [r for r in all_rows if 2000 <= r["actual_payout"] < 8000]  # 20〜80倍
hito = [r for r in all_rows if r["actual_payout"] < 2000]             # 20倍未満

print(f"=== 結果記録済みレース総数: {total}件 ===")
print(f"  大穴（≥80倍/8000円）: {len(okuma)}件 ({len(okuma)/total*100:.1f}%)")
print(f"  中穴（20〜80倍）:      {len(chuuma)}件 ({len(chuuma)/total*100:.1f}%)")
print(f"  本命圏（<20倍）:       {len(hito)}件 ({len(hito)/total*100:.1f}%)")

# ---- PT帯別 大穴出現率 ----
print()
print("=== PT帯別 大穴出現率 ===")
pt_all = defaultdict(int)
pt_ok = defaultdict(int)
for r in all_rows:
    pt_all[r["arare_score"]] += 1
    if r["actual_payout"] >= 8000:
        pt_ok[r["arare_score"]] += 1
for pt in sorted(pt_all.keys()):
    total_pt = pt_all[pt]
    ok_pt = pt_ok.get(pt, 0)
    print(f"  PT{pt:2d}: {ok_pt}/{total_pt}件 ({ok_pt/total_pt*100:.1f}%)")

# ---- 会場別 大穴出現率（件数5件以上） ----
print()
print("=== 会場別 大穴出現率（5件以上） ===")
v_all = defaultdict(int)
v_ok = defaultdict(int)
for r in all_rows:
    v = r["venue_name"]
    v_all[v] += 1
    if r["actual_payout"] >= 8000:
        v_ok[v] += 1
venue_rates = [(v, v_ok.get(v,0), v_all[v]) for v in v_all if v_all[v] >= 5]
venue_rates.sort(key=lambda x: -x[1]/x[2])
for v, ok, tot in venue_rates:
    print(f"  {v}: {ok}/{tot}件 ({ok/tot*100:.1f}%)")

# ---- レース番号別 大穴出現率 ----
print()
print("=== レース番号別 大穴出現率 ===")
rn_all = defaultdict(int)
rn_ok = defaultdict(int)
for r in all_rows:
    rn_all[r["race_no"]] += 1
    if r["actual_payout"] >= 8000:
        rn_ok[r["race_no"]] += 1
for rn in sorted(rn_all.keys()):
    tot = rn_all[rn]
    ok = rn_ok.get(rn, 0)
    print(f"  {rn:2d}R: {ok}/{tot}件 ({ok/tot*100:.1f}%)")

# ---- arare_reasons キーワード頻度（大穴 vs 全体） ----
def extract_reasons(rows):
    cnt = Counter()
    for r in rows:
        reasons = r.get("arare_reasons") or ""
        # スラッシュ区切りで分割
        parts = [p.strip() for p in reasons.split("/") if p.strip()]
        for p in parts:
            # 数値部分を除いてキーワード化
            kw = re.sub(r"[\d\.]+", "N", p)
            cnt[kw] += 1
    return cnt

print()
print("=== arare_reasons キーワード（大穴レースで多いもの Top20） ===")
ok_cnt = extract_reasons(okuma)
all_cnt = extract_reasons(all_rows)
# 大穴での出現率 vs 全体出現率のリフト値で並べる
lift_data = []
for kw, ok_n in ok_cnt.items():
    all_n = all_cnt.get(kw, 1)
    ok_rate = ok_n / max(len(okuma), 1)
    all_rate = all_n / max(total, 1)
    lift = ok_rate / max(all_rate, 0.001)
    lift_data.append((kw, ok_n, all_n, ok_rate, all_rate, lift))
lift_data.sort(key=lambda x: -x[5])
print(f"  {'キーワード':<35} {'大穴出現':>6} {'全体':>6} {'リフト':>6}")
for kw, ok_n, all_n, ok_rate, all_rate, lift in lift_data[:20]:
    print(f"  {kw:<35} {ok_rate*100:5.1f}%  {all_rate*100:5.1f}%  {lift:5.2f}x")

# ---- bet_label別 大穴出現率 ----
print()
print("=== 参戦ラベル別 大穴出現率 ===")
lbl_all = defaultdict(int)
lbl_ok = defaultdict(int)
for r in all_rows:
    lbl = r["bet_label"]
    lbl_all[lbl] += 1
    if r["actual_payout"] >= 8000:
        lbl_ok[lbl] += 1
for lbl in ["プチュン", "黒船熱", "中穴", "見送り"]:
    if lbl_all[lbl]:
        print(f"  {lbl}: {lbl_ok.get(lbl,0)}/{lbl_all[lbl]}件 ({lbl_ok.get(lbl,0)/lbl_all[lbl]*100:.1f}%)")

# ---- 大穴レース詳細一覧（払戻上位10件） ----
print()
print("=== 大穴レース払戻上位10件 ===")
for r in okuma[:10]:
    print(f"  {r['date']} {r['venue_name']} {r['race_no']}R  PT={r['arare_score']}  {r['actual_combination']}  ¥{r['actual_payout']:,}")
    reasons_short = (r["arare_reasons"] or "")[:80]
    print(f"    [{r['bet_label']}] {reasons_short}")

conn.close()
