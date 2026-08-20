import sqlite3

conn = sqlite3.connect("data/perry.db")
cur = conn.cursor()

# カラム名確認
cur.execute("PRAGMA table_info(predictions)")
cols = [row[1] for row in cur.fetchall()]
print("カラム一覧:", cols)
print()

# 払戻カラム名を自動判定
pay_col = "result_payout" if "result_payout" in cols else ("payout" if "payout" in cols else cols[0])
print(f"払戻カラム: {pay_col}")
print()

print("=== PT別 的中分析 ===")
cur.execute(f"""
SELECT arare_score, COUNT(*) as total,
       SUM(CASE WHEN is_hit=1 THEN 1 ELSE 0 END) as hits,
       ROUND(AVG(CASE WHEN is_hit=1 THEN {pay_col} ELSE NULL END), 0) as avg_payout
FROM predictions
WHERE result_recorded_at IS NOT NULL
GROUP BY arare_score ORDER BY arare_score
""")
print(f"{'PT':>4} {'予想数':>6} {'的中':>5} {'的中率':>7} {'平均払戻':>9}")
for row in cur.fetchall():
    pt, total, hits, avg_pay = row
    rate = f"{hits/total*100:.1f}%" if total > 0 else "-"
    avg = f"¥{int(avg_pay):,}" if avg_pay else "-"
    print(f"{pt:>4} {total:>6} {hits:>5} {rate:>7} {avg:>9}")

print()
print("=== ラベル別 的中分析 ===")
cur.execute(f"""
SELECT bet_label, COUNT(*) as total,
       SUM(CASE WHEN is_hit=1 THEN 1 ELSE 0 END) as hits,
       ROUND(AVG(CASE WHEN is_hit=1 THEN {pay_col} ELSE NULL END), 0) as avg_payout,
       SUM({pay_col}) as total_payout
FROM predictions
WHERE result_recorded_at IS NOT NULL
GROUP BY bet_label ORDER BY bet_label
""")
print(f"{'ラベル':>8} {'予想数':>6} {'的中':>5} {'的中率':>7} {'平均払戻':>10} {'総払戻':>10}")
for row in cur.fetchall():  # noqa
    lbl, total, hits, avg_pay, total_pay = row
    rate = f"{hits/total*100:.1f}%" if total > 0 else "-"
    avg = f"¥{int(avg_pay):,}" if avg_pay else "-"
    tp = f"¥{int(total_pay):,}" if total_pay else "¥0"
    print(f"{str(lbl):>8} {total:>6} {hits:>5} {rate:>7} {avg:>10} {tp:>10}")

print()
print("=== 的中レース一覧（配当順） ===")
cur.execute(f"""
SELECT date, venue_name, race_no, bet_label, arare_score, combination, {pay_col}
FROM predictions
WHERE is_hit=1 AND result_recorded_at IS NOT NULL
ORDER BY {pay_col} DESC
LIMIT 20
""")
print(f"{'日付':>12} {'会場':>6} {'R':>3} {'ラベル':>8} {'PT':>4} {'組み合わせ':>10} {'払戻':>9}")
for row in cur.fetchall():
    date, venue, race, lbl, pt, combo, pay = row
    print(f"{str(date):>12} {str(venue):>6} {race:>3}R {str(lbl):>8} {pt:>4} {str(combo):>10} ¥{int(pay):,}")

conn.close()
