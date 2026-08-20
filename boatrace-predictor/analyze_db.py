import sqlite3

conn = sqlite3.connect("data/perry.db")
cur = conn.cursor()

# カラム名確認
cur.execute("PRAGMA table_info(predictions)")
cols = [row[1] for row in cur.fetchall()]
print("=== カラム一覧 ===")
for c in cols:
    print(f"  {c}")

# サンプルデータ確認（結果が入っているもの）
print()
print("=== 結果記録済みサンプル（直近5件） ===")
cur.execute("""
SELECT * FROM predictions
WHERE result_recorded_at IS NOT NULL AND is_hit=1
ORDER BY id DESC LIMIT 5
""")
rows = cur.fetchall()
for row in rows:
    for col, val in zip(cols, row):
        print(f"  {col}: {val}")
    print("---")

# 数値が大きいカラムを探す（払戻候補）
print()
print("=== 数値カラム最大値（払戻カラム特定用） ===")
for c in cols:
    try:
        cur.execute(f"SELECT MAX({c}) FROM predictions WHERE result_recorded_at IS NOT NULL AND is_hit=1")
        val = cur.fetchone()[0]
        if val and isinstance(val, (int, float)) and val > 100:
            print(f"  {c}: 最大値={val}")
    except Exception:
        pass

conn.close()
