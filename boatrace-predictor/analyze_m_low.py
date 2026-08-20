import sqlite3

conn = sqlite3.connect("data/perry_backup_20260820.db")
cur = conn.cursor()

cur.execute("SELECT COUNT(DISTINCT date||venue_name||race_no) FROM predictions")
total = cur.fetchone()[0]

cur.execute("""SELECT COUNT(DISTINCT date||venue_name||race_no) FROM predictions
WHERE arare_reasons LIKE '%1号M低%'""")
m1 = cur.fetchone()[0]

cur.execute("""SELECT COUNT(DISTINCT date||venue_name||race_no) FROM predictions
WHERE arare_reasons LIKE '%2号M低%'""")
m2 = cur.fetchone()[0]

cur.execute("""SELECT COUNT(DISTINCT date||venue_name||race_no) FROM predictions
WHERE arare_reasons LIKE '%M低%'""")
m_any = cur.fetchone()[0]

print(f"総レース数: {total}件")
print(f"1号M低（<30%）: {m1}件 ({m1/total*100:.1f}%)")
print(f"2号M低（<35%）: {m2}件 ({m2/total*100:.1f}%)")
print(f"どちらかM低:   {m_any}件 ({m_any/total*100:.1f}%)")

conn.close()
