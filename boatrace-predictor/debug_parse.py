"""パーサーデバッグ用スクリプト"""
import re
from pathlib import Path

txt_path = Path("data/raw/B/b260325.txt")
content = open(txt_path, encoding="cp932", errors="replace").read()
lines = content.splitlines()

bbgn_count = 0
boat_count = 0
for line in lines:
    if re.match(r"^\d{2}BBGN", line):
        bbgn_count += 1
        print(f"場コード: {line[:2]}")
    if re.match(r"^[1-6] \d{4}", line):
        boat_count += 1

print(f"\nBBGNセクション数: {bbgn_count}")
print(f"選手データ行（概算）: {boat_count}")
