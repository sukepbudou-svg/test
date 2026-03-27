"""各場の選手データ行の文字を詳しく調べる"""
import re
from pathlib import Path

txt_path = Path("data/raw/B/b260325.txt")
content = open(txt_path, encoding="cp932", errors="replace").read()
lines = content.splitlines()

venue_code = None
venue_samples = {}

for line in lines:
    if re.match(r"^\d{2}BBGN", line):
        venue_code = line[:2]

    # 艇番+スペース+登録番号のパターン
    if re.match(r"^[1-6].{1}\d{4}", line) and venue_code:
        sep = line[1]  # 艇番の次の文字
        sep_code = hex(ord(sep))
        if venue_code not in venue_samples:
            venue_samples[venue_code] = (sep, sep_code, line[:20])

print("各場の区切り文字:")
for vc, (sep, code, sample) in venue_samples.items():
    print(f"  場{vc}: 文字='{sep}' コード={code} 行例='{sample}'")
