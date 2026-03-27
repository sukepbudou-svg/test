"""各場の最初の選手データ行をフルで表示し、正規表現マッチを確認する"""
import re
from pathlib import Path

txt_path = Path("data/raw/B/b260325.txt")
content = open(txt_path, encoding="cp932", errors="replace").read()
lines = content.splitlines()

pattern = re.compile(
    r"^([1-6])\s(\d{4})"
    r"([^\x00-\x7F]+)"
    r"(\d{2,3})"
    r"([^\x00-\x7F]{2})"
    r"(\d{2,3})"
    r"([AB][12])\s+"
    r"([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+"
    r"(\d+)\s+(\d{1,2}\.\d{2})"
    r"(\d+)\s+([\d.]+)"
)

venue_code = None
venue_shown = set()

for line in lines:
    if re.match(r"^\d{2}BBGN", line):
        venue_code = line[:2]

    if re.match(r"^[1-6] \d{4}", line) and venue_code and venue_code not in venue_shown:
        venue_shown.add(venue_code)
        m = pattern.match(line)
        print(f"場{venue_code}: {'✅マッチ' if m else '❌不一致'}")
        print(f"  行: {line}")
        if not m:
            # どの部分で失敗しているか段階的に確認
            for i in range(1, 10):
                partial = pattern.pattern[:pattern.pattern.find(r"([\d.]+)", 0) + i*10]
                try:
                    pm = re.match(partial, line)
                    if not pm:
                        print(f"  → ここで失敗: {partial[-30:]}")
                        break
                except:
                    pass
        print()
