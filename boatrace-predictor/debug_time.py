"""
番組表の発走時刻フォーマット診断スクリプト
レース番号付近の行を表示して、時刻がどこにあるか確認する
"""
from pathlib import Path
import re

RAW_B_DIR = Path(__file__).parent / "data" / "raw" / "B"
files = sorted(RAW_B_DIR.glob("*.txt"))
if not files:
    print("番組表テキストファイルが見つかりません")
    exit()

txt_path = files[-1]  # 最新ファイルを使用
print(f"診断ファイル: {txt_path.name}\n")

fw2hw = str.maketrans("０１２３４５６７８９Ｒ", "0123456789R")

with open(txt_path, encoding="cp932", errors="replace") as f:
    lines = f.readlines()

found = 0
for i, line in enumerate(lines):
    line_hw = line.translate(fw2hw)
    if re.match(r"[　\s]*1R\s", line_hw):  # 1レース目だけ表示
        print(f"=== 1R 発見（行{i}）周辺20行 ===")
        for j in range(max(0, i-2), min(len(lines), i+20)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} [{j:4d}] {repr(lines[j].rstrip())}")
        found += 1
        if found >= 3:  # 最大3場分
            break

if found == 0:
    print("レース番号行が見つかりませんでした")
