"""
選手の戦術スタイルと決まり手統計を K-file の着順・コースデータから計算する

統計項目:
  aggression_score  : アウトコース（3-6番）での勝率 ÷ 全選手平均 (1.0=平均的)
  inner_win_rate    : コース1番からの勝率（逃げ率プロキシ）
  course2_pressure  : コース2番からの2着内率（差し圧力プロキシ）
"""
import json
from pathlib import Path

STYLE_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "racer_style.json"

_OUTER_BASELINE = 0.145  # コース3-6全体の平均1着率（約14.5%）
_MIN_OUTER_RACES = 15    # アウト統計の最低サンプル数
_MIN_INNER_RACES = 10    # インコース統計の最低サンプル数


def build_racer_style_lookup(k_dir: Path) -> dict:
    """
    K-file の着順・コースデータから選手ごとの戦術スタイルと決まり手統計を計算する

    Returns:
        {racer_no: {"aggression_score": float, "inner_win_rate": float, "course2_pressure": float}}
    """
    from src.collector.parser import parse_all_results

    df_rank, _ = parse_all_results(k_dir)
    if df_rank.empty:
        return {}

    stats = {}
    for racer_no, group in df_rank.groupby("racer_no"):
        entry: dict = {}

        # 積極性スコア（アウト勝率 ÷ ベースライン）
        outer = group[group["course"] >= 3]
        if len(outer) >= _MIN_OUTER_RACES:
            outer_win_rate = (outer["rank"] == 1).sum() / len(outer)
            entry["aggression_score"] = round(float(outer_win_rate / _OUTER_BASELINE), 4)

        # コース1番からの逃げ率（inner_win_rate）
        inner = group[group["course"] == 1]
        if len(inner) >= _MIN_INNER_RACES:
            entry["inner_win_rate"] = round(float((inner["rank"] == 1).sum() / len(inner)), 4)

        # コース2番からの2着内率（course2_pressure）
        c2 = group[group["course"] == 2]
        if len(c2) >= _MIN_INNER_RACES:
            entry["course2_pressure"] = round(float((c2["rank"] <= 2).sum() / len(c2)), 4)

        if entry:
            stats[int(racer_no)] = entry

    return stats


def save_racer_style_stats(stats: dict, path: Path = STYLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f)
    print(f"[OK] 戦術スタイル統計を保存: {len(stats)}選手 → {path}")


def load_racer_style_stats(path: Path = STYLE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
        # 旧フォーマット（float）と新フォーマット（dict）の両方に対応
        result = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                result[int(k)] = v
            else:
                result[int(k)] = {"aggression_score": float(v)}
        return result
    except Exception as e:
        print(f"[WARN] 戦術スタイル読み込みエラー: {e}")
        return {}
