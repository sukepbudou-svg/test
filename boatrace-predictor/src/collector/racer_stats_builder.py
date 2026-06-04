"""
選手の戦術スタイル（積極性スコア）を K-file の着順・コースデータから計算する

積極性スコア:
  - アウトコース（3-6番）での実際の勝率 ÷ 全選手平均勝率
  - 1.0 = 平均的、>1.0 = アウトでも勝てる積極系、<1.0 = インコース依存型
"""
import json
from pathlib import Path

STYLE_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "racer_style.json"

_OUTER_BASELINE = 0.145  # コース3-6全体の平均1着率（約14.5%）
_MIN_OUTER_RACES = 15    # 統計に使う最低サンプル数


def build_racer_style_lookup(k_dir: Path) -> dict:
    """
    K-file の着順・コースデータから選手ごとの積極性スコアを計算する

    Returns:
        {racer_no: aggression_score}
    """
    from src.collector.parser import parse_all_results

    df_rank, _ = parse_all_results(k_dir)
    if df_rank.empty:
        return {}

    stats = {}
    for racer_no, group in df_rank.groupby("racer_no"):
        outer = group[group["course"] >= 3]
        if len(outer) < _MIN_OUTER_RACES:
            continue
        outer_win_rate = (outer["rank"] == 1).sum() / len(outer)
        aggression = outer_win_rate / _OUTER_BASELINE
        stats[int(racer_no)] = round(float(aggression), 4)

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
            return {int(k): float(v) for k, v in json.load(f).items()}
    except Exception as e:
        print(f"[WARN] 戦術スタイル読み込みエラー: {e}")
        return {}
