"""
予測モデルのトレーニング
LightGBM を使って各艇の1着確率を学習する
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from src.features.builder import get_feature_columns, add_course_advantage

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


def train_model(df: pd.DataFrame, model_name: str = "trifecta_model") -> lgb.Booster:
    """
    3連単予測モデルを学習する

    Args:
        df: build_features()で生成した特徴量DataFrame
        model_name: 保存するモデル名

    Returns:
        学習済みLightGBMモデル
    """
    df = add_course_advantage(df)
    feature_cols = get_feature_columns() + [f"boat{bn}_course_advantage" for bn in range(1, 7)]

    # 欠損値処理
    df = df.dropna(subset=["winner_boat"])
    df["winner_boat"] = df["winner_boat"].astype(int) - 1  # 0-indexed (0〜5)

    X = df[feature_cols].fillna(0)
    y = df["winner_boat"]

    # 時系列分割でクロスバリデーション
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            "objective": "multiclass",
            "num_class": 6,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        pred = model.predict(X_val)
        score = log_loss(y_val, pred)
        scores.append(score)
        print(f"Fold {fold+1}: log_loss = {score:.4f}")

    print(f"CV平均 log_loss: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # 全データで最終モデルを学習
    dtrain_full = lgb.Dataset(X, label=y)
    params["verbose"] = -1
    final_model = lgb.train(params, dtrain_full, num_boost_round=int(model.best_iteration * 1.1))

    # モデル保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.txt"
    final_model.save_model(str(model_path))
    print(f"[OK] モデル保存: {model_path}")

    return final_model


def load_model(model_name: str = "trifecta_model") -> lgb.Booster | None:
    """保存済みモデルを読み込む"""
    model_path = MODEL_DIR / f"{model_name}.txt"
    if not model_path.exists():
        print(f"[ERROR] モデルファイルが見つかりません: {model_path}")
        return None
    model = lgb.Booster(model_file=str(model_path))
    print(f"[OK] モデル読み込み: {model_path}")
    return model
