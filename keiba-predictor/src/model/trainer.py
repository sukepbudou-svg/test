"""
競馬予想モデルのトレーニング
LightGBM で馬連の的中確率を学習する
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from src.features.builder import get_feature_columns

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


def train_model(df: pd.DataFrame, model_name: str = "quinella_model") -> lgb.Booster:
    """
    馬連的中予測モデルを学習する

    Args:
        df: build_quinella_features()で生成した特徴量DataFrame
            result列(0/1)が必要
    """
    feature_cols = get_feature_columns()

    df = df.dropna(subset=["result"])
    X = df[feature_cols].fillna(0)
    y = df["result"].astype(int)

    print(f"学習データ: {len(df)}組み合わせ / 的中数: {y.sum()} ({y.mean()*100:.1f}%)")

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        pred = model.predict(X_val)
        score = log_loss(y_val, pred)
        scores.append(score)
        print(f"Fold {fold+1}: log_loss = {score:.4f}")

    print(f"CV平均 log_loss: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, dtrain_full,
                            num_boost_round=int(model.best_iteration * 1.1))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.txt"
    final_model.save_model(str(model_path))
    print(f"[OK] モデル保存: {model_path}")

    return final_model


def load_model(model_name: str = "quinella_model") -> lgb.Booster | None:
    """保存済みモデルを読み込む"""
    model_path = MODEL_DIR / f"{model_name}.txt"
    if not model_path.exists():
        print(f"[ERROR] モデルファイルが見つかりません: {model_path}")
        return None
    model = lgb.Booster(model_file=str(model_path))
    print(f"[OK] モデル読み込み: {model_path}")
    return model
