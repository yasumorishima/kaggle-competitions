"""
Kaggle Competition Training Script with W&B Integration.

Usage:
    python train.py --config config.yaml
"""

import argparse

import numpy as np
import pandas as pd
import wandb
import yaml
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict):
    # W&B init
    wb_cfg = config["wandb"]
    run = wandb.init(
        project=wb_cfg["project"],
        config=config["model"]["params"],
        tags=wb_cfg.get("tags", []),
    )

    # Load data
    train_df = pd.read_csv(config["data"]["train"])
    test_df = pd.read_csv(config["data"]["test"])

    # TODO: Define target and features
    target_col = "target"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X = train_df[feature_cols]
    y = train_df[target_col]
    X_test = test_df[feature_cols]

    # Cross validation
    cv_cfg = config["cv"]
    n_splits = cv_cfg["n_splits"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv_cfg["random_state"])

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    model_params = config["model"]["params"].copy()
    early_stopping = model_params.pop("early_stopping_rounds", 50)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(100)],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_splits

        # TODO: Calculate metric based on config["competition"]["metric"]
        fold_score = np.sqrt(np.mean((y_val - val_pred) ** 2))
        scores.append(fold_score)

        wandb.log({"fold": fold, "fold_score": fold_score})
        print(f"Fold {fold}: {fold_score:.5f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\nCV: {mean_score:.5f} ± {std_score:.5f}")

    wandb.log({"cv_mean": mean_score, "cv_std": std_score})

    # Save submission
    sub = pd.read_csv(config["data"]["submission"])
    sub[target_col] = test_preds
    sub.to_csv("submission.csv", index=False)
    wandb.save("submission.csv")

    print("Saved submission.csv")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
