import argparse
import lightgbm as lgb
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json


def train_lgbm(train_path: Path, model_out: Path, imp_out: Path,
               test_size: float = 0.2, random_state: int = 42):
    # Load windows & split features / target
    df = pd.read_parquet(train_path)
    y = df['is_intro']
    X = df.drop(columns=['episode_id', 'is_intro'])

    # Train/Val split by rows (could also use GroupKFold by episode)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set   = lgb.Dataset(X_val,   label=y_val)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1
    }

    booster = lgb.train(params,
                        train_set,
                        num_boost_round=200,
                        valid_sets=[val_set],
                        early_stopping_rounds=25)

    # Evaluate
    y_pred = booster.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    print(f"Validation AUC = {auc:.4f}")

    # Save model & feature importances
    model_out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_out))
    print(f"Model saved to {model_out}")

    # Feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': booster.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    imp_out.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(imp_out, index=False)
    print(f"Feature importances saved to {imp_out}")

    # Save meta json
    meta = {
        'train_rows': len(X_train),
        'val_rows': len(X_val),
        'auc': auc,
        'best_iteration': booster.best_iteration,
        'params': params
    }
    with open(model_out.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LightGBM intro classifier')
    parser.add_argument('--train_parquet', type=Path, required=True,
                        help='windows_train.parquet')
    parser.add_argument('--model_out',     type=Path, default=Path('models/lightgbm_intro.txt'))
    parser.add_argument('--imp_out',       type=Path, default=Path('models/feature_importances.csv'))
    args = parser.parse_args()
    train_lgbm(args.train_parquet, args.model_out, args.imp_out)
