"""
train_lr.py - train the AIRSDN traffic-level classifiers (base-paper reproduction, NSFNET).

Train the traffic-level classifiers following the AIRSDN paper (Section 4.1):
70 % train / 15 % validation / 15 % test split, six binary classifiers
(XGBoost, Random Forest, SVM, KNN, AdaBoost, Logistic Regression), report
precision / recall / F1 (Table 5), the LR confusion matrix (Table 6) and AUC.

The Logistic Regression model (best model in the paper) is saved for the
routing experiment. Labels are kept as in the dataset (1 = low traffic,
2 = high traffic) so the routing algorithm can pick the path with the
lowest predicted traffic level.
"""
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'data', 'result_data.csv')
MODELS_DIR = os.path.join(HERE, 'models')
SEED = 40


def build_models():
    # Distance / margin based models need feature scaling: the 19 features span
    # very different ranges (bits_per_second ~1e6 vs packet_loss ~1e1).
    return {
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', early_stopping_rounds=10, random_state=SEED),
        'Random Forest': RandomForestClassifier(random_state=SEED),
        'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier()),
        'AdaBoost': AdaBoostClassifier(random_state=SEED),
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    }


def train_and_save_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} does not exist.")
        return

    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # First 19 columns are features, last column is traffic_type_numeric (1 = low, 2 = high)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    high_label = int(y.max())

    # 70 % train, 15 % validation, 15 % test
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    print(f"Split -> train: {len(y_train)}, validation: {len(y_val)}, test: {len(y_test)}\n")

    # XGBoost expects 0-indexed labels
    to_xgb = lambda labels: (labels == high_label).astype(int)

    fitted = {}
    print(f"{'Model':<22}{'Precision':>10}{'Recall':>10}{'F1 Score':>10}{'AUC':>8}")
    for name, model in build_models().items():
        if name == 'XGBoost':
            model.fit(X_train, to_xgb(y_train), eval_set=[(X_val, to_xgb(y_val))], verbose=False)
            y_pred = np.where(model.predict(X_test) == 1, high_label, y.min())
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        fitted[name] = (model, y_pred)

        # High traffic is the positive class
        precision = precision_score(y_test, y_pred, pos_label=high_label)
        recall = recall_score(y_test, y_pred, pos_label=high_label)
        f1 = f1_score(y_test, y_pred, pos_label=high_label)
        auc = roc_auc_score(y_test == high_label, y_score)
        print(f"{name:<22}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}{auc:>8.3f}")

    lr_model, lr_pred = fitted['Logistic Regression']
    cm = confusion_matrix(y_test, lr_pred, labels=[y.min(), high_label])
    print("\nLogistic Regression confusion matrix (rows = actual, cols = predicted; Low, High):")
    print(cm)

    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'logistic_regression_model.pkl'))
    print(f"\nLogistic Regression model saved to {MODELS_DIR}/logistic_regression_model.pkl")



if __name__ == '__main__':
    train_and_save_models()
