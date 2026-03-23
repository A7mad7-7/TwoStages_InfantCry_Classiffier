import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from xgboost_pipeline.config_xgb import ConfigXGB

class TrainerXGB:
    """Manages Z-Score normalization, XGBoost training, and evaluation."""
    def __init__(self, config: ConfigXGB):
        self.cfg = config
        self.scaler = StandardScaler()
        # Classical parameters configured for multi-class classification
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=self.cfg.num_classes,
            eval_metric="mlogloss",
            seed=self.cfg.random_seed,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=15
        )

    def fit_scaler(self, X_train: np.ndarray):
        print("[Trainer] Fitting Z-Score Scaler on Training features...")
        self.scaler.fit(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def train(self, X_train, y_train, X_val, y_val):
        print(f"[Trainer] Training XGBoost Classifier...")
        
        # Fit model with Validation Set for early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=10
        )
        print("[Trainer] Training complete.")

    def evaluate(self, X_test, y_test):
        print("[Trainer] Evaluating on Test Set...")
        y_pred = self.model.predict(X_test)
        
        report = classification_report(
            y_test, y_pred, 
            target_names=self.cfg.class_names
        )
        print("\n============================================================")
        print("  Classification Report — Test Set")
        print("============================================================")
        print(report)

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.cfg.class_names,
                    yticklabels=self.cfg.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("XGBoost Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.cfg.confusion_matrix_path)
        plt.close()
        print(f"[Trainer] Saved confusion matrix: {self.cfg.confusion_matrix_path}")

    def save(self):
        self.model.save_model(self.cfg.model_save_path)
        joblib.dump(self.scaler, self.cfg.scaler_save_path)
        print(f"[Trainer] Saved model to {self.cfg.model_save_path}")
        print(f"[Trainer] Saved Z-Score scaler to {self.cfg.scaler_save_path}")
