import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class MenopauseGB:
    def __init__(self, file_path, perform_hyperparameter_search=False):
        self.file_path = file_path
        self.perform_hyperparameter_search = perform_hyperparameter_search
        self.data = None
        self.features = [
            "HOTFLAS", "NUMHOTF", "BOTHOTF", "NITESWE", "NUMNITS", "BOTNITS",
            "COLDSWE", "NUMCLDS", "BOTCLDS", "STIFF", "IRRITAB", "MOODCHG", "STATUS", "AGE", "RACE"
        ]
        self.target = "CognitiveDecline"
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=9, 
            subsample=0.8, 
            min_samples_split=2, 
            max_features="sqrt", 
            random_state=42
        )

    def load_data(self):
        self.data = pd.read_csv(self.file_path, low_memory=False)

    def define_cognitive_decline(self):
        """
        Define cognitive decline dynamically based on changes in a participant's scores over time.
        """
        self.data["TOTIDE1"] = pd.to_numeric(self.data["TOTIDE1"], errors="coerce")
        self.data["TOTIDE2"] = pd.to_numeric(self.data["TOTIDE2"], errors="coerce")
        self.data = self.data.dropna(subset=["TOTIDE1", "TOTIDE2"])
        self.data = self.data.sort_values(by=["SWANID", "VISIT"])
        self.data["TOTIDE1_change"] = self.data.groupby("SWANID")["TOTIDE1"].diff()
        self.data["TOTIDE2_change"] = self.data.groupby("SWANID")["TOTIDE2"].diff()
        self.data[self.target] = np.where(
            (self.data["TOTIDE1_change"] < -1) | 
            (self.data["TOTIDE2_change"] < -1) |
            (self.data["TOTIDE1_change"] / self.data["TOTIDE1"].shift(1) < -0.1) |
            (self.data["TOTIDE2_change"] / self.data["TOTIDE2"].shift(1) < -0.1), 
            1, 
            0
        )

    def preprocess_data(self):
        """
        Handle missing values, scale features, and split data into training and testing sets.
        """
        self.data[self.features] = self.data[self.features].replace(" ", np.nan)
        self.data[self.features] = self.data[self.features].apply(pd.to_numeric, errors="coerce")
        self.data = self.data.dropna(subset=self.features + [self.target])
        X = self.data[self.features]
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self):
        """
        Train the Gradient Boosting model on the training data with optional hyperparameter tuning.
        """
        print("Training Gradient Boosting model...")
        smote_tomek = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(self.X_train, self.y_train)

        if self.perform_hyperparameter_search:
            print("Performing hyperparameter search...")
            param_grid = {
                "n_estimators": [50, 100, 150, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2", None],
            }

            random_search = RandomizedSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_distributions=param_grid,
                n_iter=100,
                scoring="roc_auc",
                cv=5,
                random_state=42,
                n_jobs=-1
            )
            random_search.fit(X_train_resampled, y_train_resampled)
            self.model = random_search.best_estimator_
            print("\nBest Hyperparameters from RandomizedSearchCV:", random_search.best_params_)
        else:
            self.model.fit(X_train_resampled, y_train_resampled)

    def evaluate_model(self):
        """
        Evaluate the model on the testing set and print performance metrics.
        """
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.1):
            y_pred_adjusted = (y_prob >= threshold).astype(int)
            f1 = f1_score(self.y_test, y_pred_adjusted)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Optimal Threshold: {best_threshold:.2f}, Best F1-Score: {best_f1:.3f}")
        y_pred_adjusted = (y_prob >= best_threshold).astype(int)
        print("\nAdjusted Threshold Performance:")
        print(classification_report(self.y_test, y_pred_adjusted))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred_adjusted))
        print(f"ROC-AUC Score: {roc_auc_score(self.y_test, y_prob):.3f}")
        self.plot_roc_curve(self.y_test, y_prob)

    def plot_roc_curve(self, y_test, y_prob):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    def run(self):
        """
        Execute the workflow: load data, preprocess, train, and evaluate.
        """
        print("Loading data...")
        self.load_data()
        print("Defining cognitive decline...")
        self.define_cognitive_decline()
        print("Preprocessing data...")
        self.preprocess_data()
        print("Training model...")
        self.train_model()
        print("Evaluating model...")
        self.evaluate_model()

if __name__ == "__main__":
    file_path = "processed_combined_data.csv"  # Replace with your dataset path
    gb_model = MenopauseGB(file_path, perform_hyperparameter_search=False)  # Set to False to skip hyperparameter search
    gb_model.run()