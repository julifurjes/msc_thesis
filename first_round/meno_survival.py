import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Load dataset
df = pd.read_csv('combined_data.csv', low_memory=False)

# Replace empty strings or invalid values with NaN
df = df.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)

# Convert numeric columns and handle missing data
numeric_columns = ["ESTROG", "COMBIN", "PELVCPN", "ENDO", "EXERCIS", "SLEEPQL", "AGE", "INCOME", "MARITAL"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=numeric_columns + ["VISIT", "STATUS"], inplace=True)

# Add event and time-to-event columns
df["event"] = (df["STATUS"] == '2').astype(bool)
df["time_to_event"] = df["VISIT"].astype(float)

# Define features and target
features = numeric_columns
Xt = df[features].to_numpy()  # Convert features to a NumPy array
y = Surv.from_dataframe("event", "time_to_event", df)  # Create a structured survival target

# Train-test split
random_state = 20
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)

# Debugging: Check shapes and values
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
print("y_train sample:\n", y_train[:5])

# Random Survival Forest Model
rsf = RandomSurvivalForest(
    n_estimators=1000,
    min_samples_split=10,
    min_samples_leaf=15,
    n_jobs=-1,
    random_state=random_state
)
rsf.fit(X_train, y_train)

# Scoring
try:
    test_score = rsf.score(X_test, y_test)
    print(f"Test set score: {test_score:.4f}")
except Exception as e:
    print(f"Error during scoring: {e}")

# Predict survival probabilities
try:
    X_test_sorted = X_test[np.argsort(X_test[:, 0])]  # Sort by the first feature (e.g., AGE)
    X_test_sel = np.vstack([X_test_sorted[:3], X_test_sorted[-3:]])  # Select 3 from top and bottom
    surv = rsf.predict_survival_function(X_test_sel, return_array=True)

    # Plot survival probabilities
    plt.figure(figsize=(10, 6))
    for i, s in enumerate(surv):
        plt.step(rsf.unique_times_, s, where="post", label=f"Sample {i}")
    plt.ylabel("Survival Probability")
    plt.xlabel("Time in Days")
    plt.legend()
    plt.grid(True)
    plt.title("Predicted Survival Functions")
    plt.show()
except Exception as e:
    print(f"Error during prediction: {e}")

# Permutation importance
try:
    result = permutation_importance(rsf, X_test, y_test, n_repeats=15, random_state=random_state)
    importance_df = pd.DataFrame({
        "Mean Importance": result.importances_mean,
        "Std Deviation": result.importances_std
    }, index=features).sort_values(by="Mean Importance", ascending=False)

    print("\nFeature Importance (Permutation):")
    print(importance_df)
except Exception as e:
    print(f"Error during permutation importance: {e}")
