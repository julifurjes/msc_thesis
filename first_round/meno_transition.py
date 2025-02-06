import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MenopauseTransitionPredictor:
    def __init__(self, data_path, features, id_column, time_column, target_column):
        """
        Initialize the predictor with the dataset and required columns.
        
        :param data_path: Path to the dataset (CSV file).
        :param features: List of feature column names.
        :param id_column: Column name for subject IDs.
        :param time_column: Column name for time/visit indicator.
        :param target_column: Column name for the target variable (e.g., menopausal status).
        """
        self.data_path = data_path
        self.features = features
        self.id_column = id_column
        self.time_column = time_column
        self.target_column = target_column
        self.scaler = StandardScaler()

    def load_and_prepare_data(self):
        """
        Load the dataset, preprocess features, and create a transition target.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # Ensure all feature columns are numeric
        for feature in self.features:
            self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce')

        # Process the target column
        self.data[self.target_column] = pd.to_numeric(self.data[self.target_column], errors='coerce')

        # Group by subject ID and calculate transition target
        self.data["transition"] = self.data.groupby(self.id_column)[self.target_column].shift(-1) != self.data[self.target_column]
        self.data["transition"] = self.data["transition"].astype(int)

        # Drop rows without a valid next transition (last visit per subject)
        self.data = self.data.dropna(subset=self.features + ["transition"])

        # Normalize features
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])

    def create_sequences(self):
        """
        Create sequences grouped by subject ID and ordered by time.
        """
        grouped = self.data.groupby(self.id_column)
        X, y = [], []

        for _, group in grouped:
            group = group.sort_values(self.time_column)
            X.append(group[self.features].values)  # Sequence of features
            y.append(group["transition"].values)  # Transition labels

        # Pad sequences to ensure equal length
        self.X = pad_sequences(X, padding="post", dtype="float32")
        self.y = pad_sequences(y, padding="post", dtype="int32")  # Ensure labels match sequence length

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def build_model(self):
        """
        Build the LSTM model for transition prediction.
        """
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, input_shape=(self.X.shape[1], self.X.shape[2])))
        self.model.add(LSTM(64, activation="tanh", return_sequences=True))
        self.model.add(Dense(1, activation="sigmoid"))  # Binary classification

        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print(self.model.summary())

    def train_model(self, epochs=20, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model.
        """
        self.history = self.model.fit(
            self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split
        )

    def evaluate_model(self):
        """
        Evaluate the model on the test set.
        """
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self):
        """
        Predict probabilities on the test set.
        """
        predictions = self.model.predict(self.X_test)
        return predictions


if __name__ == "__main__":
    # Define file path and columns
    data_path = "combined_data.csv"
    features = [
        "E2AVE", "FSH", "T", "TSH", "SHBG",  # Hormones
        "DEPRESS", "QLTYLIF", "FEARFUL", "SLEEPQL",  # Mental health
        "AGE", "INCOME", "MARITAL",  # Demographics
        "BODYPAI", "PHYCTDW", "OVERHLT",  # Physical health
    ]
    id_column = "SWANID"
    time_column = "VISIT"
    target_column = "STATUS"  # Menopausal status column

    # Instantiate the predictor
    predictor = MenopauseTransitionPredictor(data_path, features, id_column, time_column, target_column)

    # Load and prepare data
    predictor.load_and_prepare_data()

    # Create sequences
    predictor.create_sequences()

    # Train-test split
    predictor.train_test_split()

    # Build and train the model
    predictor.build_model()
    predictor.train_model()

    # Evaluate the model
    predictor.evaluate_model()

    # Predict on the test set
    predictions = predictor.predict()
    print("Predictions:", predictions[:10])  # Print first 10 predictions

# TOO HIGH ACCURACY