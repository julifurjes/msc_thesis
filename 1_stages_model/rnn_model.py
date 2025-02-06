import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class MenopausePredictionModel:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, low_memory=False)
        print("Initial data shape:", self.data.shape)
        self.prepare_data()
        self.sequence_length = 3
        
    def prepare_data(self):
        """Prepare the dataset for ML modeling."""
        # Convert STATUS to numeric and filter
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        self.data = self.data[self.data['STATUS'].between(2, 5)]
        
        # Convert cognitive variables to numeric
        cognitive_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        for col in cognitive_vars:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Sort by SWANID and VISIT
        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        
        # Handle missing values within each cognitive variable
        for var in cognitive_vars:
            # Forward fill then backward fill within each subject
            self.data[var] = self.data.groupby('SWANID')[var].transform(
                lambda x: x.ffill().bfill()
            )
        
        # Calculate changes for each variable
        for var in cognitive_vars:
            self.data[f'{var}_change'] = self.data.groupby('SWANID')[var].diff()
        
        # Remove subjects with too many missing values
        missing_threshold = 0.5
        subject_missing = self.data.groupby('SWANID')[cognitive_vars].apply(
            lambda x: x.isnull().mean()
        ).mean(axis=1)
        valid_subjects = subject_missing[subject_missing < missing_threshold].index
        self.data = self.data[self.data['SWANID'].isin(valid_subjects)]
        
        # Final cleanup of any remaining NaN values
        all_vars = cognitive_vars + [f'{var}_change' for var in cognitive_vars]
        self.data = self.data.dropna(subset=all_vars)
        
        print(f"Data shape after preprocessing: {self.data.shape}")
        print(f"Number of unique subjects: {self.data['SWANID'].nunique()}")
        print(f"Number of visits per subject (min, max): ({self.data.groupby('SWANID').size().min()}, {self.data.groupby('SWANID').size().max()})")

    def create_sequences(self):
        """Create sequences for RNN input."""
        features = ['STATUS', 'TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        target_vars = ['TOTIDE1_change', 'TOTIDE2_change', 'NERVES_change', 
                      'SAD_change', 'FEARFULA_change']
        
        sequences = []
        targets = []
        
        for _, subject_data in self.data.groupby('SWANID'):
            if len(subject_data) >= self.sequence_length + 1:
                for i in range(len(subject_data) - self.sequence_length):
                    seq = subject_data[features].iloc[i:i+self.sequence_length].values
                    target = subject_data[target_vars].iloc[i+self.sequence_length].values
                    
                    if not (np.isnan(seq).any() or np.isnan(target).any()):
                        sequences.append(seq)
                        targets.append(target)
        
        if not sequences:
            raise ValueError("No valid sequences could be created from the data")
        
        print(f"Created {len(sequences)} valid sequences")
        return np.array(sequences), np.array(targets)

    def build_model(self, input_shape, output_shape):
        """Build the RNN model."""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def train_model(self):
        """Train the prediction model."""
        try:
            # Create sequences
            print("\nCreating sequences...")
            X, y = self.create_sequences()
            
            print(f"Sequence shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale the data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            # Reshape and scale
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
            X_test_scaled = scaler_X.transform(X_test_reshaped)
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)
            
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Build and train model
            model = self.build_model(
                input_shape=(self.sequence_length, X_train.shape[-1]),
                output_shape=y_train.shape[-1]
            )
            
            print("\nTraining model...")
            history = model.fit(
                X_train_scaled, y_train_scaled,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Evaluate model
            predictions_scaled = model.predict(X_test_scaled)
            predictions = scaler_y.inverse_transform(predictions_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            return model, mse, r2, predictions, y_test
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise

    def analyze_predictions(self, predictions, y_test):
        """Analyze and visualize prediction results."""
        target_vars = ['TOTIDE1_change', 'TOTIDE2_change', 'NERVES_change', 
                      'SAD_change', 'FEARFULA_change']
        
        for i, var in enumerate(target_vars):
            plt.figure(figsize=(10, 6))
            
            plt.scatter(y_test[:, i], predictions[:, i], alpha=0.5)
            plt.plot([y_test[:, i].min(), y_test[:, i].max()], 
                    [y_test[:, i].min(), y_test[:, i].max()], 
                    'r--', lw=2)
            
            plt.title(f'Predicted vs Actual Changes in {var}')
            plt.xlabel('Actual Changes')
            plt.ylabel('Predicted Changes')
            
            r2 = r2_score(y_test[:, i], predictions[:, i])
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
                    transform=plt.gca().transAxes)
            
            plt.show()

def run_ml_analysis():
    print("Initializing analysis...")
    analysis = MenopausePredictionModel("processed_combined_data.csv")
    
    print("\nTraining model...")
    trained_model, mse, r2, predictions, y_test = analysis.train_model()
    
    print(f"\nOverall Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    print("\nAnalyzing predictions...")
    analysis.analyze_predictions(predictions, y_test)

if __name__ == "__main__":
    run_ml_analysis()