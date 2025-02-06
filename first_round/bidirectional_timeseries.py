import pandas as pd
import numpy as np
from semopy import Model
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.correlation_tools import cov_nearest
import networkx as nx
import matplotlib.pyplot as plt


class LongitudinalSEMModel:
    def __init__(self, data_path):
        """
        Initialize the model with the dataset.

        :param data_path: Path to the dataset (CSV file).
        """
        self.data_path = data_path
        self.data = None

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # Convert necessary columns to numeric
        numeric_columns = self.data.select_dtypes(include=["object"]).columns
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop rows with all missing values
        self.data.dropna(how="all", inplace=True)

        # Ensure VISIT is categorical
        if "VISIT" in self.data.columns:
            self.data["VISIT"] = self.data["VISIT"].astype("category")

        print("Data loaded and prepared successfully.")

    def check_missing_data(self):
        """
        Check for missing data and print a summary.
        """
        self.data.fillna(0, inplace=True)
        missing_summary = self.data.isnull().sum()
        print("Missing values summary:")
        print(missing_summary[missing_summary > 0])

    def standardize_data(self):
        """
        Standardize numeric variables to have mean=0 and std=1.
        """
        numeric_columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        scaler = StandardScaler()
        self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
        print("Data standardized successfully.")

    def remove_collinear_variables(self, threshold=0.95):
        """
        Remove one of two variables that are highly correlated (above a threshold).
        """
        corr_matrix = self.data.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        print(f"Dropping highly correlated columns: {to_drop}")
        self.data.drop(columns=to_drop, inplace=True)

    def regularize_covariance_matrix(self):
        """
        Regularize the covariance matrix to ensure it is positive definite.
        """
        cov_matrix = self.data.cov()
        reg_cov_matrix = cov_nearest(cov_matrix, threshold=1e-4)
        print("Covariance matrix regularized successfully.")
        return reg_cov_matrix

    def fit_longitudinal_model(self):
        """
        Fit a longitudinal SEM model and print results.
        """
        # Define the SEM model for longitudinal analysis
        model_desc = """
        # Repeated measures for mental health indicators
        DEPRESS.VISIT1 ~ BLUES.VISIT1 + MOODCHG.VISIT1 + HAPPY.VISIT1 + ENDO.VISIT1
        DEPRESS.VISIT2 ~ DEPRESS.VISIT1 + BLUES.VISIT2 + MOODCHG.VISIT2 + HAPPY.VISIT2 + ENDO.VISIT2
        BLUES.VISIT1 ~ ENDO.VISIT1 + PELVCPN.VISIT1 + ABBLEED.VISIT1 + FIBRUTR.VISIT1
        BLUES.VISIT2 ~ BLUES.VISIT1 + ENDO.VISIT2 + PELVCPN.VISIT2 + ABBLEED.VISIT2 + FIBRUTR.VISIT2

        # Lagged relationships
        DEPRESS.VISIT2 ~ DEPRESS.VISIT1
        BLUES.VISIT2 ~ BLUES.VISIT1

        # Moderators and mediators
        SLEEPQL.VISIT1 ~ EXERCIS.VISIT1 + YOGA.VISIT1 + OVERHLT.VISIT1 + BODYPAI.VISIT1
        RESTLES.VISIT1 ~ EXERCIS.VISIT1 + YOGA.VISIT1 + OVERHLT.VISIT1 + BODYPAI.VISIT1

        # Controls
        DEPRESS.VISIT1 ~ OVERHLT.VISIT1 + BODYPAI.VISIT1
        BLUES.VISIT1 ~ OVERHLT.VISIT1 + BODYPAI.VISIT1
        """

        # Initialize the SEM model
        self.sem_model = Model(model_desc)

        # Regularize covariance matrix
        reg_cov_matrix = self.regularize_covariance_matrix()

        # Load dataset into the model
        self.sem_model.load_dataset(self.data, covariance=reg_cov_matrix)

        # Fit the SEM model
        self.sem_model.fit()

        # Print model summary
        print("\nLongitudinal Model Summary:")
        print(self.sem_model.inspect())

        # Save results to a CSV file
        self.sem_model.inspect().to_csv("longitudinal_model_summary.csv", index=False)

    def visualize_results_networkx(self):
        """
        Visualize the SEM model relationships using networkx.
        """
        G = nx.DiGraph()

        # Add nodes for each time point
        nodes = [
            "DEPRESS.VISIT1", "BLUES.VISIT1", "DEPRESS.VISIT2", "BLUES.VISIT2",
            "ENDO.VISIT1", "ENDO.VISIT2", "EXERCIS.VISIT1", "SLEEPQL.VISIT1"
        ]
        G.add_nodes_from(nodes)

        # Add edges (example relationships from SEM)
        edges = [
            ("ENDO.VISIT1", "DEPRESS.VISIT1"), ("DEPRESS.VISIT1", "DEPRESS.VISIT2"),
            ("BLUES.VISIT1", "DEPRESS.VISIT2"), ("EXERCIS.VISIT1", "SLEEPQL.VISIT1"),
            ("SLEEPQL.VISIT1", "DEPRESS.VISIT1"),
        ]
        G.add_edges_from(edges)

        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)  # Layout for nodes
        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_weight="bold", node_size=2000)
        plt.title("Longitudinal SEM Path Diagram")
        plt.show()

    def run(self):
        """
        Execute the full pipeline: Load, preprocess, and fit the model.
        """
        self.load_and_prepare_data()
        self.check_missing_data()
        self.standardize_data()
        self.remove_collinear_variables()
        self.fit_longitudinal_model()
        self.visualize_results_networkx()


if __name__ == "__main__":
    # Define the dataset path
    data_path = "combined_data.csv"

    # Initialize and run the model
    longitudinal_model = LongitudinalSEMModel(data_path)
    longitudinal_model.run()