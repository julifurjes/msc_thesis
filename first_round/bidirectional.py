import pandas as pd
from semopy import Model, Optimizer
import networkx as nx
import matplotlib.pyplot as plt

class BidirectionalModel:
    def __init__(self, data_path):
        """
        Initialize the model with the dataset.

        :param data_path: Path to the dataset (CSV file).
        """
        self.data_path = data_path

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # Convert necessary columns to numeric
        all_columns = [
            # Mental health indicators
            "DEPRESS", "BLUES", "MOODCHG", "HAPPY", "CONTROL", "PILING", "FEARFUL", 
            "SLEEPQL", "RESTLES", "QLTYLIF",
            # Gynecological symptoms
            "ENDO", "PELVCPN", "ABBLEED", "FIBRUTR", "STATUS",
            # Moderators/mediators
            "EXERCIS", "YOGA", "AGE", "MARITAL", "CRNTMAR", "INCOME", "SLEEPQL",
            # Controls
            "OVERHLT", "BODYPAI", "COURTES", "RESPECT", "INSULTE", "HARASSE",
        ]
        for col in all_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop rows with missing values
        self.data = self.data.dropna(subset=all_columns)

    def fit_bidirectional_model(self):
        """
        Fit a bidirectional SEM model and print results.
        """
        # Define the SEM model
        model_desc = """
        # Mental health indicators as dependent variables
        DEPRESS ~ ENDO + PELVCPN + ABBLEED + FIBRUTR + STATUS + SLEEPQL + RESTLES
        BLUES ~ ENDO + PELVCPN + ABBLEED + FIBRUTR + STATUS + SLEEPQL + RESTLES
        QLTYLIF ~ ENDO + PELVCPN + ABBLEED + FIBRUTR + STATUS + SLEEPQL + RESTLES
        FEARFUL ~ ENDO + PELVCPN + ABBLEED + FIBRUTR + STATUS + SLEEPQL + RESTLES

        # Gynecological symptoms as dependent variables
        ENDO ~ DEPRESS + BLUES + QLTYLIF + FEARFUL + AGE + MARITAL + INCOME
        PELVCPN ~ DEPRESS + BLUES + QLTYLIF + FEARFUL + AGE + MARITAL + INCOME
        ABBLEED ~ DEPRESS + BLUES + QLTYLIF + FEARFUL + AGE + MARITAL + INCOME
        FIBRUTR ~ DEPRESS + BLUES + QLTYLIF + FEARFUL + AGE + MARITAL + INCOME

        # Moderators and mediators
        SLEEPQL ~ EXERCIS + YOGA + OVERHLT + BODYPAI
        RESTLES ~ EXERCIS + YOGA + OVERHLT + BODYPAI

        # Controls
        DEPRESS ~ OVERHLT + BODYPAI + COURTES + RESPECT + INSULTE + HARASSE
        BLUES ~ OVERHLT + BODYPAI + COURTES + RESPECT + INSULTE + HARASSE
        QLTYLIF ~ OVERHLT + BODYPAI + COURTES + RESPECT + INSULTE + HARASSE
        """

        # Initialize the SEM model
        self.sem_model = Model(model_desc)

        # Load dataset into the model
        self.sem_model.load_dataset(self.data)

        # Fit the SEM model
        self.sem_model.fit()

        # Print model summary
        print("\nModel Summary:")
        print(self.sem_model.inspect())

        self.sem_model.inspect().to_csv("bidirectional_model_summary.csv", index=False)

    def visualize_results_networkx(self):
        """
        Visualize the SEM model relationships using networkx.
        """
        G = nx.DiGraph()

        # Add nodes
        nodes = [
            "DEPRESS", "BLUES", "QLTYLIF", "FEARFUL",
            "ENDO", "PELVCPN", "ABBLEED", "FIBRUTR",
            "SLEEPQL", "RESTLES", "EXERCIS", "YOGA", "OVERHLT", "BODYPAI"
        ]
        G.add_nodes_from(nodes)

        # Add edges (example relationships from SEM)
        edges = [
            ("ENDO", "DEPRESS"), ("PELVCPN", "DEPRESS"), ("ABBLEED", "DEPRESS"),
            ("DEPRESS", "ENDO"), ("DEPRESS", "PELVCPN"), ("QLTYLIF", "ENDO"),
            ("EXERCIS", "SLEEPQL"), ("YOGA", "RESTLES"), ("OVERHLT", "DEPRESS"),
        ]
        G.add_edges_from(edges)

        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)  # Layout for nodes
        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_weight="bold", node_size=2000)
        plt.title("SEM Path Diagram")
        plt.show()

    def visualize_results(self):
        """
        Visualize SEM model relationships with matplotlib.
        """
        plt.figure(figsize=(10, 6))
        
        # Draw arrows manually
        plt.arrow(0.2, 0.8, 0.3, -0.2, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
        plt.arrow(0.5, 0.6, 0.3, -0.2, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
        plt.text(0.2, 0.8, "ENDO", fontsize=12, ha="center")
        plt.text(0.5, 0.6, "DEPRESS", fontsize=12, ha="center")
        plt.text(0.8, 0.4, "QLTYLIF", fontsize=12, ha="center")
        
        # Customize appearance
        plt.axis("off")
        plt.title("SEM Path Diagram")
        plt.show()
            

if __name__ == "__main__":
    # Define the dataset path
    data_path = "combined_data.csv"

    # Initialize and run the model
    bidi_model = BidirectionalModel(data_path)

    # Load and prepare the data
    bidi_model.load_and_prepare_data()

    # Fit the bidirectional SEM model
    bidi_model.fit_bidirectional_model()

    # Visualize the SEM diagram
    bidi_model.visualize_results()