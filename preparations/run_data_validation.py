import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_validation import DataValidator

variables1 = ['STATUS', 'TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
variables2 = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS', 'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG']
variables3 = ['LISTEN', 'TAKETOM', 'HELPSIC', 'CONFIDE', 'EMOCTDW', 'EMOACCO', 'EMOCARE', 'INTERFR', 'SOCIAL', 'INCOME', 'HOW_HAR', 'BCINCML', 'DEGREE']

data = pd.read_csv('processed_combined_data.csv')
variables = variables1 + variables2 + variables3
output_dir = 'data_validation_output'

validator = DataValidator(
    data=data,
    variables=variables,
    output_dir=output_dir,
    plotting=False
)

# Run validation checks
validation_results = validator.run_checks(
    checks=['distributions', 'group_sizes', 'multicollinearity', 'stationarity', 'heteroscedasticity']
)