from utils.data_validation import DataValidator
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

variables1 = ['STATUS', 'TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
variables2 = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS', 'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG', 'LANGCOG']
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
    checks=['missing', 'distributions', 'group_sizes', 'homogeneity', 'multicollinearity', 'independence'],
    grouping_var='STATUS'
)