# Longitudinal Data Analysis Project

This repository contains a complete data analysis framework and the corresponding AI declaration.

## Setting Up Your Research Environment

### System Requirements

This project requires Python 3.x and several libraries. A virtual environment can be used to keep all dependencies organized.

### Getting the Repository

Start by downloading the repository to your computer:

```bash
git clone https://github.com/julifurjes/msc_thesis.git
cd msc_thesis
```

### Creating a Virtual Environment

To ensure your analysis runs consistently, create a Python environment:

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Installing Required Libraries

Install all necessary packages using the requirements file:

```bash
pip install -r requirements.txt
```

### Data Requirements and Setup

**Important**: This project requires external data that cannot be included in the repository due to GDPR.

You need to download the dataset from the ICPSR: https://www.icpsr.umich.edu/web/ICPSR/series/00253

After downloading the data:
1. Create a `data` folder in the main project directory
2. Place all downloaded files in this `data` folder and follow the naming regulations in the code

Your project structure should look like this:

```
project-folder/
├── data/           # Your downloaded ICPSR data goes here
├── preparations/   # Data processing scripts
├── etc.
```

## Running the Analysis

### Step 1: Data Processing

The analysis requires specific data preparation steps that must be completed in order. Navigate to: the preparations folder:

```bash
cd preparations
```

Run these two scripts in the exact same order:

```bash
# First: Create the main data structure
python create_dataframe.py

# Second: Handle missing data
python impute_data.py
```

### Step 2: Optional Data Description

After preparing your data, you can run additional analysis:

```bash
# Descriptive statistics
python data_desc.py

# Validate data quality
python run_data_validation.py
```

### Step 3: Running the Models

The project includes three different modeling approaches. Each model is in its own folder and can be run separately:

```bash
cd [model-folder-name]
python longitudinal.py
```

Repeat this process for each of the three model folders to complete the full analysis.

## AI Declaration

Details about AI assistance in development can be found in the `AI_declaration.txt` file in the main directory.