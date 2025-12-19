# Quick Start Guide

This guide will help you run the split notebooks from start to finish.

## Prerequisites

1. **Python Environment**: Make sure you have Python 3.9+ installed
2. **Install Dependencies**: Install required packages

## Setup Steps

### 1. Install Dependencies

Open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

Or if you prefer using a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Jupyter

From the project root directory:

```bash
jupyter notebook
```

Or if you prefer JupyterLab:

```bash
jupyter lab
```

## Running the Notebooks

Run the notebooks **in order** (01 → 02 → 03 → 04 → 05):

### Notebook 1: Data Collection

**File**: `notebooks/01_data_exploration/01_data_collection.ipynb`

**What it does**:

- Generates game IDs for the 2023 NHL season
- Fetches play-by-play data from NHL API
- Saves to `data/raw/nhl_raw_plays.parquet`

**Time**: ~10-30 minutes (depending on API speed)

**Note**: If the data file already exists, it will skip collection and just load the existing file.

**To run**: Open the notebook and run all cells (Cell → Run All)

---

### Notebook 2: Data Cleaning

**File**: `notebooks/02_data_cleaning/02_data_cleaning.ipynb`

**What it does**:

- Filters for shot events only
- Flattens nested JSON columns
- Removes unnecessary columns
- Standardizes coordinates
- Handles missing data
- Creates target variable (is_goal)
- Saves to `data/processed/cleaned_shots.parquet`

**Time**: ~1-2 minutes

**Prerequisites**: Notebook 1 must have created `data/raw/nhl_raw_plays.parquet`

**To run**: Open the notebook and run all cells

---

### Notebook 3: Exploratory Data Analysis

**File**: `notebooks/03_exploratory_analysis/03_exploratory_analysis.ipynb`

**What it does**:

- Calculates basic features (distance, angle)
- Performs exploratory data analysis
- Creates visualizations (heatmaps, bar charts, scatter plots)
- Correlation analysis
- Saves to `data/processed/shots_with_basic_features.parquet`

**Time**: ~2-3 minutes

**Prerequisites**: Notebook 2 must have created `data/processed/cleaned_shots.parquet`

**To run**: Open the notebook and run all cells

---

### Notebook 4: Advanced Feature Engineering

**File**: `notebooks/04_advanced_features/04_advanced_features.ipynb`

**What it does**:

- Creates game situation features
- Creates time-based features
- Creates shot quality metrics
- Creates rebound features
- Encodes shot types and zones
- Saves to `data/processed/shots_with_all_features.parquet`

**Time**: ~1-2 minutes

**Prerequisites**: Notebook 3 must have created `data/processed/shots_with_basic_features.parquet`

**To run**: Open the notebook and run all cells

---

### Notebook 5: Modeling

**File**: `notebooks/05_modeling/05_modeling.ipynb`

**What it does**:

- Prepares data for modeling
- Trains Logistic Regression model (baseline)
- Trains XGBoost model
- Evaluates and compares models
- Shows feature importance
- Saves trained model to `data/models/xgboost_xg_model.pkl`

**Time**: ~2-5 minutes (depending on your machine)

**Prerequisites**: Notebook 4 must have created `data/processed/shots_with_all_features.parquet`

**To run**: Open the notebook and run all cells

---

## Quick Run (All at Once)

If you want to run everything in sequence, you can use this approach:

1. Open Jupyter from the project root
2. Navigate to each notebook folder in order
3. Run all cells in each notebook
4. Wait for each to complete before moving to the next

## Troubleshooting

### Import Errors

If you get import errors like `ModuleNotFoundError: No module named 'src'`:

**Solution**: Make sure you're running the notebooks from within their respective directories, or adjust the `sys.path.append('../../')` line if needed.

The notebooks use relative paths (`../../`) to access:

- `src/` modules (for reusable functions)
- `data/` directory (for data files)

### Path Issues

If you get file not found errors:

1. Make sure you're running notebooks from the correct directory structure
2. Check that previous notebooks have created the required data files
3. Verify the paths in each notebook match your directory structure

### Data Already Exists

If data files already exist:

- **Notebook 1**: Will skip API calls and load existing data
- **Notebooks 2-5**: Will overwrite existing processed data files

### API Rate Limiting

If Notebook 1 fails due to API rate limiting:

- The code includes a small delay (0.1 seconds) between requests
- If you still hit limits, you can increase the delay in `src/data/collectors/nhl_api.py`

## Expected Output Files

After running all notebooks, you should have:

```
data/
├── raw/
│   └── nhl_raw_plays.parquet          # From Notebook 1
├── processed/
│   ├── cleaned_shots.parquet          # From Notebook 2
│   ├── shots_with_basic_features.parquet  # From Notebook 3
│   └── shots_with_all_features.parquet    # From Notebook 4
└── models/
    ├── xgboost_xg_model.pkl          # From Notebook 5
    └── label_encoders.pkl             # From Notebook 5
```

## Running Individual Notebooks

You can run any notebook independently **as long as its input data file exists**. For example:

- To re-run just the modeling notebook: Make sure `data/processed/shots_with_all_features.parquet` exists
- To re-run feature engineering: Make sure `data/processed/shots_with_basic_features.parquet` exists

## Next Steps

After running all notebooks:

1. **Explore Results**: Check the model performance metrics in Notebook 5
2. **Experiment**: Try modifying features in Notebook 4
3. **Visualize**: Review the EDA plots in Notebook 3
4. **Deploy**: Use the saved model (`data/models/xgboost_xg_model.pkl`) for predictions
