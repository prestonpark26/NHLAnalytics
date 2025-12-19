# Project Structure for NHL Shot Analytics

This document outlines the recommended structure for scaling your NHL analytics project from data exploration to production-ready applications.

```
NHLAnalytics/
├── README.md                    # Project overview and setup instructions
├── .gitignore                   # Git ignore patterns
├── requirements.txt             # Python dependencies (or pyproject.toml)
├── environment.yml              # Conda environment (optional)
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Original, immutable data
│   │   └── nhl_raw_plays.parquet
│   ├── processed/               # Cleaned, transformed data
│   │   └── .gitkeep
│   ├── external/                # Third-party data sources
│   │   └── .gitkeep
│   └── models/                  # Trained model artifacts
│       └── .gitkeep
│
├── notebooks/                    # Jupyter notebooks (renamed from dataExploration)
│   ├── 01_data_exploration/     # Initial exploration
│   │   ├── nhlAnalyticsV1.ipynb
│   │   └── nhlAnalyticsV2.ipynb
│   ├── 02_feature_engineering/  # Feature creation
│   │   └── .gitkeep
│   ├── 03_modeling/             # Model development
│   │   └── .gitkeep
│   └── 04_visualization/        # Dashboard prototypes
│       └── .gitkeep
│
├── src/                         # Source code (production-ready)
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── collectors/          # API data collection
│   │   │   ├── __init__.py
│   │   │   └── nhl_api.py       # NHL API client
│   │   ├── processors/          # Data transformation
│   │   │   ├── __init__.py
│   │   │   └── data_cleaning.py
│   │   └── loaders/             # Data loading utilities
│   │       ├── __init__.py
│   │       └── parquet_loader.py
│   │
│   ├── features/                # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_builder.py
│   │
│   ├── models/                  # ML models and training
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── visualization/           # Plotting and visualization
│   │   ├── __init__.py
│   │   └── charts.py
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── helpers.py
│
├── config/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   └── logging.yaml             # Logging configuration
│
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_data/
│   │   └── sample_data.parquet
│   ├── test_collectors/
│   │   └── test_nhl_api.py
│   └── test_processors/
│       └── test_data_cleaning.py
│
├── scripts/                     # Standalone scripts
│   ├── download_data.py         # Data collection scripts
│   ├── train_model.py           # Training scripts
│   └── generate_report.py       # Report generation
│
├── docs/                        # Documentation
│   ├── api/                     # API documentation
│   ├── analysis/                # Analysis reports
│   └── architecture.md          # System architecture
│
├── backend/                     # FastAPI backend (when ready)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── endpoints/
│   │   ├── models/
│   │   └── services/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                    # React frontend (when ready)
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── Dockerfile
│
└── .github/                     # GitHub workflows (CI/CD)
    └── workflows/
        └── ci.yml
```
