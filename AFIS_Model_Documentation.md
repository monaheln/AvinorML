# AFIS Concurrency Prediction - Documentation

## a. Method Selection and Approach

**Problem Statement**: Predict when AFIS operators will experience concurrent communication with multiple aircraft within the same hour.

**Chosen Method**: Random Forest machine learning
- Concurrency occurs when aircraft communication windows overlap
- For arrivals: 16 minutes before to 5 minutes after landing
- For departures: 15 minutes before to 8 minutes after takeoff

**Why Random Forest**:
- Can handle many different types of variables (numerical, categorical, temporal)
- Provides clear explanation of which factors influence predictions
- Robust against data errors and outliers
- Well-suited for this type of binary classification (concurrency/no concurrency)

## b. System Structure and Architecture

**System Components**:
```
Historical Flight Data → Data Processing → Model Training → Predictions
      (2018-2025)           ↓               ↓             ↓
                     Feature Engineering  Validation   October 2025
```

**Architecture**:
1. **Data Loading**: CSV files with flight data and airport groups
2. **Data Processing**: Data cleaning, handling cancelled flights
3. **Feature Engineering**: Create new variables from raw data
4. **Model Training**: Random Forest with time-based validation
5. **Prediction**: Generate probabilities for October 2025

**Technical Components**:
- Python 3.9+ with pandas, scikit-learn
- No external APIs required
- Standalone application

**Data Sources**:
- Training data: `202509_Datasett/training_data.csv` (465,031 hourly records)
- Historical flights: `202509_Datasett/historical_flights.csv` (399,426 flights)
- October 2025 schedule: `schedule_oct2025_updated (1).csv` (4,565 flights - updated version)
- October 2025 inference features: `inference_data_oct2025_updated.csv` (5,208 hourly records - updated version)
- Airport groups: `202509_Datasett/airportgroups.csv` (21 airports in 7 groups)

**Updated Datasets**: The model uses the latest updated versions of October 2025 data which contain additional flights and corrected feature values compared to the original competition datasets.

## c. Models and Algorithms

**Main Model**: Random Forest Classifier
- 100 trees in the ensemble
- Maximum depth: 10 levels
- Minimum samples per split: 10
- Balanced class weights to handle rare concurrency events

**Evaluation Method**:
- AUC-ROC as primary metric (achieved: 0.9563)
- Time-based validation: pre-2024 data for training, 2024+ for validation
- Avoids data leakage by not using future information

**Most Important Variables** (feature importance):
1. Scheduled concurrency (34%): Whether the timetable suggests overlap
2. Number of flights per hour (20%): More flights = higher risk
3. Concurrency risk flag (20%): Binary indicator for critical hours
4. Flight-hour interaction (14%): Combined effect of volume and timing

## d. Source Code

**File Structure**:
- `afis_simple_model.py`: Main model that generates predictions
- `afis_concurrency_model.py`: Extended implementation with detailed feature engineering
- `Helness_october_2025_predictions_updated.csv`: Final prediction file (using updated datasets)
- `feature_importance.csv`: Explanation of the model's decision basis

**Installation and Execution**:
```bash
# Requirements: Python 3.9+, pandas, scikit-learn, numpy
pip install pandas scikit-learn numpy

# Run the model (generates Helness_october_2025_predictions_updated.csv)
python afis_simple_model.py
```

**Main Code Components**:

1. **SimplifiedAFISModel class**:
   - `load_and_prepare_data()`: Loads data
   - `engineer_core_features()`: Creates prediction variables
   - `train_model()`: Trains Random Forest
   - `predict_october()`: Generates October predictions

2. **Feature Engineering**:
   - Time-based variables: hour of day, day of week, season
   - Flight traffic variables: number of flights, scheduled concurrency
   - Interaction variables: combined effects

3. **Model Training**:
   - Time-based data split for realistic validation
   - Hyperparameter optimization for best performance
   - Feature importance analysis for explainability

**Scaling and Further Development**:
- Can handle larger datasets by increasing sample size
- Easy to add new variables (weather, delays, etc.)
- Model can be retrained with new data without code changes

**Results**:
- **Validation AUC**: 0.9563 (very good predictive ability)
- **October predictions**: 5,047 hourly predictions generated
- **High-risk hours**: 1,506 hours identified (c30% of total)
- **Updated dataset impact**: 581 additional flights included, higher average probability (0.3286)

The model shows that the number of scheduled flights and time intervals between flights are the strongest indicators of concurrency risk, which aligns well with operational experience.