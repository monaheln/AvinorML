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
- `october_2025_predictions.csv`: Final prediction file
- `feature_importance.csv`: Explanation of the model's decision basis

**Installation and Execution**:
```bash
# Requirements: Python 3.9+, pandas, scikit-learn, numpy
pip install pandas scikit-learn numpy

# Run the model
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
- **High-risk hours**: 1,256 hours identified (25% of total)

The model shows that the number of scheduled flights and time intervals between flights are the strongest indicators of concurrency risk, which aligns well with operational experience.