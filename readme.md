# AFIS Concurrency Prediction - Avinor Data Competition

## Overview

This repository contains the complete solution for Avinor's "Når går det på høygir?" data competition, predicting when AFIS (Aerodrome Flight Information Service) operators will experience concurrent aircraft communications.

**Competition Goal**: Predict hourly probability of concurrency events for 7 airport groups in October 2025.

**Final Result**: AUC-ROC 0.9563 on validation data, 5,047 hourly predictions generated.

## Repository Structure

```
├── 202509_Datasett/           # Original competition data
│   ├── training_data.csv      # Historical hourly data with targets
│   ├── historical_flights.csv # Flight-level data 2018-2025
│   ├── schedule_oct2025.csv   # October 2025 flight schedule (original)
│   ├── inference_data_oct2025.csv # Original inference features
│   └── airportgroups.csv      # Airport group mappings
├── schedule_oct2025_updated (1).csv   # Updated October 2025 schedule 
├── inference_data_oct2025_updated.csv # Updated inference features 
├── afis_simple_model.py       # Main prediction model (WORKING)
├── afis_concurrency_model.py  # Advanced implementation with full features
├── oktober_backtest.py        # Validation against October 2024 data
├── Helness_october_2025_predictions_updated.csv  # Final submission file
├── Helness_october_2025_predictions.csv  # Previous submission file
├── october_2025_predictions.csv  # Latest model output
├── feature_importance.csv     # Model explainability data
├── AFIS_Rapport.tex          # Technical documentation (LaTeX)
├── AFIS_Model_Documentation.md # Technical documentation (Markdown)
└── README.md                 # Competition submission overview
```

## Quick Start

### Requirements
- Python 3.9+
- pandas, scikit-learn, numpy

### Run Main Model
```bash
pip install pandas scikit-learn numpy
python afis_simple_model.py
```

**Output**:
- `october_2025_predictions.csv` - Latest model output
- `Helness_october_2025_predictions_updated.csv` - Final competition submission (copy of above)
- `feature_importance.csv` - Model explainability
- Console output with model performance metrics

## Model Overview

### Approach
- **Algorithm**: Random Forest Classifier
- **Problem Type**: Binary classification (concurrency/no concurrency per hour)
- **Evaluation Metric**: AUC-ROC (competition requirement)
- **Validation**: Time-based split (pre-2024 training, 2024+ validation)

### Key Features
1. **Scheduled concurrency** (34% importance) - Direct overlap indicators
2. **Flight count per hour** (20% importance) - More flights = higher risk
3. **Concurrency risk flag** (20% importance) - Binary indicator for >1 flights
4. **Flight-hour interaction** (14% importance) - Combined volume/timing effects
5. **Time-based features** - Hour of day, day of week, seasonality

### Results
- **Validation AUC**: 0.9563 (excellent discrimination)
- **October 2024 backtest**: 0.7322 (realistic expectation)
- **Predictions**: 1,506 high-risk hours identified (>50% probability)
- **Volume calibration**: Higher risk assessment due to 581 additional flights in updated data
- **Average probability**: 0.3286 

## Files Description

### Core Implementation
- **`afis_simple_model.py`**: Main working model, production-ready
- **`afis_concurrency_model.py`**: Extended implementation with communication window calculations

### Analysis & Validation
- **`oktober_backtest.py`**: Tests our Oct 2025 predictions against Oct 2024 reality
- **`feature_importance.csv`**: Ranked feature importance for model interpretability

### Documentation
- **`AFIS_Rapport.tex`**: Comprehensive technical report following competition requirements
- **`AFIS_Model_Documentation.md`**: Technical documentation in Markdown format
- **`README.md`**: Competition submission summary

### Data & Results
- **`Helness_october_2025_predictions_updated.csv`**: Final competition submission (5,047 predictions using updated data)
- **`Helness_october_2025_predictions.csv`**: Previous submission (original data)
- **`october_2025_predictions.csv`**: Latest model output
- **`202509_Datasett/`**: Original competition datasets
- **`schedule_oct2025_updated (1).csv`**: Updated October 2025 schedule with 581 additional flights
- **`inference_data_oct2025_updated.csv`**: Updated inference features with corrected values

## Methodology Highlights

### Domain-Specific Approach
- **Communication Windows**: Arrivals (16min before to 5min after), Departures (15min before to 8min after)
- **Concurrency Definition**: Overlapping communication windows within airport groups
- **AFIS Context**: Remote tower operations with 1 operator managing up to 3 airports

### Model Validation
- **Time-based splits** prevent data leakage
- **October backtest** provides realistic performance expectations
- **Feature importance** ensures regulatory transparency

### Innovation Points
- Custom feature engineering for aviation domain
- Balanced approach between accuracy and explainability
- Comprehensive validation including year-over-year testing

## Competition Performance

**Strengths**:
- High validation AUC (0.956) demonstrates strong pattern recognition
- Volume-calibrated predictions match historical rates
- Explainable features align with aviation domain knowledge
- Robust validation methodology

**Realistic Expectations**:
- October backtest (AUC 0.732) suggests actual performance will be good but not perfect
- Model captures fundamental patterns but may not predict all year-to-year variations

## Development Notes

### Alternative Approaches Considered
1. **XGBoost**: Implemented but Random Forest performed similarly with better interpretability
2. **October-only training**: Could have trained only on October historical data for seasonal specificity
3. **Weather integration**: METAR data integration planned but not implemented due to time constraints
4. **Communication window simulation**: Full implementation available in `afis_concurrency_model.py`

### Lessons Learned
1. **Feature importance matters**: Simple features (flight count) often outperform complex ones
2. **Validation strategy is critical**: Time-based splits essential for temporal prediction tasks
3. **Domain knowledge valuable**: Aviation-specific features significantly improved performance

## Future Improvements

1. **Weather data integration**: Add METAR weather conditions
2. **Delay modeling**: Incorporate cascading delay effects
3. **Real-time adaptation**: Dynamic model updating with new observations
4. **Ensemble methods**: Combine multiple model types for robustness

## Competition Context

**Avinor's Challenge**: "Når går det på høygir?" - predicting AFIS workload peaks
**Business Value**: Optimize staffing for remote tower operations
**Regulatory Requirement**: Explainable models for aviation safety compliance
**Evaluation**: AUC-ROC on actual October 2025 outcomes (post-competition)

---

*Developed for Avinor Data Competition 2025*
*Author: Mona Helness*
*Final AUC: 0.9563 (validation), realistic expectation: ~0.73 based on backtesting*