#!/usr/bin/env python3
"""
Simplified AFIS Concurrency Prediction Model
Focus on core features and fast execution for demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

class SimplifiedAFISModel:
    """Streamlined model focusing on key features."""

    def __init__(self, data_path="202509_Datasett/"):
        self.data_path = data_path

    def load_and_prepare_data(self):
        """Load existing training data and enhance with key features."""
        print("Loading and preparing data...")

        # Load existing training data
        self.training_data = pd.read_csv(f"{self.data_path}training_data.csv")
        self.training_data['date'] = pd.to_datetime(self.training_data['date'])

        # Load October schedule for predictions
        self.oct_schedule = pd.read_csv(f"{self.data_path}schedule_oct2025.csv")

        print(f"Training data: {len(self.training_data)} records")
        print(f"October schedule: {len(self.oct_schedule)} flights")

        return self

    def engineer_core_features(self, data_df):
        """Engineer the most important features efficiently."""
        features_df = data_df.copy()

        # Time-based features
        features_df['hour_of_day'] = features_df['hour']
        features_df['day_of_week'] = features_df['date'].dt.day_of_week
        features_df['month'] = features_df['date'].dt.month
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)

        # Hour groupings for patterns
        features_df['hour_group'] = pd.cut(features_df['hour'],
                                         bins=[0, 6, 12, 18, 24],
                                         labels=['night', 'morning', 'afternoon', 'evening'])

        # Airport group encoding
        le_group = LabelEncoder()
        features_df['airport_group_encoded'] = le_group.fit_transform(features_df['airport_group'])

        # Season encoding
        le_season = LabelEncoder()
        features_df['season_encoded'] = le_season.fit_transform(features_df['feat_season'])

        # Existing scheduled features 
        # feat_sched_flights_cnt and feat_sched_concurrence are key predictors

        # Interaction features
        features_df['flights_x_hour'] = features_df['feat_sched_flights_cnt'] * features_df['hour_of_day']
        features_df['concurrence_risk'] = (features_df['feat_sched_flights_cnt'] > 1).astype(int)

        print(f"Core features engineered: {features_df.shape[1]} columns")

        return features_df

    def train_model(self, features_df):
        """Train Random Forest model with core features."""
        print("Training model...")

        # Select feature columns
        feature_cols = [
            'hour_of_day', 'day_of_week', 'month', 'is_weekend',
            'airport_group_encoded', 'season_encoded',
            'feat_sched_flights_cnt', 'feat_sched_concurrence',
            'flights_x_hour', 'concurrence_risk'
        ]

        # Add hour group dummies
        hour_group_dummies = pd.get_dummies(features_df['hour_group'], prefix='hour_group')
        features_df = pd.concat([features_df, hour_group_dummies], axis=1)
        feature_cols.extend(hour_group_dummies.columns.tolist())

        X = features_df[feature_cols].fillna(0)
        y = features_df['target']

        # Time-based split
        cutoff_date = pd.Timestamp('2024-01-01')
        train_mask = features_df['date'] < cutoff_date
        val_mask = features_df['date'] >= cutoff_date

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
        print(f"Training positive rate: {y_train.mean():.3f}")
        print(f"Validation positive rate: {y_val.mean():.3f}")

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)

        print(f"Validation AUC: {auc:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

        return model, feature_cols, auc, feature_importance

    def predict_october(self, model, feature_cols):
        """Generate October 2025 predictions."""
        print("Generating October 2025 predictions...")

        # Create October template
        oct_dates = pd.date_range('2025-10-01', '2025-10-31', freq='H')
        groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        oct_data = []
        for date_hour in oct_dates:
            for group in groups:
                # Simple flight count estimation from schedule
                hour_flights = len(self.oct_schedule[
                    (pd.to_datetime(self.oct_schedule['std']).dt.date == date_hour.date()) &
                    (pd.to_datetime(self.oct_schedule['std']).dt.hour == date_hour.hour) &
                    ((self.oct_schedule['dep_airport_group'] == group) |
                     (self.oct_schedule['arr_airport_group'] == group))
                ])

                oct_data.append({
                    'airport_group': group,
                    'date': date_hour,
                    'hour': date_hour.hour,
                    'target': 0,  # Placeholder
                    'feat_season': 'autumn',
                    'feat_sched_flights_cnt': hour_flights,
                    'feat_sched_concurrence': 1 if hour_flights > 1 else 0
                })

        oct_df = pd.DataFrame(oct_data)

        # Engineer features
        oct_features = self.engineer_core_features(oct_df)

        # Add missing hour group columns
        hour_group_dummies = pd.get_dummies(oct_features['hour_group'], prefix='hour_group')
        oct_features = pd.concat([oct_features, hour_group_dummies], axis=1)

        # Generate predictions
        X_oct = oct_features[feature_cols].fillna(0)
        predictions = model.predict_proba(X_oct)[:, 1]

        # Format results
        results = pd.DataFrame({
            'airport_group': oct_features['airport_group'],
            'date': oct_features['date'].dt.date,
            'hour': oct_features['hour'],
            'pred': predictions
        })

        print(f"October predictions: {len(results)} records")
        print(f"Average probability: {predictions.mean():.4f}")
        print(f"High risk hours (>0.5): {(predictions > 0.5).sum()}")

        return results

if __name__ == "__main__":
    print("=== Simplified AFIS Concurrency Model ===")

    # Initialize and run
    model = SimplifiedAFISModel()
    model.load_and_prepare_data()

    # Engineer features on a subset for testing
    sample_data = model.training_data.sample(50000, random_state=42)
    features_df = model.engineer_core_features(sample_data)

    # Train model
    trained_model, feature_cols, auc, importance = model.train_model(features_df)

    # Generate October predictions
    oct_predictions = model.predict_october(trained_model, feature_cols)

    # Save results
    oct_predictions.to_csv('october_2025_predictions.csv', index=False)
    importance.to_csv('feature_importance.csv', index=False)

    print("\n=== Results Saved ===")
    print("october_2025_predictions.csv - Final predictions")
    print("feature_importance.csv - Model explainability")
    print(f"Model AUC: {auc:.4f}")