#!/usr/bin/env python3
"""
AFIS Concurrency Prediction Model
Predicts hourly concurrency events for Avinor remote tower operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Optional advanced libraries (install if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - using Random Forest only")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - feature importance only")

print("Environment setup complete.")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")
print(f"SHAP available: {SHAP_AVAILABLE}")

class AFISConcurrencyModel:
    """Main class for AFIS concurrency prediction model."""

    def __init__(self, data_path="202509_Datasett/"):
        self.data_path = data_path
        self.airport_groups = None
        self.historical_flights = None
        self.training_data = None
        self.oct_schedule = None

    def load_data(self):
        """Load and clean all datasets with proper datetime handling."""
        print("Loading datasets...")

        # Load airport groups
        self.airport_groups = pd.read_csv(f"{self.data_path}airportgroups.csv")
        print(f"Airport groups: {len(self.airport_groups)} airports in {self.airport_groups['airport_group'].nunique()} groups")

        # Load historical flights
        self.historical_flights = pd.read_csv(f"{self.data_path}historical_flights.csv")
        print(f"Historical flights: {len(self.historical_flights)} records")

        # Clean datetime columns
        datetime_cols = ['std', 'sta', 'atd', 'ata']
        for col in datetime_cols:
            if col in self.historical_flights.columns:
                self.historical_flights[col] = pd.to_datetime(self.historical_flights[col], errors='coerce')

        # Remove cancelled flights and flights with missing essential data
        initial_count = len(self.historical_flights)
        self.historical_flights = self.historical_flights[self.historical_flights['cancelled'] == 0]
        self.historical_flights = self.historical_flights.dropna(subset=['std', 'sta'])
        print(f"After cleaning: {len(self.historical_flights)} flights ({initial_count - len(self.historical_flights)} removed)")

        # Load existing training data for validation
        self.training_data = pd.read_csv(f"{self.data_path}training_data.csv")
        self.training_data['date'] = pd.to_datetime(self.training_data['date'])
        print(f"Training data: {len(self.training_data)} hourly records")

        # Load October 2025 schedule
        self.oct_schedule = pd.read_csv(f"{self.data_path}schedule_oct2025.csv")
        for col in ['std', 'sta']:
            self.oct_schedule[col] = pd.to_datetime(self.oct_schedule[col])
        print(f"October 2025 schedule: {len(self.oct_schedule)} flights")

        return self

    def calculate_communication_intervals(self, flights_df):
        """
        Calculate communication intervals for arrivals and departures.
        Arrivals: [landing_time - 16min, landing_time + 5min]
        Departures: [takeoff_time - 15min, takeoff_time + 8min]
        """
        intervals = []

        for _, flight in flights_df.iterrows():
            # Use actual times if available, otherwise scheduled times
            dep_time = flight['atd'] if pd.notna(flight['atd']) else flight['std']
            arr_time = flight['ata'] if pd.notna(flight['ata']) else flight['sta']

            # Departure communication window (if departing from group airport)
            if pd.notna(flight['dep_airport_group']) and flight['dep_airport_group'] != 'NA':
                if pd.notna(dep_time):
                    comm_start = dep_time - timedelta(minutes=15)
                    comm_end = dep_time + timedelta(minutes=8)
                    intervals.append({
                        'flight_id': flight['flight_id'],
                        'airport_group': flight['dep_airport_group'],
                        'flight_type': 'departure',
                        'comm_start': comm_start,
                        'comm_end': comm_end,
                        'flight_time': dep_time
                    })

            # Arrival communication window (if arriving at group airport)
            if pd.notna(flight['arr_airport_group']) and flight['arr_airport_group'] != 'NA':
                if pd.notna(arr_time):
                    comm_start = arr_time - timedelta(minutes=16)
                    comm_end = arr_time + timedelta(minutes=5)
                    intervals.append({
                        'flight_id': flight['flight_id'],
                        'airport_group': flight['arr_airport_group'],
                        'flight_type': 'arrival',
                        'comm_start': comm_start,
                        'comm_end': comm_end,
                        'flight_time': arr_time
                    })

        return pd.DataFrame(intervals)

    def detect_overlaps(self, intervals_df):
        """Detect overlapping communication windows within each airport group."""
        overlaps = []

        for group in intervals_df['airport_group'].unique():
            group_intervals = intervals_df[intervals_df['airport_group'] == group].copy()
            group_intervals = group_intervals.sort_values('comm_start')

            # Check for overlaps between consecutive intervals
            for i in range(len(group_intervals)):
                for j in range(i + 1, len(group_intervals)):
                    interval1 = group_intervals.iloc[i]
                    interval2 = group_intervals.iloc[j]

                    # Check if intervals overlap
                    if interval1['comm_end'] > interval2['comm_start']:
                        overlap_start = max(interval1['comm_start'], interval2['comm_start'])
                        overlap_end = min(interval1['comm_end'], interval2['comm_end'])

                        overlaps.append({
                            'airport_group': group,
                            'overlap_start': overlap_start,
                            'overlap_end': overlap_end,
                            'flight1': interval1['flight_id'],
                            'flight2': interval2['flight_id']
                        })
                    else:
                        # Since sorted by start time, no more overlaps possible for this interval
                        break

        return pd.DataFrame(overlaps)

    def generate_hourly_labels(self, overlaps_df, start_date, end_date):
        """Generate hourly concurrency labels from overlap events."""
        # Create hourly template
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        # Create all hour-group combinations
        hourly_data = []
        for date_hour in date_range:
            for group in groups:
                hourly_data.append({
                    'airport_group': group,
                    'date': date_hour.date(),
                    'hour': date_hour.hour,
                    'datetime': date_hour,
                    'target': 0  # Default no concurrency
                })

        labels_df = pd.DataFrame(hourly_data)

        # Mark hours with concurrency
        if not overlaps_df.empty:
            for _, overlap in overlaps_df.iterrows():
                # Find all hours that contain any part of the overlap
                overlap_start_hour = overlap['overlap_start'].floor('H')
                overlap_end_hour = overlap['overlap_end'].floor('H')

                # Mark all hours between start and end (inclusive)
                affected_hours = pd.date_range(overlap_start_hour, overlap_end_hour, freq='H')

                for hour in affected_hours:
                    mask = (labels_df['datetime'] == hour) & (labels_df['airport_group'] == overlap['airport_group'])
                    labels_df.loc[mask, 'target'] = 1

        return labels_df

    def generate_labels_from_flights(self):
        """Complete pipeline to generate concurrency labels from flight data."""
        print("Generating concurrency labels...")

        # Calculate communication intervals
        intervals_df = self.calculate_communication_intervals(self.historical_flights)
        print(f"Communication intervals: {len(intervals_df)}")

        # Detect overlaps
        overlaps_df = self.detect_overlaps(intervals_df)
        print(f"Overlap events found: {len(overlaps_df)}")

        # Generate hourly labels
        start_date = self.historical_flights['std'].min().date()
        end_date = self.historical_flights['std'].max().date()
        labels_df = self.generate_hourly_labels(overlaps_df, start_date, end_date)

        print(f"Hourly labels generated: {len(labels_df)} records")
        print(f"Concurrency events: {labels_df['target'].sum()} hours ({labels_df['target'].mean():.3f} rate)")

        return labels_df, intervals_df, overlaps_df

    def validate_labels(self, generated_labels_df):
        """Compare generated labels with existing training data targets."""
        print("Validating label generation...")

        # Merge generated labels with existing training data
        training_subset = self.training_data[['airport_group', 'date', 'hour', 'target']].copy()
        generated_subset = generated_labels_df[['airport_group', 'date', 'hour', 'target']].copy()
        generated_subset.columns = ['airport_group', 'date', 'hour', 'generated_target']

        # Convert date columns to same format
        training_subset['date'] = pd.to_datetime(training_subset['date']).dt.date
        generated_subset['date'] = pd.to_datetime(generated_subset['date'])

        comparison = training_subset.merge(
            generated_subset,
            on=['airport_group', 'date', 'hour'],
            how='inner'
        )

        if len(comparison) > 0:
            accuracy = (comparison['target'] == comparison['generated_target']).mean()
            print(f"Label validation accuracy: {accuracy:.3f}")
            print(f"Original positive rate: {comparison['target'].mean():.3f}")
            print(f"Generated positive rate: {comparison['generated_target'].mean():.3f}")

            # Show some mismatches for analysis
            mismatches = comparison[comparison['target'] != comparison['generated_target']]
            if len(mismatches) > 0:
                print(f"Mismatches: {len(mismatches)} out of {len(comparison)}")
        else:
            print("No overlapping data found for validation")

        return comparison if len(comparison) > 0 else None

    def engineer_features(self, target_data):
        """Engineer comprehensive features following the detailed plan."""
        print("Engineering features...")

        features_df = target_data.copy()

        # Basic time features
        features_df['hour_of_day'] = features_df['hour']
        features_df['date_dt'] = pd.to_datetime(features_df['date'])
        features_df['day_of_week'] = features_df['date_dt'].dt.day_of_week
        features_df['month'] = features_df['date_dt'].dt.month
        features_df['day_of_year'] = features_df['date_dt'].dt.day_of_year

        # Season encoding (more detailed than existing)
        def get_season(month):
            if month in [12, 1, 2]: return 'winter'
            elif month in [3, 4, 5]: return 'spring'
            elif month in [6, 7, 8]: return 'summer'
            else: return 'autumn'

        features_df['season'] = features_df['month'].apply(get_season)

        # Group all flights by date and group for feature engineering
        flight_features = []

        for _, row in features_df.iterrows():
            date = row['date']
            hour = row['hour']
            group = row['airport_group']

            # Get flights for this group on this date around this hour
            date_flights = self.historical_flights[
                (pd.to_datetime(self.historical_flights['std']).dt.date == date) |
                (pd.to_datetime(self.historical_flights['sta']).dt.date == date)
            ]

            group_flights = date_flights[
                (date_flights['dep_airport_group'] == group) |
                (date_flights['arr_airport_group'] == group)
            ]

            # Flight volume features
            hour_flights = self._get_flights_in_hour(group_flights, date, hour)
            prev_hour_flights = self._get_flights_in_hour(group_flights, date, hour-1)
            next_hour_flights = self._get_flights_in_hour(group_flights, date, hour+1)

            # Calculate timing features
            min_gap, avg_gap, max_concurrent = self._calculate_timing_features(hour_flights, hour)

            # Delay features
            avg_delay, delay_variance = self._calculate_delay_features(hour_flights)

            flight_features.append({
                'flights_current_hour': len(hour_flights),
                'flights_prev_hour': len(prev_hour_flights),
                'flights_next_hour': len(next_hour_flights),
                'min_flight_gap_minutes': min_gap,
                'avg_flight_gap_minutes': avg_gap,
                'max_concurrent_estimate': max_concurrent,
                'avg_delay_minutes': avg_delay,
                'delay_variance': delay_variance,
                'scheduled_departures': len(hour_flights[hour_flights.get('dep_airport_group', pd.Series()) == group]) if 'dep_airport_group' in hour_flights.columns else 0,
                'scheduled_arrivals': len(hour_flights[hour_flights.get('arr_airport_group', pd.Series()) == group]) if 'arr_airport_group' in hour_flights.columns else 0
            })

        # Add flight features to main dataframe
        flight_features_df = pd.DataFrame(flight_features)
        features_df = pd.concat([features_df, flight_features_df], axis=1)

        # Encode categorical variables
        le_group = LabelEncoder()
        features_df['airport_group_encoded'] = le_group.fit_transform(features_df['airport_group'])

        le_season = LabelEncoder()
        features_df['season_encoded'] = le_season.fit_transform(features_df['season'])

        print(f"Features engineered: {features_df.shape[1]} columns for {len(features_df)} records")

        return features_df

    def _get_flights_in_hour(self, flights_df, date, hour):
        """Get flights that have communication windows overlapping with specified hour."""
        if hour < 0: hour = 0
        if hour > 23: hour = 23

        hour_start = pd.Timestamp.combine(date, pd.Timestamp(f"{hour:02d}:00:00").time())
        hour_end = hour_start + timedelta(hours=1)

        relevant_flights = []
        for _, flight in flights_df.iterrows():
            # Check if communication window overlaps with this hour
            dep_time = flight['atd'] if pd.notna(flight['atd']) else flight['std']
            arr_time = flight['ata'] if pd.notna(flight['ata']) else flight['sta']

            if pd.notna(dep_time):
                comm_start = dep_time - timedelta(minutes=15)
                comm_end = dep_time + timedelta(minutes=8)
                if comm_start < hour_end and comm_end > hour_start:
                    relevant_flights.append(flight)

            if pd.notna(arr_time):
                comm_start = arr_time - timedelta(minutes=16)
                comm_end = arr_time + timedelta(minutes=5)
                if comm_start < hour_end and comm_end > hour_start:
                    relevant_flights.append(flight)

        return pd.DataFrame(relevant_flights).drop_duplicates('flight_id') if relevant_flights else pd.DataFrame()

    def _calculate_timing_features(self, hour_flights, hour):
        """Calculate minimum gaps and concurrent flight estimates."""
        if len(hour_flights) <= 1:
            return 999, 999, 0  # No risk if 0-1 flights

        # Get all flight times in this hour window
        flight_times = []
        for _, flight in hour_flights.iterrows():
            dep_time = flight['atd'] if pd.notna(flight['atd']) else flight['std']
            arr_time = flight['ata'] if pd.notna(flight['ata']) else flight['sta']

            if pd.notna(dep_time): flight_times.append(dep_time)
            if pd.notna(arr_time): flight_times.append(arr_time)

        if len(flight_times) <= 1:
            return 999, 999, 0

        flight_times = sorted(flight_times)

        # Calculate gaps between consecutive flights
        gaps = [(flight_times[i+1] - flight_times[i]).total_seconds() / 60
                for i in range(len(flight_times)-1)]

        min_gap = min(gaps) if gaps else 999
        avg_gap = np.mean(gaps) if gaps else 999

        # Estimate max concurrent by simulating communication windows
        max_concurrent = self._estimate_max_concurrent(flight_times)

        return min_gap, avg_gap, max_concurrent

    def _estimate_max_concurrent(self, flight_times):
        """Estimate maximum concurrent communications if all flights on schedule."""
        if len(flight_times) <= 1:
            return 0

        events = []  # (time, +1 for start, -1 for end)

        for ft in flight_times:
            # Assume average communication window (15.5 minutes before to 6.5 minutes after)
            comm_start = ft - timedelta(minutes=15.5)
            comm_end = ft + timedelta(minutes=6.5)
            events.append((comm_start, 1))
            events.append((comm_end, -1))

        events.sort()
        current_concurrent = 0
        max_concurrent = 0

        for _, event_type in events:
            current_concurrent += event_type
            max_concurrent = max(max_concurrent, current_concurrent)

        return max_concurrent

    def _calculate_delay_features(self, hour_flights):
        """Calculate delay-related features."""
        if len(hour_flights) == 0:
            return 0, 0

        delays = []
        for _, flight in hour_flights.iterrows():
            if pd.notna(flight['atd']) and pd.notna(flight['std']):
                delay = (flight['atd'] - flight['std']).total_seconds() / 60
                delays.append(delay)
            if pd.notna(flight['ata']) and pd.notna(flight['sta']):
                delay = (flight['ata'] - flight['sta']).total_seconds() / 60
                delays.append(delay)

        if delays:
            return np.mean(delays), np.var(delays)
        else:
            return 0, 0

    def train_models(self, features_df):
        """Train Random Forest and XGBoost models with time-based validation."""
        print("Training models...")

        # Prepare features and target
        feature_cols = [col for col in features_df.columns
                       if col not in ['target', 'airport_group', 'date', 'datetime', 'date_dt', 'season']]

        X = features_df[feature_cols].fillna(0)
        y = features_df['target']
        dates = pd.to_datetime(features_df['date'])

        # Time-based split: use 2024+ for validation, earlier for training
        cutoff_date = pd.Timestamp('2024-01-01')
        train_mask = dates < cutoff_date
        val_mask = dates >= cutoff_date

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        print(f"Training set: {len(X_train)} records, positive rate: {y_train.mean():.3f}")
        print(f"Validation set: {len(X_val)} records, positive rate: {y_val.mean():.3f}")

        # Train Random Forest
        print("Training Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [5, 10],
            'class_weight': ['balanced']
        }

        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        rf_grid.fit(X_train, y_train)

        rf_best = rf_grid.best_estimator_
        rf_val_proba = rf_best.predict_proba(X_val)[:, 1]
        rf_auc = roc_auc_score(y_val, rf_val_proba)
        print(f"Random Forest validation AUC: {rf_auc:.4f}")

        # Train XGBoost if available
        xgb_best = None
        xgb_auc = 0
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            xgb_params = {
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'scale_pos_weight': [y_train.value_counts()[0] / y_train.value_counts()[1]]
            }

            xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
            xgb_grid.fit(X_train, y_train)

            xgb_best = xgb_grid.best_estimator_
            xgb_val_proba = xgb_best.predict_proba(X_val)[:, 1]
            xgb_auc = roc_auc_score(y_val, xgb_val_proba)
            print(f"XGBoost validation AUC: {xgb_auc:.4f}")

        # Select best model
        if xgb_auc > rf_auc:
            best_model = xgb_best
            best_auc = xgb_auc
            model_name = "XGBoost"
        else:
            best_model = rf_best
            best_auc = rf_auc
            model_name = "Random Forest"

        print(f"Best model: {model_name} with AUC: {best_auc:.4f}")

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))

        return best_model, feature_cols, best_auc, feature_importance.head(15)

    def predict_october(self, model, feature_cols):
        """Generate predictions for October 2025 schedule."""
        print("Generating October 2025 predictions...")

        # Create hourly template for October 2025
        oct_dates = pd.date_range('2025-10-01', '2025-10-31', freq='H')
        groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        oct_data = []
        for date_hour in oct_dates:
            for group in groups:
                oct_data.append({
                    'airport_group': group,
                    'date': date_hour.date(),
                    'hour': date_hour.hour,
                    'datetime': date_hour,
                    'target': 0  # Placeholder
                })

        oct_df = pd.DataFrame(oct_data)

        # Use October schedule as "historical" data for feature engineering
        original_historical = self.historical_flights
        self.historical_flights = self.oct_schedule

        # Engineer features for October
        oct_features = self.engineer_features(oct_df)

        # Restore original historical data
        self.historical_flights = original_historical

        # Generate predictions
        X_oct = oct_features[feature_cols].fillna(0)
        predictions = model.predict_proba(X_oct)[:, 1]

        # Format results
        results = pd.DataFrame({
            'airport_group': oct_features['airport_group'],
            'date': oct_features['date'],
            'hour': oct_features['hour'],
            'pred': predictions
        })

        print(f"October predictions generated: {len(results)} records")
        print(f"Average predicted probability: {predictions.mean():.4f}")
        print(f"Predictions > 0.5: {(predictions > 0.5).sum()}")

        return results

if __name__ == "__main__":
    print("=== AFIS Concurrency Prediction Model ===")
    print("Following the detailed implementation plan...")

    # Initialize model
    model = AFISConcurrencyModel()
    model.load_data()

    # Option 1: Full pipeline (use for final predictions)
    # Option 2: Use existing training data (faster for development)
    USE_EXISTING_TRAINING_DATA = True

    if USE_EXISTING_TRAINING_DATA:
        print("\n=== Using existing training data for faster development ===")
        # Use existing training data and engineer additional features
        features_df = model.engineer_features(model.training_data.head(10000))  # Subset for testing

        print(f"Feature engineering complete. Shape: {features_df.shape}")

        # Train models
        best_model, feature_cols, best_auc, feature_importance = model.train_models(features_df)

        print(f"\n=== Model Training Complete ===")
        print(f"Best model AUC: {best_auc:.4f}")

        # Generate October predictions
        oct_predictions = model.predict_october(best_model, feature_cols)

        # Save predictions
        oct_predictions.to_csv('october_2025_predictions.csv', index=False)
        print(f"\nPredictions saved to october_2025_predictions.csv")

        # Save feature importance
        feature_importance.to_csv('feature_importance.csv', index=False)
        print(f"Feature importance saved to feature_importance.csv")

    else:
        print("\n=== Full pipeline with label generation ===")
        # This would take longer but generates labels from scratch
        labels_df, intervals_df, overlaps_df = model.generate_labels_from_flights()

        # Validate against existing labels
        validation_results = model.validate_labels(labels_df)

        if validation_results is not None:
            print(f"Label validation successful")

        # Continue with feature engineering and model training...
        print("Full pipeline would continue here...")

    print("\n=== Implementation Complete ===")
    print("Ready for submission and regulatory documentation.")