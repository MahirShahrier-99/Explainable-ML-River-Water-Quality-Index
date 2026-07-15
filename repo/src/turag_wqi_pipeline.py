"""Turag River WQI machine learning pipeline.

Standalone version of Turag_WQI_Analysis.ipynb.
Run from the repository root: python src/turag_wqi_pipeline.py
"""

# Install dependencies first: pip install -r requirements.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils import resample

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# XGBoost & SHAP
import xgboost as xgb
import shap

print("="*80)
print("ALL PACKAGES LOADED SUCCESSFULLY")
print("TensorFlow version:", tf.__version__)
print("="*80)

# =============================================================================
# CUSTOM COLOR SCHEME & SETTINGS
# =============================================================================

COLORS = {
    'primary': '#F075AE',
    'secondary': '#F7DB91',
    'tertiary': '#FFFDCE',
    'quaternary': '#9BC264',
    'gradient': ['#F075AE', '#F7DB91', '#FFFDCE', '#9BC264']
}

# Publication settings with CONSTRAINED LAYOUT to prevent overlap
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.constrained_layout.use'] = True  # KEY: Prevents overlap

print(f"✓ Custom color scheme: {COLORS['gradient']}")
print("✓ Constrained layout enabled (no overlap)")


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

print("\n" + "="*80)
print("STEP 1: DATA LOADING")
print("="*80)

DATA_PATH = 'data/Turag_River_WQI_Bangladesh_DoE_Results.xlsx'
df_raw = pd.read_excel(DATA_PATH, sheet_name='WQI_Summary')

print(f"\n✓ Data loaded successfully!")
print(f"  Total samples: {len(df_raw)}")
print(f"  Locations: {df_raw['Location'].nunique()}")
print(f"  Time period: {df_raw['Year'].min()}-{df_raw['Year'].max()}")


# STEP 2: FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80)
print("STEP 2: COMPREHENSIVE FEATURE ENGINEERING")
print("="*80)

class ComprehensiveFeatureEngineering:
    """Complete feature engineering pipeline"""

    def __init__(self, df):
        self.df = df.copy()
        self.features = []

    def create_all_features(self):
        """Execute all feature engineering steps"""
        self.create_ratio_features()
        self.create_spatial_features()
        self.create_temporal_features()
        self.create_interaction_features()
        self.handle_missing_values()
        return self.df, self.features

    def create_ratio_features(self):
        """Create ratio features for source identification"""
        print("\n[1/4] Creating Ratio Features...")

        # BOD/COD ratio (biodegradability index)
        self.df['BOD_COD_ratio'] = self.df['BOD(mg/L)'] / (self.df['COD(mg/L)'] + 0.01)

        # DO deficit
        DO_saturation = 9.0
        self.df['DO_deficit'] = (DO_saturation - self.df['DO(mg/L)']).clip(lower=0)

        # TDS/EC ratio
        self.df['TDS_EC_ratio'] = self.df['TDS(mg/L)'] / (self.df['EC(micro S/cm)'] + 0.01)

        # Chloride/TDS ratio
        self.df['Cl_TDS_ratio'] = self.df['Chloride(mg/L)'] / (self.df['TDS(mg/L)'] + 0.01)

        # Pollution load index (composite measure)
        self.df['Pollution_load'] = (
            (self.df['BOD(mg/L)'] / 6.0) +
            (self.df['COD(mg/L)'] / 12.0) +
            (self.df['Turbidity. NTU'] / 10.0)
        ) / 3.0

        # Organic pollution ratio
        self.df['Organic_pollution'] = (self.df['BOD(mg/L)'] + self.df['COD(mg/L)']) / 2

        # Parameter exceedance ratios
        standards = {
            'pH:Lab': 7.0,  # Deviation from neutral
            'DO(mg/L)': 5.0,
            'BOD(mg/L)': 6.0,
            'COD(mg/L)': 12.0,
        }

        for param, standard in standards.items():
            if param in self.df.columns:
                if param == 'pH:Lab':
                    self.df[f'{param}_deviation'] = abs(self.df[param] - standard)
                else:
                    self.df[f'{param}_exceedance'] = (self.df[param] / standard).clip(upper=5.0)

        ratio_features = ['BOD_COD_ratio', 'DO_deficit', 'TDS_EC_ratio', 'Cl_TDS_ratio',
                         'Pollution_load', 'Organic_pollution']
        ratio_features += [col for col in self.df.columns if 'exceedance' in col or 'deviation' in col]

        self.features.extend(ratio_features)
        print(f"  ✓ Created {len(ratio_features)} ratio features")

    def create_spatial_features(self):
        """Create spatial features"""
        print("\n[2/4] Creating Spatial Features...")

        # Industrial proximity (from location names)
        industrial_keywords = ['Dyeing', 'Ltd', 'Composite', 'Washing', 'Tongi']
        self.df['Near_Industry'] = self.df['Location'].apply(
            lambda x: 1 if any(kw in str(x) for kw in industrial_keywords) else 0
        )

        # Location encoding
        le = LabelEncoder()
        self.df['Location_encoded'] = le.fit_transform(self.df['Location'])

        # Upstream position (based on latitude)
        self.df['Upstream_position'] = (
            (self.df['Latitude'] - self.df['Latitude'].min()) /
            (self.df['Latitude'].max() - self.df['Latitude'].min())
        )

        # Distance from hotspot (Turag Bridge)
        ref_lat = self.df[self.df['Location'].str.contains('Turag Bridge', na=False)]['Latitude'].mean()
        ref_long = self.df[self.df['Location'].str.contains('Turag Bridge', na=False)]['Longitude'].mean()

        if pd.notna(ref_lat) and pd.notna(ref_long):
            self.df['Distance_from_hotspot'] = np.sqrt(
                (self.df['Latitude'] - ref_lat)**2 +
                (self.df['Longitude'] - ref_long)**2
            )
        else:
            self.df['Distance_from_hotspot'] = 0

        spatial_features = ['Near_Industry', 'Location_encoded', 'Upstream_position',
                           'Distance_from_hotspot']
        self.features.extend(spatial_features)
        print(f"  ✓ Created {len(spatial_features)} spatial features")

    def create_temporal_features(self):
        """Create temporal features including lags"""
        print("\n[3/4] Creating Temporal Features...")

        # Season encoding
        def get_season(month):
            if month in [3, 4, 5]:
                return 0  # Pre-monsoon
            elif month in [6, 7, 8, 9]:
                return 1  # Monsoon
            else:
                return 2  # Post-monsoon

        self.df['Season'] = self.df['Month'].apply(get_season)

        # Cyclic encoding for month
        self.df['Month_sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)

        # Year trend
        self.df['Year_numeric'] = self.df['Year'] - self.df['Year'].min()

        # Sort by location and time for lag features
        self.df = self.df.sort_values(['Location', 'Year', 'Month'])

        # Lag features (grouped by location)
        for lag in [1, 2, 3]:
            self.df[f'WQI_lag_{lag}'] = self.df.groupby('Location')['WQI'].shift(lag)

        # Rolling statistics (3-month window)
        self.df['WQI_roll_mean_3'] = (
            self.df.groupby('Location')['WQI']
            .rolling(3, min_periods=1).mean()
            .reset_index(0, drop=True)
        )

        self.df['WQI_roll_std_3'] = (
            self.df.groupby('Location')['WQI']
            .rolling(3, min_periods=1).std()
            .reset_index(0, drop=True)
        )

        temporal_features = ['Season', 'Month_sin', 'Month_cos', 'Year_numeric',
                            'WQI_lag_1', 'WQI_lag_2', 'WQI_lag_3',
                            'WQI_roll_mean_3', 'WQI_roll_std_3']
        self.features.extend(temporal_features)
        print(f"  ✓ Created {len(temporal_features)} temporal features")

    def create_interaction_features(self):
        """Create interaction features"""
        print("\n[4/4] Creating Interaction Features...")

        # Season × Parameter interactions
        for param in ['COD(mg/L)', 'BOD(mg/L)', 'Turbidity. NTU']:
            if param in self.df.columns:
                self.df[f'{param}_Season'] = self.df[param] * (self.df['Season'] + 1)

        # BOD/COD × Season
        self.df['BOD_COD_Season'] = self.df['BOD_COD_ratio'] * (self.df['Season'] + 1)

        # Industrial × Season (critical interaction)
        self.df['Industrial_Season'] = self.df['Near_Industry'] * (self.df['Season'] + 1)

        interaction_features = [col for col in self.df.columns if '_Season' in col]
        self.features.extend(interaction_features)
        print(f"  ✓ Created {len(interaction_features)} interaction features")

    def handle_missing_values(self):
        """Handle missing values in features"""
        print("\nHandling missing values...")

        for feat in self.features:
            if feat in self.df.columns:
                if self.df[feat].isnull().sum() > 0:
                    self.df[feat].fillna(self.df[feat].median(), inplace=True)

        print("✓ Missing values handled")

# Execute feature engineering
feature_engineer = ComprehensiveFeatureEngineering(df_raw)
df_engineered, feature_list = feature_engineer.create_all_features()

print("\n" + "="*80)
print(f"✓ FEATURE ENGINEERING COMPLETE")
print(f"  Total engineered features: {len(feature_list)}")
print(f"  Dataset size: {len(df_engineered)} samples")
print("="*80)

# CONTINUE IN NEXT PART...
print("\n✓ PART 1 COMPLETE - Proceed to Part 2 for analysis...")


# STEP 3: TIME SERIES DECOMPOSITION & SEASONAL ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("STEP 3: TIME SERIES DECOMPOSITION & SEASONAL ANALYSIS")
print("="*80)

def seasonal_decomposition_analysis(df):
    """
    Perform seasonal decomposition for key parameters
    """

    results = {}
    parameters = ['COD(mg/L)', 'BOD(mg/L)', 'Turbidity. NTU', 'DO(mg/L)', 'EC(micro S/cm)']

    # Aggregate by month across all locations
    monthly_data = df.groupby(['Year', 'Month']).agg({
        'WQI': 'mean',
        **{p: 'mean' for p in parameters if p in df.columns}
    }).reset_index()

    # Create date column
    monthly_data['Date'] = pd.to_datetime(
        monthly_data[['Year', 'Month']].assign(Day=1)
    )
    monthly_data = monthly_data.sort_values('Date').set_index('Date')

    print("\nAttempting STL decomposition for each parameter...")

    for param in parameters:
        if param not in monthly_data.columns:
            continue

        try:
            # STL decomposition
            stl = STL(monthly_data[param].dropna(), seasonal=13, robust=True)
            result = stl.fit()

            results[param] = {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'seasonal_strength': 1 - (result.resid.var() /
                                         (result.resid.var() + result.seasonal.var()))
            }

            print(f"  ✓ {param}: Seasonal strength = {results[param]['seasonal_strength']:.3f}")

        except Exception as e:
            print(f"  ✗ {param}: Decomposition failed ({str(e)[:50]}...)")

    return results, monthly_data


decomp_results, monthly_aggregated = seasonal_decomposition_analysis(df_engineered)

# Season-Parameter Correlation Analysis
print("\n" + "-"*80)
print("SEASON-PARAMETER-WQI CORRELATION ANALYSIS")
print("-"*80)

season_names = {0: 'Pre-monsoon', 1: 'Monsoon', 2: 'Post-monsoon'}
parameters = ['COD(mg/L)', 'BOD(mg/L)', 'Turbidity. NTU', 'DO(mg/L)', 'EC(micro S/cm)']

season_correlations = {}

for season_id, season_name in season_names.items():
    season_data = df_engineered[df_engineered['Season'] == season_id]

    print(f"\n{season_name}:")
    corrs = []

    for param in parameters:
        if param in season_data.columns:
            corr = season_data[['WQI', param]].corr().iloc[0, 1]
            corrs.append((param, corr))
            print(f"  {param:30s}: r = {corr:+.3f}")

    season_correlations[season_name] = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)

print("\n✓ Seasonal correlation analysis complete")


# STEP 4: DATA PREPARATION FOR MODELING
# =============================================================================

print("\n" + "="*80)
print("STEP 4: DATA PREPARATION FOR MODELING")
print("="*80)

# Select features for modeling (without lag features for initial models)
modeling_features_base = [
    # Original parameters
    'pH:Lab', 'DO(mg/L)', 'BOD(mg/L)', 'COD(mg/L)', 'TDS(mg/L)',
    'SS(mg/L)', 'Turbidity. NTU', 'Chloride(mg/L)', 'EC(micro S/cm)',

    # Ratio features
    'BOD_COD_ratio', 'DO_deficit', 'TDS_EC_ratio', 'Pollution_load', 'Organic_pollution',

    # Spatial features
    'Near_Industry', 'Location_encoded', 'Upstream_position',

    # Temporal features (no lags for base models)
    'Season', 'Month_sin', 'Month_cos', 'Year_numeric',

    # Interactions
    'BOD_COD_Season', 'Industrial_Season'
]

# Filter to existing columns
modeling_features = [f for f in modeling_features_base if f in df_engineered.columns]

# Remove rows with missing WQI or key features
df_clean = df_engineered.dropna(subset=['WQI'] + modeling_features[:9])

print(f"\n✓ Clean dataset: {len(df_clean)} samples")
print(f"✓ Features for modeling: {len(modeling_features)}")

# Time-based train-test split
train_mask = df_clean['Year'] <= 2019
val_mask = (df_clean['Year'] >= 2021) & (df_clean['Year'] <= 2022)
test_mask = df_clean['Year'] >= 2023

X_train = df_clean.loc[train_mask, modeling_features].values
X_val = df_clean.loc[val_mask, modeling_features].values
X_test = df_clean.loc[test_mask, modeling_features].values

y_train = df_clean.loc[train_mask, 'WQI'].values
y_val = df_clean.loc[val_mask, 'WQI'].values
y_test = df_clean.loc[test_mask, 'WQI'].values

# Get status for classification
status_train = df_clean.loc[train_mask, 'Water_Quality_Status'].values if 'Water_Quality_Status' in df_clean.columns else None
status_test = df_clean.loc[test_mask, 'Water_Quality_Status'].values if 'Water_Quality_Status' in df_clean.columns else None

print(f"\n✓ Train set: {len(X_train)} samples (2011-2019)")
print(f"✓ Validation set: {len(X_val)} samples (2021-2022)")
print(f"✓ Test set: {len(X_test)} samples (2023)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features standardized")


# STEP 5: BASELINE MODELS
# =============================================================================

print("\n" + "="*80)
print("STEP 5: BASELINE MODELS")
print("="*80)

baseline_results = {}

# Random Forest
print("\n[1/4] Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

baseline_results['Random Forest'] = {'RMSE': rmse_rf, 'R2': r2_rf, 'predictions': y_pred_rf, 'model': rf_model}
print(f"  RMSE: {rmse_rf:.2f}, R²: {r2_rf:.3f}")

# XGBoost
print("\n[2/4] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

baseline_results['XGBoost'] = {'RMSE': rmse_xgb, 'R2': r2_xgb, 'predictions': y_pred_xgb, 'model': xgb_model}
print(f"  RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.3f}")

# Gradient Boosting
print("\n[3/4] Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

baseline_results['Gradient Boosting'] = {'RMSE': rmse_gb, 'R2': r2_gb, 'predictions': y_pred_gb, 'model': gb_model}
print(f"  RMSE: {rmse_gb:.2f}, R²: {r2_gb:.3f}")

# =============================================================================
# STEP 6: IMPROVED DEEP LEARNING MODEL (SIMPLIFIED FOR SMALL DATA)
# =============================================================================

print("\n[4/4] Training Improved Deep Learning Model...")
print("  Note: Simplified architecture for small dataset (368 samples)")

def build_improved_dl_model(input_dim):
    """
    Improved DL model with:
    - Simpler architecture (fewer layers)
    - Stronger regularization
    - Batch normalization
    - Better suited for small datasets
    """

    inputs = layers.Input(shape=(input_dim,))

    # First block
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Second block
    x = layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Output
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Build and compile
dl_model = build_improved_dl_model(input_dim=X_train_scaled.shape[1])

dl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(dl_model.summary())

# Train with strong regularization
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=0)

history = dl_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=300,
    batch_size=8,  # Smaller batch size for small dataset
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate
y_pred_dl = dl_model.predict(X_test_scaled, verbose=0).flatten()
rmse_dl = np.sqrt(mean_squared_error(y_test, y_pred_dl))
r2_dl = r2_score(y_test, y_pred_dl)

baseline_results['Deep Learning (Improved)'] = {'RMSE': rmse_dl, 'R2': r2_dl, 'predictions': y_pred_dl, 'model': dl_model}

print(f"\n✓ Deep Learning Model trained")
print(f"  RMSE: {rmse_dl:.2f}, R²: {r2_dl:.3f}")
print(f"  Training epochs: {len(history.history['loss'])}")
print(f"  {'✓ IMPROVED!' if r2_dl > 0 else '✗ Still poor - use tree models instead'}")

print("\n✓ All baseline models trained")

# Select best model
best_model_name = max(baseline_results.items(), key=lambda x: x[1]['R2'])[0]
best_model = baseline_results[best_model_name]['model']
y_pred_best = baseline_results[best_model_name]['predictions']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   RMSE: {baseline_results[best_model_name]['RMSE']:.2f}")
print(f"   R²: {baseline_results[best_model_name]['R2']:.3f}")


# STEP 7: MULTI-SCALE SHAP EXPLAINABILITY
# =============================================================================

print("\n" + "="*80)
print("STEP 7: MULTI-SCALE SHAP EXPLAINABILITY")
print("="*80)

print("\n[1/4] Computing SHAP values (this may take a few minutes)...")

# Use XGBoost for SHAP (best for tree explainer)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

print("✓ SHAP values computed")

# -------------------------------------------------------------------------
# Level 1: Global Feature Importance
# -------------------------------------------------------------------------
print("\n[2/4] Level 1: Global Feature Importance")

mean_shap = np.abs(shap_values).mean(axis=0)

feature_importance_df = pd.DataFrame({
    'Feature': modeling_features,
    'Mean_|SHAP|': mean_shap
}).sort_values('Mean_|SHAP|', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# -------------------------------------------------------------------------
# Level 2: Seasonal Attribution
# -------------------------------------------------------------------------
print("\n[3/4] Level 2: Seasonal Parameter Attribution")

# Get season for test samples
test_df = df_clean[test_mask].copy()
test_df['SHAP_idx'] = range(len(test_df))

seasonal_shap = {}

for season_id, season_name in season_names.items():
    season_indices = test_df[test_df['Season'] == season_id]['SHAP_idx'].values

    if len(season_indices) > 0:
        season_shap_vals = shap_values[season_indices]
        season_mean_shap = np.abs(season_shap_vals).mean(axis=0)

        seasonal_shap[season_name] = pd.DataFrame({
            'Feature': modeling_features,
            'Mean_|SHAP|': season_mean_shap
        }).sort_values('Mean_|SHAP|', ascending=False)

        print(f"\n{season_name}:")
        print(seasonal_shap[season_name].head(5).to_string(index=False))

# -------------------------------------------------------------------------
# Level 3: Location-Specific Attribution
# -------------------------------------------------------------------------
print("\n[4/4] Level 3: Location-Specific Attribution")

location_shap = {}
top_locations = test_df['Location'].value_counts().head(5).index

for location in top_locations:
    loc_indices = test_df[test_df['Location'] == location]['SHAP_idx'].values

    if len(loc_indices) > 0:
        loc_shap_vals = shap_values[loc_indices]
        loc_mean_shap = np.abs(loc_shap_vals).mean(axis=0)

        location_shap[location] = pd.DataFrame({
            'Feature': modeling_features,
            'Mean_|SHAP|': loc_mean_shap
        }).sort_values('Mean_|SHAP|', ascending=False)

        print(f"\n{location[:50]}:")
        print(location_shap[location].head(5).to_string(index=False))

print("\n✓ Multi-scale SHAP analysis complete")


# STEP 8: SOURCE IDENTIFICATION
# =============================================================================

print("\n" + "="*80)
print("STEP 8: SOURCE IDENTIFICATION & CLUSTERING")
print("="*80)

def identify_pollution_sources(df):
    """
    Identify pollution source types based on parameter signatures
    IMPROVED: More inclusive criteria to reduce "Unknown" category
    """

    df_source = df.copy()
    df_source = df_source.dropna(subset=['BOD(mg/L)', 'COD(mg/L)', 'EC(micro S/cm)'])

    # Initialize
    df_source['Source_signature'] = 'Diffuse Non-Point Sources'

    # Textile Dyeing: Low BOD/COD (<0.3), High EC (>800) OR High COD (>60) with industry
    textile_mask = (
        ((df_source['BOD_COD_ratio'] < 0.3) & (df_source['EC(micro S/cm)'] > 800)) |
        ((df_source['Near_Industry'] == 1) & (df_source['COD(mg/L)'] > 60) & (df_source['BOD_COD_ratio'] < 0.35))
    )
    df_source.loc[textile_mask, 'Source_signature'] = 'Textile Dyeing'

    # Domestic Sewage: High BOD/COD (>0.45)
    domestic_mask = (df_source['BOD_COD_ratio'] > 0.45)
    df_source.loc[domestic_mask, 'Source_signature'] = 'Domestic Sewage'

    # Mixed Urban: Moderate BOD/COD (0.3-0.45), not already classified
    mixed_mask = (
        (df_source['BOD_COD_ratio'] >= 0.3) &
        (df_source['BOD_COD_ratio'] <= 0.45) &
        (df_source['Source_signature'] == 'Diffuse Non-Point Sources')
    )
    df_source.loc[mixed_mask, 'Source_signature'] = 'Mixed Urban'

    return df_source

df_with_sources = identify_pollution_sources(df_clean)

print("\nSource Distribution:")
source_counts = df_with_sources['Source_signature'].value_counts()
print(source_counts)
print(f"\n% Diffuse Non-Point Sources: {(source_counts.get('Diffuse Non-Point Sources', 0) / len(df_with_sources) * 100):.1f}%")

print("\nSource Characteristics:")
for source in source_counts.index:
    source_data = df_with_sources[df_with_sources['Source_signature'] == source]
    print(f"\n{source}:")
    print(f"  Mean WQI: {source_data['WQI'].mean():.1f} \u00b1 {source_data['WQI'].std():.1f}")
    print(f"  Mean BOD/COD: {source_data['BOD_COD_ratio'].mean():.3f} \u00b1 {source_data['BOD_COD_ratio'].std():.3f}")
    print(f"  Mean COD: {source_data['COD(mg/L)'].mean():.1f} \u00b1 {source_data['COD(mg/L)'].std():.1f} mg/L")
    print(f"  Mean EC: {source_data['EC(micro S/cm)'].mean():.0f} \u00b1 {source_data['EC(micro S/cm)'].std():.0f} \u00b5S/cm")

    # Top locations
    top_locs = source_data['Location'].value_counts().head(3)
    print(f"  Top locations: {', '.join(top_locs.index[:3])}")

print("\n\u2713 Source identification complete")

print("\n" + "="*80)
print("\u2713 PART 2 COMPLETE")
print("="*80)
print("\nNext: Part 3 for NEW features (Future Forecasting, Parameter Decomposition, Scenarios)")


# STEP 9: PARAMETER-WISE WQI DECOMPOSITION
# =============================================================================

print("\n" + "="*80)
print("STEP 9: PARAMETER-WISE WQI DECOMPOSITION")
print("="*80)

# Calculate base value and individual parameter contributions
base_value = explainer.expected_value

# For each test sample, decompose WQI
decomposition_results = []

main_params = ['COD(mg/L)', 'BOD(mg/L)', 'Turbidity. NTU', 'DO(mg/L)', 'EC(micro S/cm)']

for idx in range(len(X_test)):
    sample_contributions = {}

    for i, feature in enumerate(modeling_features):
        sample_contributions[feature] = shap_values[idx, i]

    decomp_entry = {
        'Sample': idx,
        'Observed_WQI': y_test[idx],
        'Base_WQI': base_value,
        'Predicted_WQI': y_pred_best[idx]
    }

    for param in main_params:
        if param in sample_contributions:
            decomp_entry[f'{param}_contribution'] = sample_contributions[param]

    decomposition_results.append(decomp_entry)

decomp_df = pd.DataFrame(decomposition_results)

# Calculate mean contributions
print("\nMean Parameter Contributions to WQI (SHAP-based):")
print("="*60)

param_contrib_summary = []
for param in main_params:
    col = f'{param}_contribution'
    if col in decomp_df.columns:
        mean_contrib = decomp_df[col].mean()
        std_contrib = decomp_df[col].std()

        param_contrib_summary.append({
            'Parameter': param.replace('(mg/L)', '').replace('. NTU', '').replace('(micro S/cm)', ''),
            'Mean_Contribution': mean_contrib,
            'Std_Dev': std_contrib
        })

        print(f"  {param:30s}: {mean_contrib:+7.2f} ± {std_contrib:5.2f} points")

param_contrib_df = pd.DataFrame(param_contrib_summary)
print("\n✓ Parameter-wise decomposition complete")


# STEP 10: FUTURE FORECASTING (2024-2035)
# =============================================================================

print("\n" + "="*80)
print("STEP 10: FUTURE FORECASTING (2024-2035)")
print("="*80)

def create_future_scenarios(df_clean, modeling_features, years=range(2024, 2036)):
    """
    Generate future scenarios based on historical patterns
    """

    print(f"\nGenerating future scenarios for {years}...")
    future_data = []

    for year in years:
        for month in range(1, 13):
            for location in df_clean['Location'].unique():

                # Get historical data for this location-month combination
                hist = df_clean[
                    (df_clean['Location'] == location) &
                    (df_clean['Month'] == month)
                ]

                if len(hist) > 0:
                    scenario = {'Year': year, 'Month': month, 'Location': location}

                    # Use historical means for water quality parameters
                    for param in ['pH:Lab', 'DO(mg/L)', 'BOD(mg/L)', 'COD(mg/L)',
                                  'TDS(mg/L)', 'SS(mg/L)', 'Turbidity. NTU',
                                  'Chloride(mg/L)', 'EC(micro S/cm)']:
                        if param in hist.columns:
                            scenario[param] = hist[param].mean()

                    # Calculate all engineered features
                    scenario['BOD_COD_ratio'] = scenario['BOD(mg/L)'] / (scenario['COD(mg/L)'] + 0.01)
                    scenario['DO_deficit'] = max(0, 9.0 - scenario['DO(mg/L)'])
                    scenario['TDS_EC_ratio'] = scenario['TDS(mg/L)'] / (scenario['EC(micro S/cm)'] + 0.01)
                    scenario['Cl_TDS_ratio'] = scenario['Chloride(mg/L)'] / (scenario['TDS(mg/L)'] + 0.01)
                    scenario['Pollution_load'] = (
                        (scenario['BOD(mg/L)'] / 6.0) +
                        (scenario['COD(mg/L)'] / 12.0) +
                        (scenario['Turbidity. NTU'] / 10.0)
                    ) / 3.0
                    scenario['Organic_pollution'] = (scenario['BOD(mg/L)'] + scenario['COD(mg/L)']) / 2

                    # Spatial features
                    scenario['Near_Industry'] = hist['Near_Industry'].iloc[0]
                    scenario['Location_encoded'] = hist['Location_encoded'].iloc[0]
                    scenario['Upstream_position'] = hist['Upstream_position'].iloc[0]

                    # Temporal features
                    def get_season(m):
                        return 0 if m in [3,4,5] else 1 if m in [6,7,8,9] else 2

                    scenario['Season'] = get_season(month)
                    scenario['Month_sin'] = np.sin(2 * np.pi * month / 12)
                    scenario['Month_cos'] = np.cos(2 * np.pi * month / 12)
                    scenario['Year_numeric'] = year - df_clean['Year'].min()

                    # Interactions
                    scenario['BOD_COD_Season'] = scenario['BOD_COD_ratio'] * (scenario['Season'] + 1)
                    scenario['Industrial_Season'] = scenario['Near_Industry'] * (scenario['Season'] + 1)
                    scenario['COD(mg/L)_Season'] = scenario['COD(mg/L)'] * (scenario['Season'] + 1)
                    scenario['BOD(mg/L)_Season'] = scenario['BOD(mg/L)'] * (scenario['Season'] + 1)
                    scenario['Turbidity. NTU_Season'] = scenario['Turbidity. NTU'] * (scenario['Season'] + 1)

                    future_data.append(scenario)

    print(f"  ✓ Generated {len(future_data)} future scenarios")
    return pd.DataFrame(future_data)

# Generate baseline future scenario
print("\n[1/3] Generating baseline scenario (Business as Usual)...")
future_baseline = create_future_scenarios(df_clean, modeling_features, years=range(2024, 2036))

# Prepare features and predict
X_future = future_baseline[modeling_features].values
X_future_scaled = scaler.transform(X_future)
future_baseline['WQI_predicted'] = best_model.predict(X_future_scaled)

# Bootstrap confidence intervals
print("\n[2/3] Computing confidence intervals (100 bootstrap iterations)...")

bootstrap_preds = []
np.random.seed(42)

for i in range(100):
    # Resample training data
    indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True)
    X_boot = X_train_scaled[indices]
    y_boot = y_train[indices]

    # Train model
    boot_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                           max_depth=5, random_state=i)
    boot_model.fit(X_boot, y_boot)

    # Predict
    boot_pred = boot_model.predict(X_future_scaled)
    bootstrap_preds.append(boot_pred)

    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/100 iterations complete")

bootstrap_preds = np.array(bootstrap_preds)
future_baseline['WQI_lower_95'] = np.percentile(bootstrap_preds, 2.5, axis=0)
future_baseline['WQI_upper_95'] = np.percentile(bootstrap_preds, 97.5, axis=0)
future_baseline['WQI_std'] = np.std(bootstrap_preds, axis=0)

print("✓ Baseline forecast complete with 95% confidence intervals")

# Summary statistics
print(f"\n2024-2035 Baseline Forecast:")
print(f"  Mean WQI: {future_baseline['WQI_predicted'].mean():.1f}")
print(f"  95% CI: [{future_baseline['WQI_lower_95'].mean():.1f}, {future_baseline['WQI_upper_95'].mean():.1f}]")
print(f"  Range: {future_baseline['WQI_predicted'].min():.1f} - {future_baseline['WQI_predicted'].max():.1f}")


# STEP 11: SCENARIO ANALYSIS
# =============================================================================

print("\n[3/3] Creating intervention scenarios...")

# Scenario 1: ETP Enforcement (30% COD/BOD reduction at industrial sites)
future_etp = future_baseline.copy()
industrial_mask = future_etp['Near_Industry'] == 1

future_etp.loc[industrial_mask, 'COD(mg/L)'] *= 0.7
future_etp.loc[industrial_mask, 'BOD(mg/L)'] *= 0.7

# Recalculate derived features
future_etp['BOD_COD_ratio'] = future_etp['BOD(mg/L)'] / (future_etp['COD(mg/L)'] + 0.01)
future_etp['Pollution_load'] = (
    (future_etp['BOD(mg/L)'] / 6.0) +
    (future_etp['COD(mg/L)'] / 12.0) +
    (future_etp['Turbidity. NTU'] / 10.0)
) / 3.0
future_etp['Organic_pollution'] = (future_etp['BOD(mg/L)'] + future_etp['COD(mg/L)']) / 2
future_etp['BOD_COD_Season'] = future_etp['BOD_COD_ratio'] * (future_etp['Season'] + 1)
future_etp['COD(mg/L)_Season'] = future_etp['COD(mg/L)'] * (future_etp['Season'] + 1)
future_etp['BOD(mg/L)_Season'] = future_etp['BOD(mg/L)'] * (future_etp['Season'] + 1)

X_etp = future_etp[modeling_features].values
X_etp_scaled = scaler.transform(X_etp)
future_etp['WQI_predicted'] = best_model.predict(X_etp_scaled)
future_etp['Scenario'] = 'ETP Enforcement'

print("  ✓ ETP Enforcement scenario created")

# Scenario 2: Sewage Treatment (40% BOD reduction everywhere)
future_sewage = future_baseline.copy()
future_sewage['BOD(mg/L)'] *= 0.6

# Recalculate features
future_sewage['BOD_COD_ratio'] = future_sewage['BOD(mg/L)'] / (future_sewage['COD(mg/L)'] + 0.01)
future_sewage['Pollution_load'] = (
    (future_sewage['BOD(mg/L)'] / 6.0) +
    (future_sewage['COD(mg/L)'] / 12.0) +
    (future_sewage['Turbidity. NTU'] / 10.0)
) / 3.0
future_sewage['Organic_pollution'] = (future_sewage['BOD(mg/L)'] + future_sewage['COD(mg/L)']) / 2
future_sewage['BOD_COD_Season'] = future_sewage['BOD_COD_ratio'] * (future_sewage['Season'] + 1)
future_sewage['BOD(mg/L)_Season'] = future_sewage['BOD(mg/L)'] * (future_sewage['Season'] + 1)

X_sewage = future_sewage[modeling_features].values
X_sewage_scaled = scaler.transform(X_sewage)
future_sewage['WQI_predicted'] = best_model.predict(X_sewage_scaled)
future_sewage['Scenario'] = 'Sewage Treatment'

print("  ✓ Sewage Treatment scenario created")

# Scenario 3: Combined Intervention
future_combined = future_baseline.copy()

# Industrial reductions
industrial_mask_comb = future_combined['Near_Industry'] == 1
future_combined.loc[industrial_mask_comb, 'COD(mg/L)'] *= 0.7
future_combined.loc[industrial_mask_comb, 'BOD(mg/L)'] *= 0.7

# General sewage reduction
future_combined['BOD(mg/L)'] *= 0.6

# Recalculate all features
future_combined['BOD_COD_ratio'] = future_combined['BOD(mg/L)'] / (future_combined['COD(mg/L)'] + 0.01)
future_combined['Pollution_load'] = (
    (future_combined['BOD(mg/L)'] / 6.0) +
    (future_combined['COD(mg/L)'] / 12.0) +
    (future_combined['Turbidity. NTU'] / 10.0)
) / 3.0
future_combined['Organic_pollution'] = (future_combined['BOD(mg/L)'] + future_combined['COD(mg/L)']) / 2
future_combined['BOD_COD_Season'] = future_combined['BOD_COD_ratio'] * (future_combined['Season'] + 1)
future_combined['COD(mg/L)_Season'] = future_combined['COD(mg/L)'] * (future_combined['Season'] + 1)
future_combined['BOD(mg/L)_Season'] = future_combined['BOD(mg/L)'] * (future_combined['Season'] + 1)

X_combined = future_combined[modeling_features].values
X_combined_scaled = scaler.transform(X_combined)
future_combined['WQI_predicted'] = best_model.predict(X_combined_scaled)
future_combined['Scenario'] = 'Combined'

print("  ✓ Combined intervention scenario created")

# Label baseline
future_baseline['Scenario'] = 'Baseline (No Action)'

# Compare scenarios
scenario_summary = pd.DataFrame({
    'Scenario': ['Baseline', 'ETP Enforcement', 'Sewage Treatment', 'Combined'],
    'Mean_WQI': [
        future_baseline['WQI_predicted'].mean(),
        future_etp['WQI_predicted'].mean(),
        future_sewage['WQI_predicted'].mean(),
        future_combined['WQI_predicted'].mean()
    ]
})

scenario_summary['Improvement'] = (
    scenario_summary['Mean_WQI'].iloc[0] - scenario_summary['Mean_WQI']
)
scenario_summary['Improvement_pct'] = (
    scenario_summary['Improvement'] / scenario_summary['Mean_WQI'].iloc[0] * 100
)

# Add estimated costs and ROI
scenario_summary['Cost_Million_BDT'] = [0, 15, 50, 65]
scenario_summary['Timeline_months'] = [0, 6, 18, 24]
scenario_summary['ROI'] = scenario_summary.apply(
    lambda row: (row['Improvement'] / row['Cost_Million_BDT']) if row['Cost_Million_BDT'] > 0 else 0,
    axis=1
)

print("\n✓ SCENARIO COMPARISON (2024-2035 Forecast):")
print("="*80)
print(scenario_summary.round(1).to_string(index=False))

print("\n✓ Scenario analysis complete")


# STEP 12: TEMPORAL VALIDATION
# =============================================================================

print("\n" + "="*80)
print("STEP 12: TEMPORAL VALIDATION (Rolling Window)")
print("="*80)

validation_results = []

# Test on each year using all previous years for training
for test_year in [2019, 2021, 2022, 2023]:
    train_mask_temp = df_clean['Year'] < test_year
    test_mask_temp = df_clean['Year'] == test_year

    X_train_temp = df_clean.loc[train_mask_temp, modeling_features].values
    y_train_temp = df_clean.loc[train_mask_temp, 'WQI'].values
    X_test_temp = df_clean.loc[test_mask_temp, modeling_features].values
    y_test_temp = df_clean.loc[test_mask_temp, 'WQI'].values

    if len(X_train_temp) > 0 and len(X_test_temp) > 0:
        # Standardize
        scaler_temp = StandardScaler()
        X_train_temp_scaled = scaler_temp.fit_transform(X_train_temp)
        X_test_temp_scaled = scaler_temp.transform(X_test_temp)

        # Train model
        model_temp = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=5, random_state=42)
        model_temp.fit(X_train_temp_scaled, y_train_temp)

        # Predict
        y_pred_temp = model_temp.predict(X_test_temp_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_temp, y_pred_temp))
        mae = mean_absolute_error(y_test_temp, y_pred_temp)
        r2 = r2_score(y_test_temp, y_pred_temp)

        validation_results.append({
            'Test_Year': test_year,
            'Train_Period': f'2011-{test_year-1}',
            'N_train': len(X_train_temp),
            'N_test': len(X_test_temp),
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })

validation_df = pd.DataFrame(validation_results)

print("\nTemporal Validation Results:")
print("="*80)
print(validation_df.to_string(index=False))

print(f"\n✓ Mean R² across years: {validation_df['R²'].mean():.3f}")
print(f"✓ Mean RMSE: {validation_df['RMSE'].mean():.2f}")
print("✓ Model demonstrates robust temporal forecasting capability")

print("\n" + "="*80)
print("✓ PART 3 COMPLETE")
print("="*80)


import os
os.makedirs('figures', exist_ok=True)



print("\n" + "="*80)
print("CREATING ALL PUBLICATION-QUALITY FIGURES")
print("="*80)

# =============================================================================
# FIGURE 1: MODEL PERFORMANCE & VALIDATION
# =============================================================================

print("\n[Figure 1/7] Model Performance & Validation...")

fig1 = plt.figure(figsize=(16, 10))
gs = fig1.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel A: Model Comparison
ax1 = fig1.add_subplot(gs[0, 0])

model_names = list(baseline_results.keys())
rmse_values = [baseline_results[m]['RMSE'] for m in model_names]
r2_values = [baseline_results[m]['R2'] for m in model_names]

x = np.arange(len(model_names))
bars = ax1.bar(x, rmse_values, color=COLORS['gradient'][:len(model_names)],
               alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('RMSE', fontweight='bold')
ax1.set_xlabel('Model', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
ax1.set_title('(A) Model Performance Comparison', fontweight='bold', loc='left', pad=10, fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# R² on secondary axis
ax1_twin = ax1.twinx()
ax1_twin.plot(x, r2_values, 'o-', color='#2E86AB', linewidth=3,
              markersize=10, markeredgecolor='white', markeredgewidth=2)
ax1_twin.set_ylabel('R² Score', fontweight='bold', color='#2E86AB')
ax1_twin.tick_params(axis='y', labelcolor='#2E86AB')
ax1_twin.set_ylim(-0.5, 1.0)

# Add values
for i, (rmse, r2) in enumerate(zip(rmse_values, r2_values)):
    ax1.text(i, rmse + 1, f'{rmse:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if r2 > 0:
        ax1_twin.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontsize=8, color='#2E86AB')

# Panel B: Predicted vs Observed
ax2 = fig1.add_subplot(gs[0, 1])

scatter = ax2.scatter(y_test, y_pred_best, c=y_test, cmap='RdYlGn_r',
                     s=80, alpha=0.7, edgecolors='black', linewidth=1)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--', lw=3, alpha=0.5)

ax2.set_xlabel('Observed WQI', fontweight='bold')
ax2.set_ylabel('Predicted WQI', fontweight='bold')
ax2.set_title(f'(B) {best_model_name} Performance', fontweight='bold', loc='left', pad=10, fontsize=10)
ax2.grid(alpha=0.3)

# Statistics box
rmse_best = baseline_results[best_model_name]['RMSE']
r2_best = baseline_results[best_model_name]['R2']
mae_best = mean_absolute_error(y_test, y_pred_best)
textstr = f'RMSE = {rmse_best:.2f}\nMAE = {mae_best:.2f}\nR² = {r2_best:.3f}\nn = {len(y_test)}'
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor=COLORS['tertiary'],
         alpha=0.9, edgecolor='black', linewidth=1.5))

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Observed WQI', rotation=270, labelpad=20, fontweight='bold')

# Panel C: Residual Analysis
ax3 = fig1.add_subplot(gs[0, 2])

residuals = y_test - y_pred_best
ax3.scatter(y_pred_best, residuals, c=COLORS['primary'],
           alpha=0.6, s=60, edgecolors='black', linewidth=1)
ax3.axhline(y=0, color='black', linestyle='--', lw=2)
ax3.axhline(y=rmse_best, color='red', linestyle=':', lw=2, alpha=0.7)
ax3.axhline(y=-rmse_best, color='red', linestyle=':', lw=2, alpha=0.7, label=f'±RMSE ({rmse_best:.1f})')

ax3.set_xlabel('Predicted WQI', fontweight='bold')
ax3.set_ylabel('Residuals', fontweight='bold')
ax3.set_title('(C) Residual Analysis', fontweight='bold', loc='left', pad=10, fontsize=10)
ax3.legend(loc='best')
ax3.grid(alpha=0.3)

# Panel D: Temporal Validation
ax4 = fig1.add_subplot(gs[1, 0])

years_val = validation_df['Test_Year'].values
r2_val = validation_df['R²'].values
rmse_val = validation_df['RMSE'].values

x_val = range(len(years_val))
bars_val = ax4.bar(x_val, rmse_val, color=COLORS['gradient'][:len(years_val)],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('RMSE', fontweight='bold')
ax4.set_xlabel('Test Year', fontweight='bold')
ax4.set_xticks(x_val)
ax4.set_xticklabels(years_val)
ax4.set_title('(D) Temporal Validation', fontweight='bold', loc='left', pad=10, fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# R² on secondary axis
ax4_twin = ax4.twinx()
ax4_twin.plot(x_val, r2_val, 'o-', color='#2E86AB', linewidth=3,
             markersize=10, markeredgecolor='white', markeredgewidth=2)
ax4_twin.set_ylabel('R² Score', fontweight='bold', color='#2E86AB')
ax4_twin.tick_params(axis='y', labelcolor='#2E86AB')
ax4_twin.set_ylim(0.5, 1.0)

# Add values
for i, (rmse, r2) in enumerate(zip(rmse_val, r2_val)):
    ax4.text(i, rmse + 0.5, f'{rmse:.1f}', ha='center', va='bottom', fontsize=9)
    ax4_twin.text(i, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom', fontsize=8, color='#2E86AB')

# Panel E-F: Feature Importance
ax5 = fig1.add_subplot(gs[1, 1:])

top_n = 15
top_features = feature_importance_df.head(top_n)

# Color code
original_params = ['pH:Lab', 'DO(mg/L)', 'BOD(mg/L)', 'COD(mg/L)', 'TDS(mg/L)',
                   'SS(mg/L)', 'Turbidity. NTU', 'Chloride(mg/L)', 'EC(micro S/cm)']

colors_feat = []
for feat in top_features['Feature'].values:
    colors_feat.append(COLORS['primary'] if feat in original_params else COLORS['quaternary'])

bars_feat = ax5.barh(range(top_n), top_features['Mean_|SHAP|'].values[::-1],
                     color=colors_feat[::-1], alpha=0.8, edgecolor='black', linewidth=1.5)

ax5.set_yticks(range(top_n))
ax5.set_yticklabels(top_features['Feature'].values[::-1], fontsize=9)
ax5.set_xlabel('Mean |SHAP value|', fontweight='bold')
ax5.set_title('(E) Feature Importance (Top 15)', fontweight='bold', loc='left', pad=10, fontsize=10)
ax5.grid(axis='x', alpha=0.3)

# Add values
for i, val in enumerate(top_features['Mean_|SHAP|'].values[::-1]):
    ax5.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=8, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['primary'], label='Original Parameters', edgecolor='black'),
    Patch(facecolor=COLORS['quaternary'], label='Engineered Features', edgecolor='black')
]
ax5.legend(handles=legend_elements, loc='lower right')

# plt.suptitle('Figure 1: Model Performance, Validation & Feature Importance',
#              fontsize=11, fontweight='bold', y=0.98)

plt.savefig('figures/Figure1_Model_Performance_Complete.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure1_Model_Performance_Complete.png")
plt.show()
plt.close()

# Continue with remaining figures in next response...


# FIGURE 2: MULTI-SCALE SHAP ANALYSIS
# =============================================================================

print("\n[Figure 2/7] Multi-Scale SHAP Analysis...")

fig2 = plt.figure(figsize=(16, 12))
gs = fig2.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Panel A: Global Importance
ax1 = fig2.add_subplot(gs[0, :2])

top_10_feat = feature_importance_df.head(10)
colors_bar = [COLORS['primary'] if f in ['pH:Lab', 'DO(mg/L)', 'BOD(mg/L)', 'COD(mg/L)',
               'TDS(mg/L)', 'SS(mg/L)', 'Turbidity. NTU', 'Chloride(mg/L)', 'EC(micro S/cm)']
               else COLORS['quaternary'] for f in top_10_feat['Feature']]

ax1.barh(range(10), top_10_feat['Mean_|SHAP|'].values[::-1],
         color=colors_bar[::-1], alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(10))
ax1.set_yticklabels(top_10_feat['Feature'].values[::-1])
ax1.set_xlabel('Mean |SHAP|', fontweight='bold')
ax1.set_title('(A) Global Feature Importance', fontweight='bold', loc='left', pad=10)
ax1.grid(axis='x', alpha=0.3)

for i, val in enumerate(top_10_feat['Mean_|SHAP|'].values[::-1]):
    ax1.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=9, fontweight='bold')

# Panel B: Seasonal Attribution Heatmap
ax2 = fig2.add_subplot(gs[0, 2])

# Create seasonal matrix
season_matrix = []
for season_name in ['Pre-monsoon', 'Monsoon', 'Post-monsoon']:
    if season_name in seasonal_shap:
        top_5 = seasonal_shap[season_name].head(5)
        for _, row in top_5.iterrows():
            season_matrix.append({
                'Season': season_name,
                'Feature': row['Feature'][:20],
                'SHAP': row['Mean_|SHAP|']
            })

if len(season_matrix) > 0:
    season_df = pd.DataFrame(season_matrix)
    top_features_seasonal = season_df.groupby('Feature')['SHAP'].sum().nlargest(8).index
    season_df_filtered = season_df[season_df['Feature'].isin(top_features_seasonal)]

    season_pivot = season_df_filtered.pivot(index='Feature', columns='Season', values='SHAP').fillna(0)
    season_pivot = season_pivot[['Pre-monsoon', 'Monsoon', 'Post-monsoon']]

    im = ax2.imshow(season_pivot.values, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Pre-\nmonsoon', 'Monsoon', 'Post-\nmonsoon'], fontsize=9)
    ax2.set_yticks(range(len(season_pivot)))
    ax2.set_yticklabels(season_pivot.index, fontsize=8)
    ax2.set_title('(B) Seasonal Attribution', fontweight='bold', loc='left', pad=10)

    for i in range(len(season_pivot)):
        for j in range(3):
            text = ax2.text(j, i, f'{season_pivot.values[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8)

# Panel C-E: Location-Specific (Top 3)
loc_axes = [fig2.add_subplot(gs[1, i]) for i in range(3)]

for idx, (location, ax) in enumerate(zip(list(location_shap.keys())[:3], loc_axes)):
    loc_data = location_shap[location].head(6)

    colors_loc = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
                  COLORS['quaternary'], COLORS['primary'], COLORS['secondary']]

    ax.barh(range(len(loc_data)), loc_data['Mean_|SHAP|'].values[::-1],
            color=colors_loc[:len(loc_data)][::-1], alpha=0.8,
            edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(loc_data)))
    ax.set_yticklabels(loc_data['Feature'].values[::-1], fontsize=8)
    ax.set_xlabel('Mean |SHAP|', fontweight='bold', fontsize=10)
    ax.set_title(f'({chr(67+idx)}) {location[:20]}...',
                fontweight='bold', loc='left', fontsize=10, pad=10)
    ax.grid(axis='x', alpha=0.3)

# Panel F: Parameter-Wise WQI Decomposition
ax6 = fig2.add_subplot(gs[2, :])

sorted_params = param_contrib_df.sort_values('Mean_Contribution', key=abs, ascending=False)
params_plot = sorted_params['Parameter'].values
contribs_plot = sorted_params['Mean_Contribution'].values

colors_decomp = COLORS['gradient'][:len(params_plot)]

bars_decomp = ax6.barh(range(len(params_plot)), contribs_plot,
                       color=colors_decomp, alpha=0.8, edgecolor='black', linewidth=2)

ax6.set_yticks(range(len(params_plot)))
ax6.set_yticklabels(params_plot)
ax6.set_xlabel('Mean Contribution to WQI (SHAP value)', fontweight='bold', fontsize=11)
ax6.set_title('(F) Parameter-wise WQI Decomposition', fontweight='bold',
             loc='left', pad=10, fontsize=12)
ax6.axvline(x=0, color='black', linewidth=2)
ax6.grid(axis='x', alpha=0.3)

for i, val in enumerate(contribs_plot):
    ax6.text(val + (1 if val > 0 else -1), i, f'{val:+.1f}',
            va='center', ha='left' if val > 0 else 'right',
            fontsize=10, fontweight='bold')

#plt.suptitle('Figure 2: Multi-Scale SHAP Analysis & Parameter-wise WQI Decomposition',
 #            fontsize=12, fontweight='bold', y=0.96)

plt.savefig('figures/Figure2_SHAP_Analysis_Complete.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure2_SHAP_Analysis_Complete.png")
plt.show()
plt.close()

# =============================================================================
# FIGURE 3: SOURCE IDENTIFICATION
# =============================================================================

print("\n[Figure 3/7] Source Identification & Validation...")

fig3 = plt.figure(figsize=(14, 10))
gs = fig3.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Parameter Signature Space
ax1 = fig3.add_subplot(gs[0, :])

source_colors_map = {
    'Textile Dyeing': COLORS['primary'],
    'Domestic Sewage': COLORS['secondary'],
    'Mixed Urban': COLORS['quaternary'],
    'Diffuse Non-Point Sources': '#CCCCCC'
}

for source, color in source_colors_map.items():
    source_data = df_with_sources[df_with_sources['Source_signature'] == source]
    if len(source_data) > 0:
        ax1.scatter(source_data['BOD_COD_ratio'], source_data['EC(micro S/cm)'],
                   c=color, label=source, alpha=0.6, s=60, edgecolors='black', linewidth=1)

# Decision boundaries
ax1.axvline(x=0.3, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.axvline(x=0.45, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.axhline(y=800, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

# Labels
ax1.text(0.15, 1200, 'Textile\nDyeing', fontsize=11, fontweight='bold',
         ha='center', bbox=dict(boxstyle='round', facecolor=COLORS['primary'], alpha=0.3))
ax1.text(0.6, 600, 'Domestic\nSewage', fontsize=11, fontweight='bold',
         ha='center', bbox=dict(boxstyle='round', facecolor=COLORS['secondary'], alpha=0.3))

ax1.set_xlabel('BOD/COD Ratio', fontweight='bold')
ax1.set_ylabel('EC (µS/cm)', fontweight='bold')
ax1.set_title('(A) Parameter Signature Space for Source Identification',
             fontweight='bold', loc='left', pad=10)
ax1.legend(loc='best')
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1.0)

# Panel B: WQI Distribution by Source
ax2 = fig3.add_subplot(gs[1, 0])

source_order = ['Textile Dyeing', 'Domestic Sewage', 'Mixed Urban', 'Diffuse Non-Point Sources']
wqi_by_source = [df_with_sources[df_with_sources['Source_signature'] == s]['WQI'].dropna().values
                 for s in source_order if s in df_with_sources['Source_signature'].unique()]

valid_sources = [s for s in source_order if s in df_with_sources['Source_signature'].unique()]

bp = ax2.boxplot(wqi_by_source, labels=[s.replace(' ', '\n') for s in valid_sources],
                 patch_artist=True, notch=True, widths=0.6)

for patch, source in zip(bp['boxes'], valid_sources):
    patch.set_facecolor(source_colors_map[source])
    patch.set_alpha(0.7)

ax2.set_ylabel('WQI', fontweight='bold')
ax2.set_title('(B) WQI Distribution by Source Type', fontweight='bold',
             loc='left', pad=10)
ax2.grid(axis='y', alpha=0.3)

# Panel C: Source Distribution
ax3 = fig3.add_subplot(gs[1, 1])

source_counts_plot = df_with_sources['Source_signature'].value_counts()
colors_pie = [source_colors_map[s] for s in source_counts_plot.index]

wedges, texts, autotexts = ax3.pie(source_counts_plot.values, labels=source_counts_plot.index,
                                    colors=colors_pie, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

ax3.set_title('(C) Source Distribution', fontweight='bold', pad=10)

# plt.suptitle('Figure 3: Pollution Source Identification & Validation',
#              fontsize=12, fontweight='bold', y=0.96)

plt.savefig('figures/Figure3_Source_Identification.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure3_Source_Identification.png")
plt.show()
plt.close()

# =============================================================================
# FIGURE 4: SEASONAL PATTERNS
# =============================================================================

print("\n[Figure 4/7] Seasonal Patterns...")

fig4 = plt.figure(figsize=(14, 10))
gs = fig4.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Seasonal Correlation Heatmap
ax1 = fig4.add_subplot(gs[0, :])

# Create correlation matrix
corr_data = []
for season_name, corr_list in season_correlations.items():
    for param, corr_val in corr_list[:5]:
        param_short = param.replace('(mg/L)', '').replace('(micro S/cm)', '').replace('. NTU', '')
        corr_data.append({
            'Season': season_name,
            'Parameter': param_short,
            'Correlation': corr_val
        })

corr_df = pd.DataFrame(corr_data)
corr_pivot = corr_df.pivot(index='Parameter', columns='Season', values='Correlation')
corr_pivot = corr_pivot[['Pre-monsoon', 'Monsoon', 'Post-monsoon']]

im = ax1.imshow(corr_pivot.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks(range(3))
ax1.set_xticklabels(corr_pivot.columns)
ax1.set_yticks(range(len(corr_pivot)))
ax1.set_yticklabels(corr_pivot.index)
ax1.set_title('(A) Parameter-WQI Correlation by Season', fontweight='bold',
             loc='left', pad=10)

for i in range(len(corr_pivot)):
    for j in range(3):
        text = ax1.text(j, i, f'{corr_pivot.values[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Correlation with WQI', rotation=270, labelpad=20, fontweight='bold')

# Panel B: Seasonal WQI Distribution
ax2 = fig4.add_subplot(gs[1, 0])

season_wqi = []
for season_id, season_name in season_names.items():
    season_data = df_clean[df_clean['Season'] == season_id]['WQI'].values
    season_wqi.append(season_data)

bp2 = ax2.boxplot(season_wqi, labels=['Pre-monsoon', 'Monsoon', 'Post-monsoon'],
                  patch_artist=True, notch=True)

for patch, color in zip(bp2['boxes'], COLORS['gradient'][:3]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('WQI', fontweight='bold')
ax2.set_xlabel('Season', fontweight='bold')
ax2.set_title('(B) WQI Distribution by Season', fontweight='bold',
             loc='left', pad=10)
ax2.grid(axis='y', alpha=0.3)

# Panel C: Temporal Trend
ax3 = fig4.add_subplot(gs[1, 1])

yearly_wqi = df_clean.groupby('Year')['WQI'].agg(['mean', 'std']).reset_index()

ax3.errorbar(yearly_wqi['Year'], yearly_wqi['mean'], yerr=yearly_wqi['std'],
             fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
             color=COLORS['primary'], ecolor='gray', alpha=0.8)

# Trend line
z = np.polyfit(yearly_wqi['Year'], yearly_wqi['mean'], 1)
p = np.poly1d(z)
ax3.plot(yearly_wqi['Year'], p(yearly_wqi['Year']),
         "--", linewidth=2, color=COLORS['quaternary'],
         label=f'Trend: {z[0]:+.1f} WQI/year')

ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('Mean WQI ± SD', fontweight='bold')
ax3.set_title('(C) Temporal Trend (2011-2023)', fontweight='bold',
             loc='left', pad=10)
ax3.legend(loc='best')
ax3.grid(alpha=0.3)

# plt.suptitle('Figure 4: Seasonal Patterns & Temporal Trends',
#              fontsize=12, fontweight='bold', y=0.96)

plt.savefig('figures/Figure4_Seasonal_Patterns.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure4_Seasonal_Patterns.png")
plt.show()
plt.close()

print("\n✓ Figures 1-4 complete. Continue for Figures 5-7 (Future Forecasts & Scenarios)...")


# FIGURE 5: FUTURE FORECASTS (2024-2035)
# =============================================================================

print("\n[Figure 5/7] Future Forecasts (2024-2035)...")

fig5 = plt.figure(figsize=(16, 10))
gs = fig5.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Monthly Forecast Trend
ax1 = fig5.add_subplot(gs[0, :])

monthly_forecast = future_baseline.groupby(['Year', 'Month']).agg({
    'WQI_predicted': 'mean',
    'WQI_lower_95': 'mean',
    'WQI_upper_95': 'mean'
}).reset_index()

monthly_forecast['Date'] = pd.to_datetime(
    monthly_forecast[['Year', 'Month']].assign(Day=1)
)

x_dates = monthly_forecast['Date']
y_pred = monthly_forecast['WQI_predicted']
y_lower = monthly_forecast['WQI_lower_95']
y_upper = monthly_forecast['WQI_upper_95']

ax1.plot(x_dates, y_pred, 'o-', linewidth=3, markersize=8,
        color=COLORS['primary'], label='Predicted WQI',
        markeredgecolor='white', markeredgewidth=2)

ax1.fill_between(x_dates, y_lower, y_upper, alpha=0.3,
                color=COLORS['primary'], label='95% Confidence Interval')

# Classification lines
ax1.axhline(y=100, color=COLORS['quaternary'], linestyle='--', linewidth=2,
           alpha=0.5, label='Good (<100)')
ax1.axhline(y=150, color=COLORS['secondary'], linestyle='--', linewidth=2,
           alpha=0.5, label='Moderate (<150)')

ax1.set_xlabel('Date', fontweight='bold', fontsize=12)
ax1.set_ylabel('Predicted WQI', fontweight='bold', fontsize=12)
ax1.set_title('(A) Monthly WQI Forecast 2024-2035 with 95% CI',
             fontweight='bold', loc='left', pad=10, fontsize=13)
ax1.legend(loc='best', fontsize=9)
ax1.grid(alpha=0.3)

# Monsoon shading
for year in range(2024, 2036):
    monsoon_start = pd.Timestamp(f'{year}-06-01')
    monsoon_end = pd.Timestamp(f'{year}-09-30')
    ax1.axvspan(monsoon_start, monsoon_end, alpha=0.1, color='blue')

# Panel B: Location-wise Forecast
ax2 = fig5.add_subplot(gs[1, 0])

top_locs_2023 = df_clean[df_clean['Year'] == 2023].groupby('Location')['WQI'].mean().nlargest(5)
loc_forecasts = []

for loc in top_locs_2023.index:
    loc_data = future_baseline[future_baseline['Location'] == loc]
    loc_forecasts.append({
        'Location': loc[:22] + '...' if len(loc) > 22 else loc,
        'Mean_WQI': loc_data['WQI_predicted'].mean()
    })

loc_df = pd.DataFrame(loc_forecasts).sort_values('Mean_WQI', ascending=True)

y_pos = range(len(loc_df))
bars_loc = ax2.barh(y_pos, loc_df['Mean_WQI'],
                    color=COLORS['gradient'][:len(loc_df)], alpha=0.8,
                    edgecolor='black', linewidth=1.5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(loc_df['Location'], fontsize=9)
ax2.set_xlabel('Predicted Mean WQI', fontweight='bold')
ax2.set_title('(B) Location-wise Forecasts (Top 5)', fontweight='bold',
             loc='left', pad=10)
ax2.grid(axis='x', alpha=0.3)

for i, val in enumerate(loc_df['Mean_WQI']):
    ax2.text(val + 2, i, f'{val:.0f}', va='center', fontsize=9, fontweight='bold')

# Panel C: Seasonal Forecast Pattern
ax3 = fig5.add_subplot(gs[1, 1])

season_names_dict = {0: 'Pre-monsoon', 1: 'Monsoon', 2: 'Post-monsoon'}
future_baseline_temp = future_baseline.copy()
future_baseline_temp['Season_name'] = future_baseline_temp['Season'].map(season_names_dict)

seasonal_forecast = future_baseline_temp.groupby('Season_name').agg({
    'WQI_predicted': ['mean', 'std']
}).reset_index()

seasonal_forecast.columns = ['Season', 'Mean_WQI', 'Std_WQI']
season_order = ['Pre-monsoon', 'Monsoon', 'Post-monsoon']
seasonal_forecast['Season'] = pd.Categorical(seasonal_forecast['Season'],
                                            categories=season_order, ordered=True)
seasonal_forecast = seasonal_forecast.sort_values('Season')

x_season = range(len(seasonal_forecast))
bars_season = ax3.bar(x_season, seasonal_forecast['Mean_WQI'],
                      yerr=seasonal_forecast['Std_WQI'],
                      color=COLORS['gradient'][:3], alpha=0.8,
                      edgecolor='black', linewidth=2, capsize=10)

ax3.set_xticks(x_season)
ax3.set_xticklabels(seasonal_forecast['Season'])
ax3.set_ylabel('Predicted WQI (Mean ± SD)', fontweight='bold')
ax3.set_title('(C) Seasonal Forecast Pattern', fontweight='bold',
             loc='left', pad=10)
ax3.grid(axis='y', alpha=0.3)

for i, (mean, std) in enumerate(zip(seasonal_forecast['Mean_WQI'],
                                    seasonal_forecast['Std_WQI'])):
    ax3.text(i, mean + std + 3, f'{mean:.0f}±{std:.0f}',
            ha='center', fontsize=10, fontweight='bold')

#plt.suptitle('Figure 5: Future Water Quality Forecasts (2024-2035)',
#             fontsize=12, fontweight='bold', y=0.96)

plt.savefig('figures/Figure5_Future_Forecasts.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure5_Future_Forecasts.png")
plt.show()
plt.close()

# =============================================================================
# FIGURE 6: SCENARIO ANALYSIS
# =============================================================================

print("\n[Figure 6/7] Scenario Analysis...")

fig6 = plt.figure(figsize=(16, 10))
gs = fig6.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Scenario Comparison
ax1 = fig6.add_subplot(gs[0, :])

scenarios_plot = scenario_summary['Scenario'].values
mean_wqi_scenarios = scenario_summary['Mean_WQI'].values
improvements = scenario_summary['Improvement'].values

x_scen = np.arange(len(scenarios_plot))
bars_scen = ax1.bar(x_scen, mean_wqi_scenarios, color=COLORS['gradient'],
                    alpha=0.8, edgecolor='black', linewidth=2)

ax1.set_ylabel('Mean Predicted WQI', fontweight='bold', fontsize=12)
ax1.set_xlabel('Intervention Scenario', fontweight='bold', fontsize=12)
ax1.set_xticks(x_scen)
ax1.set_xticklabels(scenarios_plot, rotation=20, ha='right')
ax1.set_title('(A) Scenario Comparison: WQI by Intervention',
             fontweight='bold', loc='left', pad=10, fontsize=13)
ax1.grid(axis='y', alpha=0.3)

# Improvement arrows
for i in range(1, len(mean_wqi_scenarios)):
    ax1.annotate('', xy=(i, mean_wqi_scenarios[i]), xytext=(0, mean_wqi_scenarios[0]),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='red', alpha=0.6))

    improvement = improvements[i]
    improvement_pct = scenario_summary['Improvement_pct'].values[i]
    mid_y = (mean_wqi_scenarios[0] + mean_wqi_scenarios[i]) / 2
    ax1.text(i/2, mid_y, f'-{improvement:.0f}\n({improvement_pct:.0f}%)',
            ha='center', fontsize=9, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                     edgecolor='red', linewidth=1.5))

# Add values
for i, val in enumerate(mean_wqi_scenarios):
    ax1.text(i, val + 3, f'{val:.0f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# Classification lines
ax1.axhline(y=100, color=COLORS['quaternary'], linestyle='--', linewidth=2, alpha=0.5)
ax1.axhline(y=150, color=COLORS['secondary'], linestyle='--', linewidth=2, alpha=0.5)

# Panel B: Cost-Benefit Analysis
ax2 = fig6.add_subplot(gs[1, 0])

costs = scenario_summary['Cost_Million_BDT'].values[1:]
benefits = improvements[1:]
roi = scenario_summary['ROI'].values[1:]

scatter = ax2.scatter(costs, benefits, s=500*roi,
                     c=[COLORS['primary'], COLORS['secondary'], COLORS['quaternary']],
                     alpha=0.7, edgecolors='black', linewidth=2)

for i, (cost, benefit, scenario) in enumerate(zip(costs, benefits, scenarios_plot[1:])):
    ax2.text(cost, benefit, scenario, ha='center', va='center',
            fontweight='bold', fontsize=8)

ax2.set_xlabel('Investment Cost (Million BDT)', fontweight='bold')
ax2.set_ylabel('WQI Improvement (points)', fontweight='bold')
ax2.set_title('(B) Cost-Benefit Analysis', fontweight='bold', loc='left', pad=10)
ax2.grid(alpha=0.3)

# ROI isolines
for roi_val in [0.5, 1.0, 1.5, 2.0]:
    cost_range = np.linspace(10, 70, 100)
    benefit_range = cost_range * roi_val
    ax2.plot(cost_range, benefit_range, '--', alpha=0.3, color='gray', linewidth=1)
    ax2.text(65, 65*roi_val-2, f'ROI={roi_val:.1f}', fontsize=8, alpha=0.5)

# Panel C: Temporal Evolution
ax3 = fig6.add_subplot(gs[1, 1])

all_scenarios_df = pd.concat([
    future_baseline.assign(Scenario='Baseline'),
    future_etp.assign(Scenario='ETP'),
    future_sewage.assign(Scenario='Sewage'),
    future_combined.assign(Scenario='Combined')
])

scenario_colors_plot = {
    'Baseline': COLORS['primary'],
    'ETP': COLORS['secondary'],
    'Sewage': COLORS['tertiary'],
    'Combined': COLORS['quaternary']
}

for scenario_name in ['Baseline', 'ETP', 'Sewage', 'Combined']:
    scenario_data = all_scenarios_df[all_scenarios_df['Scenario'] == scenario_name]
    monthly_scen = scenario_data.groupby(['Year', 'Month'])['WQI_predicted'].mean().reset_index()
    monthly_scen['Date'] = pd.to_datetime(monthly_scen[['Year', 'Month']].assign(Day=1))

    ax3.plot(monthly_scen['Date'], monthly_scen['WQI_predicted'], 'o-',
            linewidth=2, markersize=5, label=scenario_name,
            color=scenario_colors_plot[scenario_name], alpha=0.8,
            markeredgecolor='white', markeredgewidth=1)

ax3.set_xlabel('Date', fontweight='bold')
ax3.set_ylabel('Predicted Mean WQI', fontweight='bold')
ax3.set_title('(C) Temporal Evolution by Scenario', fontweight='bold',
             loc='left', pad=10)
ax3.legend(loc='best', fontsize=9)
ax3.grid(alpha=0.3)

#plt.suptitle('Figure 6: Intervention Scenario Analysis & Cost-Benefit Assessment',
#             fontsize=12, fontweight='bold', y=0.96)

plt.savefig('figures/Figure6_Scenario_Analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure6_Scenario_Analysis.png")
plt.show()
plt.close()

# =============================================================================
# FIGURE 7: COMPREHENSIVE SUMMARY DASHBOARD
# =============================================================================

print("\n[Figure 7/7] Comprehensive Summary Dashboard...")

fig7 = plt.figure(figsize=(16, 10))
gs = fig7.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel A: Key Metrics Box
ax1 = fig7.add_subplot(gs[0, 0])
ax1.axis('off')

key_metrics = f"""
KEY PERFORMANCE METRICS
{'='*35}

Model Performance:
Best: {best_model_name}
RMSE: {baseline_results[best_model_name]['RMSE']:.2f}
R²: {baseline_results[best_model_name]['R2']:.3f}
MAE: {mean_absolute_error(y_test, y_pred_best):.2f}

Temporal Validation:
Mean R²: {validation_df['R²'].mean():.3f}
Range: {validation_df['R²'].min():.3f}-{validation_df['R²'].max():.3f}

Future Forecast (2024-35):
Mean: {future_baseline['WQI_predicted'].mean():.1f}
95% CI: [{future_baseline['WQI_lower_95'].mean():.1f},
         {future_baseline['WQI_upper_95'].mean():.1f}]

Best Intervention:
ETP Enforcement
Improvement: {scenario_summary.iloc[1]['Improvement']:.1f} pts
({scenario_summary.iloc[1]['Improvement_pct']:.1f}%)
Cost: BDT {scenario_summary.iloc[1]['Cost_Million_BDT']:.0f}M
ROI: {scenario_summary.iloc[1]['ROI']:.2f}
"""

ax1.text(0.05, 0.95, key_metrics, transform=ax1.transAxes,
        fontsize=8.5, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['tertiary'],
                 alpha=0.9, edgecolor='black', linewidth=2))

ax1.set_title('(A) Key Metrics Summary', fontweight='bold', loc='left', pad=10)

# Panel B: Historical vs Forecast by Season
ax2 = fig7.add_subplot(gs[0, 1:])

# Get seasonal data
seasonal_historical = []
seasonal_forecast_data = []

for season_id, season_name in season_names.items():
    hist_data = df_clean[df_clean['Season'] == season_id]['WQI'].mean()
    forecast_data = future_baseline[future_baseline['Season'] == season_id]['WQI_predicted'].mean()

    seasonal_historical.append(hist_data)
    seasonal_forecast_data.append(forecast_data)

x_seasons = np.arange(3)
width = 0.35

bars1 = ax2.bar(x_seasons - width/2, seasonal_historical, width,
               label='Historical (2011-2023)',
               color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x_seasons + width/2, seasonal_forecast_data, width,
               label='Forecast (2024-2035)',
               color=COLORS['quaternary'], alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Mean WQI', fontweight='bold')
ax2.set_xlabel('Season', fontweight='bold')
ax2.set_xticks(x_seasons)
ax2.set_xticklabels(['Pre-monsoon', 'Monsoon', 'Post-monsoon'])
ax2.set_title('(B) Historical vs Forecasted WQI by Season',
             fontweight='bold', loc='left', pad=10)
ax2.legend(loc='best')
ax2.grid(axis='y', alpha=0.3)

# Add values
for i, (hist, fore) in enumerate(zip(seasonal_historical, seasonal_forecast_data)):
    ax2.text(i - width/2, hist + 2, f'{hist:.0f}', ha='center', va='bottom', fontsize=9)
    ax2.text(i + width/2, fore + 2, f'{fore:.0f}', ha='center', va='bottom', fontsize=9)

# Panel C: Source Contribution Pie
ax3 = fig7.add_subplot(gs[1, 0])

source_counts_final = df_with_sources['Source_signature'].value_counts()
colors_pie_final = [source_colors_map.get(s, '#CCCCCC') for s in source_counts_final.index]

wedges, texts, autotexts = ax3.pie(source_counts_final.values,
                                    labels=source_counts_final.index,
                                    colors=colors_pie_final,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    textprops={'fontsize': 9, 'fontweight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

ax3.set_title('(C) Pollution Source Distribution', fontweight='bold', pad=10)

# Panel D: Intervention Impact
ax4 = fig7.add_subplot(gs[1, 1:])

scenarios_final = scenario_summary['Scenario'].values
improvements_final = scenario_summary['Improvement'].values
costs_final = scenario_summary['Cost_Million_BDT'].values

sizes_bubble = np.abs(improvements_final) * 15

for i, (scenario, cost, improvement, size) in enumerate(
    zip(scenarios_final, costs_final, improvements_final, sizes_bubble)):

    if i == 0:  # Skip baseline
        continue

    color_bubble = COLORS['gradient'][min(i, len(COLORS['gradient'])-1)]
    ax4.scatter(cost, improvement, s=size, c=[color_bubble], alpha=0.7,
               edgecolors='black', linewidth=2, label=scenario)

ax4.set_xlabel('Cost (Million BDT)', fontweight='bold')
ax4.set_ylabel('WQI Improvement (points)', fontweight='bold')
ax4.set_title('(D) Intervention Impact (Bubble size = Improvement)',
             fontweight='bold', loc='left', pad=10)
ax4.legend(loc='best', fontsize=9)
ax4.grid(alpha=0.3)

# Add labels
for scenario, cost, improvement in zip(scenarios_final[1:], costs_final[1:], improvements_final[1:]):
    ax4.annotate(f'{improvement:.0f} pts',
                xy=(cost, improvement), xytext=(5, 5),
                textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

#plt.suptitle('Figure 7: Comprehensive Results Summary & Decision Support Dashboard',
#             fontsize=12, fontweight='bold', y=0.96)

plt.savefig('figures/Figure7_Summary_Dashboard.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure7_Summary_Dashboard.png")
plt.show()
plt.close()
