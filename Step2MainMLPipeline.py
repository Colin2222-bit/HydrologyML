!pip install -q optuna lightgbm cartopy

import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import os
import sys
import shutil

# Define base directory depending on environment
if os.path.exists('/content'):
    print("Detected Google Colab environment.")
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = Path('/content/drive/MyDrive/SWE_Project')
else:
    print("Detected NERSC/Jupyter environment.")
    PROJECT_DIR = Path('/global/cfs/cdirs/m4062/colin_brown/')

# Define and create high-resolution plots output directory
PLOTS_DIR = PROJECT_DIR / "poster_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Project directory set to: {PROJECT_DIR}")
print(f"High-resolution figures will be saved to: {PLOTS_DIR}")

# --- THE METRICS ENGINE ---
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)

    # --- SAFE CORRELATION GUARD ---
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = 0.0
    else:
        r_matrix = np.corrcoef(y_true, y_pred)
        r = r_matrix[0, 1] if not np.isnan(r_matrix[0, 1]) else 0.0

    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-8)
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-8)

    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (alpha - 1)**2)

    return r2, rmse, mae, mbe, kge

# --- DIAGNOSTIC PLOTTER ---
def plot_diagnostic_suite(val_df, model_name, scenario, exp_name, target, model=None, feature_names=None):
    pred_col = f'{model_name}_Predicted_{scenario}'

    # 1. Standardize and Clean Up Metadata labels for Titles
    target_clean = "SWE" if target == "WTEQ" else "Snow Depth" if target == "SNWD" else target
    scen_clean = "Daily Meteorology" if "Meteo_Only" in scenario else "Cumulative Meteorology" if "Meteo_Eng" in scenario else scenario
    
    # Perfect alignment with abstract terminology
    if "Temporal" in exp_name and "Regional" not in exp_name:
        exp_clean = "Temporal"
    elif "Region" in exp_name or "Regional" in exp_name:
        reg_suffix = exp_name.split("_")[-1]
        exp_clean = f"Regional-Temporal ({reg_suffix})"
    elif "Pure_Spatial" in exp_name:
        exp_clean = "Pure Spatial"
    elif "Spatiotemporal" in exp_name:
        exp_clean = "Spatiotemporal"
    elif "Zero" in exp_name:
        exp_clean = "Sierra Zero-Shot"
    else:
        exp_clean = exp_name.replace("_", " ").title()

    # Generate safe filesystem name templates for saving
    safe_exp = exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    safe_scen = scenario.replace(" ", "_")
    file_prefix = f"{safe_exp}_{target}_{model_name}_{safe_scen}"

    # ==================================================
    # 1. Global Aggregate Time Series
   
    agg = val_df.groupby('DOWY')[[target, pred_col]].mean()
    plt.figure(figsize=(8, 3))
    line1, = plt.plot(agg.index, agg[target], label='Observed', color='#1f77b4', linewidth=2)
    line2, = plt.plot(agg.index, agg[pred_col], label='Predicted', color='#ff7f0e', linestyle='--')
    
    # Title & Axis Labels
    plt.title(f"Daily Mean {target_clean}\n"
              f"{exp_clean} • LightGBM ({scen_clean})", 
              fontsize=11, fontweight='bold', pad=10)
    plt.xlabel("Day of Water Year (DOWY)", fontsize=10)
    plt.ylabel(f"Mean {target_clean} (m)", fontsize=10)
    plt.legend(handles=[line1, line2], labels=[f'Observed {target}', f'Predicted {target}'])
    plt.tight_layout()
    
    # Save Figure
    plt.savefig(PLOTS_DIR / f"{file_prefix}_timeseries.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Memory safety release

   
    # 2. 1:1 Scatter Plot

    plt.figure(figsize=(6, 5))

    r2_val, _, _, _, kge_val = calc_metrics(val_df[target], val_df[pred_col])
    hb = plt.hexbin(val_df[target], val_df[pred_col], gridsize=50, cmap='Blues', bins='log', mincnt=1)

    max_val = max(val_df[target].max(), val_df[pred_col].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='1:1 Line')

    cb = plt.colorbar(hb, label='log10(count)')

    text_str = f"$R^2$: {r2_val:.2f}\nKGE: {kge_val:.2f}"
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Title & Axis Labels
    plt.title(f"Predicted vs. Observed {target_clean}\n"
              f"{exp_clean} • LightGBM ({scen_clean})", 
              fontsize=11, fontweight='bold', pad=10)
    plt.xlabel(f"Observed {target_clean} (m)", fontsize=10)
    plt.ylabel(f"Predicted {target_clean} (m)", fontsize=10)
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save Figure
    plt.savefig(PLOTS_DIR / f"{file_prefix}_scatter.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Memory safety release


    # 3. Spatial Geo Performance Map 
    site_perf = val_df.groupby(['station', 'latitude', 'longitude'], group_keys=False).apply(
        lambda x: pd.Series([calc_metrics(x[target], x[pred_col])[4]], index=['KGE']),
        include_groups=False
    ).reset_index()

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    sc = ax.scatter(site_perf['longitude'], site_perf['latitude'], c=site_perf['KGE'],
                    cmap='RdYlGn', transform=ccrs.PlateCarree(), vmin=-0.5, vmax=1, s=20, edgecolors='k', zorder=3)
    plt.colorbar(sc, label='Kling-Gupta Efficiency (KGE)')
    
    # Decoupled, Publication-Grade Title
    plt.title(f"Spatial Model Performance (KGE): {target_clean}\n"
              f"{exp_clean} • LightGBM ({scen_clean})", 
              fontsize=11, fontweight='bold', pad=10)
    
    # Clean publication-style coordinates and ticks
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    
    ax.set_xlabel("Longitude (°W)", fontsize=10, labelpad=20)
    ax.set_ylabel("Latitude (°N)", fontsize=10, labelpad=20)
    
    # Save Figure
    plt.savefig(PLOTS_DIR / f"{file_prefix}_spatial_map.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Memory safety release

    # ==================================================
    # 4. Relative Feature Importance (Only for Tree Models)
    # ==================================================
    if model_name == 'LightGBM' and model is not None and feature_names is not None:
        name_mapping = {
            'prcp_cumsum': 'Cumulative Precipitation',
            'MDD_cumsum': 'Melting Degree Days (Cumulative)',
            'FDD_cumsum': 'Freezing Degree Days (Cumulative)',
            'elevation_m': 'Elevation',
            'srad_cumsum': 'Cumulative Solar Radiation',
            'vp_cumsum': 'Cumulative Vapor Pressure',
            'DOWY': 'Day of Water Year (DOWY)',
            'longitude': 'Longitude',
            'latitude': 'Latitude',
            'prcp': 'Daily Precipitation',
            'tmax': 'Daily Max Temperature',
            'tmin': 'Daily Min Temperature',
            'vp': 'Vapor Pressure',
            'srad': 'Solar Radiation',
            'dayl': 'Daylength'
        }

        raw_importances = model.feature_importances_
        total_importance = np.sum(raw_importances) + 1e-8
        relative_importances = (raw_importances / total_importance) * 100

        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': relative_importances})
        df_imp['Clean_Feature'] = df_imp['Feature'].map(name_mapping).fillna(df_imp['Feature'])
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(20)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_imp, x='Importance', y='Clean_Feature', hue='Clean_Feature', palette='viridis', legend=False)

        # Decoupled, Publication-Grade Title & Axis Labels
        plt.title(f"Relative Feature Importance for {target_clean} Prediction\n"
                  f"{exp_clean} • LightGBM ({scen_clean})", 
                  fontsize=11, fontweight='bold', pad=15)
        plt.xlabel("Relative Feature Importance (%)", fontsize=11)
        plt.ylabel("Predictor Features", fontsize=11)
        plt.xlim(0, max(df_imp['Importance']) * 1.1)
        plt.tight_layout()
        
        # Save Figure
        plt.savefig(PLOTS_DIR / f"{file_prefix}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()  # Memory safety release

# --- CONFIG ---
cat_vars = ['mountainRange', 'state']
targets = ['WTEQ', 'SNWD']  # Set to both targets cleanly

feature_scenarios = {
    "1_Meteo_Only": ['tmax', 'tmin', 'DOWY', 'prcp', 'vp', 'srad', 'dayl', 'elevation_m', 'longitude', 'latitude'],
    "2_Meteo_Eng": ['DOWY', 'prcp_cumsum', 'elevation_m', 'longitude', 'latitude', 'vp_cumsum', 'srad_cumsum', 'MDD_cumsum', 'FDD_cumsum']
}
models_to_test = {
    'LinReg': LinearRegression(),

    #LightGBM parameters were tuned for the baseline temporal test using optuna. I wanted to keep all parameters the same for all tests. 
    'LightGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                              reg_lambda=2.0, max_bin=63, subsample=0.8, subsample_freq=5,
                              colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
}

# --- DATA LOADING & CLEANING ---
print("Loading Master Data...")
master_df = pd.read_parquet(PROJECT_DIR / "master_dataset_engineered.parquet")
master_df.rename(columns=lambda x: x.split(' ')[0], inplace=True)

# 1. SMART NETWORK-AWARE DEDUPLICATION
print("Scanning and resolving duplicate SNOTEL/CCSS stations...")
master_df['lat_round'] = master_df['latitude'].round(2)
master_df['lon_round'] = master_df['longitude'].round(2)
stations_meta = master_df[['station', 'network', 'lat_round', 'lon_round']].drop_duplicates(subset=['station'])

ccss_to_drop = []
for _, group in stations_meta.groupby(['lat_round', 'lon_round']):
    networks = group['network'].unique()
    if 'SNOTEL' in networks and 'CCSS' in networks:
        ccss_to_drop.extend(group[group['network'] == 'CCSS']['station'].unique().tolist())

master_df = master_df[~master_df['station'].isin(ccss_to_drop)]
master_df = master_df.drop(columns=['lat_round', 'lon_round'])


# 2. DYNAMICALLY DROP ROWS MISSING OUR DESIRED TARGETS
master_df = master_df.dropna(subset=targets)

# 3. GEOGRAPHIC MAPPING (WITH SOUTH DAKOTA TO ROCKIES)
region_map = {
    'Alaska': 'Alaska', 'Washington': 'PNW', 'Oregon': 'PNW',
    'California': 'Sierra', 'Nevada': 'Sierra', 'Montana': 'Rockies',
    'Wyoming': 'Rockies', 'Colorado': 'Rockies', 'Idaho': 'Rockies',
    'Utah': 'Rockies', 'Arizona': 'Southwest', 'New Mexico': 'Southwest',
    'South Dakota': 'Rockies'  # South Dakota mapped to Rockies to keep Black Hills stations active
}
master_df['macro_region'] = master_df['state'].map(region_map).fillna('Other')
master_df = master_df[master_df['macro_region'] != 'Other']

# 4. SAFETY FEATURES DROPNA (FOR LINEAR REGRESSION ROBUSTNESS)
all_features = list(set(feature_scenarios["1_Meteo_Only"] + feature_scenarios["2_Meteo_Eng"]))
master_df = master_df.dropna(subset=all_features)

# 5. CATEGORICAL TYPE CASTING
for col in cat_vars:
    master_df[col] = master_df[col].astype('category')



unique_stations_df = master_df[['station', 'network']].drop_duplicates(subset=['station'])
snotel_count = (unique_stations_df['network'] == 'SNOTEL').sum()
ccss_count = (unique_stations_df['network'] == 'CCSS').sum()
total_stations = len(unique_stations_df)


# Experiment
def run_experiment(train_df, val_df, exp_name):
    val_df = val_df.copy()
    all_res = []

    for target in targets:
        print(f"\n Target Variable: {target}")

        # Baseline Climatology
        print("Scenario: 0_Baseline (Climatology)")
        clim_map = train_df.groupby('DOWY')[target].mean()

        preds_train_bl = train_df['DOWY'].map(clim_map).fillna(train_df[target].mean())
        r2_tr_bl = r2_score(train_df[target], preds_train_bl)

        preds_val_bl = val_df['DOWY'].map(clim_map).fillna(train_df[target].mean())
        r2_val_bl, rmse_val_bl, mae_val_bl, mbe_val_bl, kge_val_bl = calc_metrics(val_df[target], preds_val_bl)

        all_res.append({
            "Target": target, "Scenario": "0_Baseline", "Algorithm": "Climatology",
            "R2": r2_val_bl, "RMSE": rmse_val_bl, "KGE": kge_val_bl, "MAE": mae_val_bl,
            "MBE": mbe_val_bl,
        })
        print(f" Baseline Done. | Train R2: {r2_tr_bl:.3f} | Val R2: {r2_val_bl:.3f} | Diff: {r2_tr_bl - r2_val_bl:.3f}")

        # Machine Learning Scenarios
        for scenario_name, num_vars in feature_scenarios.items():
            print(f"  🎬 Running Scenario: {scenario_name}")
            feats = [col for col in cat_vars + num_vars if col in train_df.columns]

            pre = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), [c for c in cat_vars if c in feats]),
                ('num', StandardScaler(), [n for n in num_vars if n in feats])
            ], remainder='drop', verbose_feature_names_out=False)

            X_train, X_val = train_df[feats].copy(), val_df[feats].copy()
            y_train, y_val = train_df[target], val_df[target]

            X_train_p = pre.fit_transform(X_train)
            X_val_p = pre.transform(X_val)

            for model_name, model_algo in models_to_test.items():
                print(f"  Training Model: {model_name}...")
                fresh_model = clone(model_algo)

                if model_name == 'LightGBM':
                    fresh_model.fit(X_train, y_train)
                    preds_train = fresh_model.predict(X_train)
                    preds_val = np.maximum(fresh_model.predict(X_val), 0)
                else:
                    fresh_model.fit(X_train_p, y_train)
                    preds_train = fresh_model.predict(X_train_p)
                    preds_val = np.maximum(fresh_model.predict(X_val_p), 0)

                r2_tr = r2_score(y_train, preds_train)
                r2_val, rmse_val, mae, mbe, kge_val = calc_metrics(y_val, preds_val)

                val_df[f'{model_name}_Predicted_{scenario_name}'] = preds_val

                all_res.append({
                    "Target": target, "Scenario": scenario_name, "Algorithm": model_name,
                    "R2": r2_val, "RMSE": rmse_val, "KGE": kge_val, "MAE": mae,
                    "MBE": mbe
                })
                print(f" {model_name} Done. | Train R2: {r2_tr:.3f} | Val R2: {r2_val:.3f} | Diff: {r2_tr - r2_val:.3f}")

                if model_name == 'LightGBM':
                    plot_diagnostic_suite(val_df, model_name, scenario_name, exp_name, target, model=fresh_model, feature_names=feats)

                del fresh_model, preds_train, preds_val; gc.collect()

    return pd.DataFrame(all_res)

# --- EXECUTION ---
np.random.seed(42)
unique_stations = master_df['station'].unique()
new_stations = np.random.choice(unique_stations, size=int(len(unique_stations) * 0.20), replace=False)


print(" EXPERIMENT 1: TEMPORAL")

train1 = master_df[(master_df["WY"] >= 2001) & (master_df["WY"] <= 2019)].copy()
test1 = master_df[(master_df["WY"] >= 2020) & (master_df["WY"] <= 2025)].copy()
res1 = run_experiment(train1, test1, "Temporal")
display(res1)
del train1, test1; gc.collect()
print('All done with EXP1')

print(" EXPERIMENT 2: TEMPORAL-REGIONAL")
regions = ['PNW', 'Sierra', 'Rockies', 'Southwest', 'Alaska']
for reg in regions:
    print(f"\n📍 Sub-Region Pipeline: {reg}")
    reg_df = master_df[master_df['macro_region'] == reg]
    t = reg_df[(reg_df["WY"] >= 2001) & (reg_df["WY"] <= 2019)].copy()
    v = reg_df[(reg_df["WY"] >= 2020) & (reg_df["WY"] <= 2025)].copy()
    res_reg = run_experiment(t, v, f"Region_{reg}")
    display(res_reg)
    del t, v, reg_df; gc.collect()

print('All done with EXP2')

print(" EXPERIMENT 3: SPATIAL")
train_pure = master_df[(~master_df['station'].isin(new_stations)) & (master_df["WY"] >= 2001) & (master_df["WY"] <= 2019)].copy()
test_pure = master_df[(master_df['station'].isin(new_stations)) & (master_df["WY"] >= 2001) & (master_df["WY"] <= 2019)].copy()
res3 = run_experiment(train_pure, test_pure, "Pure_Spatial")
display(res3)
del train_pure, test_pure; gc.collect()
print('All done with EXP3')

print(" EXPERIMENT 4: SPATIoTEMPORL")
train_spatio = master_df[(~master_df['station'].isin(new_stations)) & (master_df["WY"] >= 2001) & (master_df["WY"] <= 2019)].copy()
test_spatio = master_df[(master_df['station'].isin(new_stations)) & (master_df["WY"] >= 2020) & (master_df["WY"] <= 2025)].copy()
res4 = run_experiment(train_spatio, test_spatio, "Spatiotemporal")
display(res4)
del train_spatio, test_spatio; gc.collect()
print('All done with EXP4')

print(" EXPERIMENT 5: Sierra Zer-Shot (Spatiotemporal")
train2 = master_df[(master_df['macro_region'] != 'Sierra') & (master_df["WY"] >= 2001) & (master_df["WY"] <= 2019)].copy()
test2 = master_df[(master_df['macro_region'] == 'Sierra') & (master_df["WY"] >= 2020) & (master_df["WY"] <= 2025)].copy()
res2 = run_experiment(train2, test2, "Sierra Zero-Shot")
display(res2)
del train2, test2; gc.collect()

print('All done with final testing! 🚀')

#Downloads

zip_output_path = PROJECT_DIR / "final_poster_plots"
shutil.make_archive(zip_output_path, 'zip', PLOTS_DIR)
print(f" Compression Complete! Download your files here: {zip_output_path}.zip")
print("="*50 + "\n")
