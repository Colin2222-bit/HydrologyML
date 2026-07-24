#Use for ML SNOTEL training on its own feature data. Be warned that apply_climatology is computationally expensive for a low # of stations affected. 


audit_climatology_qc(ds, train_climatology):
    """Prints the number of anomalies that would be flagged by statistical QC."""
    vars_to_check = ['TMAX_trans', 'TMIN_trans', 'TAVG_trans', 'posSWE_trans', 'negSWE_trans', 'PRCP_trans']
    print(f"--- Statistical QC Audit ---")

    for var in vars_to_check:
        anomaly def = ds[var].groupby('time.month') - train_climatology[f'{var}_mean']
        # Count 5-STD flags
        raw_5std = abs(anomaly).groupby('time.month') > (5 * train_climatology[f'{var}_std'])
        count = int(raw_5std.sum().values)
        print(f"Variable {var}: {count} points would be flagged as 5-STD outliers.")

    print(f"Note: These flags are currently NOT applied to the dataset.")

    return ds, train_climatology

def audit_mass_balance_qc(ds):
    """Prints how many years/stations would be dropped by Meyer/Sun mass-balance filters."""
    # 1. Cumulative PRCP (assuming incremental)
    cum_prcp = ds['PRCPSA'].groupby('WY').cumsum(dim='time')

    # 2. Max SWE days
    max_swe_mask = ds['WTEQ'] == ds['WTEQ'].groupby('WY').max(dim='time')
    max_swe_vals = ds['WTEQ'].where(max_swe_mask)
    assoc_prcp_vals = cum_prcp.where(max_swe_mask)

    # 3. Identify Inconsistent Years
    inconsistent_wy = (max_swe_vals > (1.05 * assoc_prcp_vals)).groupby('WY').any(dim='time')

    # 4. Identify Bad Stations
    rel_diff = ((max_swe_vals - assoc_prcp_vals) / assoc_prcp_vals).where(max_swe_mask & inconsistent_wy)
    bad_stations = (rel_diff.mean(dim='time') > 0.20).fillna(False)

    print(f"--- Mass Balance Audit ---")
    print(f"Station-Years that would be dropped: {int(inconsistent_wy.sum().values)}")
    print(f"Stations that would be fully excluded: {int(bad_stations.sum().values)}")

    return ds

def prepare_qc_variables(ds):
    """Calculates daily differences and transformations required for Serreze QC."""
    ds['SWE_DIFF'] = ds['WTEQ'] - ds['WTEQ'].shift(time=1)
    ds['TAVG_INC'] = ds['TAVG'] - ds['TAVG'].shift(time=1)

    ds['posSWE_trans'] = np.sqrt(ds['SWE_DIFF'].where(ds['SWE_DIFF'] > 0))
    ds['negSWE_trans'] = np.sqrt((-ds['SWE_DIFF']).where(ds['SWE_DIFF'] < 0))
    ds['PRCP_trans'] = np.sqrt(ds['PRCPSA'].where(ds['PRCPSA'] > 0))

    ds['TMAX_trans'] = ds['TMAX']
    ds['TMIN_trans'] = ds['TMIN']
    ds['TAVG_trans'] = ds['TAVG']
    ds['TAVG_INC_trans'] = ds['TAVG_INC']

    return ds
import numpy as np

def audit_mass_balance_qc(ds):
    """
    Meyer/Sun mass-balance filter. Checks if Maximum SWE exceeds 
    cumulative precipitation by more than 5%.
    """
    cum_prcp = ds['PRCPSA'].groupby('WY').cumsum(dim='time')
    wy_max_swe = ds['WTEQ'].groupby('WY').max(dim='time')
    wy_max_expanded = wy_max_swe.sel(WY=ds['WY'])
    
    max_swe_mask = ds['WTEQ'] == wy_max_expanded
    max_swe_vals = ds['WTEQ'].where(max_swe_mask)
    assoc_prcp_vals = cum_prcp.where(max_swe_mask)
    
    inconsistent_wy = (max_swe_vals > (1.05 * assoc_prcp_vals)).groupby('WY').any(dim='time')
    inconsistent_wy_mask = inconsistent_wy.sel(WY=ds['WY'])
    
    rel_diff = ((max_swe_vals - assoc_prcp_vals) / assoc_prcp_vals).where(max_swe_mask & inconsistent_wy_mask)
    bad_stations = (rel_diff.mean(dim='time') > 0.20).fillna(False)
    
    print(f"Station-Years dropped (5% rule): {int(inconsistent_wy.sum().values)}")
    print(f"Entire Stations excluded (20% rule): {int(bad_stations.sum().values)}")
    return ds

def fit_climatology(training_ds):
    """Calculates baseline monthly stats ONLY from training data (prevents leakage)."""
    train_climatology = {}
    for var in ['TMAX', 'TMIN', 'TAVG', 'WTEQ', 'PRCPSA']:
        train_climatology[f'{var}_mean'] = training_ds[var].groupby('time.month').mean()
        train_climatology[f'{var}_std'] = training_ds[var].groupby('time.month').std()
        count = training_ds[var].groupby("time.month").count("time")
        std = training_ds[var].groupby("time.month").std("time")
        train_climatology[f'{var}_valid'] = (count >= 30) & (std > 0)
    return train_climatology

def apply_climatology_qc(ds, train_climatology):
    """Applies the training baselines to flag 3-SD and 5-SD statistical outliers."""
    for var in ['TMAX', 'TMIN', 'TAVG', 'WTEQ', 'PRCPSA']:
        anomaly = ds[var].groupby('time.month') - train_climatology[f'{var}_mean']
        raw_5std = abs(anomaly).groupby('time.month') > (5 * train_climatology[f'{var}_std'])
        raw_3std = abs(anomaly).groupby('time.month') > (3 * train_climatology[f'{var}_std'])
        is_valid_month = train_climatology[f'{var}_valid'].sel(month=ds['time.month']).drop_vars('month')
        
        ds[f'FLAG_{var}_5STD'] = (raw_5std & is_valid_month).fillna(False).astype(bool)
    # 2. Extreme Precipitation (Serreze: > 254 mm/day)
    ds['FLAG_PRCP_EXTREME'] = ds['PRCPSA'] > 0.254
    ds['PRCPSA'] = ds['PRCPSA'].where(~ds['FLAG_PRCP_EXTREME'])

    # 4. Temperature Bounds (Serreze with Alaska Mod)
    alaska_mask = ds['state'] == 'Alaska'
    temp_extreme_any = xr.zeros_like(ds["TAVG"], dtype=bool)

    for temp in ['TMAX', 'TMIN', 'TAVG']:
        flag = (ds[temp] > 40) | \
               ((ds[temp] < -40) & ~alaska_mask) | \
               ((ds[temp] < -60) & alaska_mask)
        ds[f'FLAG_{temp}_EXTREME'] = flag
        temp_extreme_any = temp_extreme_any | flag # If one breaks, they all break

    # Apply the mask to all temperature variables simultaneously
    for temp in ['TMAX', 'TMIN', 'TAVG']:
        ds[temp] = ds[temp].where(~temp_extreme_any)

    return ds
    
def apply_hardware_spike_qc(ds):
    """
    Module 2: Identifies broken or stuck sensors using Durre's methods.
    """
    # 1. Temperature Streaks (20 days)
    for var in ['TMAX', 'TMIN', 'TAVG']:
        # min_periods=20 ensures it only flags if 20 valid days exist
        ds[f'FLAG_{var}_STREAK'] = ds[var].rolling(time=20, min_periods=20).std() < 1e-5
        flag = ds[f'FLAG_{var}_STREAK']
        expanded = flag.rolling(time=20, min_periods=1).max().astype(bool)
        ds[f'FLAG_{var}_STREAK_EXPANDED'] = expanded
        ds[var] = ds[var].where(~expanded)

    # 2. Temperature Spikes/Dips (Isolated 25C jumps)
    for var in ['TMAX', 'TMIN']:
        diff_yesterday = ds[var] - ds[var].shift(time=1)
        diff_tomorrow = ds[var] - ds[var].shift(time=-1)

        spike = (diff_yesterday >= 25) & (diff_tomorrow >= 25)
        dip = (diff_yesterday <= -25) & (diff_tomorrow <= -25)
        ds[f'FLAG_{var}_SPIKE_DIP'] = spike | dip

        # Mask the spikes/dips (Added the missing closing parenthesis here)
        ds[var] = ds[var].where(~ds[f'FLAG_{var}_SPIKE_DIP'])

    # 3. SNWD Stagnation (90 days of NON-ZERO snow)
    if 'SNWD' in ds:
        snwd_nonzero = ds['SNWD'].where(ds['SNWD'] > 0)

        # Get the strict 90-day flag
        stagnant = snwd_nonzero.rolling(time=90, min_periods=90).std() == 0
        ds['FLAG_SNWD_STAGNANT'] = stagnant

        # Expand it backwards by 90 days to mask the WHOLE stuck period
        expanded = stagnant.rolling(time=90, min_periods=1).max().astype(bool)
        ds['FLAG_SNWD_STAGNANT_EXPANDED'] = expanded

        # Explicitly mask SNWD 
        ds['SNWD'] = ds['SNWD'].where(~expanded)

    return ds


def apply_snotel_bc(ds):
    """Module 4: Applies Harms et al. temperature correction to SNOTEL only."""
    is_snotel = ds['network'] == 'SNOTEL'
    ds['is_snotel'] = is_snotel
    for var in ['TMAX', 'TMIN', 'TAVG']:
        corrected_temp = (1.03 * ds[var]) - 0.9
        ds[var] = xr.where(is_snotel, corrected_temp, ds[var])
    return dsdef gap_fill_predictors(ds):
    # 1. Add the missing loop for temperature variables
    for var in ['TMAX', 'TMIN', 'TAVG']:
        was_null = ds[var].isnull()
        ds[var] = ds[var].interpolate_na(dim='time', method='linear', limit=2)
        ds[f'FLAG_{var}_INTERP'] = was_null & ds[var].notnull()

    # 2. SNWD Logic
    was_missing = ds['SNWD'].isnull()
    ds['SNWD'] = ds['SNWD'].interpolate_na(dim='time', method='linear', limit=7)
    ds['FLAG_SNWD_FILLED'] = was_missing & ds['SNWD'].notnull()

    # 3. PRCPSA Logic (Moved UP so it actually runs before the return)
    prcp_nulls = ds['PRCPSA'].isnull()
    is_gap_end = prcp_nulls.rolling(time=4, min_periods=4).sum() == 4
    large_prcp_gap = (is_gap_end | is_gap_end.shift(time=-1, fill_value=False) |
                      is_gap_end.shift(time=-2, fill_value=False) |
                      is_gap_end.shift(time=-3, fill_value=False)) & prcp_nulls

    fill_mask = prcp_nulls & ~large_prcp_gap
    ds['PRCPSA'] = ds['PRCPSA'].fillna(0)

    ds['FLAG_PRCP_FILLED'] = fill_mask
    ds['PRCPSA'] = ds['PRCPSA'].where(~large_prcp_gap)

    return ds


        ds[f'FLAG_{var}_3STD'] = (raw_3std & is_valid_month).fillna(False).astype(bool)
APPENDIX: SNOTEL data plotting;
SNOTEL Station Plots 
fig, ax = plt.subplots(1, 6, figsize=(26, 5))
colors = {
    "SNWD": "#1f77b4",    # Deep Blue for Snow Depth
    "WTEQ": "#aec7e8",    # Light Blue for Snow Water Equivalent (SWE)
    "PRCPSA": "#ff7f0e"   # Orange for Accumulated Precipitation
}
# --- Row 1 / Subplot 0-2: Monthly Averages ---
# --- Row 1 / Subplot 0-2: Monthly Averages ---
# Store the grouped data to keep your code clean
snwd_monthly = ds.SNWD.groupby(ds.time.dt.month).mean(dim=['time', 'station'])
wteq_monthly = ds.WTEQ.groupby(ds.time.dt.month).mean(dim=['time', 'station'])
prcpsa_monthly = ds.PRCPSA.groupby(ds.time.dt.month).mean(dim=['time', 'station'])

# Pass the '.month' coordinate explicitly as the X-axis value
ax[0].plot(snwd_monthly.month, snwd_monthly, color=colors["SNWD"], linewidth=2.5)
ax[0].set_title("Monthly Mean Snow Depth", fontsize=11, fontweight='bold')
ax[0].set_ylabel("Snow Depth (m)", fontsize=10)

ax[1].plot(wteq_monthly.month, wteq_monthly, color=colors["WTEQ"], linewidth=2.5)
ax[1].set_title("Monthly Mean SWE", fontsize=11, fontweight='bold')
ax[1].set_ylabel("SWE (m)", fontsize=10)

ax[2].plot(prcpsa_monthly.month, prcpsa_monthly, color=colors["PRCPSA"], linewidth=2.5)
ax[2].set_title("Monthly Mean Accum. Precip", fontsize=11, fontweight='bold')
ax[2].set_ylabel("Precipitation (m)", fontsize=10)

# Apply standard monthly formatting to the first 3 plots
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i in range(3):
    ax[i].set_xlabel("Month", fontsize=10)
    ax[i].set_xticks(range(1, 13))
    ax[i].set_xticklabels(month_labels, rotation=45)
    # FIX 1: Lock the boundaries precisely onto your data range (1 to 12)
    ax[i].set_xlim(1,12)
    ax[i].grid(True, linestyle="--", alpha=0.5)
# 1. Map boundaries based on your total dataset with a 2-degree padding
lon_min, lon_max = float(ds.longitude.min()) - 2, float(ds.longitude.max()) + 2
lat_min, lat_max = float(ds.latitude.min()) - 2, float(ds.latitude.max()) + 2


# 3. Create a 2x3 grid of subplots with PlateCarree projection
fig, axes = plt.subplots(2, 3, figsize=(22, 14), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

# Helper function to apply map features
def style_snotel_map(ax, title):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8)
    ax.add_feature(cfeature.STATES, linestyle='--', edgecolor='gray', linewidth=0.5)

# --- Row 1 / Subplot 3-5: Day of Year (Daily Climatology) ---
# FIX 2: Exclude day 366 from the calculated data arrays using .sel()
snwd_daily = ds.SNWD.groupby(ds.time.dt.dayofyear).mean(dim=['time', 'station'])
wteq_daily = ds.WTEQ.groupby(ds.time.dt.dayofyear).mean(dim=['time', 'station'])
prcpsa_daily = ds.PRCPSA.groupby(ds.time.dt.dayofyear).mean(dim=['time', 'station'])

# Safely drop day 366 if it exists in the grouped coordinate index
days_to_keep = [d for d in snwd_daily.dayofyear.values if d <= 365]
snwd_daily = snwd_daily.sel(dayofyear=days_to_keep)
wteq_daily = wteq_daily.sel(dayofyear=days_to_keep)
prcpsa_daily = prcpsa_daily.sel(dayofyear=days_to_keep)

ax[3].plot(snwd_daily.dayofyear, snwd_daily, color=colors["SNWD"], alpha=0.8)
ax[3].set_title("Daily Climatology: Snow Depth", fontsize=11, fontweight='bold')

ax[4].plot(wteq_daily.dayofyear, wteq_daily, color=colors["WTEQ"], alpha=0.8)
ax[4].set_title("Daily Climatology: SWE", fontsize=11, fontweight='bold')

ax[5].plot(prcpsa_daily.dayofyear, prcpsa_daily, color=colors["PRCPSA"], alpha=0.8)
ax[5].set_title("Daily Climatology: Accum. Precip", fontsize=11, fontweight='bold')

# Apply daily formatting to the last 3 plots
for i in range(3, 6):
    ax[i].set_xlabel("Day of Year (1-365)", fontsize=10)

    ax[i].set_xlim(1, 365)
    ax[i].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("SNOTEL Seasonal Climatology Profiles", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()





    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linestyle=':', color='gainsboro')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=13, weight='bold', pad=10)

# Map configurations matching your requested order
vars_to_plot = ['TMAX', 'TMIN', 'TAVG', 'SNWD', 'WTEQ', 'PRCPSA']
cmaps = ['YlOrRd', 'coolwarm', 'RdYlBu_r', 'Blues', 'YlGnBu', 'viridis']
labels = ['Mean Max Temp (°C)', 'Mean Min Temp (°C)', 'Mean Temp (°C)', 'Mean Snow Depth (m)', 'Mean Water Equiv. (m)', 'Mean Precip Acc. (m)']

# Populate the SNOTEL grid
for i, var in enumerate(vars_to_plot):
    style_snotel_map(axes[i], f'SNOTEL Stations\n{var} (All-Time Mean)')

    spatial_data = ds[var].mean(dim='time', skipna=True)

    sc = axes[i].scatter(
        ds.longitude, ds.latitude,
        c=spatial_data,
        transform=ccrs.PlateCarree(),
        cmap=cmaps[i], edgecolors='black', linewidth=0.5, s=55, zorder=3
    )
    cbar = fig.colorbar(sc, ax=axes[i], orientation='horizontal', pad=0.08, shrink=0.8, aspect=25)
    cbar.set_label(labels[i], fontsize=10)

fig.suptitle("SNOTEL Network Regional Spatial Analysis", fontsize=16, fontweight='bold', y=0.98)
AllStations = easysnowdata.automatic_weather_stations.StationCollection()
'''
OTHER QC/BC (For SNOTEL temps and prcp and now that I have daymet, do not apply. If I ever go back and retest with snotel temp data for a full comparison in paper I will readd them.)

    # 2. Extreme Precipitation (Serreze: > 254 mm/day)
    ds['FLAG_PRCP_EXTREME'] = ds['PRCPSA'] > 0.254
    ds['PRCPSA'] = ds['PRCPSA'].where(~ds['FLAG_PRCP_EXTREME'])

    # 4. Temperature Bounds (Serreze with Alaska Mod)
    alaska_mask = ds['state'] == 'Alaska'
    temp_extreme_any = xr.zeros_like(ds["TAVG"], dtype=bool)
