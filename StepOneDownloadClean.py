import io
import time
import requests
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# Paths (Define globally so both execution and functions see it)
archive_path = "snotel_data.nc" 
PROJECT_DIR = Path('/content/drive/MyDrive/SWE_Project') # Update if on NERSC

def apply_physical_bounds_qc(ds):
    """Removes physically impossible target values."""
    for var in ['WTEQ', 'SNWD']:
        ds[f'FLAG_{var}_NEGATIVE'] = ds[var] < 0
        ds[var] = ds[var].where(ds[var] >= 0)
    if 'SNWD' in ds:
        ds['FLAG_SNWD_EXTREME'] = ds['SNWD'] > 11.46
        ds['FLAG_WTEQ_EXTREME'] = ds['SNWD'] > 4
        ds['SNWD'] = ds['SNWD'].where(~ds['FLAG_SNWD_EXTREME'])
        ds['WTEQ'] = ds['WTEQ'].where(~ds['FLAG_WTEQ_EXTREME'])
    return ds

def apply_snwd_stagnation_qc(ds):
    """Flags completely stuck snow depth sensors (90 days)."""
    snwd_nonzero = ds['SNWD'].where(ds['SNWD'] > 0)
    stagnant = snwd_nonzero.rolling(time=90, min_periods=90).std() == 0
    expanded = stagnant.rolling(time=90, min_periods=1).max().astype(bool)
    ds['SNWD'] = ds['SNWD'].where(~expanded)
    return ds

def apply_snow_physics_qc(ds):
    """Flags impossible daily density/mass changes."""
    swe_diff = ds['WTEQ'] - ds['WTEQ'].shift(time=1)
    snwd_diff = ds['SNWD'] - ds['SNWD'].shift(time=1)
    density = ds['WTEQ'] / ds['SNWD'].where(ds['SNWD'] > 0)
    density_diff = density - density.shift(time=1)
    valid_days = ds['DOWY'] != 1

    rule1 = (swe_diff < 0) & (density_diff < 0) & (snwd_diff > 0)
    rule2 = (swe_diff > 0) & (snwd_diff < 0) & (density_diff > 0.0005)
    ds['FLAG_PHYSICS_QC_FAIL'] = (rule1 | rule2) & valid_days

    ds['WTEQ'] = ds['WTEQ'].where(~ds['FLAG_PHYSICS_QC_FAIL'])
    ds['SNWD'] = ds['SNWD'].where(~ds['FLAG_PHYSICS_QC_FAIL'])
    
    is_gross_jump = abs(swe_diff.where(valid_days)) > 0.254
    ds['FLAG_SWE_GROSS'] = is_gross_jump
    ds['WTEQ'] = ds['WTEQ'].where(~is_gross_jump)

    swe_diff_tomorrow = swe_diff.shift(time=-1)
    swe_jump = ((swe_diff > 0.0635) & (swe_diff_tomorrow < -0.0635)) | \
               ((swe_diff < -0.0635) & (swe_diff_tomorrow > 0.0635))
    ds['FLAG_SWE_REVERSAL'] = (swe_jump | swe_jump.shift(time=1, fill_value=False)) & valid_days
    ds['WTEQ'] = ds['WTEQ'].where(~(ds['FLAG_SWE_GROSS'] | ds['FLAG_SWE_REVERSAL']))
    return ds

def fetch_daymet_rest(station_id, lat, lon, start_year, end_year, retries=3):
    """Fetches Daymet single-pixel data."""
    url = (f"https://daymet.ornl.gov/single-pixel/api/data?"
           f"lat={lat}&lon={lon}&vars=tmin,tmax,prcp,vp,srad,dayl"
           f"&start={start_year}-01-01&end={end_year}-12-31")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            lines = r.text.splitlines()
            header_idx = next(i for i, l in enumerate(lines) if l.startswith('year,'))
            df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
            df['station'] = station_id
            return df
        except Exception as e:
            time.sleep(3)
    return None

def engineer_climate_memory(df):
    """Calculates water-year cumulative features for snowpack memory."""
    print("Engineering features...")
    master_df = df.sort_values(['station', 'time']).copy()
    
    master_df['tavg'] = (master_df['tmax'] + master_df['tmin']) / 2

    master_df['prcp_cumsum'] = master_df.groupby(['station', 'WY'])['prcp'].cumsum()
    master_df['vp_cumsum']   = master_df.groupby(['station', 'WY'])['vp'].cumsum()
    master_df['srad_cumsum'] = master_df.groupby(['station', 'WY'])['srad'].cumsum()

    master_df['degrees_below_freezing'] = np.where(master_df['tavg'] < 0, master_df['tavg'].abs(), 0)
    master_df['FDD_cumsum'] = master_df.groupby(['station', 'WY'])['degrees_below_freezing'].cumsum()

    master_df['degrees_above_freezing'] = np.where(master_df['tavg'] > 0, master_df['tavg'], 0)
    master_df['MDD_cumsum'] = master_df.groupby(['station', 'WY'])['degrees_above_freezing'].cumsum()

    master_df = master_df.drop(columns=['degrees_below_freezing', 'degrees_above_freezing'])
    
    # Save directly to file for Step 2
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_parquet(PROJECT_DIR / "master_dataset_engineered.parquet")
    print("✅ Feature engineering complete and dataset exported!")
    return master_df

if __name__ == "__main__":
    print("1. Loading and Cleaning SNOTEL Targets...")
    ds = xr.open_dataset(archive_path)
    ds = ds.where(ds['WY'] > 2000, drop=True)
    ds = apply_physical_bounds_qc(ds)
    ds = apply_snwd_stagnation_qc(ds)
    ds = apply_snow_physics_qc(ds)
    
    ds = ds.dropna(dim='time', how='any', subset=['WTEQ', 'SNWD'])
    df_targets = ds.to_dataframe().reset_index().dropna(subset=['WTEQ', 'SNWD'])
    df_targets['time'] = pd.to_datetime(df_targets['time'])

    print("2. Fetching Daymet Data...")
    station_info = df_targets.groupby('station').agg(
        lat=('latitude', 'first'), lon=('longitude', 'first'),
        start_date=('time', 'min'), end_date=('time', 'max')
    ).reset_index()

    all_daymet_data = []
    for _, row in station_info.iterrows():
        station_daymet = fetch_daymet_rest(row['station'], row['lat'], row['lon'], row['start_date'].year, row['end_date'].year)
        if station_daymet is not None:
            all_daymet_data.append(station_daymet)

    df_daymet = pd.concat(all_daymet_data, ignore_index=True)
    df_daymet['time'] = pd.to_datetime(df_daymet['year'].astype(str), format='%Y') + pd.to_timedelta(df_daymet['yday'] - 1, unit='D')
    df_daymet = df_daymet.drop(columns=['year', 'yday'])

    master_df = pd.merge(df_targets, df_daymet, on=['station', 'time'], how='inner')
    master_df.rename(columns=lambda x: x.split(' ')[0], inplace=True)
    flag_cols = [c for c in master_df.columns if 'FLAG' in c]
    master_df = master_df.drop(columns=flag_cols)
    
    # Run feature engineering function with corrected data-type structure
    master_df = engineer_climate_memory(master_df)
