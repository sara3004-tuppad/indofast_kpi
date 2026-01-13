"""
IndoFast KPI Calculator Module
Calculates all KPIs for station utilization analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

# Constants
DAILY_CAPACITY = 264  # kWh/day
SATURATION_THRESHOLD = 224.4  # kWh/day (85% of capacity)
SATURATION_PERCENT = 85  # %
EMA_ALPHA = 0.3


def parse_weekly_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse the uploaded XLSX data and identify weekly columns.
    
    Args:
        df: Raw dataframe from XLSX
        
    Returns:
        Tuple of (cleaned dataframe, list of week column names)
    """
    # Identify week columns (w01, w02, etc.)
    week_cols = [col for col in df.columns if col.lower().startswith('w') and col[1:].isdigit()]
    week_cols = sorted(week_cols, key=lambda x: int(x[1:]))
    df_cols_lower = {c.lower(): c for c in df.columns}
    # Ensure required columns exist
    required_cols = ['station_id', 'zone', 'city']
    for col in required_cols:
        if col.lower() not in df_cols_lower:
            raise ValueError(f"Missing required column: {col}")
    
    return df, week_cols


def calculate_raw_utilization(kwh: float) -> float:
    """Calculate raw utilization percentage."""
    return (kwh / DAILY_CAPACITY) * 100


def calculate_ema_series(values: pd.Series, alpha: float = EMA_ALPHA) -> pd.Series:
    """
    Calculate Exponential Moving Average for a series of values.
    
    Args:
        values: Series of raw utilization values
        alpha: Smoothing factor (default 0.3)
        
    Returns:
        Series of EMA values
    """
    ema = np.zeros(len(values))
    ema[0] = values.iloc[0] if not pd.isna(values.iloc[0]) else 0
    
    for i in range(1, len(values)):
        if pd.isna(values.iloc[i]):
            ema[i] = ema[i-1]
        else:
            ema[i] = alpha * values.iloc[i] + (1 - alpha) * ema[i-1]
    
    return pd.Series(ema, index=values.index)


def calculate_velocity(ema_series: pd.Series) -> pd.Series:
    """Calculate velocity (ppt/week) = EMA(t) - EMA(t-1)."""
    return ema_series.diff()


def calculate_acceleration(velocity_series: pd.Series) -> pd.Series:
    """Calculate acceleration (ppt/week²) = Vel(t) - Vel(t-1)."""
    return velocity_series.diff()


def calculate_tts(ema_util: float, velocity: float) -> float:
    """
    Calculate Time-to-Saturation (weeks).
    
    Args:
        ema_util: Current EMA utilization
        velocity: Current velocity
        
    Returns:
        Weeks to reach 85% saturation, or infinity if velocity <= 0
    """
    if velocity > 0:
        tts = (SATURATION_PERCENT - ema_util) / velocity
        return max(0, tts)  # Can't be negative
    return float('inf')


def calculate_headroom(kwh: float) -> float:
    """Calculate headroom (kWh/day) = 224.4 - avg_daily_kWh."""
    return SATURATION_THRESHOLD - kwh


def calculate_hbr(current_kwh: float, kwh_4_weeks_ago: float, headroom: float) -> float:
    """
    Calculate Headroom Burn Ratio (%).
    
    HBR = ((kWh(t) - kWh(t-4)) / Headroom) * 100
    """
    if headroom <= 0:
        return 100.0  # Already at or over saturation
    
    if pd.isna(kwh_4_weeks_ago):
        return 0.0
    
    return ((current_kwh - kwh_4_weeks_ago) / headroom) * 100


def calculate_station_kpis(df: pd.DataFrame, week_cols: List[str]) -> pd.DataFrame:
    """
    Calculate all KPIs for each station across all weeks.
    
    Args:
        df: Dataframe with station data
        week_cols: List of week column names
        
    Returns:
        Dataframe with all KPIs calculated
    """
    result_rows = []
    
    for _, row in df.iterrows():
        station_id = row['station_id']
        zone = row['zone']
        city = row['city']
        start_date = row.get('start_date', None)
        
        # Get weekly kWh values
        kwh_values = pd.Series([row[col] for col in week_cols], index=week_cols)
        
        # Calculate raw utilization
        raw_util = kwh_values.apply(calculate_raw_utilization)
        
        # Calculate EMA utilization
        ema_util = calculate_ema_series(raw_util)
        
        # Calculate velocity and acceleration
        velocity = calculate_velocity(ema_util)
        acceleration = calculate_acceleration(velocity)
        
        # Store results for each week
        for i, week in enumerate(week_cols):
            kwh = kwh_values.iloc[i]
            headroom = calculate_headroom(kwh) if not pd.isna(kwh) else None
            
            # Calculate HBR (need 4 weeks of history)
            kwh_4_ago = kwh_values.iloc[i-4] if i >= 4 else np.nan
            hbr = calculate_hbr(kwh, kwh_4_ago, headroom) if headroom is not None else None
            
            # Calculate TTS
            tts = calculate_tts(ema_util.iloc[i], velocity.iloc[i]) if not pd.isna(velocity.iloc[i]) else float('inf')
            
            result_rows.append({
                'station_id': station_id,
                'zone': zone,
                'city': city,
                'start_date': start_date,
                'week': week,
                'week_num': i + 1,
                'kwh': kwh,
                'raw_util': raw_util.iloc[i],
                'ema_util': ema_util.iloc[i],
                'velocity': velocity.iloc[i],
                'acceleration': acceleration.iloc[i],
                'tts': tts,
                'headroom': headroom,
                'hbr': hbr
            })
    
    return pd.DataFrame(result_rows)


def calculate_zone_city_ema(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate zone and city level EMA averages for ZHI calculation.
    
    Args:
        kpi_df: Dataframe with station KPIs
        
    Returns:
        Dataframe with zone_ema, city_ema, and ZHI added
    """
    # Calculate zone average EMA for each week
    zone_ema = kpi_df.groupby(['zone', 'week'])['ema_util'].mean().reset_index()
    zone_ema.columns = ['zone', 'week', 'zone_ema']
    
    # Calculate city average EMA for each week
    city_ema = kpi_df.groupby(['city', 'week'])['ema_util'].mean().reset_index()
    city_ema.columns = ['city', 'week', 'city_ema']
    
    # Merge back to main dataframe
    kpi_df = kpi_df.merge(zone_ema, on=['zone', 'week'], how='left')
    kpi_df = kpi_df.merge(city_ema, on=['city', 'week'], how='left')
    
    # Calculate ZHI
    kpi_df['zhi'] = kpi_df['zone_ema'] / kpi_df['city_ema']
    kpi_df['zhi'] = kpi_df['zhi'].replace([np.inf, -np.inf], np.nan)
    
    return kpi_df


def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main processing function that calculates all KPIs.
    
    Args:
        df: Raw dataframe from XLSX
        
    Returns:
        Tuple of (processed dataframe with all KPIs, list of week columns)
    """
    # Parse and validate data
    df, week_cols = parse_weekly_data(df)
    
    # Calculate station-level KPIs
    kpi_df = calculate_station_kpis(df, week_cols)
    
    # Calculate zone/city averages and ZHI
    kpi_df = calculate_zone_city_ema(kpi_df)
    
    return kpi_df, week_cols

