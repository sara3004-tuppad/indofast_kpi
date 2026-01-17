"""
IndoFast Color Classification Module
Classifies stations into RED, AMBER, GREEN based on KPI thresholds.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# Classification thresholds
RED_TTS_THRESHOLD = 6  # weeks
RED_TTS_CONSECUTIVE = 2  # weeks

RED_EMA_THRESHOLD = 80  # %
RED_VEL_THRESHOLD = 3.5  # ppt/week
RED_OVERLOAD_CONSECUTIVE = 2  # weeks

RED_HBR_THRESHOLD = 80  # %
RED_HBR_EMA_THRESHOLD = 20  # % - EMA util threshold for Group C
RED_HBR_EMA_CONSECUTIVE = 2  # weeks - consecutive weeks for EMA condition in Group C
RED_ZHI_THRESHOLD = 1.40

AMBER_ACC_THRESHOLD = 0.7  # ppt/week²
AMBER_ACC_CONSECUTIVE = 2  # weeks

AMBER_VEL_THRESHOLD = 2.5  # ppt/week
AMBER_VEL_CONSECUTIVE = 3  # weeks

AMBER_ZHI_THRESHOLD = 1.20


def check_consecutive_condition(series: pd.Series, condition_func, n_consecutive: int) -> pd.Series:
    """
    Check if a condition is met for n consecutive periods.
    
    Args:
        series: Series of values to check
        condition_func: Function that returns True/False for each value
        n_consecutive: Number of consecutive periods required
        
    Returns:
        Boolean series indicating if condition met for n consecutive periods
    """
    condition_met = series.apply(condition_func)
    
    # Rolling window to check consecutive True values
    result = condition_met.rolling(window=n_consecutive, min_periods=n_consecutive).sum() >= n_consecutive
    
    return result.fillna(False)


def classify_red_group_a(station_df: pd.DataFrame) -> pd.Series:
    """
    Group A - Confirmed Saturation Risk:
    TTS < 6 weeks for 2 consecutive weeks
    """
    condition = check_consecutive_condition(
        station_df['tts'],
        lambda x: x < RED_TTS_THRESHOLD and not np.isinf(x),
        RED_TTS_CONSECUTIVE
    )
    return condition


def classify_red_group_b(station_df: pd.DataFrame) -> pd.Series:
    """
    Group B - Actual Overload Zone:
    EMA Util > 80% AND Vel > 3.5, sustained for 2 weeks
    """
    combined_condition = (station_df['ema_util'] > RED_EMA_THRESHOLD) & (station_df['velocity'] > RED_VEL_THRESHOLD)
    
    # Check for 2 consecutive weeks
    result = combined_condition.rolling(window=RED_OVERLOAD_CONSECUTIVE, min_periods=RED_OVERLOAD_CONSECUTIVE).sum() >= RED_OVERLOAD_CONSECUTIVE
    
    return result.fillna(False)


def classify_red_group_c(station_df: pd.DataFrame) -> pd.Series:
    """
    Group C - Extreme Headroom Burn:
    HBR > 80% AND EMA Util > 20% sustained for 2 consecutive weeks
    """
    hbr_condition = station_df['hbr'].fillna(0) > RED_HBR_THRESHOLD
    
    # Check EMA util condition for consecutive weeks
    ema_consecutive = check_consecutive_condition(
        station_df['ema_util'].fillna(0),
        lambda x: x > RED_HBR_EMA_THRESHOLD,
        RED_HBR_EMA_CONSECUTIVE
    )
    
    return hbr_condition & ema_consecutive


def classify_red_group_d(station_df: pd.DataFrame) -> pd.Series:
    """
    Group D - Zone Stress Confirmed:
    ZHI > 1.40
    """
    return station_df['zhi'].fillna(0) > RED_ZHI_THRESHOLD


def classify_amber_acceleration(station_df: pd.DataFrame) -> pd.Series:
    """
    AMBER Condition 1:
    Acc > 0.7 for 2 consecutive weeks
    """
    condition = check_consecutive_condition(
        station_df['acceleration'].fillna(0),
        lambda x: x > AMBER_ACC_THRESHOLD,
        AMBER_ACC_CONSECUTIVE
    )
    return condition


def classify_amber_velocity(station_df: pd.DataFrame) -> pd.Series:
    """
    AMBER Condition 2:
    Vel > 2.5 for 3 consecutive weeks
    """
    condition = check_consecutive_condition(
        station_df['velocity'].fillna(0),
        lambda x: x > AMBER_VEL_THRESHOLD,
        AMBER_VEL_CONSECUTIVE
    )
    return condition


def classify_amber_zhi(station_df: pd.DataFrame) -> pd.Series:
    """
    AMBER Condition 3:
    ZHI > 1.20
    """
    return station_df['zhi'].fillna(0) > AMBER_ZHI_THRESHOLD


def classify_station(station_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify a single station's data across all weeks.
    
    Args:
        station_df: Dataframe containing one station's weekly data
        
    Returns:
        Dataframe with 'color' and 'red_reason'/'amber_reason' columns added
    """
    station_df = station_df.copy().sort_values('week_num')
    
    # Check RED conditions
    red_a = classify_red_group_a(station_df)
    red_b = classify_red_group_b(station_df)
    red_c = classify_red_group_c(station_df)
    red_d = classify_red_group_d(station_df)
    
    # Check AMBER conditions
    amber_acc = classify_amber_acceleration(station_df)
    amber_vel = classify_amber_velocity(station_df)
    amber_zhi = classify_amber_zhi(station_df)
    
    # Determine color and reasons
    colors = []
    red_reasons = []
    amber_reasons = []
    
    for i in range(len(station_df)):
        reasons_red = []
        reasons_amber = []
        
        # Check RED conditions
        if red_a.iloc[i]:
            reasons_red.append("TTS < 6 weeks (2 consecutive)")
        if red_b.iloc[i]:
            reasons_red.append("EMA > 80% & Vel > 3.5 (2 consecutive)")
        if red_c.iloc[i]:
            reasons_red.append(f"HBR > {RED_HBR_THRESHOLD}% & EMA > {RED_HBR_EMA_THRESHOLD}% ({RED_HBR_EMA_CONSECUTIVE} consecutive)")
        if red_d.iloc[i]:
            reasons_red.append("ZHI > 1.40")
        
        # Check AMBER conditions
        if amber_acc.iloc[i]:
            reasons_amber.append("Acc > 0.7 (2 consecutive)")
        if amber_vel.iloc[i]:
            reasons_amber.append("Vel > 2.5 (3 consecutive)")
        if amber_zhi.iloc[i]:
            reasons_amber.append("ZHI > 1.20")
        
        # Determine final color (RED takes priority)
        if reasons_red:
            colors.append("RED")
            red_reasons.append("; ".join(reasons_red))
            amber_reasons.append("")
        elif reasons_amber:
            colors.append("AMBER")
            red_reasons.append("")
            amber_reasons.append("; ".join(reasons_amber))
        else:
            colors.append("GREEN")
            red_reasons.append("")
            amber_reasons.append("")
    
    station_df['color'] = colors
    station_df['red_reason'] = red_reasons
    station_df['amber_reason'] = amber_reasons
    
    return station_df


def classify_all_stations(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify all stations in the dataframe.
    
    Args:
        kpi_df: Dataframe with all KPIs calculated
        
    Returns:
        Dataframe with color classifications added
    """
    classified_dfs = []
    
    for station_id in kpi_df['station_id'].unique():
        station_data = kpi_df[kpi_df['station_id'] == station_id].copy()
        classified_station = classify_station(station_data)
        classified_dfs.append(classified_station)
    
    return pd.concat(classified_dfs, ignore_index=True)


def get_station_overall_color(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get overall station color based on historical data.
    If station was ever RED in any week → RED
    If station was ever AMBER in any week (and never RED) → AMBER
    Otherwise → GREEN
    
    Args:
        classified_df: Dataframe with weekly color classifications
        
    Returns:
        Dataframe with one row per station showing overall status
    """
    station_summaries = []
    
    for station_id in classified_df['station_id'].unique():
        station_data = classified_df[classified_df['station_id'] == station_id].copy()
        station_data = station_data.sort_values('week_num')
        
        # Get the latest week's data for current metrics
        latest_week = station_data.iloc[-1]
        
        # Check if ever RED or AMBER
        ever_red = (station_data['color'] == 'RED').any()
        ever_amber = (station_data['color'] == 'AMBER').any()
        
        # Collect all reasons from history
        all_red_reasons = station_data[station_data['red_reason'] != '']['red_reason'].unique()
        all_amber_reasons = station_data[station_data['amber_reason'] != '']['amber_reason'].unique()
        
        # Determine overall color
        if ever_red:
            overall_color = 'RED'
            overall_reason = '; '.join(all_red_reasons) if len(all_red_reasons) > 0 else ''
        elif ever_amber:
            overall_color = 'AMBER'
            overall_reason = '; '.join(all_amber_reasons) if len(all_amber_reasons) > 0 else ''
        else:
            overall_color = 'GREEN'
            overall_reason = ''
        
        # Calculate WORST values (most critical) and their weeks
        # For TTS: lower is worse (closer to saturation)
        tts_values = station_data['tts'].replace([np.inf], np.nan)
        if tts_values.notna().any():
            worst_tts_idx = tts_values.idxmin()
            worst_tts = tts_values.loc[worst_tts_idx]
            worst_tts_week = station_data.loc[worst_tts_idx, 'week']
        else:
            worst_tts = np.inf
            worst_tts_week = None
        
        # For ZHI: higher is worse (zone stress)
        zhi_values = station_data['zhi'].dropna()
        if len(zhi_values) > 0:
            worst_zhi_idx = station_data['zhi'].idxmax()
            worst_zhi = station_data.loc[worst_zhi_idx, 'zhi']
            worst_zhi_week = station_data.loc[worst_zhi_idx, 'week']
        else:
            worst_zhi = np.nan
            worst_zhi_week = None
        
        # For HBR: higher is worse (burning headroom faster)
        hbr_values = station_data['hbr'].dropna()
        if len(hbr_values) > 0:
            worst_hbr_idx = station_data['hbr'].idxmax()
            worst_hbr = station_data.loc[worst_hbr_idx, 'hbr']
            worst_hbr_week = station_data.loc[worst_hbr_idx, 'week']
        else:
            worst_hbr = np.nan
            worst_hbr_week = None
        
        # For EMA Util: higher is worse (closer to saturation)
        ema_values = station_data['ema_util'].dropna()
        if len(ema_values) > 0:
            worst_ema_idx = station_data['ema_util'].idxmax()
            worst_ema_util = station_data.loc[worst_ema_idx, 'ema_util']
            worst_ema_week = station_data.loc[worst_ema_idx, 'week']
        else:
            worst_ema_util = np.nan
            worst_ema_week = None
        
        # For Velocity: higher is worse (growing faster)
        vel_values = station_data['velocity'].dropna()
        if len(vel_values) > 0:
            worst_vel_idx = station_data['velocity'].idxmax()
            worst_velocity = station_data.loc[worst_vel_idx, 'velocity']
            worst_velocity_week = station_data.loc[worst_vel_idx, 'week']
        else:
            worst_velocity = np.nan
            worst_velocity_week = None
        
        # For Headroom: lower is worse (less capacity left)
        headroom_values = station_data['headroom'].dropna()
        if len(headroom_values) > 0:
            worst_headroom_idx = station_data['headroom'].idxmin()
            worst_headroom = station_data.loc[worst_headroom_idx, 'headroom']
            worst_headroom_week = station_data.loc[worst_headroom_idx, 'week']
        else:
            worst_headroom = np.nan
            worst_headroom_week = None
        
        # Average utilization
        avg_util = station_data['kwh'].mean()
        
        station_summaries.append({
            'station_id': station_id,
            'city': latest_week['city'],
            'zone': latest_week['zone'],
            'start_date': latest_week['start_date'],
            'overall_color': overall_color,
            'overall_reason': overall_reason,
            'average_util': avg_util,
            # Worst (most critical) values and their weeks
            'worst_ema_util': worst_ema_util,
            'worst_ema_week': worst_ema_week,
            'worst_velocity': worst_velocity,
            'worst_velocity_week': worst_velocity_week,
            'worst_tts': worst_tts,
            'worst_tts_week': worst_tts_week,
            'worst_headroom': worst_headroom,
            'worst_headroom_week': worst_headroom_week,
            'worst_hbr': worst_hbr,
            'worst_hbr_week': worst_hbr_week,
            'worst_zhi': worst_zhi,
            'worst_zhi_week': worst_zhi_week,
        })
    
    return pd.DataFrame(station_summaries)


def get_color_summary(df: pd.DataFrame, color_column: str = 'color') -> Dict:
    """
    Get summary statistics for color distribution.
    
    Args:
        df: Classified dataframe
        color_column: Name of column containing color values ('color' or 'overall_color')
        
    Returns:
        Dictionary with color counts and percentages
    """
    total = len(df)
    if total == 0:
        return {
            'total': 0,
            'red_count': 0, 'red_pct': 0,
            'amber_count': 0, 'amber_pct': 0,
            'green_count': 0, 'green_pct': 0
        }
    
    color_counts = df[color_column].value_counts()
    
    return {
        'total': total,
        'red_count': color_counts.get('RED', 0),
        'red_pct': (color_counts.get('RED', 0) / total) * 100,
        'amber_count': color_counts.get('AMBER', 0),
        'amber_pct': (color_counts.get('AMBER', 0) / total) * 100,
        'green_count': color_counts.get('GREEN', 0),
        'green_pct': (color_counts.get('GREEN', 0) / total) * 100
    }
