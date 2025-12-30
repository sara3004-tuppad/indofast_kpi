# IndoFast Early Warning Framework

A Streamlit-based dashboard for monitoring battery swapping station utilization and identifying stations at risk of saturation.

## Overview

This application analyzes weekly station utilization data and classifies stations into three categories:
- **RED** - Immediate action required (station at risk of saturation)
- **AMBER** - Early warning (trending towards saturation)
- **GREEN** - Normal operation

## Input Data Format

The application expects an Excel file (`.xlsx`) with the following structure:

| Column | Description | Required |
|--------|-------------|----------|
| `station_id` | Unique identifier for each station | Yes |
| `city` | City where station is located | Yes |
| `zone` | Zone/area within the city | Yes |
| `Energized Date` | Date when station became operational | Optional |
| `W01`, `W02`, ... `Wnn` | Weekly average daily kWh consumption | Yes |

### Example Data Structure

```
station_id,city,zone,Energized Date,W01,W02,W03,...,W47
WMQISXM1V1-00013,Bengaluru,Koramangala,2024-07-24,135.12,148.42,156.78,...,157.67
WMQISXM1V1-00023,Bengaluru,Whitefield,2024-09-06,140.21,135.36,142.89,...,197.45
```

**Notes:**
- Week columns should be named `W01`, `W02`, etc. (case-insensitive)
- Weekly values represent **average daily kWh consumption** for that week
- Missing/NaN values are handled gracefully

## Core Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Daily Capacity | 264 kWh/day | Maximum daily capacity per station |
| Saturation Threshold | 224.4 kWh/day | 85% of capacity |
| EMA Alpha | 0.3 | Smoothing factor for EMA calculation |

## KPI Calculations

### 1. Raw Utilization
```
U(t) = (avg_daily_kWh / 264) × 100
```

### 2. EMA Utilization (Smoothed)
```
EMA_U(t) = 0.3 × U(t) + 0.7 × EMA_U(t-1)
```

### 3. Velocity (ppt/week)
```
Vel(t) = EMA_U(t) - EMA_U(t-1)
```

### 4. Acceleration (ppt/week²)
```
Acc(t) = Vel(t) - Vel(t-1)
```

### 5. Time-to-Saturation (TTS)
```
TTS(t) = (85 - EMA_U(t)) / Vel(t)   [if Vel > 0]
TTS(t) = ∞                          [if Vel ≤ 0]
```

### 6. Headroom
```
Headroom = 224.4 - avg_daily_kWh
```

### 7. Headroom Burn Ratio (HBR)
```
HBR = ((kWh(t) - kWh(t-4)) / Headroom) × 100
```

### 8. Zone Heat Index (ZHI)
```
ZHI = EMA_Util(zone) / EMA_Util(city)
```

## Classification Rules

### RED - Immediate Action Required

A station is classified as **RED** if ANY of the following conditions were EVER met:

| Group | Condition |
|-------|-----------|
| A - Saturation Risk | TTS < 6 weeks for 2 consecutive weeks |
| B - Overload Zone | EMA Util > 80% AND Velocity > 3.5 for 2 consecutive weeks |
| C - Headroom Burn | HBR > 80% |
| D - Zone Stress | ZHI > 1.40 |

### AMBER - Early Warning

A station is classified as **AMBER** if ANY of the following conditions were EVER met (and never RED):

| Condition |
|-----------|
| Acceleration > 0.7 for 2 consecutive weeks |
| Velocity > 2.5 for 3 consecutive weeks |
| ZHI > 1.20 |

### GREEN - Normal Operation

Station is **GREEN** if no RED or AMBER conditions were ever triggered.

## Application Flow

```
┌─────────────────┐
│  Upload XLSX    │
│  Station Data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parse Weekly   │
│  Columns        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Calculate KPIs │
│  per Station    │
│  per Week       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classify Each  │
│  Week (R/A/G)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Determine      │
│  Overall Color  │
│  (Historical)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Display        │
│  Dashboard      │
└─────────────────┘
```

## Project Structure

```
indofast_kpi/
├── app.py                 # Main Streamlit application
├── kpi_calculator.py      # KPI calculation logic
├── color_classifier.py    # RED/AMBER/GREEN classification
├── pyproject.toml         # Poetry dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry package manager

### Setup

```bash
# Clone/navigate to project directory
cd indofast_kpi

# Install dependencies with Poetry
poetry install

# Run the application
poetry run streamlit run app.py
```

## Dashboard Features

### Filters
- **City** - Filter by city
- **Zone** - Filter by zone (filtered based on selected city)
- **Station ID** - Search for specific station
- **Status** - Filter by RED/AMBER/GREEN

### Visualizations
- **Status Distribution Pie Chart** - Overview of station health
- **Utilization Distribution Histogram** - Distribution of average utilization
- **Utilization Trends** - Weekly trend of utilization and velocity
- **Zone Performance Heatmap** - Treemap showing zone-level performance

### Station Details Table
Shows for each station:
- Average utilization (kWh/day)
- Worst EMA utilization and which week it occurred
- Worst velocity and which week
- Worst TTS (Time-to-Saturation) and which week
- Worst HBR (Headroom Burn Ratio) and which week
- Worst ZHI (Zone Heat Index) and which week
- Alert reason explaining why station is RED/AMBER

### Critical Stations Section
Expandable cards for RED stations showing:
- All worst metrics with their occurrence weeks
- Alert reasons

## Output

The dashboard allows exporting filtered station data as CSV for further analysis.

## Version

Bangalore Locked v3.0

