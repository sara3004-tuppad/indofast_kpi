"""
IndoFast Early Warning Framework - Streamlit Dashboard
Interactive station planning and monitoring application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kpi_calculator import process_data, DAILY_CAPACITY, SATURATION_THRESHOLD
from color_classifier import classify_all_stations, get_color_summary, get_station_overall_color

# Page configuration
st.set_page_config(
    page_title="IndoFast Early Warning Framework",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --red-color: #FF4B4B;
        --amber-color: #FFA500;
        --green-color: #00C853;
        --bg-dark: #0E1117;
        --card-bg: #1E1E1E;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    
    .metric-card.red { border-left-color: #FF4B4B; }
    .metric-card.amber { border-left-color: #FFA500; }
    .metric-card.green { border-left-color: #00C853; }
    .metric-card.total { border-left-color: #4A90D9; }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-pct {
        font-size: 1rem;
        color: #aaa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }
    
    /* Status badges */
    .status-red { 
        background-color: #FF4B4B; 
        color: white; 
        padding: 4px 12px; 
        border-radius: 20px; 
        font-weight: 600;
    }
    .status-amber { 
        background-color: #FFA500; 
        color: black; 
        padding: 4px 12px; 
        border-radius: 20px; 
        font-weight: 600;
    }
    .status-green { 
        background-color: #00C853; 
        color: white; 
        padding: 4px 12px; 
        border-radius: 20px; 
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def create_metric_card(label: str, value: str, pct: str = None, color_class: str = "total"):
    """Create a styled metric card."""
    pct_html = f'<p class="metric-pct">{pct}</p>' if pct else ''
    return f"""
    <div class="metric-card {color_class}">
        <p class="metric-label">{label}</p>
        <p class="metric-value" style="color: {'#FF4B4B' if color_class == 'red' else '#FFA500' if color_class == 'amber' else '#00C853' if color_class == 'green' else '#4A90D9'}">{value}</p>
        {pct_html}
    </div>
    """


def load_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    cities = ['Bangalore', 'Mumbai', 'Delhi', 'Chennai']
    zones = {
        'Bangalore': ['Koramangala', 'Whitefield', 'Indiranagar', 'Electronic City'],
        'Mumbai': ['Andheri', 'Bandra', 'Powai', 'Thane'],
        'Delhi': ['Connaught Place', 'Gurgaon', 'Noida', 'Dwarka'],
        'Chennai': ['T Nagar', 'Anna Nagar', 'Velachery', 'OMR']
    }
    
    data = []
    station_id = 1
    
    for city in cities:
        for zone in zones[city]:
            for _ in range(3):  # 3 stations per zone
                row = {
                    'station_id': f'STN_{station_id:03d}',
                    'zone': zone,
                    'city': city,
                    'start_date': '2024-01-01'
                }
                
                # Generate weekly kWh data with some trend
                base = np.random.uniform(100, 200)
                trend = np.random.uniform(-2, 5)
                noise_scale = np.random.uniform(5, 20)
                
                for week in range(1, 25):
                    kwh = base + trend * week + np.random.normal(0, noise_scale)
                    kwh = max(0, min(kwh, 300))  # Clamp values
                    row[f'w{week:02d}'] = round(kwh, 2)
                
                data.append(row)
                station_id += 1
    
    return pd.DataFrame(data)


def main():
    # Header
    st.markdown('<h1 class="main-header">IndoFast Early Warning Framework</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Station Planning & Monitoring Dashboard | Bangalore Locked v3.0</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Source")
        
        data_source = st.radio(
            "Select data source:",
            ["Upload XLSX", "Use Sample Data"],
            index=1
        )
        
        df = None
        
        if data_source == "Upload XLSX":
            uploaded_file = st.file_uploader(
                "Upload your station data",
                type=['xlsx', 'xls'],
                help="File should contain: station_id, zone, city, start_date, w01, w02, ..."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_excel(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} stations")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            df = load_sample_data()
            st.info("📊 Using sample data (48 stations)")
        
        st.markdown("---")
    
    if df is None:
        st.warning("Please upload a file or select sample data to begin.")
        return
    
    # Process data
    with st.spinner("Processing data..."):
        try:
            kpi_df, week_cols = process_data(df)
            classified_df = classify_all_stations(kpi_df)
            # Get overall station colors based on historical data
            station_summary_df = get_station_overall_color(classified_df)
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        # City filter
        cities = ['All'] + sorted(station_summary_df['city'].unique().tolist())
        selected_city = st.selectbox("City", cities)
        
        # Zone filter (filtered by city)
        if selected_city != 'All':
            zone_options = ['All'] + sorted(
                station_summary_df[station_summary_df['city'] == selected_city]['zone'].unique().tolist()
            )
        else:
            zone_options = ['All'] + sorted(station_summary_df['zone'].unique().tolist())
        selected_zone = st.selectbox("Zone", zone_options)
        
        # Station search
        station_search = st.text_input(
            "Search Station ID",
            placeholder="e.g., STN_001"
        )
        
        # Color filter
        color_filter = st.multiselect(
            "Filter by Status",
            ["RED", "AMBER", "GREEN"],
            default=["RED", "AMBER", "GREEN"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Legend")
        st.markdown("""
        <span class="status-red">RED</span> Ever triggered critical condition<br><br>
        <span class="status-amber">AMBER</span> Ever triggered warning condition<br><br>
        <span class="status-green">GREEN</span> Always normal operation
        """, unsafe_allow_html=True)
    
    # Apply filters to station summary
    filtered_df = station_summary_df.copy()
    
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city'] == selected_city]
    
    if selected_zone != 'All':
        filtered_df = filtered_df[filtered_df['zone'] == selected_zone]
    
    if station_search:
        filtered_df = filtered_df[
            filtered_df['station_id'].str.contains(station_search, case=False, na=False)
        ]
    
    if color_filter:
        filtered_df = filtered_df[filtered_df['overall_color'].isin(color_filter)]
    
    # Get summary statistics
    summary = get_color_summary(filtered_df, color_column='overall_color')
    
    # Main content area
    st.markdown("### 📊 Station Status Overview (Based on Historical Data)")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card("Total Stations", str(summary['total']), color_class="total"),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_metric_card(
                "Critical (RED)", 
                str(summary['red_count']), 
                f"{summary['red_pct']:.1f}%",
                color_class="red"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_metric_card(
                "Warning (AMBER)", 
                str(summary['amber_count']), 
                f"{summary['amber_pct']:.1f}%",
                color_class="amber"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            create_metric_card(
                "Normal (GREEN)", 
                str(summary['green_count']), 
                f"{summary['green_pct']:.1f}%",
                color_class="green"
            ),
            unsafe_allow_html=True
        )
    
    # Additional metrics row
    st.markdown("---")
    
    if len(filtered_df) > 0:
        avg_worst_util = filtered_df['worst_ema_util'].mean()
        avg_worst_velocity = filtered_df['worst_velocity'].mean()
        avg_worst_headroom = filtered_df['worst_headroom'].mean()
        # stations_approaching = len(filtered_df[(filtered_df['worst_tts'] < 12) & (filtered_df['worst_tts'] > 0)])
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        
        with mcol1:
            st.metric("Avg Worst EMA Util", f"{avg_worst_util:.1f}%", delta=None, help="Average of worst utilization across stations")
        
        with mcol2:
            st.metric("Avg Worst Velocity", f"{avg_worst_velocity:.2f} ppt/week", help="Average of worst velocity across stations")
        
        with mcol3:
            st.metric("Avg Worst Headroom", f"{avg_worst_headroom:.1f} kWh/day", help="Average of minimum headroom across stations")
        
        # with mcol4:
        #     st.metric("Approaching Saturation", f"{stations_approaching}", help="Stations with worst TTS < 12 weeks")
    
    st.markdown("---")
    
    # Charts section
    st.markdown("### 📈 Visualizations")
    
    chart_col1 = st.columns(1)
    
    with chart_col1:
        # Color distribution pie chart
        if len(filtered_df) > 0:
            color_counts = filtered_df['overall_color'].value_counts()
            fig_pie = px.pie(
                values=color_counts.values,
                names=color_counts.index,
                title="Station Status Distribution (Historical)",
                color=color_counts.index,
                color_discrete_map={
                    'RED': '#FF4B4B',
                    'AMBER': '#FFA500',
                    'GREEN': '#00C853'
                },
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#888',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    

    
    # Zone heatmap
    st.markdown("### 🗺️ Zone Performance Heatmap")
    
    zone_summary = filtered_df.groupby(['city', 'zone']).agg({
        'worst_ema_util': 'mean',
        'station_id': 'count',
        'worst_zhi': 'mean'
    }).reset_index()
    zone_summary.columns = ['city', 'zone', 'avg_worst_util', 'station_count', 'avg_worst_zhi']
    
    if len(zone_summary) > 0:
        fig_heatmap = px.treemap(
            zone_summary,
            path=['city', 'zone'],
            values='station_count',
            color='avg_worst_util',
            color_continuous_scale=['#00C853', '#FFA500', '#FF4B4B'],
            range_color=[0, 100],
            title="Zone Utilization (size = station count, color = avg worst utilization)"
        )
        fig_heatmap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Station details table
    st.markdown("### 📋 Station Details")
    
    # Prepare display dataframe
    display_cols = [
        'station_id', 'city', 'zone', 'overall_color', 
        'average_util', 
        'worst_ema_util', 'worst_ema_week',
        'worst_velocity', 'worst_velocity_week',
        'worst_tts', 'worst_tts_week',
        'worst_hbr', 'worst_hbr_week',
        'worst_zhi', 'worst_zhi_week',
        'overall_reason'
    ]
    
    display_df = filtered_df[display_cols].copy()
    
    # Format numeric columns
    display_df['average_util'] = display_df['average_util'].round(2)
    display_df['worst_ema_util'] = display_df['worst_ema_util'].round(2)
    display_df['worst_velocity'] = display_df['worst_velocity'].round(3)
    display_df['worst_tts'] = display_df['worst_tts'].apply(lambda x: f"{x:.1f}" if x != float('inf') and pd.notna(x) else "∞")
    display_df['worst_hbr'] = display_df['worst_hbr'].round(2)
    display_df['worst_zhi'] = display_df['worst_zhi'].round(3)
    
    # Rename columns for display
    display_df.columns = [
        'Station ID', 'City', 'Zone', 'Status',
        'Avg Util',
        'Worst EMA %', 'EMA Week',
        'Worst Velocity', 'Vel Week',
        'Worst TTS', 'TTS Week',
        'Worst HBR %', 'HBR Week',
        'Worst ZHI', 'ZHI Week',
        'Alert Reason'
    ]
    
    # Color coding function
    def color_status(val):
        if val == 'RED':
            return 'background-color: #FF4B4B; color: white'
        elif val == 'AMBER':
            return 'background-color: #FFA500; color: black'
        elif val == 'GREEN':
            return 'background-color: #00C853; color: white'
        return ''
    
    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        ['Status', 'Avg Util', 'Worst EMA %', 'Worst Velocity', 'Worst TTS', 'Worst HBR %', 'Worst ZHI'],
        index=2
    )
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    
    ascending = sort_order == "Ascending"
    if sort_col == 'Worst TTS':
        # Handle infinity in sorting
        display_df['_tts_sort'] = display_df['Worst TTS'].apply(
            lambda x: float('inf') if x == '∞' else float(x)
        )
        display_df = display_df.sort_values('_tts_sort', ascending=ascending)
        display_df = display_df.drop('_tts_sort', axis=1)
    elif sort_col == 'Status':
        # Custom sort order for status
        status_order = {'RED': 0, 'AMBER': 1, 'GREEN': 2}
        display_df['_status_sort'] = display_df['Status'].map(status_order)
        display_df = display_df.sort_values('_status_sort', ascending=ascending)
        display_df = display_df.drop('_status_sort', axis=1)
    else:
        display_df = display_df.sort_values(sort_col, ascending=ascending)
    
    # Apply styling and display
    styled_df = display_df.style.applymap(color_status, subset=['Status'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="indofast_stations_summary.csv",
        mime="text/csv"
    )
    
    # Top critical stations
    st.markdown("### ⚠️ Stations Requiring Immediate Attention")
    
    critical_df = filtered_df[filtered_df['overall_color'] == 'RED'].copy()
    # Sort by worst TTS (lowest first - most critical)
    critical_df['_tts_sort'] = critical_df['worst_tts'].replace([np.inf], 9999)
    critical_df = critical_df.sort_values('_tts_sort', ascending=True).head(10)
    
    if len(critical_df) > 0:
        for _, row in critical_df.iterrows():
            with st.expander(f"🔴 {row['station_id']} - {row['zone']}, {row['city']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Util", f"{row['average_util']:.1f} kWh/day")
                    st.metric("Worst EMA Util", f"{row['worst_ema_util']:.1f}%", help=f"Week: {row['worst_ema_week']}")
                with col2:
                    tts_display = f"{row['worst_tts']:.1f} weeks" if row['worst_tts'] != float('inf') and pd.notna(row['worst_tts']) else "∞"
                    st.metric("Worst TTS", tts_display, help=f"Week: {row['worst_tts_week']}")
                    st.metric("Worst Velocity", f"{row['worst_velocity']:.2f} ppt/week", help=f"Week: {row['worst_velocity_week']}")
                with col3:
                    st.metric("Worst HBR", f"{row['worst_hbr']:.1f}%" if pd.notna(row['worst_hbr']) else "N/A", help=f"Week: {row['worst_hbr_week']}")
                    st.metric("Worst ZHI", f"{row['worst_zhi']:.2f}" if pd.notna(row['worst_zhi']) else "N/A", help=f"Week: {row['worst_zhi_week']}")
                
                st.markdown(f"**Alert Reason:** {row['overall_reason']}")
    else:
        st.success("✅ No critical stations in current selection!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>IndoFast Early Warning Framework v3.0 | Capacity: 264 kWh/day | Saturation: 85% (224.4 kWh/day)</p>
        <p>EMA α = 0.3 | Data refreshed weekly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

