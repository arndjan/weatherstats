import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import xml.etree.ElementTree as ET
import json

import plotly.io as pio
pio.renderers.default = 'browser'

import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

# Database connection parameters
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_DATABASE")
}

# KNMI weather stations
KNMI_STATIONS = {
    210: "Valkenburg",
    235: "De Kooy",
    240: "Schiphol",
    249: "Berkhout",
    260: "De Bilt",
    270: "Leeuwarden",
    280: "Eelde",
    290: "Twenthe",
    310: "Vlissingen",
    344: "Rotterdam",
    370: "Eindhoven",
    380: "Maastricht",
}

def insert_dataframe_to_mysql(df, table_name="knmi_minmaxtemp"):
    """
    Inserts or updates temperature data in MySQL table.
    Uses ON DUPLICATE KEY UPDATE to handle existing records.
    """
    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        sql = f"""
            INSERT INTO {table_name} (event_date, tmp_min, tmp_max, station)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                tmp_min = VALUES(tmp_min),
                tmp_max = VALUES(tmp_max);
        """

        values = [
            (row["date"], row["min_temp"], row["max_temp"], row["station"])
            for _, row in df.iterrows()
        ]

        cursor.executemany(sql, values)
        conn.commit()

        print(f"Synced {len(values)} rows to {table_name}")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

def get_data_from_database(table_name="knmi_minmaxtemp"):
    """
    Fetch all temperature data from MySQL database, grouped by station.
    Returns a dictionary with station_id as key and DataFrame as value.
    """
    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        sql = f"""
            SELECT event_date as date, tmp_min as min_temp, tmp_max as max_temp, station
            FROM {table_name}
            ORDER BY station, event_date;
        """

        cursor.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            return {}

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])

        # Group by station
        all_station_data = {}
        for station_id in df['station'].unique():
            station_df = df[df['station'] == station_id].copy()
            station_df = station_df.reset_index(drop=True)
            all_station_data[station_id] = station_df

        return all_station_data

    except mysql.connector.Error as err:
        print(f"Error reading from database: {err}")
        return {}

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

def get_knmi_temperature_data(start_date, end_date, station=240):
    """
    Fetch KNMI temperature data for a given station.
    """
    variables = ['TX', 'TN']

    url = 'https://www.daggegevens.knmi.nl/klimatologie/daggegevens'
    params = {
        'stns': str(station),
        'vars': ':'.join(variables),
        'start': start_date,
        'end': end_date,
        'fmt': 'json'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        if 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df['TX'] = df['TX'] / 10
        df['TN'] = df['TN'] / 10
        df = df.rename(columns={
            'TX': 'max_temp',
            'TN': 'min_temp'
        })
        df['station'] = station
        return df[['date', 'max_temp', 'min_temp', 'station']]
    return None

def save_to_xml(df, filename="temperature_data.xml"):
    """
    Save temperature data to an XML file.
    """
    root = ET.Element("TemperatureData")

    for _, row in df.iterrows():
        entry = ET.SubElement(root, "Day")
        date_elem = ET.SubElement(entry, "Date")
        date_elem.text = row["date"].strftime("%Y-%m-%d")

        max_temp_elem = ET.SubElement(entry, "MaxTemperature")
        max_temp_elem.text = f"{row['max_temp']:.1f}"

        min_temp_elem = ET.SubElement(entry, "MinTemperature")
        min_temp_elem.text = f"{row['min_temp']:.1f}"

    tree = ET.ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"Data saved to {filename}")

def save_to_json(df, filename="temperature_data.json"):
    """
    Save temperature data to a JSON file.
    """
    data = []
    
    for _, row in df.iterrows():
        data.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "max_temp": row["max_temp"],
            "min_temp": row["min_temp"]
        })
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Data saved to {filename}")

# Step 1: Fetch recent data from KNMI API and store in database
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')

print("Fetching recent data from KNMI API...")
for station_id, station_name in KNMI_STATIONS.items():
    print(f"  Fetching {station_name}...", end=" ")
    data = get_knmi_temperature_data(start_date, end_date, station_id)
    if data is not None:
        insert_dataframe_to_mysql(data)
        print("OK")
    else:
        print("No data")

# Step 2: Load all data from database for visualization
print("\nLoading data from database...")
all_station_data = get_data_from_database()

# Filter to only include known stations
all_station_data = {k: v for k, v in all_station_data.items() if k in KNMI_STATIONS}

if all_station_data:
    print(f"Loaded data for {len(all_station_data)} stations")

    # Get date range from data
    all_dates = pd.concat([df['date'] for df in all_station_data.values()])
    print(f"Date range: {all_dates.min().strftime('%Y-%m-%d')} to {all_dates.max().strftime('%Y-%m-%d')}")

    default_station = 240

    # Create the visualization with dropdown
    fig = go.Figure()

    station_ids = list(all_station_data.keys())
    num_stations = len(station_ids)

    # Colors for comparison view
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78'
    ]

    # Day type definitions
    day_types = [
        {'name': 'Tropische dag', 'condition': lambda df: df['max_temp'] >= 30, 'color': '#8B0000', 'symbol': 'star', 'temp_col': 'max_temp'},
        {'name': 'Zomerse dag', 'condition': lambda df: (df['max_temp'] >= 25) & (df['max_temp'] < 30), 'color': '#FF4500', 'symbol': 'diamond', 'temp_col': 'max_temp'},
        {'name': 'Warme dag', 'condition': lambda df: (df['max_temp'] >= 20) & (df['max_temp'] < 25), 'color': '#FFA500', 'symbol': 'circle', 'temp_col': 'max_temp'},
        {'name': 'Vorstdag', 'condition': lambda df: df['min_temp'] < 0, 'color': '#87CEEB', 'symbol': 'circle', 'temp_col': 'min_temp'},
        {'name': 'IJsdag', 'condition': lambda df: df['max_temp'] < 0, 'color': '#00CED1', 'symbol': 'star', 'temp_col': 'max_temp'},
    ]
    num_day_types = len(day_types)

    # === SECTION 1: Individual station traces (2 lines + 5 marker types per station) ===
    traces_per_station = 2 + num_day_types  # max line, min line, + 5 day type markers

    for i, station_id in enumerate(station_ids):
        temp_data = all_station_data[station_id]
        visible = bool(station_id == default_station)

        # Max temperature line
        fig.add_trace(go.Scatter(
            x=temp_data['date'],
            y=temp_data['max_temp'],
            fill=None,
            mode='lines',
            line_color='rgba(255, 99, 71, 0.8)',
            name='Maximum Temperature',
            visible=visible,
            legendgroup='individual',
            showlegend=visible
        ))

        # Min temperature line with fill
        fig.add_trace(go.Scatter(
            x=temp_data['date'],
            y=temp_data['min_temp'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(65, 105, 225, 0.8)',
            name='Minimum Temperature',
            visible=visible,
            legendgroup='individual',
            showlegend=visible
        ))

        # Day type markers
        for day_type in day_types:
            mask = day_type['condition'](temp_data)
            filtered = temp_data[mask]

            fig.add_trace(go.Scatter(
                x=filtered['date'],
                y=filtered[day_type['temp_col']],
                mode='markers',
                marker=dict(
                    color=day_type['color'],
                    size=10,
                    symbol=day_type['symbol'],
                    line=dict(width=1, color='white')
                ),
                name=day_type['name'],
                visible=visible,
                legendgroup='daytypes',
                showlegend=(visible and i == station_ids.index(default_station)),
                hoverinfo='skip'
            ))

    # === SECTION 2: Comparison traces - min temps ===
    for i, station_id in enumerate(station_ids):
        temp_data = all_station_data[station_id]
        station_name = KNMI_STATIONS[station_id]

        fig.add_trace(go.Scatter(
            x=temp_data['date'],
            y=temp_data['min_temp'],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=station_name,
            visible=False,
            legendgroup='comparison_min'
        ))

    # === SECTION 3: Comparison traces - max temps ===
    for i, station_id in enumerate(station_ids):
        temp_data = all_station_data[station_id]
        station_name = KNMI_STATIONS[station_id]

        fig.add_trace(go.Scatter(
            x=temp_data['date'],
            y=temp_data['max_temp'],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=station_name,
            visible=False,
            legendgroup='comparison_max'
        ))

    # Total traces: individual (lines + markers) + comparison traces
    num_individual_traces = num_stations * traces_per_station  # 7 traces per station (2 lines + 5 marker types)
    num_comparison_min_traces = num_stations
    num_comparison_max_traces = num_stations
    total_traces = num_individual_traces + num_comparison_min_traces + num_comparison_max_traces

    # Calculate annotations for individual station view
    def get_annotations(temp_data, station_name):
        max_temp_idx = temp_data['max_temp'].idxmax()
        min_temp_idx = temp_data['min_temp'].idxmin()
        return [
            dict(
                x=temp_data['date'][max_temp_idx],
                y=temp_data['max_temp'][max_temp_idx],
                text=f"Highest: {temp_data['max_temp'][max_temp_idx]:.1f}°C",
                showarrow=True,
                arrowhead=1
            ),
            dict(
                x=temp_data['date'][min_temp_idx],
                y=temp_data['min_temp'][min_temp_idx],
                text=f"Lowest: {temp_data['min_temp'][min_temp_idx]:.1f}°C",
                showarrow=True,
                arrowhead=1
            )
        ]

    # Calculate annotations for comparison views (across all stations)
    def get_comparison_annotations(temp_type='min_temp'):
        # Find global min and max across all stations
        global_min_val = float('inf')
        global_max_val = float('-inf')
        global_min_station = None
        global_max_station = None
        global_min_date = None
        global_max_date = None

        for station_id, temp_data in all_station_data.items():
            station_name = KNMI_STATIONS[station_id]
            min_idx = temp_data[temp_type].idxmin()
            max_idx = temp_data[temp_type].idxmax()

            if temp_data[temp_type][min_idx] < global_min_val:
                global_min_val = temp_data[temp_type][min_idx]
                global_min_station = station_name
                global_min_date = temp_data['date'][min_idx]

            if temp_data[temp_type][max_idx] > global_max_val:
                global_max_val = temp_data[temp_type][max_idx]
                global_max_station = station_name
                global_max_date = temp_data['date'][max_idx]

        return [
            dict(
                x=global_max_date,
                y=global_max_val,
                text=f"Highest: {global_max_val:.1f}°C ({global_max_station})",
                showarrow=True,
                arrowhead=1
            ),
            dict(
                x=global_min_date,
                y=global_min_val,
                text=f"Lowest: {global_min_val:.1f}°C ({global_min_station})",
                showarrow=True,
                arrowhead=1
            )
        ]

    # === Station dropdown buttons (for individual view) ===
    station_buttons = []
    for i, station_id in enumerate(station_ids):
        station_name = KNMI_STATIONS[station_id]
        temp_data = all_station_data[station_id]
        # Visibility: all traces for this station (lines + markers)
        visibility = [False] * total_traces
        start_idx = i * traces_per_station
        for j in range(traces_per_station):
            visibility[start_idx + j] = True

        station_buttons.append(dict(
            label=station_name,
            method='update',
            args=[
                {'visible': visibility},
                {
                    'title': f'Daily Temperature Range - {station_name}',
                    'annotations': get_annotations(temp_data, station_name)
                }
            ]
        ))

    # === View type dropdown buttons ===
    # Individual station view (default)
    default_idx = station_ids.index(default_station)
    individual_visibility = [False] * total_traces
    start_idx = default_idx * traces_per_station
    for j in range(traces_per_station):
        individual_visibility[start_idx + j] = True

    # Comparison view - all min temps
    comparison_min_visibility = [False] * total_traces
    for i in range(num_comparison_min_traces):
        comparison_min_visibility[num_individual_traces + i] = True

    # Comparison view - all max temps
    comparison_max_visibility = [False] * total_traces
    for i in range(num_comparison_max_traces):
        comparison_max_visibility[num_individual_traces + num_comparison_min_traces + i] = True

    # Define station dropdown config (to toggle visibility)
    station_dropdown_visible = dict(
        active=default_idx,
        buttons=station_buttons,
        direction='down',
        showactive=True,
        x=0.99,
        xanchor='right',
        y=1.15,
        yanchor='top',
        visible=True
    )

    station_dropdown_hidden = dict(
        active=default_idx,
        buttons=station_buttons,
        direction='down',
        showactive=True,
        x=0.99,
        xanchor='right',
        y=1.15,
        yanchor='top',
        visible=False
    )

    view_buttons = [
        dict(
            label='Per station',
            method='update',
            args=[
                {'visible': individual_visibility},
                {
                    'title': f'Daily Temperature Range - {KNMI_STATIONS[default_station]}',
                    'annotations': get_annotations(
                        all_station_data[default_station],
                        KNMI_STATIONS[default_station]
                    ),
                    'updatemenus[1].visible': True
                }
            ]
        ),
        dict(
            label='Vergelijk min. temp.',
            method='update',
            args=[
                {'visible': comparison_min_visibility},
                {
                    'title': 'Minimum Temperature - All Stations',
                    'annotations': get_comparison_annotations('min_temp'),
                    'updatemenus[1].visible': False
                }
            ]
        ),
        dict(
            label='Vergelijk max. temp.',
            method='update',
            args=[
                {'visible': comparison_max_visibility},
                {
                    'title': 'Maximum Temperature - All Stations',
                    'annotations': get_comparison_annotations('max_temp'),
                    'updatemenus[1].visible': False
                }
            ]
        )
    ]

    # Get default annotations
    default_annotations = get_annotations(
        all_station_data[default_station],
        KNMI_STATIONS[default_station]
    )

    # Update layout with dropdowns
    fig.update_layout(
        updatemenus=[
            # View type dropdown (left)
            dict(
                active=0,
                buttons=view_buttons,
                direction='down',
                showactive=True,
                x=0.01,
                xanchor='left',
                y=1.15,
                yanchor='top'
            ),
            # Station dropdown (right)
            dict(
                active=default_idx,
                buttons=station_buttons,
                direction='down',
                showactive=True,
                x=0.99,
                xanchor='right',
                y=1.15,
                yanchor='top'
            )
        ],
        annotations=default_annotations,
        title={
            'text': f'Daily Temperature Range - {KNMI_STATIONS[default_station]}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=150)
    )

    # Add a range slider
    fig.update_xaxes(rangeslider_visible=True)

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Show the plot
    fig.show()

else:
    print("Failed to fetch data")