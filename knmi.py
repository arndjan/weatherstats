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
    Inserts an entire Pandas DataFrame into a MySQL table.
    """
    conn = None
    cursor = None
    
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # SQL INSERT statement
        sql = f"""
            INSERT INTO {table_name} (event_date, tmp_min, tmp_max, station)
            VALUES (%s, %s, %s, %s);
        """

        # Convert DataFrame to list of tuples for batch insert
        values = [
            (row["date"], row["min_temp"], row["max_temp"], row["station"])
            for _, row in df.iterrows()
        ]

        # Execute batch insert
        cursor.executemany(sql, values)
        conn.commit()

        print(f"Inserted {cursor.rowcount} rows into {table_name}")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # Close cursor and connection only if they were successfully created
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

# Get last 90 days of data
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')

# Fetch data for all stations
print("Fetching data for all stations...")
all_station_data = {}
for station_id, station_name in KNMI_STATIONS.items():
    print(f"  Fetching {station_name}...", end=" ")
    data = get_knmi_temperature_data(start_date, end_date, station_id)
    if data is not None:
        all_station_data[station_id] = data
        print("OK")
    else:
        print("No data")

if all_station_data:
    # Save data for first station (Schiphol) as default
    default_station = 240
    if default_station in all_station_data:
        save_to_xml(all_station_data[default_station])
        save_to_json(all_station_data[default_station])
        insert_dataframe_to_mysql(all_station_data[default_station])

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

    # === SECTION 1: Individual station traces (2 per station) ===
    for i, station_id in enumerate(station_ids):
        temp_data = all_station_data[station_id]
        visible = (station_id == default_station)

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

    # Total traces: (num_stations * 2) individual + num_stations min comparison + num_stations max comparison
    num_individual_traces = num_stations * 2
    num_comparison_min_traces = num_stations
    num_comparison_max_traces = num_stations
    total_traces = num_individual_traces + num_comparison_min_traces + num_comparison_max_traces

    # Calculate annotations for each station
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

    # === Station dropdown buttons (for individual view) ===
    station_buttons = []
    for i, station_id in enumerate(station_ids):
        station_name = KNMI_STATIONS[station_id]
        temp_data = all_station_data[station_id]
        # Visibility: only this station's 2 traces, hide all comparison traces
        visibility = [False] * total_traces
        visibility[i * 2] = True      # max temp trace
        visibility[i * 2 + 1] = True  # min temp trace

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
    individual_visibility[default_idx * 2] = True
    individual_visibility[default_idx * 2 + 1] = True

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
                    'annotations': [],
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
                    'annotations': [],
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
            x=0.15
        )
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