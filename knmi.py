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
            (row["date"], row["min_temp"], row["max_temp"], 240)
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

def get_knmi_temperature_data(start_date, end_date):
    """
    Fetch KNMI temperature data for Amsterdam (Schiphol)
    """
    station = [240]
    variables = ['TX', 'TN']
    
    url = 'https://www.daggegevens.knmi.nl/klimatologie/daggegevens'
    params = {
        'stns': ':'.join(map(str, station)),
        'vars': ':'.join(variables),
        'start': start_date,
        'end': end_date,
        'fmt': 'json'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['TX'] = df['TX'] / 10
        df['TN'] = df['TN'] / 10
        df = df.rename(columns={
            'TX': 'max_temp',
            'TN': 'min_temp'
        })
        return df[['date', 'max_temp', 'min_temp']]
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

temp_data = get_knmi_temperature_data(start_date, end_date)

if temp_data is not None:
    # Save data to XML
    save_to_xml(temp_data)
    
    # save data to json
    save_to_json(temp_data)
    
    # store in mysql database, table knmi_minmaxtemp
    insert_dataframe_to_mysql(temp_data)
    
    # Create the visualization
    fig = go.Figure()

    # Add temperature range as a filled area
    fig.add_trace(go.Scatter(
        x=temp_data['date'],
        y=temp_data['max_temp'],
        fill=None,
        mode='lines',
        line_color='rgba(255, 99, 71, 0.8)',
        name='Maximum Temperature'
    ))

    fig.add_trace(go.Scatter(
        x=temp_data['date'],
        y=temp_data['min_temp'],
        fill='tonexty',  # fill area between traces
        mode='lines',
        line_color='rgba(65, 105, 225, 0.8)',
        name='Minimum Temperature'
    ))

    # Update layout with better styling
    fig.update_layout(
        title={
            'text': 'Daily Temperature Range in Amsterdam (Schiphol)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        template='plotly_white',  # clean template
        height=600,  # larger figure
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Add a range slider
    fig.update_xaxes(rangeslider_visible=True)

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Add annotations for extreme temperatures
    max_temp_idx = temp_data['max_temp'].idxmax()
    min_temp_idx = temp_data['min_temp'].idxmin()

    fig.add_annotation(
        x=temp_data['date'][max_temp_idx],
        y=temp_data['max_temp'][max_temp_idx],
        text=f"Highest: {temp_data['max_temp'][max_temp_idx]:.1f}°C",
        showarrow=True,
        arrowhead=1
    )

    fig.add_annotation(
        x=temp_data['date'][min_temp_idx],
        y=temp_data['min_temp'][min_temp_idx],
        text=f"Lowest: {temp_data['min_temp'][min_temp_idx]:.1f}°C",
        showarrow=True,
        arrowhead=1
    )

    # Show the plot
    fig.show()

else:
    print("Failed to fetch data")