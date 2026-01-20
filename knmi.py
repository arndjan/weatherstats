import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

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

# Day type definitions
DAY_TYPES = [
    {'name': 'Tropische dag', 'condition': lambda df: df['max_temp'] >= 30, 'color': '#8B0000', 'symbol': 'star', 'temp_col': 'max_temp'},
    {'name': 'Zomerse dag', 'condition': lambda df: (df['max_temp'] >= 25) & (df['max_temp'] < 30), 'color': '#FF4500', 'symbol': 'diamond', 'temp_col': 'max_temp'},
    {'name': 'Warme dag', 'condition': lambda df: (df['max_temp'] >= 20) & (df['max_temp'] < 25), 'color': '#FFA500', 'symbol': 'circle', 'temp_col': 'max_temp'},
    {'name': 'Vorstdag', 'condition': lambda df: df['min_temp'] < 0, 'color': '#87CEEB', 'symbol': 'circle', 'temp_col': 'min_temp'},
    {'name': 'IJsdag', 'condition': lambda df: df['max_temp'] < 0, 'color': '#00CED1', 'symbol': 'star', 'temp_col': 'max_temp'},
]

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78'
]


def insert_dataframe_to_mysql(df, table_name="knmi_minmaxtemp"):
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
        df = df.rename(columns={'TX': 'max_temp', 'TN': 'min_temp'})
        df['station'] = station
        return df[['date', 'max_temp', 'min_temp', 'station']]
    return None


def sync_data_from_api():
    """Fetch recent data from KNMI API and store in database."""
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


def calculate_stats(temp_data):
    """Calculate statistics for temperature data."""
    return {
        'Max. maximumtemp.': f"{temp_data['max_temp'].max():.1f}°C",
        'Min. maximumtemp.': f"{temp_data['max_temp'].min():.1f}°C",
        'Gem. maximumtemp.': f"{temp_data['max_temp'].mean():.1f}°C",
        'Max. minimumtemp.': f"{temp_data['min_temp'].max():.1f}°C",
        'Min. minimumtemp.': f"{temp_data['min_temp'].min():.1f}°C",
        'Gem. minimumtemp.': f"{temp_data['min_temp'].mean():.1f}°C",
        'Tropische dagen': int((temp_data['max_temp'] >= 30).sum()),
        'Zomerse dagen': int(((temp_data['max_temp'] >= 25) & (temp_data['max_temp'] < 30)).sum()),
        'Warme dagen': int(((temp_data['max_temp'] >= 20) & (temp_data['max_temp'] < 25)).sum()),
        'Vorstdagen': int((temp_data['min_temp'] < 0).sum()),
        'IJsdagen': int((temp_data['max_temp'] < 0).sum()),
    }


def create_station_figure(temp_data, station_name):
    """Create a figure for a single station with day type markers."""
    fig = go.Figure()

    # Max temperature line
    fig.add_trace(go.Scatter(
        x=temp_data['date'],
        y=temp_data['max_temp'],
        fill=None,
        mode='lines',
        line_color='rgba(255, 99, 71, 0.8)',
        name='Maximum Temperature'
    ))

    # Min temperature line with fill
    fig.add_trace(go.Scatter(
        x=temp_data['date'],
        y=temp_data['min_temp'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(65, 105, 225, 0.8)',
        name='Minimum Temperature'
    ))

    # Day type markers
    for day_type in DAY_TYPES:
        mask = day_type['condition'](temp_data)
        filtered = temp_data[mask]
        if len(filtered) > 0:
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
                hoverinfo='skip'
            ))

    # Add annotations for extremes
    max_temp_idx = temp_data['max_temp'].idxmax()
    min_temp_idx = temp_data['min_temp'].idxmin()

    fig.add_annotation(
        x=temp_data['date'][max_temp_idx],
        y=temp_data['max_temp'][max_temp_idx],
        text=f"Hoogste: {temp_data['max_temp'][max_temp_idx]:.1f}°C",
        showarrow=True, arrowhead=1
    )
    fig.add_annotation(
        x=temp_data['date'][min_temp_idx],
        y=temp_data['min_temp'][min_temp_idx],
        text=f"Laagste: {temp_data['min_temp'][min_temp_idx]:.1f}°C",
        showarrow=True, arrowhead=1
    )

    fig.update_layout(
        title=f'Temperatuur - {station_name}',
        xaxis_title="Datum",
        yaxis_title="Temperatuur (°C)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150)
    )
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def create_comparison_figure(all_station_data, temp_type='min_temp'):
    """Create a comparison figure for all stations."""
    fig = go.Figure()

    title = 'Minimumtemperatuur' if temp_type == 'min_temp' else 'Maximumtemperatuur'

    for i, (station_id, temp_data) in enumerate(all_station_data.items()):
        station_name = KNMI_STATIONS.get(station_id, str(station_id))
        fig.add_trace(go.Scatter(
            x=temp_data['date'],
            y=temp_data[temp_type],
            mode='lines',
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            name=station_name
        ))

    # Find global extremes
    global_min_val, global_max_val = float('inf'), float('-inf')
    global_min_station, global_max_station = None, None
    global_min_date, global_max_date = None, None

    for station_id, temp_data in all_station_data.items():
        station_name = KNMI_STATIONS.get(station_id, str(station_id))
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

    fig.add_annotation(
        x=global_max_date, y=global_max_val,
        text=f"Hoogste: {global_max_val:.1f}°C ({global_max_station})",
        showarrow=True, arrowhead=1
    )
    fig.add_annotation(
        x=global_min_date, y=global_min_val,
        text=f"Laagste: {global_min_val:.1f}°C ({global_min_station})",
        showarrow=True, arrowhead=1
    )

    fig.update_layout(
        title=f'{title} - Alle stations',
        xaxis_title="Datum",
        yaxis_title="Temperatuur (°C)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150)
    )
    fig.update_xaxes(rangeslider_visible=True)

    return fig


# Sync data on startup
sync_data_from_api()

# Load data from database
print("\nLoading data from database...")
ALL_STATION_DATA = get_data_from_database()
ALL_STATION_DATA = {k: v for k, v in ALL_STATION_DATA.items() if k in KNMI_STATIONS}
print(f"Loaded data for {len(ALL_STATION_DATA)} stations")

if ALL_STATION_DATA:
    all_dates = pd.concat([df['date'] for df in ALL_STATION_DATA.values()])
    print(f"Date range: {all_dates.min().strftime('%Y-%m-%d')} to {all_dates.max().strftime('%Y-%m-%d')}")

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Station options for dropdown
station_options = [{'label': name, 'value': sid} for sid, name in KNMI_STATIONS.items() if sid in ALL_STATION_DATA]

app.layout = dbc.Container([
    html.H1("KNMI Temperatuurdata", className="my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Weergave:"),
            dcc.Dropdown(
                id='view-type',
                options=[
                    {'label': 'Per station', 'value': 'station'},
                    {'label': 'Vergelijk min. temp.', 'value': 'compare_min'},
                    {'label': 'Vergelijk max. temp.', 'value': 'compare_max'},
                ],
                value='station',
                clearable=False
            )
        ], width=3),
        dbc.Col([
            html.Label("Weerstation:"),
            dcc.Dropdown(
                id='station-select',
                options=station_options,
                value=240,
                clearable=False
            )
        ], width=3, id='station-select-col'),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='temperature-graph')
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='stats-table')
        ])
    ], id='stats-row', className="mt-4"),

], fluid=True)


@callback(
    Output('station-select-col', 'style'),
    Input('view-type', 'value')
)
def toggle_station_dropdown(view_type):
    if view_type == 'station':
        return {'display': 'block'}
    return {'display': 'none'}


@callback(
    Output('stats-row', 'style'),
    Input('view-type', 'value')
)
def toggle_stats_table(view_type):
    if view_type == 'station':
        return {'display': 'block'}
    return {'display': 'none'}


@callback(
    Output('temperature-graph', 'figure'),
    Input('view-type', 'value'),
    Input('station-select', 'value')
)
def update_graph(view_type, station_id):
    if view_type == 'station':
        temp_data = ALL_STATION_DATA.get(station_id)
        if temp_data is not None:
            station_name = KNMI_STATIONS.get(station_id, str(station_id))
            return create_station_figure(temp_data, station_name)
    elif view_type == 'compare_min':
        return create_comparison_figure(ALL_STATION_DATA, 'min_temp')
    elif view_type == 'compare_max':
        return create_comparison_figure(ALL_STATION_DATA, 'max_temp')

    return go.Figure()


@callback(
    Output('stats-table', 'children'),
    Input('view-type', 'value'),
    Input('station-select', 'value'),
    Input('temperature-graph', 'relayoutData')
)
def update_stats(view_type, station_id, relayout_data):
    if view_type != 'station':
        return None

    temp_data = ALL_STATION_DATA.get(station_id)
    if temp_data is None:
        return None

    # Filter data based on zoom range
    filtered_data = temp_data.copy()
    if relayout_data:
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            start = pd.to_datetime(relayout_data['xaxis.range[0]'])
            end = pd.to_datetime(relayout_data['xaxis.range[1]'])
            filtered_data = temp_data[(temp_data['date'] >= start) & (temp_data['date'] <= end)]
        elif 'xaxis.range' in relayout_data:
            start = pd.to_datetime(relayout_data['xaxis.range'][0])
            end = pd.to_datetime(relayout_data['xaxis.range'][1])
            filtered_data = temp_data[(temp_data['date'] >= start) & (temp_data['date'] <= end)]

    if len(filtered_data) == 0:
        return html.P("Geen data in geselecteerde periode")

    stats = calculate_stats(filtered_data)

    # Create date range text
    date_range = f"{filtered_data['date'].min().strftime('%d-%m-%Y')} t/m {filtered_data['date'].max().strftime('%d-%m-%Y')} ({len(filtered_data)} dagen)"

    # Create table
    rows = []
    stat_items = list(stats.items())
    mid = (len(stat_items) + 1) // 2

    for i in range(mid):
        row_cells = [
            html.Td(stat_items[i][0], style={'fontWeight': 'bold'}),
            html.Td(str(stat_items[i][1]))
        ]
        if i + mid < len(stat_items):
            row_cells.extend([
                html.Td(stat_items[i + mid][0], style={'fontWeight': 'bold'}),
                html.Td(str(stat_items[i + mid][1]))
            ])
        rows.append(html.Tr(row_cells))

    return html.Div([
        html.H5(f"Statistieken: {date_range}"),
        dbc.Table(
            [html.Tbody(rows)],
            bordered=True,
            striped=True,
            hover=True,
            size='sm'
        )
    ])


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Dash app on http://localhost:8050")
    print("="*50 + "\n")
    app.run(debug=True, port=8050)
