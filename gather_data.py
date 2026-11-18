import argparse
import os
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd

# ==============================================================================
# CONFIGURATION (Align with Settings/settings_main.yaml)
# ==============================================================================
OUTPUT_DIR = "forecasting/Data"

FULL_START_DATE = "2022-01-01"
FULL_END_DATE = "2024-01-01"

DEBUG_START_DATE = "2023-01-01"
DEBUG_END_DATE = "2023-04-01"

# Seattle Open Data Socrata Endpoints
SEATTLE_911_URL = "https://data.seattle.gov/resource/kzjm-xkqj.json"
SEATTLE_EVENTS_URL = "https://data.seattle.gov/resource/dm95-f8w5.json"
SEATTLE_NEIGHBORHOODS_URL = (
    "https://data-seattlecitygis.opendata.arcgis.com/api/download/v1/items/"
    "b4a142f592e94d39a3bf787f3c112c1d/geojson?layers=0"
)

# Open-Meteo Archive API (Seattle coordinates, free, no API key required)
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
SEATTLE_LAT = 47.6062
SEATTLE_LON = -122.3321

WEATHER_FEATURE_COLS = [
    "Min_Temp_F",
    "Avg_Temp_F",
    "Max_Temp_F",
    "Min_Wind_Speed_mph",
    "Avg_Wind_Speed_mph",
    "Max_Wind_Speed_mph",
    "Min_Humidity_Percent",
    "Avg_Humidity_Percent",
    "Max_Humidity_Percent",
    "Min_Dew_Point_F",
    "Avg_Dew_Point_F",
    "Max_Dew_Point_F",
    "Min_Pressure_Hg",
    "Avg_Pressure_Hg",
    "Max_Pressure_Hg",
    "Total_Precipitation_intensity",
]


def fetch_socrata_paginated(base_url, where_clause, page_size=50000):
    """Fetches all matching rows from a Socrata SODA endpoint using $limit/$offset."""
    all_rows = []
    offset = 0
    while True:
        params = {
            "$where": where_clause,
            "$limit": page_size,
            "$offset": offset,
            "$order": ":id",
        }
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        print(f"  Fetched {len(all_rows)} rows...")
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(0.25)
    return pd.DataFrame(all_rows)


def fetch_911_calls(start_date, end_date, out_path):
    """Fetches Seattle 911 calls and formats columns/timestamps for event_handler.py."""
    print(f"[1/4] Fetching Seattle 911 Calls ({start_date} to {end_date})...")
    where = f"datetime >= '{start_date}T00:00:00' AND datetime < '{end_date}T00:00:00'"
    df = fetch_socrata_paginated(SEATTLE_911_URL, where)

    if df.empty:
        raise RuntimeError(f"No 911 calls returned between {start_date} and {end_date}.")

    df["dt_parsed"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["Longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df = df.dropna(subset=["dt_parsed", "Latitude", "Longitude"]).copy()

    formatted = pd.DataFrame({
        "Address": df.get("address", "UNKNOWN").fillna("UNKNOWN").astype(str),
        "Type": df.get("type", "Aid Response").fillna("Aid Response").astype(str),
        "Datetime": df["dt_parsed"].dt.strftime("%m/%d/%Y %I:%M:%S %p"),
        "Latitude": df["Latitude"],
        "Longitude": df["Longitude"],
        "Report Location": df.apply(
            lambda r: f"POINT ({r['Longitude']} {r['Latitude']})", axis=1
        ),
        "Incident Number": df.get("incident_number", df.index.astype(str)).astype(str),
    })

    formatted.to_csv(out_path, index=False)
    print(f"  Saved {len(formatted)} clean 911 records -> {out_path}")


def fetch_weather(start_date, end_date, out_path):
    """
    Fetches daily Seattle weather from Open-Meteo and writes semicolon-delimited CSV
    matching WeatherHandler's exact Year/Month/Day and COLUMN_LABELS_MAP schema.
    """
    print(f"[2/4] Fetching Daily Weather ({start_date} to {end_date})...")

    # Buffer by +/- 14 days so look-back intervals and boundary merges never produce NaNs
    buffered_start = (pd.to_datetime(start_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    buffered_end = (pd.to_datetime(end_date) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    params = {
        "latitude": SEATTLE_LAT,
        "longitude": SEATTLE_LON,
        "start_date": buffered_start,
        "end_date": buffered_end,
        "daily": [
            "temperature_2m_min",
            "temperature_2m_mean",
            "temperature_2m_max",
            "wind_speed_10m_min",
            "wind_speed_10m_mean",
            "wind_speed_10m_max",
            "relative_humidity_2m_min",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_max",
            "dew_point_2m_min",
            "dew_point_2m_mean",
            "dew_point_2m_max",
            "surface_pressure_min",
            "surface_pressure_mean",
            "surface_pressure_max",
            "precipitation_sum",
        ],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles",
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    daily = resp.json().get("daily", {})

    dt = pd.to_datetime(daily["time"])
    hpa_to_inhg = 0.02953  # Open-Meteo reports surface pressure in hPa; convert to inHg

    weather_df = pd.DataFrame({
        "Year": dt.year.astype(int),
        "Month": dt.month.astype(int),
        "Day": dt.day.astype(int),
        "Min_Temp_F": daily["temperature_2m_min"],
        "Avg_Temp_F": daily["temperature_2m_mean"],
        "Max_Temp_F": daily["temperature_2m_max"],
        "Min_Wind_Speed_mph": daily["wind_speed_10m_min"],
        "Avg_Wind_Speed_mph": daily["wind_speed_10m_mean"],
        "Max_Wind_Speed_mph": daily["wind_speed_10m_max"],
        "Min_Humidity_Percent": daily["relative_humidity_2m_min"],
        "Avg_Humidity_Percent": daily["relative_humidity_2m_mean"],
        "Max_Humidity_Percent": daily["relative_humidity_2m_max"],
        "Min_Dew_Point_F": daily["dew_point_2m_min"],
        "Avg_Dew_Point_F": daily["dew_point_2m_mean"],
        "Max_Dew_Point_F": daily["dew_point_2m_max"],
        "Min_Pressure_Hg": pd.to_numeric(daily["surface_pressure_min"], errors="coerce") * hpa_to_inhg,
        "Avg_Pressure_Hg": pd.to_numeric(daily["surface_pressure_mean"], errors="coerce") * hpa_to_inhg,
        "Max_Pressure_Hg": pd.to_numeric(daily["surface_pressure_max"], errors="coerce") * hpa_to_inhg,
        "Total_Precipitation_intensity": daily["precipitation_sum"],
    })

    # Fill any missing values cleanly
    weather_df[WEATHER_FEATURE_COLS] = (
        weather_df[WEATHER_FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    # Guard against range_f == 0 in WeatherHandler.__preprocess_weather_data_seattle
    # which divides (col - min_f) / (max_f - min_f)
    for col in WEATHER_FEATURE_COLS:
        if weather_df[col].max() - weather_df[col].min() == 0:
            weather_df.iloc[0, weather_df.columns.get_loc(col)] += 1e-6

    # WeatherHandler reads with sep=";"
    weather_df.to_csv(out_path, sep=";", index=False)
    print(f"  Saved {len(weather_df)} daily weather rows -> {out_path}")


def fetch_special_events(start_date, end_date, out_path):
    """Fetches Seattle Special Events Permits and enforces clean dates/schemas."""
    print(f"[3/4] Fetching Special Events ({start_date} to {end_date})...")
    where = (
        f"event_start_date >= '{start_date}T00:00:00' "
        f"AND event_start_date < '{end_date}T00:00:00'"
    )
    try:
        df = fetch_socrata_paginated(SEATTLE_EVENTS_URL, where)
    except Exception as exc:
        print(f"  Warning: Could not reach Special Events API ({exc}); creating empty schema.")
        df = pd.DataFrame()

    if df.empty:
        events_df = pd.DataFrame(
            columns=["Event Name", "Event Start Date", "Event End Date", "Neighborhood", "Attendance", "Status"]
        )
    else:
        start_dt = pd.to_datetime(df.get("event_start_date"), errors="coerce")
        end_dt = pd.to_datetime(df.get("event_end_date", df.get("event_start_date")), errors="coerce").fillna(start_dt)
                # Safely extract columns from df as Series
        name_col = df["name_of_event"] if "name_of_event" in df.columns else (
            df["event_name"] if "event_name" in df.columns else pd.Series("Special Event", index=df.index)
        )
        hood_col = df["event_location_neighborhood"] if "event_location_neighborhood" in df.columns else (
            df["neighborhood"] if "neighborhood" in df.columns else pd.Series("Downtown", index=df.index)
        )
        status_col = df["permit_status"] if "permit_status" in df.columns else (
            df["status"] if "status" in df.columns else pd.Series("Complete", index=df.index)
        )
        att_col = df["attendance"] if "attendance" in df.columns else pd.Series(0.0, index=df.index)
        events_df = pd.DataFrame({
            "Event Name": name_col.fillna("Special Event").astype(str),
            "Event Start Date": start_dt.dt.strftime("%Y-%m-%d"),
            "Event End Date": end_dt.dt.strftime("%Y-%m-%d"),
            "Neighborhood": hood_col.fillna("Downtown").astype(str),
            "Attendance": pd.to_numeric(att_col, errors="coerce").fillna(0.0),
            "Status": status_col.fillna("Complete").astype(str),
        }).dropna(subset=["Event Start Date", "Event End Date"])
    events_df.to_csv(out_path, index=False)
    print(f"  Saved {len(events_df)} special event rows -> {out_path}")


def fetch_neighborhoods(out_path):
    """Downloads Seattle Neighborhoods GeoJSON and explodes MultiPolygons into single Polygons."""
    print("[4/4] Fetching & Normalizing Seattle Neighborhoods GeoJSON...")
    gdf = gpd.read_file(SEATTLE_NEIGHBORHOODS_URL)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"].copy()

    if "S_HOOD" not in gdf.columns:
        name_candidates = [c for c in gdf.columns if "HOOD" in c.upper() or "NAME" in c.upper()]
        if name_candidates:
            gdf["S_HOOD"] = gdf[name_candidates[0]].fillna("Unknown").astype(str)
        else:
            gdf["S_HOOD"] = [f"Neighborhood_{i}" for i in range(len(gdf))]

    gdf.to_file(out_path, driver="GeoJSON")
    print(f"  Saved {len(gdf)} single-polygon neighborhood geometries -> {out_path}")


def audit_outputs(calls_path, weather_path, events_path, geojson_path):
    """Verifies non-empty outputs, exact weather schema, and zero NaNs."""
    print("\n--- Auditing Generated Files ---")
    calls = pd.read_csv(calls_path)
    weather = pd.read_csv(weather_path, sep=";")
    events = pd.read_csv(events_path)
    hoods = gpd.read_file(geojson_path)

    expected_weather_cols = {"Year", "Month", "Day", *WEATHER_FEATURE_COLS}
    missing_weather = expected_weather_cols - set(weather.columns)
    assert not missing_weather, f"Missing weather columns: {missing_weather}"
    assert not weather.isna().any().any(), "NaNs found in Weather_Seattle.csv!"
    for col in WEATHER_FEATURE_COLS:
        assert weather[col].max() - weather[col].min() > 0, f"Zero range in {col}!"

    assert len(calls) > 0, "911 calls CSV is empty!"
    assert not calls[["Latitude", "Longitude"]].isna().any().any(), "NaNs found in 911 coordinates!"
    assert (hoods.geometry.geom_type == "Polygon").all(), "MultiPolygon still present in GeoJSON!"

    print(f"  911 Calls:     {len(calls):,} rows | Range: {calls['Datetime'].iloc[0]} -> {calls['Datetime'].iloc[-1]}")
    print(f"  Weather:       {len(weather):,} days | Columns: {len(weather.columns)} | NaNs: 0")
    print(f"  Events:        {len(events):,} rows | NaNs: {events.isna().sum().sum()}")
    print(f"  Neighborhoods: {len(hoods):,} Polygons")
    print("All checks passed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Fetch and align municipal datasets for tumBAIS.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Use 3-month debug window ({DEBUG_START_DATE} to {DEBUG_END_DATE}) instead of full 2-year range.",
    )
    parser.add_argument("--start", type=str, default=None, help="Custom start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Custom end date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.start and args.end:
        start_date, end_date = args.start, args.end
    elif args.debug:
        start_date, end_date = DEBUG_START_DATE, DEBUG_END_DATE
    else:
        start_date, end_date = FULL_START_DATE, FULL_END_DATE

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    calls_path = os.path.join(OUTPUT_DIR, "Seattle_Real_Time_Fire_911_Calls.csv")
    weather_path = os.path.join(OUTPUT_DIR, "Weather_Seattle.csv")
    events_path = os.path.join(OUTPUT_DIR, "Special_Events_Permits_Seattle.csv")
    geojson_path = os.path.join(OUTPUT_DIR, "Seattle_Neighborhoods.geojson")

    fetch_911_calls(start_date, end_date, calls_path)
    fetch_weather(start_date, end_date, weather_path)
    fetch_special_events(start_date, end_date, events_path)
    fetch_neighborhoods(geojson_path)
    audit_outputs(calls_path, weather_path, events_path, geojson_path)


if __name__ == "__main__":
    main()