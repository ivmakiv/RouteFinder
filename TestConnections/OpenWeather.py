import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests, json
from shapely.geometry import Point

# -----------------------------
# 1. Load Configuration and Define the Area
# -----------------------------
with open("../config.json") as f:
    config = json.load(f)

place_name = "5th arrondissement, Paris, France"

# -----------------------------
# 2. Download Road Network for the 5th Arrondissement
# -----------------------------
G = ox.graph_from_place(place_name, network_type="drive")
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
print("First few road segments:")
print(gdf_edges.head())

# -----------------------------
# 3. Reproject Road Network to a Metric CRS (UTM zone 31N)
# -----------------------------
gdf_edges = gdf_edges.to_crs(epsg=32631)


# -----------------------------
# 4. Define Function to Fetch Weather Data from OpenWeather
# -----------------------------
def fetch_weather_data(lat, lon, api_key):
    """
    Fetch weather data from OpenWeather for a given lat, lon.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def compute_weather_fields(weather_json):
    """
    Extracts key weather fields from the JSON response.
    """
    main_data = weather_json.get("main", {})
    weather_info = weather_json.get("weather", [{}])[0]
    wind_data = weather_json.get("wind", {})

    return {
        "weather_main": weather_info.get("main", ""),
        "weather_desc": weather_info.get("description", ""),
        "temperature": main_data.get("temp", None),
        "feels_like": main_data.get("feels_like", None),
        "pressure": main_data.get("pressure", None),
        "humidity": main_data.get("humidity", None),
        "wind_speed": wind_data.get("speed", None),
        "wind_deg": wind_data.get("deg", None)
    }


# -----------------------------
# 5. Loop Over Road Segments to Fetch Weather Data
# -----------------------------
weather_api_key = config["openweather"]["api_key"]

weather_fields_list = []

# Iterate over each road segment in the network
# Note: This might result in many API calls if there are many segments.
for idx, row in gdf_edges.iterrows():
    # Compute the centroid of the road segment in metric CRS.
    centroid_metric = row.geometry.centroid
    # Convert the centroid to geographic coordinates (EPSG:4326)
    centroid_geo = gpd.GeoSeries([centroid_metric], crs="EPSG:32631").to_crs(epsg=4326).iloc[0]
    lat, lon = centroid_geo.y, centroid_geo.x
    try:
        weather_json = fetch_weather_data(lat, lon, weather_api_key)
        fields = compute_weather_fields(weather_json)
    except Exception as e:
        print(f"Error fetching weather for segment {idx}: {e}")
        fields = {
            "weather_main": None,
            "weather_desc": None,
            "temperature": None,
            "feels_like": None,
            "pressure": None,
            "humidity": None,
            "wind_speed": None,
            "wind_deg": None
        }
    weather_fields_list.append(fields)

# Convert the list of weather field dictionaries to a DataFrame.
df_weather = pd.DataFrame(weather_fields_list)

# -----------------------------
# 6. Add Weather Data to the Road Network DataFrame
# -----------------------------
# Reset index so that the order is preserved.
gdf_edges = gdf_edges.reset_index(drop=True)
# Concatenate the weather DataFrame with the road segments GeoDataFrame.
gdf_edges = pd.concat([gdf_edges, df_weather], axis=1)

# -----------------------------
# 7. Visualize the Temperature Distribution Across Road Segments
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the road segments colored by temperature.
gdf_edges.plot(column="temperature", cmap="coolwarm", legend=True, ax=ax, linewidth=1, alpha=0.7)
plt.title("5th Arrondissement: Road Segments Colored by Temperature")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

# Optional: Print the first few rows of the final GeoDataFrame with weather fields.
print("First few road segments with weather data:")
print(gdf_edges[["weather_main", "weather_desc", "temperature", "feels_like", "wind_speed", "humidity"]])
