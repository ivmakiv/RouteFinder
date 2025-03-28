import math
import requests
import json
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import mapbox_vector_tile

# -----------------------------
# 1. Load Configuration
# -----------------------------
with open("../config.json") as f:
    config = json.load(f)

# -----------------------------
# 2. Get User Input: Start and Finish Points
# -----------------------------
start_lat = float(input("Enter start latitude: "))   # e.g., 48.8462
start_lon = float(input("Enter start longitude: "))  # e.g., 2.3460
finish_lat = float(input("Enter finish latitude: ")) # e.g., 48.8470
finish_lon = float(input("Enter finish longitude: "))# e.g., 2.3480

# -----------------------------
# 3. Check that Finish is within 2 km of Start (Haversine formula)
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

distance = haversine(start_lat, start_lon, finish_lat, finish_lon)
print(f"Distance between start and finish: {distance:.1f} meters")
if distance > 2000:
    print("Finish point is more than 2 km from the start point. Please choose closer points.")
    exit()

# -----------------------------
# 4. Define Center and Radius; Download Road Network & POIs
# -----------------------------
center_point = (start_lat, start_lon)
radius = 200  # in meters

# Download road network using graph_from_point
G = ox.graph_from_point(center_point, dist=radius, network_type="drive")
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
print("First few road segments:")
print(gdf_edges.head())

# Retrieve POIs using features_from_point with extended tags
tags = {'amenity': True, 'shop': True, 'tourism': True, 'public_transport': True}
gdf_pois = ox.features_from_point(center_point, tags, dist=radius)
print("First few POIs from OSM:")
print(gdf_pois.head())

# -----------------------------
# 5. Reproject Datasets to Metric CRS (UTM Zone 31N)
# -----------------------------
gdf_edges = gdf_edges.to_crs(epsg=32631)
gdf_pois = gdf_pois.to_crs(epsg=32631)

# -----------------------------
# 6. Buffer Road Segments (for Venue Counting)
# -----------------------------
buffer_distance = 30  # in meters for each road segment
gdf_edges["buffer"] = gdf_edges.geometry.buffer(buffer_distance)

# -----------------------------
# 7. Count Venues per Road Segment into Separate Columns
# -----------------------------
def count_in_buffer(buffer, key, poi_gdf):
    if key in poi_gdf.columns:
        return poi_gdf[poi_gdf[key].notnull() & poi_gdf.within(buffer)].shape[0]
    else:
        return 0

gdf_edges["shop_count"] = gdf_edges["buffer"].apply(lambda buff: count_in_buffer(buff, "shop", gdf_pois))
gdf_edges["tourism_count"] = gdf_edges["buffer"].apply(lambda buff: count_in_buffer(buff, "tourism", gdf_pois))
gdf_edges["amenity_count"] = gdf_edges["buffer"].apply(lambda buff: count_in_buffer(buff, "amenity", gdf_pois))
gdf_edges["public_transport_count"] = gdf_edges["buffer"].apply(lambda buff: count_in_buffer(buff, "public_transport", gdf_pois))
print("Road segments with separate venue counts:")
print(gdf_edges[["shop_count", "tourism_count", "amenity_count", "public_transport_count"]].head())

# -----------------------------
# 8. Define Functions for TomTom Vector Tile Requests (Traffic Data)
# -----------------------------
def latlon_to_tile(lat, lon, zoom):
    """
    Converts geographic coordinates (EPSG:4326) to tile x,y indices at a given zoom level.
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_tomtom_tile_pbf(zoom, x, y, api_key, flow_type="relative"):
    """
    Fetches a TomTom traffic vector tile in PBF format.
    flow_type: "relative" (for relative flow data) or "absolute" or "relative-delay"
    """
    url = f"https://api.tomtom.com/traffic/map/4/tile/flow/{flow_type}/{zoom}/{x}/{y}.pbf"
    params = {"key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.content

def compute_tile_traffic_score_for_segment(segment_geometry, zoom, api_key, flow_type="relative"):
    """
    Computes a traffic score for a road segment using TomTom's vector tile API.
    Steps:
      1. Compute the segment's centroid (in metric CRS).
      2. Convert the centroid to geographic coordinates (EPSG:4326).
      3. Convert that lat/lon into tile x,y indices.
      4. Request the corresponding tile (PBF).
      5. Decode the tile using mapbox-vector-tile and extract the "traffic_level" values.
      6. Return the average traffic_level as the traffic score.
         (For relative flow, values are between 0 and 1, where 1 indicates high congestion.)
    """
    centroid_metric = segment_geometry.centroid
    centroid_geo = gpd.GeoSeries([centroid_metric], crs="EPSG:32631").to_crs(epsg=4326).iloc[0]
    lat, lon = centroid_geo.y, centroid_geo.x
    tile_x, tile_y = latlon_to_tile(lat, lon, zoom)
    try:
        pbf_data = fetch_tomtom_tile_pbf(zoom, tile_x, tile_y, api_key, flow_type=flow_type)
    except Exception as e:
        print(f"Tile fetch error for segment centroid ({lat},{lon}) at tile ({tile_x},{tile_y}): {e}")
        return 0
    tile_decoded = mapbox_vector_tile.decode(pbf_data)
    traffic_layer = tile_decoded.get("Traffic flow")
    if traffic_layer is None:
        return 0
    features = traffic_layer.get("features", [])
    if not features:
        return 0
    levels = []
    for feat in features:
        props = feat.get("properties", {})
        level = props.get("traffic_level")
        if level is not None:
            levels.append(level)
    if levels:
        avg_level = sum(levels) / len(levels)
        return avg_level
    return 0

# -----------------------------
# 9. Compute Traffic Score for Each Road Segment Using Tile Requests
# -----------------------------
zoom_level = 12  # Adjust as needed (try higher zoom if needed)
traffic_scores = []
for idx, row in gdf_edges.iterrows():
    try:
        score = compute_tile_traffic_score_for_segment(row.geometry, zoom_level, config["tomtom"]["api_key"], flow_type="relative")
    except Exception as e:
        print(f"Error fetching traffic for segment {idx}: {e}")
        score = 0
    traffic_scores.append(score)
gdf_edges["traffic_score"] = traffic_scores
print("First few road segments with traffic scores:")
print(gdf_edges[["traffic_score"]].head())

# -----------------------------
# 10. Define Functions for Weather Data via OpenWeather
# -----------------------------
def fetch_weather_data(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def compute_weather_fields(weather_json):
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
# 11. Loop Over Road Segments to Fetch Weather Data
# -----------------------------
weather_fields_list = []
for idx, row in gdf_edges.iterrows():
    centroid_metric = row.geometry.centroid
    centroid_geo = gpd.GeoSeries([centroid_metric], crs="EPSG:32631").to_crs(epsg=4326).iloc[0]
    lat, lon = centroid_geo.y, centroid_geo.x
    try:
        weather_json = fetch_weather_data(lat, lon, config["openweather"]["api_key"])
        fields = compute_weather_fields(weather_json)
    except Exception as e:
        print(f"Error fetching weather for segment {idx}: {e}")
        fields = {"weather_main": None, "weather_desc": None, "temperature": None,
                  "feels_like": None, "pressure": None, "humidity": None,
                  "wind_speed": None, "wind_deg": None}
    weather_fields_list.append(fields)
df_weather = pd.DataFrame(weather_fields_list)
gdf_edges = gdf_edges.reset_index(drop=True)
gdf_edges = pd.concat([gdf_edges, df_weather], axis=1)
print("First few road segments with weather data:")
print(gdf_edges[["shop_count", "tourism_count", "amenity_count", "public_transport_count",
                  "traffic_score", "weather_main", "weather_desc", "temperature", "wind_speed"]].head())

# -----------------------------
# 12. Final Visualization of the Area
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 12))
gdf_edges.plot(ax=ax, color="black", linewidth=1, alpha=0.5)
gdf_edges["buffer"].plot(ax=ax, color="lightblue", edgecolor="blue", alpha=0.3)
gdf_pois.plot(ax=ax, color="green", markersize=5, alpha=0.8)
plt.title("Area: Road Network, Buffers, POIs, Traffic & Weather")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

print("Final road segments dataset (first 10 rows):")
print(gdf_edges[
      ["shop_count", "tourism_count", "amenity_count", "public_transport_count",
       "traffic_score", "weather_main", "weather_desc", "temperature", "feels_like", "pressure", "humidity", "wind_speed", "wind_deg"]].head(10))

# -----------------------------
# 13. Additional Visualizations
# -----------------------------

# (A) Traffic Visualization: color road segments by traffic_score
fig, ax = plt.subplots(figsize=(12, 12))
gdf_edges.plot(ax=ax, column="traffic_score", cmap="Reds", linewidth=1, legend=True)
plt.title("Traffic Visualization: Road Network Colored by Traffic Score")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

# (B) Weather Temperature Visualization: color road segments by temperature
fig, ax = plt.subplots(figsize=(12, 12))
gdf_edges.plot(ax=ax, column="temperature", cmap="coolwarm", linewidth=1, legend=True)
plt.title("Weather Visualization: Road Network Colored by Temperature")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

# -----------------------------
# 14. Write Final Dataset to Files
# -----------------------------
# Drop extra geometry columns (like 'buffer') to avoid issues.
gdf_final = gdf_edges.drop(columns=["buffer"])
# Write as GeoJSON:
gdf_final.to_file("final_road_segments.geojson", driver="GeoJSON")
print("Final dataset written to final_road_segments.geojson")
