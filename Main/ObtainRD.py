import math
import requests
import json
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, LineString
import mapbox_vector_tile

with open("../config.json") as f:
    config = json.load(f)


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def latlon_to_tile(lat, lon, zoom):
    """Convert geographic coordinates (EPSG:4326) to tile x,y indices at a given zoom level."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_tomtom_tile_pbf(zoom, x, y, api_key, flow_type="relative"):
    """Fetch TomTom traffic vector tile in PBF format."""
    url = f"https://api.tomtom.com/traffic/map/4/tile/flow/{flow_type}/{zoom}/{x}/{y}.pbf"
    params = {"key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.content


def compute_tile_traffic_score_for_segment(segment_geometry, zoom, api_key, flow_type="relative"):
    """
    Compute a traffic score for a road segment using TomTom's vector tile API.
    Returns the average 'traffic_level' from the tile (for relative flow, a value between 0 and 1).
    """
    centroid_metric = segment_geometry.centroid
    centroid_geo = gpd.GeoSeries([centroid_metric], crs="EPSG:32631").to_crs(epsg=4326).iloc[0]
    lat, lon = centroid_geo.y, centroid_geo.x
    tile_x, tile_y = latlon_to_tile(lat, lon, zoom)
    try:
        pbf_data = fetch_tomtom_tile_pbf(zoom, tile_x, tile_y, api_key, flow_type=flow_type)
    except Exception as e:
        print(f"Tile fetch error for centroid ({lat},{lon}) at tile ({tile_x},{tile_y}): {e}")
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


def fetch_weather_data(lat, lon, api_key):
    """Fetch weather data from OpenWeather for a given lat/lon."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def compute_weather_fields(weather_json):
    """Extract key weather fields from OpenWeather JSON response."""
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


def count_in_buffer(buffer, key, poi_gdf):
    """Count number of POIs within buffer that have a non-null value for a given key."""
    if key in poi_gdf.columns:
        return poi_gdf[poi_gdf[key].notnull() & poi_gdf.within(buffer)].shape[0]
    else:
        return 0


def generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=2000):
    """
    Generate a road dataset with computed metrics from a given area defined by a start point and finish point.

    Input:
      start_lat, start_lon, finish_lat, finish_lon: coordinates for start and finish points.
      radius: radius (in meters) around the start point to download the road network and POIs.

    Returns:
      A GeoDataFrame with the final road segments dataset, including separate venue counts,
      traffic scores (from TomTom vector tiles), and weather data (from OpenWeather).
    """
    # Check that finish is within allowed distance
    dist = haversine(start_lat, start_lon, finish_lat, finish_lon)
    print(f"Distance between start and finish: {dist:.1f} meters")
    if dist > 2000:
        raise ValueError("Finish point is more than 2 km from the start point. Please choose closer points.")

    # Use start point as center
    center_point = (start_lat, start_lon)

    # Download road network and POIs
    G = ox.graph_from_point(center_point, dist=radius, network_type="drive")
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    print("First few road segments:")
    print(gdf_edges.head())

    tags = {'amenity': True, 'shop': True, 'tourism': True, 'public_transport': True}
    gdf_pois = ox.features_from_point(center_point, tags, dist=radius)
    print("First few POIs from OSM:")
    print(gdf_pois.head())

    # Reproject to UTM Zone 31N
    gdf_edges = gdf_edges.to_crs(epsg=32631)
    gdf_pois = gdf_pois.to_crs(epsg=32631)

    # Buffer road segments
    buffer_dist = 30  # meters
    gdf_edges["buffer"] = gdf_edges.geometry.buffer(buffer_dist)

    # Count venues per road segment for each type
    gdf_edges["shop_count"] = gdf_edges["buffer"].apply(lambda buff: count_in_buffer(buff, "shop", gdf_pois))
    gdf_edges["tourism_count"] = gdf_edges["buffer"].apply(lambda buff: count_in_buffer(buff, "tourism", gdf_pois))
    gdf_edges["amenity_count"] = gdf_edges["buffer"].apply(lambda buff: count_in_buffer(buff, "amenity", gdf_pois))
    gdf_edges["public_transport_count"] = gdf_edges["buffer"].apply(
        lambda buff: count_in_buffer(buff, "public_transport", gdf_pois))
    print("Road segments with separate venue counts:")
    print(gdf_edges[["shop_count", "tourism_count", "amenity_count", "public_transport_count"]].head())

    # Compute traffic scores using TomTom vector tile API
    zoom_level = 12  # adjust if needed
    traffic_scores = []
    for idx, row in gdf_edges.iterrows():
        try:
            score = compute_tile_traffic_score_for_segment(row.geometry, zoom_level, config["tomtom"]["api_key"],
                                                           flow_type="relative")
        except Exception as e:
            print(f"Error fetching traffic for segment {idx}: {e}")
            score = 0
        traffic_scores.append(score)
    gdf_edges["traffic_score"] = traffic_scores
    print("First few road segments with traffic scores:")
    print(gdf_edges[["traffic_score"]].head())

    # Fetch weather data for each road segment
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
    gdf_edges = gdf_edges.reset_index()
    gdf_edges = pd.concat([gdf_edges, df_weather], axis=1)
    print("First few road segments with weather data:")
    print(gdf_edges[["shop_count", "tourism_count", "amenity_count", "public_transport_count",
                     "traffic_score", "weather_main", "weather_desc", "temperature", "wind_speed"]].head())

    # Drop extra geometry columns (like 'buffer') to prepare final dataset
    gdf_final = gdf_edges.drop(columns=["buffer"])
    return gdf_final


if __name__ == '__main__':
    try:
        with open("../config.json") as f:
            config = json.load(f)
        start_lat = float(input("Enter start latitude: "))  # e.g., 48.8462
        start_lon = float(input("Enter start longitude: "))  # e.g., 2.3460
        finish_lat = float(input("Enter finish latitude: "))  # e.g., 48.8470
        finish_lon = float(input("Enter finish longitude: "))  # e.g., 2.3480
        final_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=200)
        # Visualize the final dataset (base visualization)
        fig, ax = plt.subplots(figsize=(12, 12))
        final_dataset.plot(ax=ax, color="black", linewidth=1, alpha=0.5)
        plt.title("Final Road Segments Dataset")
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.show()
        # Write the final dataset to file (GeoJSON, for example)
        final_dataset.to_file("final_road_segments.geojson", driver="GeoJSON")
        print("Final dataset written to final_road_segments.geojson")
    except Exception as e:
        print("Error generating road dataset:", e)
