import requests
import pandas as pd
import json

# Load configuration from config.json
with open("../config.json") as f:
    config = json.load(f)
#
#
# def fetch_foursquare_data_v3(lat, lon, radius=500):
#     """
#     Fetches nearby places from Foursquare V3 Places API.
#     Parameters:
#       lat (float): Latitude.
#       lon (float): Longitude.
#       radius (int): Search radius in meters (default 500).
#     Returns:
#       DataFrame: Contains place name, latitude, longitude, and a placeholder for popularity.
#     """
#     url = "https://api.foursquare.com/v3/places/search"
#     headers = {
#         "Accept": "application/json",
#         "Authorization": config["foursquare"]["api_key"]
#     }
#     params = {
#         "ll": f"{lat},{lon}",
#         "radius": radius,
#         "limit": 50
#     }
#
#     response = requests.get(url, headers=headers, params=params)
#     data = response.json()
#
#     # For debugging, you can print the raw response:
#     # print(json.dumps(data, indent=2))
#
#     results = data.get("results", [])
#     venues = []
#     for place in results:
#         name = place.get("name")
#         geocodes = place.get("geocodes", {}).get("main", {})
#         lat_place = geocodes.get("latitude")
#         lon_place = geocodes.get("longitude")
#         # V3 API might not provide a check-in count; you may use another available metric or leave as None.
#         venues.append({
#             "name": name,
#             "lat": lat_place,
#             "lon": lon_place,
#             "popularity": None  # Placeholder since the V3 API may not include this directly.
#         })
#     return pd.DataFrame(venues)
#
#
# # Example usage: Fetch data for Paris (latitude, longitude)
# if __name__ == "__main__":
#     df_ped = fetch_foursquare_data_v3(48.8566, 2.3522)
#     print(df_ped.head())
#
#
# import requests
# import json
#
#
# def fetch_tomtom_traffic_data(point, zoom, api_key):
#     """
#     Fetches traffic flow data from the TomTom Traffic API.
#
#     Parameters:
#       point (str): A string with "latitude,longitude" (e.g., "48.8566,2.3522").
#       zoom (int): A zoom level indicating the granularity (e.g., 10).
#       api_key (str): Your TomTom API key.
#
#     Returns:
#       dict: The JSON response from the API.
#     """
#     url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
#     params = {
#         "point": point,
#         "key": api_key
#     }
#     response = requests.get(url, params=params)
#     response.raise_for_status()
#     return response.json()
#
#
# if __name__ == "__main__":
#     # Example: Fetch traffic data for Paris center at a zoom level of 10
#     point = "48.8566,2.3522"
#     zoom = 10  # Adjust zoom for different resolution; lower zoom gives broader area.
#     tomtom_api_key = config["tomtom"]["api_key"] # Replace with your actual API key
#
#     try:
#         traffic_data = fetch_tomtom_traffic_data(point, zoom, tomtom_api_key)
#         print(json.dumps(traffic_data, indent=2))
#     except Exception as e:
#         print("Error fetching traffic data from TomTom:", e)
#
#
import requests
import json

def fetch_weather_data(lat, lon, api_key):
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

# Example usage:
if __name__ == "__main__":
    weather_api_key = config["openweather"]["api_key"]
    weather_data = fetch_weather_data(48.8566, 2.3522, weather_api_key)
    print(json.dumps(weather_data, indent=2))




# import osmnx as ox
#
# # Download the driving network for Paris
# G = ox.graph_from_place("Paris, France", network_type="drive")
# print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
#
# # Convert edges (road segments) to a GeoDataFrame
# gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
# print(gdf_edges.head())




