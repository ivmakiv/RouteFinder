# Main.py

from ObtainRD import generate_road_dataset
from ConvertToGraph import dataset_to_graph
from RouteFinder import find_route
from Visualizations import visualize_route
from pyproj import Transformer

def main():
    print("Input for route finding:")
    start_lat = float(input("Enter start latitude (EPSG:4326): "))    # e.g., 48.8600
    start_lon = float(input("Enter start longitude (EPSG:4326): "))   # e.g., 2.3522
    finish_lat = float(input("Enter finish latitude (EPSG:4326): "))  # e.g., 48.8550
    finish_lon = float(input("Enter finish longitude (EPSG:4326): ")) # e.g., 2.3700
    max_length = float(input("Enter maximum route length in meters: "))  # e.g., 3000

    # 1. Generate the road dataset using ObtainRD.py
    road_dataset = generate_road_dataset(start_lat, start_lon, finish_lat, finish_lon, radius=800)
    print("Road dataset generated.")

    # 2. Convert the road dataset into a graph using ConvertToGraph.py
    G = dataset_to_graph(road_dataset)
    print("Graph created with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges.")

    # 3. Call the find_route() function from RouteFinder.py, passing the graph and user coordinates.
    best_route, total_length, status = find_route(G, road_dataset, start_lat, start_lon, finish_lat, finish_lon, max_length)
    print(status)
    if best_route is None:
        return
    print("Best route (node sequence):", best_route)
    print("Total route length: {:.1f} m".format(total_length))

    # 4. Visualize the route using the visualize_route() function from Visualization.py
    visualize_route(G, best_route, start_lon, start_lat, finish_lon, finish_lat)

if __name__ == '__main__':
    main()
