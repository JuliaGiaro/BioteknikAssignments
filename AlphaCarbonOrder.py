import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to find the nearest neighbor for each point
def find_nearest_neighbors(coordinates):
    connections = []
    n = len(coordinates)

    for i in range(n):
        min_distance = float('inf')
        nearest_point = None

        for j in range(n):
            if i != j:
                distance = calculate_distance(coordinates[i, 1:], coordinates[j, 1:])
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = j

        if nearest_point is not None:
            connections.append((int(coordinates[i, 0]), int(coordinates[nearest_point, 0])))

    return connections

# Function to perform depth-first search traversal
def dfs(graph, start, visited, order):
    visited[start] = True
    order.append(start)

    for neighbor in graph[start]:
        if not visited[neighbor]:
            dfs(graph, neighbor, visited, order)

# Function to find and print the order of alpha carbons
def print_order(connections, coordinates):
    graph = {i: [] for i in range(1, 11)}

    for connection in connections:
        graph[connection[0]].append(connection[1])


    visited = {i: False for i in range(1, 11)}
    order = []

    # Find the alpha carbon with the lowest X and Y coordinates as the starting point
    starting_point = find_starting_point(coordinates)
    dfs(graph, starting_point, visited, order)

    print(order)

# Function to find the alpha carbon with the lowest X and Y coordinates
def find_starting_point(coordinates):
    min_xy_index = np.argmin(coordinates[:, 1] + coordinates[:, 2])
    return int(coordinates[min_xy_index, 0])

def connect_clusters(coordinates, threshold_distance):
    connections = find_nearest_neighbors(coordinates)
    clusters = []

    for connection in connections:
        connected = False

        for cluster in clusters:
            for point in cluster:
                distance = calculate_distance(coordinates[connection[0] - 1, 1:], coordinates[point - 1, 1:])
                if distance < threshold_distance:
                    cluster.append(connection[1])
                    connected = True
                    break

        if not connected:
            clusters.append([connection[0], connection[1]])

    # Connect clusters into one directed graph
    directed_connections = []

    for i in range(len(clusters) - 1):
        current_cluster = clusters[i]
        next_cluster = clusters[i + 1]

        # Find nearest endpoint in the next cluster from the current cluster's endpoint
        current_endpoint = current_cluster[-1]
        min_distance = float('inf')
        nearest_point = None

        for point in next_cluster:
            distance = calculate_distance(coordinates[current_endpoint - 1, 1:], coordinates[point - 1, 1:])
            if distance < min_distance:
                min_distance = distance
                nearest_point = point

        # Add directed connection
        directed_connections.append((current_endpoint, nearest_point))

    return directed_connections


# Function to plot 3D coordinates with color-coded points, annotations, and connections
def plot_coordinates_with_connections(coordinates, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = coordinates[:, 1]
    y = coordinates[:, 2]
    z = coordinates[:, 3]

    # Color-coded points based on numbers
    colors = coordinates[:, 0]

    # Scatter plot with color-coded points
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', marker='o', s=50, label='Alpha Carbons')

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Alpha Carbon Number')

    # Annotate points with alpha carbon numbers
    for i, txt in enumerate(coordinates[:, 0]):
        ax.text(x[i], y[i], z[i], str(int(txt)), color='black')

    # Find and plot connections between alpha carbons
    connections = find_nearest_neighbors(coordinates)

    # Sort connections based on the X-coordinate


    # Print connected alpha carbons


    for connection in connections:
        ax.plot([x[connection[0] - 1], x[connection[1] - 1]],
                [y[connection[0] - 1], y[connection[1] - 1]],
                [z[connection[0] - 1], z[connection[1] - 1]], color='gray', linestyle='--')
    threshold_distance = 4.0 # Adjust this threshold distance as needed
    clusters = connect_clusters(coordinates, threshold_distance)

    # Plot connections between clusters
    for cluster in clusters:
        for i in range(len(cluster) - 1):
            ax.plot([x[cluster[i] - 1], x[cluster[i + 1] - 1]],
                    [y[cluster[i] - 1], y[cluster[i + 1] - 1]],
                    [z[cluster[i] - 1], z[cluster[i + 1] - 1]], color='gray', linestyle='--')
    print_order(connections, coordinates)
    # Label axes
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')

    # Set title
    ax.set_title(title)

    # Show plot
    plt.show()

# Coordinates for the first file
file1_coordinates = np.array([
    [1, 5.598, 5.767, 11.082],
    [2, 8.496, 4.609, 8.837],
    [3, 13.660, 10.707, 9.787],
    [4, 10.646, 8.991, 11.408],
    [5, 8.912, 2.083, 13.258],
    [6, 9.448, 9.034, 15.012],
    [7, 8.673, 5.314, 15.279],
    [8, 16.967, 12.784, 4.338],
    [9, 13.856, 11.469, 6.066],
    [10, 5.145, 2.209, 12.453]
])

# Coordinates for the second file
file2_coordinates = np.array([
    [1, 13.257, 10.745, 15.081],
    [2, 13.512, 5.395, 12.878],
    [3, 13.564, 11.573, 18.836],
    [4, 15.445, 7.667, 15.246],
    [5, 17.334, 10.956, 18.691],
    [6, 17.924, 13.421, 15.877],
    [7, 18.504, 12.312, 12.298],
    [8, 21.452, 16.969, 6.513],
    [9, 21.936, 12.911, 10.809],
    [10, 22.019, 13.242, 7.020]
])

# Plot coordinates for the first file with color-coded points, annotations, and connections
plot_coordinates_with_connections(file1_coordinates, 'Alpha Carbons - File 1')

# Plot coordinates for the second file with color-coded points, annotations, and connections
plot_coordinates_with_connections(file2_coordinates, 'Alpha Carbons - File 2')
