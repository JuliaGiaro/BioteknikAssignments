import numpy as np
import networkx as nx

#I decided to traverse trough the nodes x y ans z coordinates and deciding the shortest path between them
#the best method to do that was to take the weight as the distance and then check which one is the closest while keeping
#track of which ones you have visited, lastly print them out.
#before I started with these methods I wanted to visualize the points in a 3d graph (not shown) and wanted to include it as well in
#my code but decided it was redundant (also it casued wierd behaviour with the output)

#the only things you need for this program to run is numpy and networkx while using python. (I used the python version 3.12)
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def find_shortest_chain(coordinates, target_distance=3.8):
    n = len(coordinates)
    graph = nx.Graph()

    for i in range(n):
        for j in range(i + 1, n):
            distance = calculate_distance(coordinates[i, 1:], coordinates[j, 1:])
            graph.add_edge(int(coordinates[i, 0]), int(coordinates[j, 0]), weight=distance)

    best_order = None
    best_total_distance = float('inf')

    for start_node in range(1, n + 1):
        order = [start_node]
        visited = {start_node}

        total_distance = 0.0

        while len(visited) < n:
            current_node = order[-1]

            
            neighbors = [(neighbor, data['weight']) for neighbor, data in graph[current_node].items() if
                         neighbor not in visited]
            if neighbors:
                next_node, weight = min(neighbors, key=lambda x: x[1])
                order.append(next_node)
                visited.add(next_node)
                total_distance += weight

        if abs(total_distance - target_distance * (n - 1)) < abs(best_total_distance - target_distance * (n - 1)):
            best_order = order
            best_total_distance = total_distance

    return best_order


test_coordinates = np.array([
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


def find_and_print_order(coordinates):
    order = find_shortest_chain(coordinates)
    print("Order of alpha carbons:", order)

print("data file:")
find_and_print_order(file1_coordinates)

print("test file:")
find_and_print_order(test_coordinates)