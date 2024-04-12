import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
import os

sys.setrecursionlimit(10000)
maze_number = 0

# def generate_maze(width, height):
#     # Initialize the maze grid as a solid block (walls everywhere)
#     maze = [[0] * (2 * width + 1) for _ in range(2 * height + 1)]

#     # Carve out the internal area for the maze paths
#     for x in range(width):
#         for y in range(height):
#             maze[2 * y + 1][2 * x + 1] = 1

#     stack = [(0, 0)]
#     visited = {(0, 0)}

#     while stack:
#         x, y = stack[-1]
#         neighbors = []
#         for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
#             if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
#                 neighbors.append((nx, ny))

#         if neighbors:
#             nx, ny = random.choice(neighbors)
#             # Remove the wall between the current cell and the chosen cell
#             maze[2 * y + 1 + (ny - y)][2 * x + 1 + (nx - x)] = 0
#             stack.append((nx, ny))
#             visited.add((nx, ny))
#         else:
#             stack.pop()

#     # Randomize the start and end points on the boundary
#     # Ensure start and end are on different boundaries if chosen from top/bottom or left/right
#     sides = ['top', 'right', 'bottom', 'left']
#     anakin_skywalker = random.randint(0, 3)
#     start_side = sides[anakin_skywalker]  # Choose two distinct sides
#     end_side = sides[(anakin_skywalker + 2) % 4]

#     # Start point
#     if start_side == 'top':
#         start = (-1, -1)
#         while start[0] == -1 or maze[1][start[1]] == 0:
#             start = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))
#     elif start_side == 'bottom':
#         start = (-1, -1)
#         while start[0] == -1 or maze[height - 2][start[1]] == 0:
#             start = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))
#     elif start_side == 'left':
#         start = (-1, -1)
#         while start[0] == -1 or maze[start[0]][1] == 0:
#             start = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))
#     else:  # 'right'
#         start = (-1, -1)
#         while start[0] == -1 or maze[start[0]][width - 2] == 0:
#             start = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))

#     # End point
#     if end_side == 'top':
#         end = (-1, -1)
#         while end[0] == -1 or maze[1][end[1]] == 0:
#             end = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))
#     elif end_side == 'bottom':
#         end = (-1, -1)
#         while end[0] == -1 or maze[height - 2][end[1]] == 0:
#             end = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))
#     elif end_side == 'left':
#         end = (-1, -1)
#         while end[0] == -1 or maze[end[0]][1] == 0:
#             end = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))
#     else:  # 'right'
#         end = (-1, -1)
#         while end[0] == -1 or maze[end[0]][width - 2] == 0:
#             end = (random.choice(range(1, height - 1)), random.choice(range(1, width - 1)))

#     # Open the start and end points
#     maze[start[0]][start[1]] = 1
#     maze[end[0]][end[1]] = 1

#     return maze, start, end

import random

def generate_maze(width, height):
    # Initialize the maze grid as a solid block (walls everywhere)
    maze = [[0] * (2 * width + 1) for _ in range(2 * height + 1)]

    # Carve out the internal area for the maze paths
    for x in range(width):
        for y in range(height):
            maze[2 * y + 1][2 * x + 1] = 1

    stack = [(0, 0)]
    visited = {(0, 0)}

    while stack:
        x, y = stack[-1]
        neighbors = []
        for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            # Remove the wall between the current cell and the chosen cell
            maze[2 * y + 1 + (ny - y)][2 * x + 1 + (nx - x)] = 1
            stack.append((nx, ny))
            visited.add((nx, ny))
        else:
            stack.pop()

    # Randomize the start and end points on the boundary
    # Ensure start and end are on different boundaries if chosen from top/bottom or left/right
    sides = ['top', 'right', 'bottom', 'left']
    start_side = random.choice(sides)
    end_side = sides[(sides.index(start_side) + 2) % 4]
    # end_side = random.choice([side for side in sides if side != start_side])

    # Start point
    if start_side == 'top':
        start = (1, random.randint(1, width * 2 - 1))
    elif start_side == 'bottom':
        start = (height * 2 - 1, random.randint(1, width * 2 - 1))
    elif start_side == 'left':
        start = (random.randint(1, height * 2 - 1), 1)
    else:  # 'right'
        start = (random.randint(1, height * 2 - 1), width * 2 - 1)

    # End point
    if end_side == 'top':
        end = (1, random.randint(1, width * 2 - 1))
    elif end_side == 'bottom':
        end = (height * 2 - 1, random.randint(1, width * 2 - 1))
    elif end_side == 'left':
        end = (random.randint(1, height * 2 - 1), 1)
    else:  # 'right'
        end = (random.randint(1, height * 2 - 1), width * 2 - 1)

    # Open the start and end points
    maze[start[0]][start[1]] = 1
    maze[end[0]][end[1]] = 1

    return maze, start, end

def visualize_maze(maze, start, end):
    maze_np = np.array(maze)
    # Color code the start and end for visualization
    maze_np[start[0]][start[1]] = 2  # Start point
    maze_np[end[0]][end[1]] = 3  # End point

    plt.figure(figsize=(10, 5))
    # Custom color map: walls in black, path in white, start in green, end in red
    cmap = ListedColormap(['black', 'white', 'green', 'red'])
    bounds = [0,1,2,3,4]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(maze_np, cmap=cmap, norm=norm)
    plt.xticks([]), plt.yticks([])
    plt.show()

def find_path_dfs(maze, start, end):
    # Convert maze to a more convenient format for pathfinding
    height = len(maze)
    width = len(maze[0])
    visited = [[False for _ in range(width)] for _ in range(height)]
    path = []
    
    def dfs(x, y):
        if x == end[0] and y == end[1]:
            path.append((x, y))
            return True
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and not visited[nx][ny] and maze[nx][ny] != 0:
                visited[nx][ny] = True
                if dfs(nx, ny):
                    path.append((x, y))
                    return True
        return False

    visited[start[0]][start[1]] = True
    dfs(start[0], start[1])
    return path[::-1]  # Return reversed path to start from the beginning

def dfs_search(maze, start, end):
    stack = [(start, [start])]  # Stack of (position, path) tuples
    visited = set()
    while stack:
        (x, y), path = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        if (x, y) == end:
            return path
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Move 1 cells at a time
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
                stack.append(((nx, ny), path + [(nx, ny)]))
    return None  # No path found

def visualize_maze_with_path(maze, start, end, path):
    maze_np = np.array(maze)
    for x, y in path:
        maze_np[x][y] = 4  # Mark the path
    
    # Assign special values for the start and end for visualization
    maze_np[start[0]][start[1]] = 2
    maze_np[end[0]][end[1]] = 3

    plt.figure(figsize=(10, 5))
    cmap = ListedColormap(['black', 'white', 'green', 'red', 'blue'])
    bounds = [0,1,2,3,4,5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(maze_np, cmap=cmap, norm=norm)
    plt.xticks([]), plt.yticks([])
    plt.show()

def construct_datapoint(maze, dir, reward=0.1):
    mazevals = []
    data = ""
    for r in maze:
        for c in r:
            mazevals.append(c)
    data += str(mazevals)
    data += '\n'
    data += dir
    data += '\n'
    data += str(reward)
    data += '\n'
    return data
    
def export_dataset(folder, dataset, n):
    with open(f"{folder}/maze{n}.txt", 'w') as file:
        for i in range(len(dataset)):
            file.write(str(dataset[i]))

def generate_dataset(maze, path):
    print(path)
    dataset = []
    maze[path[0][0]][path[0][1]] = 3
    maze[path[len(path) - 1][0]][path[len(path) - 1][1]] = 5
    for i in range(1, len(path)):
        if i != len(path) - 1:
            path_ = (path[i], path[i + 1])
            foobar = maze[path_[0][0]][path_[0][1]]
            maze[path_[0][0]][path_[0][1]] = 4
            maze[path_[0][0]][path_[0][1]] = foobar
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        if dx == 1: dir = "D"
        elif dx == -1: dir = "U"
        if dy == 1: dir = "R"
        elif dy == -1: dir = "L"

        data = construct_datapoint(maze, dir, reward=(i / len(path)))
        dataset.append(data)
    return dataset

from queue import PriorityQueue

def heuristic(a, b):
    """Calculate the Manhattan distance between two points"""
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def a_star_search(maze, start, end):
    """Perform A* search to find the path from start to end"""
    start = (start[0], start[1])  # Ensure start is a tuple
    end = (end[0], end[1])  # Ensure end is a tuple
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    while not queue.empty():
        current = queue.get()[1]

        if current == end:
            break

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next = (current[0] + dx, current[1] + dy)
            if 0 <= next[0] < len(maze) and 0 <= next[1] < len(maze[0]) and maze[next[0]][next[1]] == 1:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(end, next)
                    queue.put((priority, next))
                    came_from[next] = current
    # Reconstruct path
    current = end
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# Adjust the path to fit the maze's structure, considering walls
def adjust_path_for_maze(path):
    adjusted_path = []
    for point in path:
        adjusted_path.append((point[0]*2+1, point[1]*2+1))
    return adjusted_path

# visualize_maze(maze, start, end)
# path = find_path_dfs(maze, start, end)
# path = dfs_search(maze, start, end)

# clears maze folder
# for filename in os.listdir("mazes"):
#     if "small" not in filename and "medium" not in filename and "large" not in filename:
#         os.remove(f"mazes/{filename}")

# if __name__=="main":
# for j in range(4):
for n in range(1000):
    width, height = 5, 5
    maze, start, end = generate_maze(width, height)
    path = a_star_search(maze, (start[0], start[1]), (end[0], end[1]))
    dataset = generate_dataset(maze, path)
    export_dataset("dataset_of_1000_small", dataset, n)
    # visualize_maze_with_path(maze, start, end, path)
