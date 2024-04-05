import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
import os

sys.setrecursionlimit(10000)
maze_number = 0

def generate_maze(width, height):
    # Initialize the maze grid as a solid block (walls everywhere)
    maze = [[1] * (2 * width + 1) for _ in range(2 * height + 1)]

    # Carve out the internal area for the maze paths
    for x in range(width):
        for y in range(height):
            maze[2 * y + 1][2 * x + 1] = 0

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
            maze[2 * y + 1 + (ny - y)][2 * x + 1 + (nx - x)] = 0
            stack.append((nx, ny))
            visited.add((nx, ny))
        else:
            stack.pop()

    # Randomize the start and end points on the boundary
    # Ensure start and end are on different boundaries if chosen from top/bottom or left/right
    sides = ['top', 'right', 'bottom', 'left']
    anakin_skywalker = random.randint(0, 3)
    start_side = sides[anakin_skywalker]  # Choose two distinct sides
    end_side = sides[(anakin_skywalker + 2) % 4]

    # Start point
    if start_side == 'top':
        start = (-1, -1)
        while start[0] == -1 or maze[0][start[1]+1] == 0:
            start = (0, random.choice(range(1, 2 * width - 2, 2)))
    elif start_side == 'bottom':
        start = (-1, -1)
        while start[0] == -1 or maze[2 * height][start[1]-1] == 0:
            start = (2 * height, random.choice(range(1, 2 * width - 2, 2)))
    elif start_side == 'left':
        start = (-1, -1)
        while start[0] == -1 or maze[start[0]+1][0] == 0:
            start = (random.choice(range(1, 2 * height - 2, 2)), 0)
    else:  # 'right'
        start = (-1, -1)
        while start[0] == -1 or maze[start[0]-1][2 * width] == 0:
            start = (random.choice(range(1, 2 * height - 2, 2)), 2 * width)

    # End point
    if end_side == 'top':
        end = (-1, -1)
        while end[0] == -1 or maze[0][end[1]+1] == 0:
            end = (0, random.choice(range(1, 2 * width - 2, 2)))
    elif end_side == 'bottom':
        end = (-1, -1)
        while end[0] == -1 or maze[2 * height][end[1]-1] == 0:
            end = (2 * height, random.choice(range(1, 2 * width - 2, 2)))
    elif end_side == 'left':
        end = (-1, -1)
        while end[0] == -1 or maze[end[0]+1][0] == 0:
            end = (random.choice(range(1, 2 * height - 2, 2)), 0)
    else:  # 'right'
        end = (-1, -1)
        while end[0] == -1 or maze[end[0]-1][2 * width] == 0:
            end = (random.choice(range(1, 2 * height - 2, 2)), 2 * width)

    # Open the start and end points
    maze[start[0]][start[1]] = 0
    maze[end[0]][end[1]] = 0

    return maze, start, end

def visualize_maze(maze, start, end):
    maze_np = np.array(maze)
    # Color code the start and end for visualization
    maze_np[start[0]][start[1]] = 2  # Start point
    maze_np[end[0]][end[1]] = 3  # End point

    plt.figure(figsize=(10, 5))
    # Custom color map: walls in black, path in white, start in green, end in red
    cmap = ListedColormap(['white', 'black', 'green', 'red'])
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
            if 0 <= nx < height and 0 <= ny < width and not visited[nx][ny] and maze[nx][ny] != 1:
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
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])
    bounds = [0,1,2,3,4,5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(maze_np, cmap=cmap, norm=norm)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def save_maze_and_solutions(maze, path):
    maze[path[0][0]][path[0][1]] = 3
    maze[path[len(path) - 1][0]][path[len(path) - 1][1]] = 5
    # path = path[::-1]
    for i in range(1, len(path) - 2):
        if random.randint(0, 100) == 36:
            path_ = (path[i], path[i + 1])
            # Save the maze grid
            with open(f"mazes/large_maze{maze_number}_iteration_{i}.txt", "w") as file:
                og0 = maze[path_[0][0]][path_[0][1]]
                maze[path_[0][0]][path_[0][1]] = 4
                for row in maze:
                    file.write(''.join(str(cell) for cell in row) + "\n")
                file.write("\n")

                maze[path_[0][0]][path_[0][1]] = og0

                # Calculate moves from the path_
                moves = []
                for i in range(1, len(path_)):
                    dx = path_[i][0] - path_[i-1][0]
                    dy = path_[i][1] - path_[i-1][1]
                    # print(dx, dy)
                    if dx == 1: file.write("D\n")
                    elif dx == -1: file.write("U\n")
                    if dy == 1: file.write("R\n")
                    elif dy == -1: file.write("L\n")
                
                # Save the solution moves
                # file.write("".join(moves) + "\n")
            break

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
            if 0 <= next[0] < len(maze) and 0 <= next[1] < len(maze[0]) and maze[next[0]][next[1]] == 0:
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

for j in range(1000):
    width, height = 50, 50
    maze, start, end = generate_maze(width, height)
    maze_number = j
    path = a_star_search(maze, (start[0], start[1]), (end[0], end[1]))
    save_maze_and_solutions(maze, path)
    # visualize_maze_with_path(maze, start, end, path)

