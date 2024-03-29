# Maze generator -- Randomized Prim Algorithm

## Imports
import random
import time
from colorama import init
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import heapq
import copy

## Functions
def printMaze(maze):
	for i in range(0, len(maze)):
		for j in range(0, len(maze[0])):
			if (maze[i][j] == 'u'):
				print(Fore.WHITE + str(maze[i][j]), end=" ")
			elif (maze[i][j] == 'c'):
				print(Fore.GREEN + str(maze[i][j]), end=" ")
			else:
				print(Fore.RED + str(maze[i][j]), end=" ")
            # else: (maze[i][j] == 'x'):
            #     print(Fore.BLUE + str(maze[i][j]), end=" ")
			
		print('\n')
     
def purify_maze(maze):
    for col in maze[len(maze) - 1]:
        if col == 0:
            col = 1
    return maze

def display_maze(maze):
	# Convert maze to a numpy array for visualization
    maze_array = np.array([[1 if cell == 'w' else 0 for cell in row] for row in maze])

    # Plot the maze
    plt.imshow(maze_array, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Find number of surrounding cells
def surroundingCells(maze, width, height, rand_wall):
	s_cells = 0
	if (maze[rand_wall[0]-1][rand_wall[1]] == 'c'):
		s_cells += 1
	if (maze[rand_wall[0]+1][rand_wall[1]] == 'c'):
		s_cells += 1
	if (maze[rand_wall[0]][rand_wall[1]-1] == 'c'):
		s_cells +=1
	if (maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
		s_cells += 1

	return s_cells



def generate_maze(width, height):
    wall = 'w'
    cell = 'c'
    unvisited = 'u'
    maze = []

    # Initialize colorama
    init()

    # Denote all cells as unvisited
    for i in range(0, height):
        line = []
        for j in range(0, width):
            line.append(unvisited)
        maze.append(line)

    # Randomize starting point and set it a cell
    starting_height = int(random.random()*height)
    starting_width = int(random.random()*width)
    if (starting_height == 0):
        starting_height += 1
    if (starting_height == height-1):
        starting_height -= 1
    if (starting_width == 0):
        starting_width += 1
    if (starting_width == width-1):
        starting_width -= 1

    # Mark it as cell and add surrounding walls to the list
    maze[starting_height][starting_width] = cell
    walls = []
    walls.append([starting_height - 1, starting_width])
    walls.append([starting_height, starting_width - 1])
    walls.append([starting_height, starting_width + 1])
    walls.append([starting_height + 1, starting_width])

    # Denote walls in maze
    maze[starting_height-1][starting_width] = 'w'
    maze[starting_height][starting_width - 1] = 'w'
    maze[starting_height][starting_width + 1] = 'w'
    maze[starting_height + 1][starting_width] = 'w'
    while (walls):
        # Pick a random wall
        rand_wall = walls[int(random.random()*len(walls))-1]

        # Check if it is a left wall
        if (rand_wall[1] != 0):
            if (maze[rand_wall[0]][rand_wall[1]-1] == 'u' and maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
                # Find the number of surrounding cells
                s_cells = surroundingCells(maze, width, height, rand_wall)

                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    # Upper cell
                    if (rand_wall[0] != 0):
                        if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])


                    # Bottom cell
                    if (rand_wall[0] != height-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])

                    # Leftmost cell
                    if (rand_wall[1] != 0):	
                        if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])
                

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Check if it is an upper wall
        if (rand_wall[0] != 0):
            if (maze[rand_wall[0]-1][rand_wall[1]] == 'u' and maze[rand_wall[0]+1][rand_wall[1]] == 'c'):

                s_cells = surroundingCells(maze, width, height, rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    # Upper cell
                    if (rand_wall[0] != 0):
                        if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])

                    # Leftmost cell
                    if (rand_wall[1] != 0):
                        if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])

                    # Rightmost cell
                    if (rand_wall[1] != width-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Check the bottom wall
        if (rand_wall[0] != height-1):
            if (maze[rand_wall[0]+1][rand_wall[1]] == 'u' and maze[rand_wall[0]-1][rand_wall[1]] == 'c'):

                s_cells = surroundingCells(maze, width, height, rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    if (rand_wall[0] != height-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])
                    if (rand_wall[1] != 0):
                        if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])
                    if (rand_wall[1] != width-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)


                continue

        # Check the right wall
        if (rand_wall[1] != width-1):
            if (maze[rand_wall[0]][rand_wall[1]+1] == 'u' and maze[rand_wall[0]][rand_wall[1]-1] == 'c'):

                s_cells = surroundingCells(maze, width, height, rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    if (rand_wall[1] != width-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])
                    if (rand_wall[0] != height-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])
                    if (rand_wall[0] != 0):	
                        if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Delete the wall from the list anyway
        for wall in walls:
            if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                walls.remove(wall)
        


    # Mark the remaining unvisited cells as walls
    for i in range(0, height):
        for j in range(0, width):
            if (maze[i][j] == 'u'):
                maze[i][j] = 'w'

    # Set entrance and exit
    for i in range(0, width):
        if (maze[1][i] == 'c'):
            maze[0][i] = 'c'
            break

    for i in range(width-1, 0, -1):
        if (maze[height-2][i] == 'c'):
            maze[height-1][i] = 'c'
            break
	
    return maze

# for i in range(0, 1):
#      with open(f"maze{i}.txt", "w") as file:
#           maze = generate_maze(21, 21)
        #   maze = purify_maze(maze)
        #   display_maze(maze)
          

def convert_maze_for_visualization(maze):
    num_rows = len(maze)
    num_cols = len(maze[0])
    converted_maze = np.zeros((num_rows, num_cols), dtype=int)  # Initialize maze array with zeros
    for i in range(num_rows):
        for j in range(num_cols):
            if maze[i][j] == 'w':
                converted_maze[i][j] = 1  # Set walls to 1
            elif maze[i][j] == 'c':
                converted_maze[i][j] = 0  # Set clear spaces to 0
            elif maze[i][j] == 's':
                converted_maze[i][j] = 2  # Set start cell to 2
            elif maze[i][j] == 'f':
                converted_maze[i][j] = 3  # Set finish cell to 3
            elif maze[i][j] == 'p':
                converted_maze[i][j] = 4  # Set path cell to 4
    return converted_maze

def heuristic(start, goal):
    # Manhattan distance heuristic
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def astar(maze, start, goal):
    # Define the dimensions of the maze
    rows = len(maze)
    cols = len(maze[0])

    # Define movement directions (up, down, left, right)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    open_set = []

    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    path = []
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
            # Mark the path in the maze
            for cell in path:
                maze[cell[0]][cell[1]] = 'p'
            
            # return path[::-1]
        
        for direction in directions:
            dx, dy = direction
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] in ['c', 'f']:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return path

def mark_path_on_maze(maze, path):
    marked_mazes = [np.copy(maze)]  # Store the original maze as the first element
    for i in range(1, len(path)):
        maze_copy = np.copy(maze)  # Create a copy of the original maze
        for x, y in path[:i+1]:  # Mark the path up to and including the current step
            maze_copy[x][y] = 'p'  # Mark the path with 'p'
        marked_mazes.append(maze_copy)  # Add the marked maze to the list
    return marked_mazes

def visualize_mazes(mazes):
    plt.figure(figsize=(len(mazes) * 2, 2))
    for i, maze in enumerate(mazes):
        plt.subplot(1, len(mazes), i + 1)
        plt.imshow(maze, cmap='binary', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    plt.show()

WIDTH = 10
HEIGHT = 10
start = (0, 1)
goal = (HEIGHT - 1, WIDTH - 2)
# Find the shortest path using A*
maze = generate_maze(WIDTH, HEIGHT)
path = astar(maze, start, goal)
maze_list = mark_path_on_maze(maze, path)
print(maze_list)
# # print(maze_list)
# for i, marked_maze in enumerate(maze_list):
#     print(f"Step {i}:")
#     print(marked_maze)
#     print()

visualize_mazes([convert_maze_for_visualization(maze) for maze in maze_list])
