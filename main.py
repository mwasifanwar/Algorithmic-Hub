import pygame as pg
import math
import random
import colors as clr
import vibrant_colors as vclr
from queue import PriorityQueue
from collections import deque
from queue import Queue
import sys
import time

pg.init() # initiates pygame

speed = 0
width = 600  # width set to 800 of the console window
console_window = pg.display.set_mode((width, width))  #  Dimensions of console window set to square
flags = pg.NOFRAME  # Use the NOFRAME flag to create a borderless window
console_window = pg.display.set_mode((width, width), flags=flags)

def screen_shake(console_window):
    shake_intensity = 10  # Adjust the intensity of the screen shake
    shake_duration = 0.5  # Adjust the duration of the screen shake in seconds
    shake_frequency = 10  # Adjust the frequency of the screen shake
    flash_duration = 0.1  # Adjust the duration of the red flash in seconds

    original_pos = console_window.get_rect()  # Save the original position of the screen
    shake_amount = shake_intensity * shake_duration  # Calculate the amount of shake

    # Shake the screen
    start_time = pg.time.get_ticks()
    while pg.time.get_ticks() - start_time < shake_duration * 1500:
        # Generate random offsets for the screen shake
        offset_x = random.uniform(-shake_amount, shake_amount)
        offset_y = random.uniform(-shake_amount, shake_amount)

        # Shift the screen by the calculated offsets
        screen_offset = original_pos.move(offset_x, offset_y)
        console_window.blit(console_window, screen_offset)

        # Create red flashes with adjustable transparency
        flash_alpha = 128  # Adjust the transparency (0-255) of the red flash
        flash_surface = pg.Surface(console_window.get_size(), pg.SRCALPHA)
        flash_surface.fill((255, 0, 0, flash_alpha))
        console_window.blit(flash_surface, (0, 0))

        # Update the Pygame display
        pg.display.flip()

        # Wait for a short interval to control the shake frequency
        pg.time.wait(int(1000 / shake_frequency))

    # Restore the original position of the screen
    console_window.blit(console_window, original_pos)
    pg.display.flip()


class Nodes:  #  Keep track of nodes(cubes) of what color it is of
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width  #  Keep track of the row coorinate on where the cursor is
        self.y = col * width  #  Keep track of the col coorinate on where the cursor is
        self.color = clr.WHITE  #  Initially the color of the grid will be white
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows
        self.visited = False  #  For Depth First Search
        self.cost = float('inf')  # Initialize the cost to infinity for UCS
        self.parent = None
        
    
    def get_pos(self):
        return self.row, self.col
    
    def is_closed(self):
        return self.color == clr.RED
    
    def is_open(self):
        return self.color == clr.LIME
    
    def is_wall(self):
        return self.color == clr.BLACK
    
    def is_start(self):
        return self.color == clr.ORANGE
    
    def is_goal(self):
        return self.color == clr.TURQUOISE
    
    def is_reset(self):
        self.color = clr.WHITE
    
    def make_start(self):
        self.color = clr.ORANGE

    def make_closed(self):
        self.color = clr.RED
    
    def make_open(self):
        self.color = clr.LIME
    
    def make_wall(self):
        self.color = clr.BLACK
    
    def make_goal(self):
        self.color = clr.PURPLE
    
    def make_path(self):
        self.color = clr.PURPLE
    
    def draw(self, console_window):
        pg.draw.rect(console_window, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        # Get the dimensions of the grid
        rows = len(grid)
        cols = len(grid[0])
        
        #  Step 1: Check if the neighbours is not a wall(bareer) then append neighbour into the neighbours list
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_wall():  # Check if the DOWN of current node(cube) is not a wall
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_wall():  # Check if the UP of current node(cube) is not a wall
            self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_wall():  # Check if the RIGHT of current node(cube) is not a wall
            self.neighbours.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_wall():  # Check if the LEFT of current node(cube) is not a wall
            self.neighbours.append(grid[self.row][self.col - 1])

    
    def __lt__(self, other):  #  "less-than function for comparing two Nodes as current Node is less than second Node"
        return False
    
def manhattan_dist(p1, p2):
    x1, y1 = p1  #  p1 is start
    x2, y2 = p2  #  p2 is goal
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

# => A Star Algo ------------------------------------------------------------------------------------------------------------------------------------------------

def a_star_algorithm(draw, grid, start, end, speed):  #  F-score = G-score + H-score   --->   F(n) = G(n) + H(n)  // Formula for A* algorithm
    count = 0
    open_set = PriorityQueue()  #  PriorityQueue is efficient way to get the smallest element everytime

    open_set.put((0, count, start))  #  Add (start node with its F-score) to the priority queue
    came_from = {}  # Keep track of What node did this node come from
    g_score = {nodes: float("inf") for row in grid for nodes in row}
    g_score[start] = 0
    f_score = {nodes: float("inf") for row in grid for nodes in row}
    f_score[start] = manhattan_dist(start.get_pos(), end.get_pos())  #  We want to make an estimate of how far is start node from end node

    open_set_hash = {start}

    while not open_set.empty():  #  if open_set is empty then that means that we have considered every single possible node we have gone through, and if there is no path then the path does not exist
        for event in pg.event.get():
            if event.type == pg.QUIT:  #  To quit the execution of program in the middle if someone wants to
                pg.quit()

        current = open_set.get()[2]  #  The reason behind setting the index to 2 is that, in our ```open_set``` we have three parameters, and the third one, which is on the second index is node and we want to store that node inside the current
        open_set_hash.remove(current)  #  For make sure we dont have any duplicates

        if current == end:  # We have found the shortest path and we need to reconstruct the path
            reconstruct_path(came_from, end, draw)
            end.make_closed()
            return True
        
        for neighbour in current.neighbours:  #  To consider all of the neighbours of current node
            temp_g_score = g_score[current] + 1  #  To go to the next neighbour

            if temp_g_score < g_score[neighbour]:  #  if we have found the best neghbour than previous one then update it to be the better path
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + manhattan_dist(neighbour.get_pos(), end.get_pos())  #  Manhattan Distance deals with the positions

                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))  #  Consider this new neighbour now as it has better path
                    open_set_hash.add(neighbour)  #  We are just going to store the node here
                    neighbour.make_open()

        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        if current != start:  #  This will check if node we just looked at is not start then it will not be added in open_set()
            current.make_closed()

    return None

# -------------------------------------------------------------------------------------------------------------------------------------------------

# => Best First Search ----------------------------------------------------------------------------------------------------------------------------

def get_neighbors(node, grid):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    for direction in directions:
        row, col = node.row + direction[0], node.col + direction[1]

        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            neighbor = grid[row][col]
            if not neighbor.is_wall():
                neighbors.append(neighbor)

    return neighbors


def best_first_search(draw, grid, start, end, speed):
    open_set = PriorityQueue()  # Priority queue to store nodes based on their heuristic scores
    open_set.put((0, start))  # Add start node to the priority queue with priority 0
    came_from = {}  # Dictionary to keep track of the parent nodes

    while not open_set.empty():
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            

        current = open_set.get()[1]  # Get the node with the lowest heuristic score from the priority queue

        if current == end:  # Reached the goal node
            reconstruct_path(came_from, end, draw)  # Reconstruct the path
            end.make_closed()
            return True
        
        for neighbor in current.neighbours:  # Iterate over the neighbors of the current node
            if neighbor not in came_from:  # If the neighbor has not been visited
                came_from[neighbor] = current  # Set the current node as the parent of the neighbor
                priority = manhattan_dist(neighbor.get_pos(), end.get_pos())  # Calculate the heuristic score (Manhattan distance)
                open_set.put((priority, neighbor))  # Add the neighbor to the priority queue
                neighbor.make_open()

        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        if current != start:
            current.make_closed()

    return False

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

# => Dijkestra Algo ------------------------------------------------------------------------------------------------------

def dijkstra(draw, grid, start, end, speed):
    count = 0
    open_set = PriorityQueue()  # Priority queue to store nodes based on their distances
    open_set.put((0, count, start))  # Add start node to the priority queue with distance 0
    came_from = {}  # Dictionary to keep track of the parent nodes
    distance = {node: sys.maxsize for row in grid for node in row}  # Initialize all distances to infinity
    distance[start] = 0  # Set the distance of the start node to 0

    while not open_set.empty():
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        current = open_set.get()[2]  # Get the node with the lowest distance from the priority queue

        if current == end:  # Reached the goal node
            reconstruct_path(came_from, end, draw)  # Reconstruct the path
            end.make_closed()
            return True

        for neighbor in current.neighbours:  # Iterate over the neighbors of the current node
            if neighbor not in came_from:  # If the neighbor has not been visited
                new_distance = distance[current] + 1  # Calculate the distance from the start node to the neighbor
                if new_distance < distance[neighbor]:  # If the new distance is smaller than the current distance
                    came_from[neighbor] = current  # Set the current node as the parent of the neighbor
                    distance[neighbor] = new_distance  # Update the distance of the neighbor
                    count += 1
                    open_set.put((distance[neighbor], count, neighbor))  # Add the neighbor to the priority queue with the new distance
                    neighbor.make_open()

        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        if current != start:
            current.make_closed()

    return False

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------


# => Breadth First Algo ------------------------------------------------------------------------------------------------------

def get_neighbors_Breadth_First(node, grid, rows, cols):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    for direction in directions:
        # Get the row and column of the neighbor based on the direction
        row, col = node.row + direction[0], node.col + direction[1]

        # Check if the neighbor is within the bounds of the grid
        if 0 <= row < rows and 0 <= col < cols:
            neighbor = grid[row][col]
            if not neighbor.is_wall():
                neighbors.append(neighbor)

    return neighbors

def breadth_first(draw, grid, start, end, speed):
    # Get the dimensions of the grid
    rows = len(grid)
    cols = len(grid[0])

    # Create a visited set to keep track of visited nodes
    visited = set()

    # Create a queue for BFS
    queue = deque()

    # Add the start node to the queue and mark it as visited
    queue.append(start)
    visited.add(start)

    # Create a dictionary to store the parent nodes
    parent = {}

    while queue:
        current = queue.popleft()

        # If the current node is the end node, we have found the path
        if current == end:
            reconstruct_path(parent, end, draw)
            end.make_closed()
            return True

        # Get the neighbors of the current node
        neighbors = get_neighbors_Breadth_First(current, grid, rows, cols)

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current
                neighbor.make_open()
        
        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        if current != start:
            current.make_closed()

    return False

# ----------------------------------------------------------------------------------------------------------------


# => Depth First Algo ------------------------------------------------------------------------------------------------------

def get_neighbors_Depth_First(node, grid):
    neighbors = []
    row, col = node.get_pos()

    if row > 0 and not grid[row - 1][col].is_wall():
        neighbors.append(grid[row - 1][col])  # Up neighbor

    if row < len(grid) - 1 and not grid[row + 1][col].is_wall():
        neighbors.append(grid[row + 1][col])  # Down neighbor

    if col > 0 and not grid[row][col - 1].is_wall():
        neighbors.append(grid[row][col - 1])  # Left neighbor

    if col < len(grid[0]) - 1 and not grid[row][col + 1].is_wall():
        neighbors.append(grid[row][col + 1])  # Right neighbor

    return neighbors

def depth_first(draw, grid, start, end, speed):
    if start == end:
        return True

    stack = []          # Create an empty stack for DFS
    visited = set()     # Keep track of visited nodes
    parent = {}         # Store parent nodes for reconstructing the path

    stack.append(start)     # Push the start node to the stack
    visited.add(start)      # Mark the start node as visited

    while stack:
        current = stack.pop()    # Pop a node from the stack

        if current == end:
            reconstruct_path(parent, end, draw)   # Reconstruct the path
            end.make_closed()   # Mark the end node as closed
            return True

        neighbors = get_neighbors_Depth_First(current, grid)   # Get unvisited neighbors

        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)   # Push unvisited neighbors to the stack
                visited.add(neighbor)    # Mark the neighbor as visited
                parent[neighbor] = current   # Store the parent node for path reconstruction
                neighbor.make_open()    # Mark the neighbor as open

        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        if current != start:
            current.make_closed()   # Mark the current node as closed if it's not the start node

    return False


# --------------------------------------------------------------------------------------------------------------------------



# => Uniform Cost Search ---------------------------------------------------------------------------------------------------------------------------------

def get_neighbors(node, grid):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    for direction in directions:
        row, col = node.row + direction[0], node.col + direction[1]

        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            neighbor = grid[row][col]
            if not neighbor.is_wall():
                neighbors.append((neighbor, 1))  # Adjust the cost based on your implementation

    return neighbors

def uniform_cost_search(draw, grid, start, end, speed):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))  # Add start node to the priority queue
    came_from = {}  # Keep track of the parent nodes
    g_score = {node: float('inf') for row in grid for node in row}
    g_score[start] = 0

    while not open_set.empty():
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        current = open_set.get()[2]  # Get the node with the lowest cost
        if current == end:  # Reached the goal node
            reconstruct_path(came_from, end, draw)
            end.make_closed()
            return True

        for neighbor, cost in get_neighbors(current, grid):  # Get the neighboring nodes
            new_cost = g_score[current] + cost  # Adjust the cost based on your implementation
            if new_cost < g_score[neighbor]:  # Found a better path
                came_from[neighbor] = current
                g_score[neighbor] = new_cost
                open_set.put((g_score[neighbor], count, neighbor))
                neighbor.make_open()

        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        if current != start:
            current.make_closed()

    return False

#---------------------------------------------------------------------------------------------------------------------------------

# => Bidirectional Search ----------------------------------------------------------------------------------------------------------------

def bidirectional_bfs(draw, grid, start, end, speed):
    # Initialize the queues for the forward and backward searches
    queue_start = deque([start])
    queue_end = deque([end])

    # Initialize dictionaries to store the parent nodes for the forward and backward searches
    came_from_start = {start: None}
    came_from_end = {end: None}

    # Main loop of the bidirectional BFS
    while queue_start and queue_end:
        # Perform one step of the forward search
        current_start = queue_start.popleft()
        current_start.make_closed()  # Mark the current node as closed
        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        # Check if the forward search intersects with the backward search
        if current_start in came_from_end:
            intersect_node = current_start
            break

        # Iterate over the neighbors of the current node for the forward search
        for neighbor_start in current_start.neighbours:
            if neighbor_start not in came_from_start:
                came_from_start[neighbor_start] = current_start  # Update the parent node
                queue_start.append(neighbor_start)  # Add the neighbor to the queue
                neighbor_start.make_open()  # Mark the neighbor as open
                draw()
                time.sleep(speed)  # Introduce a speed to control the speed of execution

        # Perform one step of the backward search
        current_end = queue_end.popleft()
        current_end.make_closed()  # Mark the current node as closed
        draw()
        time.sleep(speed)  # Introduce a speed to control the speed of execution

        # Check if the backward search intersects with the forward search
        if current_end in came_from_start:
            intersect_node = current_end
            break

        # Iterate over the neighbors of the current node for the backward search
        for neighbor_end in current_end.neighbours:
            if neighbor_end not in came_from_end:
                came_from_end[neighbor_end] = current_end  # Update the parent node
                queue_end.append(neighbor_end)  # Add the neighbor to the queue
                neighbor_end.make_open()  # Mark the neighbor as open
                draw()
                time.sleep(speed)  # Introduce a speed to control the speed of execution

    # Reconstruct the path from the start node to the intersecting node
    path_start = []
    current = intersect_node
    while current != start:
        path_start.append(current)
        current = came_from_start[current]

    # Reconstruct the path from the end node to the intersecting node
    path_end = []
    current = intersect_node
    while current != end:
        path_end.append(current)
        current = came_from_end[current]

    # Combine the paths in reverse order
    path = path_start + list(reversed(path_end))
    
    # Mark the nodes along the final path
    for node in path:
        node.make_path()
        draw()  # Introduce a speed to control the speed of execution
        time.sleep(speed)  # Introduce a speed to control the speed of execution

    return path

# -----------------------------------------------------------------------------------------------------------------------------------------

# => Depth Limited Search ----------------------------------------------------------------------------------------------------------------
def depth_limited_search(draw, grid, node, end, depth_limit, speed):
    # Check if the current node is the goal node
    if node == end:
        return True

    # Check if depth limit has been reached
    if depth_limit <= 0:
        return False

    # Mark the current node as closed and update the visualization
    node.make_closed()
    draw()
    time.sleep(speed)  # Introduce a speed to control the speed of execution

    # Explore the neighbors of the current node
    for neighbor in node.neighbours:
        # Check if the neighbor is not closed
        if not neighbor.is_closed():
            # Mark the neighbor as open and update the visualization
            neighbor.make_open()
            draw()
            time.sleep(speed)  # Introduce a speed to control the speed of execution

            # Recursively call depth_limited_search on the neighbor with reduced depth limit and speed
            if depth_limited_search(draw, grid, neighbor, end, depth_limit - 1, speed):
                # If the path is found, mark the neighbor as part of the path and update the visualization
                neighbor.make_path()
                draw()
                time.sleep(speed)  # Introduce a speed to control the speed of execution
                return True

    return False

# -----------------------------------------------------------------------------------------------------------------------------------------

# => Iterative Deepening Search ----------------------------------------------------------------------------------------------------------------

def dls(draw, grid, node, end, depth_limit, came_from, speed):
    # Check if the current node is the goal node
    if node == end:
        return True

    # Check if the depth limit has been reached
    if depth_limit <= 0:
        return False

    # Iterate over the neighbors of the current node
    for neighbor in node.neighbours:
        # Check if the neighbor has not been visited before
        if neighbor not in came_from:
            came_from[neighbor] = node  # Update the parent node for the neighbor
            neighbor.make_open()  # Mark the neighbor as open
            draw()
            time.sleep(speed)  # Introduce a speed to control the speed of execution  # Update the visualization
            # Recursively call dls with the neighbor as the new current node
            if dls(draw, grid, neighbor, end, depth_limit - 1, came_from, speed):
                return True
            neighbor.make_closed()  # Mark the neighbor as closed
            draw()
            time.sleep(speed)  # Introduce a speed to control the speed of execution  # Update the visualization

    return False


def iterative_deepening_search(draw, grid, start, end, speed):
    # Iterate over the depth limits from 0 to sys.maxsize
    for depth_limit in range(sys.maxsize):
        came_from = {}  # Dictionary to store the parent nodes
        # Call dls to perform depth-limited search with the current depth limit
        if dls(draw, grid, start, end, depth_limit, came_from, speed):
            reconstruct_path(came_from, end, draw)  # Reconstruct the path
            end.make_closed()  # Mark the end node as closed
            return True

    return False


# -----------------------------------------------------------------------------------------------------------------------------------------
def make_grid(rows, width):  #  The parameters will be that how many rows will be inside the grid and what will be the width of the grid
    grid = []
    gap = width // rows  #  Returns the width of each cube inside the grid
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            nodes = Nodes(i, j, gap, rows)
            grid[i].append(nodes)

    return grid

def draw_grid(console_window, rows, width):
    _gap = width // rows  #  Returns the width of each cube inside the grid
    for i in range(rows):  #  For drawing horizontal lines
        pg.draw.line(console_window, clr.WHITE,(0, i*_gap),(width, i*_gap))
        for j in range(rows):  #  For drawing vertical lines
            pg.draw.line(console_window, clr.WHITE, (j*_gap, 0), (j*_gap, width))

def draw(console_window, grid, rows, width):
    console_window.fill(clr.WHITE)

    for row in grid:
        for node in row:
            node.draw(console_window)

    draw_grid(console_window, rows, width)  # Draw grid lines

    for row in grid:  # Draw walls again to cover grid lines inside walls
        for node in row:
            if node.is_wall():
                node.draw(console_window)
            if node.color == clr.PURPLE:
                node.draw(console_window)
            if node.color == clr.LIME:
                node.draw(console_window)

    pg.display.update()  #  update whatever we have drawn on the screen



def get_mouseclicked_position(position, rows, width):
    gap = width // rows  #  Returns the width of each cube inside the grid
    y, x = position
    
    # Check if the mouse position is within the grid
    if x < 0 or y < 0 or x >= width or y >= width:  #  The conditions x < 0 and y < 0 are included to handle scenarios where the mouse cursor might go beyond the left or upper boundaries of the pygame window AND The conditions x >= width and y >= width are included to handle scenarios where the mouse cursor might go beyond the right or bottom boundaries of the pygame window. 
        return None, None

    row = y // gap  #  Takes the position in y from line 102 and divides by gap that is basically width of each cube telling us where we are in the cube
    col = x // gap  #  Takes the position in x from line 102 and divides by gap that is basically width of each cube telling us where we are in the cube

    return row, col

def main(console_window, width):
    rows = 50
    grid = make_grid(rows, width - (width % rows))  #  The parameters of this make_grid() function are set to ```width - (width % rows)```  instead of simple width only to prevent ```the list index out of range error``` in a way that it evenly divides no. of rows with width
    
    start = None
    end = None

    run = True    

    while run:
        draw(console_window, grid, rows, width)
        for event in pg.event.get():  #  Any event that might occur in the program like mouseclick event etc
            if event.type == pg.QUIT:  #  ipgf the event is to quit then exit the loop
                run = False
            if event.type == pg.QUIT:
                is_running = False

            # Skip event processing if mouse position is outside the frame
            if not pg.mouse.get_focused():
                continue

            if pg.mouse.get_pressed()[0]:  #  if left mouse button pressed
                position = pg.mouse.get_pos()  #  Gives us the position of mouse on screen of pygame
                row, col = get_mouseclicked_position(position, rows, width)  #  Gives us the row and column on which we clicked on
                if row is None or col is None:
                    continue
                nodes = grid[row][col]
                if not start and nodes != end:
                    start = nodes
                    start.make_start()
                elif not end and nodes != start:
                    end = nodes
                    end.make_goal()
                elif nodes != start and nodes != end:
                    nodes.make_wall()

            elif pg.mouse.get_pressed()[2]:  # if right mouse button pressed
                position = pg.mouse.get_pos()
                row, col = get_mouseclicked_position(position, rows, width)
                nodes = grid[row][col]
                nodes.is_reset()

                if nodes == start:
                    start = None
                elif nodes == end:
                    end = None  # Update the variable here with assignment operator (=) instead of comparison operator (==)

            depth_limit = 100  # Set the depth limit to the desired value
            if event.type == pg.KEYDOWN:  #  To start the algorithm
                if event.key == pg.K_SPACE and start and end:  #  if the program has not yet started and space bar is pressed then call the ```update_neighbours()``` function
                    #  First update the neighbours of the Nodes class
                    for row in grid:
                        for nodes in row:
                            nodes.update_neighbours(grid)
                    path = a_star_algorithm(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)
                    #path = best_first_search(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)
                    #path = breadth_first(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)
                    #path = depth_first(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)
                    #path = uniform_cost_search(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)
                    #path = depth_limited_search(lambda: draw(console_window, grid, rows, width), grid, start, end, depth_limit, speed)
                    #path = iterative_deepening_search(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)
                    #path = bidirectional_bfs(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)
                    #path = dijkstra(lambda: draw(console_window, grid, rows, width), grid, start, end, speed)

                    if not path:
                        screen_shake(console_window)

                if event.key == pg.K_c:  #  Clear the entire frame
                    start = None
                    end = None
                    grid = make_grid(rows, width - (width % rows))  #  Remake the grid

    pg.quit()

main(console_window, width)