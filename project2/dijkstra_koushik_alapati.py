# Importing necessary modules/libraries for computation
import numpy as np  # Library for numerical operations
import cv2 as cv    # OpenCV library for image processing
import math as m    # Math library for mathematical operations
import time as tm   # Library for time-related functions

# Definition of the Node class for representing nodes in the graph
class Node:
    def __init__(self, position, cost, parent):
        self.position = position  # Position of the node
        self.cost = cost          # Cost associated with reaching this node
        self.parent = parent      # Reference to the parent node
        
# Drawing polyline shapes
def poly_Shape(img, polyPts, color, type):
    """
    Function to draw polyline shapes on an image.

    Parameters:
        img (numpy.ndarray): The image to draw on.
        polyPts (numpy.ndarray): Array of points defining the shape.
        color (tuple): Color of the shape.
        type (str): Type of shape ('Obstacle' or 'border').

    Returns:
        None
    """
    polyPts = polyPts.reshape((-1, 1, 2))
    if type == "Obstacle":
        cv.fillPoly(img, [polyPts], color)
    else:
        cv.polylines(img, [polyPts], True, color, thickness=5)


def gen_Map():
    """
    Function to generate the map with obstacles.

    Returns:
        numpy.ndarray: The generated map image.
    """
    clearance = 5

    # Generating a black background arena
    arena = np.zeros((500, 1200, 3), dtype="uint8")
    canvas_height, canvas_width, _ = arena.shape
    # Defining colors
    white = (255, 255, 255)
    blue = (255, 0, 0)

    for x in range(canvas_width):
        for y in range(canvas_height):

            # Defining clearance on the borders
            if x <= clearance or x >= 1200 - clearance or y >= 500 - clearance or y <= clearance:
                arena[499 - y][x] = white

            # Drawing the Top rectangle obstacle Border
            if (100 - clearance) <= x <= (175 + clearance) and (0 - clearance) <= y <= 400 + clearance:
                arena[y - 499][x] = white
            # Drawing the Top rectangle obstacle
            if 100 <= x <= 175 and 0 <= y <= 400:
                arena[y - 499][x] = blue

            # Drawing the Bottom rectangle obstacle Border
            if (275 - clearance) <= x <= (350 + clearance) and (0) <= y <= 400 + clearance:
                arena[499 - y][x] = white
            # Drawing the Bottom rectangle obstacle
            if 275 <= x <= 350 and 0 <= y <= 400:
                arena[499 - y][x] = blue

            # Drawing Borders for reverse C shape
            if (900 - clearance) <= x <= (1100 + clearance) and (50 - clearance) <= y <= 125 + clearance:
                arena[y - 499][x] = white
            if (1020 - clearance) <= x <= (1100 + clearance) and (125 - clearance) <= y <= 375 + clearance:
                arena[y - 499][x] = white
            if (900 - clearance) <= x <= (1100 + clearance) and (375 - clearance) <= y <= 450 + clearance:
                arena[y - 499][x] = white

            # Drawing the reverse C shape
            if 900 <= x <= 1100 and 50 <= y <= 125:
                arena[y - 499][x] = blue
            if 1020 <= x <= 1100 and 125 <= y <= 375:
                arena[y - 499][x] = blue
            if 900 <= x <= 1100 and 375 <= y <= 450:
                arena[y - 499][x] = blue

            # Defining clearance on the borders
            if x <= clearance or x >= 1200 - clearance or y >= 500 - clearance or y <= clearance:
                arena[499 - y][x] = white

            # Define the vertices of the hexagon
    center = (650, 250)
    radius = 150
    num_sides = 6
    angle_step = 2 * np.pi / num_sides
    hexagon_pts = []
    for i in range(num_sides):
        x = int(center[0] + radius * np.cos(i * angle_step + np.pi / 6))
        y = int(center[1] + radius * np.sin(i * angle_step + np.pi / 6))
        hexagon_pts.append((x, y))

    # Convert the list of points to a numpy array
    hexagon_pts = np.array(hexagon_pts, np.int32)
    poly_Shape(arena, hexagon_pts, blue, "Obstacle")

    radius = 150 + 5
    num_sides = 6
    angle_step = 2 * np.pi / num_sides
    hexagon_pts_bor = []
    for i in range(num_sides):
        x = int(center[0] + radius * np.cos(i * angle_step + np.pi / 6))
        y = int(center[1] + radius * np.sin(i * angle_step + np.pi / 6))
        hexagon_pts_bor.append((x, y))

    # Convert the list of points to a numpy array
    hexagon_pts_bor = np.array(hexagon_pts_bor, np.int32)
    poly_Shape(arena, hexagon_pts_bor, white, "border")

    return arena
def markers(strtNode, goalNode):
    """
    Function to mark the start and goal points on the map.

    Parameters:
        strtNode (tuple): Tuple containing the start point coordinates.
        goalNode (tuple): Tuple containing the goal point coordinates.

    Returns:
        numpy.ndarray: The map image with start and goal points marked.
    """
    map_image[(strtNode[1]), strtNode[0]] = np.array([0, 0, 255])
    map_image[(goalNode[1]), goalNode[0]] = np.array([0, 0, 255])
    top_left = (int(strtNode[0] - 5 / 2), int(499-strtNode[1] - 5 / 2))
    bottom_right = (int(strtNode[0] + 5 / 2), int(499-strtNode[1] + 5 / 2))
    cv.rectangle(map_image, top_left, bottom_right, (0, 0, 255), thickness=2)
    
    top_left = (int(goalNode[0] - 5 / 2), int(499-goalNode[1] - 5 / 2))
    bottom_right = (int(goalNode[0] + 5 / 2), int(499-goalNode[1] + 5 / 2))
    cv.rectangle(map_image, top_left, bottom_right, (0, 0, 255), thickness=2)
    return map_image

def coordinates():
    """
    Function to get the start and end coordinates from the user.

    Returns:
        tuple: Tuple containing the start and end coordinates.
    """
    start_x = int(input("Enter the x-coordinate of the start point: "))
    start_y = int(input("Enter the y-coordinate of the start point: "))
    goal_x = int(input("Enter the x-coordinate of the goal point: "))
    goal_y = int(input("Enter the y-coordinate of the goal point: "))

    start_pt = (start_x, start_y)
    goal_pt = (goal_x, goal_y)
    return start_pt, goal_pt


def check_coordinates(startPt, endPt):
    """
    Function to check if the start and end points are valid.

    Parameters:
        startPt (tuple): Tuple containing the start point coordinates.
        endPt (tuple): Tuple containing the end point coordinates.

    Returns:
        bool: True if the points are valid, False otherwise.
    """
    canvas_height, canvas_width, _ = map_image.shape  # Get canvas dimensions

    if map_image[(canvas_height - startPt[1] - 1), startPt[0]].any() == np.array([0, 0, 0]).all():
        if map_image[(canvas_height - endPt[1] - 1), endPt[0]].any() == np.array([0, 0, 0]).all():
            print("started")
            status = True
        elif map_image[(canvas_height - endPt[1] - 1), endPt[0]].all() == np.array([255, 255, 255]).all():
            print("The goal is on the border change the goal point")
            status = False
        else:
            print("The goal is inside obstacle change the goal point")
            status = False 
    elif map_image[(canvas_height - startPt[1] - 1), startPt[0]].all() == np.array([255, 255, 255]).all():
        print("The start is on the border point change the start point")
        status = False
    else:
        print("The start is on the obstacle point change the start point")
        status = False 

    return status

# Function to move up
def move_Up(curPos):
    """
    Function to move the current position up by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[1] += 1
    return curPos, 1

# Function to move down
def move_Down(curPos):
    """
    Function to move the current position down by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[1] -= 1
    return curPos, 1

# Function to move right
def move_Right(curPos):
    """
    Function to move the current position right by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[0] += 1
    return curPos, 1

# Function to move left
def move_Left(curPos):
    """
    Function to move the current position left by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[0] -= 1
    return curPos, 1

# Function to move up-right
def move_Up_Right(curPos):
    """
    Function to move the current position diagonally up-right by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[0] += 1
    curPos[1] += 1
    return curPos, 1.4

# Function to move up-left
def move_Up_Left(curPos):
    """
    Function to move the current position diagonally up-left by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[0] -= 1
    curPos[1] += 1
    return curPos, 1.4

# Function to move down-right
def move_Down_Right(curPos):
    """
    Function to move the current position diagonally down-right by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[0] += 1
    curPos[1] -= 1
    return curPos, 1.4

# Function to move down-left
def move_Down_Left(curPos):
    """
    Function to move the current position diagonally down-left by one unit.

    Parameters:
        curPos (list): List containing the current position coordinates.

    Returns:
        tuple: Tuple containing the new position coordinates and the cost of the move.
    """
    curPos[0] -= 1
    curPos[1] -= 1
    return curPos, 1.4


def feasibility_check(curPos):
    """
    Function to check the feasibility of computed neighbors in terms of their location and obstacle presence.

    Parameters:
        curPos (list): List of tuples containing the coordinates and costs of potential neighbors.

    Returns:
        list: List of feasible neighbors.
    """
    next_neighbour = []
    for element in curPos:
        if element[0][1] >= 0 and element[0][1] < 500 and element[0][0] >= 0 and element[0][0] < 1200:
            if map_image[(499 - element[0][1]), element[0][0]].all() == np.array([0, 0, 0]).all():
                next_neighbour.append([element[0], element[1]])
    return next_neighbour

def neighbour_list(node):
    """
    Function to accept node coordinates and generate a list of feasible neighbors for each node.

    Parameters:
        node (Node): Node object representing the current position.

    Returns:
        list: List of feasible neighbors for the given node.
    """
    current_Pos = [node.position[0], node.position[1]]
    conNeighList = [move_Up(current_Pos.copy()), move_Down(current_Pos.copy()),
                    move_Right(current_Pos.copy()), move_Left(current_Pos.copy()),
                    move_Up_Right(current_Pos.copy()), move_Down_Right(current_Pos.copy()),
                    move_Up_Left(current_Pos.copy()), move_Down_Left(current_Pos.copy())]
    
    neighbour = feasibility_check(conNeighList) # Check for feasible neighbors
    return neighbour

    
def main():
    """
    Main function to execute the Dijkstra algorithm and visualize the pathfinding process.
    """
    global map_image
    que = []  # Queue to store nodes based on their cost
    node_dict = {}  # Dictionary to store nodes with their positions as keys
    visited_set = set([])  # Set to store visited nodes
    nodeObj = {}  # Dictionary to store the goal node and its details
    
    # Generate the map image
    map_image = gen_Map()
    videoStr = []  # List to store frames for video creation
    
    # Get the start and goal coordinates from user input
    status = False
    while not status:
        strtNode, goalNode = coordinates()
        status = check_coordinates(strtNode, goalNode) 
    
    
    start_t = tm.time()  # Record the start time of the algorithm
    
    # Initialize cost dictionary with infinity for all positions
    cost_Come = {}
    for i in range(500):
        for j in range(1200):
            cost_Come[str([i, j])] = m.inf
    
    cost_Come[str(strtNode)] = 0  # Cost to reach start node is 0
    visited_set.add(str(strtNode))  # Mark start node as visited
    node = Node(strtNode, 0, None)  # Create a node object for the start position
    node_dict[str(node.position)] = node  # Add start node to the dictionary
    que.append([node.cost, node.position])  # Add start node to the priority queue
    
    iter = 0  # Initialize iteration counter
    
    # Main loop of Dijkstra algorithm
    while que:
        que.sort()  # Sort the priority queue based on cost
        queNode = que.pop(0)  # Pop the node with minimum cost from the queue
        node = node_dict[str(queNode[1])]  # Get the corresponding node object
        
        # Check if the goal node is reached
        if queNode[1][0] == goalNode[0] and queNode[1][1] == goalNode[1]:
            nodeObj[str(goalNode)] = Node(goalNode, queNode[0], node)  # Store goal node details
            break
        
        # Iterate over the feasible neighbors of the current node
        for neighbour_Node, neighCost in neighbour_list(node):
            if str(neighbour_Node) in visited_set:  # Check if the neighbor has been visited
                # Update cost if the new path through the current node is cheaper
                neighbour_updated_Cost = neighCost + cost_Come[str(node.position)]
                if neighbour_updated_Cost < cost_Come[str(neighbour_Node)]:
                    cost_Come[str(neighbour_Node)] = neighbour_updated_Cost
                    node_dict[str(neighbour_Node)].parent = node  # Update parent of the neighbor node
                
            else:
                # Mark neighbor as visited and update its cost
                visited_set.add(str(neighbour_Node))
                map_image[(499 - neighbour_Node[1]), neighbour_Node[0], :] = np.array([0, 255, 0])
    
                # Append frame to videoStr for visualization
                if iter % 1000 == 0:
                    videoStr.append(map_image.copy())
                    cv.imshow("Dijkstra Algorithm", map_image)
                    cv.waitKey(1)
                    
    
                updated_cost = neighCost + cost_Come[str(node.position)]
                cost_Come[str(neighbour_Node)] = updated_cost
                nxtNode = Node(neighbour_Node, updated_cost, node_dict[str(node.position)])
                que.append([updated_cost, nxtNode.position])  # Add neighbor to priority queue
                node_dict[str(nxtNode.position)] = nxtNode  # Add neighbor node to dictionary
        iter += 1
    
    # Backtrack to find the optimal path
    bckTrackNode = nodeObj[str(goalNode)]
    prntNode = bckTrackNode.parent
    bckTrackLst = []
    
    # Generate list of nodes in the optimal path
    while prntNode:
        bckTrackLst.append([(499 - prntNode.position[1]), prntNode.position[0]])
        prntNode = prntNode.parent
    
    bckTrackLst.reverse()  # Reverse the list to get correct order
    
    # Visualize the optimal path
    for val in bckTrackLst:
        map_image[val[0], val[1], :] = np.array([255, 0, 0])
        map_image = markers(strtNode, goalNode)  # Mark the start and goal points on the map 
        videoStr.append(map_image.copy())
        cv.imshow("Dijkstra Algorithm", map_image)
        cv.waitKey(1)
    
    end_t = tm.time()  # Record the end time of the algorithm

    # Print total time taken to find the optimal path
    print("Total time taken to find the optimal path: {:.2f}".format(end_t - start_t), "seconds")  
    
    # Video writing
    clip = cv.VideoWriter(
        'dijkstra_animation_video.mp4', cv.VideoWriter_fourcc(*'MP4V'), 50, (1200, 500))
    
    for idx in range(len(videoStr)):
        frame = videoStr[idx]
        clip.write(frame)
    clip.release()
    
    print("Clip stored at terminal's directory path")

    # Wait for any key to be pressed before closing the window
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()