# Dijkstra Pathfinding Algorithm Visualization

This project demonstrates the Dijkstra algorithm for pathfinding in a 2D grid-based environment. The algorithm finds the shortest path between a given start and end point while avoiding obstacles.

## Instructions

### 1. Libraries/Dependencies:
- numpy
- cv2 (OpenCV)
- math
- time

### 2. Running the Code:
- Ensure that you have the required dependencies installed.
- Execute the `main()` function in the provided Python script (`dijkstra_koushik_alapati.py`).

### 3. Input:
- Upon running the code, you will be prompted to enter the coordinates for the start and goal points.
- Input the x and y coordinates for both the start and goal points when prompted.
- The Input will be taken till the right coordinates are given, that is when the start and goal are not in obstacle and boundary.

### 4. Output:
- The program will display the visualization of the Dijkstra algorithm's execution, showing the pathfinding process step by step.
- Once the optimal path is found, the final path will be highlighted in blue.

### 5. Video Output:
- A video animation of the pathfinding process will be generated and stored as `dijkstra_animation_video.mp4` in the terminal's directory path.
- The video will provide a visual representation of the algorithm's execution, making it easier to understand the pathfinding process.

## Note
- The program includes functions to generate the map with obstacles, move the agent in different directions, and check the feasibility of potential neighbor nodes.
- The Dijkstra algorithm implementation uses a priority queue, and dictionary to efficiently explore the search space.
- The visualization helps in understanding how the algorithm explores the grid and finds the optimal path from the start to the goal point.

