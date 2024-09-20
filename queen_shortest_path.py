#!/usr/bin/env python3
"""
queen_shortest_path.py

A comprehensive Python implementation to find the shortest path for a Queen on a binary grid
using multiple algorithms: Brute-Force (DFS), Breadth-First Search (BFS), A* Search, and
Bidirectional BFS.

The grid can be configured by the user via a JSON file. If no grid file is provided, a default
8x8 grid is used.

Author: Samir Musali
Date: 2024-09-20
"""

import logging
import sys
import argparse
from collections import deque
import heapq
import json
import os
import time  # Added for time measurement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define default grid as per the problem statement
DEFAULT_GRID = [
    [1, 0, 0, 1, 0, 1, 1, 0],  # Row 0
    [1, 0, 0, 1, 0, 1, 1, 0],  # Row 1
    [1, 0, 0, 0, 1, 1, 1, 0],  # Row 2
    [1, 1, 1, 0, 1, 1, 0, 1],  # Row 3
    [1, 1, 0, 1, 0, 1, 1, 0],  # Row 4
    [0, 0, 0, 1, 0, 1, 0, 1],  # Row 5
    [1, 0, 0, 1, 0, 1, 1, 0],  # Row 6
    [1, 0, 0, 1, 0, 1, 1, 1],  # Row 7
]

START_POS_DEFAULT = (0, 0)
TARGET_POS_DEFAULT = (7, 7)

# Define Queen's possible movement directions
DIRECTIONS = [
    (-1, -1),  # Up-Left
    (-1, 0),   # Left
    (-1, 1),   # Down-Left
    (0, -1),   # Up
    (0, 1),    # Down
    (1, -1),   # Up-Right
    (1, 0),    # Right
    (1, 1),    # Down-Right
]

def load_grid(grid_path):
    """
    Load grid from a JSON file.

    The JSON file should contain a 2D list of integers (0s and 1s).

    Example:
    [
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ]
    """
    if not os.path.exists(grid_path):
        logger.error(f"Grid file '{grid_path}' does not exist.")
        sys.exit(1)
    
    try:
        with open(grid_path, 'r') as f:
            grid = json.load(f)
        
        if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
            logger.error("Grid file must contain a 2D list.")
            sys.exit(1)
        
        # Validate grid contents
        for row in grid:
            for cell in row:
                if cell not in (0, 1):
                    logger.error("Grid cells must contain only 0s and 1s.")
                    sys.exit(1)
        
        return grid
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from grid file: {e}")
        sys.exit(1)

def is_valid_position(x, y, grid_size):
    """Check if the position (x, y) is within the grid boundaries."""
    return 0 <= x < grid_size['cols'] and 0 <= y < grid_size['rows']

def get_neighbors(x, y, grid, grid_size):
    """Generate all valid neighbors for the Queen from position (x, y)."""
    neighbors = []
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        while is_valid_position(nx, ny, grid_size) and grid[ny][nx] == 1:
            neighbors.append((nx, ny))
            nx += dx
            ny += dy
    return neighbors

def reconstruct_path(parents, end, start):
    """Reconstruct the path from start to end using the parents dictionary."""
    path = []
    current = end
    while current in parents:
        path.append(current)
        current = parents[current]
    if current == start:
        path.append(current)
    path.reverse()
    return path

def brute_force_dfs(grid, grid_size, start, target):
    """
    Brute-Force Approach using Depth-First Search (DFS) with Backtracking.

    Note: This approach is highly inefficient for larger grids due to its exponential time complexity.
    """

    logger.info("Starting Brute-Force DFS Approach")
    shortest_path = []
    min_length = [float('inf')]

    def dfs(current, target, path, visited):
        x, y = current
        logger.debug(f"Visiting: {current}, Path Length: {len(path)}")

        if current == target:
            if len(path) < min_length[0]:
                min_length[0] = len(path)
                shortest_path.clear()
                shortest_path.extend(path)
                logger.info(f"New shortest path found: {shortest_path} with length {min_length[0]}")
            return

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            while is_valid_position(nx, ny, grid_size) and grid[ny][nx] == 1:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    dfs((nx, ny), target, path + [(nx, ny)], visited)
                    visited.remove((nx, ny))
                nx += dx
                ny += dy

    visited = set()
    visited.add(start)
    dfs(start, target, [start], visited)

    if shortest_path:
        logger.info(f"Brute-Force DFS Shortest path length: {min_length[0]}")
        logger.info(f"Brute-Force DFS Path: {shortest_path}")
        return min_length[0], shortest_path
    else:
        logger.info("Brute-Force DFS: No path found")
        return -1, []

def bfs(grid, grid_size, start, target):
    """
    Breadth-First Search (BFS) Approach.

    Efficiently finds the shortest path in unweighted graphs.
    """

    logger.info("Starting BFS Approach")
    queue = deque()
    queue.append((start[0], start[1], 0))
    visited = [[False for _ in range(grid_size['cols'])] for _ in range(grid_size['rows'])]
    parents = {}
    visited[start[1]][start[0]] = True

    while queue:
        x, y, length = queue.popleft()
        logger.debug(f"Dequeued: ({x}, {y}), Current Path Length: {length}")

        if (x, y) == target:
            path = reconstruct_path(parents, (x, y), start)
            logger.info(f"BFS Shortest path length: {length}")
            logger.info(f"BFS Path: {path}")
            return length, path

        for dx, dy in DIRECTIONS:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy

                if not is_valid_position(nx, ny, grid_size):
                    break  # Out of bounds

                if grid[ny][nx] == 0:
                    break  # Blocked cell

                if not visited[ny][nx]:
                    visited[ny][nx] = True
                    parents[(nx, ny)] = (x, y)
                    queue.append((nx, ny, length + 1))
                    logger.debug(f"Enqueued: ({nx}, {ny}), New Path Length: {length + 1}")

    logger.info("BFS: No path found")
    return -1, []

def heuristic(a, b):
    """
    Heuristic function for A* Search: Euclidean distance.
    """
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

def a_star_search(grid, grid_size, start, target):
    """
    A* Search Algorithm Approach.

    Utilizes a heuristic to potentially reduce the search space.
    """

    logger.info("Starting A* Search Approach")
    open_set = []
    heapq.heappush(open_set, (heuristic(start, target), 0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        logger.debug(f"Popped from heap: {current}, f: {current_f}, g: {current_g}")

        if current == target:
            path = reconstruct_path(came_from, current, start)
            logger.info(f"A* Search Shortest path length: {current_g}")
            logger.info(f"A* Search Path: {path}")
            return current_g, path

        if current in visited:
            continue

        visited.add(current)

        for dx, dy in DIRECTIONS:
            nx, ny = current
            while True:
                nx += dx
                ny += dy

                if not is_valid_position(nx, ny, grid_size):
                    break  # Out of bounds

                if grid[ny][nx] == 0:
                    break  # Blocked cell

                neighbor = (nx, ny)
                tentative_g_score = current_g + 1

                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue  # Not a better path

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, target)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
                logger.debug(f"Pushed to heap: {neighbor}, f: {f_score}, g: {tentative_g_score}")

    logger.info("A* Search: No path found")
    return -1, []

def bidirectional_bfs(grid, grid_size, start, target):
    """
    Bidirectional BFS Approach.

    Simultaneously searches from the start and target positions to potentially reduce search space.
    """

    logger.info("Starting Bidirectional BFS Approach")
    if start == target:
        logger.info("Start and Target positions are the same")
        return 0, [start]

    # Initialize frontiers
    frontier_start = deque()
    frontier_end = deque()
    frontier_start.append((start[0], start[1], 0))
    frontier_end.append((target[0], target[1], 0))

    # Visited dictionaries
    visited_start = {start: 0}
    visited_end = {target: 0}

    # Parents dictionaries for path reconstruction
    parents_start = {}
    parents_end = {}

    while frontier_start and frontier_end:
        # Expand from start
        x, y, length = frontier_start.popleft()
        logger.debug(f"Start BFS dequeued: ({x}, {y}), Length: {length}")

        for dx, dy in DIRECTIONS:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy

                if not is_valid_position(nx, ny, grid_size):
                    break

                if grid[ny][nx] == 0:
                    break

                neighbor = (nx, ny)
                if neighbor not in visited_start:
                    visited_start[neighbor] = length + 1
                    parents_start[neighbor] = (x, y)
                    frontier_start.append((nx, ny, length + 1))
                    logger.debug(f"Start BFS enqueued: {neighbor}, Length: {length + 1}")

                # Check for intersection
                if neighbor in visited_end:
                    total_length = visited_start[neighbor] + visited_end[neighbor]
                    path_start = reconstruct_path(parents_start, neighbor, start)
                    path_end = reconstruct_path(parents_end, neighbor, target)
                    path_end.reverse()
                    full_path = path_start + path_end[1:]
                    logger.info(f"Bidirectional BFS Shortest path length: {total_length}")
                    logger.info(f"Bidirectional BFS Path: {full_path}")
                    return total_length, full_path

        # Expand from end
        x, y, length = frontier_end.popleft()
        logger.debug(f"End BFS dequeued: ({x}, {y}), Length: {length}")

        for dx, dy in DIRECTIONS:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy

                if not is_valid_position(nx, ny, grid_size):
                    break

                if grid[ny][nx] == 0:
                    break

                neighbor = (nx, ny)
                if neighbor not in visited_end:
                    visited_end[neighbor] = length + 1
                    parents_end[neighbor] = (x, y)
                    frontier_end.append((nx, ny, length + 1))
                    logger.debug(f"End BFS enqueued: {neighbor}, Length: {length + 1}")

                # Check for intersection
                if neighbor in visited_start:
                    total_length = visited_start[neighbor] + visited_end[neighbor]
                    path_start = reconstruct_path(parents_start, neighbor, start)
                    path_end = reconstruct_path(parents_end, neighbor, target)
                    path_end.reverse()
                    full_path = path_start + path_end[1:]
                    logger.info(f"Bidirectional BFS Shortest path length: {total_length}")
                    logger.info(f"Bidirectional BFS Path: {full_path}")
                    return total_length, full_path

    logger.info("Bidirectional BFS: No path found")
    return -1, []

def main():
    """Main function to parse arguments and execute the chosen algorithm."""
    parser = argparse.ArgumentParser(
        description="Find the shortest path for a Queen on a binary grid using various algorithms."
    )
    parser.add_argument(
        '-a', '--algorithm',
        type=str,
        choices=['bfs', 'brute-force', 'astar', 'bidirectional'],
        default='bfs',
        help='Algorithm to use for finding the shortest path.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG level) logging.'
    )
    parser.add_argument(
        '-g', '--grid',
        type=str,
        default=None,
        help='Path to JSON file containing the grid configuration.'
    )
    parser.add_argument(
        '-s', '--start',
        type=int,
        nargs=2,
        metavar=('X', 'Y'),
        default=None,
        help='Starting position as two integers: X Y (0-based indices). Default is (0,0).'
    )
    parser.add_argument(
        '-t', '--target',
        type=int,
        nargs=2,
        metavar=('X', 'Y'),
        default=None,
        help='Target position as two integers: X Y (0-based indices). Default is (cols-1, rows-1).'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    logger.info(f"Selected Algorithm: {args.algorithm.upper()}")

    # Load grid
    if args.grid:
        grid = load_grid(args.grid)
        logger.debug(f"Loaded grid from '{args.grid}':")
    else:
        grid = DEFAULT_GRID
        logger.debug("Using default grid:")

    # Log grid if in debug mode
    if logger.level == logging.DEBUG:
        for idx, row in enumerate(grid):
            logger.debug(f"Row {idx}: {row}")

    # Determine grid size
    grid_size = {
        'rows': len(grid),
        'cols': len(grid[0]) if grid else 0
    }

    # Validate that all rows have the same number of columns
    for row in grid:
        if len(row) != grid_size['cols']:
            logger.error("All rows in the grid must have the same number of columns.")
            sys.exit(1)

    # Set start and target positions
    if args.start:
        start = tuple(args.start)
        if not is_valid_position(start[0], start[1], grid_size):
            logger.error("Start position is out of grid bounds.")
            sys.exit(1)
    else:
        start = START_POS_DEFAULT
        # If default start is out of bounds for custom grid, adjust
        if not is_valid_position(start[0], start[1], grid_size):
            start = (0, 0)
            logger.warning("Default start position (0,0) is out of grid bounds. Adjusting start position to (0,0).")

    if args.target:
        target = tuple(args.target)
        if not is_valid_position(target[0], target[1], grid_size):
            logger.error("Target position is out of grid bounds.")
            sys.exit(1)
    else:
        target = (grid_size['cols'] - 1, grid_size['rows'] - 1)
        # If default target is out of bounds for custom grid, adjust
        if not is_valid_position(target[0], target[1], grid_size):
            target = (grid_size['cols'] - 1, grid_size['rows'] - 1)
            logger.warning(f"Default target position ({grid_size['cols'] -1},{grid_size['rows'] -1}) is out of grid bounds.")

    # Check if start and target positions are traversable
    if grid[start[1]][start[0]] == 0:
        logger.error("Start position is blocked (contains 0). No path possible.")
        sys.exit(1)

    if grid[target[1]][target[0]] == 0:
        logger.error("Target position is blocked (contains 0). No path possible.")
        sys.exit(1)

    try:
        # Start timing
        start_time = time.perf_counter()

        # Execute the chosen algorithm
        if args.algorithm == 'bfs':
            length, path = bfs(grid, grid_size, start, target)
        elif args.algorithm == 'brute-force':
            length, path = brute_force_dfs(grid, grid_size, start, target)
        elif args.algorithm == 'astar':
            length, path = a_star_search(grid, grid_size, start, target)
        elif args.algorithm == 'bidirectional':
            length, path = bidirectional_bfs(grid, grid_size, start, target)
        else:
            logger.error("Invalid algorithm selection.")
            sys.exit(1)

        # End timing
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"Total execution time: {elapsed_time:.6f} seconds")

        if length != -1:
            print(f"\nShortest path length: {length}")
            print("Path taken:")
            for step in path:
                print(step)
        else:
            print("\nNo path found from start to target.")

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user. Exiting gracefully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
