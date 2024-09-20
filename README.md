# Queen Shortest Path Finder

## Overview

`queen_shortest_path.py` is a comprehensive Python implementation to find the shortest path for a Queen on a binary grid using multiple algorithms:

- **Brute-Force (DFS)**
- **Breadth-First Search (BFS)**
- **A\* Search**
- **Bidirectional BFS**

The Queen can move in all eight possible directions: up, down, left, right, and diagonally. The goal is to find the shortest path from the starting position `(0, 0)` (top-left corner) to the target position `(7, 7)` (bottom-right corner) while navigating through cells containing `1`s on the grid.

## Problem Statement

Given an 8x8 binary grid where `1` represents a traversable cell and `0` represents a blocked cell, find the shortest path for a Queen to move from `(0, 0)` to `(7, 7)`. The Queen can move in any of the eight directions (up, down, left, right, and diagonally) as long as it moves only through cells containing `1`s.

## Features

- Multiple pathfinding algorithms implemented:
  - Brute-Force Depth-First Search (DFS)
  - Breadth-First Search (BFS)
  - A\* Search
  - Bidirectional BFS
- Configurable grid via a JSON file
- Default 8x8 grid provided
- Verbose logging option for detailed execution information
- Execution time measurement

## Installation

1. **Clone or Download the Repository**

   Download the `queen_shortest_path.py` script to your local machine.

2. **Install Dependencies**

   Ensure you have Python 3 installed. The script uses only standard Python libraries, so no additional installations are required.

## Usage

Run the script using the command line:

```bash
python queen_shortest_path.py [OPTIONS]
```

### Options

- `-a`, `--algorithm`: Algorithm to use for finding the shortest path.
  - Choices: `bfs`, `brute-force`, `astar`, `bidirectional`
  - Default: `bfs`

- `-v`, `--verbose`: Enable verbose (DEBUG level) logging.

- `-g`, `--grid`: Path to JSON file containing the grid configuration.

- `-s`, `--start`: Starting position as two integers: `X Y` (0-based indices). Default is `(0, 0)`.

- `-t`, `--target`: Target position as two integers: `X Y` (0-based indices). Default is `(cols-1, rows-1)`.

### Examples

1. **Using Default Grid**

   ```bash
   python queen_shortest_path.py
   ```

2. **Using A* Search Algorithm with Verbose Logging**

   ```bash
   python queen_shortest_path.py -a astar -v
   ```

3. **Specifying a Custom Grid and Positions**

   ```bash
   python queen_shortest_path.py -g custom_grid.json -s 0 0 -t 7 7
   ```

## Custom Grid Configuration

You can define your custom grid by creating a JSON file containing a 2D list of integers (`0`s and `1`s).

### Example `custom_grid.json`

```json
[
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1]
]
```

## Author

**Samir Musali**  
Date: 2024-09-20

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
