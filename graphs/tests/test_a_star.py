"""
Unit tests for the A* pathfinding algorithm (graphs.a_star.search).

This module tests the search function which implements the A* algorithm
for finding the shortest path on a 2D grid while avoiding obstacles.
"""

import pytest

from graphs.a_star import search


# Helper functions
def create_manhattan_heuristic(grid, goal):
    """
    Create a Manhattan distance heuristic for a given grid and goal.
    
    Args:
        grid: 2D grid where 0 is traversable and 1 is obstacle
        goal: Goal position [row, col]
    
    Returns:
        2D heuristic matrix with Manhattan distances and obstacle penalties
    """
    heuristic = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            heuristic[i][j] = abs(i - goal[0]) + abs(j - goal[1])
            if grid[i][j] == 1:
                heuristic[i][j] = 99
    return heuristic


def is_valid_path(path, grid, init, goal):
    """
    Validate that a path is correct.
    
    Args:
        path: List of [row, col] positions
        grid: 2D grid
        init: Initial position [row, col]
        goal: Goal position [row, col]
    
    Returns:
        True if path is valid, False otherwise
    """
    if not path:
        return False
    if path[0] != init or path[-1] != goal:
        return False
    for pos in path:
        if grid[pos[0]][pos[1]] != 0:
            return False
    return True


def is_path_continuous(path):
    """
    Check that all consecutive positions in path are adjacent (no diagonal moves).
    
    Args:
        path: List of [row, col] positions
    
    Returns:
        True if path is continuous, False otherwise
    """
    if len(path) <= 1:
        return True
    for i in range(len(path) - 1):
        row_diff = abs(path[i + 1][0] - path[i][0])
        col_diff = abs(path[i + 1][1] - path[i][1])
        # Each step should move exactly 1 cell in one direction
        if row_diff + col_diff != 1:
            return False
    return True


def has_no_repeated_cells(path):
    """
    Check that path doesn't visit the same cell twice.
    
    Args:
        path: List of [row, col] positions
    
    Returns:
        True if all cells are unique, False otherwise
    """
    seen = set()
    for pos in path:
        pos_tuple = tuple(pos)
        if pos_tuple in seen:
            return False
        seen.add(pos_tuple)
    return True


# TC-01: Happy Path - Simple Grid Without Obstacles
def test_search_simple_grid_no_obstacles():
    """
    Test finding path on a simple 2x2 grid with no obstacles.
    
    Verifies that the algorithm can find a valid path from init to goal
    on a basic grid without any obstacles.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init position"
    assert path[-1] == goal, "Path should end at goal position"
    assert len(path) == 3, "Path length should be 3 for 2x2 grid"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"
    assert is_path_continuous(path), "Path should be continuous"


# TC-02: Happy Path - Complex Grid With Obstacles (Doctest Scenario)
def test_search_complex_grid_with_obstacles():
    """
    Test the doctest example with a complex grid containing obstacles.
    
    Verifies that the algorithm correctly navigates around obstacles
    and finds the optimal path as documented in the doctest.
    """
    grid = [
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
    ]
    init = [0, 0]
    goal = [4, 5]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    expected_path = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3],
        [3, 3], [2, 3], [2, 4], [2, 5], [3, 5], [4, 5],
    ]
    
    assert path == expected_path, "Path should match expected path from doctest"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"
    assert len(action) == len(grid), "Action grid should have same rows as grid"
    assert len(action[0]) == len(grid[0]), "Action grid should have same cols as grid"


# TC-03: Boundary - Init Same As Goal
def test_search_init_same_as_goal():
    """
    Test when initial position equals goal position.
    
    Verifies that the algorithm handles the trivial case where
    no movement is needed.
    """
    grid = [[0, 0], [0, 0]]
    init = [1, 1]
    goal = [1, 1]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path == [[1, 1]], "Path should contain only the goal position"
    assert len(path) == 1, "Path length should be 1"


# TC-04: Boundary - Minimum Grid Size (1x1)
def test_search_minimum_grid_size():
    """
    Test with the smallest possible grid (1x1) with init at goal.
    
    Verifies that the algorithm handles edge case of minimal grid.
    """
    grid = [[0]]
    init = [0, 0]
    goal = [0, 0]
    cost = 1
    heuristic = [[0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path == [[0, 0]], "Path should be single cell"
    assert action == [[0]], "Action grid should be single cell"


# TC-05: Boundary - Single Row Grid
def test_search_single_row_grid():
    """
    Test grid with only one row (horizontal movement only).
    
    Verifies that the algorithm works correctly when movement
    is restricted to a single dimension.
    """
    grid = [[0, 0, 0, 0, 0]]
    init = [0, 0]
    goal = [0, 4]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    expected_path = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]
    assert path == expected_path, "Path should traverse entire row"
    assert len(path) == 5, "Path length should be 5"


# TC-06: Boundary - Single Column Grid
def test_search_single_column_grid():
    """
    Test grid with only one column (vertical movement only).
    
    Verifies that the algorithm works correctly when movement
    is restricted to a single dimension.
    """
    grid = [[0], [0], [0], [0], [0]]
    init = [0, 0]
    goal = [4, 0]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    expected_path = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    assert path == expected_path, "Path should traverse entire column"
    assert len(path) == 5, "Path length should be 5"


# TC-07: Boundary - Rectangular Grid (Non-Square)
def test_search_rectangular_grid():
    """
    Test grid with different row and column counts (2x5 grid).
    
    Verifies that the algorithm handles non-square grids correctly.
    """
    grid = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    init = [0, 0]
    goal = [1, 4]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# TC-08: Boundary - Large Grid (Performance)
def test_search_large_grid():
    """
    Test with a large 20x20 grid to verify performance and correctness.
    
    Verifies that the algorithm can handle larger grids efficiently
    and still find the optimal path.
    """
    grid = [[0] * 20 for _ in range(20)]
    init = [0, 0]
    goal = [19, 19]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert len(path) == 39, "Optimal path length should be 39 (19+19+1)"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"
    assert is_path_continuous(path), "Path should be continuous"


# TC-09: Exception - Unreachable Goal (Blocked Path)
def test_search_unreachable_goal_blocked():
    """
    Test when goal exists but is completely blocked by obstacles.
    
    Verifies that the algorithm raises ValueError when no path exists.
    """
    grid = [[0, 1], [1, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 99], [99, 0]]
    
    with pytest.raises(ValueError, match="Algorithm is unable to find solution"):
        search(grid, init, goal, cost, heuristic)


# TC-10: Exception - Goal Is An Obstacle
def test_search_goal_is_obstacle():
    """
    Test when goal position is marked as an obstacle in the grid.
    
    Verifies that the algorithm raises ValueError when goal is unreachable.
    """
    grid = [[0, 0], [0, 1]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 99]]
    
    with pytest.raises(ValueError, match="Algorithm is unable to find solution"):
        search(grid, init, goal, cost, heuristic)


# TC-11: Exception - Completely Blocked Grid
def test_search_completely_blocked_grid():
    """
    Test when all cells in the grid are obstacles.
    
    Verifies that the algorithm raises ValueError when no traversable cells exist.
    """
    grid = [[1, 1], [1, 1]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[99, 99], [99, 99]]
    
    with pytest.raises(ValueError, match="Algorithm is unable to find solution"):
        search(grid, init, goal, cost, heuristic)


# TC-12: Exception - Empty Grid
def test_search_empty_grid():
    """
    Test when grid is an empty list.
    
    Verifies behavior with invalid empty grid input.
    Note: Current implementation raises IndexError.
    """
    grid = []
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = []
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# TC-13: Exception - Init Out Of Bounds
def test_search_init_out_of_bounds():
    """
    Test when initial position is outside grid boundaries.
    
    Verifies behavior with invalid init coordinates.
    Note: Current implementation raises IndexError.
    """
    grid = [[0, 0], [0, 0]]
    init = [5, 5]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# TC-14: Exception - Goal Out Of Bounds
def test_search_goal_out_of_bounds():
    """
    Test when goal position is outside grid boundaries.
    
    Verifies that the algorithm raises ValueError when goal is unreachable
    due to being out of bounds.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [5, 5]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(ValueError, match="Algorithm is unable to find solution"):
        search(grid, init, goal, cost, heuristic)


# TC-15: Edge Case - Negative Init Coordinates
def test_search_negative_init_coordinates():
    """
    Test when init contains negative coordinates.
    
    Verifies behavior with negative coordinates.
    Note: Current implementation allows this - negative indices wrap around
    in Python, so init=[-1, 0] accesses grid[1][0]. This documents current
    behavior which may be unintended.
    """
    grid = [[0, 0], [0, 0]]
    init = [-1, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    # This test documents current behavior
    # Negative indices in Python wrap around, so init=[-1, 0] becomes grid[1][0]
    path, action = search(grid, init, goal, cost, heuristic)
    
    # The algorithm accepts negative coordinates due to Python's negative indexing
    assert path[0] == init, "Path should start at init (with negative index)"
    assert path[-1] == goal, "Path should end at goal"


# TC-16: Edge Case - Different Cost Values
def test_search_different_cost_values():
    """
    Test with cost value greater than 1.
    
    Verifies that the algorithm handles different cost values correctly.
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 5
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# TC-17: Edge Case - Zero Cost
def test_search_zero_cost():
    """
    Test with cost value of 0.
    
    Verifies that the algorithm still works with zero cost.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 0
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# TC-18: Edge Case - Negative Cost
def test_search_negative_cost():
    """
    Test with negative cost value.
    
    Verifies behavior with negative cost.
    Note: Current implementation accepts this - may not be intended.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = -1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Documents current behavior - negative cost is accepted
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-19: Edge Case - Large Heuristic Values
def test_search_large_heuristic_values():
    """
    Test with very large heuristic values.
    
    Verifies that the algorithm handles large numbers correctly.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[1000000, 999999], [999999, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# TC-20: Edge Case - Negative Heuristic Values
def test_search_negative_heuristic_values():
    """
    Test with negative heuristic values.
    
    Verifies that the algorithm handles negative heuristics.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[-100, -50], [-50, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-21: Edge Case - Non-Integer Grid Values
def test_search_non_integer_grid_values():
    """
    Test when grid contains non-integer values (floats).
    
    Verifies behavior with float grid values.
    Note: Current behavior treats non-zero as traversable.
    """
    grid = [[0.5, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Documents current behavior - float values are accepted
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-22: Edge Case - String Grid Values
def test_search_string_grid_values():
    """
    Test when grid contains string values.
    
    Verifies behavior with string grid values.
    Note: Current behavior accepts strings - indicates lack of input validation.
    """
    grid = [["a", 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Documents current behavior - string values are accepted
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-23: Complex Scenario - Maze-Like Structure
def test_search_maze_structure():
    """
    Test navigating through a maze with multiple obstacles.
    
    Verifies that the algorithm can find a path through a complex
    maze-like structure.
    """
    grid = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ]
    init = [0, 0]
    goal = [4, 4]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"
    assert len(path) == 17, "Path length should be 17 for this maze"


# TC-24: Verification - Action Grid Correctness
def test_search_action_grid_correctness():
    """
    Verify action grid correctly encodes movement directions.
    
    Verifies that the action grid has correct dimensions and
    action values are in valid range [0, 3].
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert len(action) == len(grid), "Action grid rows should match grid"
    assert len(action[0]) == len(grid[0]), "Action grid cols should match grid"
    
    # Verify action values are in valid range
    for row in action:
        for val in row:
            assert 0 <= val <= 3, f"Action value {val} should be in range [0, 3]"


# TC-25: Verification - Path Continuity
def test_search_path_continuity():
    """
    Verify all consecutive positions in path are adjacent.
    
    Ensures that the path doesn't have any jumps or diagonal movements.
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert is_path_continuous(path), "Path should be continuous with no diagonal moves"


# TC-26: Verification - No Repeated Cells In Path
def test_search_no_repeated_cells():
    """
    Verify path doesn't visit the same cell twice.
    
    Ensures that the path is optimal and doesn't contain cycles.
    """
    grid = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ]
    init = [0, 0]
    goal = [4, 4]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert has_no_repeated_cells(path), "Path should not visit the same cell twice"


# TC-27: Edge Case - Mismatched Heuristic Dimensions
def test_search_mismatched_heuristic_dimensions():
    """
    Test when heuristic matrix has different dimensions than grid.
    
    Verifies behavior with mismatched input dimensions.
    Note: Current behavior may ignore extra columns - indicates lack of validation.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1, 2], [1, 0, 1]]  # 3 columns instead of 2
    
    # This test documents current behavior
    # May raise IndexError or work with partial heuristic
    try:
        path, action = search(grid, init, goal, cost, heuristic)
        assert path[0] == init, "Path should start at init"
        assert path[-1] == goal, "Path should end at goal"
    except IndexError:
        # Also acceptable - dimensions don't match
        pass


# TC-28: Edge Case - Init Is Obstacle
def test_search_init_is_obstacle():
    """
    Test when initial position is marked as obstacle in grid.
    
    Verifies behavior when starting on an obstacle.
    Note: Current behavior doesn't validate init cell - the algorithm
    starts from the init position even if it's marked as an obstacle.
    This documents current behavior which may be unintended.
    """
    grid = [[1, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[99, 1], [1, 0]]
    
    # This test documents current behavior
    # The algorithm doesn't check if init is an obstacle
    path, action = search(grid, init, goal, cost, heuristic)
    
    # The algorithm accepts starting on an obstacle
    assert path[0] == init, "Path should start at init (even if obstacle)"
    assert path[-1] == goal, "Path should end at goal"


# Additional test: Parametrized test for various grid sizes
@pytest.mark.parametrize("size,expected_path_length", [
    ((2, 2), 3),
    ((3, 3), 5),
    ((4, 4), 7),
    ((5, 5), 9),
])
def test_search_various_grid_sizes(size, expected_path_length):
    """
    Test A* algorithm on various grid sizes.
    
    Uses parametrized tests to verify the algorithm works correctly
    on different grid sizes with optimal path lengths.
    """
    rows, cols = size
    grid = [[0] * cols for _ in range(rows)]
    init = [0, 0]
    goal = [rows - 1, cols - 1]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert len(path) == expected_path_length, f"Path length should be {expected_path_length}"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# Additional test: Verify action grid can reconstruct path
def test_search_action_grid_reconstructs_path():
    """
    Verify that the action grid can be used to reconstruct the path.
    
    This tests that the action grid correctly encodes the path taken.
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Reconstruct path from action grid
    DIRECTIONS = [
        [-1, 0],  # left
        [0, -1],  # down
        [1, 0],  # right
        [0, 1],  # up
    ]
    
    reconstructed = []
    x, y = goal
    reconstructed.append([x, y])
    
    while x != init[0] or y != init[1]:
        act = action[x][y]
        x = x - DIRECTIONS[act][0]
        y = y - DIRECTIONS[act][1]
        reconstructed.append([x, y])
    
    reconstructed.reverse()
    
    assert reconstructed == path, "Reconstructed path should match original path"


# Additional test: Verify path avoids obstacles
def test_search_path_avoids_obstacles():
    """
    Verify that the found path avoids all obstacles.
    
    Tests a grid with scattered obstacles to ensure the path
    doesn't pass through any obstacle cells.
    """
    grid = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    init = [0, 0]
    goal = [4, 4]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Verify no cell in path is an obstacle
    for pos in path:
        assert grid[pos[0]][pos[1]] == 0, f"Path position {pos} should not be an obstacle"
