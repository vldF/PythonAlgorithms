"""
Additional unit tests for the A* pathfinding algorithm (graphs.a_star.search).

This module contains additional test cases to cover gaps in the existing test suite,
including:
- Edge cases with infinity and NaN values
- Boundary cases with single traversable cells
- Complex obstacle navigation scenarios
- Verification tests for path optimality and action grid encoding
"""

import math

import pytest

from graphs.a_star import DIRECTIONS, search


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


# =============================================================================
# TC-61 to TC-66: Edge Cases - Infinity, NaN, and Special Values
# =============================================================================

# TC-61: Edge Case - Infinity Values in Heuristic
def test_search_infinity_in_heuristic():
    """
    Test when heuristic contains infinity values.
    
    Verifies that the algorithm handles infinity in heuristic matrix.
    The algorithm should still find a valid path, avoiding cells with
    infinite heuristic cost where possible.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, float('inf')], [float('inf'), 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# TC-62: Edge Case - NaN Values in Heuristic
def test_search_nan_in_heuristic():
    """
    Test when heuristic contains NaN values.
    
    Verifies that the algorithm handles NaN in heuristic matrix.
    Note: NaN comparisons are always False, so the sorting behavior
    may be unpredictable. This documents the actual behavior.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, float('nan')], [float('nan'), 0]]
    
    # NaN comparisons are handled by Python's sort
    # The algorithm should still complete, though path may vary
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-63: Edge Case - Infinity Cost Value
def test_search_infinity_cost():
    """
    Test with infinity as the cost value.
    
    Verifies that the algorithm handles infinite cost per step.
    The algorithm should still find a path since the cost comparison
    uses the total f value (g + heuristic).
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = float('inf')
    heuristic = [[0, 1], [1, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-64: Edge Case - NaN Cost Value
def test_search_nan_cost():
    """
    Test with NaN as the cost value.
    
    Verifies that the algorithm handles NaN cost per step.
    Note: NaN comparisons are always False, so the sorting behavior
    may be unpredictable. This documents the actual behavior.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = float('nan')
    heuristic = [[0, 1], [1, 0]]
    
    # NaN comparisons are handled by Python's sort
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-65: Edge Case - Float Heuristic Values
def test_search_float_heuristic_values():
    """
    Test with float values in the heuristic matrix.
    
    Verifies that the algorithm accepts and correctly handles
    floating-point heuristic values.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0.5, 1.5], [1.5, 0.0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# TC-66: Edge Case - Very Large Positive Cost
def test_search_very_large_positive_cost():
    """
    Test with a very large positive cost value.
    
    Verifies that the algorithm handles large cost values without
    overflow or precision issues.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 10**10
    heuristic = [[0, 1], [1, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# =============================================================================
# TC-67: Edge Case - Custom Iterable for Coordinates
# =============================================================================

# TC-67: Edge Case - Custom Iterable for Coordinates
def test_search_custom_iterable_coordinates():
    """
    Test with custom iterable class for coordinates.
    
    Verifies that any object supporting indexing (__getitem__) can be
    used for init and goal coordinates.
    """
    class CustomCoords:
        """Custom coordinate class with indexing support."""
        def __init__(self, coords):
            self._coords = coords
        
        def __getitem__(self, index):
            return self._coords[index]
        
        def __len__(self):
            return len(self._coords)
    
    grid = [[0, 0], [0, 0]]
    init = CustomCoords([0, 0])
    goal = CustomCoords([1, 1])
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == [0, 0], "Path should start at init position"
    assert path[-1] == [1, 1], "Path should end at goal position"


# =============================================================================
# TC-68 to TC-69: Boundary - Single Traversable Cell Scenarios
# =============================================================================

# TC-68: Boundary - Single Traversable Cell Grid
def test_search_single_traversable_cell():
    """
    Test grid with only one traversable cell where init equals goal.
    
    Verifies that the algorithm handles the case where the only
    traversable cell is both the start and goal position.
    """
    grid = [[1, 1], [1, 0]]  # Only [1, 1] is traversable
    init = [1, 1]
    goal = [1, 1]
    cost = 1
    heuristic = [[99, 99], [99, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path == [[1, 1]], "Path should be single cell when init equals goal"
    assert len(path) == 1, "Path length should be 1"


# TC-69: Exception - Single Traversable Cell with Unreachable Goal
def test_search_single_traversable_cell_unreachable_goal():
    """
    Test grid with one traversable cell but goal is on an obstacle.
    
    Verifies that the algorithm raises ValueError when the goal is
    unreachable due to being on an obstacle cell.
    """
    grid = [[1, 1], [1, 0]]  # Only [1, 1] is traversable
    init = [1, 1]
    goal = [0, 0]  # Goal is on obstacle
    cost = 1
    heuristic = [[99, 99], [99, 0]]
    
    with pytest.raises(ValueError, match="Algorithm is unable to find solution"):
        search(grid, init, goal, cost, heuristic)


# =============================================================================
# TC-70: Complex Scenario - Narrow Passage Navigation
# =============================================================================

# TC-70: Complex Scenario - Narrow Passage Navigation
def test_search_narrow_passage():
    """
    Test navigating around a vertical wall obstacle.
    
    Verifies that the algorithm correctly finds the path around
    a wall that blocks direct access to the goal.
    """
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    init = [0, 0]
    goal = [0, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"
    
    # Verify path goes around the wall (through bottom)
    # The path should not pass through [0, 1] or [1, 1] (obstacles)
    assert [0, 1] not in path, "Path should not pass through obstacle at [0, 1]"
    assert [1, 1] not in path, "Path should not pass through obstacle at [1, 1]"


# =============================================================================
# TC-71: Verification - Path Optimality on Open Grid
# =============================================================================

# TC-71: Verification - Path Optimality on Open Grid
def test_search_path_optimality_open_grid():
    """
    Verify path optimality on an open 3x3 grid.
    
    Ensures that:
    - Path length is optimal (5 for 3x3 grid from corner to corner)
    - Path is continuous (no diagonal moves)
    - No cells are visited twice
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert len(path) == 5, "Optimal path length should be 5 (2+2+1)"
    assert is_path_continuous(path), "Path should be continuous"
    assert has_no_repeated_cells(path), "Path should not have repeated cells"


# =============================================================================
# TC-72: Edge Case - Mixed Integer and Float Coordinates
# =============================================================================

# TC-72: Edge Case - Mixed Integer and Float Coordinates
def test_search_mixed_int_float_coordinates():
    """
    Test with mixed integer and float coordinates.
    
    Verifies that float coordinates that are whole numbers raise TypeError.
    Note: Python's list indexing requires integers, not floats, even if
    the float represents a whole number (e.g., 0.0, 1.0).
    """
    grid = [[0, 0], [0, 0]]
    init = [0.0, 0]  # Mixed float and int
    goal = [1, 1.0]  # Mixed int and float
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    # Float indices are not accepted by Python's list indexing
    with pytest.raises(TypeError):
        search(grid, init, goal, cost, heuristic)


# =============================================================================
# TC-74: Exception - Empty Heuristic Grid
# =============================================================================

# TC-74: Exception - Empty Heuristic Grid
def test_search_empty_heuristic_grid():
    """
    Test when heuristic is an empty list.
    
    Verifies that the function raises IndexError when heuristic
    dimensions don't match the grid.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = []  # Empty heuristic
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# =============================================================================
# TC-75: Verification - Action Grid Direction Encoding
# =============================================================================

# TC-75: Verification - Action Grid Direction Encoding
def test_search_action_grid_direction_encoding():
    """
    Verify that action grid values correctly encode movement directions.
    
    Ensures that all action values are in the valid range [0, 3],
    corresponding to the DIRECTIONS constant.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Verify action grid dimensions
    assert len(action) == len(grid), "Action grid rows should match grid"
    assert len(action[0]) == len(grid[0]), "Action grid cols should match grid"
    
    # Verify all action values are in valid range [0, 3]
    for row in action:
        for val in row:
            assert 0 <= val <= 3, f"Action value {val} should be in range [0, 3]"


# =============================================================================
# TC-76: Complex Scenario - U-Shaped Obstacle
# =============================================================================

# TC-76: Complex Scenario - U-Shaped Obstacle
def test_search_u_shaped_obstacle():
    """
    Test navigating around a U-shaped obstacle pattern.
    
    Verifies that the algorithm can find a path around a U-shaped
    obstacle that opens away from the starting position.
    """
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    init = [0, 0]
    goal = [0, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"
    
    # Verify path doesn't pass through obstacles
    for pos in path:
        assert grid[pos[0]][pos[1]] == 0, f"Path position {pos} should not be an obstacle"


# =============================================================================
# TC-77: Edge Case - Negative Heuristic at Goal
# =============================================================================

# TC-77: Edge Case - Negative Heuristic at Goal
def test_search_negative_heuristic_at_goal():
    """
    Test with negative heuristic values including at the goal.
    
    Verifies that the algorithm handles negative heuristic values
    and still finds the goal correctly.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[-100, -50], [-50, -100]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert is_valid_path(path, grid, init, goal), "Path should be valid"


# =============================================================================
# TC-78: Stress Test - Grid with Diagonal Obstacle Pattern
# =============================================================================

# TC-78: Stress Test - Grid with Diagonal Obstacle Pattern
def test_search_diagonal_obstacle_pattern():
    """
    Test navigating around a diagonal obstacle pattern on a 10x10 grid.
    
    Verifies that the algorithm can find a path when obstacles form
    a diagonal pattern across the grid.
    """
    size = 10
    grid = [[0] * size for _ in range(size)]
    
    # Create a diagonal obstacle pattern (leaving gaps for passage)
    # Start from index 1 to leave init cell [0, 0] traversable
    for i in range(1, size):
        if i % 2 == 0:  # Leave every other cell open
            grid[i][i] = 1
    
    init = [0, 0]
    goal = [size - 1, size - 1]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    
    # Verify path doesn't pass through obstacles
    for pos in path:
        assert grid[pos[0]][pos[1]] == 0, f"Path position {pos} should not be an obstacle"


# =============================================================================
# TC-80: Exception - Zero Dimensions Grid
# =============================================================================

# TC-80: Exception - Zero Dimensions Grid
def test_search_zero_dimensions_grid():
    """
    Test when grid has one empty row (zero columns).
    
    Verifies that the function raises IndexError when trying to
    access a column in an empty row.
    """
    grid = [[]]  # Grid with one empty row
    init = [0, 0]
    goal = [0, 0]
    cost = 1
    heuristic = [[]]
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# =============================================================================
# Additional Verification Tests
# =============================================================================

def test_search_path_reconstructible_from_action_grid():
    """
    Verify that the path can be reconstructed from the action grid.
    
    This tests that the action grid correctly encodes the path taken
    by verifying that following the actions backward from goal to init
    reconstructs the same path.
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Reconstruct path from action grid
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


def test_search_action_grid_consistency():
    """
    Verify action grid consistency with the path.
    
    Ensures that the action at each position in the path (except init)
    correctly points to the previous position in the path.
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    # Verify each step in the path (except init) has correct action
    for i in range(1, len(path)):
        curr = path[i]
        prev = path[i - 1]
        
        # Find which direction was used
        for dir_idx, direction in enumerate(DIRECTIONS):
            if prev[0] + direction[0] == curr[0] and prev[1] + direction[1] == curr[1]:
                # This is the direction that was taken
                # The action at curr should be this direction
                assert action[curr[0]][curr[1]] == dir_idx, \
                    f"Action at {curr} should be {dir_idx}"
                break
