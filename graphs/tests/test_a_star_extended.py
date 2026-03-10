"""
Extended unit tests for the A* pathfinding algorithm (graphs.a_star.search).

This module contains additional test cases to cover gaps in the existing test suite,
including:
- Invalid input validation (None values, wrong types)
- Malformed inputs (empty lists, single-element coordinates)
- Edge cases (boolean grid values, negative grid values, tuples)
- Stress tests (very large grids)
- Verification tests (return types, input mutation)
"""

import copy
import doctest
import pytest

from graphs import a_star
from graphs.a_star import search


# Helper functions (duplicated from test_a_star.py for self-containment)
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


# =============================================================================
# TC-32 to TC-35: Invalid Input - None Values
# =============================================================================

# TC-32: Invalid Input - None as Grid
def test_search_none_grid():
    """
    Test when grid is None.
    
    Verifies that the function raises TypeError when grid is None.
    Note: Current implementation doesn't validate input types.
    """
    grid = None
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(TypeError):
        search(grid, init, goal, cost, heuristic)


# TC-33: Invalid Input - None as Init
def test_search_none_init():
    """
    Test when init is None.
    
    Verifies that the function raises TypeError when init is None.
    Note: Current implementation doesn't validate input types.
    """
    grid = [[0, 0], [0, 0]]
    init = None
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(TypeError):
        search(grid, init, goal, cost, heuristic)


# TC-34: Invalid Input - None as Goal
def test_search_none_goal():
    """
    Test when goal is None.
    
    Verifies that the function raises TypeError when goal is None.
    Note: Current implementation doesn't validate input types.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = None
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(TypeError):
        search(grid, init, goal, cost, heuristic)


# TC-35: Invalid Input - None as Heuristic
def test_search_none_heuristic():
    """
    Test when heuristic is None.
    
    Verifies that the function raises TypeError when heuristic is None.
    Note: Current implementation doesn't validate input types.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = None
    
    with pytest.raises(TypeError):
        search(grid, init, goal, cost, heuristic)


# =============================================================================
# TC-36 to TC-40: Malformed Input Validation
# =============================================================================

# TC-36: Invalid Input - Empty Init List
def test_search_empty_init_list():
    """
    Test when init is an empty list.
    
    Verifies that the function raises IndexError when init is empty.
    Note: Current implementation doesn't validate init length.
    """
    grid = [[0, 0], [0, 0]]
    init = []
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# TC-37: Invalid Input - Single Element Init
def test_search_single_element_init():
    """
    Test when init has only one element.
    
    Verifies that the function raises IndexError when init is incomplete.
    Note: Current implementation expects [row, col] format.
    """
    grid = [[0, 0], [0, 0]]
    init = [0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# TC-38: Invalid Input - Empty Goal List
def test_search_empty_goal_list():
    """
    Test when goal is an empty list.
    
    Verifies that the function raises IndexError when goal is empty.
    Note: Current implementation doesn't validate goal length.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = []
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# TC-39: Invalid Input - String as Cost
def test_search_string_cost():
    """
    Test when cost is a string instead of a number.
    
    Verifies that the function raises TypeError when cost is invalid type.
    Note: Current implementation doesn't validate cost type.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = "abc"
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(TypeError):
        search(grid, init, goal, cost, heuristic)


# TC-40: Invalid Input - Float Coordinates
def test_search_float_coordinates():
    """
    Test when init contains float values instead of integers.
    
    Verifies that the function raises TypeError when coordinates are floats.
    Note: Current implementation expects integer indices.
    """
    grid = [[0, 0], [0, 0]]
    init = [0.5, 0.5]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    with pytest.raises(TypeError):
        search(grid, init, goal, cost, heuristic)


# =============================================================================
# TC-41 to TC-46: Edge Cases - Type Flexibility
# =============================================================================

# TC-41: Edge Case - Boolean Grid Values
def test_search_boolean_grid_values():
    """
    Test when grid contains boolean values instead of integers.
    
    Verifies that True is treated as obstacle (1) and False as traversable (0).
    Note: In Python, True == 1 and False == 0, so True is treated as obstacle.
    """
    grid = [[False, True], [False, False]]  # True should be obstacle
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 99], [1, 0]]
    
    # The path should avoid the True cell at [0, 1]
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    # Verify path doesn't go through True (obstacle)
    assert [0, 1] not in path, "Path should not pass through True (obstacle)"


# TC-42: Edge Case - None in Grid
def test_search_none_in_grid():
    """
    Test when grid contains None values.
    
    Verifies behavior with None in grid cells.
    Note: Current implementation compares grid[x2][y2] == 0, so None != 0
    means None is treated as traversable. This documents current behavior.
    """
    grid = [[None, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    # None != 0, so it's treated as traversable
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-43: Edge Case - Negative Grid Values
def test_search_negative_grid_values():
    """
    Test when grid contains negative values.
    
    Verifies that negative values are treated as traversable.
    Note: Current implementation only treats 1 as obstacle.
    """
    grid = [[-1, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    # -1 != 1, so it's treated as traversable
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-44: Edge Case - Grid Value Greater Than 1
def test_search_grid_value_greater_than_one():
    """
    Test when grid contains values greater than 1.
    
    Verifies that only value 1 is treated as obstacle.
    Note: Current implementation only treats 1 as obstacle.
    """
    grid = [[2, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    # 2 != 1, so it's treated as traversable
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-45: Edge Case - Tuple Instead of List for Coordinates
def test_search_tuple_coordinates():
    """
    Test when init and goal are tuples instead of lists.
    
    Verifies that tuples are accepted since they support indexing.
    Note: Tuples work because the implementation only uses indexing.
    """
    grid = [[0, 0], [0, 0]]
    init = (0, 0)  # tuple instead of list
    goal = (1, 1)  # tuple instead of list
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == list(init), "Path should start at init"
    assert path[-1] == list(goal), "Path should end at goal"


# TC-46: Edge Case - Tuple Grid
def test_search_tuple_grid():
    """
    Test when grid is a tuple of tuples instead of list of lists.
    
    Verifies that tuple grids are accepted since they support indexing.
    Note: Tuples work because the implementation only uses indexing.
    """
    grid = ((0, 0), (0, 0))  # tuple of tuples
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# =============================================================================
# TC-47 to TC-49: Edge Cases - Inconsistent Dimensions
# =============================================================================

# TC-47: Edge Case - Empty Row in Grid
def test_search_empty_row_in_grid():
    """
    Test when grid contains an empty row.
    
    Verifies behavior with malformed grid structure.
    Note: Current implementation may raise IndexError.
    """
    grid = [[], [0, 0]]
    init = [1, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[], [0, 0]]
    
    with pytest.raises(IndexError):
        search(grid, init, goal, cost, heuristic)


# TC-48: Edge Case - Inconsistent Row Lengths in Grid
def test_search_inconsistent_row_lengths():
    """
    Test when grid rows have different lengths.
    
    Verifies behavior with jagged grid structure.
    Note: Current implementation doesn't validate grid structure.
    """
    grid = [[0, 0, 0], [0, 0]]  # jagged grid
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1, 2], [1, 0]]
    
    # This may work or raise IndexError depending on path taken
    try:
        path, action = search(grid, init, goal, cost, heuristic)
        assert path[0] == init, "Path should start at init"
        assert path[-1] == goal, "Path should end at goal"
    except IndexError:
        # Also acceptable - jagged grids are invalid
        pass


# TC-49: Edge Case - Inconsistent Heuristic Row Lengths
def test_search_inconsistent_heuristic_row_lengths():
    """
    Test when heuristic rows have different lengths.
    
    Verifies behavior with jagged heuristic structure.
    Note: Current implementation may work if the path doesn't access
    the extra columns. This documents the actual behavior.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1, 2], [1, 0]]  # jagged heuristic
    
    # The algorithm may work if it doesn't need to access the extra columns
    # or may raise IndexError depending on the path taken
    try:
        path, action = search(grid, init, goal, cost, heuristic)
        assert path[0] == init, "Path should start at init"
        assert path[-1] == goal, "Path should end at goal"
    except IndexError:
        # Also acceptable - jagged heuristic is invalid
        pass


# =============================================================================
# TC-50 to TC-52: Edge Cases - Heuristic Edge Cases
# =============================================================================

# TC-50: Edge Case - Zero Heuristic (Dijkstra-like Behavior)
def test_search_zero_heuristic():
    """
    Test with all-zero heuristic matrix.
    
    Verifies that algorithm degrades to Dijkstra's algorithm when heuristic is zero.
    This should still find a valid path, just without heuristic guidance.
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # all zeros
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert len(path) == 5, "Path length should be 5 for 3x3 grid"


# TC-51: Edge Case - Init=Goal on Obstacle
def test_search_init_equals_goal_on_obstacle():
    """
    Test when init equals goal but the cell is marked as obstacle.
    
    Verifies that the algorithm returns immediately without checking obstacle status.
    Note: When init == goal, the algorithm returns immediately without obstacle check.
    """
    grid = [[1]]  # single obstacle cell
    init = [0, 0]
    goal = [0, 0]
    cost = 1
    heuristic = [[99]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path == [[0, 0]], "Path should be single cell when init == goal"
    assert len(path) == 1, "Path length should be 1"


# TC-52: Edge Case - Large Heuristic at Goal
def test_search_large_heuristic_at_goal():
    """
    Test when heuristic has very large value at goal position.
    
    Verifies that the algorithm still finds the goal despite high heuristic cost.
    The algorithm should still reach the goal since it checks goal condition before expanding.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = [[0, 1], [1, 999999]]  # very large at goal
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# =============================================================================
# TC-53 to TC-54: Stress Tests
# =============================================================================

# TC-53: Stress Test - Very Large Grid (100x100)
def test_search_very_large_grid():
    """
    Test with a very large 100x100 grid.
    
    Verifies performance and correctness on large grids.
    The optimal path from [0,0] to [99,99] should have length 199.
    """
    grid = [[0] * 100 for _ in range(100)]
    init = [0, 0]
    goal = [99, 99]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    assert len(path) == 199, "Optimal path length should be 199 (99+99+1)"


# TC-54: Stress Test - Large Grid with Obstacles
def test_search_large_grid_with_obstacles():
    """
    Test with a 50x50 grid containing a vertical wall obstacle.
    
    Verifies performance and correctness when navigating around obstacles
    on a larger grid.
    """
    size = 50
    grid = [[0] * size for _ in range(size)]
    
    # Create a vertical wall obstacle in the middle
    wall_col = size // 2
    for row in range(size - 5):  # Leave a gap at the bottom
        grid[row][wall_col] = 1
    
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
# TC-55 to TC-56: Verification Tests
# =============================================================================

# TC-55: Verification - Return Type
def test_search_return_type():
    """
    Verify that search returns correct types.
    
    Ensures that:
    - path is a list
    - action is a list
    - path elements are lists
    - action elements are lists
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert isinstance(path, list), "path should be a list"
    assert isinstance(action, list), "action should be a list"
    assert all(isinstance(p, list) for p in path), "path elements should be lists"
    assert all(isinstance(row, list) for row in action), "action rows should be lists"
    assert all(isinstance(val, int) for row in action for val in row), \
        "action values should be integers"


# TC-56: Verification - Input Not Modified
def test_search_input_not_modified():
    """
    Verify that search does not modify input parameters.
    
    Ensures that grid, init, goal, and heuristic remain unchanged after the call.
    """
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    init = [0, 0]
    goal = [2, 2]
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    # Create deep copies to compare later
    original_grid = copy.deepcopy(grid)
    original_init = copy.deepcopy(init)
    original_goal = copy.deepcopy(goal)
    original_heuristic = copy.deepcopy(heuristic)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert grid == original_grid, "grid should not be modified"
    assert init == original_init, "init should not be modified"
    assert goal == original_goal, "goal should not be modified"
    assert heuristic == original_heuristic, "heuristic should not be modified"


# =============================================================================
# TC-57 to TC-59: Additional Edge Cases
# =============================================================================

# TC-57: Edge Case - Very Large Negative Cost
def test_search_very_large_negative_cost():
    """
    Test with very large negative cost value.
    
    Verifies behavior with extreme negative cost.
    Note: Current implementation accepts this - documents current behavior.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = -1000000
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-58: Edge Case - Zero Cost and Zero Heuristic
def test_search_zero_cost_zero_heuristic():
    """
    Test with both zero cost and zero heuristic.
    
    Verifies that the algorithm works with all zeros.
    """
    grid = [[0, 0], [0, 0]]
    init = [0, 0]
    goal = [1, 1]
    cost = 0
    heuristic = [[0, 0], [0, 0]]
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"


# TC-59: Edge Case - Spiral Obstacle Pattern
def test_search_spiral_obstacle():
    """
    Test navigating through a spiral-like obstacle arrangement.
    
    Verifies that the algorithm can find a path through complex obstacle patterns.
    """
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    init = [0, 0]
    goal = [4, 3]  # Center of spiral
    cost = 1
    heuristic = create_manhattan_heuristic(grid, goal)
    
    path, action = search(grid, init, goal, cost, heuristic)
    
    assert path[0] == init, "Path should start at init"
    assert path[-1] == goal, "Path should end at goal"
    
    # Verify path doesn't pass through obstacles
    for pos in path:
        assert grid[pos[0]][pos[1]] == 0, f"Path position {pos} should not be an obstacle"


# =============================================================================
# TC-60: Doctest Verification
# =============================================================================

# TC-60: Verification - Doctest Examples Work
def test_search_doctest_examples():
    """
    Verify that all doctest examples in the module pass.
    
    Runs doctest on the a_star module to ensure documentation examples work.
    """
    # Run doctest on the a_star module
    results = doctest.testmod(a_star, verbose=False)
    
    assert results.failed == 0, f"Doctest failed {results.failed} tests"
    assert results.attempted > 0, "At least one doctest should be present"
