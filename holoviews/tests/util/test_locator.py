import numpy as np
import pytest

from ...util.locator import MaxNLocator


@pytest.mark.parametrize(
    "n, vmin, vmax, expected",
    [
        (1, 20, 100, [0, 80, 160]),
        (2, 20, 100, [0, 40, 80, 120]),
        (3, 20, 100, [0, 30, 60, 90, 120]),
        (4, 20, 100, [20, 40, 60, 80, 100]),
        (5, 20, 100, [20, 40, 60, 80, 100]),
        (6, 20, 100, [15, 30, 45, 60, 75, 90, 105]),
        (7, 20, 100, [15, 30, 45, 60, 75, 90, 105]),
        (8, 20, 100, [20, 30, 40, 50, 60, 70, 80, 90, 100]),
        (9, 20, 100, [20, 30, 40, 50, 60, 70, 80, 90, 100]),
        (10, 20, 100, [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104]),
        (1, 1e-4, 1e-3, [0, 1e-3]),
        (2, 1e-4, 1e-3, [0, 5e-4, 1e-3]),
        (3, 1e-4, 1e-3, [0, 3e-4, 6e-4, 9e-4, 1.2e-3]),
        (4, 1e-4, 1e-3, [0, 2.5e-4, 5e-4, 7.5e-4, 1e-3]),
        (5, 1e-4, 1e-3, [0, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3]),
        (6, 1e-4, 1e-3, [0, 1.5e-4, 3e-4, 4.5e-4, 6e-4, 7.5e-4, 9e-4, 1.05e-3]),
        (7, 1e-4, 1e-3, [0, 1.5e-4, 3e-4, 4.5e-4, 6e-4, 7.5e-4, 9e-4, 1.05e-3]),
        (8, 1e-4, 1e-3, [0, 1.5e-4, 3e-4, 4.5e-4, 6e-4, 7.5e-4, 9e-4, 1.05e-3]),
        (9, 1e-4, 1e-3, [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]),
        (10, 1e-4, 1e-3, [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]),
        (2, -1e15, 1e15, [-1e15, 0, 1e15]),
        (4, -1e15, 1e15, [-1e15, -5e14, 0, 5e14, 1e15]),
        (8, -1e15, 1e15, [-1e15, -7.5e14, -5e14, -2.5e14, 0, 2.5e14, 5e14, 7.5e14, 1e15]),
        (5, 0, 0.85e-50, [0, 2e-51, 4e-51, 6e-51, 8e-51, 1e-50]),
        (5, -0.85e-50, 0, [-1e-50, -8e-51, -6e-51, -4e-51, -2e-51, 0]),
        (5, 1.23, 1.23, [1.23]*5),
    ],
)
def test_max_n_locator(n, vmin, vmax, expected):
    locator = MaxNLocator(n)
    ticks = locator.tick_values(vmin, vmax)
    np.testing.assert_almost_equal(ticks, expected)

    # Same results if swap vmin and vmax
    ticks = locator.tick_values(vmax, vmin)
    np.testing.assert_almost_equal(ticks, expected)


@pytest.mark.parametrize("n", (0, -1, -2))
def test_max_n_locator_invalid_n(n):
    with pytest.raises(ValueError):
        _ = MaxNLocator(n)
