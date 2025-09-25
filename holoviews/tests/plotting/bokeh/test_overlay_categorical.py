import numpy as np
import pandas as pd

from holoviews.element import Points
from holoviews.util.transform import dim

from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer


class TestOverlayCategoricalColorMapping(LoggingComparisonTestCase, TestBokehPlot):
    """Tests for categorical color mapping in overlays (Issue #6691)."""

    def test_overlay_categorical_subset_factors(self):
        """Test that overlaying Points with different categorical subsets 
        maintains correct color mapping factors for each element."""
        
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.uniform(0, 10, 100),
            'y': np.random.uniform(0, 10, 100),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100)
        })
        
        # Create first plot with all categories
        p1 = Points(data, ['x', 'y'], 'category').opts(
            color=dim('category'), show_legend=True
        )
        
        # Create second plot with subset of categories  
        subset_data = data[data.category.isin(['C', 'D', 'E'])]
        p2 = Points(subset_data, ['x', 'y'], 'category').opts(
            color=dim('category'), show_legend=True
        )
        
        # Create overlay
        overlay = p1 * p2
        overlay_plot = bokeh_renderer.get_plot(overlay)
        
        # Get individual subplot color mappers
        p1_plot = overlay_plot.subplots[('Points', 'I')]
        p2_plot = overlay_plot.subplots[('Points', 'II')]
        
        p1_factors = set(p1_plot.handles['color_color_mapper'].factors)
        p2_factors = set(p2_plot.handles['color_color_mapper'].factors)
        
        # Verify each plot has correct factors
        expected_p1_factors = set(data.category.unique())
        expected_p2_factors = set(subset_data.category.unique())
        
        self.assertEqual(p1_factors, expected_p1_factors,
                        "P1 should have factors for all categories in its data")
        self.assertEqual(p2_factors, expected_p2_factors,
                        "P2 should have factors only for categories in its subset data")
        
        # P2 should not have factors for categories not in its data
        p1_only_factors = expected_p1_factors - expected_p2_factors  
        self.assertFalse(p1_only_factors.intersection(p2_factors),
                        "P2 should not have factors for categories not in its data")

    def test_overlay_categorical_disjoint_factors(self):
        """Test overlaying Points with completely disjoint categorical sets."""
        
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6],
            'y': [1, 2, 3, 4, 5, 6],
            'category': ['A', 'B', 'C', 'X', 'Y', 'Z']
        })
        
        # Create plots with disjoint categories
        p1_data = data[data.category.isin(['A', 'B', 'C'])]
        p2_data = data[data.category.isin(['X', 'Y', 'Z'])]
        
        p1 = Points(p1_data, ['x', 'y'], 'category').opts(
            color=dim('category'), show_legend=True
        )
        p2 = Points(p2_data, ['x', 'y'], 'category').opts(
            color=dim('category'), show_legend=True
        )
        
        overlay = p1 * p2
        overlay_plot = bokeh_renderer.get_plot(overlay)
        
        p1_plot = overlay_plot.subplots[('Points', 'I')]
        p2_plot = overlay_plot.subplots[('Points', 'II')]
        
        p1_factors = set(p1_plot.handles['color_color_mapper'].factors)
        p2_factors = set(p2_plot.handles['color_color_mapper'].factors)
        
        # Verify disjoint factors
        self.assertEqual(p1_factors, {'A', 'B', 'C'})
        self.assertEqual(p2_factors, {'X', 'Y', 'Z'})
        self.assertTrue(p1_factors.isdisjoint(p2_factors),
                       "Factors should be completely disjoint")

    def test_overlay_categorical_same_factors(self):
        """Test that overlaying Points with same categories still works correctly."""
        
        data1 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1, 2, 3],
            'category': ['A', 'B', 'C']
        })
        
        data2 = pd.DataFrame({
            'x': [4, 5, 6], 
            'y': [4, 5, 6],
            'category': ['A', 'B', 'C']  # Same categories
        })
        
        p1 = Points(data1, ['x', 'y'], 'category').opts(
            color=dim('category'), show_legend=True
        )
        p2 = Points(data2, ['x', 'y'], 'category').opts(
            color=dim('category'), show_legend=True
        )
        
        overlay = p1 * p2
        overlay_plot = bokeh_renderer.get_plot(overlay)
        
        p1_plot = overlay_plot.subplots[('Points', 'I')]
        p2_plot = overlay_plot.subplots[('Points', 'II')]
        
        p1_factors = set(p1_plot.handles['color_color_mapper'].factors)
        p2_factors = set(p2_plot.handles['color_color_mapper'].factors)
        
        # Both should have the same factors when data has same categories
        expected_factors = {'A', 'B', 'C'}
        self.assertEqual(p1_factors, expected_factors)
        self.assertEqual(p2_factors, expected_factors)