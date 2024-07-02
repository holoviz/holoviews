import numpy as np
import pandas as pd

from holoviews.element import Bars

from .test_plot import TestPlotlyPlot


class TestBarsPlot(TestPlotlyPlot):

    def test_bars_plot(self):
        bars = Bars([3, 2, 1])
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], [0, 1, 2])
        self.assertEqual(state['data'][0]['y'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['xaxis']['range'], [None, None])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 3.2])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_bars_plot_inverted(self):
        bars = Bars([3, 2, 1]).opts(invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], [0, 1, 2])
        self.assertEqual(state['data'][0]['x'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 3.2])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')
        self.assertEqual(state['layout']['yaxis']['range'], [None, None])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')

    def test_bars_grouped(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B'])
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], [['A', 'B', 'C', 'C'], ['1', '2', '2', '1']])
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'group')
        self.assertEqual(state['layout']['xaxis']['range'], [None, None])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'A, B')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 4.3])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_bars_grouped_inverted(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).opts(invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], [['A', 'B', 'C', 'C'], ['1', '2', '2', '1']])
        self.assertEqual(state['data'][0]['x'], np.array([1, 2, 3, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'group')
        self.assertEqual(state['layout']['yaxis']['range'], [None, None])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'A, B')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 4.3])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')

    def test_bars_stacked(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).opts(stacked=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][0]['y'], np.array([0, 2, 3]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['x'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][1]['y'], np.array([1, 0, 4]))
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'stack')
        self.assertEqual(state['layout']['xaxis']['range'], [None, None])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'A')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 7.6])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_bars_stacked_inverted(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).opts(stacked=True, invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][0]['x'], np.array([0, 2, 3]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['y'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][1]['x'], np.array([1, 0, 4]))
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'stack')
        self.assertEqual(state['layout']['yaxis']['range'], [None, None])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'A')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 7.6])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')

    def test_visible(self):
        element = Bars([3, 2, 1]).opts(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)

    def test_bars_continuous_data_list_same_interval(self):
        bars = Bars(([0, 1, 2], [10, 20, 30]))
        plot = self._get_plot_state(bars)
        np.testing.assert_equal(plot['data'][0]['x'], [0, 1, 2])
        np.testing.assert_equal(plot['data'][0]['y'], [10, 20, 30])

    def test_bars_continuous_data_list_diff_interval(self):
        bars = Bars(([0, 3, 10], [10, 20, 30]))
        plot = self._get_plot_state(bars)
        np.testing.assert_equal(plot['data'][0]['x'], [0, 3, 10])
        np.testing.assert_equal(plot['data'][0]['y'], [10, 20, 30])

    def test_bars_continuous_datetime(self):
        y = np.random.rand(10)
        bars = Bars((pd.date_range("1/1/2000", periods=10), y))
        plot = self._get_plot_state(bars)
        np.testing.assert_equal(plot['data'][0]['x'], pd.date_range("1/1/2000", periods=10).values.astype(float))
        np.testing.assert_equal(plot['data'][0]['y'], y)

    def test_bars_not_continuous_data_list(self):
        bars = Bars([("A", 1), ("B", 2), ("C", 3)])
        plot = self._get_plot_state(bars)
        np.testing.assert_equal(plot['data'][0]['x'], ["A", "B", "C"])
        np.testing.assert_equal(plot['data'][0]['y'], [1, 2, 3])

    def test_bars_group(self):
        samples = 100

        pets = ["Cat", "Dog", "Hamster", "Rabbit"]
        genders = ["Female", "Male", "N/A"]

        np.random.seed(100)
        pets_sample = np.random.choice(pets, samples)
        gender_sample = np.random.choice(genders, samples)

        bars = Bars(
            (pets_sample, gender_sample, np.ones(samples)), ["Pets", "Gender"]
        ).aggregate(function=np.sum)
        plot = self._get_plot_state(bars)
        np.testing.assert_equal(set(plot['data'][0]['x'][0]), set(pets))
        np.testing.assert_equal(
            plot['data'][0]['y'], np.array([6., 10., 10., 10.,  7., 10.,  6., 10.,  9.,  7.,  8.,  7.])
        )

    def test_bar_group_stacked(self):
        samples = 100

        pets = ["Cat", "Dog", "Hamster", "Rabbit"]
        genders = ["Female", "Male", "N/A"]

        np.random.seed(100)
        pets_sample = np.random.choice(pets, samples)
        gender_sample = np.random.choice(genders, samples)

        bars = (
            Bars((pets_sample, gender_sample, np.ones(samples)), ["Pets", "Gender"])
            .aggregate(function=np.sum)
            .opts(stacked=True)
        )
        plot = self._get_plot_state(bars)
        np.testing.assert_equal(set(plot['data'][0]['x']), set(pets))
        np.testing.assert_equal(plot['data'][0]['y'], np.array([8, 7, 6, 7]))

    def test_bar_color(self):
        data = pd.DataFrame({"A": range(5)})
        bars = Bars(data).opts(color="gold")
        fig = self._get_plot_state(bars)
        data = fig["data"][0]
        assert data["marker"]["color"] == "gold"
