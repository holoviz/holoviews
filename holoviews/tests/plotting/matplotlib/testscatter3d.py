from holoviews.element import Scatter3D

from .testplot import TestMPLPlot, mpl_renderer


class TestPointPlot(TestMPLPlot):

    def test_scatter3d_padding_square(self):
        scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).options(padding=0.1)
        plot = mpl_renderer.get_plot(scatter3d)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        z_range = plot.handles['axis'].get_zlim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
        self.assertEqual(z_range[0], 1.8)
        self.assertEqual(z_range[1], 4.2)

    def test_curve_padding_square_per_axis(self):
        curve = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).options(padding=((0, 0.1), (0.1, 0.2), (0.2, 0.3)))
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        z_range = plot.handles['axis'].get_zlim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.4)
        self.assertEqual(z_range[0], 1.6)
        self.assertEqual(z_range[1], 4.6)

    def test_scatter3d_padding_hard_zrange(self):
        scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).redim.range(z=(0, 3)).options(padding=0.1)
        plot = mpl_renderer.get_plot(scatter3d)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        z_range = plot.handles['axis'].get_zlim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
        self.assertEqual(z_range[0], 0)
        self.assertEqual(z_range[1], 3)

    def test_scatter3d_padding_soft_zrange(self):
        scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).redim.soft_range(z=(0, 3)).options(padding=0.1)
        plot = mpl_renderer.get_plot(scatter3d)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        z_range = plot.handles['axis'].get_zlim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
        self.assertEqual(z_range[0], 0)
        self.assertEqual(z_range[1], 4.2)

    def test_scatter3d_padding_unequal(self):
        scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).options(padding=(0.05, 0.1, 0.2))
        plot = mpl_renderer.get_plot(scatter3d)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        z_range = plot.handles['axis'].get_zlim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
        self.assertEqual(z_range[0], 1.6)
        self.assertEqual(z_range[1], 4.4)

    def test_scatter3d_padding_nonsquare(self):
        scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).options(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(scatter3d)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        z_range = plot.handles['axis'].get_zlim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
        self.assertEqual(z_range[0], 1.8)
        self.assertEqual(z_range[1], 4.2)

    def test_scatter3d_padding_logz(self):
        scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).options(padding=0.1, logz=True)
        plot = mpl_renderer.get_plot(scatter3d)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        z_range = plot.handles['axis'].get_zlim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
        self.assertEqual(z_range[0], 1.8660659830736146)
        self.assertEqual(z_range[1], 4.2870938501451725)

