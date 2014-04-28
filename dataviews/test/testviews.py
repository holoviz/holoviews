import unittest
from dataviews.boundingregion import BoundingBox
from dataviews.views import View, Annotation, Layout, Overlay, GridLayout


class ViewTest(unittest.TestCase):

    def test_view(self):
        View('An example of arbitrary data')

    def test_constant_label(self):
        label = 'label'
        view = View('An example of arbitrary data', label=label)
        self.assertEqual(view.label, label)
        try:
            view.label = 'another label'
            raise AssertionError("Label should be a constant parameter.")
        except TypeError: pass

    def test_add_operator(self):
        view1 = View('data1', label='view1')
        view2 = View('data2', label='view2')
        self.assertEqual(type(view1 + view2), GridLayout)



class AnnotationTest(unittest.TestCase):

    def setUp(self):
        self.constructor = dict(boxes=[BoundingBox(), ((-0.25,-0.25),(0.25, 0.25))],
                                vlines = [-0.1, 0.2], hlines = [0.2, -0.1],
                                arrows = [((0.2,0.2),{'text':'arrow','points':20})])
        self.data =  set([
                    ('line', ((0.5, -0.5), (0.5, 0.5), (-0.5, 0.5),
                              (-0.5, -0.5), (0.5, -0.5)), None),
                    ('line', ((0.25, -0.25), (0.25, 0.25), (-0.25, 0.25),
                              (-0.25, -0.25), (0.25, -0.25)), None),
                    ('vline', -0.1, None), ('vline', 0.2, None),
                    ('hline', 0.2, None), ('hline', -0.1, None),
                    ('<', 'arrow', (0.2, 0.2), 20, '->', None)])

    def test_annotation_init(self):
        annotation = Annotation(**self.constructor)
        self.assertEqual(set(annotation.data), self.data)


    def test_annotation_add_box(self):
        annotation = Annotation(**self.constructor)
        annotation.box(BoundingBox(radius=0.75))
        data = self.data | set([('line', ((0.75, -0.75), (0.75, 0.75),
                                          (-0.75, 0.75), (-0.75, -0.75),
                                          (0.75, -0.75)), None)])
        self.assertEqual(set(annotation.data), data)

    def test_annotation_add_vline(self):
        annotation = Annotation(**self.constructor)
        annotation.vline(0.99)
        self.assertEqual(set(annotation.data),
                         self.data | set([('vline', 0.99, None)]))


    def test_annotation_add_hline_interval(self):
        annotation = Annotation(**self.constructor)
        annotation.hline(-0.8, {'Time':(10,20)})
        self.assertEqual(set(annotation.data),
                         self.data | set([('hline', -0.8, (('Time', (10, 20)),))]))



class CompositeTest(unittest.TestCase):
    "For testing of basic composite view types"

    def setUp(self):
        self.data1 ='An example of arbitrary data'
        self.data2 = 'Another example...'
        self.data3 = 'A third example.'

        self.view1 = View(self.data1, label='view1')
        self.view2 = View(self.data2, label='view2')
        self.view3 = View(self.data3, label='view3')


class LayoutTest(CompositeTest):

    def test_layout_single(self):
        Layout([self.view1])

    def test_layout_double(self):
        layout = self.view1 << self.view2
        self.assertEqual(layout.main.data, self.data1)
        self.assertEqual(layout.right.data, self.data2)

    def test_layout_triple(self):
        layout = self.view3 << self.view2 << self.view1
        self.assertEqual(layout.main.data, self.data3)
        self.assertEqual(layout.right.data, self.data2)
        self.assertEqual(layout.top.data, self.data1)

    def test_layout_iter(self):
        layout = self.view3 << self.view2 << self.view1
        for el, data in zip(layout, [self.data3, self.data2, self.data1]):
            self.assertEqual(el.data, data)

    def test_layout_add_operator(self):
        layout1 = self.view3 << self.view2
        layout2 = self.view2 << self.view1
        self.assertEqual(type(layout1 + layout2), GridLayout)



class OverlayTest(CompositeTest):

    def test_overlay(self):
        Overlay([self.view1, self.view2, self.view3])

    def test_overlay_mul(self):
        try:
            (self.view1 * self.view2 * self.view3)
            raise AssertionError("The View class should not support __mul__.")
        except TypeError: pass

    def test_overlay_iter(self):
        views = [self.view1, self.view2, self.view3]
        overlay = Overlay(views)
        for el, v in zip(overlay, views):
            self.assertEqual(el, v)

    def test_overlay_labels(self):
        views = [self.view1, self.view2, self.view3]
        overlay = Overlay(views)
        self.assertEqual(overlay.labels, [v.label for v in views])

    def test_overlay_set(self):
        new_layers = [self.view1, self.view3]
        overlay = Overlay([self.view2])
        overlay.set(new_layers)
        self.assertEqual(len(overlay), len(new_layers))


    def test_overlay_add(self):
        overlay = Overlay([self.view1])
        overlay.add(self.view2)


    def test_overlay_integer_indexing(self):
        overlay = Overlay([self.view1, self.view2, self.view3])
        self.assertEqual(overlay[0], self.view1)
        self.assertEqual(overlay[1], self.view2)
        self.assertEqual(overlay[2], self.view3)
        try:
            overlay[3]
            raise AssertionError("Index should be out of range.")
        except IndexError: pass


    def test_overlay_str_indexing(self):
        overlay = Overlay([self.view1, self.view2, self.view3])

        self.assertEqual(overlay[self.view1.label], self.view1)
        self.assertEqual(overlay[self.view2.label], self.view2)
        self.assertEqual(overlay[self.view3.label], self.view3)

        try:
            overlay['Invalid key']
            raise AssertionError("Index should be an invalid key.")
        except KeyError: pass



class GridLayoutTest(CompositeTest):

    def test_gridlayout_init(self):
        grid = GridLayout([self.view1, self.view2, self.view3, self.view2])
        self.assertEqual(grid.shape, (1,4))

    def test_gridlayout_cols(self):
        GridLayout([self.view1, self.view2,self.view3, self.view2])



if __name__ == "__main__":
    import nose
    nose.runmodule()
