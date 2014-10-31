import unittest

from holoviews.core import BoundingBox, View
from holoviews.view import Annotation


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


class AnnotationTest(unittest.TestCase):

    def setUp(self):
        self.constructor = dict(boxes=[BoundingBox(), ((-0.25,-0.25),(0.25, 0.25))],
                                vlines = [-0.1, 0.2], hlines = [0.2, -0.1],
                                arrows = [((0.2,0.2),{'text':'arrow','points':20})])
        self.data =  [
            ('line', ((0.5, -0.5), (0.5, 0.5), (-0.5, 0.5),
                      (-0.5, -0.5), (0.5, -0.5)), None),
            ('line', ((0.25, -0.25), (0.25, 0.25), (-0.25, 0.25),
                      (-0.25, -0.25), (0.25, -0.25)), None),
            ('vline', -0.1, None), ('vline', 0.2, None),
            ('hline', 0.2, None), ('hline', -0.1, None),
            ('<', 'arrow', (0.2, 0.2), 20, '->', None)]

    def test_annotation_init(self):
        annotation = Annotation(**self.constructor)
        self.assertEqual(annotation.data, self.data)


    def test_annotation_add_box(self):
        annotation = Annotation(**self.constructor)
        annotation.box(BoundingBox(radius=0.75))
        data = self.data[:]
        data.append(('line', ((0.75, -0.75), (0.75, 0.75),
                              (-0.75, 0.75), (-0.75, -0.75),
                              (0.75, -0.75)), None))
        self.assertEqual(annotation.data, data)

    def test_annotation_add_vline(self):
        annotation = Annotation(**self.constructor)
        annotation.vline(0.99)
        data = self.data[:]
        data.append(('vline', 0.99, None))
        self.assertEqual(annotation.data, data)


    def test_annotation_add_hline_interval(self):
        annotation = Annotation(**self.constructor)
        annotation.hline(-0.8, {'Time':(10,20)})
        data = self.data[:]
        data.append(('hline', -0.8, {'Time':(10, 20)}))
        self.assertEqual(annotation.data, data)


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
