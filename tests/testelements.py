import unittest

from holoviews.core import BoundingBox, ViewableElement
from holoviews.element import VLine, HLine


class ViewableElementTest(unittest.TestCase):

    def test_view(self):
        ViewableElement('An example of arbitrary data')

    def test_constant_label(self):
        label = 'label'
        view = ViewableElement('An example of arbitrary data', label=label)
        self.assertEqual(view.label, label)
        try:
            view.label = 'another label'
            raise AssertionError("Label should be a constant parameter.")
        except TypeError: pass


class AnnotationTest(unittest.TestCase):


    def test_annotation_vline_init(self):  VLine(0.1)

    def test_annotation_add_hline_init(self): HLine(0.1)


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
