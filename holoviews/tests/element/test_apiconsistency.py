"""
Tests to make sure all components follow the appropriate API
"""
from holoviews import element
from holoviews.element import __all__ as all_elements
from holoviews.element.comparison import ComparisonTestCase


class TestParameterDeclarations(ComparisonTestCase):

    def test_element_group_parameter_declared_constant(self):
        for element_name in all_elements:
            el = getattr(element, element_name)
            self.assertEqual(el.param['group'].constant, True,
                             msg=f'Group parameter of element {element_name} not constant')

    def test_element_label_parameter_declared_constant(self):
        """
        Checking all elements in case LabelledData.label is redefined
        """
        for element_name in all_elements:
            el = getattr(element, element_name)
            self.assertEqual(el.param['label'].constant, True,
                             msg=f'Label parameter of element {element_name} not constant')
