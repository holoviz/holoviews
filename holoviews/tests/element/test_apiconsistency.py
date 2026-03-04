"""
Tests to make sure all components follow the appropriate API
"""
import pytest

from holoviews import element
from holoviews.element import elements_list


@pytest.mark.parametrize("element_name", sorted(elements_list))
def test_element_group_parameter_declared_constant(element_name):
    el = getattr(element, element_name)
    assert el.param['group'].constant


@pytest.mark.parametrize("element_name", sorted(elements_list))
def test_element_label_parameter_declared_constant(element_name):
    """
    Checking all elements in case LabelledData.label is redefined
    """
    el = getattr(element, element_name)
    assert el.param['label'].constant
