"""
Tests to make sure all components follow the appropriate API
"""

import pytest

import holoviews as hv


@pytest.mark.parametrize("element_name", sorted(hv.elements_list))
def test_element_group_parameter_declared_constant(element_name):
    el = getattr(hv.element, element_name)
    assert el.param["group"].constant


@pytest.mark.parametrize("element_name", sorted(hv.elements_list))
def test_element_label_parameter_declared_constant(element_name):
    """
    Checking all elements in case LabelledData.label is redefined
    """
    el = getattr(hv.element, element_name)
    assert el.param["label"].constant
