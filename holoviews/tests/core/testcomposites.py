"""
Test cases for the composite types built with + and *, i.e. Layout
and Overlay (does *not* test HoloMaps).
"""

from holoviews import Element, Layout, Overlay, HoloMap
from holoviews.element.comparison import ComparisonTestCase


class ElementTestCase(ComparisonTestCase):

    def setUp(self):
        self.el1 = Element('data1')
        self.el2 = Element('data2')
        self.el3 = Element('data3')
        self.el4 = Element('data4', group='ValA')
        self.el5 = Element('data5', group='ValB')
        self.el6 = Element('data6', label='LabelA')
        self.el7 = Element('data7', group='ValA', label='LabelA')
        self.el8 = Element('data8', group='ValA', label='LabelB')

    def test_element_init(self):
        Element('data1')


class LayoutTestCase(ElementTestCase):

    def setUp(self):
        super(LayoutTestCase, self).setUp()

    def test_layouttree_keys_1(self):
        t = self.el1 + self.el2
        self.assertEqual(t.keys(),
                         [('Element', 'I'), ('Element', 'II')])

    def test_layouttree_keys_2(self):
        t = Layout([self.el1, self.el2])
        self.assertEqual(t.keys(),
                         [('Element', 'I'), ('Element', 'II')])

    def test_layouttree_deduplicate(self):
        for i in range(2, 10):
            l = Layout([Element([], label='0') for _ in range(i)])
            self.assertEqual(len(l), i)

    def test_layouttree_values_1(self):
        t = self.el1 + self.el2
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_layouttree_values_2(self):
        t = Layout([self.el1, self.el2])
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_triple_layouttree_keys(self):
        t = self.el1 + self.el2 + self.el3
        expected_keys = [('Element', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_triple_layouttree_values(self):
        t = self.el1 + self.el2 + self.el3
        self.assertEqual(t.values(), [self.el1 , self.el2 , self.el3])

    def test_layouttree_varying_value_keys(self):
        t = self.el1 + self.el4
        self.assertEqual(t.keys(), [('Element', 'I'), ('ValA', 'I')])

    def test_layouttree_varying_value_keys2(self):
        t = self.el4 + self.el5
        self.assertEqual(t.keys(), [('ValA', 'I'), ('ValB', 'I')])

    def test_triple_layouttree_varying_value_keys(self):
        t = self.el1 + self.el4 + self.el2 + self.el3
        expected_keys = [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_four_layouttree_varying_value_values(self):
        t = self.el1 + self.el4 + self.el2 + self.el3
        self.assertEqual(t.values(), [self.el1 , self.el4 , self.el2 , self.el3])

    def test_layouttree_varying_label_keys(self):
        t = self.el1 + self.el6
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'LabelA')])

    def test_triple_layouttree_varying_label_keys(self):
        t = self.el1 + self.el6 + self.el2
        expected_keys = [('Element', 'I'), ('Element', 'LabelA'), ('Element', 'II')]
        self.assertEqual(t.keys(), expected_keys)

    def test_layouttree_varying_label_keys2(self):
        t = self.el7 + self.el8
        self.assertEqual(t.keys(), [('ValA', 'LabelA'), ('ValA', 'LabelB')] )

    def test_layouttree_varying_label_and_values_keys(self):
        t = self.el6 + self.el7 + self.el8
        expected_keys = [('Element', 'LabelA'), ('ValA', 'LabelA'), ('ValA', 'LabelB')]
        self.assertEqual(t.keys(), expected_keys)

    def test_layouttree_varying_label_and_values_values(self):
        t = self.el6 + self.el7 + self.el8
        self.assertEqual(t.values(), [self.el6, self.el7, self.el8])

    def test_layouttree_associativity(self):
        t1 = (self.el1 + self.el2 + self.el3)
        t2 = ((self.el1 + self.el2) + self.el3)
        t3 = (self.el1 + (self.el2 + self.el3))
        self.assertEqual(t1.keys(), t2.keys())
        self.assertEqual(t2.keys(), t3.keys())

    def test_layouttree_constructor1(self):
        t = Layout([self.el1])
        self.assertEqual(t.keys(),  [('Element', 'I')])

    def test_layouttree_constructor2(self):
        t = Layout([self.el8])
        self.assertEqual(t.keys(),  [('ValA', 'LabelB')])

    def test_layouttree_group(self):
        t1 = (self.el1 + self.el2)
        t2 = Layout(list(t1.relabel(group='NewValue')))
        self.assertEqual(t2.keys(), [('NewValue', 'I'), ('NewValue', 'II')])

    def test_layouttree_quadruple_1(self):
        t = self.el1 + self.el1 + self.el1 + self.el1
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II'),
                                    ('Element', 'III'), ('Element', 'IV')])

    def test_layouttree_quadruple_2(self):
        t = self.el6 + self.el6 + self.el6 + self.el6
        self.assertEqual(t.keys(), [('Element', 'LabelA', 'I'),
                                    ('Element', 'LabelA', 'II'),
                                    ('Element', 'LabelA', 'III'),
                                    ('Element', 'LabelA', 'IV')])

    def test_layout_constructor_with_layouts(self):
        layout1 = self.el1 + self.el4
        layout2 = self.el2 + self.el5
        paths = Layout([layout1, layout2]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'),
                                 ('Element', 'II'), ('ValB', 'I')])

    def test_layout_constructor_with_mixed_types(self):
        layout1 = self.el1 + self.el4 + self.el7
        layout2 = self.el2 + self.el5 + self.el8
        paths = Layout([layout1, layout2, self.el3]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'),
                                 ('ValA', 'LabelA'), ('Element', 'II'),
                                 ('ValB', 'I'), ('ValA', 'LabelB'),
                                 ('Element', 'III')])

    def test_layout_constructor_retains_custom_path(self):
        layout = Layout([('Custom', self.el1)])
        paths = Layout([layout, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'I'), ('Element', 'I')])

    def test_layout_constructor_retains_custom_path_with_label(self):
        layout = Layout([('Custom', self.el6)])
        paths = Layout([layout, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'LabelA'), ('Element', 'I')])

    def test_layout_integer_index(self):
        t = self.el1 + self.el2
        self.assertEqual(t[0], self.el1)
        self.assertEqual(t[1], self.el2)

    def test_layout_overlay_element(self):
        t = (self.el1 + self.el2) * self.el3
        self.assertEqual(t, Layout([self.el1*self.el3, self.el2*self.el3]))

    def test_layout_overlay_element_reverse(self):
        t = self.el3 * (self.el1 + self.el2)
        self.assertEqual(t, Layout([self.el3*self.el1, self.el3*self.el2]))

    def test_layout_overlay_overlay(self):
        t = (self.el1 + self.el2) * (self.el3 * self.el4)
        self.assertEqual(t, Layout([self.el1*self.el3*self.el4, self.el2*self.el3*self.el4]))

    def test_layout_overlay_overlay_reverse(self):
        t = (self.el3 * self.el4) * (self.el1 + self.el2)
        self.assertEqual(t, Layout([self.el3*self.el4*self.el1,
                                    self.el3*self.el4*self.el2]))

    def test_layout_overlay_holomap(self):
        t = (self.el1 + self.el2) * HoloMap({0: self.el3})
        self.assertEqual(t, Layout([HoloMap({0: self.el1*self.el3}),
                                    HoloMap({0: self.el2*self.el3})]))

    def test_layout_overlay_holomap_reverse(self):
        t = HoloMap({0: self.el3}) * (self.el1 + self.el2)
        self.assertEqual(t, Layout([HoloMap({0: self.el3*self.el1}),
                                    HoloMap({0: self.el3*self.el2})]))



class OverlayTestCase(ElementTestCase):
    """
    The tests here match those in LayoutTestCase; Overlays inherit
    from Layout and behave in a very similar way (except for being
    associated with * instead of the + operator)
    """

    def setUp(self):
        super(OverlayTestCase, self).setUp()

    def test_overlay_keys(self):
        t = self.el1 * self.el2
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II')])

    def test_overlay_keys_2(self):
        t = Overlay([self.el1, self.el2])
        self.assertEqual(t.keys(),
                         [('Element', 'I'), ('Element', 'II')])

    def test_overlay_values(self):
        t = self.el1 * self.el2
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_overlay_values_2(self):
        t = Overlay([self.el1, self.el2])
        self.assertEqual(t.values(), [self.el1, self.el2])

    def test_triple_overlay_keys(self):
        t = self.el1 * self.el2 * self.el3
        expected_keys = [('Element', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_triple_overlay_values(self):
        t = self.el1 * self.el2 * self.el3
        self.assertEqual(t.values(), [self.el1 , self.el2 , self.el3])

    def test_overlay_varying_value_keys(self):
        t = self.el1 * self.el4
        self.assertEqual(t.keys(), [('Element', 'I'), ('ValA', 'I')])

    def test_overlay_varying_value_keys2(self):
        t = self.el4 * self.el5
        self.assertEqual(t.keys(), [('ValA', 'I'), ('ValB', 'I')])

    def test_triple_overlay_varying_value_keys(self):
        t = self.el1 * self.el4 * self.el2 * self.el3
        expected_keys = [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('Element', 'III')]
        self.assertEqual(t.keys(), expected_keys)

    def test_four_overlay_varying_value_values(self):
        t = self.el1 * self.el4 * self.el2 * self.el3
        self.assertEqual(t.values(), [self.el1 , self.el4 , self.el2 , self.el3])

    def test_overlay_varying_label_keys(self):
        t = self.el1 * self.el6
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'LabelA')])

    def test_triple_overlay_varying_label_keys(self):
        t = self.el1 * self.el6 * self.el2
        expected_keys = [('Element', 'I'), ('Element', 'LabelA'), ('Element', 'II')]
        self.assertEqual(t.keys(), expected_keys)

    def test_overlay_varying_label_keys2(self):
        t = self.el7 * self.el8
        self.assertEqual(t.keys(), [('ValA', 'LabelA'), ('ValA', 'LabelB')] )

    def test_overlay_varying_label_and_values_keys(self):
        t = self.el6 * self.el7 * self.el8
        expected_keys = [('Element', 'LabelA'), ('ValA', 'LabelA'), ('ValA', 'LabelB')]
        self.assertEqual(t.keys(), expected_keys)

    def test_overlay_varying_label_and_values_values(self):
        t = self.el6 * self.el7 * self.el8
        self.assertEqual(t.values(), [self.el6, self.el7, self.el8])

    def test_deep_overlay_keys(self):
        o1 = (self.el1 * self.el2)
        o2 = (self.el1 * self.el2)
        o3 = (self.el1 * self.el2)
        t = (o1 * o2 * o3)
        expected_keys = [('Element', 'I'), ('Element', 'II'), ('Element', 'III'),
                         ('Element', 'IV'), ('Element', 'V'), ('Element', 'VI')]
        self.assertEqual(t.keys(), expected_keys)

    def test_deep_overlay_values(self):
        o1 = (self.el1 * self.el2)
        o2 = (self.el1 * self.el2)
        o3 = (self.el1 * self.el2)
        t = (o1 * o2 * o3)
        self.assertEqual(t.values(), [self.el1 , self.el2,
                                      self.el1 , self.el2,
                                      self.el1 , self.el2])

    def test_overlay_associativity(self):
        o1 = (self.el1 * self.el2 * self.el3)
        o2 = ((self.el1 * self.el2) * self.el3)
        o3 = (self.el1 * (self.el2 * self.el3))
        self.assertEqual(o1.keys(), o2.keys())
        self.assertEqual(o2.keys(), o3.keys())

    def test_overlay_constructor1(self):
        t = Overlay([self.el1])
        self.assertEqual(t.keys(),  [('Element', 'I')])

    def test_overlay_constructor2(self):
        t = Overlay([self.el8])
        self.assertEqual(t.keys(),  [('ValA', 'LabelB')])

    def test_overlay_group(self):
        t1 = (self.el1 * self.el2)
        t2 = Overlay(list(t1.relabel(group='NewValue', depth=1)))
        self.assertEqual(t2.keys(), [('NewValue', 'I'), ('NewValue', 'II')])

    def test_overlay_quadruple_1(self):
        t = self.el1 * self.el1 * self.el1 * self.el1
        self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II'),
                                    ('Element', 'III'), ('Element', 'IV')])

    def test_overlay_quadruple_2(self):
        t = self.el6 * self.el6 * self.el6 * self.el6
        self.assertEqual(t.keys(), [('Element', 'LabelA', 'I'),
                                    ('Element', 'LabelA', 'II'),
                                    ('Element', 'LabelA', 'III'),
                                    ('Element', 'LabelA', 'IV')])

    def test_overlay_constructor_with_layouts(self):
        layout1 = self.el1 + self.el4
        layout2 = self.el2 + self.el5
        paths = Layout([layout1, layout2]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'),
                                 ('Element', 'II'), ('ValB', 'I')])

    def test_overlay_constructor_with_mixed_types(self):
        overlay1 = self.el1 + self.el4 + self.el7
        overlay2 = self.el2 + self.el5 + self.el8
        paths = Layout([overlay1, overlay2, self.el3]).keys()
        self.assertEqual(paths, [('Element', 'I'), ('ValA', 'I'),
                                 ('ValA', 'LabelA'), ('Element', 'II'),
                                 ('ValB', 'I'), ('ValA', 'LabelB'),
                                 ('Element', 'III')])

    def test_overlay_constructor_retains_custom_path(self):
        overlay = Overlay([('Custom', self.el1)])
        paths = Overlay([overlay, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'I'), ('Element', 'I')])

    def test_overlay_constructor_retains_custom_path_with_label(self):
        overlay = Overlay([('Custom', self.el6)])
        paths = Overlay([overlay, self.el2]).keys()
        self.assertEqual(paths, [('Custom', 'LabelA'), ('Element', 'I')])

    def test_overlay_with_holomap(self):
        overlay = Overlay([('Custom', self.el6)])
        composite = overlay * HoloMap({0: Element(None, group='HoloMap')})
        self.assertEqual(composite.last.keys(), [('Custom', 'LabelA'), ('HoloMap', 'I')])

    def test_overlay_id_inheritance(self):
        overlay = Overlay([], id=1)
        self.assertEqual(overlay.clone().id, 1)
        self.assertEqual(overlay.clone()._plot_id, overlay._plot_id)
        self.assertNotEqual(overlay.clone([])._plot_id, overlay._plot_id)


class CompositeTestCase(ElementTestCase):
    """
    Test case for trees involving both + (Layout) and * (Overlay)
    """

    def test_composite1(self):
        t = (self.el1 * self.el2) + (self.el1 * self.el2)
        self.assertEqual(t.keys(),  [('Overlay', 'I'), ('Overlay', 'II')])

    def test_composite_relabelled_value1(self):
        t = (self.el1 * self.el2) + (self.el1 * self.el2).relabel(group='Val2')
        self.assertEqual(t.keys(), [('Overlay', 'I'), ('Val2', 'I')])

    def test_composite_relabelled_label1(self):
        t = ((self.el1 * self.el2)
             + (self.el1 * self.el2).relabel(group='Val1', label='Label2'))
        self.assertEqual(t.keys(), [('Overlay', 'I'), ('Val1', 'Label2')])

    def test_composite_relabelled_label2(self):
        t = ((self.el1 * self.el2).relabel(label='Label1')
             + (self.el1 * self.el2).relabel(group='Val1', label='Label2'))
        self.assertEqual(t.keys(), [('Overlay', 'Label1'), ('Val1', 'Label2')])

    def test_composite_relabelled_value2(self):
        t = ((self.el1 * self.el2).relabel(group='Val1')
             + (self.el1 * self.el2).relabel(group='Val2'))
        self.assertEqual(t.keys(), [('Val1', 'I'), ('Val2', 'I')])

    def test_composite_relabelled_value_and_label(self):
        t = ((self.el1 * self.el2).relabel(group='Val1', label='Label1')
             + (self.el1 * self.el2).relabel(group='Val2', label='Label2'))
        self.assertEqual(t.keys(), [('Val1', 'Label1'), ('Val2', 'Label2')])


    def test_triple_composite_relabelled_value_and_label_keys(self):
        t = ((self.el1 * self.el2)
             +(self.el1 * self.el2).relabel(group='Val1', label='Label1')
             + (self.el1 * self.el2).relabel(group='Val2', label='Label2'))
        excepted_keys = [('Overlay', 'I'), ('Val1', 'Label1'), ('Val2', 'Label2')]
        self.assertEqual(t.keys(), excepted_keys)

    def test_deep_composite_values(self):
        o1 = (self.el1 * self.el2)
        o2 = (self.el1 * self.el2)
        o3 = (self.el7 * self.el8)
        t = (o1 + o2 + o3)
        self.assertEqual(t.values(), [o1, o2, o3])

    def test_deep_composite_keys(self):
        o1 = (self.el1 * self.el2)
        o2 = (self.el1 * self.el2)
        o3 = (self.el7 * self.el8)
        t = (o1 + o2 + o3)
        expected_keys = [('Overlay', 'I'), ('Overlay', 'II'), ('ValA', 'I')]
        self.assertEqual(t.keys(), expected_keys)

    def test_deep_composite_indexing(self):
        o1 = (self.el1 * self.el2)
        o2 = (self.el1 * self.el2)
        o3 = (self.el7 * self.el8)
        t = (o1 + o2 + o3)
        expected_keys = [('Overlay', 'I'), ('Overlay', 'II'), ('ValA', 'I')]
        self.assertEqual(t.keys(), expected_keys)
        self.assertEqual(t.ValA.I, o3)
        self.assertEqual(t.ValA.I.ValA.LabelA, self.el7)
        self.assertEqual(t.ValA.I.ValA.LabelB, self.el8)

    def test_deep_composite_getitem(self):
        o1 = (self.el1 * self.el2)
        o2 = (self.el1 * self.el2)
        o3 = (self.el7 * self.el8)
        t = (o1 + o2 + o3)
        expected_keys = [('Overlay', 'I'), ('Overlay', 'II'), ('ValA', 'I')]
        self.assertEqual(t.keys(), expected_keys)
        self.assertEqual(t['ValA']['I'], o3)
        self.assertEqual(t['ValA']['I'].get('ValA').get('LabelA'), self.el7)
        self.assertEqual(t['ValA']['I'].get('ValA').get('LabelB'), self.el8)

    def test_invalid_tree_structure(self):
        try:
            (self.el1 + self.el2) * (self.el1 + self.el2)
        except TypeError as e:
            self.assertEqual(str(e), "unsupported operand type(s) for *: 'Layout' and 'Layout'")
