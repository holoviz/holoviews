"""
Test cases for the composite types built with + and * i.e Layout
and Overlay (does *not* test HoloMaps).
"""

from holoviews import Element, Layout, Overlay
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

    def test_layouttree_from_values1(self):
        t = Layout.from_values(self.el1)
        self.assertEqual(t.keys(),  [('Element', 'I')])

    def test_layouttree_from_values2(self):
        t = Layout.from_values(self.el8)
        self.assertEqual(t.keys(),  [('ValA', 'LabelB')])

    def test_layouttree_group(self):
        t1 = (self.el1 + self.el2)
        t2 = t1.regroup('NewValue')
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

    def test_overlay_from_values1(self):
        t = Overlay.from_values(self.el1)
        self.assertEqual(t.keys(),  [('Element', 'I')])

    def test_overlay_from_values2(self):
        t = Overlay.from_values(self.el8)
        self.assertEqual(t.keys(),  [('ValA', 'LabelB')])

    def test_overlay_group(self):
        t1 = (self.el1 * self.el2)
        t2 = t1.regroup('NewValue')
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
