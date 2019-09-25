"""
Unit test of the (non-rendering) exporters and importers.
"""

import os
import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Serializer, Pickler, Unpickler, Deserializer
from holoviews.element.comparison import ComparisonTestCase


class TestSerialization(ComparisonTestCase):
    """
    Test the basic serializer and deserializer (i.e. using pickle),
    including metadata access.
    """

    def setUp(self):
        self.image1 = Image(np.array([[1,2],[4,5]]))
        self.image2 = Image(np.array([[5,4],[3,2]]))

    def tearDown(self):
        for f in os.listdir('.'):
            if f.endswith('.pkl'):
                os.remove(f)

    def test_serializer_save(self):
        Serializer.save(self.image1, 'test_serializer_save.pkl',
                        info={'info':'example'}, key={1:2})

    def test_serializer_save_no_file_extension(self):
        Serializer.save(self.image1, 'test_serializer_save_no_ext',
                        info={'info':'example'}, key={1:2})
        if 'test_serializer_save_no_ext.pkl' not in os.listdir('.'):
            raise AssertionError('File test_serializer_save_no_ext.pkl not found')

    def test_serializer_save_and_load_data_1(self):
        Serializer.save(self.image1, 'test_serializer_save_and_load_data_1.pkl')
        loaded = Deserializer.load('test_serializer_save_and_load_data_1.pkl')
        self.assertEqual(loaded, self.image1)

    def test_serializer_save_and_load_data_2(self):
        Serializer.save(self.image2, 'test_serializer_save_and_load_data_2.pkl')
        loaded = Deserializer.load('test_serializer_save_and_load_data_2.pkl')
        self.assertEqual(loaded, self.image2)

    def test_serializer_save_and_load_key(self):
        input_key = {'test_key':'key_val'}
        Serializer.save(self.image1, 'test_serializer_save_and_load_data.pkl', key=input_key)
        key = Deserializer.key('test_serializer_save_and_load_data.pkl')
        self.assertEqual(key, input_key)

    def test_serializer_save_and_load_info(self):
        input_info = {'info':'example'}
        Serializer.save(self.image1, 'test_serializer_save_and_load_data.pkl', info=input_info)
        info = Deserializer.info('test_serializer_save_and_load_data.pkl')
        self.assertEqual(info['info'], input_info['info'])

    def test_serialize_deserialize_1(self):
        data,_ = Serializer(self.image1)
        obj =   Deserializer(data)
        self.assertEqual(obj, self.image1)

    def test_serialize_deserialize_2(self):
        data,_ = Serializer(self.image2)
        obj =   Deserializer(data)
        self.assertEqual(obj, self.image2)



class TestBasicPickler(ComparisonTestCase):
    """
    Test pickler and unpickler using the .hvz format, including
    metadata access.
    """

    def setUp(self):
        self.image1 = Image(np.array([[1,2],[4,5]]))
        self.image2 = Image(np.array([[5,4],[3,2]]))

    def tearDown(self):
        for f in os.listdir('.'):
            if f.endswith('.hvz'):
                os.remove(f)

    def test_pickler_save(self):
        Pickler.save(self.image1, 'test_pickler_save.hvz',
                     info={'info':'example'}, key={1:2})

    def test_pickler_save_no_file_extension(self):
        Pickler.save(self.image1, 'test_pickler_save_no_ext',
                        info={'info':'example'}, key={1:2})
        if 'test_pickler_save_no_ext.hvz' not in os.listdir('.'):
            raise AssertionError('File test_pickler_save_no_ext.hvz not found')

    def test_pickler_save_and_load_data_1(self):
        Pickler.save(self.image1, 'test_pickler_save_and_load_data_1.hvz')
        loaded = Unpickler.load('test_pickler_save_and_load_data_1.hvz')
        self.assertEqual(loaded, self.image1)

    def test_pickler_save_and_load_data_2(self):
        Pickler.save(self.image2, 'test_pickler_save_and_load_data_2.hvz')
        loaded = Unpickler.load('test_pickler_save_and_load_data_2.hvz')
        self.assertEqual(loaded, self.image2)

    def test_pickler_save_and_load_key(self):
        input_key = {'test_key':'key_val'}
        Pickler.save(self.image1, 'test_pickler_save_and_load_data.hvz', key=input_key)
        key = Unpickler.key('test_pickler_save_and_load_data.hvz')
        self.assertEqual(key, input_key)

    def test_pickler_save_and_load_info(self):
        input_info = {'info':'example'}
        Pickler.save(self.image1, 'test_pickler_save_and_load_data.hvz', info=input_info)
        info = Unpickler.info('test_pickler_save_and_load_data.hvz')
        self.assertEqual(info['info'], input_info['info'])

    def test_serialize_deserialize_1(self):
        data,_ = Pickler(self.image1)
        obj =   Unpickler(data)
        self.assertEqual(obj, self.image1)

    def test_serialize_deserialize_2(self):
        data,_ = Pickler(self.image2)
        obj =   Unpickler(data)
        self.assertEqual(obj, self.image2)



class TestPicklerAdvanced(ComparisonTestCase):
    """
    Test advanced pickler and unpickler functionality supported by the
    .hvz format.
    """

    def setUp(self):
        self.image1 = Image(np.array([[1,2],[4,5]]))
        self.image2 = Image(np.array([[5,4],[3,2]]))

    def tearDown(self):
        for f in os.listdir('.'):
            if f.endswith('.hvz'):
                os.remove(f)

    def test_pickler_save_layout(self):
        Pickler.save(self.image1+self.image2, 'test_pickler_save_layout',
                        info={'info':'example'}, key={1:2})

    def test_pickler_save_load_layout(self):
        Pickler.save(self.image1+self.image2, 'test_pickler_save_load_layout',
                        info={'info':'example'}, key={1:2})
        loaded = Unpickler.load('test_pickler_save_load_layout.hvz')
        self.assertEqual(loaded, self.image1+self.image2)

    def test_pickler_save_load_layout_entries(self):
        Pickler.save(self.image1+self.image2, 'test_pickler_save_load_layout_entries',
                        info={'info':'example'}, key={1:2})
        entries = Unpickler.entries('test_pickler_save_load_layout_entries.hvz')
        self.assertEqual(entries, ['Image.I', 'Image.II'] )

    def test_pickler_save_load_layout_entry1(self):
        Pickler.save(self.image1+self.image2, 'test_pickler_save_load_layout_entry1',
                        info={'info':'example'}, key={1:2})
        entries = Unpickler.entries('test_pickler_save_load_layout_entry1.hvz')
        assert ('Image.I' in entries), "Entry 'Image.I' missing"
        loaded = Unpickler.load('test_pickler_save_load_layout_entry1.hvz',
                                entries=['Image.I'])
        self.assertEqual(loaded, self.image1)

    def test_pickler_save_load_layout_entry2(self):
        Pickler.save(self.image1+self.image2, 'test_pickler_save_load_layout_entry2',
                        info={'info':'example'}, key={1:2})
        entries = Unpickler.entries('test_pickler_save_load_layout_entry2.hvz')
        assert ('Image.II' in entries), "Entry 'Image.II' missing"
        loaded = Unpickler.load('test_pickler_save_load_layout_entry2.hvz',
                                entries=['Image.II'])
        self.assertEqual(loaded, self.image2)

    def test_pickler_save_load_single_layout(self):
        single_layout = Layout([self.image1])
        Pickler.save(single_layout, 'test_pickler_save_load_single_layout',
                        info={'info':'example'}, key={1:2})

        entries = Unpickler.entries('test_pickler_save_load_single_layout.hvz')
        self.assertEqual(entries, ['Image.I(L)'])
        loaded = Unpickler.load('test_pickler_save_load_single_layout.hvz',
                                entries=['Image.I(L)'])
        self.assertEqual(single_layout, loaded)

