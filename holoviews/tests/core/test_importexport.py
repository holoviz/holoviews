"""
Unit test of the (non-rendering) exporters and importers.
"""

import numpy as np

from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal


class TestSerialization:
    """
    Test the basic serializer and deserializer (i.e. using pickle),
    including metadata access.
    """

    def setup_method(self):
        self.image1 = Image(np.array([[1,2],[4,5]]))
        self.image2 = Image(np.array([[5,4],[3,2]]))

    def test_serializer_save(self, tmp_path):
        Serializer.save(self.image1, tmp_path / 'test_serializer_save.pkl',
                        info={'info':'example'}, key={1:2})

    def test_serializer_save_no_file_extension(self, tmp_path):
        Serializer.save(self.image1, tmp_path / 'test_serializer_save_no_ext',
                        info={'info':'example'}, key={1:2})
        assert tmp_path / 'test_serializer_save_no_ext.pkl' in tmp_path.iterdir()

    def test_serializer_save_and_load_data_1(self, tmp_path):
        Serializer.save(self.image1, tmp_path / 'test_serializer_save_and_load_data_1.pkl')
        loaded = Deserializer.load(tmp_path / 'test_serializer_save_and_load_data_1.pkl')
        assert_element_equal(loaded, self.image1)

    def test_serializer_save_and_load_data_2(self, tmp_path):
        Serializer.save(self.image2, tmp_path / 'test_serializer_save_and_load_data_2.pkl')
        loaded = Deserializer.load(tmp_path / 'test_serializer_save_and_load_data_2.pkl')
        assert_element_equal(loaded, self.image2)

    def test_serializer_save_and_load_key(self, tmp_path):
        input_key = {'test_key':'key_val'}
        Serializer.save(self.image1, tmp_path / 'test_serializer_save_and_load_data.pkl', key=input_key)
        key = Deserializer.key(tmp_path / 'test_serializer_save_and_load_data.pkl')
        assert key == input_key

    def test_serializer_save_and_load_info(self, tmp_path):
        input_info = {'info':'example'}
        Serializer.save(self.image1, tmp_path / 'test_serializer_save_and_load_data.pkl', info=input_info)
        info = Deserializer.info(tmp_path / 'test_serializer_save_and_load_data.pkl')
        assert info['info'] == input_info['info']

    def test_serialize_deserialize_1(self, tmp_path):
        data,_ = Serializer(self.image1)
        obj =   Deserializer(data)
        assert_element_equal(obj, self.image1)

    def test_serialize_deserialize_2(self, tmp_path):
        data,_ = Serializer(self.image2)
        obj =  Deserializer(data)
        assert_element_equal(obj, self.image2)



class TestBasicPickler:
    """
    Test pickler and unpickler using the .hvz format, including
    metadata access.
    """

    def setup_method(self):
        self.image1 = Image(np.array([[1,2],[4,5]]))
        self.image2 = Image(np.array([[5,4],[3,2]]))

    def test_pickler_save(self, tmp_path) -> None:
        Pickler.save(self.image1, tmp_path / 'test_pickler_save.hvz',
                     info={'info':'example'}, key={1:2})

    def test_pickler_save_no_file_extension(self, tmp_path):
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_no_ext',
                        info={'info':'example'}, key={1:2})
        assert tmp_path / 'test_pickler_save_no_ext.hvz' in tmp_path.iterdir()

    def test_pickler_save_and_load_data_1(self, tmp_path):
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_and_load_data_1.hvz')
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_and_load_data_1.hvz')
        assert_element_equal(loaded, self.image1)

    def test_pickler_save_and_load_data_2(self, tmp_path):
        Pickler.save(self.image2, tmp_path / 'test_pickler_save_and_load_data_2.hvz')
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_and_load_data_2.hvz')
        assert_element_equal(loaded, self.image2)

    def test_pickler_save_and_load_key(self, tmp_path):
        input_key = {'test_key':'key_val'}
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_and_load_data.hvz', key=input_key)
        key = Unpickler.key(tmp_path / 'test_pickler_save_and_load_data.hvz')
        assert key == input_key

    def test_pickler_save_and_load_info(self, tmp_path):
        input_info = {'info':'example'}
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_and_load_data.hvz', info=input_info)
        info = Unpickler.info(tmp_path / 'test_pickler_save_and_load_data.hvz')
        assert info['info'] == input_info['info']

    def test_serialize_deserialize_1(self):
        data,_ = Pickler(self.image1)
        obj =   Unpickler(data)
        assert_element_equal(obj, self.image1)

    def test_serialize_deserialize_2(self):
        data,_ = Pickler(self.image2)
        obj =   Unpickler(data)
        assert_element_equal(obj, self.image2)


class TestPicklerAdvanced:
    """
    Test advanced pickler and unpickler functionality supported by the
    .hvz format.
    """

    def setup_method(self):
        self.image1 = Image(np.array([[1,2],[4,5]]))
        self.image2 = Image(np.array([[5,4],[3,2]]))

    def test_pickler_save_layout(self, tmp_path):
        Pickler.save(self.image1+self.image2, tmp_path / 'test_pickler_save_layout',
                        info={'info':'example'}, key={1:2})

    def test_pickler_save_load_layout(self, tmp_path):
        Pickler.save(self.image1+self.image2, tmp_path / 'test_pickler_save_load_layout',
                        info={'info':'example'}, key={1:2})
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout.hvz')
        assert_element_equal(loaded, self.image1+self.image2)

    def test_pickler_save_load_layout_entries(self, tmp_path):
        Pickler.save(self.image1+self.image2, tmp_path / 'test_pickler_save_load_layout_entries',
                        info={'info':'example'}, key={1:2})
        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entries.hvz')
        assert entries == ['Image.I', 'Image.II']

    def test_pickler_save_load_layout_entry1(self, tmp_path):
        Pickler.save(self.image1+self.image2, tmp_path / 'test_pickler_save_load_layout_entry1',
                        info={'info':'example'}, key={1:2})
        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entry1.hvz')
        assert 'Image.I' in entries, "Entry 'Image.I' missing"
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout_entry1.hvz',
                                entries=['Image.I'])
        assert_element_equal(loaded, self.image1)

    def test_pickler_save_load_layout_entry2(self, tmp_path):
        Pickler.save(self.image1+self.image2, tmp_path / 'test_pickler_save_load_layout_entry2',
                        info={'info':'example'}, key={1:2})
        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entry2.hvz')
        assert 'Image.II' in entries, "Entry 'Image.II' missing"
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout_entry2.hvz',
                                entries=['Image.II'])
        assert_element_equal(loaded, self.image2)

    def test_pickler_save_load_single_layout(self, tmp_path):
        single_layout = Layout([self.image1])
        Pickler.save(single_layout, tmp_path / 'test_pickler_save_load_single_layout',
                        info={'info':'example'}, key={1:2})

        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_single_layout.hvz')
        assert entries == ['Image.I(L)']
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_single_layout.hvz',
                                entries=['Image.I(L)'])
        assert_element_equal(single_layout, loaded)
