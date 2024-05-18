"""
Unit test of the archive system, namely FileArchive with different
exporters (not including renderers).
"""
import json
import os
import tarfile
import zipfile

import numpy as np

from holoviews import Image
from holoviews.core.io import FileArchive, Serializer


class TestFileArchive:

    def setup_method(self):
        self.image1 = Image(np.array([[1,2],[4,5]]), group='Group1', label='Im1')
        self.image2 = Image(np.array([[5,4],[3,2]]), group='Group2', label='Im2')

    def test_filearchive_init(self):
        FileArchive()

    def test_filearchive_image_pickle(self, tmp_path):
        export_name = 'archive_image'
        filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
        archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=False)
        archive.add(self.image1)
        archive.add(self.image2)
        assert len(archive) == 2
        assert archive.listing() == filenames
        archive.export()
        assert os.path.isdir(tmp_path / export_name), f"No directory {str(export_name)!r} created on export."
        assert sorted(filenames) == sorted(os.listdir(tmp_path / export_name))
        assert archive.listing() == []

    def test_filearchive_image_pickle_zip(self, tmp_path):
        export_name = 'archive_image'
        filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
        archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name,
                              pack=True, archive_format='zip')
        archive.add(self.image1)
        archive.add(self.image2)
        assert len(archive) == 2
        assert archive.listing() == filenames
        archive.export()
        export_folder = os.fspath(tmp_path / export_name) + '.zip'
        assert os.path.isfile(export_folder)
        namelist = [f'{export_name}/{f}' for f in filenames]
        with zipfile.ZipFile(export_folder, 'r') as f:
            expected = sorted(map(os.path.abspath, namelist))
            output = sorted(map(os.path.abspath, f.namelist()))
            assert expected == output
        assert archive.listing() == []


    def test_filearchive_image_pickle_tar(self, tmp_path):
        export_name = 'archive_image'
        filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
        archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name,
                              pack=True, archive_format='tar')
        archive.add(self.image1)
        archive.add(self.image2)
        assert len(archive) == 2
        assert archive.listing() == filenames
        archive.export()
        export_folder = os.fspath(tmp_path / export_name) + '.tar'
        assert os.path.isfile(export_folder)
        namelist = [f'{export_name}/{f}' for f in filenames]
        with tarfile.TarFile(export_folder, 'r') as f:
            assert sorted(namelist) == sorted([el.path for el in f.getmembers()])
        assert archive.listing() == []


    def test_filearchive_image_serialize(self, tmp_path):
        export_name = 'archive_image_serizalize'
        filenames = ['Group1-Im1.pkl', 'Group2-Im2.pkl']
        archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, exporters=[Serializer], pack=False)
        archive.add(self.image1)
        archive.add(self.image2)
        assert len(archive) == 2
        assert archive.listing() == filenames
        archive.export()
        assert os.path.isdir(tmp_path / export_name)
        assert sorted(filenames) == sorted(os.listdir(tmp_path / export_name))
        assert archive.listing() == []

    def test_filearchive_image_pickle_name_clash(self, tmp_path):
        export_name = 'archive_image_test_clash'
        filenames = ['Group1-Im1.hvz', 'Group1-Im1-1.hvz']
        archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=False)
        archive.add(self.image1)
        archive.add(self.image1)
        assert len(archive) == 2
        assert archive.listing() == filenames
        archive.export()
        assert os.path.isdir(tmp_path / export_name)
        assert sorted(filenames) == sorted(os.listdir(tmp_path / export_name))
        assert archive.listing() == []

    def test_filearchive_json_single_file(self, tmp_path):
        export_name = "archive_json"
        data = {'meta':'test'}
        archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=False)
        archive.add(filename='metadata.json', data=json.dumps(data),
                    info={'mime_type':'text/json'})
        assert len(archive) == 1
        assert archive.listing() == ['metadata.json']
        archive.export()
        fname = os.fspath(tmp_path / f"{export_name}_metadata.json")
        assert os.path.isfile(fname)
        with open(fname) as f:
            assert json.load(f) == data
        assert archive.listing() == []

    """
    A test case that examines the clear() method of
    FileArchives of io.py
    """
    def test_filearchive_clear_file(self, tmp_path):
        export_name = "archive_for_clear"
        export_name = "archive_for_clear"
        archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=False)
        archive.add(self.image1)
        archive.add(self.image2)
        archive.clear()
        assert archive._files == {}
