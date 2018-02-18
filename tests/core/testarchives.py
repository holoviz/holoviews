"""
Unit test of the archive system, namely FileArchive with different
exporters (not including renderers).
"""
import os
import shutil
import json
import zipfile
import tarfile
import numpy as np
from holoviews import Image
from holoviews.core.io import Serializer, FileArchive
from holoviews.element.comparison import ComparisonTestCase


class TestFileArchive(ComparisonTestCase):

    def setUp(self):
        self.image1 = Image(np.array([[1,2],[4,5]]), group='Group1', label='Im1')
        self.image2 = Image(np.array([[5,4],[3,2]]), group='Group2', label='Im2')

    def tearDown(self):
        for f in os.listdir('.'):
            if any(f.endswith(ext) for ext in ['.json', '.zip', '.tar']):
                os.remove(f)
            if os.path.isdir(f) and f.startswith('archive_'):
                shutil.rmtree(f)

    def test_filearchive_init(self):
        FileArchive()

    def test_filearchive_image_pickle(self):
        export_name = 'archive_image'
        filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
        archive = FileArchive(export_name=export_name, pack=False)
        archive.add(self.image1)
        archive.add(self.image2)
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.listing(), filenames)
        archive.export()
        if not os.path.isdir(export_name):
            raise AssertionError("No directory %r created on export." % export_name)
        self.assertEqual(sorted(filenames), sorted(os.listdir(export_name)))
        self.assertEqual(archive.listing(), [])

    def test_filearchive_image_pickle_zip(self):
        export_name = 'archive_image'
        filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
        archive = FileArchive(export_name=export_name,
                              pack=True, archive_format='zip')
        archive.add(self.image1)
        archive.add(self.image2)
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.listing(), filenames)
        archive.export()
        if not os.path.isfile(export_name+'.zip'):
            raise AssertionError("No zip file %r created on export." % export_name)

        namelist = ['archive_image/%s' % f for f in filenames]
        with zipfile.ZipFile(export_name+'.zip', 'r') as f:
            self.assertEqual(sorted(namelist), sorted(f.namelist()))
        self.assertEqual(archive.listing(), [])


    def test_filearchive_image_pickle_tar(self):
        export_name = 'archive_image'
        filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
        archive = FileArchive(export_name=export_name,
                              pack=True, archive_format='tar')
        archive.add(self.image1)
        archive.add(self.image2)
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.listing(), filenames)
        archive.export()
        if not os.path.isfile(export_name+'.tar'):
            raise AssertionError("No tar file %r created on export." % export_name)

        namelist = ['archive_image/%s' % f for f in filenames]
        with tarfile.TarFile(export_name+'.tar', 'r') as f:
            self.assertEqual(sorted(namelist),
                             sorted([el.path for el in f.getmembers()]))
        self.assertEqual(archive.listing(), [])


    def test_filearchive_image_serialize(self):
        export_name = 'archive_image_serizalize'
        filenames = ['Group1-Im1.pkl', 'Group2-Im2.pkl']
        archive = FileArchive(export_name=export_name, exporters=[Serializer], pack=False)
        archive.add(self.image1)
        archive.add(self.image2)
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.listing(), filenames)
        archive.export()
        if not os.path.isdir(export_name):
            raise AssertionError("No directory %r created on export." % export_name)
        self.assertEqual(sorted(filenames), sorted(os.listdir(export_name)))
        self.assertEqual(archive.listing(), [])

    def test_filearchive_image_pickle_name_clash(self):
        export_name = 'archive_image_test_clash'
        filenames = ['Group1-Im1.hvz', 'Group1-Im1-1.hvz']
        archive = FileArchive(export_name=export_name, pack=False)
        archive.add(self.image1)
        archive.add(self.image1)
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.listing(), filenames)
        archive.export()
        if not os.path.isdir(export_name):
            raise AssertionError("No directory %r created on export." % export_name)
        self.assertEqual(sorted(filenames), sorted(os.listdir(export_name)))
        self.assertEqual(archive.listing(), [])

    def test_filearchive_json_single_file(self):
        export_name = 'archive_json'
        data = {'meta':'test'}
        archive = FileArchive(export_name=export_name, pack=False)
        archive.add(filename='metadata.json', data=json.dumps(data),
                    info={'mime_type':'text/json'})
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.listing(), ['metadata.json'])
        archive.export()
        fname = '%s_%s' % (export_name, 'metadata.json')
        if not os.path.isfile(fname):
            raise AssertionError("No file %r created on export." % fname)
        self.assertEqual(json.load(open(fname, 'r')), data)
        self.assertEqual(archive.listing(), [])
