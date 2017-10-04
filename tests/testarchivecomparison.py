"""
Unit test of ArchiveComparison.
"""

import os
import shutil
from collections import namedtuple

from unittest import TestCase

import numpy as np

from holoviews import Image
from holoviews.core.io import FileArchive
from holoviews.element.comparison import ArchiveComparison

# TODO:
# * test the actual comparison information (rather than just match or not)
# * test single file archive
# * test different matching functions


fspec = namedtuple('fspec',('name','contents'))


class TestArchiveComparison(TestCase):

    archive_format = None

    @classmethod
    def setUpClass(cls):
        cls.root = '_testarchivecomp'

        for f in os.listdir('.'):
            if f.startswith(cls.root):
                # CEBALERT: can't remember how to indicate test error (not error with stuff being tested)
                raise AssertionError('%s already exists: move away'%f)

        cls.image1 = Image(np.array([[1,2],[4,5]]), group='Group1', label='Im1')
        cls.image2 = Image(np.array([[5,4],[3,2]]), group='Group2', label='Im1')
        cls.image3 = Image(np.array([[6,7],[9,8]]), group='Group3', label='Im1')
        cls.export_name = 'test'

        cls.fa_opts = {}
        if cls.archive_format is not None:
            cls.fa_opts['pack']=True
            cls.fa_opts['archive_format']=cls.archive_format
        else:
            cls.fa_opts['pack']=False
        cls.ext = '.%s'%cls.archive_format if cls.archive_format else ''


    def setUp(self):
        os.mkdir(self.root)

 
    def tearDown(self):
        for f in os.listdir('.'):
            if os.path.isdir(f) and f.startswith(self.root):
                shutil.rmtree(f)

       
    def _compare_archives(self,fspec1,fspec2,match=True):
        self._make_archive(fspec1)
        self._make_archive(fspec2)

        a = ArchiveComparison(os.path.join(self.root,fspec1.name)+self.ext,
                              os.path.join(self.root,fspec2.name)+self.ext)

        if match:
            if not a.equal():
                raise AssertionError(a.fails)
        else:
            self.assertFalse(a.equal())


    def _make_archive(self,fspec):
        a = FileArchive(root=self.root,export_name=fspec.name,**self.fa_opts)
        for x in fspec.contents:
            a.add(x)
        a.export()


    def test_missing_test_archive(self):
        self._make_archive(fspec(self.export_name,(self.image1,self.image1)))

        with self.assertRaises(IOError):
            ArchiveComparison(os.path.join(self.root,self.export_name)+self.ext,
                              os.path.join(self.root,'doesnotexist'+self.ext))
            

    def test_missing_ref_archive(self):
        self._make_archive(fspec(self.export_name,(self.image1,self.image1)))

        with self.assertRaises(IOError):
            ArchiveComparison(os.path.join(self.root,'doesnotexist')+self.ext,
                              os.path.join(self.root,self.export_name+self.ext))


    def test_matching_contents(self):
        self._compare_archives(fspec(self.export_name+'1',(self.image1,self.image2)),
                               fspec(self.export_name+'2',(self.image1,self.image2)),
                               match=True)

    def test_nonmatching_contents(self):
        self._compare_archives(fspec(self.export_name+'1',(self.image1,self.image2)),
                               fspec(self.export_name+'2',(self.image1,self.image1)),
                               match=False)
        
    def test_extra_data(self):
        self._compare_archives(fspec(self.export_name+'1',(self.image1,self.image2)),
                               fspec(self.export_name+'2',(self.image1,self.image2,self.image3)),
                               match=False)

    def test_missing_data(self):
        self._compare_archives(fspec(self.export_name+'1',(self.image1,self.image2,self.image3)),
                               fspec(self.export_name+'2',(self.image1,self.image2)),
                               match=False)



class TestZipArchiveComparison(TestArchiveComparison):
    archive_format = 'zip'



class TestTarArchiveComparison(TestArchiveComparison):
    archive_format = 'tar'
