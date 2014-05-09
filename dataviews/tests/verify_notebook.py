#! /usr/bin/env ipython
"""
As of IPython 2.0 there seems to be no good way to automatically run
notebooks, nevermind test them. This script attempts to use a set of
ugly hacks (as cleanly as possible) to help remedy this situation.

As IPython Notebooks get used more (and as more code integrates with
them) the more important it becomes that we can:

1) Automatically run notebooks (including display hooks with HTML/rich content)
2) Automatically test notebook content for unexpected changes.
3) Do this easily and quickly.

It is impossible to maintain reproducible notebooks without this
functionality. Thankfully, this may become easier using nbconvert in
IPython 3.0: https://github.com/ipython/ipython/pull/5639

Summary
-------

* This script must be run with IPython (no notebook server required).

* HTML and plaintext (i.e. pretty-printed) display hook are patched
  to grab object and display data.

* Animations are configured to render as single frames (more reliable
  output format than video).

* Standard output is captured to collect printed content.

* Booleans follow bizarre rules: instead of using display hooks, they
  must be grabbed using the _ (underscore variable) in a post
  execution hook.

* Reference data is saved (and potentially archived) if not available.

* If reference data is found, test data is generated from the given
  notebook by running it in the same manner.

* Each item of test data is paired up with a correspond reference
  file. Each of these pairs becomes a test method that is dynamically
  patched onto a unittest.TestCase class (for nose to find).

* Nosetests runs each of the dynamically defined tests. Each item of
  test data is compared to the corresponding reference data.

Limitations
-----------

* Only one notebook can be tested at once (the IPython namespace is
  shared and impossible to clear properly).

* There needs to be cleaner way to achieve all this!
"""

import sys, os, pickle,  shutil, time, zipfile
from StringIO import StringIO

# Standardize backend due to random inconsistencies
from matplotlib import pyplot
pyplot.switch_backend('agg')

import IPython
from IPython import get_ipython
from IPython.display import clear_output
from IPython.nbformat import current

from utils import IPTestCase
from dataviews import ipython

from nose.plugins.skip import SkipTest


ARCHIVE_REF = False  # Whether or not to zip/unzip the reference data
CLEANUP_DATA = False # Whether to delete the generated test data when complete


class NBTester(IPTestCase):
    """
    Tester class used by nosetests. Test methods dynamically generated
    from notebook data.
    """
    @classmethod
    def tearDownClass(cls):
        if ARCHIVE_REF:
            shutil.rmtree(NB_PATH[:-6]+'_reference')
        if CLEANUP_DATA:
            shutil.rmtree(NB_PATH[:-6])


class Capture(object):
    """
    Class that tries many dirty tricks to try and capture Python
    objects and display their data from the notebook. Capture occurs
    when cells are executed by NBRunner.
    """
    def __init__(self, ip, name, reference, code_cell_count):
        self.shell = ip
        self.name = name
        self.reference = reference
        self.code_cell_count = code_cell_count
        # Captured object and display data
        self.object_data = None
        self.display_data = None
        # Counters set from outside by NBRunner
        self.counter = {'display':0, 'data':0, 'code':0}
        self.set_display_hooks() # Hooks patched to grab data
        self.wait = True

    def post_execute(self):
        """
        Hook executed after each cell is run. Used to capture the _
        (last variable). Made necessary because booleans slip past
        stderr/stdout and all display hooks (including
        sys.displayhook!)
        """
        self.object_data = None
        self.display_data = None
        obj = self.shell.user_ns.get('_')
        # Necessary in case the extension is reloaded
        ipython.render_anim = ipython.middle_frame
        self.shell.display_formatter.format(obj)[0]['text/plain']
        # Only set if bool and no captured via displayhook
        if self.object_data is None and isinstance(obj, bool):
            self.object_data = obj
        # Needed to indicate the execution of the notebook is over.
        if isinstance(obj,str) and obj =='__EXECUTION_TERMINATED__':
            self.wait = False

    #=======================#
    # Patched Display hooks #
    #=======================#

    def empty_hook(self, hook_type):
        """
        Capture arbitrary Python objects without testing display
        output.
        """
        def capture_hook(obj, pprinter, cycles):
            self.object_data = obj if obj != '' else None
            info = (self.counter['code'], self.code_cell_count,
                    ' reference ' if self.reference else ' ', self.name)
            pprinter.text("[Code cell %d/%d] Captured%sdata from '%s' notebook" % info)
            self.display_data = None
            clear_output()
        return capture_hook


    def html_hook(self, display_hook):
        """
        Capture Python objects that have HTML display and the
        corresponding markup.
        """
        def capture_hook(obj, pprinter, cycles):
            self.object_data = obj
            self.display_data = display_hook(obj)
            info = (self.counter['code'], self.code_cell_count,
                    ' reference ' if self.reference else ' ', self.name)
            pprinter.text("[%d/%d] Captured%sdisplay from '%s' notebook" % info)
            clear_output()
        return capture_hook


    def set_display_hooks(self):
        """
        Patch the IPython display hooks to capture raw object data and display.
        """
        # Patch normal pretty-printing to grab those objects.
        plain_formatter = self.shell.display_formatter.formatters['text/plain']
        plain_printers = dict((tp, self.empty_hook(tp))
                              for (tp,h) in plain_formatter.type_printers.items())
        for tp, hook in plain_printers.items():
            plain_formatter.for_type(tp, hook)
        # Transfer custom HTML hooks over to plain/text hooks for views
        html_formatter = self.shell.display_formatter.formatters['text/html']
        html_printers = dict((tp, self.html_hook(h))
                             for (tp,h) in html_formatter.type_printers.items())

        # Set the combined set of patched hooks on text/plain (html notebook only)
        for tp, hook in html_printers.items():
            plain_formatter.for_type(tp, hook)

        # Attempt to capture anything that is of type object...
        plain_formatter.for_type(object, self.empty_hook(object))



class NBRunner(object):
    """
    Run the code cells of a notebook and save their return values
    (possibly including display representation) to disk.
    """
    def __init__(self, shell, name, nb, data_dir, reference=False):
        self.nb = nb
        self.shell = shell
        self.data_dir = data_dir
        self.reference = reference

        self.code_cells = self.get_code_cells(nb)
        self.capture = Capture(shell, name, reference, len(self.code_cells))
        if IPython.version_info[0] < 2:
            self.shell.register_post_execute(self.capture.post_execute)
        else:
            self.shell.events.register('post_run_cell', self.capture.post_execute)



    def get_code_cells(self, nb):
        """
        Read the contents of all code cells from given notebook.
        """
        code_cells = []
        for i, cell in enumerate(nb.worksheets[0].cells):
            if cell.cell_type == 'code':
                code_cells.append(cell.input)
        return code_cells


    def run_cell(self, cell, buff, seekpos):
        """
        Run a code cell and capture the output to stdout. Requires a
        StringIO buffer and the current seekpos (which is updated in
        the returned tuple).
        """
        stdout_handle =  sys.stdout
        sys.stdout = buff
        self.shell.run_cell(cell)
        buff.flush()
        buff.seek(seekpos)
        print_output = buff.read()[:]
        seekpos += len(print_output)
        sys.stdout = stdout_handle
        return print_output, seekpos


    def run(self):
        """
        Run contents of code cells, capturing and saving object (with
        their corresponding display/print output when appropriate)
        """
        buff = StringIO()
        seekpos = 0
        filelist = []
        for i, cell in enumerate(self.code_cells):
            self.object_data = None
            self.display_data = None

            print_output, seekpos = self.run_cell(cell, buff, seekpos)
            self.capture.counter['code'] += 1 # Cell has been run

            object_data = self.capture.object_data
            display_data = self.capture.display_data

            # Save object data (and code executed)
            if object_data is not None:
                pickle_path = os.path.join(self.data_dir, 'data_%03d.pkl' %
                                           self.capture.counter['data'])
                filelist.append(pickle_path)
                with open(pickle_path,'w') as f:
                    pickle.dump((object_data, cell), f)
                self.capture.counter['data'] += 1

            # Save object display data (and code executed)
            if display_data is not None or print_output!='':
                html_path = os.path.join(self.data_dir, 'display_%03d.html'
                                         % self.capture.counter['display'])
                filelist.append(html_path)
                title = ('<b>[Display %d]</b></br></br>' % self.capture.counter['display'])
                display_str = display_data if display_data else ''
                with open(html_path, 'w') as f:
                    f.write(title+print_output.replace('\n', '<br>')+"<br><br>"+display_str)
                self.capture.counter['display'] += 1

        # Signal that the last cell has executed (wait till it has)
        self.shell.run_cell('"__EXECUTION_TERMINATED__"')
        while self.capture.wait:
             time.sleep(1)

        # Archive generated reference data as appropriate.
        if self.reference and ARCHIVE_REF:
            zipf = zipfile.ZipFile("%s.zip" % self.data_dir, 'w')
            for f in filelist:
                zipf.write(f, os.path.basename(f))

            shutil.rmtree(self.data_dir)



class Configure(object):
    """
    Set up Capture, execute the notebooks with NBRunner to generate
    files (reference and/or test data) and finally build unit tests
    (that nose can find) on NBTester.
    """
    def __init__(self, notebook):
        ip = get_ipython()   # Get IPython instance (if possible)
        if ip is None: raise SkipTest("No IPython")
        # Booleans cannot be silenced (or captured normally)!
        prompt = "[Unsilenceable Boolean (ignore)]"
        ipython.load_ipython_extension(ip, verbose=False)
        ip.run_cell("%config PromptManager.out_template = '"+ prompt+"'", silent=True)

        msg = self.generate_data_files(ip, notebook)  # Create test pickle/ html files
        msg += self.set_nose_methods(notebook)        # Compare files as unit tests

        # Display message
        if msg.strip():
            sys.stderr.write("\n%s\n" % msg)


    def compare_files(self, data_path, ref_path):
        """
        Create a test method given two file paths to be compared.
        """
        def data_comparison(*args):
            # Needs *args and following due to odd behaviour (due to nose?)
            if not args: return
            self = args[0]
            # No reference data. Nothing to test.
            if not os.path.isfile(ref_path): return
            # Load reference and test data
            test_code, ref_code = None, None
            if data_path.endswith('.pkl'):
                with open(data_path,'r') as data_file:
                    test_data, test_code = pickle.load(data_file)
                with open(ref_path,'r') as ref_file:
                    ref_data,  ref_code =  pickle.load(ref_file)
                kwargs = {}
            elif data_path.endswith('.html'):
                test_data = open(data_path,'r').read()
                ref_data =  open(ref_path,'r').read()
                kwargs = {'msg':'Display output mismatch.'}
            try:
                # Compare the contents of the two files
                self.assertEqual(test_data, ref_data, **kwargs)
            # Show the code at the point of inconsistent results
            except AssertionError as e:
                if test_code:
                    print "Code cell executed:\n   %s" %           '\n   '.join(test_code.splitlines())
                    print "Reference code cell executed:\n   %s" % '\n   '.join(ref_code.splitlines())
                raise e

        return data_comparison


    def set_nose_methods(self, notebook):
        """
        Dynamically create all test methods that nosetests can find
        and execute.
        """
        report_msg = ''
        nb_name = os.path.basename(notebook).rsplit('.')[0]
        data_paths, ref_paths, msg = self.match_files(notebook)
        report_msg += msg
        # Zipped data/reference file paths for testing
        for (data_path, ref_path) in zip(data_paths, ref_paths):
            basename = os.path.basename(data_path).rsplit('.')[0]
            test_method = self.compare_files(data_path[:], ref_path[:])
            test_method.__name__ = 'test_'+ nb_name+'_'+ basename
            setattr (NBTester, test_method.__name__, test_method)
        return report_msg


    def generate_data_files(self, ip, notebook):
        """
        Generate both testand new reference data (if needed) using
        NBRunner.
        """
        msg =''
        # Find the data directory (create it if necessary)
        nb_dir, name = os.path.split(notebook)
        basename = name.rsplit('.ipynb')[0]
        data_dir = os.path.join(nb_dir, basename)
        ref_dir= data_dir + '_reference'
        nb = current.read(open(notebook,'r'), 'ipynb')
        # Reference data not found - regenerate it and exit
        archive = not os.path.isfile(ref_dir + '.zip')
        missing_dir =  not os.path.isdir(ref_dir)
        if (not archive and ARCHIVE_REF) or (missing_dir and not ARCHIVE_REF):
            if not os.path.isdir(ref_dir): os.mkdir(ref_dir)
            reference_runner =  NBRunner(ip, basename, nb, ref_dir, reference=True)
            reference_runner.run()
            return ''

        # Remove any pre-existing test data.
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        # Generate the test data
        os.mkdir(data_dir)
        NBRunner(ip, basename, nb, data_dir, reference=False).run()
        return msg


    def unzip_data(self, data_dir):
        """
        Decompress reference data zip archive.
        """
        archive = data_dir + '_reference.zip'
        unpack_dir = data_dir + '_reference'
        if os.path.isfile(archive):
            if os.path.isdir(unpack_dir):
                shutil.rmtree(unpack_dir)
            os.makedirs(unpack_dir)
            zfile = zipfile.ZipFile(archive)
            for name in zfile.namelist():
                (_, filename) = os.path.split(name)
                zfile.extract(name, unpack_dir)


    def match_files(self, notebook):
        """
        Given a set of test files in the data directory, look up the
        corresponding files in reference directory and generate
        warnings about any mismatches.
        """
        nb_dir, name = os.path.split(notebook)
        basename = name.rsplit('.ipynb')[0]
        data_dir = os.path.join(nb_dir, basename)
        # Find the test data files (if they exist at all)
        if not os.path.isdir(data_dir):
            data_files = []
        else:
            data_files = [f for f in sorted(os.listdir(data_dir))
                          if f.endswith('.pkl') or f.endswith('.html')]
        # Find the reference files
        if ARCHIVE_REF: self.unzip_data(data_dir)
        ref_files = [f for f in sorted(os.listdir(data_dir + '_reference'))
                     if f.endswith('.pkl') or f.endswith('.html')]
        # For each data file, look up the corresponding reference file
        data_paths, ref_paths, msg = [], [], ''
        for data_file in data_files:
            data_path = os.path.join(data_dir, data_file)
            ref_path = os.path.join(data_dir + '_reference', data_file)

            if not os.path.isfile(ref_path):
                msg += "No reference file found for %s\n" % data_file
            else:
                data_paths.append(data_path)
                ref_paths.append(ref_path)
        # Message about references files that were not looked up
        unused = set(ref_files) - set(data_files)
        if unused:
            msg += 'Following reference files are unused: %s' % ', '.join(sorted(unused))

        return data_paths, ref_paths, msg

#=========================================#
# Set up tests for the specified notebook #
#=========================================#

if len (sys.argv) == 2:
    NB_PATH = os.path.abspath(sys.argv[1])
    Configure(NB_PATH)
    sys.stderr.write('\nDynamically generated tests: %s\n'
                     % ', '.join(sorted(el for el in dir(NBTester) if el.startswith('test_'))))
else:
    raise Exception("Run as follows: ipython notebook_test.py <notebook_path>")


if __name__ == '__main__':
    """
    Run to test a single notebook as follows:

    $ ipython notebook_test.py Demo.ipynb
    """
    import nose
    sys.argv = sys.argv[:1] # Stop nose consuming notebook path
    nose.runmodule()
