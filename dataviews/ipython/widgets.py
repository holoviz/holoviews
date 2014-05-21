import sys, math, time

try:
    import IPython
    from IPython.core.display import clear_output
except:
    clear_output = None
    from nose.plugins.skip import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.12")


ip2 = IPython.version_info[0] == 2

import param

class ProgressBar(param.Parameterized):
    """
    A simple text progress bar suitable for the IPython notebook.
    """

    label = param.String(default='Progress', doc="""
        The label for the current progress bar.""")

    width = param.Integer(default=70, doc="""
        The width of the progress bar in multiples of 'char'.""")

    fill_char = param.String(default='#', doc="""
        The character used to fill the progress bar.""")

    def __init__(self, **kwargs):
        super(ProgressBar,self).__init__(**kwargs)

    def __call__(self, percentage):
        " Update the progress bar to the given percentage value "
        if clear_output and not ip2: clear_output()
        if clear_output and ip2: clear_output(wait=True)
        percent_per_char = 100.0 / self.width
        char_count = int(math.floor(percentage/percent_per_char) if percentage<100.0 else self.width)
        blank_count = self.width - char_count
        sys.stdout.write('\r' + "%s:\n[%s%s] %0.1f%%" % (self.label,
                                                        self.fill_char * char_count,
                                                        ' '*len(self.fill_char)*blank_count,
                                                        percentage))
        sys.stdout.flush()
        time.sleep(0.0001)
