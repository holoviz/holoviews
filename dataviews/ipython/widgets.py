import sys, math, time

try:
    import IPython
    from IPython.core.display import clear_output
except:
    clear_output = None
    from nose.plugins.skip import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.12")

import param
ipython2 = (IPython.version_info[0] == 2)



class ProgressBar(param.Parameterized):
    """
    A simple text progress bar suitable for both the IPython notebook
    and the IPython interactive prompt.
    """

    label = param.String(default='Progress', allow_None=True, doc="""
        The label of the current progress bar.""")

    width = param.Integer(default=70, doc="""
        The width of the progress bar as the number of chararacters""")

    fill_char = param.String(default='#', doc="""
        The character used to fill the progress bar.""")

    blank_char = param.String(default=' ', doc="""
        The character for the blank portion of the progress bar.""")

    percent_range = param.NumericTuple(default=(0.0,100.0), doc="""
        The total percentage spanned by the progress bar when called
        with a value between 0% and 100%. This allows an overall
        completion in percent to be broken down into smaller sub-tasks
        that individually complete to 100 percent.""")

    def __init__(self, **kwargs):
        super(ProgressBar,self).__init__(**kwargs)

    def __call__(self, percentage):
        " Update the progress bar within the specified percent_range"
        span = (self.percent_range[1]-self.percent_range[0])
        percentage = self.percent_range[0] + ((percentage/100.0) * span)
        if clear_output and not ipython2: clear_output()
        if clear_output and ipython2: clear_output(wait=True)
        percent_per_char = 100.0 / self.width
        char_count = int(math.floor(percentage/percent_per_char)
                         if percentage<100.0 else self.width)
        blank_count = self.width - char_count
        sys.stdout.write('\r' + "%s[%s%s] %0.1f%%" % (self.label+':\n' if self.label else '',
                                                      self.fill_char * char_count,
                                                      ' '*len(self.fill_char)*blank_count,
                                                      percentage))
        sys.stdout.flush()
        time.sleep(0.0001)



class RunProgress(ProgressBar):
    """
    RunProgress breaks up the execution of a slow running command so
    that the level of completion can be displayed during execution.

    This class is designed to run commands that take a single numeric
    argument that acts additively. Namely, it is expected that a slow
    running command 'run_hook(X+Y)' can be arbitrarily broken up into
    multiple, faster executing calls 'run_hook(X)' and 'run_hook(Y)'
    without affecting the overall result.

    For instance, this is suitable for simulations where the numeric
    argument is the simulated time - typically, advancing 10 simulated
    seconds takes about twice as long as advancing by 5 seconds.
    """

    interval = param.Number(default=100, doc="""
        The run interval used to break up updates to the progress bar.""")

    run_hook = param.Callable(default=param.Dynamic.time_fn.advance, doc="""
        By default updates time in param which is very fast and does
        not need a progress bar. Should be set to some slower running
        callable where display of progress level is desired.""")


    def __init__(self, **kwargs):
        super(RunProgress,self).__init__(**kwargs)

    def __call__(self, value):
        """
        Execute the run_hook to a total of value, breaking up progress
        updates by the value specified by interval.
        """
        completed = 0
        while (value - completed) >= self.interval:
            self.run_hook(self.interval)
            completed += self.interval
            super(RunProgress, self).__call__(100 * (completed / float(value)))
        remaining = value - completed
        if remaining != 0:
            self.run_hook(remaining)
            super(RunProgress, self).__call__(100)
