"""
Tests the OptsCompleter class for tab-completion in the opts magic.
"""

from unittest import SkipTest
try:
    from holoviews.ipython import IPTestCase
    from holoviews.ipython.magics import OptsCompleter
except ImportError:
    raise SkipTest("Required dependencies not satisfied for testing OptsCompleter")


class TestOptsCompleter(IPTestCase):

    def setUp(self):
        self.completions = {
            'AnElement':(
                ['plotoptA1', 'plotoptA2'],
                ['styleoptA1', 'styleoptA2']),
            'AnotherElement':(
                ['plotoptB1', 'plotoptB2'],
                ['styleoptB1', 'styleoptB2']),
            'BarElement':(
                ['plotoptC1', 'plotoptC2'],
                ['styleoptC1', 'styleoptC2'])}

        self.compositor_defs = {}
        self.all_keys = sorted(self.completions.keys()) + ['style(', 'plot[', 'norm{']

        super().setUp()

    def test_completer_setup(self):
        "Test setup_completions for the real completion set"
        completions = OptsCompleter.setup_completer()
        self.assertEqual(completions, OptsCompleter._completions)
        self.assertNotEqual(completions, {})

    def test_completions_simple1(self):
        suggestions = OptsCompleter.line_completer('%%opts An',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions, self.all_keys)

    def test_completions_simple2(self):
        "Same as above even though the selected completion is different"
        suggestions = OptsCompleter.line_completer('%%opts Anoth',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions, self.all_keys)

    def test_completions_invalid_plot1(self):
        "Same as above although the syntax is invalid"
        suggestions = OptsCompleter.line_completer('%%opts Ano [',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions, self.all_keys)

    def test_completions_short_plot1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement [',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions, ['plotoptA1=', 'plotoptA2='])

    def test_completions_long_plot1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement plot[',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions, ['plotoptA1=', 'plotoptA2='])

    def test_completions_short_style1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement (',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  ['styleoptA1=', 'styleoptA2='])

    def test_completions_long_style1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement style(',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  ['styleoptA1=', 'styleoptA2='])

    def test_completions_short_norm1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement {',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  ['+axiswise', '+framewise'])

    def test_completions_long_norm1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement norm{',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  ['+axiswise', '+framewise'])

    def test_completions_short_plot_closed1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement [test=1]',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  self.all_keys)

    def test_completions_long_plot_closed1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement plot[test=1]',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  self.all_keys)

    def test_completions_short_plot_long_style1(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement [test=1] AnotherElement style(',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  ['styleoptB1=', 'styleoptB2='])

    def test_completions_short_plot_long_style2(self):
        "Suggest corresponding plot options"
        suggestions = OptsCompleter.line_completer('%%opts AnElement [test=1] BarElement style(',
                                                   self.completions, self.compositor_defs)
        self.assertEqual(suggestions,  ['styleoptC1=', 'styleoptC2='])
