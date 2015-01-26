"""
The magics offered by the holoviews IPython extension are powerful and
support rich, compositional specifications. To avoid the the brittle,
convoluted code that results from trying to support the syntax in pure
Python, this file defines suitable parsers using pyparsing that are
cleaner and easier to understand.

Pyparsing is required by matplotlib and will therefore be available if
holoviews is being used in conjunction with matplotlib.
"""

import string
# Should pass some explicit namespace to eval...
from holoviews.core.settings import Settings, Cycle
from itertools import groupby
import pyparsing as pp

class Parser(object):
    """
    Base class for magic line parsers, designed for forgiving parsing
    of keyword lists.
    """

    @classmethod
    def _strip_commas(cls, kw):
        "Strip out any leading/training commas from the token"
        kw = kw[:-1] if kw[-1]==',' else kw
        return kw[1:] if kw[0]==',' else kw

    @classmethod
    def collect_tokens(cls, parseresult, mode):
        """
        Collect the tokens from a (potentially) nested parse result.
        """
        inner = '(%s)' if mode=='parens' else '[%s]'
        if parseresult is None: return []
        tokens = []
        for token in parseresult.asList():
            # If value is a tuple, the token will be a list
            if isinstance(token, list):
                tokens[-1] = tokens[-1] + (inner % ''.join(token))
            else:
                if token.strip() == ',': continue
                tokens.append(cls._strip_commas(token))
        return tokens

    @classmethod
    def todict(cls, parseresult, mode='parens'):
        """
        Helper function to return dictionary given the parse results
        from a pyparsing.nestedExpr object (containing keywords).
        """
        grouped, kwargs = [], {}
        tokens = cls.collect_tokens(parseresult, mode)
        # Group tokens without '=' and append to last token containing '='
        for group in groupby(tokens, lambda el: '=' in el):
            (val, items) = group
            if val is True:
                grouped += [el for el in items]
            if val is False:
                grouped[-1] += ''.join(items)

        for keyword in grouped:
            try:     kwargs.update(eval('dict(%s)' % keyword))
            except:  raise SyntaxError("Could not evaluate keyword: %r" % keyword)
        return kwargs



class OptsSpec(Parser):
    """
    An OptsSpec is a string specification that describes an
    OptionTree. It is a list of tree path specifications (using dotted
    syntax) separated by keyword lists for either plotting options (in
    square brackets) and/or style options (in parentheses). Both sets
    of keywords lists are optional, by an styles keywords must follow
    plotting keywords (if present).

    For instance, the following string:

    Matrix [show_title=False] (interpolation='nearest') Curve color='r'

    Would specify an OptionTree with Settings(show_title=False) plot
    options for Matrix and style options Settings(color='r') for
    Curve.

    The parser is fairly forgiving; commas between keywords are
    optional and additional spaces are often allowed. The only
    restriction is that keywords *must* be immediately followed by the
    '=' sign (no space).
    """

    plot_options = pp.nestedExpr(opener='[',
                              closer=']',
                              ignoreExpr=None
                                ).setResultsName("plot_options")

    style_options = pp.nestedExpr(opener='(',
                                  closer=')',
                                  ignoreExpr=None
                                  ).setResultsName("style_options")

    pathspec = pp.Combine(
                          pp.Word(string.uppercase, exact=1)
                        + pp.Word(pp.alphas+'._')
                         ).setResultsName("pathspec")

    spec_group = pp.Group(pathspec
                          + pp.Optional(plot_options)
                          + pp.Optional(style_options))

    opts_spec = pp.OneOrMore(spec_group)


    @classmethod
    def parse(cls, line):
        """
        Parse an options specification, returning a dictionary with
        path keys and {'plot':<settings>, 'style':<settings>} values.
        """
        parses  = [p for p in cls.opts_spec.scanString(line)]
        if len(parses) != 1:
            raise SyntaxError("Invalid specification syntax.")
        else:
            (k,s,e) = parses[0]
            processed = line[:e]
            if (processed != line):
                raise SyntaxError("Failed to parse remainder of string: %r" % line[e:])

        parse = {}
        for group in cls.opts_spec.parseString(line):
            settings = {}
            if 'plot_options' in group:
                plotopts =  group['plot_options'][0]
                settings['plot'] = Settings(**cls.todict(plotopts, 'brackets'))
            if 'style_options' in group:
                styleopts = group['style_options'][0]
                settings['style'] = Settings(**cls.todict(styleopts, 'parens'))
            parse[group['pathspec']] = settings
        return parse
