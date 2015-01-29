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
from holoviews.core.options import Options, Cycle
from itertools import groupby
import pyparsing as pp

from ..operation.channel import ChannelOperation
from ..plotting import Plot

import numpy as np  # pyflakes:ignore (API import for keyword eval)

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
    syntax) separated by keyword lists for any of the normalization,
    plotting options or style options (in braces, square brackets and
    parentheses respectively). All these option sets are optional, but
    the style options must follow the plotting options which must
    follow the normalization options.

    For instance, the following string:

    Matrix [show_title=False] (interpolation='nearest') Curve color='r'

    Would specify an OptionTree with Options(show_title=False) plot
    options for Matrix and style options Options(color='r') for
    Curve.

    Note that the normalization options are just syntactic sugar for
    the the normalization plotting option. In other words:

    Matrix {-groupwise -mapwise}

    Is equivalent to:

    Matrix [normalization=3]

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

    norm_options = pp.nestedExpr(opener='{',
                                 closer='}',
                                 ignoreExpr=None
                                 ).setResultsName("norm_options")


    channelops = pp.MatchFirst(
        [pp.Literal(el.value) for el in ChannelOperation.channel_ops])

    dotted_path = pp.Combine( pp.Word(string.uppercase, exact=1)
                              + pp.Word(pp.alphas+'._'))


    pathspec = (dotted_path | channelops).setResultsName("pathspec")


    spec_group = pp.Group(pathspec
                          + pp.Optional(norm_options)
                          + pp.Optional(plot_options)
                          + pp.Optional(style_options))

    opts_spec = pp.OneOrMore(spec_group)


    @classmethod
    def process_normalization(cls, parse_group):
        """
        Given a normalization parse group (i.e. the contents of the
        braces), validate the option list and compute the appropriate
        integer value for the normalization plotting option.
        """
        if ('norm_options' not in parse_group): return None
        opts = parse_group['norm_options'][0].asList()
        if opts == []: return None

        options = ['+mapwise', '-mapwise', '+groupwise', '-groupwise']

        for normopt in options:
            if opts.count(normopt) > 1:
                raise SyntaxError("Normalization specification must not"
                                  " contain repeated %r" % normopt)

        if not all(opt in options for opt in opts):
            raise SyntaxError("Normalization option not one of %s"
                              % ", ".join(options))
        excluded = [('+mapwise', '-mapwise'), ('+groupwise', '-groupwise')]
        for pair in excluded:
            if all(exclude in opts for exclude in pair):
                raise SyntaxError("Normalization specification cannot"
                                  " contain both %s and %s" % (pair[0], pair[1]))

        # If unspecified, default is +groupwise and +mapwise
        if len(opts) == 1 and opts[0].endswith('mapwise'):
            groupwise = True
            mapwise =   True if '+mapwise' in opts else False
        elif len(opts) == 1 and opts[0].endswith('groupwise'):
            mapwise = True
            groupwise = True if '+groupwise' in opts else False
        else:
            groupwise = True if '+groupwise' in opts else False
            mapwise =   True if '+mapwise' in opts else False

        return dict(groupwise=groupwise,
                    mapwise=mapwise)



    @classmethod
    def parse(cls, line):
        """
        Parse an options specification, returning a dictionary with
        path keys and {'plot':<options>, 'style':<options>} values.
        """
        parses  = [p for p in cls.opts_spec.scanString(line)]
        if len(parses) != 1:
            raise SyntaxError("Invalid specification syntax.")
        else:
            (k,s,e) = parses[0]
            processed = line[:e]
            if (processed.strip() != line.strip()):
                raise SyntaxError("Failed to parse remainder of string: %r" % line[e:])

        parse = {}
        for group in cls.opts_spec.parseString(line):
            options, plot_options = {}, {}

            normalization = cls.process_normalization(group)
            if normalization is not None:
                options['norm'] = Options(**normalization)

            if 'plot_options' in group:
                plotopts =  group['plot_options'][0]
                options['plot'] = Options(**cls.todict(plotopts, 'brackets'))

            if 'style_options' in group:
                styleopts = group['style_options'][0]
                options['style'] = Options(**cls.todict(styleopts, 'parens'))
            parse[group['pathspec']] = options
        return parse



class ChannelSpec(Parser):
    """
    The syntax for defining a set of channel operations is as follows:

    [ op(spec) [settings] value ]+

    The components are:

    value     : Value identifier with capitalized initial letter.
    op        : The name of the operation to apply.
    spec      : Overlay specification of form (A * B) where A and B are
                 dotted path specifications.
    settings : Optional list of keyword arguments to be used as
               parameters to the operation (in square brackets).
    """

    value = pp.Word(pp.alphas+pp.nums+'_').setResultsName("value")

    op = pp.Word(pp.alphas+pp.nums+'_').setResultsName("op")

    overlay_spec = pp.nestedExpr(opener='(',
                                 closer=')',
                                 ignoreExpr=None
                             ).setResultsName("spec")

    op_settings = pp.nestedExpr(opener='[',
                                closer=']',
                                ignoreExpr=None
                            ).setResultsName("op_settings")

    channel_spec = pp.OneOrMore(pp.Group(op + overlay_spec + value
                                         + pp.Optional(op_settings)))


    @classmethod
    def parse(cls, line):
        """
        Parse a list of channel specification, returning a ChannelOperation
        """
        channel_ops = []
        parses  = [p for p in cls.channel_spec.scanString(line)]
        if len(parses) != 1:
            raise SyntaxError("Invalid specification syntax.")
        else:
            (k,s,e) = parses[0]
            processed = line[:e]
            if (processed.strip() != line.strip()):
                raise SyntaxError("Failed to parse remainder of string: %r" % line[e:])

        opmap = {op.__name__:op for op in ChannelOperation.operations}
        for group in cls.channel_spec.parseString(line):

            kwargs = {}
            operation = opmap[group['op']]
            spec = ' '.join(group['spec'].asList()[0])

            if '*' not in spec:
                raise SyntaxError("Overlay specification must contain at least one * operation")
            if  group['op'] not in opmap:
                raise SyntaxError("Operation %s not available for channel operations"
                                  % group['op'])
            if  'op_settings' in group:
                kwargs = cls.todict(group['op_settings'][0], 'brackets')

            channel_op = ChannelOperation(str(spec), operation, str(group['value']), **kwargs)
            channel_ops.append(channel_op)
        return channel_ops
