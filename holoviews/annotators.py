from __future__ import absolute_import, unicode_literals

import sys

from collections import OrderedDict

import param

from panel.pane import PaneBase
from panel.layout import Row, Tabs
from panel.util import param_name

from .core import DynamicMap, Element, Layout, Overlay, Store
from .core.spaces import Callable
from .core.util import isscalar
from .element import Path, Polygons, Points, Table
from .plotting.links import VertexTableLink, DataLink, SelectionLink
from .streams import PolyDraw, PolyEdit, Selection1D, PointDraw


def preprocess(function, current=[]):
    """
    Turns a param.depends watch call into a preprocessor method, i.e.
    skips all downstream events triggered by it.
    NOTE: This is a temporary hack while the addition of preprocessors
          in param is under discussion. This only works for the first
          method which depends on a particular parameter.
          (see https://github.com/pyviz/param/issues/332)
    """
    def inner(*args, **kwargs):
        self = args[0]
        self.param._BATCH_WATCH = True
        function(*args, **kwargs)
        self.param._BATCH_WATCH = False
        self.param._watchers = []
        self.param._events = []
    return inner


class annotate(param.ParameterizedFunction):
    """
    The annotate function allows drawing, editing and annotating any
    given Element (if it is supported). The annotate function returns
    a Layout of the editable plot and an Overlay of table(s), which
    allow editing the data of the element. The edited and annotated
    data may be accessed using the element and selected properties.
    """

    annotator = param.Parameter(doc="""The current Annotator instance.""")

    annotations = param.ClassSelector(default=[], class_=(dict, list), doc="""
        Annotations to associate with each object.""")

    edit_vertices = param.Boolean(default=True, doc="""
        Whether to add tool to edit vertices.""")

    num_objects = param.Integer(default=None, bounds=(0, None), doc="""
        The maximum number of objects to draw.""")

    show_vertices = param.Boolean(default=True, doc="""
        Whether to show vertices when drawing the Path.""")

    table_transforms = param.HookList(default=[], doc="""
        Transform(s) to apply to element when converting data to Table.
        The functions should accept the Annotator and the transformed
        element as input.""")

    table_opts = param.Dict(default={'editable': True, 'width': 400}, doc="""
        Opts to apply to the editor table(s).""")

    vertex_annotations = param.ClassSelector(default=[], class_=(dict, list), doc="""
        Columns to annotate the Polygons with.""")

    vertex_style = param.Dict(default={'nonselection_alpha': 0.5}, doc="""
        Options to apply to vertices during drawing and editing.""")

    _annotator_types = OrderedDict()

    @property
    def annotated(self):
        annotated = self.annotator.object
        if Store.current_backend == 'bokeh':
            return annotated.opts(clone=True, tools=['hover'])

    @property
    def selected(self):
        selected = self.annotator.selected
        if Store.current_backend == 'bokeh':
            return selected.opts(clone=True, tools=['hover'])

    @classmethod
    def compose(cls, *annotators):
        """Composes multiple annotator layouts and elements

        The composed Layout will contain all the elements in the
        supplied annotators and an overlay of all editor tables.

        Args:
            annotators: Annotator layouts or elements to compose

        Returns:
            A new layout consisting of the overlaid plots and tables
        """
        layers = []
        tables = []
        for annotator in annotators:
            if isinstance(annotator, Layout):
                l, ts = annotator
                layers.append(l)
                tables += ts
            elif isinstance(annotator, annotate):
                layers.append(annotator.plot)
                tables += [t[0].object for t in annotator.editor]
            elif isinstance(annotator, Element):
                layers.append(annotator)
            else:
                raise ValueError("Cannot compose %s type with annotators." %
                                 type(annotator).__name__)
        tables = Overlay(tables, group='Annotator').opts(tabs=True)
        return (Overlay(layers).collate() + tables).opts(sizing_mode='stretch_width')

    def __call__(self, element, **params):
        for eltype, annotator_type in self._annotator_types.items():
            if isinstance(element, eltype):
                break
            else:
                annotator_type = None
        if annotator_type is None:
            raise ValueError('Annotation of %s element types is not '
                             'supported.' % type(element).__name__)
        self.annotator = annotator_type(element, **params)
        tables = Overlay([t[0].object for t in self.annotator.editor], group='Annotator').opts(tabs=True)
        return (self.annotator.plot + tables).opts(sizing_mode='stretch_width')



class Annotator(PaneBase):
    """
    An Annotator allows drawing, editing and annotating a specific
    type of element. Each Annotator consists of the `plot` to draw and
    edit the element and the `editor`, which contains a list of tables,
    which make it possible to annotate each object in the element with
    additional properties defined in the `annotations`.
    """

    annotations = param.ClassSelector(default=[], class_=(dict, list), doc="""
        Annotations to associate with each object.""")

    default_opts = param.Dict(default={'responsive': True, 'min_height': 400,
                                       'padding': 0.1}, doc="""
        Opts to apply to the element.""")

    object = param.ClassSelector(class_=Element, doc="""
        The Element to edit and annotate.""")

    num_objects = param.Integer(default=None, bounds=(0, None), doc="""
        The maximum number of objects to draw.""")

    table_transforms = param.HookList(default=[], doc="""
        Transform(s) to apply to element when converting data to Table.
        The functions should accept the Annotator and the transformed
        element as input.""")

    table_opts = param.Dict(default={'editable': True, 'width': 400}, doc="""
        Opts to apply to the editor table(s).""")

    # Once generic editing tools are merged into bokeh this could
    # include snapshot, restore and clear tools
    _tools = []

    # Allows patching on custom behavior
    _extra_opts = {}

    # Triggers for updates to the table
    _triggers = ['annotations', 'object', 'table_opts']

    # Links between plot and table
    _link_type = DataLink
    _selection_link_type = SelectionLink

    priority = 0.7

    @classmethod
    def applies(cls, obj):
        if 'holoviews' not in sys.modules:
            return False
        return isinstance(obj, cls.param.object.class_)

    @property
    def _element_type(self):
        return self.param.object.class_

    @property
    def _object_name(self):
        return self._element_type.__name__

    def __init__(self, object=None, **params):
        super(Annotator, self).__init__(None, **params)
        self.object = self._process_element(object)
        self._table_row = Row()
        self.editor = Tabs(('%s' % param_name(self.name), self._table_row))
        self.plot = DynamicMap(Callable(self._get_plot, inputs=[self.object]))
        self._tables = []
        self._init_stream()
        self._stream.add_subscriber(self._update_object, precedence=0.1)
        self._selection = Selection1D(source=self.plot)
        self._update_table()
        self._update_links()
        self.param.watch(self._update, self._triggers)
        self.layout[:] = [self.plot, self.editor]

    @param.depends('annotations', 'object', 'default_opts')
    def _get_plot(self):
        return self._process_element(self.object)

    def _get_model(self, doc, root=None, parent=None, comm=None):
        return self.layout._get_model(doc, root, parent, comm)

    @preprocess
    def _update(self, event=None):
        if event and event.name == 'object':
            with param.discard_events(self):
                self.object = self._process_element(event.new)
        self._update_table()

    def _update_links(self):
        if hasattr(self, '_link'): self._link.unlink()
        self._link = self._link_type(self.plot, self._table)
        if self._selection_link_type:
            if hasattr(self, '_selection_link'): self._selection_link.unlink()
            self._selection_link = SelectionLink(self.plot, self._table)

    def _update_object(self, data=None):
        with param.discard_events(self):
            self.object = self._stream.element

    def _update_table(self):
        object = self.object
        for transform in self.table_transforms:
            object = transform(object)
        self._table = Table(object, label=param_name(self.name)).opts(
            show_title=False, **self.table_opts)
        self._update_links()
        self._table_row[:] = [self._table]

    def select(self, selector=None):
        return self.layout.select(selector)

    @classmethod
    def compose(cls, *annotators):
        """Composes multiple Annotator instances and elements

        The composed Panel will contain all the elements in the
        supplied Annotators and Tabs containing all editors.

        Args:
            annotators: Annotator objects or elements to compose

        Returns:
            A new Panel consisting of the overlaid plots and tables
        """
        layers, tables = [], []
        for a in annotators:
            if isinstance(a, Annotator):
                layers.append(a.plot)
                tables += a.tables
            elif isinstance(a, Element):
                layers.append(a)
        return Row(Overlay(layers).collate(), Tabs(*tables))

    @property
    def tables(self):
        return list(zip(self.editor._names, self.editor))

    @property
    def selected(self):
        return self.object.iloc[self._selection.index]



class PathAnnotator(Annotator):
    """
    Annotator which allows drawing and editing Paths and associating
    values with each path and each vertex of a path using a table.
    """

    edit_vertices = param.Boolean(default=True, doc="""
        Whether to add tool to edit vertices.""")

    object = param.ClassSelector(class_=Path, doc="""
        Path object to edit and annotate.""")

    show_vertices = param.Boolean(default=True, doc="""
        Whether to show vertices when drawing the Path.""")

    vertex_annotations = param.ClassSelector(default=[], class_=(dict, list), doc="""
        Columns to annotate the Polygons with.""")

    vertex_style = param.Dict(default={'nonselection_alpha': 0.5}, doc="""
        Options to apply to vertices during drawing and editing.""")

    _vertex_table_link = VertexTableLink

    _triggers = ['annotations', 'edit_vertices', 'object', 'table_opts',
                 'vertex_annotations']

    def __init__(self, object=None, **params):
        self._vertex_table_row = Row()
        super(PathAnnotator, self).__init__(object, **params)
        self.editor.append(('%s Vertices' % param_name(self.name),
                            self._vertex_table_row))

    def _init_stream(self):
        name = param_name(self.name)
        self._stream = PolyDraw(
            source=self.plot, data={}, num_objects=self.num_objects,
            show_vertices=self.show_vertices, tooltip='%s Tool' % name,
            vertex_style=self.vertex_style
        )
        if self.edit_vertices:
            self._vertex_stream = PolyEdit(
                source=self.plot, tooltip='%s Edit Tool' % name,
                vertex_style=self.vertex_style,
            )

    def _process_element(self, element=None):
        if element is None or not isinstance(element, self._element_type):
            datatype = list(self._element_type.datatype)
            datatype.remove('multitabular')
            datatype.append('multitabular')
            element = self._element_type(element, datatype=datatype)

        # Add annotation columns to poly data
        validate = []
        for col in self.annotations:
            if col in element:
                validate.append(col)
                continue
            init = self.annotations[col]() if isinstance(self.annotations, dict) else ''
            element = element.add_dimension(col, 0, init, True)
        for col in self.vertex_annotations:
            if col in element:
                continue
            elif isinstance(self.vertex_annotations, dict):
                init = self.vertex_annotations[col]()
            else:
                init = ''
            element = element.add_dimension(col, 0, init, True)

        # Validate annotations
        poly_data = {c: element.dimension_values(c, expanded=False)
                     for c in validate}
        if validate and len(set(len(v) for v in poly_data.values())) != 1:
            raise ValueError('annotations must refer to value dimensions '
                             'which vary per path while at least one of '
                             '%s varies by vertex.' % validate)

        # Add options to element
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, color_index=None, **self.default_opts)
        opts.update(self._extra_opts)
        return element.options(**opts)

    def _update_links(self):
        super(PathAnnotator, self)._update_links()
        if hasattr(self, '_vertex_link'): self._vertex_link.unlink()
        self._vertex_link = self._vertex_table_link(self.plot, self._vertex_table)

    def _update_object(self, data=None):
        if self._stream.element is not None:
            element = self._stream.element
            if (element.interface.datatype == 'multitabular' and
                element.data and isinstance(element.data[0], dict)):
                for path in element.data:
                    for col in self.annotations:
                        if not isscalar(path[col]) and len(path[col]):
                            path[col] = path[col][0]
            with param.discard_events(self):
                self.object = element

    def _update_table(self):
        name = param_name(self.name)
        annotations = list(self.annotations)
        table = self.object
        for transform in self.table_transforms:
            table = transform(table)
        table_data = {a: list(table.dimension_values(a, expanded=False))
                      for a in annotations}
        self._table = Table(table_data, annotations, [], label=name).opts(
            show_title=False, **self.table_opts)
        self._vertex_table = Table(
            [], table.kdims, list(self.vertex_annotations), label='%s Vertices' % name
        ).opts(show_title=False, **self.table_opts)
        self._update_links()
        self._table_row[:] = [self._table]
        self._vertex_table_row[:] = [self._vertex_table]

    @property
    def selected(self):
        index = self._selection.index
        data = [p for i, p in enumerate(self._stream.element.split()) if i in index]
        return self.object.clone(data)


class PolyAnnotator(PathAnnotator):
    """
    Annotator which allows drawing and editing Polygons and associating
    values with each polygon and each vertex of a Polygon using a table.
    """

    object = param.ClassSelector(class_=Polygons, doc="""
         Polygon element to edit and annotate.""")


class PointAnnotator(Annotator):
    """
    Annotator which allows drawing and editing Points and associating
    values with each point using a table.
    """

    object = param.ClassSelector(class_=Points, doc="""
        Points element to edit and annotate.""")

    opts = param.Dict(default={'responsive': True, 'min_height': 400,
                               'padding': 0.1, 'size': 10}, doc="""
        Opts to apply to the element.""")

    def _init_stream(self):
        name = param_name(self.name)
        self._stream = PointDraw(
            source=self.plot, data={}, num_objects=self.num_objects,
            tooltip='%s Tool' % name
        )

    def _process_element(self, object):
        if object is None or not isinstance(object, self._element_type):
            object = self._element_type(object)

        # Add annotations
        for col in self.annotations:
            if col in object:
                continue
            init = self.annotations[col]() if isinstance(self.annotations, dict) else None
            object = object.add_dimension(col, 0, init, True)

        # Add options
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, **self.default_opts)
        opts.update(self._extra_opts)
        return object.options(**opts)


# Register Annotators
annotate._annotator_types.update([
    (Polygons, PolyAnnotator),
    (Path, PathAnnotator),
    (Points, PointAnnotator)
])
