from __future__ import absolute_import

import param

from panel.param import ParamMethod
from panel.layout import Row, Tabs
from panel.util import param_name

from .core import Element, Overlay
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


class AnnotationManager(param.Parameterized):
    """
    The AnnotationManager allows combining any number of Annotators
    and elements together into a single linked, displayable panel
    object.

    The manager consists of two main components, the `plot` to
    annotate and the `editor` containing the annotator tables. These
    are combined into a `layout` by default which are displayed when
    the manager is visualized.
    """

    layers = param.List(default=[], doc="""
        Annotators and/or Elements to manage.""")

    opts = param.Dict(default={'responsive': True, 'min_height': 400}, doc="""
        The options to apply to the plot layers.""")

    table_opts = param.Dict(default={'width': 400}, doc="""
        The options to apply to editor tables.""")

    def __init__(self, layers=[], **params):
        super(AnnotationManager, self).__init__(**params)
        for layer in layers:
            self.add_layer(layer)
        self.plot = ParamMethod(self._get_plot)
        self.editor = Tabs()
        self._update_editor()
        self.layout = Row(self.plot, self.editor, sizing_mode='stretch_width')

    @param.depends('layers')
    def _get_plot(self):
        layers = []
        for layer in self.layers:
            if isinstance(layer, Annotator):
                layers.append(layer.element)
            else:
                layers.append(layer)
        overlay = Overlay(layers)
        if len(overlay):
            overlay = overlay.collate()
        return overlay.opts(**self.opts)

    @param.depends('layers', watch=True)
    def _update_editor(self):
        tables = []
        for layer in self.layers:
            if not isinstance(layer, Annotator):
                continue
            tables += [(name, t.opts(**self.table_opts)) for name, t in layer._tables]
        self.editor[:] = tables

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.layout._repr_mimebundle_(include, exclude)

    def add_layer(self, layer):
        """Adds a layer to the manager.

        Adds an Annotator or Element to the layers managed by the
        annotation manager.

        Args:
            layer (Annotator or Element): Layer to add to the manager
        """
        if isinstance(layer, Annotator):
            layer.param.watch(lambda event: self.param.trigger('layers'), 'element')
        elif isinstance(layer, Element):
            raise ValueError('Annotator layer must be a Annotator subclass '
                             'or a HoloViews/GeoViews element.')
        self.layers.append(layer)
        self.param.trigger('layers')



class Annotator(param.Parameterized):
    """
    An Annotator allows drawing, editing and annotating a specific
    type of element. Each Annotator consists of the `plot` to draw and
    edit the element and the `editor`, which contains a list of tables,
    which make it possible to annotate each object in the element with
    additional properties defined in the `annotations`.
    """

    annotations = param.ClassSelector(default=[], class_=(dict, list), doc="""
        Annotations to associate with each object.""")

    element = param.ClassSelector(class_=Element, doc="""
        The Element to edit and annotate.""")

    num_objects = param.Integer(default=None, bounds=(0, None), doc="""
        The maximum number of objects to draw.""")

    opts = param.Dict(default={}, doc="""
        Opts to apply to the element.""")

    table_transforms = param.HookList(default=[], doc="""
        Transform(s) to apply to element when converting data to Table.
        The functions should accept the Annotator and the transformed
        element as input.""")

    table_opts = param.Dict(default={'editable': True}, doc="""
        Opts to apply to the editor table(s).""")

    # Once generic editing tools are merged into bokeh this could
    # include snapshot, restore and clear tools
    _tools = []

    _draw_stream = None

    _stream_kwargs = {}

    @property
    def _element_type(self):
        return self.param.element.class_

    @property
    def _element_name(self):
        return self._element_type.__name__

    @param.depends('element')
    def _get_plot(self):
        return self.element.options(responsive=True, min_height=600)

    def __init__(self, element=None, **params):
        super(Annotator, self).__init__(**params)
        self._tables = []
        self.editor = Tabs()
        self._selection = Selection1D()
        self._initialize(element)
        self.plot = ParamMethod(self._get_plot)
        self._layout = Row(self.plot, self.editor, sizing_mode='stretch_width')

    @param.depends('element', watch=True)
    @preprocess
    def _initialize(self, element=None):
        """
        Initializes the element ready for annotation.
        """
        element = self.element if element is None else element
        self._init_element(element)
        self._init_table()
        self._selection.source = self.element
        self.editor[:] = self._tables
        self._stream.add_subscriber(self._update_element)

    def _update_element(self, data):
        with param.discard_events(self):
            self.element = self._stream.element

    def _table_data(self):
        """
        Returns data used to initialize the table.
        """
        element = self.element
        for transform in self.table_transforms:
            element = transform(self, element)
        return element

    def _init_element(self, element):
        """
        Subclasses should implement this method.
        """

    def _init_table(self):
        """
        Subclasses should implement this method.
        """

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._layout._repr_mimebundle_(include, exclude)

    @property
    def selected(self):
        """
        Subclasses should return a new element containing currently selected objects.
        """


class PathAnnotator(Annotator):
    """
    PathAnnotator allows drawing and editing Paths and associating
    values with each path and each vertex of a path using a table.
    """

    element = param.ClassSelector(class_=Path, doc="""
        Path element to edit and annotate.""")

    show_vertices = param.Boolean(default=True, doc="""
        Whether to show vertices when drawing the Path.""")

    vertex_annotations = param.ClassSelector(default=[], class_=(dict, list), doc="""
        Columns to annotate the Polygons with.""")

    vertex_style = param.Dict(default={'nonselection_alpha': 0.5}, doc="""
        Options to apply to vertices during drawing and editing.""")

    def _init_element(self, element=None):
        if element is None or not isinstance(element, self._element_type):
            element = self._element_type(element)

        # Add annotation columns to poly data
        validate = []
        for col in self.annotations:
            if col in element:
                validate.append(col)
                continue
            init = self.annotations[col] if isinstance(self.annotations, dict) else ''
            element = element.add_dimension(col, 0, init, True)
        for col in self.vertex_annotations:
            if col in element:
                continue
            elif isinstance(self.vertex_annotations, dict):
                init = self.vertex_annotations[col]
            else:
                init = ''
            element = element.add_dimension(col, 0, init, True)

        # Validate annotations
        poly_data = {c: self.element.dimension_values(c, expanded=False)
                     for c in validate}
        if validate and len(set(len(v) for v in poly_data.values())) != 1:
            raise ValueError('annotations must refer to value dimensions '
                             'which vary per path while at least one of '
                             '%s varies by vertex.' % validate)

        # Add options to element
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, color_index=None, **self.opts)
        self.element = element.options(**opts)

    def _init_table(self):
        self._stream = PolyDraw(
            source=self.element, data={}, num_objects=self.num_objects,
            show_vertices=self.show_vertices, vertex_style=self.vertex_style
        )
        self._vertex_stream = PolyEdit(
            source=self.element, vertex_style=self.vertex_style,
        )

        table_data = self._table_data()
        self._table = Table(table_data, list(self.annotations), []).opts(**self.table_opts)
        self._poly_link = DataLink(self.element, self._table)
        self._vertex_table = Table(
            [], self.element.kdims, list(self.vertex_annotations)
        ).opts(**self.table_opts)
        self._vertex_link = VertexTableLink(self.element, self._vertex_table)
        self._tables = [
            ('%s' % param_name(self.name), self._table),
            ('%s Vertices' % param_name(self.name), self._vertex_table)
        ]

    def _update_element(self):
        element = self._poly_stream.element
        if (element.interface.datatype == 'multitabular' and
            element.data and isinstance(element.data[0], dict)):
            for path in element.data:
                for col in self.annotations:
                    if len(path[col]):
                        path[col] = path[col][0]
        with param.discard_events(self):
            self.element = element

    @property
    def selected(self):
        index = self._selection.index
        data = [p for i, p in enumerate(self._stream.element.split()) if i in index]
        return self.output.clone(data)


class PolyAnnotator(PathAnnotator):
    """
    PolyAnnotator allows drawing and editing Polygons and associating
    values with each polygon and each vertex of a Polygon using a table.
    """

    element = param.ClassSelector(class_=Polygons, doc="""
         Polygon element to edit and annotate.""")


class PointAnnotator(Annotator):
    """
    PolyAnnotator allows drawing and editing Points and associating
    values with each point using a table.
    """

    element = param.ClassSelector(class_=Points, doc="""
        Points element to edit and annotate.""")

    # Link between Points and Table
    _point_table_link = lambda self, source, target: DataLink(source=source, target=target)

    def _init_element(self, element):
        if element is None or not isinstance(element, self._element_type):
            element = self._element_type(element)

        # Add annotations
        for col in self.annotations:
            if col in element:
                continue
            init = self.annotations[col] if isinstance(self.annotations, dict) else None
            element = element.add_dimension(col, 0, init, True)

        # Add options
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, **self.opts)
        self.element = element.options(**opts)

    def _init_table(self):
        self._stream = PointDraw(
            source=self.element, data={}, num_objects=self.num_objects
        )
        table_data = self._table_data()
        self._table = Table(table_data).opts(**self.table_opts)
        self._point_link = self._point_table_link(self.element, self._table)
        self._point_selection_link = SelectionLink(self.element, self._table)
        self._tables = [('%s' % param_name(self.name), self._table)]

    @property
    def selected(self):
        return self.element.iloc[self._point_selection.index]
