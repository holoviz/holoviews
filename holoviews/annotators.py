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
    together into a single
    """

    layers = param.List(default=[])

    table_height = param.Integer(default=150, doc="Height of the table",
                                 precedence=-1)

    table_width = param.Integer(default=400, doc="Width of the table",
                                 precedence=-1)

    def __init__(self, layers=[], **params):
        super(AnnotationManager, self).__init__(**params)
        for layer in layers:
            self.add_layer(layer)
        self.plot = ParamMethod(self._plots)
        self.editor = Tabs()
        self._tables()
        self._layout = Row(self.plot, self.editor, sizing_mode='stretch_width')

    @param.depends('layers')
    def _plots(self):
        layers = []
        for layer in self.layers:
            if isinstance(layer, Annotator):
                layers.append(layer.element)
            else:
                layers.append(layer)
        return Overlay(layers).opts(responsive=True, min_height=600)

    @param.depends('layers', watch=True)
    def _tables(self):
        tables = []
        for layer in self.layers:
            if not isinstance(layer, Annotator):
                continue
            tables += [(name, t.opts(width=self.table_width, height=self.table_height))
                       for name, t in layer.tables]
        self.editor[:] = tables

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._layout._repr_mimebundle_(include, exclude)

    def add_layer(self, layer):
        """
        Adds an Annotator or Element to the layers.

        Parameters
        ----------
        layer: (Annotator or Element)
            Layer to add to AnnotationManager
        """
        if isinstance(layer, Annotator):
            annotator.param.watch(lambda event: self.param.trigger('layers'), 'element')
        elif isinstance(layer, Element):
            raise ValueError('Annotator layer must be a Annotator subclass or a HoloViews/GeoViews element.')
        self.layers.append(layer)
        self.param.trigger('layers')



class Annotator(param.Parameterized):

    element = param.ClassSelector(class_=Element, precedence=-1, doc="""
        Element to annotate.""")

    num_objects = param.Integer(default=None, bounds=(0, None), doc="""
        The maximum number of objects to draw.""")

    style = param.Dict(default={}, doc="""
        Style to apply to the element""")

    tables = param.List(default=[], doc="""
        List of tables to edit object properties.""")

    # Once generic editing tools are merged into bokeh this could
    # include snapshot, restore and clear tools
    _tools = []    

    @property
    def _element_type(self):
        return self.param.element.class_

    @property
    def _element_name(self):
        return self._element_type.__name__

    @param.depends('element')
    def _element(self):
        return self.element.options(responsive=True, min_height=600)

    @param.depends('tables', watch=True)
    def _tables(self):
        self.editor[:] = self.tables

    def __init__(self, element=None, **params):
        super(Annotator, self).__init__(**params)
        self._initialize(element)
        self.editor = Tabs()
        self._tables()
        self.plot = ParamMethod(self._element)
        self._layout = Row(self.plot, self.editor, sizing_mode='stretch_width')

    @param.depends('element', watch=True)
    @preprocess
    def _initialize(self, element=None):
        """
        Initializes the element ready for annotation.
        """
        element = self.element if element is None else element
        self._init_element(element)
        self._link_element()

    def _init_element(self, element):
        """
        Subclasses should implement this method.
        """

    def _link_element(self):
        """
        Subclasses should implement this method.
        """

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._layout._repr_mimebundle_(include, exclude)

    @property
    def output(self):
        """
        Subclasses should return the element including all annotations.
        """

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

    element = param.ClassSelector(class_=Path, precedence=-1, doc="""
         Polygon or Path element to annotate""")

    poly_columns = param.List(default=['Group'], doc="""
        Columns to annotate the Polygons with.""")

    vertex_columns = param.List(default=[], doc="""
        Columns to annotate the Polygons with.""")

    def _init_element(self, element=None):
        if element is None or not isinstance(element, self._element_type):
            element = self._element_type(element)
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, finalize_hooks=[initialize_tools], color_index=None,
                    **self.style)
        self.element = element.options(**opts)
        poly_draw, poly_edit = PolyDraw, PolyEdit
        self._poly_stream = poly_draw(
            source=self.element, data={}, show_vertices=True,
            num_objects=self.num_objects)
        self._vertex_stream = poly_edit(
            source=self.element, vertex_style={'nonselection_alpha': 0.5},
            **style_kwargs)
        self._poly_selection = Selection1D(source=self.element)

    def _link_element(self):
        table_opts = dict(editable=True)

        # Add annotation columns to poly data
        for col in self.poly_columns+self.vertex_columns:
            if col not in self.element:
                self.element = self.element.add_dimension(col, 0, '', True)
        self._poly_stream.source = self.element
        self._vertex_stream.source = self.element
        self._poly_selection.source = self.element

        if len(self.element):
            poly_data = project(self.element).split()
            self._poly_stream.event(data={kd.name: [p.dimension_values(kd) for p in poly_data]
                                         for kd in self.element.kdims})

        poly_data = {c: self.element.dimension_values(c, expanded=False) for c in self.poly_columns}
        if len(set(len(v) for v in poly_data.values())) != 1:
            raise ValueError('poly_columns must refer to value dimensions '
                             'which vary per path while at least one of '
                             '%s varies by vertex.' % self.poly_columns)
        self._poly_table = Table(poly_data, self.poly_columns, []).opts(**table_opts)
        self._poly_link = DataLink(source=self.element, target=self._poly_table)
        self._vertex_table = Table([], self.element.kdims, self.vertex_columns).opts(**table_opts)
        self._vertex_link = VertexTableLink(self.element, self._vertex_table)
        self.tables[:] = [
            ('%s' % param_name(self.name), self._poly_table),
            ('%s Vertices' % param_name(self.name), self._vertex_table)
        ]
        self.param.trigger('tables')

    @property
    def output(self):
        element = self._poly_stream.element
        for path in element.data:
            for col in self.poly_columns:
                if len(path[col]):
                    path[col] = path[col][0]
        return element

    @property
    def selected(self):
        index = self._poly_selection.index
        if not index:
            return []
        return [p for i, p in enumerate(self._poly_stream.element.split()) if i in index]


class PolyAnnotator(PathAnnotator):
    """
    PolyAnnotator allows drawing and editing Polygons and associating
    values with each polygon and each vertex of a Polygon using a table.
    """

    element = param.ClassSelector(class_=Polygons, precedence=-1, doc="""
         Polygon or Path element to annotate""")


class PointAnnotator(Annotator):
    """
    PolyAnnotator allows drawing and editing Points and associating
    values with each point using a table.
    """

    element = param.ClassSelector(class_=Points, precedence=-1, doc="""
        Element to annotate.""")

    point_columns = param.List(default=['Size'], doc="""
        Columns to annotate the Points with.""", precedence=-1)

    # Link between Points and Table 
    _point_table_link = lambda source, target: DataLink(source=source, target=target)

    # Transform from Points element to table element data
    _table_transform = lambda element: element

    def _init_element(self, element):
        if element is None or not isinstance(element, self._element_type):
            element = self._element_type(element)
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, finalize_hooks=[initialize_tools],
                    **self.style)
        self.element = element.options(**opts)
        self._point_stream = PointDraw(source=self.element, drag=True, data={},
                                       num_objects=self.num_objects)

    def _link_element(self):
        table_opts = dict(editable=True)
        for col in self.point_columns:
            if col not in self.element:
                self.element = self.element.add_dimension(col, 0, None, True)
        self._point_stream.source = self.element
        transformed = self._table_transform(self.element)
        self._point_table = Table(transformed).opts(**table_opts)
        self._point_link = self._point_table_link(source=self.element, target=self._point_table)
        self._point_selection_link = SelectionLink(source=self.element, target=self._point_table)
        self._point_selection = Selection1D(source=self.element)
        self.tables[:] = [('%s' % param_name(self.name), self._point_table)]

    @property
    def output(self):
        return self._point_stream.element

    @property
    def selected(self):
        return self.element.iloc[self._point_selection.index]
