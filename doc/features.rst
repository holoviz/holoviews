Features
--------

**Overview**

* Lets you build data structures that both contain and visualize your data.
* Includes a rich `library of composable elements <http://www.holoviews.org/Tutorials/Elements.html>`_ that can be overlaid, nested and positioned with ease.
* Supports `rapid data exploration <http://www.holoviews.org/Tutorials/Exploring_Data.html>`_ that naturally develops into a `fully reproducible workflow <http://www.holoviews.org/Tutorials/Exporting.html>`_.
* Create interactive visualizations that can be controlled via widgets or via custom events in Python using the 'streams' system. When using the bokeh backend, you can use streams to directly interact with your plots.
* Rich semantics for `indexing and slicing of data in arbitrarily high-dimensional spaces <http://www.holoviews.org/Tutorials/Sampling_Data.html>`_.
* Plotting output using the `Matplotlib <http://www.holoviews.org/Tutorials/Elements.html>`_, `Bokeh <http://www.holoviews.org/Tutorials/Bokeh_Elements.html>`_, and `plotly <http://plot.ly/>`_ backends.
* A variety of data interfaces to work with tabular and N-dimensional array data using `NumPy <http://www.numpy.org/>`_, `pandas <http://pandas.pydata.org/>`_, `dask <http://dask.pydata.org/en/latest/>`_, `iris <http://scitools.org.uk/iris/>`_ and `xarray <http://xarray.pydata.org/en/stable/>`_.
* Every parameter of every object includes easy-to-access documentation.
* All features `available in vanilla Python 2 or 3 <http://www.holoviews.org/Tutorials/Options.html>`_, with minimal dependencies.

**Support for maintainable, reproducible research**
  
* Supports a truly reproducible workflow by minimizing the code needed for analysis and visualization.
* Already used in a variety of research projects, from conception to final publication.
* All HoloViews objects can be pickled and unpickled.
* Provides comparison utilities for testing, so you know when your results have changed and why.
* Core data structures only depend on the numpy and param libraries.
* Provides `export and archival facilities <http://www.holoviews.org/Tutorials/Exporting.html>`_ for keeping track of your work throughout the lifetime of a project.

**Analysis and data access features**

* Allows you to annotate your data with dimensions, units, labels and data ranges.
* Easily `slice and access <http://www.holoviews.org/Tutorials/Sampling_Data.html>`_ regions of your data, no matter how high the dimensionality.
* Apply any suitable function to collapse your data or reduce dimensionality.
* Helpful textual representation to inform you how every level of your data may be accessed.
* Includes small library of common operations for any scientific or engineering data.
* Highly extensible: add new operations to easily apply the data transformations you need.

**Visualization features**

* Useful default settings make it easy to inspect data, with minimal code.
* Powerful normalization system to make understanding your data across plots easy.
* Build `complex animations or interactive visualizations in seconds  <http://www.holoviews.org/Tutorials/Exploring_Data.html>`_ instead of hours or days.
* Refine the visualization of your data interactively and incrementally.
* Separation of concerns: all visualization settings are kept separate from your data objects.
* Support for fully interactive plots using the `Bokeh backend <http://www.holoviews.org/Tutorials/Bokeh_Backend.html>`_.

**Jupyter Notebook support**

* Support for all recent releases of IPython and Jupyter Notebooks.
* Automatic tab-completion everywhere.
* Exportable sliders and scrubber widgets.
* Custom interactivity using streams and notebook comms to dynamically updating plots.
* Automatic display of animated formats in the notebook or for export, including gif, webm, and mp4.
* Useful IPython magics for configuring global display options and for customizing objects.
* `Automatic archival and export of notebooks <http://www.holoviews.org/Tutorials/Exporting.html>`_, including extracting figures as SVG, generating a static HTML copy of your results for reference, and storing your optional metadata like version control information.
