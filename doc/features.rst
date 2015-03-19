Features
________


**Overview**

* Lets you build data structures that both contain and visualize your data.
* Includes a rich `library of composable elements <Tutorials/Elements>`_ that can be overlaid, nested and positioned with ease.
* Supports `rapid data exploration <Tutorials/Exploring_Data>`_ that naturally develops into a `fully reproducible workflow <Tutorials/Exporting>`_.
* You can create complex animated or interactive visualizations with minimal code.
* Rich semantics for `indexing and slicing of data in arbitrarily high-dimensional spaces <Tutorials/Transforming_Data>`_.
* Every parameter of every object includes easy-to-access documentation.
* All features `available in vanilla Python 2 or 3 <Tutorials/Options>`_, with minimal dependencies.
* All examples on the website are tested automatically each night, using the latest version of the code.

**Support for maintainable, reproducible research**
  
* Supports a truly reproducible workflow by minimizing the code needed for analysis and visualization.
* Already used in a variety of research projects, from conception to final publication.
* All HoloViews objects can be pickled and unpickled.
* Provides `comparison utilities <https://ioam.github.io/holoviews/Reference_Manual/holoviews.element.html#holoviews.element.comparison.Comparison>`_ for testing, so you know when your results have changed and why.
* Core data structures only depend on the numpy and param libraries.
* Provides `export and archival facilities <Tutorials/Exporting>`_ for keeping track of your work throughout the lifetime of a project.

**Analysis and data access features**

* Allows you to annotate your data with dimensions, units, labels and data ranges.
* Easily `slice and access <Tutorials/Transforming_Data>`_ regions of your data, no matter how high the dimensionality.
* Apply any suitable function to collapse your data or reduce dimensionality.
* Helpful textual representation to inform you how every level of your data may be accessed.
* Includes small library of common operations for any scientific or engineering data.
* Highly extensible: add new operations to easily apply the data transformations you need.

**Visualization features**

* Useful default settings make it easy to inspect data, with minimal code.
* Powerful normalization system to make understanding your data across plots easy.
* Build `complex animations or interactive visualizations in seconds  <Tutorials/Exploring_Data>`_ instead of hours or days.
* Refine the visualization of your data interactively and incrementally.
* Separation of concerns: all visualization settings are kept separate from your data objects.
* Support for interactive tooltips/panning/zooming, via the optional mpld3 backend.

**IPython Notebook support**

* Support for both IPython 2 and 3.
* Automatic tab-completion everywhere.
* Exportable sliders and scrubber widgets.
* Automatic display of animated formats in the notebook or for export, including gif, webm, and mp4.
* Useful IPython magics for configuring global display options and for customizing objects.
* `Automatic archival and export of notebooks <Tutorials/Exporting>`_, including extracting figures as SVG, generating a static HTML copy of your results for reference, and storing your optional metadata like version control information.

**Integration with third-party libraries**  

* Flexible interface to both the `pandas and Seaborn libraries <Tutorials/Pandas_Seaborn>`_
* Immediately visualize pandas data as any HoloViews object.
* Seamlessly combine and animate your Seaborn plots in HoloViews rich, compositional data-structures.
