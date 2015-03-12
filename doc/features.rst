**Overview**

* A rich library of composable elements that can be overlaid, nested and positioned with ease.
* Build datastructures that both contain and visualize your data.
* Supports rapid data exploration that naturally develops into a fully reprobucible workflow.
* Achieve complex, animated or interactive visualizations with minimal code.
* Rich semantics for indexing and slicing of data in arbitrarily high-dimensional spaces.
* Every parameter of every object includes easy-to-access documentation.
* All features available in vanilla Python 2 or 3.

**Support for maintainable, reproducible research**
  
* Enables a truly reproducible workflow by minimizing the code needed for analysis and visualization.
* Used in real research projects from conception to final publication.
* All HoloViews objects can be pickled and unpickled.
* Comparison utilities for testing purposes so you know when your results have changed and why.
* Core datastructures only depend on the numpy library.
* Archival facilities for keeping track of your work throughout a project.

**Analysis and data access features**

* Add useful semantic information to your data with Dimension objects including units, labels and data ranges.
* Easily slice and access into your data, no matter how high the dimensionality.
* Apply any suitable function to collapse your data or reduce dimensionality.
* Helpful textual representation to inform you how every level of your data may be accessed.
* Small library of operations common when working with any scientific or engineering data.
* Highly extensible: add new operations to easily apply the data transformations you need.

**Visualization features**

* Useful default settings make it easy to inspect with minimal code.
* Powerful normalization system to make understanding your data across plots easy.
* Build complex animations or interactive visualizations in seconds instead of hours or days.
* Refine the visualization of your data in an interactive, incremental manner.
* Separation of concerns: all visualization settings are kept separate from your data objects.
* Support for interactive tooltips/panning/zooming via the mpld3 backend.

**IPython Notebook support**

* Support for both IPython 2 and 3.
* Automatic, tab-completion everywhere.
* Exportable sliders and scrubber widgets.
* Display of animated formats in the notebook including gif, webm and mp4
* Useful IPython magics for configuring global display options and for customizing objects.
* Automatic archival and export of notebooks:

  + Automatic archival of HTML snapshots and cleared .ipynb files.
  + Render your data in the browser as PNG but save to SVG in the background. 

**Integration with third-party libraries**  

* Flexible interface to both the pandas and seaborn libraries
* Immediately visualize pandas data as any HoloViews object.
* Seamlessly combine and animate your seaborn plots in HoloViews rich, compositional data-structures.
