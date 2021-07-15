HoloViews Roadmap, as of 3/2019
===============================

HoloViews is maintained by a core development team who coordinate contributions from many other different users/developers. The core-developer priorities depend on funding, usage in ongoing projects, and other factors. For 2019, the scheduled tasks are:

1. **Ongoing maintenance, improved documentation and examples**:
   As always, there are various bugs and usability issues reported on the issue tracker, and we will address these as time permits.

2. **More flexible and maintainable widgets and layouts using Panel** (`#805 <https://github.com/pyviz/holoviews/issues/805>`__):
   - Re-implement HoloViews widgets using Panel, to allow them to be modified and rearranged flexibly
   - Re-implement HoloViews layouts using Panel where feasible, to allow more powerful and flexible arrangements

3. **Separate packages into holoviews-core and extensions**:
   HoloViews has always been designed with a backend-independent core that helps you describe and work with your data, along with plotting-library-specific backends that generate visualizations. To make this separation explicit and to make it simpler to generate objects in contexts where no backend is available, the holoviews package needs to be split into a core (probably to be called `holoviews-core`) along with packages per extension and possibly a `holoviews` metapackage that installs all of them as the single current package does.

4. **Improved developer docs**:
   Because HoloViews includes both JavaScript and Python code and both core data-description features and optional backend-specific plotting support, it can be difficult to understand how to contribute to HoloViews development. We need much better developer docs to make it simpler to join the HoloViews team!

5. **Generalize Annotator objects**:
   The examples at `EarthSim.pyviz.org <https://earthsim.pyviz.org>`__ show that the drawing-tool support we added to Bokeh and HoloViews makes it possible to create sophisticated applications for collecting user inputs such as annotations on plots and ML training examples.  However, it is currently difficult to specify and create such objects for new tasks, and we hope to be able to provide more general mechanisms for collecting user inputs.

6. **Finalize HoloViews 2.0 API**:
   The HoloViews API includes various older and discouraged programming styles that we would like to deprecate and remove, but first we need to discuss those and get consensus before committing to any breaking changes as we move to 2.0.

7. **Deeper Datashader support**:
   HoloViews offers an interface to `Datashader <https://datashader.org>`__ to allow working with large datasets in web browsers while avoiding overplotting and other issues. Improving this interface will help make using Datashader with HoloViews more seamless, including porting Datashader's colormapping support into Bokeh (and thus supporting colorbars and hover), adding datashader operations for the various Datashader geo functions to GeoViews, adding support for datashading additional HoloViews element types more efficiently, and making it simpler to switch between datashaded and regular Bokeh plots.


Other things we'd like to see in HoloViews but have not currently scheduled for 2019 include:

1. **More fully support Plotly backend**:
   The Plotly backend is not quite as well supported as the Bokeh or Matplotlib backends, and we'd love to get contributions from Plotly users who can expand that support, especially for using linked streams to provide dynamic behavior in plots.

2. **Use Matplotlib's OO interface instead of pyplot**:
   HoloViews currently uses Matplotlib's pyplot interface, which causes some headaches for selecting appropriate backends.  Rewriting to use the OO interface would take a good bit of work, but would make things run more smoothly.

3. **SVG export for Bokeh**:
   HoloViews supports SVG export of Matplotlib plots, but does not currentlky use Bokeh's SVG generation support because it produces separate SVG files per plot instead of a coherent layout.  If Bokeh could be extended to generate a laid out SVG, then Bokeh plots would be far more usable for publications.

4. **Additional element types**:
   There are always more plotting types that can be added (see e.g. the
   `PyViz roadmap <http://pyviz.org/Roadmap.html>`__), but none of these are
   needed by the core developers for current projects and so are unlikely
   to be added unless contributed by users.

5. **Better 3D support**:
   There is some improvement planned to 3D support in 2019, but there would still be a long way to go after that, and so anyone who routinely goes from 2D to 3D plotting and back could consider improving the 3D functionality available in HoloViews to make that simpler.

If any of the functionality above is interesting to you (or you have ideas of your own!) and can offer help with implementation, please open an issue on this repository or on the specific subproject repository involved. And if you are lucky enough to be in a position to fund our developers to work on it, please contact ``jbednar@anaconda.com``.

And please note that many of the features that you might think should be part of HoloViews may already be available or planned for one of the other `PyViz tools <http://pyviz.org>`__ that are designed to work well with HoloViews, so please also check out the   `PyViz Roadmap <http://http://pyviz.org/Roadmap.html>`__.


