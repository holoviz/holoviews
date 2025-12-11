# HoloViews Roadmap, as of December 2025

HoloViews is maintained by a core development team who coordinate contributions from many other different users/developers. The core-developer priorities depend on funding, usage in ongoing projects, and other factors. For 2026, the scheduled tasks are:

1. **Ongoing maintenance, improved documentation and examples**:
   As always, there are various bugs and usability issues reported on the issue tracker, and we will address these as time permits. This includes reviewing and closing old issues and pull requests as part of regular project maintenance to keep the project organized and ensure relevant issues get attention.

2. **Donut plot**:
   Introduce a donut plot as a first-class visualization type in HoloViews.

3. **Backend consistency**:
   Verify that new features added to the Bokeh backend, such as sizebars and scalebars, are also supported in the Matplotlib backend. This work ensures feature parity across the backends and provides users with consistent functionality regardless of their plotting backend choice.

4. **Deprecate and remove old features** ([#6445](https://github.com/holoviz/holoviews/issues/6445)):
   Remove outdated and redundant features to streamline the codebase and reduce maintenance burden.

5. **Test suite modernization** ([#6735](https://github.com/holoviz/holoviews/pull/6735)):
   Migrate the existing test suite from unittest to pytest. This modernization will make it easier for new contributors to write and understand tests, improve test discoverability, and provide better test output and debugging capabilities.

6. **Code formatting**:
   Adopt ruff for formatting to provide a consistent contributor experience. Automated formatting reduces friction in code reviews and ensures the codebase maintains a uniform style, making it easier for contributors to focus on functionality rather than style issues.

7. **Typing improvements**:
   Improve type coverage for Non-Parameterized classes to support better editor integration and autocomplete. Better type hints will enhance the developer experience by providing more accurate IDE suggestions and catching potential type-related bugs earlier in development.

8. **Import cleanup**:
   Remove star imports across the codebase to improve clarity and maintainability. Explicit imports make it easier to understand dependencies, improve code navigation in editors, and help avoid naming conflicts.

9. **Versioned documentation**:
   Add version-specific documentation so users can access docs for current and past releases. This is particularly important for users working with older versions of HoloViews who need access to documentation matching their installed version.

10. **Governance** ([#6752](https://github.com/holoviz/holoviews/pull/6752)):
    Work toward establishing or updating the HoloViews governance model. Clear governance structures help ensure the long-term sustainability of the project and provide transparency about how decisions are made.

If any of the functionality above is interesting to you (or you have ideas of your own!) and can offer help with implementation, please open an issue on this repository. And if you are lucky enough to be in a position to fund our developers to work on it, please contact the HoloViz team.

And please note that many of the features that you might think should be part of HoloViews may already be available or planned for one of the other [HoloViz tools](https://holoviz.org) that are designed to work well with HoloViews, so please also check out the [HoloViz Roadmap](https://holoviz.org/about/roadmap.html).
