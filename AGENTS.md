# AGENTS.md

Instructions for AI coding agents working on the HoloViews project.

## Project overview

HoloViews is a Python library designed to make data analysis and visualization seamless and simple. It allows users to work with data and its visualization interactively, bundling data with metadata to support both analysis and visualization.

## Prerequisites

- [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Install [Pixi](https://pixi.sh/latest/#installation)

## Setup commands

- Fork the repository on GitHub: [github.com/holoviz/holoviews](https://github.com/holoviz/holoviews)
- Clone your fork: `git clone https://github.com/<Your Username>/holoviews && cd holoviews`
- Set up development environment: `pixi run setup-dev` (this runs `install`, `download-data`, and other setup tasks)
- Alternatively, just install dependencies: `pixi run install` (editable install of HoloViews)
- Download test data: `pixi run download-data`
- Sync git tags with upstream (if working from a fork): `pixi run sync-git-tags`
- See all available commands: `pixi task list`
- Install pre-commit hooks (recommended): `pixi run lint-install`
- Activate the environment (optional): `pixi shell` (equivalent to activating a virtual environment)

## Testing instructions

Run tests with these pixi tasks:

```bash
pixi run test-unit      # Unit tests (pytest)
pixi run test-example   # Example/notebook tests (nbval)
pixi run test-ui        # UI tests with Playwright (first run downloads Chrome)
```

**Testing in specific environments:**

- Available environments: `test-39`, `test-310`, `test-311`, `test-312`, `test-core`
- Run in specific environment: `pixi run -e test-312 test-unit`
- The `test-core` environment has only core dependencies

**Additional testing tips:**

- Run specific test file: `pixi run pytest holoviews/tests/path/to/test_file.py`
- Run linting: `pixi run lint`
- Run linting before each commit automatically: `pixi run lint-install`
- Fix any test errors before committing changes
- Add or update tests for code you change, even if not explicitly requested

**Before opening a PR:**

- Run tests locally to be considerate of CI resources
- Group commits into meaningful chunks before pushing

## Code style

- Follow PEP 8 Python style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Import order: standard library, third-party packages, local imports
- Use descriptive variable names
- Add type hints where appropriate

## Docstring format

### Preferred format for examples

Use indented code blocks (`::`), NOT doctest format (`>>>`):

**DO:**

```python
"""
Examples
--------
Create a basic scatter plot::

    import holoviews as hv
    hv.extension('bokeh')
    scatter = hv.Scatter(data)
    scatter
"""
```

**DON'T:**

```python
"""
Examples
--------
Create a basic scatter plot:

    >>> import holoviews as hv
    >>> scatter = hv.Scatter(data)
"""
```

### Rationale

- Code blocks are easier to copy-paste in VS Code and other editors
- No need for `>>>` and `...` continuation markers
- Cleaner, more readable code examples
- Still fully compatible with NumPy docstring conventions and Sphinx

### NumPy docstring sections

Use these standard sections in order:

1. **Short summary** (one line)
2. **Extended summary** (optional, multiple paragraphs)
3. **Parameters** - function/method arguments
4. **Returns** - return value description
5. **Raises** - exceptions that may be raised
6. **See Also** - related classes/functions
7. **Notes** - additional information
8. **References** - citations, links
9. **Examples** - usage examples (use `::` format)

### Example structure

```python
class Scatter(Selection1DExpr, Chart):
    """Scatter plots show the relationship between two variables as points.

    Each point's x-position comes from the key dimension (kdim, typically
    the independent variable), while its y-position comes from the first
    value dimension (vdim, typically the dependent variable).

    Additional value dimensions can control point color, size, and other properties.

    Examples
    --------
    Create a basic scatter plot from array data::

        import numpy as np
        import holoviews as hv
        hv.extension('bokeh')

        data = np.random.randn(20).cumsum()
        scatter = hv.Scatter(data)
        scatter

    Customize appearance with options::

        scatter.opts(color='red', size=10, marker='circle')

    Create a scatter plot from tabular data with multiple value dimensions for color and size::

        import numpy as np
        import pandas as pd
        import holoviews as hv
        hv.extension('bokeh')

        data = pd.DataFrame(
            np.random.rand(100, 4),
            columns=['x', 'y', 'z', 'size']
        )
        scatter = hv.Scatter(data, kdims='x', vdims=['y', 'z', 'size'])
        scatter.opts(color='z', size=hv.dim('size')*10)

    See Also
    --------
    Curve : Line plot element
    Points : 2D point cloud element

    References
    ----------
    https://holoviews.org/reference/elements/bokeh/Scatter.html
    """
```

## Commit guidelines

- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issue numbers when applicable: "Fix #123: Description"
- Keep commits focused on a single logical change

## Pull request guidelines

- Title format should include a prefix:
  - `build:` Changes that affect the build system
  - `chore:` Changes that are not user-facing
  - `ci:` Changes to CI configuration files and scripts
  - `compat:` Compatibility with upstream packages
  - `docs:` Documentation only changes
  - `enh:` An enhancement to existing feature
  - `feat:` A new feature
  - `fix:` A bug fix
  - `perf:` A code change that improves performance
  - `refactor:` A code change that neither fixes a bug nor adds a feature
  - `test:` Adding missing tests or correcting existing tests
  - `type:` Type annotations
- Ensure all tests pass before submitting
- Run tests locally before opening or pushing to an opened PR
- Update documentation for new features
- Add examples for new elements or functionality
- Follow the existing code style and conventions
- Group commits to meaningful chunks before pushing to GitHub

## Documentation

- Build documentation: `pixi run docs-build` (takes ~1 hour)
- To speed up local builds, disable galleries:
  - `export HV_DOC_GALLERY=False` - disable main gallery
  - `export HV_DOC_REF_GALLERY=False` - disable reference gallery
- Development docs: https://dev.holoviews.org/

## Additional resources

- Developer Guide: https://holoviews.org/developer_guide/index.html
- Discord (dev channel): https://discord.gg/rb6gPXbdAr
- Discourse: https://discourse.holoviz.org/
