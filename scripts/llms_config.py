"""Config for building HoloViews markdown docs and llms.txt."""

from __future__ import annotations

from pathlib import Path

from nbsite.scripts import LlmsBuildConfig, LlmsSection, MarkdownSource

ROOT = Path(__file__).parent.parent
DOC_DIR = ROOT / "doc"
BUILTDOCS_DIR = ROOT / "builtdocs"
OUTPUT_DIR = BUILTDOCS_DIR / "markdown"

# Files that carry no LLM code-gen value and should be excluded from the build.
EXCLUDE_FILES = (
    Path("releases.md"),
    Path("roadmap.md"),
    Path("about.md"),
    Path("site_map.rst"),
)

_MD_PAGE = lambda p: p.suffix == ".md" and p.stem != "index"

CONFIG = LlmsBuildConfig(
    project_title="HoloViews",
    project_description=(
        "HoloViews is a library that makes data analysis and visualization "
        "simple by automatically generating\n"
        "the right plot elements for your data. It supports Bokeh, Matplotlib, "
        "and other backends.\n"
        "This file lists the most important documentation pages for LLM-assisted "
        "development; not all generated doc links are shown."
    ),
    markdown_root=OUTPUT_DIR,
    llms_output_path=BUILTDOCS_DIR / "llms.txt",
    sources=(
        MarkdownSource(
            source_dir=DOC_DIR,
            output_dir=OUTPUT_DIR,
            rendered_source_dir=BUILTDOCS_DIR,
            # user_guide, getting_started, reference content comes from examples/
            exclude_dir_names=(
                ".ipynb_checkpoints",
                "governance",
                "test_data",
                "user_guide",
                "getting_started",
                "reference",
            ),
            exclude_files=EXCLUDE_FILES,
        ),
        MarkdownSource(
            source_dir=ROOT / "examples",
            output_dir=OUTPUT_DIR,
            include_suffixes=(".ipynb",),
            exclude_dir_names=(".ipynb_checkpoints",),
        ),
    ),
    sections=(
        LlmsSection(
            title="getting started",
            description="Guides for installing and getting started with HoloViews",
            path_prefix=Path("."),
            path_filter=lambda p: p == Path("install.md"),
            description_builder=lambda p: "Installing HoloViews",
            group="Documentation",
        ),
        LlmsSection(
            title="elements",
            description="Element types for visualizing data with examples.",
            path_prefix=Path("reference/elements"),
            path_filter=_MD_PAGE,
            url_pattern="/markdown/reference/elements/{path}.md",
            group="Documentation",
        ),
        LlmsSection(
            title="containers",
            description="Container types for organizing elements.",
            path_prefix=Path("reference/containers"),
            path_filter=_MD_PAGE,
            url_pattern="/markdown/reference/containers/{path}.md",
            group="Documentation",
        ),
        LlmsSection(
            title="streams",
            description="Stream types for interactivity.",
            path_prefix=Path("reference/streams"),
            path_filter=_MD_PAGE,
            url_pattern="/markdown/reference/streams/{path}.md",
            group="Documentation",
        ),
        LlmsSection(
            title="user guide",
            description="User guides and tutorials for HoloViews.",
            path_prefix=Path("user_guide"),
            path_filter=_MD_PAGE,
            url_pattern="/markdown/user_guide/{path}.md",
            group="Documentation",
        ),
        LlmsSection(
            title="reference manual",
            description="API reference for HoloViews modules and classes.",
            path_prefix=Path("reference_manual"),
            path_filter=_MD_PAGE,
            url_pattern="/markdown/reference_manual/{stem}.md",
            group="Documentation",
        ),
    ),
)
