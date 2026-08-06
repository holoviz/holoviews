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
    Path("FAQ.md"),
)

GETTING_STARTED = {
    Path("install.md"): "Installing HoloViews",
}

REFERENCE = {
    Path("reference/elements/index.md"): "Element types for visualizing data",
    Path("reference/containers/index.md"): "Container types for organizing elements",
    Path("reference/streams/index.md"): "Stream types for interactivity",
    Path("reference/features/index.md"): "Feature summaries and explanations",
    Path("reference/apps/index.md"): "Interactive app examples",
}


def _label(path: Path) -> str:
    """Title-cased label, using parent dir name for index pages."""
    stem = path.parent.name if path.stem == "index" else path.stem
    return stem.replace("_", " ").replace("-", " ").title()


def dict_filter(mapping: dict[Path, str]):
    return lambda path: path in mapping


def dict_description(mapping: dict[Path, str]):
    return lambda path: mapping[path]


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
    markdown_base_url="/markdown",
    sources=(
        MarkdownSource(
            source_dir=DOC_DIR,
            output_dir=OUTPUT_DIR,
            rendered_source_dir=BUILTDOCS_DIR,
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
            path_filter=dict_filter(GETTING_STARTED),
            label_builder=_label,
            description_builder=dict_description(GETTING_STARTED),
            group="Documentation",
        ),
        LlmsSection(
            title="reference",
            description="API reference, elements, containers, and features",
            path_prefix=Path("reference"),
            path_filter=dict_filter(REFERENCE),
            label_builder=_label,
            description_builder=dict_description(REFERENCE),
            group="Documentation",
        ),
        LlmsSection(
            title="elements",
            description="Element types for visualizing data with examples.",
            path_prefix=Path("reference/elements"),
            path_filter=lambda p: p.suffix == ".md" and p.stem != "index",
            url_pattern="/markdown/reference/elements/{path}.md",
            group="Documentation",
        ),
        LlmsSection(
            title="containers",
            description="Container types for organizing elements.",
            path_prefix=Path("reference/containers"),
            path_filter=lambda p: p.suffix == ".md" and p.stem != "index",
            url_pattern="/markdown/reference/containers/{path}.md",
            group="Documentation",
        ),
        LlmsSection(
            title="streams",
            description="Stream types for interactivity.",
            path_prefix=Path("reference/streams"),
            path_filter=lambda p: p.suffix == ".md" and p.stem != "index",
            url_pattern="/markdown/reference/streams/{path}.md",
            group="Documentation",
        ),
        LlmsSection(
            title="user guide",
            description="User guides and tutorials for HoloViews.",
            path_prefix=Path("user_guide"),
            path_filter=lambda p: p.suffix == ".md" and p.stem != "index",
            url_pattern="/markdown/user_guide/{path}.md",
            group="Documentation",
        ),
    ),
)
