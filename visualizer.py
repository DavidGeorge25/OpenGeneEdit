def generate_interactive_plasmid_plot(sequence: str):
    """Generate an interactive Bokeh plasmid map from a DNA sequence."""
    cleaned_sequence = "".join(base for base in sequence.upper() if base in {"A", "C", "G", "T"})
    if len(cleaned_sequence) < 40:
        raise ValueError("DNA sequence is too short to visualize as a plasmid.")

    # Same imports dna_features_viewer uses to set BOKEH_AVAILABLE; fail with a clear message.
    try:
        import bokeh  # noqa: F401
        from bokeh.plotting import figure, ColumnDataSource  # noqa: F401
        from bokeh.models import Range1d, HoverTool  # noqa: F401
        from bokeh.core.properties import value  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Interactive maps need Bokeh (and its deps). Install with the same Python as Streamlit:\n"
            "  python3 -m pip install --user bokeh pandas\n"
            "Then restart Streamlit (stop the terminal job and run again)."
        ) from exc

    # If Streamlit started before Bokeh was installed, dna_features_viewer may have cached
    # BOKEH_AVAILABLE = False. Drop cached submodules so the next import re-evaluates.
    import sys

    for key in list(sys.modules):
        if key == "dna_features_viewer" or key.startswith("dna_features_viewer."):
            del sys.modules[key]

    try:
        from dna_features_viewer import GraphicFeature, GraphicRecord
    except ImportError as exc:
        raise ImportError(
            "dna_features_viewer is required. Install with: "
            "python3 -m pip install dna_features_viewer bokeh"
        ) from exc

    seq_len = len(cleaned_sequence)
    quarter = seq_len // 4
    features = [
        GraphicFeature(
            start=0,
            end=max(quarter, 1),
            strand=+1,
            color="#7FB3D5",
            label="Promoter",
        ),
        GraphicFeature(
            start=max(quarter, 1),
            end=max(2 * quarter, 2),
            strand=+1,
            color="#82E0AA",
            label="RBS",
        ),
        GraphicFeature(
            start=max(2 * quarter, 2),
            end=max(3 * quarter, 3),
            strand=+1,
            color="#F7DC6F",
            label="CDS",
        ),
        GraphicFeature(
            start=max(3 * quarter, 3),
            end=seq_len,
            strand=-1,
            color="#F1948A",
            label="Terminator",
        ),
    ]

    record = GraphicRecord(
        sequence_length=len(cleaned_sequence),
        sequence=cleaned_sequence,
        features=features,
    )
    # dna_features_viewer uses figure_width in inches; figure_height is multiplied by 100
    # for Bokeh px height — must be int or you get 320.0 and Bokeh 3 rejects non-int height.
    return record.plot_with_bokeh(figure_width=9, figure_height=3)
