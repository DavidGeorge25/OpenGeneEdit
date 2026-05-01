from pathlib import Path
from tempfile import NamedTemporaryFile


def generate_circular_plasmid_png(sequence: str) -> str:
    """Generate a circular plasmid PNG from a DNA sequence."""
    cleaned_sequence = "".join(base for base in sequence.upper() if base in {"A", "C", "G", "T"})
    if len(cleaned_sequence) < 40:
        raise ValueError("DNA sequence is too short to visualize as a plasmid.")

    try:
        from dna_features_viewer import CircularGraphicRecord, GraphicFeature
    except ImportError as exc:
        raise ImportError(
            "dna_features_viewer is required. Install with: "
            "python3 -m pip install dna_features_viewer matplotlib"
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

    record = CircularGraphicRecord(
        sequence_length=len(cleaned_sequence),
        sequence=cleaned_sequence,
        features=features,
    )
    ax, _ = record.plot(figure_width=7)
    output = NamedTemporaryFile(prefix="plasmid_map_", suffix=".png", delete=False)
    output_path = Path(output.name)
    output.close()
    ax.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    ax.figure.clf()
    return str(output_path)
