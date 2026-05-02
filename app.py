import time
from datetime import datetime

import streamlit as st

from inference import parse_thought_and_sequence, run_inference
from visualizer import generate_interactive_plasmid_plot


def _apply_rag_substitution(thought: str, sequence: str):
    try:
        from igem_rag import apply_rag_substitution as rag_fn

        return rag_fn(thought, sequence, progress_cb=None, log_context="streamlit")
    except Exception as exc:
        flat = "".join((sequence or "").upper().split())
        return flat, {"enabled": False, "error": str(exc)}


def render_bokeh_figure(fig, *, use_container_width: bool = True) -> None:
    """Prefer streamlit-bokeh (non-deprecated); fall back to st.bokeh_chart on older stacks."""
    try:
        from streamlit_bokeh import streamlit_bokeh

        streamlit_bokeh(fig, use_container_width=use_container_width)
    except ImportError:
        st.bokeh_chart(fig, use_container_width=use_container_width)
        st.caption(
            "Tip: install **streamlit-bokeh** (requires **Python 3.10+** and a matching Bokeh pin) "
            "to replace deprecated `st.bokeh_chart`. On Python 3.9, use Bokeh **2.4.3** with "
            "`st.bokeh_chart` only."
        )
    except Exception as exc:
        st.warning(f"streamlit-bokeh could not render this figure ({exc}). Using legacy `st.bokeh_chart`.")
        st.bokeh_chart(fig, use_container_width=use_container_width)


def stream_tokens(text: str, delay_seconds: float = 0.03):
    """Yield progressively larger text chunks for a typewriter effect."""
    words = text.split()
    partial = []
    for word in words:
        partial.append(word)
        yield " ".join(partial)
        time.sleep(delay_seconds)


def estimate_tm_celsius(sequence: str) -> float:
    """Simple Wallace-rule estimate for demo readiness."""
    seq = sequence.upper()
    at = seq.count("A") + seq.count("T")
    gc = seq.count("G") + seq.count("C")
    return (2 * at) + (4 * gc)


def to_fasta(sequence: str, record_name: str = "compiled_sequence") -> str:
    width = 80
    lines = [sequence[i : i + width] for i in range(0, len(sequence), width)]
    return f">{record_name}\n" + "\n".join(lines) + "\n"


def to_genbank(sequence: str, record_name: str = "compiled_sequence") -> str:
    date_str = datetime.utcnow().strftime("%d-%b-%Y").upper()
    locus = f"LOCUS       {record_name[:16]:<16}{len(sequence):>11} bp    DNA     circular SYN {date_str}"
    definition = "DEFINITION  Synthetic compiler generated DNA construct."
    accession = "ACCESSION   ."
    version = "VERSION     ."
    source = "SOURCE      Synthetic DNA construct"
    organism = "  ORGANISM  Synthetic DNA construct"
    features = "FEATURES             Location/Qualifiers\n     source          1..{0}\n                     /organism=\"Synthetic DNA construct\"".format(
        len(sequence)
    )

    origin_lines = []
    seq_lower = sequence.lower()
    for start in range(0, len(seq_lower), 60):
        chunk = seq_lower[start : start + 60]
        groups = [chunk[i : i + 10] for i in range(0, len(chunk), 10)]
        origin_lines.append(f"{start + 1:>9} " + " ".join(groups))

    return (
        f"{locus}\n"
        f"{definition}\n"
        f"{accession}\n"
        f"{version}\n"
        f"{source}\n"
        f"{organism}\n"
        f"{features}\n"
        "ORIGIN\n"
        + "\n".join(origin_lines)
        + "\n//\n"
    )


st.set_page_config(page_title="Synthetic Biology Compiler", layout="centered")
st.title("Synthetic Biology Compiler")
st.write("Enter a genetic circuit design prompt to generate reasoning and a plasmid map.")


with st.form("compiler_form"):
    prompt = st.text_area(
        "Genetic circuit design prompt",
        height=140,
        placeholder="Example: Design a lead-inducible biosensor with GFP readout.",
    )
    submit = st.form_submit_button("Compile Circuit")


if submit:
    if not prompt.strip():
        st.warning("Please enter a prompt before submitting.")
    else:
        status = st.status("Running compiler backend...", expanded=False)
        raw_output = run_inference(prompt)
        thought_text, dna_sequence = parse_thought_and_sequence(raw_output)
        dna_sequence, rag_detail = _apply_rag_substitution(thought_text, dna_sequence)
        if rag_detail.get("applied"):
            st.caption(
                "iGEM parts: verified registry sequences substituted where similarity ≥ threshold; "
                "see expanded details."
            )
            with st.expander("iGEM RAG (verification)", expanded=False):
                st.json(rag_detail)
        elif rag_detail.get("error"):
            st.caption(f"iGEM RAG unavailable: {rag_detail['error']}")
        plasmid_figure = generate_interactive_plasmid_plot(dna_sequence)
        status.update(label="Compile complete", state="complete")

        with st.expander("Biological Logic Pipeline", expanded=True):
            placeholder = st.empty()
            for partial_text in stream_tokens(thought_text):
                placeholder.markdown(partial_text)

        st.subheader("Circular Plasmid Map")
        render_bokeh_figure(plasmid_figure, use_container_width=True)

        seq_len = len(dna_sequence)
        gc_count = dna_sequence.count("G") + dna_sequence.count("C")
        gc_content = (gc_count / seq_len * 100.0) if seq_len else 0.0
        tm_c = estimate_tm_celsius(dna_sequence)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sequence Length", f"{seq_len} bp")
        c2.metric("GC Content", f"{gc_content:.2f}%")
        c3.metric("Estimated Tm", f"{tm_c:.1f} °C")

        fasta_text = to_fasta(dna_sequence)
        genbank_text = to_genbank(dna_sequence)
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download FASTA",
            data=fasta_text,
            file_name="compiled_sequence.fasta",
            mime="text/plain",
            use_container_width=True,
        )
        d2.download_button(
            "Download GenBank (.gb)",
            data=genbank_text,
            file_name="compiled_sequence.gb",
            mime="text/plain",
            use_container_width=True,
        )

        st.code(dna_sequence, language="text")
