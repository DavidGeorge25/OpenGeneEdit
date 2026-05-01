import streamlit as st

from inference import parse_thought_and_sequence, run_mock_inference
from visualizer import generate_circular_plasmid_png


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
        with st.spinner("Compiling circuit and generating biological logic..."):
            raw_output = run_mock_inference(prompt)
            thought_text, dna_sequence = parse_thought_and_sequence(raw_output)
            png_path = generate_circular_plasmid_png(dna_sequence)

        with st.expander("Biological Logic Pipeline", expanded=True):
            st.write(thought_text)

        st.subheader("Circular Plasmid Map")
        st.image(png_path, caption="Generated plasmid visualization", use_container_width=True)
        st.code(dna_sequence, language="text")
