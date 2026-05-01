import re
import time
from typing import Tuple


MOCK_SEQUENCE = (
    "TTGACATGATAAGTAAGGAGGTTTAAATGATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCC"
    "CAATTCTTGTTGAATTAGATGGTCATCCGCTTGAGCTACCATTATCAACAAAATACTCCAATTGGC"
    "GATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCCACACAATCTGCCCTTTCGAAAGATCCC"
    "AACGAAAAGCGTGACCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACACATGGCATG"
    "GATGAACTATACAAATAA"
)


def run_mock_inference(prompt: str) -> str:
    """Simulate model inference and return thought + sequence payload."""
    _ = prompt  # mock endpoint ignores prompt for now
    time.sleep(3)
    return (
        "<|channel>thought\n"
        "The circuit uses a constitutive promoter and tuned RBS to establish a stable "
        "transcription-translation baseline before the reporter CDS. The terminator "
        "must be placed downstream to prevent read-through and preserve modular behavior.\n"
        "<channel|>\n"
        f"{MOCK_SEQUENCE}"
    )


def parse_thought_and_sequence(model_output: str) -> Tuple[str, str]:
    """Extract thought text inside tags and trailing DNA sequence."""
    pattern = re.compile(
        r"<\|channel\>thought\s*(.*?)\s*<channel\|>\s*([ACGTNacgtn\s]+)\s*$",
        re.DOTALL,
    )
    match = pattern.search(model_output)
    if match:
        thought = match.group(1).strip()
        sequence = re.sub(r"\s+", "", match.group(2)).upper()
        return thought, sequence

    if "<|channel>thought" in model_output and "<channel|>" in model_output:
        thought_part, seq_part = model_output.split("<channel|>", 1)
        thought = thought_part.replace("<|channel>thought", "", 1).strip()
        sequence = re.sub(r"[^ACGTNacgtn]", "", seq_part).upper()
        if thought and sequence:
            return thought, sequence

    raise ValueError("Could not parse thought and DNA sequence from model output.")
