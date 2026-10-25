from pathlib import Path
import subprocess
import sys
import streamlit as st
from generate_music import generate_midi

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "midi_dataset"
MODEL_PATH = BASE_DIR / "models" / "music_lstm.keras"

st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="centered")
st.title("🎵 AI Music Generation with LSTM")
st.write(
    "Train a small LSTM neural network on MIDI files, then generate a new melody "
    "and download it as a MIDI file."
)

st.info(
    "This project generates pitch sequences. The current version uses fixed note "
    "durations; it does not learn full rhythm, harmony, or musical structure."
)

DATA_DIR.mkdir(exist_ok=True)
midi_files = list(DATA_DIR.rglob("*.mid")) + list(DATA_DIR.rglob("*.midi"))
st.subheader("1. Training dataset")
st.write(f"MIDI files found in `midi_dataset/`: **{len(midi_files)}**")
if midi_files:
    with st.expander("View MIDI files"):
        for path in midi_files[:100]:
            st.write(path.relative_to(BASE_DIR).as_posix())
else:
    st.warning("Add MIDI files to the midi_dataset folder before training.")

if st.button("🧠 Train / Retrain Model", type="primary", disabled=not bool(midi_files)):
    with st.spinner("Training model. This may take several minutes..."):
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "train_model.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True
        )
    if result.returncode == 0:
        st.success("Model trained successfully!")
        st.code(result.stdout[-6000:] if result.stdout else "Training completed.")
    else:
        st.error("Training failed. Check the error output below.")
        st.code((result.stderr or result.stdout)[-6000:])

st.divider()
st.subheader("2. Generate new music")
st.write("Train the model first. Then choose how many notes to generate.")
number_of_notes = st.slider("Number of notes", min_value=30, max_value=300, value=120, step=10)
temperature = st.slider(
    "Creativity (temperature)", min_value=0.3, max_value=1.5, value=0.9, step=0.1,
    help="Lower values are more predictable; higher values are more varied."
)

if st.button("🎼 Generate Music", disabled=not MODEL_PATH.exists()):
    try:
        with st.spinner("Generating MIDI..."):
            output_file = BASE_DIR / "generated_music.mid"
            generate_midi(
                number_of_notes=number_of_notes,
                temperature=temperature,
                output_path=output_file
            )
        st.success("Music generated!")
        st.write("Generated file: `generated_music.mid`")
        st.download_button(
            label="⬇️ Download generated MIDI",
            data=output_file.read_bytes(),
            file_name="codealpha_generated_music.mid",
            mime="audio/midi"
        )
        st.caption("To listen, open the downloaded MIDI file in a MIDI player or music software.")
    except Exception as exc:
        st.error(f"Could not generate music: {exc}")

st.divider()
st.subheader("How it works")
st.markdown(
    """
    1. MIDI files are converted into pitch-number sequences.
    2. An LSTM learns to predict the next pitch from the previous 50 pitches.
    3. The model predicts notes one at a time.
    4. The generated sequence is exported as a MIDI file.
    """
)
