# CodeAlpha Music Generation with AI

A beginner-friendly AI music generation project using Python, TensorFlow/Keras LSTM, Music21, and Streamlit.

## Features
- Reads MIDI files from `midi_dataset/`
- Converts note and chord pitches into numeric sequences
- Trains an LSTM model to predict the next pitch
- Generates new pitch sequences
- Exports generated music as a downloadable MIDI file
- Streamlit user interface

## 1. Requirements
- Python 3.10 or 3.11 recommended
- Internet access for installing packages
- MIDI training files (`.mid` or `.midi`)

## 2. Install
Open a terminal in this project folder.

Windows:
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv` with an installed compatible Python version.

## 3. Add training data
Place MIDI files in `midi_dataset/`. Use music you have permission to use. You can use your own MIDI compositions or legally downloadable public-domain MIDI files. A larger, varied dataset generally helps, but a few short files are only suitable for demonstrating the pipeline.

The folder should look like:
```text
midi_dataset/
    piece1.mid
    piece2.mid
    piece3.midi
```

## 4. Train the model
```bash
python train_model.py
```
The trained model and training information are saved to `models/`.

## 5. Generate a MIDI file from the terminal
```bash
python generate_music.py
```
The output is `generated_music.mid`.

## 6. Run the Streamlit app
```bash
streamlit run app.py
```
Open the local URL shown in your terminal (usually http://localhost:8501).

## 7. GitHub upload
Create a repository named `CodeAlpha_MusicGeneration`, then upload:
- `app.py`
- `train_model.py`
- `generate_music.py`
- `requirements.txt`
- `README.md`
- a small set of MIDI files if their licenses allow redistribution
- `models/music_lstm.keras` if the model file is reasonably sized and allowed by your repository limits

Do not upload `.venv/`, `__pycache__/`, or other temporary files. Add a `.gitignore` file if needed.

## 8. Deployment notes
Streamlit Community Cloud has resource and build limits. TensorFlow may make installation slow or fail depending on the current Python/runtime image. Train locally first. For deployment, commit the trained model and `models/model_info.json` if repository size permits. Include only MIDI data you are allowed to redistribute. If deployment cannot install TensorFlow, use a compatible Python runtime or host the model on a service that supports TensorFlow.

## Limitations
This educational baseline models pitch only. It assigns a fixed duration to every generated note, so it does not learn note lengths, rests, chords, or complete musical form. The quality depends heavily on the MIDI dataset and training time.
