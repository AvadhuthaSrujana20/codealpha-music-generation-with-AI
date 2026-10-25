from pathlib import Path
import json
import numpy as np
import music21
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "music_lstm.keras"
INFO_PATH = BASE_DIR / "models" / "model_info.json"


def sample_next_pitch(probabilities, temperature=1.0):
    """Sample a MIDI pitch; temperature controls randomness."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.maximum(probabilities, 1e-9)
    temperature = max(float(temperature), 0.1)
    logits = np.log(probabilities) / temperature
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


def generate_midi(number_of_notes=100, temperature=1.0, output_path=None):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Put MIDI files in midi_dataset/ and run train_model.py first."
        )
    info = json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else {}
    sequence_length = int(info.get("sequence_length", 50))
    model = tf.keras.models.load_model(MODEL_PATH)

    # Start with a random seed sequence in the valid MIDI pitch range.
    seed = np.random.randint(48, 73, size=sequence_length).tolist()
    generated = []

    for _ in range(int(number_of_notes)):
        model_input = np.asarray(seed[-sequence_length:], dtype=np.int32).reshape(1, -1)
        probabilities = model.predict(model_input, verbose=0)[0]
        next_pitch = sample_next_pitch(probabilities, temperature)
        generated.append(next_pitch)
        seed.append(next_pitch)

    melody = music21.stream.Stream()
    melody.append(music21.tempo.MetronomeMark(number=100))
    for pitch_number in generated:
        note = music21.note.Note(pitch_number)
        note.quarterLength = 0.5
        melody.append(note)

    if output_path is None:
        output_path = BASE_DIR / "generated_music.mid"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    melody.write("midi", fp=str(output_path))
    return output_path


if __name__ == "__main__":
    path = generate_midi(number_of_notes=120, temperature=0.9)
    print(f"Generated MIDI saved to: {path}")
