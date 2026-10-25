from pathlib import Path
import json
import numpy as np
import music21
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "midi_dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "music_lstm.keras"
INFO_PATH = MODEL_DIR / "model_info.json"

SEQUENCE_LENGTH = 50
VOCAB_SIZE = 128  # MIDI pitch numbers 0-127


def extract_pitch_sequences():
    """Read MIDI files and return one pitch-number sequence per file."""
    midi_files = list(DATA_DIR.rglob("*.mid")) + list(DATA_DIR.rglob("*.midi"))
    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found in {DATA_DIR}. Add .mid or .midi files first."
        )

    all_sequences = []
    errors = []
    for midi_path in midi_files:
        try:
            score = music21.converter.parse(str(midi_path))
            pitches = []
            for element in score.flatten().notes:
                if isinstance(element, music21.note.Note):
                    pitches.append(int(element.pitch.midi))
                elif isinstance(element, music21.chord.Chord):
                    # Keep each chord tone as a pitch token.
                    pitches.extend(int(p.midi) for p in element.pitches)
            if len(pitches) > SEQUENCE_LENGTH:
                all_sequences.append(pitches)
                print(f"Loaded {midi_path.name}: {len(pitches)} pitch tokens")
            else:
                print(f"Skipped {midi_path.name}: not enough notes ({len(pitches)})")
        except Exception as exc:
            errors.append(f"{midi_path.name}: {exc}")

    if not all_sequences:
        detail = "\n".join(errors[:5])
        raise ValueError(
            "No usable MIDI files were found. Try standard .mid files with melody notes.\n" + detail
        )
    return all_sequences, len(midi_files), errors


def build_training_data(pitch_sequences):
    X, y = [], []
    for pitches in pitch_sequences:
        for i in range(len(pitches) - SEQUENCE_LENGTH):
            window = pitches[i:i + SEQUENCE_LENGTH + 1]
            X.append(window[:-1])
            y.append(window[-1])

    if not X:
        raise ValueError("Not enough notes to create training examples.")
    X = np.asarray(X, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    return X, y


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    pitch_sequences, file_count, errors = extract_pitch_sequences()
    X, y = build_training_data(pitch_sequences)
    print(f"\nTraining examples: {len(X)}")
    print("Training LSTM model...")

    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128),
        layers.Dense(128, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = callbacks.EarlyStopping(
        monitor="loss", patience=3, restore_best_weights=True
    )
    model.fit(
        X, y,
        epochs=20,
        batch_size=64,
        validation_split=0.1 if len(X) >= 20 else 0.0,
        callbacks=[early_stop],
        verbose=1,
        shuffle=True
    )
    model.save(MODEL_PATH)

    info = {
        "sequence_length": SEQUENCE_LENGTH,
        "vocab_size": VOCAB_SIZE,
        "midi_files_found": file_count,
        "usable_midi_files": len(pitch_sequences),
        "training_examples": int(len(X)),
        "epochs_requested": 20,
        "skipped_or_problem_files": errors
    }
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Training info saved to: {INFO_PATH}")


if __name__ == "__main__":
    main()
