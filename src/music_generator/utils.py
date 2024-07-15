import os
import logging
import numpy as np
import music21.chord as chord21
import music21.note as note21
from pathlib import Path
from collections import namedtuple
from music21 import converter, instrument, stream

MIDI_OUT_PATH = Path("./data/midi_out")

CustomNote = namedtuple(
    "CustomNote", ("pitch", "velocity", "duration", "delay"))


def parse_midi_files(path: Path) -> list[stream.Stream]:
    """Load MIDI files from a directory tree and return music21 streams."""
    scores = []

    # Iterate over all MIDI files in all subdirectories
    for root, _, files in os.walk(path):
        logging.debug("Loading MIDI files from '%s'...", root)

        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                file_path = Path(root) / file

                try:
                    midi = converter.parse(file_path)
                    scores.append(midi)
                except Exception as exc:
                    logging.warning(
                        "MIDI file '%s' could not be parsed (%s).", file_path, exc)

    logging.info("Loaded %s MIDI files.", len(scores))
    return scores


def extract_notes(
    scores: list[stream.Stream],
    inst_filter: instrument.Instrument = None,
) -> tuple[list[CustomNote], set[CustomNote]]:
    """Extract notes and chords from the provided scores."""
    all_notes: list[CustomNote] = []
    note_set: set[CustomNote] = set()

    for index, score in enumerate(scores):
        logging.debug("Extracting notes from score %s/%s...",
                      index, len(scores))

        # Filter for specific instrument
        parts = instrument.partitionByInstrument(score)
        if parts:
            parts = [part for part in parts if not inst_filter or isinstance(
                part.getInstrument(), inst_filter)]
            notes = parts[0].recurse() if parts else score.recurse()
        else:
            notes = score.recurse()

        # Differentiate between single notes and chords
        for item in notes:
            if isinstance(item, note21.Note):
                pitch = item.pitch.nameWithOctave
            elif isinstance(item, chord21.Chord):
                pitch = ":".join(
                    [pitch.nameWithOctave for pitch in item.pitches])
            else:
                continue

            # Delay is reserved for future temporal modeling; currently kept at zero for parity.
            custom_note = CustomNote(
                pitch=pitch,
                velocity=item.volume.velocity,
                duration=item.duration.quarterLength,
                delay=0,
            )
            all_notes.append(custom_note)
            note_set.add(custom_note)
            # TODO: Consider splitting pitch/velocity/duration into separate channels.

    logging.debug("Extracted %s notes (%s individual notes).",
                  len(all_notes), len(note_set))
    return all_notes, note_set


def create_sequences(
    all_notes: list[CustomNote],
    note_set: set[CustomNote],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, dict[CustomNote, int]]:
    """Convert note streams into integer-encoded input/output sequences."""
    logging.debug("Creating sequences...")
    note_map = {note: index for index, note in enumerate(note_set)}
    input_data = []
    output_data = []

    # Create sequences with inputs and outputs
    for i in range(0, len(all_notes) - seq_len):
        sequence_in = all_notes[i:i + seq_len]
        sequence_out = all_notes[i + seq_len]
        input_data.append([note_map[note] for note in sequence_in])
        output_data.append(note_map[sequence_out])

    # Convert lists to one-hot encoding
    # input_data = [keras.utils.to_categorical(seq, num_classes=len(note_map)) for seq in input_data]
    # output_data = keras.utils.to_categorical(output_data, num_classes=len(note_map))

    input_data, output_data = np.array(input_data), np.array(output_data)
    logging.debug(f"Created input-sequences: {input_data.shape}, output-sequences: {output_data.shape} "
                  f"and value-map: ({len(note_map)}).")
    return input_data, output_data, note_map


def build_midi_file(notes: list[CustomNote], file_name: str) -> None:
    """Write a list of CustomNote entries to a MIDI file."""
    logging.debug("Building MIDI file...")
    note_output = []

    file_path = MIDI_OUT_PATH / file_name
    os.makedirs(file_path.parent, exist_ok=True)

    # Differentiate between single notes and chords
    for item in notes:
        if ":" in item.pitch:
            new_note = chord21.Chord(item.pitch.split(":"))
        else:
            new_note = note21.Note(item.pitch)

        new_note.volume.velocity = item.velocity
        new_note.duration.quarterLength = item.duration
        # TODO: Use delay to insert rests between notes.
        new_note.storedInstrument = instrument.Piano()
        note_output.append(new_note)

    midi_stream = stream.Stream(note_output)
    midi_stream.write("midi", file_path)
    logging.debug("Saved MIDI file with %s notes.", len(notes))
