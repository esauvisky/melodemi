#!/usr/bin/env python3
import os
import sys

import music21
import numpy as np
from keras.layers import LSTM, Dense, Input, concatenate
from keras.models import Model
from keras.utils import to_categorical
from loguru import logger
from midi2audio import FluidSynth
from tqdm import tqdm

SEQUENCE_LENGTH = 32
BATCH_SIZE = 8
EPOCHS = 100
QL_TO_ORDINAL = {}
ORDINAL_TO_QL = {}

def setup_logging(level="DEBUG", show_module=False):
    """
    Setups better log format for loguru
    """
    logger.remove(0)    # Remove the default logger
    log_level = level
    log_fmt = u"<green>["
    log_fmt += u"{file:10.10}…:{line:<3} | " if show_module else ""
    log_fmt += u"{time:HH:mm:ss.SSS}]</green> <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, level=log_level, format=log_fmt, colorize=True, backtrace=True, diagnose=True)


def get_chord(note):
    duration = music21.duration.Duration(ORDINAL_TO_QL[note[1]])
    pitch_octave_pairs = [(note[i], note[i + 1]) for i in range(2, len(note), 2) if note[i] != 0 and note[i + 1] != 0]
    pitches = [music21.pitch.Pitch(pitch, octave=octave) for pitch, octave in pitch_octave_pairs]

    return music21.chord.Chord(pitches, duration=duration)


def get_note(note):
    duration = music21.duration.Duration(ORDINAL_TO_QL[note[1]])
    pitch, octave = note[2] - 1, note[3]
    pitch = music21.pitch.Pitch(pitch, octave=octave)

    return music21.note.Note(pitch=pitch, duration=duration)


def get_rest(note):
    duration = music21.duration.Duration(ORDINAL_TO_QL[note[1]])

    return music21.note.Rest(duration=duration)


def play_song(song):
    stream = music21.stream.Stream()
    for note in song:
        if note[0] == 1:   # Rest
            stream.append(get_rest(note))
        elif note[0] == 2: # Note
            stream.append(get_note(note))
        elif note[0] == 3: # Chord
            stream.append(get_chord(note))

    stream.write('midi', fp='generated_music.mid')
    sf2_path = '/usr/share/soundfonts/freepats-general-midi.sf2' # path to sound font file
    FluidSynth(sound_font=sf2_path).play_midi("generated_music.mid")
def parse_corpus(corpi):
    notes = []
    for corpus in corpi:
        logger.info(f'Parsing {corpus}')
        parsed = music21.converter.parse(corpus)
        elements = parsed.elements[1].flat.notesAndRests
        for element in elements:
            # duration = element.duration.fullName # too many different values
            duration = element.duration.quarterLengthNoTuplets # less precise
            if duration not in QL_TO_ORDINAL:
                ordinal = len(QL_TO_ORDINAL) + 1
                QL_TO_ORDINAL[duration] = ordinal
                ORDINAL_TO_QL[ordinal] = duration
            else:
                ordinal = QL_TO_ORDINAL[duration]

            if isinstance(element, music21.note.Rest):
                feat = [1, ordinal]

            if isinstance(element, music21.note.Note):
                feat = [2, ordinal, element.pitch.pitchClass + 1, element.pitch.octave]

            if isinstance(element, music21.chord.Chord):
                chord_notes = []
                for pitch in element.pitches:
                    chord_notes = chord_notes + [pitch.pitchClass + 1, pitch.octave]
                feat = [3, ordinal, *chord_notes]

            notes.append(feat)

    # ensures all notes are the same length by padding with zeros
    max_len = max(len(note) for note in notes)
    notes = [note + [0] * (max_len - len(note)) for note in notes]
    return notes
## Load the midi files
# corpi = [c for c in music21.corpus.getComposer("mozart")[2:5]]
# corpi = [os.path.join('./midi', f) for f in os.listdir('./midi') if f.endswith('.mid')]
# corpi = [os.path.join('./midi', f) for f in os.listdir('./midi') if f.endswith('.mid') and f.startswith("mid")][5:10]
# corpi = [os.path.join('./midi/pokemon', f) for f in os.listdir('./midi/pokemon')]
corpi = ["./midi/pokemon/Pokemon_game_-_Pokemon.mid"]
NOTES = parse_corpus(corpi)

## Preprocess the data
NUM_NOTES = (len(NOTES) // BATCH_SIZE) * BATCH_SIZE
NOTES = NOTES[:NUM_NOTES]
NUM_FEATURES = len(NOTES[0])

feature_classes_count = [max(np.max(notes, axis=0)[i] + 1 for notes in [NOTES]) for i in range(NUM_FEATURES)]
features = [np.array([note[i] for note in NOTES]) for i in range(NUM_FEATURES)]
Xs = [
    np.zeros((NUM_NOTES - SEQUENCE_LENGTH, SEQUENCE_LENGTH, feature_classes_count[i]), dtype=np.int32)
    for i in range(NUM_FEATURES)]
ys = [np.zeros((NUM_NOTES - SEQUENCE_LENGTH, feature_classes_count[i]), dtype=np.int32) for i in range(NUM_FEATURES)]

for i in range(NUM_FEATURES):
    for j in range(NUM_NOTES - SEQUENCE_LENGTH):
        Xs[i][j] = to_categorical(features[i][j:j + SEQUENCE_LENGTH], num_classes=feature_classes_count[i])
        ys[i][j] = to_categorical(features[i][j + SEQUENCE_LENGTH], num_classes=feature_classes_count[i])
# Define the model architecture
inputs = [Input(shape=(SEQUENCE_LENGTH, feature_classes_count[i])) for i in range(NUM_FEATURES)]
merged = concatenate(inputs)

lstm1 = LSTM(units=128, activation='tanh', return_sequences=True)(merged)
lstm2 = LSTM(units=128)(lstm1)
outputs = [Dense(feature_classes_count[i], activation='softmax', name=f'out{i}')(lstm2) for i in range(NUM_FEATURES)]

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# Train the model
history = model.fit(Xs, ys, batch_size=BATCH_SIZE, epochs=EPOCHS, use_multiprocessing=True)
def predict_sequence(model, seed_sequence, num_features):
    # Encode the seed sequence
    slen = len(seed_sequence[0])
    seed_sequence = [seed_sequence[i][-slen:] for i in range(num_features)]
    seed_sequence = [
        to_categorical([seed_sequence[i]], num_classes=feature_classes_count[i]) for i in range(num_features)]
    # Predict the next feature values in the sequence
    predicted_features = model.predict(seed_sequence, verbose=0)
    # Decode the predicted feature values
    next_features = [np.argmax(predicted_features[i][0]) for i in range(num_features)]
    return next_features


def generate_song(model, seed_sequence, generated_length, num_features):
    slen = len(seed_sequence[0])
    current_sequence = [list(reversed(seed_sequence[i][:slen])) for i in range(num_features)]
    # song = [seed_sequence[i][:slen] for i in range(num_features)]
    generated = np.asarray([[] for i in range(num_features)], dtype=int)
    for i in range(generated_length - slen):
        next_features = predict_sequence(model, current_sequence, num_features)
        print(f"Predicted feature: {next_features}")
        current_sequence = np.hstack((current_sequence, np.transpose([next_features])))
        current_sequence = np.delete(current_sequence, 0, axis=1)
        generated = np.hstack((generated, np.transpose([next_features])), dtype=int)
    return generated


# Set the seed sequence and the desired length of the generated song
seed_sequence = [features[i][:SEQUENCE_LENGTH] for i in range(NUM_FEATURES)]
GENERATED_LENGTH = 50

# Generate the song
song = generate_song(model, seed_sequence, GENERATED_LENGTH, NUM_FEATURES)
song = np.transpose(song)
play_song(song)
