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

SEQUENCE_LENGTH = 16
BATCH_SIZE = 16
EPOCHS = 300
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
        # sf2_path = '/usr/share/soundfonts/freepats-general-midi.sf2' # path to sound font file
        # FluidSynth(sound_font=sf2_path).play_midi(corpus)
        parsed = music21.converter.parse(corpus)
        elements = parsed.flat.notesAndRests
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
# corpi = [os.path.join('./midi', f) for f in os.listdir('./midi') if f.endswith('.mid') and f.startswith("mid")][:20]
# corpi = [os.path.join('./midi/pokemon', f) for f in os.listdir('./midi/pokemon')]
corpi = ["./midi/pokemon/Pokemon_game_-_Pokemon.mid"]
notes = parse_corpus(corpi)

## Preprocess the data
sequence_length = SEQUENCE_LENGTH
n_samples = (len(notes) // BATCH_SIZE) * BATCH_SIZE
notes = notes[:n_samples]
num_features = len(notes[0])

feature_classes_count = [max(np.max(notes, axis=0)[i] + 1 for notes in [notes]) for i in range(num_features)]
features = [np.array([note[i] for note in notes]) for i in range(num_features)]
Xs = [
    np.zeros((len(notes) - sequence_length, sequence_length, feature_classes_count[i]), dtype=np.int32)
    for i in range(num_features)]
ys = [np.zeros((len(notes) - sequence_length, feature_classes_count[i]), dtype=np.int32) for i in range(num_features)]

for i in range(num_features):
    for j in range(len(notes) - sequence_length):
        Xs[i][j] = to_categorical(features[i][j:j + sequence_length], num_classes=feature_classes_count[i])
        ys[i][j] = to_categorical(features[i][j + sequence_length], num_classes=feature_classes_count[i])

# Define the model architecture
inputs = [Input(shape=(sequence_length, feature_classes_count[i])) for i in range(num_features)]
merged = concatenate(inputs)

lstm1 = LSTM(units=64, activation='tanh', return_sequences=True)(merged)
lstm2 = LSTM(units=64)(lstm1)
outputs = [Dense(feature_classes_count[i], activation='softmax', name=f'out{i}')(lstm2) for i in range(num_features)]

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# Train the model
history = model.fit(Xs, ys, batch_size=BATCH_SIZE, epochs=EPOCHS, use_multiprocessing=True)


def predict_sequence(model, seed_sequence, sequence_length, num_features):
    # Encode the seed sequence
    seed_sequence = [seed_sequence[i][-sequence_length:] for i in range(num_features)]
    seed_sequence = [to_categorical([seed_sequence[i]], num_classes=feature_classes_count[i]) for i in range(num_features)]
    # Predict the next feature values in the sequence
    predicted_features = model.predict(seed_sequence, verbose=0)
    # Decode the predicted feature values
    next_features = [np.argmax(predicted_features[i][0]) for i in range(num_features)]

    return next_features


def generate_song(model, seed_sequence, sequence_length, song_length, num_features):
    # song = [list(reversed(seed_sequence[i][:sequence_length])) for i in range(num_features)]
    song = [seed_sequence[i][:sequence_length] for i in range(num_features)]
    for i in range(song_length - sequence_length):
        next_features = predict_sequence(model, song, sequence_length, num_features)
        print(f"Predicted feature: {next_features}")
        song = [np.append(song[i], next_features[i]) for i in range(num_features)]
    return song


# Set the seed sequence and the desired length of the generated song
seed_sequence = [features[i][:SEQUENCE_LENGTH] for i in range(num_features)]
song_length = 128

# Generate the song
song = generate_song(model, seed_sequence, sequence_length, song_length, num_features)
song = np.transpose(song)
play_song(song)
