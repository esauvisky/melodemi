#!/usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt
import music21
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GRU, LSTM, Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.utils import to_categorical
from loguru import logger
from midi2audio import FluidSynth
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEQUENCE_LENGTH = 64
GENERATED_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 100
QL_TO_ORDINAL = {}
ORDINAL_TO_QL = {}

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", format=u'<green>[{time:HH:mm:ss.SSS}]</green> <level>{level: <8}</level> | <level>{message}</level>', colorize=True, backtrace=True, diagnose=True)

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
        if note[0] == 1:
            stream.append(get_rest(note))
        elif note[0] == 2:
            stream.append(get_note(note))
        elif note[0] == 3:
            stream.append(get_chord(note))
    stream.write('midi', fp='generated_music.mid')
    sf2_path = '/usr/share/soundfonts/freepats-general-midi.sf2'
    FluidSynth(sound_font=sf2_path).play_midi("generated_music.mid")

def transpose_corpus(corpus, transpositions):
    transposed_corpi = []
    for transposition in transpositions:
        transposed_corpus = corpus.transpose(transposition)
        transposed_corpi.append(transposed_corpus)
    return transposed_corpi

def parse_corpus(corpi):
    notes = []
    for corpus in corpi:
        # logger.info(f'Parsing {corpus}')
        elements = corpus.elements[1].flat.notesAndRests
        for element in elements:
            duration = element.duration.quarterLengthNoTuplets
            if duration not in QL_TO_ORDINAL:
                ordinal = len(QL_TO_ORDINAL) + 1
                QL_TO_ORDINAL[duration] = ordinal
                ORDINAL_TO_QL[ordinal] = duration
            else:
                ordinal = QL_TO_ORDINAL[duration]

            if isinstance(element, music21.note.Rest):
                feat = [1, ordinal]
            elif isinstance(element, music21.note.Note):
                feat = [2, ordinal, element.pitch.pitchClass + 1, element.pitch.octave]
            elif isinstance(element, music21.chord.Chord):
                chord_notes = [pitch.pitchClass + 1 for pitch in element.pitches] + [pitch.octave for pitch in element.pitches]
                feat = [3, ordinal, *chord_notes]

            notes.append(feat)

    max_len = max(len(note) for note in notes)
    notes = [note + [0] * (max_len - len(note)) for note in notes]
    return notes

# corpi = [c for c in music21.corpus.getComposer("mozart")[2:5]]
# corpi = [os.path.join('./midi', f) for f in os.listdir('./midi') if f.endswith('.mid')]
corpi = [os.path.join('./midi', f) for f in os.listdir('./midi') if f.endswith('.mid') and f.startswith("mid")][5:10]
# corpi = [os.path.join('./midi/pokemon', f) for f in os.listdir('./midi/pokemon')]
# corpi = ["./midi/pokemon/Pokemon_game_-_Pokemon.mid"]

# Transpose the MIDI files to different keys
transpositions = [-2, -1, 0, 1, 2]  # Transpose to different keys
corpi_transposed = []
for corpus in corpi:
    corpus = music21.converter.parse(corpus)
    for transposed_corpus in transpose_corpus(corpus, transpositions):
        corpi_transposed.append(transposed_corpus)

# Parse the transposed MIDI files
NOTES = parse_corpus(corpi_transposed)

# NOTES = parse_corpus(corpi)

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

gru1 = GRU(units=128, activation='tanh', return_sequences=True)(merged)
dropout1 = Dropout(0.2)(gru1)
gru2 = GRU(units=128)(dropout1)
dropout2 = Dropout(0.2)(gru2)
outputs = [Dense(feature_classes_count[i], activation='softmax', name=f'out{i}')(dropout2) for i in range(NUM_FEATURES)]
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Split the data into training and validation sets
Xs_train, Xs_val, ys_train, ys_val = [], [], [], []
for i in range(NUM_FEATURES):
    X_train, X_val, y_train, y_val = train_test_split(Xs[i], ys[i], test_size=0.2, random_state=42)
    Xs_train.append(X_train)
    Xs_val.append(X_val)
    ys_train.append(y_train)
    ys_val.append(y_val)

# Train the model
history = model.fit(Xs_train, ys_train, batch_size=BATCH_SIZE, epochs=EPOCHS, use_multiprocessing=True,
                    validation_data=(Xs_val, ys_val), callbacks=[early_stopping, model_checkpoint])

# Visualize the training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy
    ax2.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.show()

plot_training_history(history)

def predict_sequence(model, seed_sequence, num_features):
    slen = len(seed_sequence[0])
    seed_sequence = [seed_sequence[i][-slen:] for i in range(num_features)]
    seed_sequence = [
        to_categorical([seed_sequence[i]], num_classes=feature_classes_count[i]) for i in range(num_features)]
    predicted_features = model.predict(seed_sequence, verbose=0)
    next_features = [np.argmax(predicted_features[i][0]) for i in range(num_features)]
    return next_features

def generate_song(model, seed_sequence, num_features):
    slen = len(seed_sequence[0])
    current_sequence = [list(seed_sequence[i][:slen]) for i in range(num_features)]
    generated = np.asarray([[] for i in range(num_features)], dtype=int)
    for i in range(GENERATED_LENGTH):
        next_features = predict_sequence(model, current_sequence, num_features)
        logger.info(f'Predicted {next_features} ({i + 1}/{GENERATED_LENGTH} notes)')
        current_sequence = np.hstack((current_sequence, np.transpose([next_features])))
        current_sequence = np.delete(current_sequence, 0, axis=1)
        generated = np.hstack((generated, np.transpose([next_features])), dtype=int)
    return generated

seed_sequence = [features[i][:SEQUENCE_LENGTH] for i in range(NUM_FEATURES)]

song = generate_song(model, seed_sequence, NUM_FEATURES)
song = np.transpose(song)
play_song(song)

song = generate_song(model, reversed(seed_sequence), NUM_FEATURES)
song = np.transpose(song)
play_song(song)
