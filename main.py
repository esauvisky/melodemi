#!/usr/bin/env python3
import os
from loguru import logger
import sys
from tqdm import tqdm
import argparse
import asyncio
from sys import platform
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import music21
from keras.callbacks import ModelCheckpoint
import numpy as np
import pygame

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

from midi2audio import FluidSynth

def play_midi(midi_file):

    #Play MIDI
    sf2_path = '/usr/share/soundfonts/freepats-general-midi.sf2'  # path to sound font file
    FluidSynth(sound_font=sf2_path).play_midi(midi_file)

setup_logging("INFO")


class Main:
    def __init__(self, args):
        self.args = args

    def start(self):
        logger.info("hello world")
        logger.info(self.args)

        # Load the musical data using Music21
        corpus = music21.corpus.getComposer("bach")
        notes = []

        # for piece in corpus:
        parsed = [music21.converter.parse(corpus[i]) for i in range(0,2)]
        for part in parsed:
            elements = part.flat.notes
            for element in tqdm(elements):
                if isinstance(element, music21.note.Note):
                    # note = (element.pitch.midi, element.duration.quarterLength)
                    # notes.append(note)
                    notes.append(element.pitch.midi)

        # Preprocess the data
        X = []
        y = []
        sequence_length = 100
        for i in tqdm(range(len(notes) - sequence_length)):
            X.append(notes[i:i + sequence_length])
            y.append(notes[i + sequence_length])

        # Convert the data to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # load the model if exists saved
        if os.path.exists('best_model.h5'):
            model = keras.models.load_model('best_model.h5')
        else:
            # Define the model
            model = Sequential()
            model.add(LSTM(units=128, input_shape=(sequence_length, 1)))
            model.add(Dense(units=128, activation='softmax'))

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

        # Train the model
        batch_size = 128
        epochs = 500
        model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])

        # Generate music
        start_note = np.zeros((1, sequence_length, 1))
        start_note[0, -1, 0] = music21.note.Note('C4').pitch.midi
        generated_notes = []
        note = start_note
        for i in range(100):
            prediction = model.predict(note)
            next_note = np.argmax(prediction)
            generated_notes.append(next_note)
            note[0, :-1, 0] = note[0, 1:, 0]
            note[0, -1, 0] = music21.note.Note(pitch=music21.pitch.Pitch(next_note)).pitch.midi

        # Convert the generated notes to a Music21 Stream
        generated_stream = music21.stream.Stream()
        for note in generated_notes:
            generated_stream.append(music21.note.Note(pitch=music21.pitch.Pitch(note)))

        # Write the generated stream to a MIDI file
        generated_stream.write('midi', fp='generated_music.mid')

        play_midi('generated_music.mid')


if __name__ == "__main__":
    if platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    parser = argparse.ArgumentParser()

    # # Required single positional argument
    # parser.add_argument("arg",
    #                     help="Required positional argument (a single thing).")

    # # Required multime positional arguments
    # parser.add_argument('items', nargs='+',
    #                     help='Required various positional arguments (a list).')

    # Optional argument flag which defaults to False
    parser.add_argument("-f", "--flag", action="store_true", default=False, help="Activate Flag (false by default)")

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-n", "--name", action="store", dest="name", help="Specifies a name if necessary.")

    # Optional extra verbosity level.
    parser.add_argument("-v", "--verbose", action="store_const", default="INFO", const="DEBUG", help="Increases verbosity. Shows debugging log messages.")

    args = parser.parse_args()
    Main(args).start()
    # asyncio.run(Main(args).start())
