import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import converter, instrument, note, chord, stream
import glob
import pickle
import os


class MusicGenerator:
    def __init__(self):
        self.notes = []
        self.model = None
        self.network_input = None
        self.network_output = None
        self.n_vocab = 0
        self.note_to_int = {}
        self.int_to_note = {}

    def load_midi_files(self, midi_path="midi_songs/*.mid"):
        """Load and parse MIDI files"""
        print("Loading MIDI files...")
        notes = []
        successful_files = 0
        failed_files = 0

        for file in glob.glob(midi_path):
            try:
                print(f"Parsing {file}")
                midi = converter.parse(file)
                notes_to_parse = None

                try:
                    parts = instrument.partitionByInstrument(midi)
                    if parts:
                        notes_to_parse = parts.parts[0].recurse()
                    else:
                        notes_to_parse = midi.flat.notes
                except:
                    notes_to_parse = midi.flat.notes

                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))

                successful_files += 1

            except Exception as e:
                print(f"❌ Error parsing {file}: {str(e)}")
                print(f"⚠️  Skipping this file and continuing...")
                failed_files += 1
                continue

        self.notes = notes
        print(f"\n✅ Successfully parsed: {successful_files} files")
        print(f"❌ Failed to parse: {failed_files} files")
        print(f"📝 Total notes extracted: {len(notes)}")

        if len(notes) == 0:
            raise Exception("No notes were extracted! Please check your MIDI files.")

        # Save notes for later use
        with open('notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

        return notes

    def prepare_sequences(self, sequence_length=100):
        """Prepare input and output sequences for training"""
        print("\nPreparing sequences...")

        # Get unique notes
        pitchnames = sorted(set(self.notes))
        self.n_vocab = len(pitchnames)

        # Create mappings
        self.note_to_int = {note: number for number, note in enumerate(pitchnames)}
        self.int_to_note = {number: note for number, note in enumerate(pitchnames)}

        network_input = []
        network_output = []

        # Create input sequences and output
        for i in range(0, len(self.notes) - sequence_length, 1):
            sequence_in = self.notes[i:i + sequence_length]
            sequence_out = self.notes[i + sequence_length]
            network_input.append([self.note_to_int[char] for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])

        n_patterns = len(network_input)

        # Reshape for LSTM
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        network_input = network_input / float(self.n_vocab)

        # One-hot encode output
        network_output = keras.utils.to_categorical(network_output)

        self.network_input = network_input
        self.network_output = network_output

        print(f"✅ Sequences prepared!")
        print(f"   - Vocabulary size: {self.n_vocab}")
        print(f"   - Training patterns: {n_patterns}")
        return network_input, network_output

    def build_model(self, sequence_length=100):
        """Build LSTM model"""
        print("\n🏗️  Building model...")

        model = keras.Sequential([
            keras.layers.LSTM(512, input_shape=(sequence_length, 1),
                              return_sequences=True),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(512, return_sequences=True),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(512),
            keras.layers.Dense(256),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self.n_vocab),
            keras.layers.Activation('softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model

        print("✅ Model built successfully!")
        print(model.summary())
        return model

    def train(self, epochs=50, batch_size=64):
        """Train the model"""
        print(f"\n🎓 Training model for {epochs} epochs...")
        print("This will take a while. Grab a coffee! ☕")

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            'weights-{epoch:02d}-{loss:.4f}.keras',
            monitor='loss',
            save_best_only=True,
            mode='min'
        )

        self.model.fit(
            self.network_input,
            self.network_output,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_callback]
        )

        print("✅ Training completed!")

    def generate_music(self, length=500, temperature=1.0):
        """Generate new music"""
        print(f"\n🎵 Generating {length} notes...")

        # Load notes if not already loaded
        if not self.notes:
            with open('notes', 'rb') as filepath:
                self.notes = pickle.load(filepath)

        # Prepare sequences if not done
        if not self.note_to_int:
            pitchnames = sorted(set(self.notes))
            self.note_to_int = {note: number for number, note in enumerate(pitchnames)}
            self.int_to_note = {number: note for number, note in enumerate(pitchnames)}

        # Pick a random sequence from the input as a starting point
        start = np.random.randint(0, len(self.network_input) - 1)
        pattern = self.network_input[start].flatten().tolist()
        prediction_output = []

        # Generate notes
        for note_index in range(length):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)

            prediction = self.model.predict(prediction_input, verbose=0)

            # Apply temperature
            prediction = np.log(prediction + 1e-10) / temperature
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)

            index = np.random.choice(len(prediction[0]), p=prediction[0])
            result = self.int_to_note[index]
            prediction_output.append(result)

            pattern.append(index / float(self.n_vocab))
            pattern = pattern[1:]

        print(f"✅ Generated {len(prediction_output)} notes")
        return prediction_output

    def create_midi(self, prediction_output, output_file='generated_music.mid'):
        """Convert predicted notes to MIDI file"""
        print(f"\n💾 Creating MIDI file: {output_file}")

        offset = 0
        output_notes = []

        for pattern in prediction_output:
            # Pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # Pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=output_file)

        print(f"✅ MIDI file saved as {output_file}")


# Usage example - QUICK TEST VERSION (5 epochs)
if __name__ == "__main__":
    print("=" * 60)
    print("🎵 AI MUSIC GENERATOR 🎵")
    print("=" * 60)

    generator = MusicGenerator()

    # Step 1: Load MIDI files
    notes = generator.load_midi_files("midi_songs/*.mid")

    # Step 2: Prepare sequences
    network_input, network_output = generator.prepare_sequences(sequence_length=100)

    # Step 3: Build model
    model = generator.build_model(sequence_length=100)

    # Step 4: Train model (REDUCED for quick testing - increase for better results)
    generator.train(epochs=5, batch_size=64)

    # Step 5: Generate music
    generated_notes = generator.generate_music(length=200, temperature=1.0)

    # Step 6: Create MIDI file
    generator.create_midi(generated_notes, 'generated_music.mid')

    print("\n" + "=" * 60)
    print("✅ COMPLETE! Check 'generated_music.mid' file")
    print("=" * 60)