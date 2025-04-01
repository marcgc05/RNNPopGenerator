import os
from music21 import key as m21key, pitch as m21pitch
import math

def parse_key(key_file):
    with open(key_file) as f:
        key_line = f.readline().strip()
        _, _, key_string = key_line.split()

        # Split and normalize
        if ":" in key_string:
            tonic, mode = key_string.split(":")
            tonic = tonic.capitalize()  # 'gb' → 'Gb'
            if mode == "maj":
                mode = "major"
            elif mode == "min":
                mode = "minor"
            else:
                print(f"Warning: Unrecognized mode '{mode}', defaulting to 'major'")
                mode = "major"
            key_string = f"{tonic} {mode}"

        print(f"Parsing key: {key_string}")
        print(m21key.Key(tonic, mode).name)
        return m21key.Key(tonic, mode)

def parse_chords(chord_file):
    chords = []
    with open(chord_file) as f:
        for line in f:
            start, end, chord = line.strip().split()
            chords.append((float(start), float(end), chord))
    return chords

def get_chord_at_time(chords, time, prev_chord=None):
    for start, end, chord in chords:
        if start <= time < end:
            if chord != "N":
                return chord
    return prev_chord or "N"

def normalize_note(note_name, pitch_num, key_obj:m21key.Key):
    p = m21pitch.Pitch(note_name)
    octave = p.octave

    # Let music21 respell to match key if needed
    p_in_key = p.getEnharmonic() if p.name not in [n.name for n in key_obj.pitches] else p
    print(p_in_key)

    degree = key_obj.getScaleDegreeAndAccidentalFromPitch(p_in_key)
    print(degree)
    if degree[1] is not None:
        degree = key_obj.getScaleDegreeAndAccidentalFromPitch(p)
        acc = degree[1]
        accidental = ""
        if acc:
            if acc.alter == 1:
                accidental = "#"
            elif acc.alter == -1:
                accidental = "b"
        return f"NOTE_ON_DEGREE_{accidental}{degree[0]}_OCT_{octave}"
    else:
        return f"NOTE_ON_DEGREE_{degree[0]}_OCT_{octave}"


def quantize(x, step=0.125):
    return round(x / step) * step

def generate_token_sequence(melody_file, chord_file, key_file):
    key_obj = parse_key(key_file)
    chords = parse_chords(chord_file)

    tokens = ["START"]
    prev_chord = None
    current_time = 0.0

    with open(melody_file) as f:
        for line in f:
            start, end, label = line.strip().split()
            start = float(start)
            end = float(end)
            duration = quantize(end - start)

            # TIME_SHIFT from last event to this one
            delta = quantize(start - current_time)
            if delta > 0:
                tokens.append(f"TIME_SHIFT_{delta}")

            current_time = end

            # Chord change
            chord = get_chord_at_time(chords, start, prev_chord)
            if chord != prev_chord and chord != "N":
                chord_token = f"CHORD_{chord.replace('#','sharp').replace('b','flat').replace(':','_').replace('/','_')}"
                tokens.append(chord_token)
                prev_chord = chord

            # Melody note or rest
            if label != "N":
                midi_num = m21pitch.Pitch(label).midi
                note_token = normalize_note(label, midi_num, key_obj)
                tokens.append(f"{note_token}_DUR_{duration}")

    tokens.append("END")
    return tokens

#print(generate_token_sequence("POP909/001/melody_midi.txt", "POP909/001/chord_midi.txt", "POP909/001/key_audio.txt"))


