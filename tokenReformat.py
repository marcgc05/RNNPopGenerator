from music21 import stream, note, chord, meter, tempo, metadata, harmony, pitch, key

def note_from_degree_and_octave(key_obj, degree, target_octave):
    """
    1) Get pitch from the scale degree (in the key).
    2) Shift pitch to the user-specified octave.
    3) Return the resulting music21.pitch.Pitch object.
    """
    # Step 1: Base pitch from scale degree
    p = key_obj.pitchFromDegree(degree)
    base_octave = p.octave  # e.g. if p was E4, base_octave = 4

    # Step 2: Shift pitch up/down if our desired target_octave differs
    octave_diff = target_octave - base_octave
    if octave_diff != 0:
        p.midi += 12 * octave_diff  # 12 semitones per octave

    return p

def tokens_to_musicxml(tokenfile, output):
    score = stream.Score()
    part = stream.Part()
    key_obj = key.Key("G-", "major")
    score.keySignature = key_obj
    score.append(part)

    #not sure how to identity these, so these are defaultish
    part.append(meter.TimeSignature('4/4'))
    part.append(tempo.MetronomeMark(number = 120))

    #create list of tokens to parse (the rest should be easy)
    with open(tokenfile, 'r') as file:
        tokens = file.read().splitlines()

    current_time_from_start = 0.0

    degree_to_note = {1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'A', 7: 'B'}
    
    for token in tokens:
        
        if (token.startswith("TIME_SHIFT_")):
            duration = (float(token.split("_")[2]))
            current_time_from_start += duration
        
        elif (token.startswith("NOTE_ON_DEGREE_")):

            duration = float(token.split("_")[7])
            degree = int(token.split("_")[3])
            octave = int(token.split("_")[5])

            pitch_obj = note_from_degree_and_octave(key_obj, degree, octave)
            n = note.Note()
            n.pitch = pitch_obj
            n.quarterLength = duration

            part.append(n)
            current_time_from_start += duration

        elif (token.startswith("CHORD_")):
            
            degree = (int(token.split('_')[1]))
            root_note = degree_to_note.get(degree) #ChordSymbol expects a root node

            quality = token.split('_')[2]

            if quality == 'maj7/5':
                quality = 'maj7'

            # Create a chord symbol
            chord_symbol = harmony.ChordSymbol()
            chord_symbol.figure = f"{root_note}{quality}"
            chord_symbol.offset = current_time_from_start

            # Add chord symbol to the part
            part.append(chord_symbol)
            
    score.write('musicxml', output)


tokens_to_musicxml("POP909/001/tokens.txt", "exampleOut.mxl")