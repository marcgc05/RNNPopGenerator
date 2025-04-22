from music21 import stream, note, meter, tempo, harmony, pitch, key, clef

def chord_root_from_degree_str(key_obj, deg_str):
    """
    Parse something like '3', '#3', or 'b7' into a pitch object
    using the key's scale-degree logic, then transpose for accidentals.
    """
    accidental = 0
    # Check if there's a leading '#' or 'b'
    if deg_str.startswith('#'):
        deg_str = deg_str[1:]
        accidental = 1
    elif deg_str.startswith('b'):
        deg_str = deg_str[1:]
        accidental = -1

    # Convert to integer degree
    degree = int(deg_str)
    # Base pitch from degree
    p = key_obj.pitchFromDegree(degree)
    # Shift if accidental is present
    if accidental != 0:
        p.transpose(accidental, inPlace=True)
    return p  # music21.pitch.Pitch

def respell_pitch_in_key(original_pitch: pitch.Pitch, key_obj: key.Key) -> pitch.Pitch:
    """
    Attempt to re-spell 'original_pitch' so it fits cleanly in 'key_obj' if possible.

    Mirrors the logic from 'normalize_note':
    1) If the note's name is not directly in the key_obj.pitches list, try getEnharmonic().
    2) Then get the scaleDegreeAndAccidentalFromPitch(...) to see if it's truly in key or has accidental.
    3) Return the possibly re-spelled pitch object.
    """

    # Copy the pitch so we don't modify the original in-place
    p = pitch.Pitch(original_pitch.nameWithOctave)

    # 1) If p is not in the key's pitch names, do getEnharmonic
    if p.name not in [pn.name for pn in key_obj.pitches]:
        p_enharm = p.getEnharmonic()
        # if that enharmonic name is in key, use it
        if p_enharm.name in [pn.name for pn in key_obj.pitches]:
            p = p_enharm

    # 2) Now check if the final pitch has an accidental in the key
    deg, acc = key_obj.getScaleDegreeAndAccidentalFromPitch(p)
    # 'deg' is the scale degree (1..7), 'acc' is a music21.accidentals.Accidental or None

    # If 'acc' is None, it's perfectly in scale. If not None, there’s a # or b.
    # We'll keep that, but if you want to transform # into b or vice versa, you could do more logic here.

    return p

def tokens_to_musicxml(tokenfile, output):
    score = stream.Score()
    part = stream.Part()
    
    # If your tokens truly came from Gb major, set that here
    # "G-" is music21's label for Gb
    key_obj = key.Key("G-", "major")
    part.insert(0, key_obj)

    # Add default time signature, tempo, etc.
    part.append(clef.TrebleClef())
    part.append(meter.TimeSignature('4/4'))
    part.append(tempo.MetronomeMark(number=120))
    score.append(part)

    current_midi = None
    duration = None
    current_time_from_start = 0.0

    with open(tokenfile, 'r') as f:
        tokens = f.read().splitlines()

    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            # Rests or time gaps
            part.append(note.Rest(duration))
            current_time_from_start += duration

        elif token.startswith("CHORD_"):
            # Example: "CHORD_#3_min"
            # Structure: CHORD_<degreeString>_<quality>
            parts = token.split('_', 2)
            # parts[0] = 'CHORD'
            # parts[1] = '#3' or 'b7' or '1'
            # parts[2] = 'maj', 'min', 'maj7', etc.
            degree_str = parts[1]      # '#3', 'b7', '1', etc.
            quality = parts[2]        # 'maj', 'min', etc.

            # If you have weird extension like 'maj7/5', handle it here
                
            if quality == "maj6":
                quality = "maj"
            elif quality == "hdim7":
                quality = "m7b5"
            elif quality.endswith(")"):
                quality = quality.split("(")[0]

            # Convert to a pitch
            root_pitch = chord_root_from_degree_str(key_obj, degree_str)
            # e.g. p.name might be 'B' if #3 in G- major is B
            root_name = root_pitch.name  # might produce 'A#', 'Bb', etc.

            # Create chord symbol
            chord_symbol = harmony.ChordSymbol()
            chord_symbol.figure = f"{root_name}{quality}"
            chord_symbol.offset = current_time_from_start
            part.append(chord_symbol)

        elif token.startswith("INITIAL_"):
            # e.g. "INITIAL_5_OCT_4"
            parts = token.split('_')
            # parts = ["INITIAL", "5", "OCT", "4"]
            deg_str = parts[1]   # e.g. "5"
            octave = int(parts[3])
            
            # Use chord_root_from_degree_str logic but also shift by octave
            anchor_pitch = chord_root_from_degree_str(key_obj, deg_str)
            base_octave = anchor_pitch.octave
            octave_diff = octave - base_octave
            if octave_diff != 0:
                anchor_pitch.midi += 12 * octave_diff
            
            current_midi = anchor_pitch.midi

        elif token.startswith("INTERVAL_"):
            # e.g. "INTERVAL_2"
            parts = token.split('_')
            # parts = ["INTERVAL", "2"]
            interval = int(parts[1])
            duration = float(duration)

            if current_midi is None:
                # No anchor set yet, default to middle C
                current_midi = 60

            new_midi = current_midi + interval
            new_pitch = pitch.Pitch()
            new_pitch.midi = new_midi

            new_pitch = respell_pitch_in_key(new_pitch, key_obj)

            n = note.Note()
            n.pitch = new_pitch
            n.quarterLength = duration
            part.append(n)

            # Advance
            current_time_from_start += duration
            current_midi = new_midi

        elif(token.startswith("DUR")):
            duration = float(token.split("_")[1])
        
        elif token == "END":
            break
    
    for thisNote in part.recurse().getElementsByClass('Note'):
        nStep = thisNote.pitch.step # e.g. 'D', 'E', 'F'
        rightAccidental = key_obj.accidentalByStep(nStep)
        thisNote.pitch.accidental = rightAccidental

    part.makeAccidentals(inPlace=True)
    part.makeMeasures(inPlace=True)
    part.makeNotation(inPlace=True)
    score.write('musicxml', output)


#tokens_to_musicxml("POP909/001/tokens.txt", "exampleOut.mxl")