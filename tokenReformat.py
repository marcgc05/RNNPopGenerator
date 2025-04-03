from music21 import stream, note, chord, meter, tempo, metadata, harmony

def tokens_to_musicxml(tokenfile, output):
    score = stream.Score()
    part = stream.Part()
    
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

            n = note.Note()
            n.octave = octave
            n.quarterLength = duration
            n.pitch.midi = 60 + (degree - 1)

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




    