import pretty_midi

def ExtractMelodyToTxt(midi_path, output_path):
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Find the melody track
    melody_track = None
    for inst in midi.instruments:
        if inst.name.upper() == "MELODY":
            melody_track = inst
            break

    if melody_track is None:
        raise ValueError("No 'MELODY' track found in the MIDI file.")

    notes = sorted(melody_track.notes, key=lambda n: n.start)
    lines = []
    current_time = 0.0
    min_rest_duration = 0.3  # adjust as needed
    for note in notes:
        gap = note.start - current_time
        # Add rest if there's a gap
        if gap > min_rest_duration:
            lines.append(f"{current_time:.6f}\t{note.start:.6f}\tN")

        # Add note
        note_name = pretty_midi.note_number_to_name(note.pitch)
        lines.append(f"{note.start:.6f}\t{note.end:.6f}\t{note_name}")
        current_time = max(current_time, note.end)

    # Optional: add a trailing rest to the end of the file
    if current_time < midi.get_end_time():
        lines.append(f"{current_time:.6f}\t{midi.get_end_time():.6f}\tN")

    # Write to file
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"Saved melody_midi.txt to {output_path}")

def GenerateMidiTxt():
    for i in range(1, 910):
        folderPath = f"POP909/{i:03}/"
        midiPath = f"{folderPath}{i:03}.mid"
        outPath=f"{folderPath}melody_midi.txt"
        ExtractMelodyToTxt(midiPath, outPath)