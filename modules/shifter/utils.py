import numpy

NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B"
]

INTERVALS = {
    0: "Unison",
    1: "Minor 2nd",
    2: "Major 2nd", 
    3: "Minor 3rd",
    4: "Major 3rd",
    5: "Perfect 4th",
    6: "Tritone",
    7: "Perfect 5th",
    8: "Minor 6th",
    9: "Major 6th",
    10: "Minor 7th",
    11: "Major 7th",
    12: "Octave",
    -1: "Minor 2nd down",
    -2: "Major 2nd down",
    -3: "Minor 3rd down",
    -4: "Major 3rd down",
    -5: "Perfect 4th down",
    -6: "Tritone down",
    -7: "Perfect 5th down",
    -12: "Octave down",
    24: "2 Octaves",
    -24: "2 Octaves down",
}

def hz2note(frequency: float) -> str:
    if frequency <= 0:
        return "N/A"
    
    MIDI_note = 69 + 12 * numpy.log2(frequency / 440.0)
    MIDI_note_rounded = int(round(MIDI_note))

    note_index = MIDI_note_rounded % 12
    octave = (MIDI_note_rounded // 12) - 1

    return f"{NOTE_NAMES[note_index]}{octave}"

def hz2note_cents(frequency: float) -> tuple[str, int]:
    if frequency <= 0:
        return "N/A", 0
    
    MIDI_note_exact = 69 + 12 * numpy.log2(frequency / 440.0)
    MIDI_note_rounded = int(round(MIDI_note_exact))
    cents_off = int(round((MIDI_note_exact - MIDI_note_rounded) * 100))

    note_index = MIDI_note_rounded % 12
    octave = (MIDI_note_rounded // 12) - 1

    return f"{NOTE_NAMES[note_index]}{octave}", cents_off

def note_hz(note: str) -> float:
    note_name = note[:-1].upper()
    octave = int(note[-1])

    note_index = NOTE_NAMES.index(note_name)
    MIDI_note = (octave + 1) * 12 + note_index

    return 440.0 * (2 ** ((MIDI_note - 69) / 12))

def format_hz(frequency: float, include_note: bool = True) -> str:
    if frequency <= 0:
        return "N/A"
    
    if include_note:
        note = hz2note(frequency)
        return f"{frequency:.1f} Hz ({note})"
    else:
        return f"{frequency:.1f} Hz"
    
def f0_range(min: float, max: float) -> str:
    note_min = hz2note(min)
    note_max = hz2note(max)

    return f"{min:.1f} Hz ({note_min}) - {max:.1f} Hz ({note_max})"

def f0_shift(min: float, max: float, key_shift: float) -> str:
    pitch_factor = 2 ** (key_shift / 12)
    shifted_min = min * pitch_factor
    shifted_max = max * pitch_factor
    
    original = f0_range(min, max)
    shifted = f0_range(shifted_min, shifted_max)

    return f"{original} -> {shifted}"

def pitch_shift(key_shift: float) -> str:
    key_number = int(key_shift) if key_shift == int(key_shift) else None
    
    if key_number is not None and key_number in INTERVALS:
        interval = INTERVALS[key_number]

        return f"{key_shift:+.0f} semitones ({interval})"
    
    else:
        pitch_factor = 2 ** (key_shift / 12)

        return f"{key_shift:+.1f} semitones ({pitch_factor:.2f}×)"

def shift_suffix(key_shift: float) -> str:
    if key_shift == int(key_shift):
        return f"_{int(key_shift)}x"
    else:
        return f"_{key_shift}x"
