from pathlib import Path

SPEAKER_ID = None  # speaker to listen to for fish sound
SOUND_THRESH = 200  # sound threshold for catching fish.
OUTPUT_FOLDER = Path(r"/Users/denlyep/Documents/Code/AqualyticsBot/temp")  # where to save outputs (images/audio plots) for debugging
SEC = 1
HALF_SEC = 0.5
SAMPLE_RATE = 44100  # audio sample rate
WAIT_PARAMETER = 1  # parameter passed to exponential distribution to sample wait time in seconds
CHUNK = 1024  # audio chunk size

KEY_LOOKUP = {
    "Interact": "8",
    "Left": "a",
    "Right": "d",
    "Fish": "9",
    "Oversized Bobber": "0",
    "Esc": "Escape",
    "Enter": "Enter"
}
