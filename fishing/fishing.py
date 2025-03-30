from . import config
import pyautogui
import numpy as np
import PIL.ImageOps
from PIL import ImageGrab
from PIL import Image
from numpy.random import uniform
from time import sleep, time
import soundcard as sc
import soundfile as sf
import matplotlib.pyplot as plt
import cv2
import sys
import os
 
import pyaudio
import numpy as np
import sys

width = 2560
height = 1600
width = 1279
height = 799
w = width // 5.5
h = height // 4.5
x = int((width // 2) - (w // 2))
y = int((height // 2) - (1.2 * h))


def hold_key(keybind, seconds=1.00):
    """
    Holds key down for seconds according to config.KEY_LOOKUP
    """
    key = config.KEY_LOOKUP[keybind]
    print(f"Action: {keybind} -> {key} for {seconds:.4f} seconds.")
    pyautogui.keyDown(key)
    sleep(seconds)
    pyautogui.keyUp(key)


def get_sound(i):
    """
    Get speaker sound (defined in config.SPEAKER_ID) and use a significant sound (volume0 as inference that a fish has
    been caught, according to config.SOUND_THRESH
    """

    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1 if sys.platform == 'darwin' else 2,
                        rate=config.SAMPLE_RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=config.CHUNK)

        # Record audio from the speaker
        data = np.frombuffer(stream.read(int(config.SAMPLE_RATE * config.HALF_SEC)), dtype=np.int16)

        # Close the stream
        stream.stop_stream()
        stream.close()

    finally:
        p.terminate()

    # Infer volume from record
    mean = np.mean(np.abs(data))
    caught_fish = mean > config.SOUND_THRESH
    print(f"{i} catch = {caught_fish}: fish volume = {mean:9.5f}, speaker = 'MAC'")

    return caught_fish


def save_img(filename: str, img: np.array):
    """
    Save image to output folder
    """
    if isinstance(img, PIL.Image.Image):
        img.save(config.OUTPUT_FOLDER / filename)
    else:
        cv2.imwrite(str(config.OUTPUT_FOLDER / filename), img)


def move_cursor_to_bait():
    """
    Move mouse cursor to fish bait using screenshot and coordinates
    """
    width = 1279
    height = 799
    w = width // 5.5
    h = height // 4.5
    x = int((width // 2) - (w // 2))
    y = int((height // 2) - (1.2 * h))
    print(f"box x size={x}")
    print(f"box y size={y}")

    x1, y1 = pyautogui.position()
    # Print the coordinates
    print(f"Mouse Position: X={x1}, Y={y1}")
    img, coords = get_fishing_zone_and_bait_coords()
    print(f"get_fishing_zone_and_bait_coords: X,Y={coords}")

    mouse_x = x + coords[0]
    mouse_y = y + coords[1]
    print(f"Moving cursor to bait @ {mouse_x, mouse_y} ...")
    pyautogui.moveTo(mouse_x, mouse_y, uniform(0.2, 0.7), pyautogui.easeOutQuad)

    img = pyautogui.screenshot(region=(x, y, w, h))
    img = np.array(img)
    save_img(f"status_cursor.png", img[:, :, ::-1])


def get_fishing_zone_and_bait_coords():
    """
    Screen shot the fishing zone, process the image and infer the bait by using the part of the red channel of the
    image with the most brightness
    """
    img = pyautogui.screenshot(region=(x, y, w, h + (y // 7)))
    save_img(f"screenshot.png", img)

    img = np.array(img)
    img_raw = img.copy()

    # Extract the red channel
    red_channel = img[:, :, 0]

    # Invert the red channel to work with least brightness instead of most brightness
    inverted_red_channel = 255 - red_channel

    # Apply blurring to the inverted red channel
    inverted_red_blurred = cv2.blur(inverted_red_channel, (20, 20))

    # Normalize for display
    inverted_red_blurred_for_display = \
        cv2.normalize(inverted_red_blurred, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Find the location with the minimum brightness
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(inverted_red_blurred)

    # Draw circles on the original and blurred images
    cv2.circle(img_raw, min_loc, 5, 255, 2)
    cv2.circle(inverted_red_blurred_for_display, min_loc, 5, 255, 2)

    save_img(f"status.png", img_raw[:, :, ::-1])
    save_img(f"status_blurred.png", inverted_red_blurred_for_display)

    return img, min_loc


def wait():
    """
    Wait for a random amount of time using exponential rng distribution
    """
    wait_time = np.random.exponential(config.WAIT_PARAMETER)
    print(f"Waiting for {wait_time:.3f} seconds ... ")
    sleep(wait_time)


def fish(hours: float = 3.0 / 6):
    """
    Main wrapper function to fish.
    :param hours: number of hours (can be decimal) to run the program for. Defaults to 30 minutes.
    """
    # import pyautogui
    # import time

    # while True:
    #     # Get the current mouse position
    #     x, y = pyautogui.position()

    #     # Print the coordinates
    #     print(f"Mouse Position: X={x}, Y={y}")

    #     # Wait for 1 second
    #     time.sleep(1)

    # setup()
    sleep(3)
    start_time = time()  # remember when we started
    seconds_to_run = hours * 60 * 60
    mins_to_run = seconds_to_run / 60
    print(f"Running for {mins_to_run:,} minutes")
    counter = 0
    not_elapsed_time = True
    while not_elapsed_time:
        elapsed_time = time() - start_time
        not_elapsed_time = elapsed_time < seconds_to_run
        print("\n")
        print("*" * 10)
        print(f"Fish iteration = {counter}, elapsed time = {elapsed_time / 60:.3f} mins (max = {mins_to_run:.3f})")
        hold_key("Fish", uniform(0.9, 1.1))  # throw fish line
        sleep(uniform(1.5, 1.9))  # wait to move cursor
        move_cursor_to_bait()
        for i in range(26):
            hear_fish_sound = get_sound(i)
            if hear_fish_sound:
                pyautogui.click(button='right')
                sleep(0.5)  # always wait at least 0.5 seconds to loot
                print("Fish caught!!!")
                wait()  # wait random amount of time after catching
                break
            sleep(0.2)  # wait between sounds
        counter += 1
    print("Finished fishing.")
