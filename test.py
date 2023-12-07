# from . import config
import pyautogui
import numpy as np
import PIL.ImageOps
from numpy.random import uniform
from time import sleep, time
import soundcard as sc
import soundfile as sf
import matplotlib.pyplot as plt
import cv2
import sys
from ultralytics import YOLO
import os
from pathlib import Path
 

model = YOLO("models\YOLOv8-12_07_23.pt")


def save_img(filename: str, img: np.array):
    """
    Save image to output folder
    """
    if isinstance(img, PIL.Image.Image):
        img.save(Path(r"temp") / filename)
    else:
        cv2.imwrite(Path(r"temp") / str(filename), img)


def move_cursor_to_bait():
    """
    Move mouse cursor to fish bait using screenshot and coordinates
    """
    output = get_fishing_zone_and_bait_coords()
    if output:
        x1, x2 = output.get("x1"), output.get("x2")
        y1, y2 = output.get("y1"), output.get("y2")
        mouse_x = x1 + (x2-x1)/2
        mouse_y = y1 + (y2-y1)/2
        print(f"Found bobber at{x1, y1, x2, y2} with precision:{output.get('prob')}")
        print(f"Moving cursor to bait @ {mouse_x, mouse_y} ...")
        pyautogui.moveTo(mouse_x, mouse_y, uniform(0.2, 0.7), pyautogui.easeOutQuad)

        img = pyautogui.screenshot(region=(x1, y1, (x2-x1)/2, (y2-y1)/2))
        img = np.array(img)
        save_img(f"latest_moved_cursor.png", img[:, :, ::-1])
        return
    
    print("Bobber not found.")
    return


def get_fishing_zone_and_bait_coords():
    """
    Screen shot the fishing zone, process the image and infer the bait by using the part of the red channel of the
    image with the most brightness
    """
    img = pyautogui.screenshot()
    img_np = np.array(img)
    
    results = model.predict(img)
    result = results[0]
    output = {
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "class_id": "",
        "prob": 0
    }
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        if prob > output.get("prob"):
            output = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
                "prob": prob
            }
    if output.get("prob") == 0:
        return None
        
    x1, x2 = output.get("x1"), output.get("x2")
    y1, y2 = output.get("y1"), output.get("y2")
    top_left = (x1, y1)
    bottom_right = (x1 + (x2-x1), y1 + (y2-y1))
    cv2.rectangle(
        img_np,
        top_left, bottom_right, 255, 2
    )
    save_img(f"latest_model_output.png", img_np)
    return output


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
        sleep(uniform(1.5, 1.9))  # wait to move cursor
        move_cursor_to_bait()
        sleep(uniform(1.5, 1.9))  # wait to move cursor
        counter += 1
    print("Finished fishing.")
    import subprocess
    # Hybernate computer
    subprocess.run("shutdown -h")


fish()