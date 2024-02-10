import cv2
import replicate
import base64

import logging

import requests
import pygame
import io
import wave


def base64ToString(b):
    return base64.b64decode(b).decode('utf-8')

def image_to_text(y):
    print(y)
    print("Image captured and processed successfully!")

    # Use Replicate to generate text from the image
    model_url = "j-min/clip-caption-reward:de37751f75135f7ebbe62548e27d6740d5155dfefdf6447db35c9865253d7e06"
    image_url = "https://replicate.delivery/mgxm/d452ef45-ce7e-4f8d-a63f-350d624fc95a/COCO_val2014_000000462565.jpeg"

    output = replicate.run(
        model_url,
        input={
            "image": y,
            "reward": "clips_grammar"
        }
    )
    return output

def expression_to_text(y):
    print(y)
    print("Image captured and processed successfully!")

    # Use Replicate to generate text from the image
    model_url ="phamquiluan/facial-expression-recognition:32e029c7b4cb59c2d82c59c7d47b61a5fced4cb1a03277ea99bca65fbfae0a3b"
    image_url = "https://replicate.delivery/mgxm/d452ef45-ce7e-4f8d-a63f-350d624fc95a/COCO_val2014_000000462565.jpeg"

    output = replicate.run(
        model_url,
        input={
            "input_path": y,
        }
    )
    return output


def text_to_speech(text1, text2):
    input=str(text1)+" "+str(text2)
    output = replicate.run(
    "lucataco/whisperspeech-small:70789b0c0bfa6d81964a43545867f34a8f8175572c429e7c3c2869fb6fa5ff95",
    input={
        "prompt": input,
        "speaker": "",
        "language": "en"
    })
    print(output)
    # play_wav_from_url(output)
    return output

def capture_image(webcam_index=0):
    # Initialize video capture object
    cap = cv2.VideoCapture(webcam_index)

    if not cap.isOpened():
        raise RuntimeError("Error opening webcam.")

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Error capturing image from webcam.")

    # Release the webcam
    cap.release()

    return frame


if __name__ == "__main__":
    # Capture the image
    image = capture_image()

    # Display the image (optional)
    cv2.imshow("Captured Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    a, buffer = cv2.imencode('.jpg', image)
    cv2.imwrite("captured_image.jpg", image) ##
    x=base64.b64encode(buffer).decode('utf-8')
    y = f'data:image/jpg;base64,{x}'
    
    statement1 = image_to_text(y)
    print(statement1)

    expressions = expression_to_text(y)
    statement2="This person is\n"+(expressions)
    print(statement2)
    
    text_to_speech(statement1, statement2)
