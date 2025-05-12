import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Model/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("Model/labels.txt", "r") as f:
    class_names = f.read().splitlines()

# Set up image parameters
offset = 20
imSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imWhite = np.ones((imSize, imSize, 3), np.uint8) * 255
        imCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imSize / h
                wCal = math.ceil(k * w)
                imResize = cv2.resize(imCrop, (wCal, imSize))
                wGap = math.ceil((imSize - wCal) / 2)
                imWhite[:, wGap:wCal+wGap] = imResize
            else:
                k = imSize / w
                hCal = math.ceil(k * h)
                imResize = cv2.resize(imCrop, (imSize, hCal))
                hGap = math.ceil((imSize - hCal) / 2)
                imWhite[hGap:hCal+hGap, :] = imResize

            # Preprocess for model
            input_image = cv2.resize(imWhite, (224, 224))  # Teachable Machine uses 224x224
            input_image = input_image.astype(np.float32) / 255.0  # Normalize
            input_image = np.expand_dims(input_image, axis=0)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.squeeze(output_data)
            index = np.argmax(prediction)
            confidence = prediction[index]

            cv2.putText(img, f"{class_names[index]} ({confidence*100:.2f}%)",
            (img.shape[1] - 300, 40),  # top-right corner with padding
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        except Exception as e:
            print("Error during prediction:", e)

        cv2.imshow("Crop", imCrop)
        cv2.imshow("White", imWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
