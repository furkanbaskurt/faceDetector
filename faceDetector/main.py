import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import visualizationUtilities as vu
from os.path import exists

modelPath = "model.tflite"

baseOptions = python.BaseOptions(model_asset_path=modelPath)
visionRunningMode = mp.tasks.vision.RunningMode
options = vision.FaceDetectorOptions(base_options=baseOptions, running_mode=visionRunningMode.IMAGE)
faceDetector = vision.FaceDetector.create_from_options(options)

#this program is suitable for the images taken by front cam
while True:
    imagePath = input("Please enter the name of the image: ")

    try:
        if exists(imagePath):
            break
    except:
        continue
    print("\nPlease enter the name of an image existing in the directory!\n")
savePath = input("Please enter the name of file to be saved after detection: ")

mpImage = mp.Image.create_from_file(imagePath)

detectionResult = faceDetector.detect(mpImage)

imageCopy = np.copy(mpImage.numpy_view())
newImage = vu.visualize(imageCopy, detectionResult)
rgbNewImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
windowName = "imageWindow"

#when the image detection completed is showed,
#if the image is too large for the window
#you may not see the face relevant detection
#please close it and check the saved file

cv2.imshow(windowName, rgbNewImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(savePath, rgbNewImage)
if cv2.imwrite(savePath, rgbNewImage) == True:
    print("Saved successfully!")


print("\n\nThanks for using! \nSoftware Developer: Furkan BASKURT ©2024")
