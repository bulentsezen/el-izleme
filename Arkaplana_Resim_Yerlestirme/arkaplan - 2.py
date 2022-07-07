import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

segmentor = SelfiSegmentation()

img_arkaplan = cv2.imread("arkaplan_resim.png")

while True:

    success, img = cap.read()

    #imgOut = segmentor.removeBG(img, (255, 0, 255), threshold=0.8)
    imgOut = segmentor.removeBG(img, img_arkaplan, threshold=0.8)

    #cv2.imshow("Image", img)
    cv2.imshow("Image-Out", imgOut)
    cv2.waitKey(1)

