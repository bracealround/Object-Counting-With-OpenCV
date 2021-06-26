#imports opencv librery
import cv2
import numpy as np
import matplotlib.pyplot as plt

#imports the image
image=cv2.imread("fruits.jpg")
cv2.imshow('image',image)

#convert the image into a grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', gray)

#Here the grayscale image is blurred to reduce the noise of the image.
blur = cv2.GaussianBlur(gray , (11,11) , 0) #(image,size of standared window , standard deviation )
cv2.imshow('Blur image', blur)

#We used canny edge detection to detect wide range of edges.
#Here I tried different higher and lower values to get the best possible result for this image.
canny=cv2.Canny(blur , 20 , 155 ,3) #(image,lower value,higher value ,kernel size)
cv2.imshow('Canny image', canny)

#dialation is used for two cases for two cases: Increasing area and accentuate features.
dialated = cv2.dilate(canny , (1,1) , iterations = 2) #(image,kernel,iteration)
cv2.imshow('Dialated image', dialated)

#used to draw edges in the picture
(cnt,heirarachy) = cv2.findContours(dialated.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb , cnt , -1 , (0,255,0) , 2)

cv2.imshow('rgb' , rgb)

print(len(cnt))


cv2.waitKey(0)
cv2.destroyAllWindows()