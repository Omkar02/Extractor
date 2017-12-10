import  cv2
import numpy as np


file='linces plate/4.JPG'
img= cv2.imread(file)
kernel = np.ones((3,3),np.uint8)

#*********************************************************
blur = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('Blur',blur)
cv2.imwrite('extracted/Blur.JPG',blur)
#*********************************************************
grey = cv2.cvtColor( blur, cv2.COLOR_RGB2GRAY )
cv2.imshow('Grey',grey)
cv2.imwrite('extracted/Grey.JPG',grey)
ret, th1 = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold',th1)
cv2.imwrite('extracted/Threshold.JPG',th1)
#*********************************************************
dilation = cv2.dilate(th1, kernel, iterations=1)
cv2.imshow('Dilation',dilation)
cv2.imwrite('extracted/Dilation.JPG',dilation)
#*********************************************************
# closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Closing',closing)
#*********************************************************
im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
cv2.drawContours(img, contours, 3, (0,255,0), 2)
cnt = contours[5]
cot=cv2.drawContours(img, [cnt], 0, (0,255,0), 2)
cv2.imshow('Contour',cot)
cv2.imwrite('extracted/Contour.JPG',cot)



cv2.waitKey(0)
cv2.destroyAllWindows()