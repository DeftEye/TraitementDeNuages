import matplotlib as plt
import cv2
import understanding_cloud_organization.train_images

img = ""
image = cv2.imread('./understanding_cloud_organization/train_images/bde641b.jpg')
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
resized = cv2.resize(image, (525, 350), interpolation = cv2.INTER_AREA)

cv2.imshow('image', image)
cv2.imshow('resize', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


