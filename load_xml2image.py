"""
Load xml image file from UTKinect Action 3D dataset
"""
import cv2

cv_file = cv2.FileStorage(
    'd:\\transfer/UTKinectAction3D_depth/s01_e01/depthImg190.xml', cv2.FILE_STORAGE_READ)
matrix = dir(cv_file.getNode("depthImg190").mat())
# matrix = cv_file.getNode("depthImg190")
print("read matrix\n", matrix)
# print(matrix)
cv_file.release()
