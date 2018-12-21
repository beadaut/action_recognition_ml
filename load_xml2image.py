"""
Load xml image file from UTKinect Action 3D dataset
"""
import cv2
import xml.etree.ElementTree
e = xml.etree.ElementTree.parse(
    '/media/tjosh/vault/UTKinectAction3D_depth/s01_e01/depthImg190.xml').getroot()


for child in e:
  print(child.tag, child.attrib)

data = dir(e[0][5])
# data = list(e[0][5].text.split(' '))
# data = [int(x) for x in data]
print(data)

# print(len(e.findall('depthImg190')))
# for atype in e.findall('depthImg190'):
#     print(atype.get('data'))

# image = cv2.Load(
#     '/media/tjosh/vault/UTKinectAction3D_depth/s01_e01/depthImg190.xml')
# cv_file = cv2.FileStorage(
#     '/media/tjosh/vault/UTKinectAction3D_depth/s01_e01/depthImg190.xml', cv2.FILE_STORAGE_READ)
# # matrix = dir(cv_file.getNode("depthImg190"))
# matrix = cv_file.getNode("depthImg190")
# print("read matrix\n", matrix)
# # print(matrix)
# cv_file.release()
