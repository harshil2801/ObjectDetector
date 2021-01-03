import cv2

thres = 0.45

# video = cv2.VideoCapture("Pexels Videos 2103099.mp4")
# video.set(3, 100)
# video.set(4,100)
img = cv2.imread("five.jpg")
img = cv2.resize(img, (1020, 420), fx = 0.1, fy = 0.1)
img1 = cv2.imread("airplane.jfif")
img1 = cv2.resize(img1, (1020, 420), fx = 0.1, fy = 0.1)
img2 = cv2.imread("dog.jfif")
img2 = cv2.resize(img2, (1020, 420), fx = 0.1, fy = 0.1)
img3 = cv2.imread("cat.jfif")
img3 = cv2.resize(img3, (1020, 420), fx = 0.1, fy = 0.1)
img4 = cv2.imread("car1.jfif")
img4 = cv2.resize(img4, (1020, 420), fx = 0.1, fy = 0.1)
img5 = cv2.imread("motorcycle.jfif")
img5 = cv2.resize(img5, (1020, 420), fx = 0.1, fy = 0.1)

className = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# while video.isOpened():
#     success, img = video.read()
# classIds, confidence, bbox = net.detect(img, confThreshold= thres)
# for classId, confidence,box in zip(classIds.flatten(), confidence.flatten(), bbox):
#     cv2.rectangle(img,box,color=(0,255,0),thickness=2)
#     cv2.putText(img,className[classId-1].upper(),(box[0]+10,box[1]+30),
#                                 cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
#     cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
#                                 cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
#
# cv2.imshow("Image", img)
# cv2.waitKey(0)

images = [img, img1, img2, img3, img4, img5]
for i in range(len(images)):
    classIds, confidence, bbox = net.detect(images[i], confThreshold=thres)
    for classId, confidence, box in zip(classIds.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(images[i], box, color=(0, 255, 0), thickness=2)
        cv2.putText(images[i], className[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(images[i], str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Images", images[i])
    cv2.waitKey(0)
