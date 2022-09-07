# Based on code by Lindevs Availible: < https://lindevs.com/yolov4-object-detection-using-opencv?fbclid=IwAR26O4FRifzUnnoUvfVPOtj3v-ag2i2t4z_YX8XE3EPyVQn2ciMhtSwvebU >
import cv2

img = cv2.imread('./input/4.png')
vid = cv2.VideoCapture('./input/license_plate.mp4')


with open('obj.names', 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('obj.cfg', 'obj.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

classIds, scores, boxes = model.detect(img, confThreshold=0.5, nmsThreshold=0.4)

for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(255, 0, 255), thickness=2)

    text = '%s: %.2f' % (classes[classId], score)
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 0, 255), thickness=2)


# cv2.imshow('Image', img)
cv2.imwrite('./output/output.png', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()