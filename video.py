# Based on code by Lindevs Availible: < https://lindevs.com/yolov4-object-detection-using-opencv?fbclid=IwAR26O4FRifzUnnoUvfVPOtj3v-ag2i2t4z_YX8XE3EPyVQn2ciMhtSwvebU >
import cv2
import time
from PIL import Image

# frame = cv2.imread('./input/4.png')
vid = cv2.VideoCapture('./video/license_plate.mp4')
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output/output.mp4', codec, fps, (width, height))


with open('obj.names', 'r') as f:
    classes = f.read().splitlines()
net = cv2.dnn.readNetFromDarknet('obj.cfg', 'obj.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

frame_num = 0
while True:
    start_time = time.time()
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
        image = Image.fromarray(frame)
    else:
        print('End of Video')
        break
        


    classIds, scores, boxes = model.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(255, 0, 255), thickness=2)

        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 0, 255), thickness=2)
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)
    out.write(frame)


    # cv2.imshow('Image', frame)
    # cv2.imwrite('./output/output.png', frame)
    # cv2.waitKey(0)
# cv2.destroyAllWindows()