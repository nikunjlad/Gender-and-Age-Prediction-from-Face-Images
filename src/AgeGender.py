# Import required modules
import time, math, argparse, cv2, sys


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()   # create a copy of the original frame
    frameHeight = frameOpencvDnn.shape[0]   # get the height of the frame
    frameWidth = frameOpencvDnn.shape[1]   # get the width of the frame
    print(frameHeight, frameWidth)

    # creating a blob from the input image. Here, we pass in the frame having original size, scaling ratio of 1.0,
    # output shape of the blob is 300x300, BGR mean values to subtract from the image, BRSwap to swap the B and R
    # color channels is set to True and lastly cropping is set to False
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)   # pass the input image through the FaceNet, blob is of shape (1, 3, 300, 300)
    detections = net.forward()  # apply forward pass
    bboxes = []  # create empty bounding box list
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # get the bounding box confidence level

        # if the confidence is > 0.7 then get the top left and bottom right (x,y) co-ordinates
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)  # top left x co-ordinate
            y1 = int(detections[0, 0, i, 4] * frameHeight)  # top left y co-ordinate
            x2 = int(detections[0, 0, i, 5] * frameWidth)   # bottom right x co-ordinate
            y2 = int(detections[0, 0, i, 6] * frameHeight)  # bottom right y co-ordinate
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

# face detector pretrained model
faceProto = "../models/face_detector/opencv_face_detector.pbtxt"
faceModel = "../models/face_detector/opencv_face_detector_uint8.pb"

# age detector pretrained model
ageProto = "../models/age_detector/age_deploy.prototxt"
ageModel = "../models/age_detector/age_net.caffemodel"

# gender detector pretrained model
genderProto = "../models/gender_detector/gender_deploy.prototxt"
genderModel = "../models/gender_detector/gender_net.caffemodel"

# model mean values to subtract from facenet model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# classes for the age and gender category
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load age, gender and face detector networks
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream
cap = cv2.VideoCapture(args.input if args.input else 0)
padding = 20  # padding the bounding box by 20 pixels on all sides

# loop while any key press
while cv2.waitKey(1) < 0:
    # Read frame
    t = time.time()   # start time for inference
    hasFrame, frame = cap.read()    # get the image frame from the capture object of the camera

    # if no frame exists, then break out of the loop and stop inference
    if not hasFrame:
        cv2.waitKey()
        break

    # pass the input frame along with face net model for getting bounding box information
    frameFace, bboxes = getFaceBox(faceNet, frame)
    break
    if not bboxes:
        print("No face Detected, Checking next frame")
        cv2.imshow("Age Gender Demo", frameFace)

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                   cv2.LINE_AA)
        cv2.imshow("Age Gender Demo", frameFace)
        # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
    print("time : {:.3f}".format(time.time() - t))
