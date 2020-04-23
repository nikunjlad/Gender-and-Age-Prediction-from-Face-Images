import time, math, argparse, cv2, sys, torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sample:

    def __init__(self, args):

        self.args = args

        # classes for the age and gender category
        self.ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.ages = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-24)", "(25-32)",
                     "(33-37)", "(38-43)", "(44-47)", "(48-53)", "(54-59)", "(60-100)"]
        self.genders = ["Male", "Female"]

        # loading face detector pretrained model
        faceProto = "../models/face_detector/opencv_face_detector.pbtxt"
        faceModel = "../models/face_detector/opencv_face_detector_uint8.pb"
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)

        # age detector pretrained model
        ageProto = "../models/age_detector/age_deploy.prototxt"
        ageModel = "../models/age_detector/age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)

        # gender detector pretrained model
        genderProto = "../models/gender_detector/gender_deploy.prototxt"
        genderModel = "../models/gender_detector/gender_net.caffemodel"
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        # model mean values to subtract from facenet model
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    @staticmethod
    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()  # create a copy of the original frame
        frameHeight = frameOpencvDnn.shape[0]  # get the height of the frame
        frameWidth = frameOpencvDnn.shape[1]  # get the width of the frame

        # creating a blob from the input image. Here, we pass in the frame having original size, scaling ratio of 1.0,
        # output shape of the blob is 300x300, BGR mean values to subtract from the image, BRSwap to swap the B and R
        # color channels is set to True and lastly cropping is set to False
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)  # pass the input image through the FaceNet, blob is of shape (1, 3, 300, 300)
        detections = net.forward()  # apply forward pass
        bboxes = []  # create empty bounding box list
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # get the bounding box confidence level

            # if the confidence is > 0.7 then get the top left and bottom right (x,y) co-ordinates
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)  # top left x co-ordinate
                y1 = int(detections[0, 0, i, 4] * frameHeight)  # top left y co-ordinate
                x2 = int(detections[0, 0, i, 5] * frameWidth)  # bottom right x co-ordinate
                y2 = int(detections[0, 0, i, 6] * frameHeight)  # bottom right y co-ordinate
                bboxes.append([x1, y1, x2, y2])  # append the co-ordinates list computed above in the bounding box list

                # draw a rectangle around the face and return the original image along with bounding box co-ordinates
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

        return frameOpencvDnn, bboxes

    def caffeInference(self):

        # Open a video file or an image file or a camera stream
        cap = cv2.VideoCapture(self.args.input if self.args.input else 0)
        padding = 20  # padding the bounding box by 20 pixels on all sides

        # loop while any key press
        while cv2.waitKey(1) < 0:
            # Read frame
            t = time.time()  # start time for inference
            hasFrame, frame = cap.read()  # get the image frame from the capture object of the camera

            # if no frame exists, then break out of the loop and stop inference
            if not hasFrame:
                cv2.waitKey()
                break

            # pass the input frame along with face net model for getting bounding box information
            frameFace, bboxes = self.getFaceBox(self.faceNet, frame)

            # if the bounding box list is empty, just display empty frames, with a message of no face detected
            if not bboxes:
                print("No face Detected, Checking next frame")
                cv2.putText(frameFace, "NO FACE DETECTED!", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA)  # render a message on the blank frame with no face
                cv2.imshow("Age Gender Demo", frameFace)  # display empty frames with message
            else:
                # loop over all the bounding box detections if they exist
                for bbox in bboxes:
                    # extend bounding box by 20 pixels on each side, essentially padding by 20 pixels taking into account
                    # the factor that face may go out of frame. max and min functions are there to limit to the box dimensions
                    # to the frame sizes
                    face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                    # reshape the face image to 227x227 using the blob function and also swapping the RB planes
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                    # predict Gender
                    self.genderNet.setInput(
                        blob)  # pass the 227x227 reshaped face blob to the gender net for prediction
                    genderPreds = self.genderNet.forward()  # do the forward pass
                    gender = self.genders[genderPreds[0].argmax()]  # get the gender

                    # print the gender along with its confidence score
                    print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                    # predict Age
                    self.ageNet.setInput(blob)  # pass the 227x227 face blob to the age net.
                    agePreds = self.ageNet.forward()  # do forward pass
                    age = self.ageList[agePreds[0].argmax()]  # get the age range

                    # print the age along with its confidence score
                    print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

                    # get the label to display on the image frame
                    label = "{},{}".format(gender, age)

                    # place the text on the image with squared bounding box
                    cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
                                2,
                                cv2.LINE_AA)

                    if self.args.output != "":
                        filename = "output/predictions/" + str(args.output)
                        cv2.imwrite(filename, frameFace)
                    cv2.imshow("Age Gender Demo", frameFace)

            print("time : {:.3f}".format(time.time() - t))

    def torchInference(self):

        # Open a video file or an image file or a camera stream
        cap = cv2.VideoCapture(self.args.input if self.args.input else 0)
        padding = 30  # padding the bounding box by 20 pixels on all sides

        # loop while any key press
        while cv2.waitKey(1) < 0:
            # Read frame
            t = time.time()  # start time for inference
            hasFrame, frame = cap.read()  # get the image frame from the capture object of the camera

            # if no frame exists, then break out of the loop and stop inference
            if not hasFrame:
                cv2.waitKey()
                break

            # pass the input frame along with face net model for getting bounding box information
            frameFace, bboxes = self.getFaceBox(self.faceNet, frame)

            # if the bounding box list is empty, just display empty frames, with a message of no face detected
            if not bboxes:
                print("No face Detected, Checking next frame")
                cv2.putText(frameFace, "No face detected!", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                            cv2.LINE_AA)  # render a message on the blank frame with no face
                cv2.imshow("Age Gender Demo", frameFace)  # display empty frames with message
            else:
                # loop over all the bounding box detections if they exist
                for bbox in bboxes:
                    # extend bounding box by 20 pixels on each side, essentially padding by 20 pixels taking into
                    # account the factor that face may go out of frame. max and min functions are there to limit to
                    # the box dimensions to the frame sizes
                    face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                    # reshape the face image to 227x227 using the blob function and also swapping the RB planes
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                    cv2.imshow("Face blob", frameFace)
                    break
            print("time : {:.3f}".format(time.time() - t))
        cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Path to output the prediction in case of single image.')

args = parser.parse_args()
s = Sample(args)
s.caffeInference()