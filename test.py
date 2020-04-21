from imutils.video import VideoStream
import imutils
import cv2

vs = VideoStream(src=0).start()

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    print(frame.shape)
    # detect faces in the frame, and for each face in the frame,
    # predict the age
    # results = detect_and_predict_age(frame, faceNet, ageNet,
    #                                  minConf=args["confidence"])
    # # loop over the results
    # for r in results:
    #     # draw the bounding box of the face along with the associated
    #     # predicted age
    #     text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
    #     (startX, startY, endX, endY) = r["loc"]
    #     y = startY - 10 if startY - 10 > 10 else startY + 10
    #     cv2.rectangle(frame, (startX, startY), (endX, endY),
    #                   (0, 0, 255), 2)
    #     cv2.putText(frame, text, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
