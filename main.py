import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)  # video object created

mpHands = mp.solutions.hands  # we have to use this before using the mediapipe model
hands = mpHands.Hands()  # this has some inbuilt parameters that we can use like static mode etc
# which we can see later by pressing ctrl and hovering over hands the default modes are already initialized
mpDraw = mp.solutions.drawing_utils  # module provided in mediapipe for landmarks or drawing

pTime = 0
cTime = 0

while True:
    success, img = cap.read()  # this will give frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to rgb
    results = hands.process(imgRGB)  # provides as the result of the handtracking improved fps
    # results.multi_hand_landmarks checks if there are multiple hands or not
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape  # for getting width and height and c is for channels
                cx, cy = int(lm.x*w), int(lm.y*h)  # centre positions
                if id == 4:  # for detecting which part we want there are 20 points in palm
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # we arent drawing on the rgb img so we give
            # img which is output and for each and every single hand detected and hand connections draw the connections

    cTime = time.time()  # getting current time
    fps = 1/(cTime-pTime)  # getting fps
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)  # placing the fps meter on
    # on the screen with inbuilt parameters
    cv2.imshow("Image", img)  # for showing in the webcam to run it
    cv2.waitKey(1)
    
