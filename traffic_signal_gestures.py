import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handLandmark in result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img):
        lmlist = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handLandmark in result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLandmark.landmark):
                    h, w, channel = img.shape
                    positionX = int(lm.x * w)
                    positionY = int(lm.y * h)
                    lmlist.append([id, positionX, positionY])
        return lmlist

# Function to recognize traffic signals based on hand positions
def recognizeTrafficSignal(fingerState):
    if fingerState == [0, 0, 0, 0, 0]:
        return "STOP 🛑"  # Fist
    elif fingerState == [1, 0, 0, 0, 0]:
        return "GO 🟢"  # Thumbs Up
    elif fingerState == [0, 1, 0, 0, 0]:
        return "TURN RIGHT ↪️"  # Index Finger Up
    elif fingerState == [0, 1, 1, 0, 0]:
        return "SLOW DOWN 🟡"  # Peace Sign (Wave Motion if moving)
    elif fingerState == [1, 1, 1, 1, 1]:
        return "STOP FOR PEDESTRIANS 🚶‍♂️"  # Open Palm
    else:
        return "UNKNOWN SIGNAL 🤷"

# Open the default camera
capture = cv2.VideoCapture(0)

# Initialize HandDetector class
detector = HandDetector()
fingertips = [4, 8, 12, 16, 20]

while True:
    success, img = capture.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    # Detect hands and get the landmarks
    img = detector.findHand(img)
    lmlist = detector.findPosition(img)

    if lmlist:
        finger = []
        # Check thumb (x-axis movement)
        if lmlist[fingertips[0]][1] < lmlist[fingertips[0] - 1][1]:
            finger.append(1)  # Thumb up
        else:
            finger.append(0)

        # Check other fingers (y-axis movement)
        for id in range(1, 5):  # Fingertips 8, 12, 16, 20
            if lmlist[fingertips[id]][2] < lmlist[fingertips[id] - 2][2]:
                finger.append(1)  # Finger is up
            else:
                finger.append(0)  # Finger is down

        # Recognize traffic signal
        traffic_signal = recognizeTrafficSignal(finger)

        # Display the recognized traffic signal
        cv2.putText(img, f'Traffic Signal: {traffic_signal}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print(f"Finger States: {finger}, Traffic Signal: {traffic_signal}")

    # Display the video feed
    cv2.imshow('TRAFFIC SIGNAL DETECTION', img)

    # Exit when 'x' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
