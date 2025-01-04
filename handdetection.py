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

# Function to recognize hand signs based on finger positions
def recognizeSign(fingerState):
    if fingerState == [0, 1, 1, 0, 0]:
        return "Peace Sign ✌️"
    elif fingerState == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"
    elif fingerState == [0, 1, 1, 1, 1]:
        return "Four Fingers Up 🖐️"
    elif fingerState == [0, 0, 0, 0, 0]:
        return "Fist ✊"
    elif fingerState == [1, 1, 1, 1, 1]:
        return "Five Fingers Open 🖐️"
    else:
        return "Unknown Gesture 🤷"

# Open the default camera
capture = cv2.VideoCapture(0)

# Initialize HandDetector class
detector = HandDetector()
fingertips = [4, 8, 12, 16, 20]

while True:
    success, img = capture.read()  # Capture a frame from the camera
    if not success:
        print("Error: Failed to capture frame.")
        break

    # Detect hands and get the landmarks
    img = detector.findHand(img)
    lmlist = detector.findPosition(img)

    if lmlist:
        finger = []
        # Check if the thumb is up
        if lmlist[fingertips[0]][1] < lmlist[fingertips[0] - 1][1]:  # Compare thumb x-coordinates
            finger.append(1)  # Thumb is up
        else:
            finger.append(0)  # Thumb is down

        # Check for other fingers
        for id in range(1, 5):  # Only check for the 4 fingertips
            if lmlist[fingertips[id]][2] < lmlist[fingertips[id] - 2][2]:  # Compare y-coordinates
                finger.append(1)  # Finger is up
            else:
                finger.append(0)  # Finger is down

        # Recognize hand sign
        sign = recognizeSign(finger)

        # Display the recognized sign on the screen
        cv2.putText(img, f'Sign: {sign}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print(f"Finger States: {finger}, Detected Sign: {sign}")

    # Display the captured frame with hand landmarks
    cv2.imshow('SIGN DETECTION', img)

    # Wait for the 'x' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the camera and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()

