import random
import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

timer = 0
stateResult = False
startGame = False
scores = [0, 0]  
TIMER_DURATION = 3  
currentAIImage = None  
currentRandomNumber = None  

def overlay_transparent(background, overlay, x, y):
    """Helper function to properly overlay transparent PNG images"""
    if overlay.shape[2] != 4:
        return background
    
    overlay_image = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0
    
    h, w = overlay_image.shape[:2]
    section = background[y:y+h, x:x+w]

    masked_section = section * (1 - mask)
    masked_overlay = overlay_image * mask
    
    background[y:y+h, x:x+w] = masked_section + masked_overlay
    
    return background

def load_and_resize_image(path, target_size=None):
    """Load and resize image while preserving transparency"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

while True:
    try:
        imgBG = cv2.imread("BG.png")
        if imgBG is None:
            raise ValueError("Failed to load background image")
        
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            continue
            
        # Flip the camera horizontally to fix mirror effect
        img = cv2.flip(img, 1)
        
        # Resize and crop camera input
        imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
        imgScaled = imgScaled[:, 80:480]
        
        # Convert to RGB for Mediapipe
        imgRGB = cv2.cvtColor(imgScaled, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if startGame:
            if not stateResult:
                timer = TIMER_DURATION - (time.time() - initialTime)
                if timer >= 0:
                    cv2.putText(imgBG, str(int(timer)), (605, 435), 
                              cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)
                
                if timer <= 0:
                    stateResult = True
                    timer = 0
                    playerMove = None

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        fingers = []
                        
                        thumb_tip = hand_landmarks.landmark[4].x
                        thumb_base = hand_landmarks.landmark[2].x
                        fingers.append(thumb_tip < thumb_base)
                        for tip in [8, 12, 16, 20]:
                            tip_y = hand_landmarks.landmark[tip].y
                            pip_y = hand_landmarks.landmark[tip - 2].y
                            fingers.append(tip_y < pip_y)
                        
                        if sum(fingers) == 0: 
                            playerMove = 1
                        elif sum(fingers) >= 4:  
                            playerMove = 2
                        elif sum(fingers) == 2 and fingers[1] and fingers[2]:  
                            playerMove = 3

                        currentRandomNumber = random.randint(1, 3)
                        try:
                            currentAIImage = load_and_resize_image(f"{currentRandomNumber}.png", (200, 200))
                        except Exception as e:
                            print(f"Error loading AI move image: {e}")

                        if playerMove is not None:
                            if playerMove == currentRandomNumber:  
                                pass
                            elif ((playerMove == 1 and currentRandomNumber == 3) or 
                                  (playerMove == 2 and currentRandomNumber == 1) or 
                                  (playerMove == 3 and currentRandomNumber == 2)):
                                scores[1] += 1  
                            else:
                                scores[0] += 1 

        if currentAIImage is not None:
            imgBG = overlay_transparent(imgBG, currentAIImage, 149, 310)

        h, w = imgScaled.shape[:2]
        imgBG[234:234+h, 795:795+w] = imgScaled
        cv2.putText(imgBG, str(scores[0]), (410, 215), 
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
        cv2.putText(imgBG, str(scores[1]), (1112, 215), 
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

        # Show image
        cv2.imshow("Rock Paper Scissors", imgBG)

        key = cv2.waitKey(1)
        if key == ord('s'):
            startGame = True
            initialTime = time.time()
            stateResult = False
            currentAIImage = None  
            currentRandomNumber = None
        elif key == ord('r'):  
            scores = [0, 0]
            startGame = False
            currentAIImage = None  
            currentRandomNumber = None
        elif key == ord('q'):  
            break

    except Exception as e:
        print(f"Error occurred: {e}")
        continue

# Clean up
cap.release()
cv2.destroyAllWindows()
