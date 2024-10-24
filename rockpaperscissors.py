import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

player_score = 0
computer_score = 0
round_time = 5  
start_time = 0
game_state = "waiting"  

def classify_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].x
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    
    if (index_tip > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        middle_tip > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        ring_tip > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        pinky_tip > landmarks[mp_hands.HandLandmark.PINKY_PIP].y):
        return "Rock"
    
    elif (index_tip < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
          middle_tip < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
          ring_tip < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
          pinky_tip < landmarks[mp_hands.HandLandmark.PINKY_PIP].y):
        return "Paper"
    
    elif (index_tip < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
          middle_tip < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
          ring_tip > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
          pinky_tip > landmarks[mp_hands.HandLandmark.PINKY_PIP].y):
        return "Scissors"
    
    return None

def determine_winner(player_choice, computer_choice):
    if player_choice == computer_choice:
        return "Tie"
    elif (player_choice == "Rock" and computer_choice == "Scissors") or \
         (player_choice == "Paper" and computer_choice == "Rock") or \
         (player_choice == "Scissors" and computer_choice == "Paper"):
        return "Player Wins"
    else:
        return "Computer Wins"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  
    h, w, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    current_time = time.time()
    
    if game_state == "waiting":
        cv2.putText(frame, "Show your hand to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if results.multi_hand_landmarks:
            start_time = current_time
            game_state = "playing"
    
    elif game_state == "playing":
        remaining_time = round_time - (current_time - start_time)
        cv2.putText(frame, f"Time: {max(0, int(remaining_time))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if remaining_time <= 0:
            game_state = "result"
            player_choice = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                player_choice = classify_gesture(hand_landmarks.landmark)
            computer_choice = random.choice(["Rock", "Paper", "Scissors"])
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    elif game_state == "result":
        if player_choice:
            cv2.putText(frame, f"Player: {player_choice}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Player: No gesture detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Computer: {computer_choice}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if player_choice:
            result = determine_winner(player_choice, computer_choice)
            cv2.putText(frame, result, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if result == "Player Wins":
                player_score += 1
            elif result == "Computer Wins":
                computer_score += 1
        else:
            cv2.putText(frame, "Computer Wins", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            computer_score += 1
        
        cv2.putText(frame, f"Player Score: {player_score}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Computer Score: {computer_score}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if current_time - start_time > round_time + 3: 
            game_state = "waiting"
    

    cv2.imshow('Rock-Paper-Scissors Game', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()