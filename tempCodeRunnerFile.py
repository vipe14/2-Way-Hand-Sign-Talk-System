
import cv2
import numpy as np
import mediapipe as mp
import pickle


def main():

    model = pickle.load(open("model.pickle", "rb"))

   
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
  

    # Initialize media capture
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.35,
                        min_tracking_confidence=0.35) as hands:
        while cap.isOpened():
        
            success, img = cap.read()
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)
            if not success:
                print("Ignoring empty camera frame")
                continue

            
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = hands.process(img)

            curr_landmark_coord = []
            xList = []
            yList = []
            
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img,
                                              landmarks,
                                              mp_hands.HAND_CONNECTIONS)
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        xList.append(x)
                        yList.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        curr_landmark_coord.append(x - min(xList))
                        curr_landmark_coord.append(y - min(yList))

                # Rectangle bounds for the hand
                x1 = int(min(xList) * img.shape[1]) - 10
                y1 = int(min(yList) * img.shape[0]) - 10
                x2 = int(max(xList) * img.shape[1]) - 10
                y2 = int(max(yList) * img.shape[0]) - 10

                prediction = model.predict([np.asarray(curr_landmark_coord)])
                predicted_character = prediction[0]
                accuracy = model.predict_proba([np.asarray(curr_landmark_coord)])

                # Draw rectangle and put alphabet with accuracy
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(img, "%.2f%%" % (np.max(accuracy) * 100), (x1 + 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 200), 2,
                            cv2.LINE_AA)

            
            cv2.imshow("Press ESC to exit", img)

           
            key_press = cv2.waitKey(1)
            if key_press == 27:
                break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()



