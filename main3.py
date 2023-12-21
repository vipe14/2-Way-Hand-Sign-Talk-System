
import cv2
import numpy as np
import mediapipe as mp
import pickle

def main():
    model = pickle.load(open("model.pickle", "rb"))
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    detected_signs = []

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.35,
                        min_tracking_confidence=0.35) as hands:
        predicted_character = ""
        sentence = ""

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

            predicted_changed = False

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

                x1 = int(min(xList) * img.shape[1]) - 10
                y1 = int(min(yList) * img.shape[0]) - 10
                x2 = int(max(xList) * img.shape[1]) - 10
                y2 = int(max(yList) * img.shape[0]) - 10

                prediction = model.predict([np.asarray(curr_landmark_coord)])
                current_predicted_character = prediction[0]

                if current_predicted_character != predicted_character:
                    predicted_character = current_predicted_character
                    predicted_changed = True

                    accuracy = model.predict_proba([np.asarray(curr_landmark_coord)])

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, predicted_character, (x1, y1 - 10), font, font_scale, (255, 0, 255), font_thickness,
                                cv2.LINE_AA)
                    cv2.putText(img, "%.2f%%" % (np.max(accuracy) * 100), (x1 + 30, y1 - 10), font, font_scale,
                                (255, 0, 200), font_thickness, cv2.LINE_AA)

                    sentence += predicted_character + " "
                    detected_signs.append(predicted_character)

                cv2.putText(img, f"Detected Sign: {predicted_character}", (10, 30), font, font_scale,
                            (0, 255, 0), font_thickness, cv2.LINE_AA)

            cv2.imshow("Press ESC to exit", img)

            if predicted_changed:
                print("Current Sentence:", sentence)

            key_press = cv2.waitKey(1)
            if key_press == 27:
                break

    # Display all detected signs
    print("Detected Signs:", " ".join(detected_signs))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
