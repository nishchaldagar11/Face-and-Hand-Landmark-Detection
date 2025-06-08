import cv2
import mediapipe as mp
import time
import csv

# Initialize MediaPipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Fullscreen window
cv2.namedWindow('Face and Hand Detection + Gestures', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Face and Hand Detection + Gestures', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Logging setup
log_data = []
frame_count = 0
prev_time = 0

# Hand gesture classification
def classify_hand_gesture(hand_landmarks, handedness):
    fingers = []

    if handedness == 'Right':
        thumb_is_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    else:
        thumb_is_open = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x

    fingers.append(1 if thumb_is_open else 0)

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        is_open = hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
        fingers.append(1 if is_open else 0)

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    else:
        return "Unknown"

with mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        frame_count += 1
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        frame_log = {'frame': frame_count, 'faces': [], 'hands': []}

        # Face expression detection
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw only contours (not full mesh)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )

                h, w, _ = image.shape

                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                left_mouth = face_landmarks.landmark[78]
                right_mouth = face_landmarks.landmark[308]

                vertical_dist = abs(top_lip.y - bottom_lip.y)
                horizontal_dist = abs(left_mouth.x - right_mouth.x)

                expression = "Smiling" if vertical_dist / horizontal_dist > 0.35 else "Neutral/Sad"

                frame_log['faces'].append({'expression': expression})

                cx = int(face_landmarks.landmark[1].x * w)
                cy = int(face_landmarks.landmark[1].y * h)

                # Add background for better visibility
                (text_w, text_h), _ = cv2.getTextSize(expression, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(image, (cx - 10, cy - 50), (cx + text_w + 10, cy - 10), (0, 0, 0), -1)
                cv2.putText(image, expression, (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Hand detection and gesture
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, hand_info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = hand_info.classification[0].label
                gesture = classify_hand_gesture(hand_landmarks, hand_label)

                landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                frame_log['hands'].append({
                    'landmarks': landmarks,
                    'gesture': gesture,
                    'hand': hand_label
                })

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = image.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)

                (text_w, text_h), _ = cv2.getTextSize(f'{hand_label} - {gesture}', cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(image, (cx - 10, cy - 50), (cx + text_w + 10, cy - 10), (0, 0, 0), -1)
                cv2.putText(image, f'{hand_label} - {gesture}', (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # FPS display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(image, f'FPS: {int(fps)}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        log_data.append(frame_log)
        cv2.imshow('Face and Hand Detection + Gestures', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save logs
with open('detection_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'face_index', 'score', 'xmin', 'ymin', 'width', 'height',
                     'hand_index', 'landmark_index', 'x', 'y', 'z', 'gesture', 'handedness', 'expression'])

    for entry in log_data:
        frame = entry['frame']
        for i, face in enumerate(entry['faces']):
            writer.writerow([
                frame, i, '', '', '', '', '', '', '', '', '', '', '', '', face.get('expression', '')
            ])
        for i, hand in enumerate(entry['hands']):
            for j, lm in enumerate(hand['landmarks']):
                writer.writerow([
                    frame, '', '', '', '', '', '', i, j, lm['x'], lm['y'], lm['z'],
                    hand['gesture'], hand['hand'], ''
                ])
