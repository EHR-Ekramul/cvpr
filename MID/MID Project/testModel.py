import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained face recognition model
model = load_model('model2.keras')

class_names = ['sabbir', 'noman', 'afif', 'ekram']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
padding_ratio = 0.2

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=7, minSize=(50, 50))

    for (x, y, w, h) in faces:
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, frame.shape[1])
        y2 = min(y + h + pad_y, frame.shape[0])
        face_img = frame[y1:y2, x1:x2]
        gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        face_img_resized = cv2.resize(gray_face_img, (256, 256))
        face_img_array = np.expand_dims(face_img_resized / 255.0, axis=(0, -1))

        # Get predictions
        predictions = model.predict(face_img_array)

        predicted_class = np.argmax(predictions)
        predicted_prob = predictions[0][predicted_class]

        if predicted_prob > 0.9:
            class_name = class_names[predicted_class]
            color = (0, 255, 0)  # Green
        else:
            class_name = 'No Prediction'
            color = (0, 0, 255)  # Red

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{class_name} ({predicted_prob*100:.2f}%)', 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Recognition using SANE', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

webcam.release()
cv2.destroyAllWindows()