import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, get_actions, mediapipe_detection

from constants import DATA_PATH, FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MIN_LENGTH_FRAMES, MODELS_PATH, MODEL_NAME

def prob_viz(res, actions, input_frame, high_threshold=0.7, low_threshold=0.49):
    output_frame = input_frame.copy()
    max_prob_idx = np.argmax(res)
    max_prob = res[max_prob_idx]

    for idx, prob in enumerate(res):
        color = (0, 255, 0) if prob >= high_threshold else (255, 0, 0)
        if prob > low_threshold:  # Mostrar solo si la probabilidad es mayor al umbral bajo
            action_text = f"{actions[idx]}: {prob*100:.2f}%"
            text_position = (10, 30 + idx * 30)  # Ajusta la posición vertical para cada acción
            cv2.putText(output_frame, action_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return output_frame

def evaluate_model(lstm_model):
    kp_sequence, sentence = [], []
    actions = get_actions(DATA_PATH)
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            image, results = mediapipe_detection(frame, holistic_model)
            kp_sequence.append(extract_keypoints(results))
            
            if len(kp_sequence) > MAX_LENGTH_FRAMES:
                res = lstm_model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]
                image = prob_viz(res, actions, image)
                
                draw_keypoints(image, results)
                cv2.imshow('Traductor UNISIMON', image)
                
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    lstm_model = load_model(model_path)
    evaluate_model(lstm_model)
