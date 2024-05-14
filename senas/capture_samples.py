import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH

# Mejoras visuales y funcionales
def center_window(root, width=400, height=150):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (width / 2))
    y_coordinate = int((screen_height / 2) - (height / 2))
    root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")

def apply_gradient_background(image, top_left, bottom_right, color_1, color_2, transparency=0.8):
    overlay = image.copy()
    for i in range(top_left[1], bottom_right[1]):
        alpha = (i - top_left[1]) / (bottom_right[1] - top_left[1])
        color = tuple([alpha * color_1[j] + (1 - alpha) * color_2[j] for j in range(3)])
        cv2.line(overlay, (top_left[0], i), (bottom_right[0], i), color, 1)
    cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)
    return image

def display_status_text(image, text, position, font, font_scale, font_thickness, text_color, bg_color_start, bg_color_end):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    top_left = (position[0], position[1] + text_height + 10)
    bottom_right = (top_left[0] + text_width, top_left[1] - text_height - 10)
    image = apply_gradient_background(image, bottom_right, top_left, bg_color_start, bg_color_end)
    cv2.putText(image, text, (position[0], position[1] + text_height), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def capture_samples(path):
    create_folder(path)
    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    count_frame = 0
    frames = []

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            _, frame = video.read()
            image, results = mediapipe_detection(frame, holistic_model)

            if there_hand(results):
                count_frame += 1
                if count_frame > 2:  # Suponiendo un margen de 2 para el inicio de la captura
                    display_status_text(image, f'Capturando... (Carpeta {path})', (10, 50), FONT, 0.75, 2, (255, 255, 255), (50, 123, 183), (183, 50, 50))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) > 7:  # Suponiendo mínimo 5 frames más el margen
                    frames = frames[:-2]  # Descartar 2 frames finales
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                    count_sample += 1

                frames = []
                count_frame = 0
                display_status_text(image, 'Listo para capturar...', (10, 50), FONT, FONT_SIZE, 2, (255, 255, 255), (50, 123, 183), (183, 50, 50))

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

# Interfaz para ingresar datos
def main_app():
    root = tk.Tk()
    root.title("Ingreso de Datos")
    center_window(root, 400, 150)  # Ajustar según sea necesario

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', background='#ececec', font=('Arial', 12))
    style.configure('TEntry', font=('Arial', 12))
    style.configure('TButton', font=('Arial', 12), background='#4CAF50', foreground='white')

    label = ttk.Label(root, text="Ingrese el nombre de la palabra u oración:")
    label.pack(pady=20)

    entry = ttk.Entry(root, width=50)
    entry.pack(pady=10)

    def on_submit():
        word_name = entry.get()
        if word_name:
            root.destroy()
            word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
            capture_samples(word_path)

    submit_button = ttk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main_app()
 