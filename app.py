import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Changer le répertoire de travail courant pour le répertoire du script pour utiliser les chemins relatifs
os.chdir(os.path.dirname(__file__))

BLUR_INITIAL = (10, 10)

def process_img(img, face_detection, blur_params):
    # Dimensions de l'image
    H, W, _ = img.shape

    # Conversion en format RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détection des visages
    out = face_detection.process(img_rgb)
                
    # Itération sur chaque visage détecté
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Floutage du visage
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], blur_params)

    return img


class App:
    # CONSTRUCTEUR
    # @param root: Instance de Tkinter.Tk, la racine de l'interface graphique.
    def __init__(self, root):
        self.root = root
        self.root.title("Floutage Visage")
        self.root.iconbitmap('./assets/icone.ico')

        self.file_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.output_name = tk.StringVar()
        self.blur_intensity = tk.DoubleVar(value=3)
         
        # Interface utilisateur
        tk.Label(root, text="Image/Video à flouter:").grid(row=0, column=0)
        tk.Entry(root, textvariable=self.file_path, width=40).grid(row=0, column=1)
        tk.Button(root, text="Parcourir", command=self.browse_file).grid(row=0, column=2)

        tk.Label(root, text="Dossier de sortie:").grid(row=1, column=0)
        tk.Entry(root, textvariable=self.output_dir, width=40).grid(row=1, column=1)
        tk.Button(root, text="Parcourir", command=self.browse_output_dir).grid(row=1, column=2)

        tk.Label(root, text="Nom du fichier de sortie:").grid(row=2, column=0)
        tk.Entry(root, textvariable=self.output_name, width=40).grid(row=2, column=1)

        tk.Label(root, text="Intensité du flou: (Faible - Forte)").grid(row=3, column=0)
        tk.Scale(root, variable=self.blur_intensity, from_=1, to=5, orient=tk.HORIZONTAL, resolution=1).grid(row=3, column=1)

        tk.Button(root, text="Flouter", command=self.process_image).grid(row=4, column=1)
    

    # METHODES DE LA CLASSE App

    # Sélectionner un fichier image/vidéo et mettre à jour le chemin dans l'interface
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image/Video Files", "*.jpg;*.png;*.mp4")])
        self.file_path.set(file_path)

    # Sélectionner un dossier de sortie et mettre à jour le chemin dans l'interface
    def browse_output_dir(self):
        output_dir = filedialog.askdirectory()
        self.output_dir.set(output_dir)

    # Afficher un message de confirmation lorsque le processus est terminé
    def show_completion_message(self):
        messagebox.showinfo("Terminé", "Le processus de floutage est terminé.")


    def process_image(self):
        mode = "image" if self.file_path.get().lower().endswith(('.jpg', '.png')) else "video"
        file_path = self.file_path.get()
        output_dir = self.output_dir.get()
        output_name = self.output_name.get()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            file_name, _ = os.path.splitext(output_name)
            source_extension = os.path.splitext(file_path)[1]
            blur_params = tuple(int(value * self.blur_intensity.get()) for value in BLUR_INITIAL)

            if mode == "image":
                img = cv2.imread(file_path)
                img = process_img(img, face_detection, blur_params)
                cv2.imwrite(os.path.join(output_dir, f'{file_name}{source_extension}'), img)

            elif mode == 'video':
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                output_video = cv2.VideoWriter(os.path.join(output_dir, f'{file_name}{source_extension}'),
                                            cv2.VideoWriter_fourcc(*'MP4V'),
                                            25,
                                            (frame.shape[1], frame.shape[0]))
                while ret:
                    frame = process_img(frame, face_detection, blur_params)
                    output_video.write(frame)
                    ret, frame = cap.read()
                cap.release()
                output_video.release()

        self.show_completion_message()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
