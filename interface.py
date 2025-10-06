
import os
import numpy as np
import cv2
import tensorflow as tf
import warnings
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
warnings.filterwarnings("ignore")

# ------------------ CONSTANTS ------------------
MODEL_PATH = "Best_COVID-19_Model.h5"  # your trained  COVID-19 model
DATA_DIR = "data/"                     # folder containing subfolders

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = sorted(os.listdir(DATA_DIR))
print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

# ------------------ IMAGE PREPROCESS ------------------
def preprocess_image(image_path):
    """Read and preprocess image for EfficientNetV2-B0 model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    return img

# ------------------ PREDICTION FUNCTION ------------------
def predict_image(image_path):
    """Predict COVID-19 Chest X-Ray Type and confidence from image path."""
    try:
        img = preprocess_image(image_path)
        preds = model.predict(img)
        class_id = np.argmax(preds[0])
        confidence = preds[0][class_id] * 100
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None

# ------------------ GUI CALLBACKS ------------------
def browse_image():
    """Open file dialog to select an image and display prediction."""
    file_path = filedialog.askopenfilename(
        title="Select X-Ray Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        predicted_class, confidence = predict_image(file_path)
        if predicted_class:
            result_label.config(
                text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
            )

# ------------------ GUI SETUP ------------------
root = Tk()
root.title("COVID-19 Chest X-Ray Image Classification")
root.geometry("400x500")

Label(root, text="COVID-19 Chest X-Ray Image Classification", font=("Arial", 14)).pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

browse_btn = Button(root, text="Select X-Ray Image", command=browse_image)
browse_btn.pack(pady=20)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()


