import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model('Cancer_detection_model.h5')
data_cat = ['Healthy', 'Leukemia - I', 'Leukemia - II', 'Leukemia - III', ' Chronic Lymphocytic Leukemia', 'Follicular Lymphoma', ' Mantle Cell Lymphoma']

# Prediction Function
def predict_image(path):
    image = tf.keras.utils.load_img(path, target_size=(180, 180))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    class_name = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100
    return class_name, confidence

# Upload Function
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    # Display image with border
    img = Image.open(file_path)
    img = ImageOps.contain(img, (250, 250))  # Resize but keep aspect
    img = ImageOps.expand(img, border=5, fill='gray')  # Add border
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Predict
    class_name, confidence = predict_image(file_path)
    result_text.set(f"🔬 Class: {class_name}\n🎯 Accuracy: {confidence:.2f}%")

# Hover Effects
def on_enter(e):
    upload_button.config(bg="#218838")

def on_leave(e):
    upload_button.config(bg="#28a745")

# GUI Setup
root = tk.Tk()
root.title("🧪 Cancer Cell Classifier")
root.geometry("450x600")
root.configure(bg="#f7f7f7")

# Header
header = tk.Label(root, text="Cancer Cell Detection", font=("Helvetica", 22, "bold"), fg="#333", bg="#f7f7f7")
header.pack(pady=20)

# Frame for image
image_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
image_frame.pack(pady=10)

image_label = tk.Label(image_frame, bg="white")
image_label.pack()

# Result Label
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 16), bg="#f7f7f7", fg="#444")
result_label.pack(pady=20)

# Upload Button
upload_button = tk.Button(root, text="📁 Upload Image", command=upload_and_predict, font=("Arial", 14, "bold"),
                          bg="#28a745", fg="white", activebackground="#218838", relief="raised", bd=4)
upload_button.pack(pady=10)
upload_button.bind("<Enter>", on_enter)
upload_button.bind("<Leave>", on_leave)

# Footer
footer = tk.Label(root, text="Model: Cancer_detection_model.h5", font=("Arial", 10), bg="#f7f7f7", fg="gray")
footer.pack(side="bottom", pady=10)

root.mainloop()
