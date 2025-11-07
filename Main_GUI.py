import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import os
import sys
 
# Setup GUI window
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")
 
app = ctk.CTk()
app.title("Plant Leaf Disease Detection")
app.geometry("700x600")
app.configure(fg_color="white")
 
image_ref = None  # Prevent garbage collection
 
# --- Functions ---
def browse_image():
    file_path = filedialog.askopenfilename(title="Select Leaf Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")])
    if file_path:
        entry_image_path.delete(0, ctk.END)
        entry_image_path.insert(0, file_path)
        display_image(file_path)
 
def display_image(path):
    global image_ref
    try:
        image = Image.open(path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo, text="")
        image_label.image = photo
        image_ref = photo
        result_box.delete("0.0", ctk.END)
    except Exception as e:
        image_label.configure(text=f"Error loading image:\n{e}")
 
def analyze_image():
    print("Loading..........")
    file_path = entry_image_path.get()
    result_box.delete("0.0", ctk.END)
    if not os.path.exists(file_path):
        result_box.insert(ctk.END, "❌ No valid image selected.\n")
        return
 
    try:
        # Call plant disease detection using CNN.py with image path and capture output
        result = subprocess.run(
           ["python", "C:/Suresh/Multi_disease_detection 95% code/plant disease detection using CNN.py", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        result = result.stdout.strip()
        result_box.delete("0.0", ctk.END)
        result_box.insert(ctk.END, result)
    except subprocess.CalledProcessError as e:
        result_box.delete("0.0", ctk.END)
        result_box.insert(ctk.END, f"❌ Error during analysis:\n{e.stderr or e}")


# --- GUI Layout ---
label = ctk.CTkLabel(app, text="Select a leaf image to detect disease:")
label.pack(pady=10)
 
entry_image_path = ctk.CTkEntry(app, width=500, placeholder_text="Selected image path...")
entry_image_path.pack(pady=5)
 
button_browse = ctk.CTkButton(app, text="Browse Image", command=browse_image)
button_browse.pack(pady=5)
 
image_label = ctk.CTkLabel(app, text="", width=200, height=200)
image_label.pack(pady=10)
 
button_analyze = ctk.CTkButton(app, text="Analyse Leaf", command=analyze_image)
button_analyze.pack(pady=10)

label_result = ctk.CTkLabel(app, text="Detection Result:")
label_result.pack(pady=5)
 
result_box = ctk.CTkTextbox(app, width=650, height=150, fg_color="black", text_color="white")
result_box.pack(pady=10)
 
app.mainloop()
