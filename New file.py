import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import cv2
import os
import datetime

# --- Setup GUI window ---
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.title("Mango Leaf Disease Detection")
app.geometry("1000x800")
app.configure(fg_color="white")

camera_running = False
image_ref = None
cap = None

# --- Pesticide Recommendation Dictionary ---
PESTICIDE_RECOMMENDATIONS = {
    "Cutting Weevil": "Carbaryl 50 WP (0.2%) or Chlorpyrifos 20 EC (2.5 ml/L). "
                      "Prune affected shoots and apply Neem oil (3%) as preventive.",
    "Gall Midge": "Dimethoate 30 EC (2 ml/L) or Malathion 50 EC (2 ml/L). "
                  "Remove infested panicles and maintain orchard hygiene.",
    "Anthracnose": "Copper oxychloride 0.3% or Carbendazim 0.1%. Repeat every 10–15 days.",
    "Powdery Mildew": "Wettable sulphur 0.2% or Hexaconazole 0.1%. Spray at early stage.",
    "Bacterial Canker": "Streptomycin sulphate (0.01%) + Copper oxychloride (0.3%). Spray fortnightly."
}


# --- Helper Functions ---
def get_pesticide_recommendation(disease_name):
    for key in PESTICIDE_RECOMMENDATIONS.keys():
        if key.lower() in disease_name.lower():
            return PESTICIDE_RECOMMENDATIONS[key]
    return "No specific pesticide recommendation available for this disease."


def log_detection(disease, pesticide):
    """Log results to detection_log.txt"""
    with open("detection_log.txt", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] Disease: {disease} | Pesticide: {pesticide}\n")


def display_selected_image(path):
    """Display selected or captured image on right side"""
    global image_ref
    try:
        image = Image.open(path)
        image.thumbnail((350, 300))
        photo = ImageTk.PhotoImage(image)
        selected_label.configure(image=photo, text="")
        selected_label.image = photo
        image_ref = photo
    except Exception as e:
        selected_label.configure(text=f"Error: {e}")


def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select Leaf Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")]
    )
    if file_path:
        entry_image_path.delete(0, ctk.END)
        entry_image_path.insert(0, file_path)
        display_selected_image(file_path)


def analyze_image(file_path):
    """Analyze a given image and show result"""
    result_box.delete("0.0", ctk.END)
    if not os.path.exists(file_path):
        result_box.insert(ctk.END, "❌ No valid image found.\n")
        return

    result_box.insert(ctk.END, "🟡 Analyzing image, please wait...\n")
    app.update_idletasks()

    try:
        # NOTE: Replace 'predict_disease.py' with your actual inference script
        script_path = os.path.join(os.path.dirname(__file__), "predict_disease.py")
        result = subprocess.run(
            ["python", script_path, file_path],
            capture_output=True,
            text=True,
            check=True
        )
        disease_name = result.stdout.strip()
        pesticide_info = get_pesticide_recommendation(disease_name)

        result_box.delete("0.0", ctk.END)
        result_box.insert(ctk.END, f"🩺 Disease Detected: {disease_name}\n\n")
        result_box.insert(ctk.END, f"💊 Pesticide Suggestion:\n{pesticide_info}\n")

        log_detection(disease_name, pesticide_info)

    except subprocess.CalledProcessError as e:
        result_box.delete("0.0", ctk.END)
        result_box.insert(ctk.END, f"❌ Error during analysis:\n{e.stderr or e}")


# --- Camera Functions ---
def start_camera():
    global camera_running, cap
    if camera_running:
        return

    camera_running = True
    result_box.delete("0.0", ctk.END)
    result_box.insert(ctk.END, "📷 Starting camera preview...\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_box.insert(ctk.END, "❌ Could not open camera.\n")
        camera_running = False
        return
    update_camera_frame()


def update_camera_frame():
    """Continuously update camera feed inside GUI"""
    global image_ref
    if not camera_running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame)
    frame_image = frame_image.resize((350, 300))
    photo = ImageTk.PhotoImage(frame_image)

    camera_label.configure(image=photo)
    camera_label.image = photo
    image_ref = photo

    app.after(10, update_camera_frame)


def capture_and_analyze():
    """Capture image from camera, display it and analyze"""
    global camera_running
    if not camera_running:
        result_box.insert(ctk.END, "⚠️ Start the camera first!\n")
        return

    ret, frame = cap.read()
    if not ret:
        result_box.insert(ctk.END, "❌ Failed to capture image.\n")
        return

    temp_path = "captured_leaf.jpg"
    cv2.imwrite(temp_path, frame)

    display_selected_image(temp_path)
    result_box.insert(ctk.END, "📸 Image captured! Analyzing...\n")

    analyze_image(temp_path)


def stop_camera():
    global camera_running
    if camera_running:
        camera_running = False
        cap.release()
        camera_label.configure(image=None, text="Camera stopped.")
        result_box.insert(ctk.END, "✅ Camera stopped.\n")


# --- GUI Layout ---
title_label = ctk.CTkLabel(app, text="🍃 Mango Leaf Disease Detection System", font=("Arial", 22, "bold"))
title_label.pack(pady=10)

# --- Top Frame for Buttons ---
frame_top = ctk.CTkFrame(app, fg_color="white")
frame_top.pack(pady=10)

entry_image_path = ctk.CTkEntry(frame_top, width=450, placeholder_text="Selected image path...")
entry_image_path.grid(row=0, column=0, padx=5)

button_browse = ctk.CTkButton(frame_top, text="Browse Image", command=browse_image)
button_browse.grid(row=0, column=1, padx=5)

button_analyze = ctk.CTkButton(frame_top, text="Analyze Image", command=lambda: analyze_image(entry_image_path.get()))
button_analyze.grid(row=0, column=2, padx=5)

# --- Camera + Selected Image Frames ---
frame_images = ctk.CTkFrame(app, fg_color="white")
frame_images.pack(pady=10)

camera_label = ctk.CTkLabel(frame_images, text="Camera Preview", width=350, height=300, corner_radius=10)
camera_label.grid(row=0, column=0, padx=15)

selected_label = ctk.CTkLabel(frame_images, text="Selected / Captured Image", width=350, height=300, corner_radius=10)
selected_label.grid(row=0, column=1, padx=15)

# --- Camera Control Buttons ---
camera_buttons = ctk.CTkFrame(app, fg_color="white")
camera_buttons.pack(pady=5)

ctk.CTkButton(camera_buttons, text="📷 Start Camera", command=start_camera).grid(row=0, column=0, padx=10)
ctk.CTkButton(camera_buttons, text="📸 Capture & Analyze", fg_color="#0C8A43", command=capture_and_analyze).grid(row=0, column=1, padx=10)
ctk.CTkButton(camera_buttons, text="🛑 Stop Camera", fg_color="#AA0000", command=stop_camera).grid(row=0, column=2, padx=10)

# --- Result Display ---
label_result = ctk.CTkLabel(app, text="Detection Result:")
label_result.pack(pady=5)

result_box = ctk.CTkTextbox(app, width=850, height=200, fg_color="black", text_color="white")
result_box.pack(pady=10)

app.mainloop()
