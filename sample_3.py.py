import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import cv2
import os

# --- Setup GUI window ---
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.title("Mango Leaf Disease Detection")
app.geometry("750x650")
app.configure(fg_color="white")

image_ref = None  # prevent garbage collection

# --- Pesticide recommendation dictionary ---
PESTICIDE_RECOMMENDATIONS = {
    "Cutting Weevil": "Carbaryl 50 WP (0.2%) or Chlorpyrifos 20 EC (2.5 ml/L). "
                      "Prune affected shoots and apply Neem oil (3%) as preventive.",
    "Gall Midge": "Dimethoate 30 EC (2 ml/L) or Malathion 50 EC (2 ml/L). "
                  "Remove infested panicles and maintain orchard hygiene.",
    "Anthracnose": "Copper oxychloride 0.3% or Carbendazim 0.1%. Repeat every 10–15 days.",
    "Powdery Mildew": "Wettable sulphur 0.2% or Hexaconazole 0.1%. Spray at early stage.",
    "Bacterial Canker": "Streptomycin sulphate (0.01%) + Copper oxychloride (0.3%). Spray fortnightly."
}


# --- Functions ---
def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select Leaf Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")]
    )
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


def get_pesticide_recommendation(disease_name):
    for key in PESTICIDE_RECOMMENDATIONS.keys():
        if key.lower() in disease_name.lower():
            return PESTICIDE_RECOMMENDATIONS[key]
    return "No specific pesticide recommendation available for this disease."


def analyze_image():
    result_box.delete("0.0", ctk.END)
    file_path = entry_image_path.get()
    if not os.path.exists(file_path):
        result_box.insert(ctk.END, "❌ No valid image selected.\n")
        return

    result_box.insert(ctk.END, "🟡 Analyzing image, please wait...\n")
    app.update_idletasks()

    try:
        script_path = os.path.join(os.path.dirname(__file__), "predict_disease.py")
        result = subprocess.run(
            ["python", script_path, file_path],
            capture_output=True,
            text=True,
            check=True
        )
        disease_name = result.stdout.strip()

        result_box.delete("0.0", ctk.END)
        result_box.insert(ctk.END, f"🩺 Detected Disease: {disease_name}\n")

        pesticide_info = get_pesticide_recommendation(disease_name)
        result_box.insert(ctk.END, f"💊 Recommended Pesticide: {pesticide_info}\n")

    except subprocess.CalledProcessError as e:
        result_box.delete("0.0", ctk.END)
        result_box.insert(ctk.END, f"❌ Error during analysis:\n{e.stderr or e}")


def realtime_detection():
    result_box.delete("0.0", ctk.END)
    result_box.insert(ctk.END, "📷 Starting real-time detection...\n")
    app.update_idletasks()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_box.insert(ctk.END, "❌ Could not open camera.\n")
        return

    script_path = os.path.join(os.path.dirname(__file__), "predict_disease.py")

    while True:
        ret, frame = cap.read()
        if not ret:
            result_box.insert(ctk.END, "⚠️ Failed to read frame.\n")
            break

        small = cv2.resize(frame, (300, 300))
        cv2.imshow("Real-time Leaf Detection - Press 'q' to quit", small)

        tmp_path = "temp_frame.jpg"
        cv2.imwrite(tmp_path, frame)

        try:
            result = subprocess.run(
                ["python", script_path, tmp_path],
                capture_output=True,
                text=True,
                check=True
            )
            disease_name = result.stdout.strip()
            pesticide_info = get_pesticide_recommendation(disease_name)
            display_text = f"{disease_name} | {pesticide_info.split('.')[0]}."
            cv2.setWindowTitle("Real-time Leaf Detection - Press 'q' to quit", display_text)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr or e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    result_box.insert(ctk.END, "✅ Real-time detection stopped.\n")


# --- GUI Layout ---
label = ctk.CTkLabel(app, text="Select a leaf image to detect disease:")
label.pack(pady=10)

entry_image_path = ctk.CTkEntry(app, width=500, placeholder_text="Selected image path...")
entry_image_path.pack(pady=5)

button_browse = ctk.CTkButton(app, text="Browse Image", command=browse_image)
button_browse.pack(pady=5)

image_label = ctk.CTkLabel(app, text="", width=200, height=200)
image_label.pack(pady=10)

button_analyze = ctk.CTkButton(app, text="Analyze Leaf Image", command=analyze_image)
button_analyze.pack(pady=10)

button_realtime = ctk.CTkButton(app, text="📷 Real-time Camera Detection", fg_color="#0C8A43", command=realtime_detection)
button_realtime.pack(pady=10)

label_result = ctk.CTkLabel(app, text="Detection Result:")
label_result.pack(pady=5)

result_box = ctk.CTkTextbox(app, width=650, height=180, fg_color="black", text_color="white")
result_box.pack(pady=10)

app.mainloop()
