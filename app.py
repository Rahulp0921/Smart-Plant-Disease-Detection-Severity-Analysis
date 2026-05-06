from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import cv2
import json
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
from collections import Counter
import csv

HISTORY_FILE = "history_store.json"

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE) as f:
        history_predictions = json.load(f)
else:
    history_predictions = []

RESULTS_FILE = os.path.join("static", "severity_results.csv")

# create file with header if not exists
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "manual_percent", "model_percent"])

with open("disease_data.json") as f:
    crop_disease_info = json.load(f)

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("plant_model_v2.h5", compile=False)

with open("classes.json") as f:
    class_names = json.load(f)

raw_class_names = json.load(open("classes.json"))
class_names = [c.replace("___", " ").replace("_", " ") for c in raw_class_names]

IMG_SIZE = model.input_shape[1]
# =========================
# LEAF VERIFICATION
# =========================
THRESHOLD = 0.35

# =========================
# GRAD-CAM
# =========================
def get_gradcam(img_array, model, last_conv_layer_name="out_relu"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

# =========================
# SEVERITY ESTIMATION
# =========================
def get_severity(img):
    import cv2
    import numpy as np

    # =========================
    # 🟢 LEAF MASK (broad HSV)
    # =========================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_leaf = np.array([15, 25, 25])
    upper_leaf = np.array([100, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    k5 = np.ones((5,5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k5)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, k5)

    # fallback if mask is too small
    if np.sum(leaf_mask > 0) < 5000:
        leaf_mask = np.ones_like(leaf_mask) * 255

    # =========================
    # 🧠 HEALTHY GREEN (strong)
    # =========================
    strong_green = cv2.inRange(
        hsv,
        np.array([35, 80, 80]),
        np.array([90, 255, 255])
    )

    # =========================
    # 🎯 LOW–MEDIUM SATURATION (key filter)
    # =========================
    s = hsv[:, :, 1]
    sat_mask = cv2.inRange(s, 0, 150)  # keeps lesions, drops vivid green

    # =========================
    # 🦠 DISEASE MASK
    # =========================
    non_green = cv2.bitwise_not(strong_green)
    disease_mask = cv2.bitwise_and(non_green, sat_mask)
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    # =========================
    # 🧹 CLEANUP
    # =========================
    k3 = np.ones((3,3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, k3)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, k3)

    # remove tiny noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_mask, connectivity=8)
    clean = np.zeros_like(disease_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 30:
            clean[labels == i] = 255
    disease_mask = clean

    # =========================
    # 📊 SEVERITY
    # =========================
    leaf_pixels = np.sum(leaf_mask > 0)
    disease_pixels = np.sum((leaf_mask > 0) & (disease_mask > 0))

    if leaf_pixels == 0:
        return "Low", 0

    severity_percent = (disease_pixels / leaf_pixels) * 100

    # light calibration (keeps values realistic)
    severity_percent *= 1.1

    # =========================
    # 📈 LEVEL
    # =========================
    if severity_percent < 30:
        level = "Low"
    elif severity_percent < 60:
        level = "Medium"
    else:
        level = "High"

    return level, round(severity_percent, 2)
# =========================
# CROP INFO
# =========================

def normalize(text):
    return text.lower().replace(" ", "_").replace("__", "_").strip()


def is_leaf_image(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # green range
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    green_ratio = np.sum(mask > 0) / mask.size

    print("Green ratio:", green_ratio)

    return green_ratio > 0.10   # 🔥 threshold

def get_crop_info(label):

    label_clean = normalize(label)

    for key in crop_disease_info:
        key_clean = normalize(key)

        # 🔥 match if key contains label OR label contains key
        if (
            label_clean == key_clean or
            label_clean.replace("__", "_") == key_clean.replace("__", "_")
        ):

            info = crop_disease_info[key]

            # 🌿 Healthy
            if "healthy" in key_clean:
                return {
                    "type": "healthy",
                    "suggestions": info.get("suggestions", [])
                }

            # 🦠 Disease
            return {
                "type": "disease",
                "symptoms": info.get("symptoms", []),
                "remedies": info.get("remedies", [])
            }

    print("❌ NO MATCH FOUND FOR:", label)

    return {
        "type": "disease",
        "symptoms": ["No data found"],
        "remedies": ["Check JSON key match"]
    }

# =========================
# DISEASE TYPE DETECTION
# =========================
def get_disease_type(name):
    n = name.lower()

    if "virus" in n or "curl" in n:
        return "viral", "Viral"

    if "bacterial" in n:
        return "bacterial", "Bacterial"

    if any(x in n for x in [
        "blight", "mold", "rust", "spot",
        "scab", "mildew", "rot", "esca", "scorch"
    ]):
        return "fungal", "Fungal"

    return "unknown", "Unknown"

# =========================
# PREDICTION PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def prediction():

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":

            # =========================
            # SAVE IMAGE
            # =========================
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = cv2.imread(filepath)

            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            img_array = np.expand_dims(img_resized, 0)



            if not is_leaf_image(img):
                session["prediction"] = "❌ Not a leaf image"
                session["confidence"] = None
                session["image_path"] = None
                session["info"] = None
                return redirect(url_for("prediction"))

            # =========================
            # MODEL PREDICTION
            # =========================
            pred = model.predict(img_array)
            raw_conf = float(np.max(pred))

            index = np.argmax(pred)
            label = class_names[index]
            raw_label = raw_class_names[index]
            conf = raw_conf * 100

            # ✅ NEW: Confidence safety check (ADD THIS)
            if raw_conf < 0.7:
                session["prediction"] = "⚠️ Uncertain prediction"
                session["confidence"] = f"{raw_conf*100:.2f}%"
                session["image_path"] = None
                session["info"] = None
                return redirect(url_for("prediction"))
            crop_name = raw_label.split("___")[0] if "___" in raw_label else raw_label
            crop_name = crop_name.split("_")[0]
            crop_name = crop_name.replace("(", "").replace(")", "").strip()

            history_predictions.append(crop_name)

            print("Stored crop:", crop_name)
            print("Full history:", history_predictions)

            # SAVE HERE
            with open(HISTORY_FILE, "w") as f:
                json.dump(history_predictions, f)

            print("Stored:", history_predictions)   # debug

            # =========================
            # 🔍 INVALID IMAGE
            # =========================
            if raw_conf < THRESHOLD:

                session["prediction"] = "❌ Not a valid leaf image.Please upload a clear leaf image"
                session["confidence"] = None
                session["image_path"] = None
                session["info"] = None

                return redirect(url_for("prediction"))

            # =========================
            # ✅ VALID IMAGE
            # =========================

            # Grad-CAM
            heatmap = get_gradcam(img_array, model)

            # Severity
            severity_level, severity_percent = get_severity(img)
            severity = f"{severity_level} ({severity_percent}%)"
            print(f"[SEVERITY] {raw_label} -> {severity_percent}%")

            # 🔥 SAVE FOR VALIDATION
            with open(RESULTS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([file.filename, "", severity_percent])

            confidence = f"{conf:.2f}%"
            info = get_crop_info(raw_label)

            # Disease type badge
            dtype, label_type = get_disease_type(raw_label)
            info["disease_type"] = dtype
            info["label"] = label_type

            # =========================
            # HEATMAP OVERLAY
            # =========================
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed = cv2.addWeighted(img, 0.5, heatmap, 0.7, 0)

            text = f"{label} | {conf:.1f}% | {severity}"

            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(superimposed,
                          (5, 5),
                          (10 + text_w, 35),
                          (0, 0, 0), -1)

            cv2.putText(superimposed, text,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0), 2)

            cv2.imwrite(filepath, superimposed)

            # =========================
            # SAVE TO SESSION (PRG)
            # =========================
            session["prediction"] = label
            session["confidence"] = confidence
            session["severity"] = severity
            session["image_path"] = filepath
            session["info"] = info

            return redirect(url_for("prediction"))

    # =========================
    # GET REQUEST (AFTER REDIRECT)
    # =========================
    prediction = session.pop("prediction", None)
    confidence = session.pop("confidence", None)
    severity = session.pop("severity", None)
    image_path = session.pop("image_path", None)
    info = session.pop("info", None)

    return render_template(
        "prediction.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        info=info,
        severity=severity
    )

# =========================
# LIVE CAMERA
# =========================
@app.route("/predict_camera", methods=["POST"])
def predict_camera():

    data = request.json["image"]
    img_data = base64.b64decode(data.split(",")[1])

    img = Image.open(BytesIO(img_data)).convert("RGB")
    img = np.array(img)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_array = np.expand_dims(img_resized, 0)


    if not is_leaf_image(img):
        return {
            "label": "invalid",
            "confidence": None,
            "severity": None,
            "image": None,
            "info": None
        }

    # =========================
    # MODEL PREDICTION
    # =========================
    pred = model.predict(img_array)
    raw_conf = float(np.max(pred))

    index = np.argmax(pred)

    label = class_names[index]            # UI label
    raw_label = raw_class_names[index]    # JSON lookup

    conf = raw_conf * 100

    if raw_conf < 0.7:
        session["prediction"] = "⚠️ Uncertain prediction"
        session["confidence"] = f"{raw_conf*100:.2f}%"
        session["image_path"] = None
        session["info"] = None
        return redirect(url_for("prediction"))

    # =========================
    # ❌ INVALID IMAGE
    # =========================
    if raw_conf < THRESHOLD:
        return {
            "label": "invalid",
            "confidence": None,
            "severity": None,
            "image": None,
            "info": None
        }

    # =========================
    # ✅ VALID IMAGE
    # =========================
    heatmap = get_gradcam(img_array, model)
    severity_level, severity_percent = get_severity(img)
    severity = f"{severity_level} ({severity_percent}%)"

    # 🔥 SAVE FOR VALIDATION
    import time
    image_name = f"camera_{int(time.time())}.jpg"

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([image_name, "", severity_percent])

    # 🔥 GET INFO (NEW)
    info = get_crop_info(raw_label)

    dtype, label_type = get_disease_type(raw_label)

    info["disease_type"] = dtype
    info["label"] = label_type

    # =========================
    # HEATMAP OVERLAY
    # =========================
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.5, heatmap, 0.7, 0)

    # TEXT OVERLAY
    text = f"{label} | {conf:.1f}% | {severity}"
    cv2.putText(superimposed, text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # Encode image
    _, buffer = cv2.imencode('.jpg', superimposed)
    encoded = base64.b64encode(buffer).decode('utf-8')

    return {
        "label": label,
        "confidence": f"{conf:.2f}%",
        "severity": severity,
        "image": encoded,
        "info": info
    }

# =========================
# CROPS
# =========================
@app.route("/crops")
def crops():

    crop_info = {
        "Apple":{"temp":"10–25°C","humidity":"60–80%","soil":"Loamy","ph":"5.5–6.5","water":"Moderate","sunlight":"Full Sun","season":"Winter"},
        "Blueberry":{"temp":"13–21°C","humidity":"60–80%","soil":"Acidic","ph":"4.5–5.5","water":"High","sunlight":"Full Sun","season":"Spring"},
        "Cherry":{"temp":"15–25°C","humidity":"50–70%","soil":"Well-drained","ph":"6.0–7.0","water":"Moderate","sunlight":"Full Sun","season":"Spring"},
        "Corn":{"temp":"18–27°C","humidity":"50–80%","soil":"Fertile","ph":"5.8–7.0","water":"High","sunlight":"Full Sun","season":"Summer"},
        "Grape":{"temp":"15–30°C","humidity":"40–60%","soil":"Sandy","ph":"5.5–7.0","water":"Moderate","sunlight":"Full Sun","season":"Summer"},
        "Orange":{"temp":"20–30°C","humidity":"50–70%","soil":"Well-drained","ph":"5.5–6.5","water":"Moderate","sunlight":"Full Sun","season":"Winter"},
        "Peach":{"temp":"15–30°C","humidity":"50–70%","soil":"Sandy loam","ph":"6.0–7.0","water":"Moderate","sunlight":"Full Sun","season":"Spring"},
        "Pepper_bell":{"temp":"20–30°C","humidity":"60–80%","soil":"Rich","ph":"6.0–6.8","water":"Moderate","sunlight":"Full Sun","season":"Summer"},
        "Potato":{"temp":"15–20°C","humidity":"60–80%","soil":"Loose","ph":"5.0–6.5","water":"Moderate","sunlight":"Partial Sun","season":"Winter"},
        "Soybean":{"temp":"20–30°C","humidity":"50–70%","soil":"Well-drained","ph":"6.0–7.0","water":"Moderate","sunlight":"Full Sun","season":"Summer"},
        "Squash":{"temp":"20–30°C","humidity":"60–80%","soil":"Moist","ph":"6.0–7.5","water":"High","sunlight":"Full Sun","season":"Summer"},
        "Strawberry":{"temp":"10–25°C","humidity":"60–80%","soil":"Organic","ph":"5.5–6.5","water":"Moderate","sunlight":"Full Sun","season":"Spring"},
        "Tomato":{"temp":"20–30°C","humidity":"60–80%","soil":"Loamy","ph":"6.0–6.8","water":"Moderate","sunlight":"Full Sun","season":"Summer"}
    }

    return render_template(
        "crops.html",
        crops=crop_info,
        raw_class_names=raw_class_names,   # ✅ ADD THIS
        disease_data=crop_disease_info
    )

# =========================
# ANALYTICS
# =========================
ALL_CROPS = [
    "Tomato", "Apple", "Corn", "Potato",
    "Grape", "Strawberry", "Orange", "Peach",
    "Blueberry", "Cherry", "Pepper",
    "Raspberry", "Soybean", "Squash",
]



@app.route("/analytics")
def analytics():

    counter = Counter(history_predictions)

    # ✅ Always show all crops
    labels = ALL_CROPS
    counts = [counter.get(crop, 0) for crop in ALL_CROPS]

    # 🔥 prevent blank pie
    if sum(counts) == 0:
        labels = ["No Data"]
        counts = [1]

    with open("history.json") as f:
        history = json.load(f)

    return render_template(
        "analytics.html",
        pie_data={"labels": labels, "counts": counts},
        history=history
    )



# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)

