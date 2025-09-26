import cv2
import os
import hashlib
import time
from strands import tool

@tool
def camera(prompt=None):
    screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    # Generate a unique hash for the filename using time and randomness
    unique_str = f"{time.time()}_{os.urandom(8).hex()}"
    unique_hash = hashlib.sha256(unique_str.encode()).hexdigest()[:16]
    img_path = os.path.join(screenshots_dir, f"screenshot_{unique_hash}.jpg")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image")

    # Enhance image quality using CLAHE and brightness/contrast adjustment
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Optionally, increase brightness and contrast
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 20    # Brightness control (0-100)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    cv2.imwrite(img_path, enhanced)
    return img_path

@tool
def analyze_image(prompt=None):
    """
    Analyze an image given its path. Returns basic info and tries to detect if there are no persons, a single person, or multiple persons.
    Usage: analyze_image('screenshots/screenshot_xxx.jpg')
    """
    if not prompt:
        return "Please provide the image path to analyze."
    img_path = prompt.strip()
    if not os.path.exists(img_path):
        return f"Image not found: {img_path}"
    img = cv2.imread(img_path)
    if img is None:
        return f"Failed to load image: {img_path}"
    height, width, channels = img.shape
    mean_color = img.mean(axis=(0, 1)).tolist()  # BGR mean

    # Person detection using Haar Cascade (frontal face)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        person_result = "Cascade file not found. Person detection unavailable."
        num_persons = None
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        num_persons = len(faces)
        if num_persons == 0:
            person_result = "No persons detected."
        elif num_persons == 1:
            person_result = "Single person detected."
        else:
            person_result = f"Multiple persons detected: {num_persons}"

    return {
        "path": img_path,
        "width": width,
        "height": height,
        "channels": channels,
        "mean_color_bgr": mean_color,
        "person_detection": person_result,
        "person_count": num_persons
    }


