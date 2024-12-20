import cv2
import subprocess
import os
import time
from pathlib import Path
from datetime import datetime

# Desired preview image filename
PREVIEW_FILENAME = "thumb_preview.jpg"

# Path to the model files
MODEL_DIR = "models"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "ssd_mobilenet_v3_large_coco.pb")
MODEL_CONFIG = os.path.join(MODEL_DIR, "ssd_mobilenet_v3_large_coco.pbtxt")

# COCO class names (subset)
COCO_CLASSES = {
    1: "Person",
    2: "Bicycle",
    3: "Car",
    4: "Motorbike",
    5: "Airplane",
    6: "Bus",
    7: "Train",
    8: "Truck",
    9: "Boat",
    # Add more COCO classes as needed
}

def load_model():
    """Load and configure the detection model"""
    if not os.path.exists(MODEL_WEIGHTS) or not os.path.exists(MODEL_CONFIG):
        raise FileNotFoundError(f"Model files not found in {MODEL_DIR}")
    
    try:
        model = cv2.dnn_DetectionModel(MODEL_WEIGHTS, MODEL_CONFIG)
        model.setInputSize(320, 320)
        model.setInputScale(1.0 / 127.5)
        model.setInputMean((127.5, 127.5, 127.5))
        model.setInputSwapRB(True)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

def capture_preview():
    """Capture a preview image using gPhoto2"""
    # Remove any existing preview file
    if os.path.exists(PREVIEW_FILENAME):
        os.remove(PREVIEW_FILENAME)
    
    try:
        # Run gPhoto2 command to capture the preview
        result = subprocess.run(
            ["gphoto2", "--capture-preview", "--filename", PREVIEW_FILENAME, "--force-overwrite"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"gPhoto2 error: {result.stderr}")
            return False
        
        # Rename the file if gPhoto2 adds an extra prefix
        for file in os.listdir("."):
            if file.startswith("thumb_") and file != PREVIEW_FILENAME:
                os.rename(file, PREVIEW_FILENAME)
                break

        return os.path.exists(PREVIEW_FILENAME)
    except subprocess.TimeoutExpired:
        print("Camera preview capture timed out")
        return False
    except Exception as e:
        print(f"Error capturing preview: {e}")
        return False

def capture_photo():
    """Capture a high-resolution photo using gPhoto2"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_filename = f"bicycle_photo_{timestamp}.jpg"
    
    try:
        result = subprocess.run(
            ["gphoto2", "--capture-image-and-download", "--filename", photo_filename, "--force-overwrite"],
            capture_output=True,
            text=True,
            timeout=20
        )
        if result.returncode != 0:
            print(f"gPhoto2 error: {result.stderr}")
            return None
        
        print(f"Photo saved as {photo_filename}")
        return photo_filename
    except subprocess.TimeoutExpired:
        print("Photo capture timed out")
        return None
    except Exception as e:
        print(f"Error capturing photo: {e}")
        return None

def process_image(image, model):
    """Process image and detect bicycles"""
    if image is None:
        return []
    
    try:
        # Run object detection
        class_ids, confidences, boxes = model.detect(image, confThreshold=0.5)

        detections = []
        if len(class_ids) > 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
                label = COCO_CLASSES.get(class_id, f"Class {class_id}")
                detections.append((label, confidence, box))
                # Draw bounding box and label
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detections
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

def main():
    """Main loop to capture and process previews"""
    print("Starting camera preview with detection. Press 'q' to quit.")

    try:
        # Load the object detection model
        model = load_model()
        
        while True:
            # Capture a preview image
            if not capture_preview():
                print("Failed to capture preview. Retrying...")
                time.sleep(1)
                continue

            # Load the captured image
            image = cv2.imread(PREVIEW_FILENAME)
            if image is None:
                print("Failed to load preview image. Retrying...")
                time.sleep(1)
                continue

            # Detect objects in the image
            detections = process_image(image, model)

            # Check for bicycle detections
            bicycle_detected = any(label == "Bicycle" for label, _, _ in detections)

            if bicycle_detected:
                print("Bicycle detected!")
                capture_photo()
                time.sleep(2)  # Avoid rapid captures

            # Print all detections to the console
            if detections:
                for label, confidence, _ in detections:
                    print(f"Detected: {label} ({confidence:.2f})")
            else:
                print("No objects detected.")

            # Display the image with detections
            cv2.imshow("Camera Preview with Detections", image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Wait for 1 second before capturing the next preview
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
