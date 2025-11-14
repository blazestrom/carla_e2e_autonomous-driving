import cv2
import numpy as np
import os
import csv

FRAMES_DIR = r"C:\Users\piyus\OneDrive\Desktop\mi\data base\tusimple_preprocessed\training\frames"
MASKS_DIR = r"C:\Users\piyus\OneDrive\Desktop\mi\data base\tusimple_preprocessed\training\lane-masks"

OUTPUT_CSV = "steering_labels.csv"

def compute_lane_center(mask):
    # Convert mask to grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # White pixels = lane lines
    ys, xs = np.where(gray > 0)

    if len(xs) == 0:
        return None

    # Lane center = mean x of white pixels
    lane_center = np.mean(xs)
    return lane_center

def compute_steering(lane_center, img_width):
    img_center = img_width / 2
    steering = (lane_center - img_center) / img_center
    return float(steering)

def main():
    files = sorted(os.listdir(FRAMES_DIR))
    with open(OUTPUT_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "steering"])

        for file in files:
            if not file.endswith(".jpg"):
                continue

            frame_path = os.path.join(FRAMES_DIR, file)
            mask_path = os.path.join(MASKS_DIR, file)

            if not os.path.exists(mask_path):
                print("Mask missing:", file)
                continue

            frame = cv2.imread(frame_path)
            mask = cv2.imread(mask_path)

            h, w, _ = frame.shape

            lane_center = compute_lane_center(mask)
            if lane_center is None:
                continue

            steering = compute_steering(lane_center, w)

            writer.writerow([frame_path, steering])

    print("Steering CSV created:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
