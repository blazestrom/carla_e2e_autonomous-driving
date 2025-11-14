import carla
import cv2
import numpy as np
import os
import csv
import time
import threading
from datetime import datetime

# output folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../dataset")
IMG_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "driving_log.csv")

os.makedirs(IMG_DIR, exist_ok=True)

class Collector:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.bp = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.actor_list = []

        # store latest camera frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def spawn_vehicle(self):
        car_bp = self.bp.filter("vehicle.*model3*")[0]
        spawn = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(car_bp, spawn)
        self.actor_list.append(self.vehicle)
        print("Vehicle spawned")

    def attach_camera(self):
        cam_bp = self.bp.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "320")
        cam_bp.set_attribute("fov", "90")

        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.6))
        self.camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.actor_list.append(self.camera)

        self.camera.listen(self._camera_callback)
        print("Camera attached")

    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        array = array[:, :, :3]  # remove alpha channel
        with self.frame_lock:
            self.latest_frame = array.copy()

    def start_collecting(self):
        # CSV file setup
        csv_file = open(CSV_PATH, "a", newline="")
        writer = csv.writer(csv_file)

        if os.stat(CSV_PATH).st_size == 0:
            writer.writerow(["image", "steering"])

        print("Recording... press CTRL+C to stop")

        try:
            while True:
                time.sleep(0.05)  # ~20 FPS

                with self.frame_lock:
                    frame = self.latest_frame

                if frame is None:
                    continue

                control = self.vehicle.get_control()
                steering = float(control.steer)

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_name = f"{timestamp}.png"
                img_path = os.path.join(IMG_DIR, img_name)

                cv2.imwrite(img_path, frame)

                # Write CSV
                writer.writerow([f"images/{img_name}", steering])
                csv_file.flush()

        except KeyboardInterrupt:
            print("Stopping recording...")

        finally:
            print("Cleaning up actors...")
            for a in self.actor_list:
                a.destroy()
            csv_file.close()
            print("Done.")

if __name__ == "__main__":
    c = Collector()
    c.spawn_vehicle()
    c.attach_camera()
    c.start_collecting()
