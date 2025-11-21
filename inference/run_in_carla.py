import carla
import torch
import cv2
import numpy as np
import time
from training.model import PilotNet

# ---------------------
# Settings
# ---------------------
MODEL_PATH = "../models/pilotnet_best.pth"
IMG_WIDTH = 320
IMG_HEIGHT = 160
SMOOTHING = 0.90         # steering smoothing factor
THROTTLE = 0.35          # fixed throttle for driving
# ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = PilotNet().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Model loaded!")

# Preprocess function
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0).to(device)

# ---------------------
# Main CARLA control class
# ---------------------
class DriveCar:
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        blueprint_library = self.world.get_blueprint_library()

        bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn = self.world.get_map().get_spawn_points()[0]

        self.vehicle = self.world.spawn_actor(bp, spawn)
        print("Vehicle spawned!")

        # Camera
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "320")
        cam_bp.set_attribute("fov", "90")

        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.6))
        self.camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.camera.listen(self.process_img)

        self.prev_steering = 0.0
        self.control = carla.VehicleControl(throttle=THROTTLE, steer=0.0)

    def process_img(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))[:, :, :3]

        # preprocess
        tensor = preprocess(img)

        with torch.no_grad():
            steering = float(model(tensor).cpu().numpy())

        # smoothing
        steering = SMOOTHING * self.prev_steering + (1 - SMOOTHING) * steering
        self.prev_steering = steering

        # apply control
        self.control.steer = float(steering)
        self.vehicle.apply_control(self.control)

    def run(self):
        print("Driving started... Press Ctrl+C to stop.")
        while True:
            time.sleep(0.03)

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    driver = DriveCar()
    try:
        driver.run()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        driver.vehicle.destroy()
