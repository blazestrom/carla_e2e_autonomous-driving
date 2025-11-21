import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from training.model import PilotNet
from training.dataset import SteeringDataset, get_transforms

CSV_PATH = "steering_labels.csv"
MODEL_PATH = "models/pilotnet_best.pth"

print("Loading dataset...")
dataset = SteeringDataset(CSV_PATH, ".", transform=get_transforms(train=False))

print(f"Total samples: {len(dataset)}")

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Model loaded!")

# Function to calculate MAE + RMSE
def evaluate_metrics(dataset, max_samples=200):
    print("Evaluating model...")
    mae_list = []
    rmse_list = []

    for i in range(min(max_samples, len(dataset))):
        img, true_steer = dataset[i]
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img).cpu().numpy()[0]

        mae_list.append(abs(pred - true_steer))
        rmse_list.append((pred - true_steer) ** 2)

    MAE = np.mean(mae_list)
    RMSE = np.sqrt(np.mean(rmse_list))

    return MAE, RMSE


# Compute metrics
MAE, RMSE = evaluate_metrics(dataset)

print("\n----- PROJECT PERFORMANCE SUMMARY -----")
print(f"Mean Absolute Error (MAE): {MAE:.5f}")
print(f"Root Mean Squared Error (RMSE): {RMSE:.5f}")
print("---------------------------------------\n")

# Pick random sample and show results
idx = random.randint(0, len(dataset) - 1)
img, true_steer = dataset[idx]
tensor = img.unsqueeze(0).to(device)

with torch.no_grad():
    pred_steer = model(tensor).cpu().numpy()[0]

# Convert image back to numpy for display
img_np = img.permute(1, 2, 0).cpu().numpy()
img_np = (img_np + 1) / 2  # scale back to [0,1]

plt.figure(figsize=(8,5))
plt.imshow(img_np)
plt.title(f"Steering\nTrue: {true_steer:.4f} | Predicted: {pred_steer:.4f}")
plt.axis("off")
plt.show()
