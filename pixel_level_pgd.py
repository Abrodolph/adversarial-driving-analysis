"""
Experiment 1: Pixel-Level Integrity
Attack Type: Projected Gradient Descent (PGD)
Target: ResNet50 (Simulating a visual perception module)
"""

import torch
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import torchattacks
from torchvision import transforms
import torchvision.models as models

def run_pixel_attack():
    # --- 1. SETUP & DOWNLOAD ---
    # Using official PyTorch Hub test image to avoid bot-blocking issues
    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    print(f"[INFO] Downloading image from {url}...")
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status Code: {response.status_code}")
    
    original_image = Image.open(BytesIO(response.content))
    print("✅ Image downloaded.")

    # --- 2. PREPARE MODEL ---
    print("[INFO] Loading ResNet50 model...")
    model = models.resnet50(pretrained=True)
    model.eval()

    # --- 3. PREPROCESS ---
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    x = preprocess(original_image).unsqueeze(0)

    # --- 4. ATTACK SETUP ---
    # PGD Attack: eps=8/255 (Perceptible noise limit), steps=10
    print("[INFO] Initializing PGD Attack...")
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)

    # --- 5. RUN EXPERIMENT ---
    original_prediction = model(x).argmax(dim=1)
    print(f"Original Class ID: {original_prediction.item()}")

    print("[INFO] Generating adversarial examples...")
    adv_images = atk(x, original_prediction)

    new_prediction = model(adv_images).argmax(dim=1)
    print(f"Adversarial Class ID: {new_prediction.item()}")

    # --- 6. VISUALIZE RESULTS ---
    save_path = "result_pixel_integrity.png"
    _plot_results(x, adv_images, original_prediction, new_prediction, save_path)
    
    if original_prediction.item() != new_prediction.item():
        print(f"\n✅ SUCCESS: Model fooled. Result saved to {save_path}")
    else:
        print(f"\n❌ FAILURE: Model not fooled. Result saved to {save_path}")

def _plot_results(clean_tensor, adv_tensor, orig_pred, new_pred, save_path):
    clean_img = transforms.ToPILImage()(clean_tensor.squeeze())
    adv_img = transforms.ToPILImage()(adv_tensor.squeeze())

    plt.figure(figsize=(12, 6))

    # Original
    plt.subplot(1, 2, 1)
    plt.title(f"Original (Integrity Intact)\nClass ID: {orig_pred.item()}")
    plt.imshow(clean_img)
    plt.axis('off')

    # Adversarial
    plt.subplot(1, 2, 2)
    plt.title(f"Adversarial (Integrity Compromised)\nClass ID: {new_pred.item()}")
    plt.imshow(adv_img)
    plt.axis('off')

    plt.savefig(save_path)
    # plt.show() # Uncomment if running in a notebook/GUI environment

if __name__ == "__main__":
    run_pixel_attack()
