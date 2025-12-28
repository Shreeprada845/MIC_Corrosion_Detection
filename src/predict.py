import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from model import CorrosionCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(img_path, model_path="models/best_model.pth"):
    model = CorrosionCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image = load_image(img_path).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prob = output.item()
        label = "CORRODED" if prob > 0.5 else "NOT CORRODED"
        print(label)
        print(prob)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    predict(sys.argv[1])
