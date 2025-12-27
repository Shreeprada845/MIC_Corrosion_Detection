from src.dataset import CORROSIONDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CORROSIONDataset("data/processed/train", transform=transform)

print("Number of training images:", len(train_dataset))
print("Sample label (0=no_corr, 1=corr):", train_dataset[0][1])
