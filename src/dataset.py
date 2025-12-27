import os
from PIL import Image
from torch.utils.data import Dataset

class CORROSIONDataset(Dataset):
    """
    Dataset for CORROSION vs no-CORROSION classification.
    Labels are inferred from folder names.
    """

    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to data/processed/train or val or test
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ['NOCORROSION', 'CORROSION']
        self.class_to_idx = {
            'NOCORROSION': 0,
            'CORROSION': 1
        }

        self.samples = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                raise FileNotFoundError(f"Expected folder not found: {class_path}")

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, file_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
