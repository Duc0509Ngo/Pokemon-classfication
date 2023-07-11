import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage, RandomAffine, ColorJitter, Normalize, CenterCrop
import matplotlib.pyplot as plt
from categories import categories

class PokemonDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.categories = categories
        self.transform = transform

        data_path = os.path.join(root, "pokemon")

        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)

            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    transform = Compose([
        Resize((224, 224)),
        CenterCrop(size=200),
        RandomAffine(degrees=(-1, 1), scale=(0.9, 1.1), center=(0.5, 0.5)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.5),
        ToTensor(),
        Normalize(mean=[0.59565047, 0.59168748, 0.56110077], std=[0.23401408, 0.22395668, 0.23227972])
    ])
    dataset = PokemonDataset(root="../data", train=True, transform=transform)
    index = 350
    image, label = dataset.__getitem__(index)
    print(image.shape)
    print(label)
    plt.imshow(ToPILImage()(image))
    plt.show()

