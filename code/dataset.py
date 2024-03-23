import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset


class CompNo1Dataset(Dataset):
    def __init__(self, config, path, files, labels, transform=None):
        super(CompNo1Dataset, self).__init__()
        self.config = config
        self.path = path
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = f"{self.path}/{self.files[item]}"

        # crop image
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        img_height = image.shape[0]
        img_width = image.shape[1]
        image = image[int(img_height * self.config.cropheight):,
                      int(img_width * self.config.cropwidth): img_width - int(img_width * self.config.cropwidth), :]

        # CLAHE
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(image)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.merge((l, a, b))
        image = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)

        # cv2 to PIL
        image = Image.fromarray(image)
        # image = transforms.PILToTensor(image)

        if self.transform:
            image = self.transform(image)

        return {
            "file": self.files[item],
            "image": image,
            "label": torch.tensor(self.labels[item], dtype=torch.float32)
        }
