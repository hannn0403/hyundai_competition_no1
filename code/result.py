import os
import time
import tqdm
import warnings
import importlib
import numpy as np
import pandas as pd

import torch
import ttach
from torchvision import transforms
from torch.utils.data import DataLoader
from config import args
from dataset import CompNo1Dataset
from model import beittransformer, swintransformer, visiontransformer


warnings.filterwarnings(action='ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    weight_files = ['experiment-beittransformer_22_best.pkl', 'experiment-vitbase_17_best.pkl', 'experiment-swintransformer_77_best.pkl']
    model_names = ['beittransformer', 'visiontransformer', 'swintransformer']
    models = []
    for idx, (weight_file, model_name) in enumerate(zip(weight_files, model_names)):
        module = importlib.import_module(f"model.{model_name}")
        checkpoint = torch.load(f"../weights/{weight_file}", torch.device("cpu"))

        print(f"Downloading ... {model_name}")
        temp_model = module.make_model().cpu()
        temp_model.load_state_dict(checkpoint["model_state_dict"])
        models.append(temp_model)
        print(f"Load model {idx + 1}: {weight_file}")

    # Load Dataset
    path = f"../datasets/test"
    path_list = ["0", "1"]
    label_list = [[1.0, 0.0], [0.0, 1.0]]

    x, y = [], []  # x: 데이터, y: 라벨
    for idx, label in enumerate(path_list):
        for file_name in os.listdir(f"{path}/{label}"):
            data = f"{label}/{file_name}"
            x.append(data)
            y.append(label_list[idx])

    # Transform
    test_transforms = transforms.Compose(
        [
            transforms.Resize((284, 284), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5465, 0.4832, 0.5847], std=[0.2334, 0.2427, 0.2349]),
        ]
    )
    test_dataset = CompNo1Dataset(args, path, x, y, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, num_workers=2, shuffle=False)
    print("Load Test Dataset")

    # Prediction
    print("\nPrediction")
    image_list = []
    label_list = []
    predicted_list = [[], [], []]

    start_time = time.time()
    for batch_item in tqdm.tqdm(test_dataloader):
        for idx, model in enumerate(models):
            model = model.to(device)
            model.eval()
            tta_model = ttach.ClassificationTTAWrapper(model, ttach.aliases.ten_crop_transform(224, 224))

            images = batch_item["image"].to(device)
            labels = batch_item["label"].to(device)

            with torch.no_grad():
                outputs = tta_model(images)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels, 1)

            predicted_list[idx] += list(predicted.detach().cpu().numpy())
        image_list += batch_item["file"]
        label_list += list(labels.detach().cpu().numpy())
    end_time = time.time()

    # Scoring
    predicted_list = np.array(predicted_list)
    predicted_list = np.round(predicted_list.sum(axis=0) / 3).astype("uint8")
    accuracy = []
    for label, predict in zip(label_list, predicted_list):
        if label == predict:
            accuracy.append(True)
        elif label != predict:
            accuracy.append(False)

    result_dict = {
        "Image list": [i.split("/")[1] for i in image_list],
        "GT": label_list,
        "Prediction": predicted_list,
        "accuracy": accuracy,
        "Average Accuracy": sum(accuracy) / len(accuracy),
        "Inference Speed (ms)": f"{(end_time - start_time) / len(image_list):0.5f}ms/image"
    }
    result_csv = pd.DataFrame(result_dict)
    result_csv.to_csv("../result.csv", index=False)
    print("T E S T   C O M P L E T E !!")


if __name__ == "__main__":
    test()
