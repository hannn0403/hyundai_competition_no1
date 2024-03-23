import os
import tqdm
import time
import random
import numpy as np
import pandas as pd
from importlib import import_module

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from dataset import CompNo1Dataset
from scheduler import CosineAnnealingWarmUpRestarts

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Solver:
    def __init__(self, config):
        self.config = config
        if not os.path.exists(f"../save/{self.config.save_model_name}"):
            os.makedirs(f"../save/{self.config.save_model_name}")

        self.device = torch.device(f"cuda:{self.config.gpu}" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 1

    # 시드 고정
    def set_seed(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

    # 데이터 셋 불러오기
    def load_dataset(self):
        x, y = [], []

        # 데이터 셋 경로 지정
        if self.config.mode == "train":
            path = f"../datasets/train"
        elif self.config.mode == "test":
            path = f"../datasets/test"

        # x: 데이터, y: 라벨
        label_list = [[1.0, 0.0], [0.0, 1.0]]
        for idx, label in enumerate(os.listdir(path)):
            for file_name in os.listdir(f"{path}/{label}"):
                data = f"{label}/{file_name}"
                x.append(data)
                y.append(label_list[idx])

        # 훈련 시에는 훈련 데이터셋과 검증 데이터셋으로 나눠서 사용
        if self.config.mode == "train":
            print("Load Training / Validation Dataset")
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=self.config.split,
                                                                random_state=self.config.seed)

            # Data Augmentation
            train_transforms = transforms.Compose(
                [
                    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomRotation(degrees=(-10, 10), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomPosterize(bits=5, p=0.2),

                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5465, 0.4832, 0.5847], std=[0.2334, 0.2427, 0.2349]),
                    transforms.RandomCrop((224, 224))
                ]
            )

            val_transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5465, 0.4832, 0.5847], std=[0.2334, 0.2427, 0.2349])
                ]
            )

            train_dataset = CompNo1Dataset(self.config, path, x_train, y_train, transform=train_transforms)
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch, num_workers=4, shuffle=True)

            val_dataset = CompNo1Dataset(self.config, path, x_test, y_test, transform=val_transforms)
            self.val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch, num_workers=4, shuffle=False)

        # 테스트 시에는 테스트 데이터셋 만 사용
        elif self.config.mode == "test":
            print("Load Test Dataset")
            test_transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5465, 0.4832, 0.5847], std=[0.2334, 0.2427, 0.2349]),
                ]
            )

            test_dataset = CompNo1Dataset(self.config, path, x, y, transform=test_transforms)
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch, num_workers=2, shuffle=False)

    # 시작 시에 초기화 (seed, 데이터셋, 모델, 목적 함수, 최적화 알고리즘)
    def build(self):
        print("Set seed:", self.config.seed)
        self.set_seed()

        print("Mode:", self.config.mode)
        print("Model:", self.config.model)
        module = import_module(f"model.{self.config.model.lower()}")
        self.model = module.make_model().cpu()  # cpu
        self.load_dataset()

        if self.config.mode == "train":
            self.model = self.model.to(self.device) # gpu
            print("Loss function: Cross Entropy")
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing).to(self.device)
            print("Optimizer: Adam")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.0001)
            print("Scheduler: CosineAnnealingWarmUpRestarts")
            self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=30, T_mult=2, eta_max=0.00001, T_up=10, gamma=0.8)

            # load checkpoint
            if self.config.resume:
                print(f"\nLoad pre-trained model {self.config.load_model_name} for training")
                model_name = self.config.load_model_name.split("_")[0]
                self.load_checkpoint(f"../save/{model_name}/{self.config.load_model_name}")  # cpu
                self.model = self.model.to(self.device)                                      # gpu
                print("Train:", self.config.load_model_name)
            else:
                print("\nTrain:", self.config.save_model_name)

        # load model
        elif self.config.mode == "test":
            print(f"Load pre-trained model {self.config.load_model_name} for test")
            model_name = self.config.load_model_name.split("_")[0]
            self.load_model(f"../save/{model_name}/{self.config.load_model_name}")  # cpu
            self.model = self.model.to(self.device)                                 # gpu
            print("\nTest:", self.config.load_model_name)

    # 훈련 (train and valid)
    def train_step(self, batch_item, training):
        images = batch_item["image"].to(self.device)
        labels = batch_item["label"].to(self.device)

        # training
        if training:
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            accuracy = (predicted == labels).sum()

        # validation
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels, 1)
                accuracy = (predicted == labels).sum()

        return loss.item(), accuracy.item()

    # 훈련 및 기록 (train and valid)
    def train(self):
        if self.config.mode == "train":
            self.writer = SummaryWriter(f"../log/{self.config.save_model_name}")
            self.model.train()

            loss_plot, val_loss_plot = [], []
            metric_plot, val_metric_plot = [], []

            # epoch
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                total_item, total_val_item = 0, 0
                total_loss, total_val_loss = 0, 0
                total_acc, total_val_acc = 0, 0

                # training
                tqdm_dataset = tqdm.tqdm(enumerate(self.train_dataloader))
                for batch, batch_item in tqdm_dataset:
                    batch_loss, batch_acc = self.train_step(batch_item, training=True)
                    total_item += len(batch_item["label"])
                    total_loss += batch_loss
                    total_acc += batch_acc

                    tqdm_dataset.set_postfix({
                        "LR": self.optimizer.param_groups[0]["lr"],
                        "Epoch": f"[{epoch}/{self.config.epochs}]",
                        "Batch": f"[{batch + 1}/{len(self.train_dataloader)}]",
                        "Loss": "{:06f}".format(batch_loss),
                        "Mean Loss": "{:06f}".format(total_loss / (batch + 1)),
                        "Accuracy": "{:06f}".format(100 * total_acc / total_item)
                    })
                loss_plot.append(total_loss / len(self.train_dataloader))
                metric_plot.append(100 * total_acc / total_item)

                # valindation
                tqdm_dataset = tqdm.tqdm(enumerate(self.val_dataloader))
                for batch, batch_item in tqdm_dataset:
                    batch_loss, batch_acc = self.train_step(batch_item, training=False)
                    total_val_item += len(batch_item["label"])
                    total_val_loss += batch_loss
                    total_val_acc += batch_acc

                    tqdm_dataset.set_postfix({
                        "LR": self.optimizer.param_groups[0]["lr"],
                        "Epoch": f"[{epoch}/{self.config.epochs}]",
                        "Batch": f"[{batch + 1}/{len(self.val_dataloader)}]",
                        "Loss": "{:06f}".format(batch_loss),
                        "Mean Loss": "{:06f}".format(total_val_loss / (batch + 1)),
                        "Accuracy": "{:06f}".format(100 * total_val_acc / total_val_item)
                    })
                val_loss_plot.append(total_val_loss / len(self.val_dataloader))
                val_metric_plot.append(100 * total_val_acc / total_val_item)

                self.writer.add_scalars('Loss', {"Train": total_loss / len(self.train_dataloader),
                                                 "Val": total_val_loss / len(self.val_dataloader)}, epoch)
                self.writer.add_scalars('Acc', {"Train": 100 * total_acc / total_item,
                                                "Val": 100 * total_val_acc / total_val_item}, epoch)

                if np.max(val_metric_plot) == val_metric_plot[-1] and epoch >= 10:
                    print("────────────────── B E S T   M O D E L !! ──────────────────")
                    model_path = f"../save/{self.config.save_model_name}/{self.config.save_model_name}_{epoch}_best.pkl"
                    self.save_checkpoint(model_path, epoch)

                if epoch % 5 == 0:
                    print("──────────────────── S A V E  M O D E L ────────────────────")
                    model_path = f"../save/{self.config.save_model_name}/{self.config.save_model_name}_{epoch}.pkl"
                    self.save_checkpoint(model_path, epoch)

                self.scheduler.step()

            print("────────────────── T R A I N   C O M P L E T E !! ──────────────────")
            self.writer.close()

        else:
            print("Not for training")

    # 테스트 (test)
    def test(self):
        if self.config.mode == "test":
            start = time.time()
            self.model.eval()
            tqdm_dataset = tqdm.tqdm(enumerate(self.test_dataloader))

            image_list = []
            label_list = []
            predicted_list = []

            for batch, batch_item in tqdm_dataset:
                images = batch_item["image"].to(self.device)
                labels = batch_item["label"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    _, labels = torch.max(labels, 1)

                image_list += batch_item["file"]
                label_list += list(labels.detach().cpu().numpy())
                predicted_list += list(predicted.detach().cpu().numpy())

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
                "Inference Speed (ms)": f"{(time.time() - start) / len(image_list)}ms/image"
            }

            result_csv = pd.DataFrame(result_dict)
            result_csv.to_csv("../result.csv", index=False)
            print("────────────────── T E S T   C O M P L E T E !! ──────────────────")

        else:
            print("Not for test")

    # 모델만 불러오기 (테스트용)
    def load_model(self, path):
        checkpoint = torch.load(path, torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])

    # 모델, 최적화 파라미터 불러오기 (훈련 체크포인트)
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, torch.device("cpu"))
        self.start_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 모델, 최적화 파라미터 저장하기
    def save_checkpoint(self, path, epoch):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(state, path)
