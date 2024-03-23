import argparse
parser = argparse.ArgumentParser(description="Hyundai Heavy Industries Competition No.1")


# Train or Test
parser.add_argument("--mode", choices=("train", "test"), default="train")
parser.add_argument("--resume", default=True)

# Preprocessing
parser.add_argument("--cropheight", type=float, default=0.3)
parser.add_argument("--cropwidth", type=float, default=0.05)

# Model Trainin
parser.add_argument("--model", type=str, default="beittransformer")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--split", type=float, default=0.3)
parser.add_argument("--lr", type=float, default=0.000001)
parser.add_argument("--label_smoothing", type=float, default=0.1)

parser.add_argument("--save_model_name", type=str, default="experiment-beittransformer-2")
parser.add_argument("--load_model_name", type=str, default="experiment-beittransformer-2_17_best.pkl")

# GPU
parser.add_argument("--gpu", type=int, default=2)
parser.add_argument("--seed", type=int, default=7777)

# args
args = parser.parse_args()
