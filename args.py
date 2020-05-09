
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="MobileNetV2", help="Model Name")
parser.add_argument("-b", "--batch_size", type=int, default=200, help="Batch Size")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning Rate")
parser.add_argument("-e", "--epochs", type=int, default=200, help="Epochs")
parser.add_argument("-n", "--name", default="", help="Run Name")
parser.add_argument("-u", "--use_cuda", type = bool, default="False", help = "Use cuda")
parser.add_argument("-d", "--dataset", default = "CIFAR10", help="Dataset")
parser.add_argument("-ls", "--latent_sizes", type=str, default="", help="latent features of a mlp model")

parser.add_argument("-o", "--optim", type=str, default="sgd", help="Optimizer select")
parser.add_argument("-mt", "--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("-wd", "--decay", type=float, default=1e-4, help="Weight Decay")

parser.add_argument("-r", "--resume", type=bool, default=False, help="Whether to resume the training")
parser.add_argument("-c", "--checkpoint", help="checkpointpath")
parser.add_argument("-t", "--testing", type=bool, default=False, help="testing code")
parser.add_argument("-i", "--inference", type=bool, default=False, help="inference code")

parser.add_argument("-k", "--kfold", type=bool, default=False, help="Kfold validation")
parser.add_argument("-rf", "--runs_folder", type=str, default="runs", help="runs folder")

parser.add_argument("-ttep", "--trans_test_pad", type=int, default=30, help="Transfrom Test Pad")
parser.add_argument("-ttec", "--trans_test_crop", type=int, default=140, help="Transfrom Test Crop")

parser.add_argument("-tr", "--transform", type=str, default="v1", help="Transform version")
parser.add_argument("-ttrt", "--train_rot", type=int, default=40, help="Transfrom Train Rot")
parser.add_argument("-ttrp", "--train_pad", type=int, default=25, help="Transfrom Train Pad")
parser.add_argument("-ttrc", "--train_crop", type=int, default=100, help="Transfrom Train Crop")
parser.add_argument("-ttrb", "--train_bright", type=float, default=0.2, help="Transfrom Train Brightness")
parser.add_argument("-ttrco", "--train_contrast", type=float, default=0.1, help="Transfrom Train Contrast")

parser.add_argument("-rp", "--run_path", help="modelrunpath")
parser.add_argument("-ia", "--inference_all", type=bool, default=False, help="Inference over all models")

parser.add_argument("-trall", "--train_all", type=bool, default=False, help="Whether to train by class or all")