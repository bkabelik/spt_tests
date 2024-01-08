import os
import sys

# Add the project's files to the python path
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
#file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
sys.path.append(file_path)
sys.path.append("/home/daktar/superpoint_transformer")

# Necessary for advanced config parsing with hydra and omegaconf
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

import hydra
from src.utils import init_config
import torch
from src.visualization import show
from src.datasets.dales import CLASS_NAMES, CLASS_COLORS
from src.datasets.dales import DALES_NUM_CLASSES as NUM_CLASSES
from src.transforms import *

# Parse the configs using hydra
cfg = init_config(overrides=[
    "experiment=dales",
    "ckpt_path=/home/daktar/superpoint_transformer/logs/train/runs/2024-01-05_11-00-33/checkpoints/epoch_279.ckpt"
])


# Instantiate the datamodule
datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

# Instantiate the model
model = hydra.utils.instantiate(cfg.model)

# Load pretrained weights from a checkpoint file
model = model.load_from_checkpoint(cfg.ckpt_path, net=model.net, criterion=model.criterion)
model = model.eval().cuda()
