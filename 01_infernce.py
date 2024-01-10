import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

# Add the project's files to the python path
#file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
#file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
#sys.path.append(file_path)
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
from src.models.segmentation import PointSegmentationModule

# Parse the configs using hydra
cfg = init_config(overrides=[
    "experiment=dales",
    "ckpt_path=/home/daktar/superpoint_transformer/logs/train/runs/2024-01-05_11-00-33/checkpoints/epoch_279.ckpt"
])

#print(cfg)
#print(cfg.model)

#ckpt_path2="/home/daktar/superpoint_transformer/logs/train/runs/2024-01-05_11-00-33/checkpoints/epoch_279.ckpt"

# Instantiate the datamodule
datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

# Instantiate the model
model = hydra.utils.instantiate(cfg.model)

#model2 = PointSegmentationModule()
#model2 = LitModule.load_from_checkpoint(cfg.ckpt_path, net=model.net, criterion=model.criterion)

#print(model2)

# Load pretrained weights from a checkpoint file
model = PointSegmentationModule.load_from_checkpoint(cfg.ckpt_path, net=model.net, criterion=model.criterion)
model = model.eval().cuda()

# Pick among train, val, and test datasets. It is important to note that
# the train dataset produces augmented spherical samples of large 
# scenes, while the val and test dataset
# dataset = datamodule.train_dataset
dataset = datamodule.val_dataset
# dataset = datamodule.test_dataset

# For the sake of visualization, we require that NAGAddKeysTo does not 
# remove input Data attributes after moving them to Data.x, so we may 
# visualize them
for t in dataset.on_device_transform.transforms:
    if isinstance(t, NAGAddKeysTo):
        t.delete_after = False

# Load a dataset item. This will return the hierarchical partition of an 
# entire tile, within a NAG object 

for i in tqdm(range(0,len(dataset))):
    nag = dataset[i]


    # Apply on-device transforms on the NAG object. For the train dataset, 
    # this will select a spherical sample of the larger tile and apply some
    # data augmentations. For the validation and test datasets, this will
    # prepare an entire tile for inference
    nag = dataset.on_device_transform(nag.cuda())
    #print(nag)

    # Inference
    logits = model(nag)

    # If the model outputs multi-stage predictions, we take the first one, 
    # corresponding to level-1 predictions 
    if model.multi_stage_loss:
        logits = logits[0]

    # Compute the level-0 (pointwise) predictions based on the predictions
    # on level-1 superpoints
    l1_preds = torch.argmax(logits, dim=1).detach()
    l0_preds = l1_preds[nag[0].super_index]

    # Save predictions for visualization in the level-0 Data attributes 
    nag[0].pred = l0_preds

    #print(nag[0])
    #export_numpy = nag[0].detach().cpu().numpy()
    #print (export_numpy)
    T1 = nag[0].pos
    #print(T1.shape)
    T2 = nag[0].intensity.unsqueeze(1)
    #print(T2)
    T3 = nag[0].pred.unsqueeze(1)
    #print(T3.shape)

    T = torch.cat((T1,T2,T3), -1)
    #print(T.shape)
    #print(T)


    numpyTensor = T.cpu().numpy()
    filename = "result_"+ str(i) + ".txt"
    np.savetxt(filename, numpyTensor)
    del T1
    del T2
    del T3
    del nag
    del logits
    torch.cuda.empty_cache()
#nag.save("output.h5")

# Visualize the hierarchical partition
'''
show(
    nag, 
    class_names=CLASS_NAMES, 
    ignore=NUM_CLASSES,
    class_colors=CLASS_COLORS,
    max_points=100000
)


# Pick a center and radius for the spherical sample
center = torch.tensor([[40, 115, 0]]).to(nag.device)
radius = 10

# Create a mask on level-0 (ie points) to be used for indexing the NAG 
# structure
mask = torch.where(torch.linalg.norm(nag[0].pos - center, dim=1) < radius)[0]

# Subselect the hierarchical partition based on the level-0 mask
nag_visu = nag.select(0, mask)




# Visualize the sample
show(
    nag_visu,
    class_names=CLASS_NAMES,
    ignore=NUM_CLASSES,
    class_colors=CLASS_COLORS, 
    max_points=100000
)

'''