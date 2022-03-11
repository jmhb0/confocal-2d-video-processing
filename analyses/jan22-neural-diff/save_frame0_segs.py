SEG_RUN_NAME="feb23"

frame=0  # extract this frame 
channel="mito" # 

overwrite=True
from pathlib import Path
PATH_PROJECT = Path("/pasteur/u/jmhb/confocal-2d-video-processing")
import sys 
sys.path.append("/pasteur/u/jmhb/confocal-2d-video-processing")

PATH_PROJECT
PATH_DATA = Path("/pasteur/data/hiPSCs_January2022")
PATH_FNAMES = PATH_DATA / "fname-lookup-timelapse.csv"

PATH_RESULTS = PATH_DATA / "seg-framewise" / SEG_RUN_NAME
PATH_RESULTS.mkdir(exist_ok=True)

import yaml
import torch
import tifffile
import pandas as pd
import numpy as np
from torchvision.utils import make_grid
import aicssegmentation
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import aicsimageio
from IPython.display import HTML
import sys
import utils
import importlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import seg_processing as sp
import pipeline_utils as pu

import matplotlib as mpl
mpl.rcParams['animation.embed_limit']=100

## load filenames 
df_fnames = utils.get_fname_lookup(PATH_FNAMES, PATH_DATA, PATH_RESULTS)
# filter out AP for now 
df_fnames=df_fnames.loc[["AP" not in s for s in df_fnames.index]]


# object to hold all the things
all_imgs_and_seg = dict()

print(len(df_fnames))
for i, row in df_fnames.iterrows():
    print(i,end=", ")
    img_obj=AICSImage(row['path_file'])
    seg_obj=AICSImage(row['path_seg'])
    idx = row.name

    ch_idx_lookup = pu.get_channel_idx_lookup(img_obj.channel_names)
    ch_idx = ch_idx_lookup[channel]

    img = img_obj.data[frame,ch_idx,0].copy()
    seg = seg_obj.data[frame,ch_idx,0].copy()

    all_imgs_and_seg[idx]=[img, seg]
    del img, seg, img_obj, seg_obj
    
torch.save(all_imgs_and_seg, "./tmp.sav" )
