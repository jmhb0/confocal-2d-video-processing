## imports 
SEG_RUN_NAME="feb23"
DO_CELL_MASK=True
overwrite=True
from pathlib import Path
PATH_PROJECT = Path("/pasteur/u/jmhb/confocal-2d-video-processing")
import sys 
import os
sys.path.append("/pasteur/u/jmhb/confocal-2d-video-processing")

PATH_PROJECT
PATH_DATA = Path("/pasteur/data/hiPSCs_January2022/")
PATH_FNAMES = PATH_DATA / "fname-lookup-timelapse.csv"
PATH_CONFIG = PATH_PROJECT / "analyses/jan22-neural-diff/configs/feb23.yaml"
PATH_SEG_FUNCS = PATH_PROJECT / "seg-func-map.yaml"

PATH_RESULTS = PATH_DATA / "seg-framewise" / SEG_RUN_NAME
PATH_RESULTS.mkdir(exist_ok=True)

print("Doing lots of segmentations")
print(f"Config file {PATH_CONFIG}")
print(f"Fname lookup {PATH_FNAMES}")
print(f"Putting results in {PATH_RESULTS}")

import yaml
import tifffile
import aicsimageio
from aicsimageio.writers import OmeTiffWriter
import pandas as pd
import numpy as np
from aicsimageio import AICSImage
import sys
import segmentation_core as sc
import pipeline_utils as pu
import utils
import importlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seg_processing as sp

## load filenames 
df_fnames = pd.read_csv(PATH_FNAMES).set_index('index')
df_fnames['path_file'] = str(PATH_DATA) +"/"+ df_fnames['folder']+"/"+df_fnames['fname']
df_fnames=df_fnames.loc[["AP" in s for s in df_fnames.index]]

## load the seg functions, and the config params for this run 
seg_funcs = sc.load_segmentation_functions(fname=PATH_SEG_FUNCS)
seg_config = sc.load_segmentation_config_from_file(fname=PATH_CONFIG, seg_funcs=seg_funcs, verbose=0)


#### update - changing the order 

for i, row in df_fnames.iterrows():
    #if row.name not in ("AP48-CTRL-8","AP48-CTRL-9","AP48-CTRLOAC-1","AP48-CTRLOAC-2","AP48-CTRLOAC-3","AP48-CTRLOAC-4","AP48-CTRLOAC-5","AP48-CTRLOAC-6","AP48-CTRLOAC-7","AP48-CTRLOAC-8","AP48-CTRLOAC-9","AP48-CTRLOAC-10",):
    #    continue
    idx=row.fname
    print("\n\n", idx)
    fname_img=Path(row['path_file'])
    print(fname_img)
    
    # duplicate the same folder structure inside the segmentation directory. 
    DIR_results = PATH_RESULTS / row['folder']
    DIR_results.mkdir(exist_ok=True, parents=True)
    fname_results = DIR_results / row['fname']

    # read data 
    img_obj = AICSImage(fname_img)
    img = img_obj.data
    # if "ApoE" in str(fname_img): 
    #     img = np.transpose(img, (2,1,0,3,4))
        
    frames, _, zs, y, x = img_obj.shape

    # create segmentation object 
    seg=np.zeros_like(img, dtype=img_obj.dtype)

    # get the channel lookup
    importlib.reload(pu)
    ch_idx_lookup = pu.get_channel_idx_lookup(img_obj.channel_names)
    ch_idx_lookup_r = dict(zip(ch_idx_lookup.values(), ch_idx_lookup.keys()))

    # do segmentation for the channels 
    channels = list(seg_config.keys())

    for frame in range(frames):
        for z in range(zs):
            # get the central-cell mask which is the same for each channel

            img_tmp = img[[frame],:,z].copy()
            if DO_CELL_MASK:
                f, (img_sum, img_sum_blurred, mask, mask_cell) = sp.whole_cell_segment_eval(fname_img, img_tmp, 
                                                title=idx, merge_channels=['lyso', 'mito', 'golgi', 'peroxy', 'er',])
                mask_cell=mask_cell[0]
            else:
                mask_cell=None

            # iterate over channels
            for ch in channels:
                ch_idx=ch_idx_lookup[ch]
                import warnings
                warnings.simplefilter(action='ignore', category=FutureWarning)
                seg_ch_fr = sc.seg_2d(img[frame,ch_idx,z], seg_config, seg_funcs, ch_name=ch, 
                                      inplace=False, save_inters=False, mask=mask_cell)
                seg[frame, ch_idx, z] = seg_ch_fr
    
        OmeTiffWriter.save(seg, fname_results, dim_order="TCZYX",
                          channel_names=img_obj.channel_names,
                           physical_pixel_sizes=aicsimageio.types.PhysicalPixelSizes(*img_obj.physical_pixel_sizes))
