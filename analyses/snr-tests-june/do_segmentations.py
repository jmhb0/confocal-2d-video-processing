# hard-coded script to segment everthing
# Currently it's dumb and assumes you're running everything from the dir where 
# it lives
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tifffile 

sys.path.insert(0,"../")
import segmentation_core as sc
import pipeline_utils as pu

#------------------Parameters-------------------------------------------------#
# `seg_frame`: which frame to segment and dump in `results_seg_frame`. 
# If seg_frame="all", then segment the whole video and put in `results_seg_video`.
seg_frame = 0
seg_frame = "all"
# `files_to_segment`: which videos to segment. The meaning of the codes mapped
# in `fname_lookup.csv`
files_to_segment = ["A0","A1","A2","A3","A4","A5","A6","A7","A8","A9","B1","B2","B3","B4","B5","B6","B7","B8","B9",]
files_to_segment = ["A0", "B2",]
files_to_segment = ["A9"]
verbose=1
binarize=True  # if segoutput is instance-level, change to binary `object/background`

#------------------Folder locations--------------------------------------------#
data_dir = "/Users/jamesburgess/vae-bio-image/data/raw_data/ips-snr-tests"
data_dir_A = f"{data_dir}/hiPSCs_setA"
data_dir_B = f"{data_dir}/hiPSCs_setB"

fname_file_lookup = "./fname-lookup.csv"
dir_config_file = "./configs"
fname_seg_funcs = '../seg-func-map.yaml'

dir_results_frame = "./results_seg_frame"
dir_results_video = "./results_seg_timelapse"

order_channels=['golgi', 'er','lyso', 'mito', 'peroxy']

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    print("Doing segmentation of full videos, for their config files")
    print(f"Segmenting files with the codes {files_to_segment}")
    n = len(files_to_segment)
    n_success = 0
    
    # load the file list and check the files are there 
    print(f"reading file lookup from {fname_file_lookup}")
    df_files = pd.read_csv(fname_file_lookup).set_index('index')
    missing_codes = set(files_to_segment).difference(df_files.index)
    if len(missing_codes)>0:
        raise ValueError(f"The file codes {missing_codes} are not in the file lookup table {fname_file_lookup}")
    else:
        print(f"All file codes have values in {files_to_segment}")

    # Check the config files exist
    config_files = os.listdir(dir_config_file)
    config_files_codes = [s.split("-")[0] for s in config_files]
    missing_configs = set(files_to_segment).difference(config_files_codes)
    if len(missing_configs)>0:
        raise ValueError(f"Config files for {missing_configs} are not in the config dir {dir_config_file}")
        
    # Create results directory. Depends on whether it's whole video or frame.
    if seg_frame=='all': 
        dir_results = dir_results_video
        axes = "TCYX"
    elif type(seg_frame) is int: 
        dir_results = dir_results_frame
        axes = "YX"
    else: 
        raise ValueError("Invalid value for `seg_frame` defined in script")
    print(f"Putting all results in {dir_results}")
    dir_results = Path(dir_results)
    dir_results.mkdir(exist_ok=True)

    # load the list of segmentation functions & their defaults
    seg_funcs = sc.load_segmentation_functions(fname=fname_seg_funcs)
    
    # read in the files and do the segmenttions
    for file_code in files_to_segment:
        print("-"*80)
        print(f"File code {file_code}")
        
        # config file name and load
        config_fname = f"{dir_config_file}/{file_code}-config.yaml"
        print(f"Loading config file {config_fname}")
        seg_config = sc.load_segmentation_config_from_file(fname=config_fname, 
                                    seg_funcs=seg_funcs, verbose=0)

        # get the filename and load image. The `seg_frame` determines whether
        # it's the whole video or just a frame
        fname = df_files.loc[file_code]['name']
        if file_code[0] == "A": fname = f"{data_dir_A}/{fname}"
        elif file_code[0] == "B": fname = f"{data_dir_B}/{fname}"
        else: raise ValueError("Error, read the code")
        seg_fname = f"{fname}-seg.czi"
        fname = f"{fname}.czi"
        print(f"Loading image from {fname}")
        img = pu.read_czi_img(fname, frame=seg_frame, order_channels=order_channels)
        print(f"Image shape is {img.shape}")
        
        ## Branch based on frame or video. Load+seg+save
        # baseline for the filename. Add file code, get the file, and strip out .czi
        fname_out_base = file_code+"_"+fname.split("/")[-1][:-4] 
        
        # whole timelapse 
        if seg_frame=='all':
            print(f"Doing timelapse segmentation")
            seg = sc.seg_2d_timelapse(img, seg_config, seg_funcs, 
                                order_channels=order_channels, verbose=verbose)
            if binarize: seg[seg>0]=1
            print(f"Out shape {seg.shape}")
            fname_out = fname_out_base + ".tiff"# fname, then strip out the .czi
            fname_out = f"{dir_results}/{fname_out}"
            print(f"Saving file {fname_out}")
            tifffile.imsave(fname_out, seg, imagej=True, metadata={'axes': axes})

        # single frame
        else:
            print(f"Doing single-frame segmentation on img shape {img.shape}")
            for i, ch_name in enumerate(order_channels):
                seg = sc.seg_2d(img[i], seg_config, seg_funcs, ch_name=ch_name)
                if binarize: 
                    seg[seg>0]=1
                    seg = seg.astype('bool')
                fname_out = f"{fname_out_base}_{ch_name}_frame_{seg_frame}.tiff"
                fname_out = f"{dir_results}/{fname_out}"
                print(f"\tSaving channel {ch_name} to {fname_out}")
                tifffile.imsave(fname_out, seg)#, imagej=True, metadata={'axes': axes})

    #-----------------------------------------------------------------------------#
    print(f"Finished all files in {files_to_segment}")
    print("Done")
