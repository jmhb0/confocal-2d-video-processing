import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from aicsimageio import AICSImage
import aicssegmentation 
import numpy as np
import pipeline_utils as pu
import os

left, right  = 0.0, 1.0    # the left side of the subplots of the figure
bottom, top = 0.0, 1.0   # the bottom of the subplots of the figure
wspace, hspace = 0.03, 0.03  # the amount of width reserved for blank space between subplots

def animate_video(x_video, figsize=(10,10), axis_off=0):
    """
    Generate animation for a video tensor with shape (frames,1,H,W).

    To display animation do: 
        ani=animate_video(x_video)
        from IPython.display import HTML
        HTML(ani.to_jshtml())
    """
    assert x_video.ndim==4
    assert x_video.shape[1]==1
    fig, axs = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if axis_off: axs.set_axis_off()

    frames = [] 
    for i in range(len(x_video)):
        frames.append([plt.imshow(x_video[i,0], cmap='gray',animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)

    plt.close()
    
    return ani 

def get_img_and_seg(df_fnames, PATH_RESULTS, idx='WT-1-1' ):
    """
    Get the image from its location according to `fnames` and its segmentation
    under 

    Output has shape (T,1,Y,X)
    """
    row = df_fnames.loc[idx]
    fname_img = row['path_file']
    DIR_results = PATH_RESULTS / row['folder']
    fname_results = (DIR_results / row['fname'])

    img_obj=AICSImage(fname_img)
    img=img_obj.data
    seg_obj=AICSImage(fname_results)
    seg=seg_obj.data
    return img, seg, img_obj, seg_obj

def make_img_and_seg_grid(img, seg, img_obj, seg_obj, channel='mito', sz=200, ac_norm=[0,17]):
    """
    Given an image and segmentation

    Args: img_ch and seg_ch shape (T,1,Y,X), the output of `get_img_and_seg`.
    """
    # get the channel
    ch_idx_lookup = pu.get_channel_idx_lookup(img_obj.channel_names)
    ch_idx = ch_idx_lookup[channel]
    img_ch, seg_ch = img[:,ch_idx], seg[:,ch_idx]

    # normalize the input image
    img_ch = aicssegmentation.core.pre_processing_utils.intensity_normalization(img_ch, ac_norm)

    # steps based on the size of the windows.
    ysteps = img_ch.shape[-2]//sz+1
    xsteps = img_ch.shape[-1]//sz+1
    cols = []
    for y in range(ysteps):
        row = []
        for x in range(xsteps):
            yslc, xslc = slice(y*sz,(y+1)*sz), slice(x*sz,(x+1)*sz)
            img_win, seg_win = img_ch[:,:,yslc, xslc], seg_ch[:,:,yslc, xslc]
            gray_bar=np.ones_like(img_win)[:,:,:,:5]*0.5
            row.extend([img_win, gray_bar, seg_win, gray_bar])
        row = np.concatenate(row, 3)
        gray_bar = np.ones_like(row)[:,:,:5,:]*0.5
        cols.extend([row,gray_bar])
    grid = np.concatenate(cols, 2)

    return grid

def get_fname_lookup(path_fname, path_data, path_results=None):
    """
    Get fname-lookup csv files as a dataframe and do extra processing like 
    getting the paths and testing that the group sizes are right.

    Args:
        path_fname: path to the csv file that has lookup info about the dataset.
        path_data: home data dir for images. All paths in the lookup file (the 
            1st arg) are relative to this directory. 
        path_results: homa dir for completed segmentations.
    """
    df_fnames = pd.read_csv(path_fname).set_index('index')
    df_fnames['path_file'] = str(path_data) +"/"+ df_fnames['folder']+"/"+df_fnames['fname']

    for idx, row in df_fnames.iterrows():
        fname = row['path_file']
        if not os.path.exists(fname):
            print(f"WARNING: did not find idx {idx} for file {fname}")

    if path_results is not None: 
        df_fnames['path_seg'] = str(path_results) + "/" + df_fnames['folder'] + "/" + df_fnames['fname']

    return df_fnames

def grid_from_2cols(x1, x2, nrow=10, ncol=8, scale_x2=1):
    """
    Given 2 tensors of images x1, x2, put them in a single imagegrid (by
    calling torchvision.utils.make_grid) where x1[i] is next to x2[i].
    It's useful for plotting an original image and its reconstruction.
    """
    n = nrow*(ncol//2)
    x1, x2 = x1[:n], x2[:n]
    if scale_x2:
        l, u = x2.min(), x2.max()
        x2 = (x2-l)/(u-l)
    # add gray bars to separate the columns
    gray_bar = 0.2*torch.ones((*x1.shape[:3], 2))
    x1 = torch.cat((gray_bar, x1), dim=3)
    x2 = torch.cat((x2, gray_bar), dim=3)


    # merge the images and create a grid
    assert x1.shape==x2.shape
    x = torch.zeros((len(x1)*2, *x1.shape[1:]), dtype=x1.dtype)
    x[0::2] = x1
    x[1::2] = x2
    grid = torchvision.utils.make_grid(x, ncol, padding=0)
    grid = torch.permute(grid, (1,2,0))
    return grid

def grid_from_3cols(x1, x2, x3, scale_vals=1, nrow=10, ncol=6):
    """
    Same as `grid_from_2cols` but for 3 columns.
    """
    n = nrow*(ncol//3)
    x1, x2, x3 = x1[:n], x2[:n], x3[:n]
    if scale_vals:
        x1 = (x1-x1.min()) / (x1.max()-x1.min())
        x2 = (x2-x2.min()) / (x2.max()-x2.min())
        x3 = (x3-x3.min()) / (x3.max()-x3.min())

    assert x1.shape==x2.shape
    x = torch.zeros((len(x1)*3, *x1.shape[1:]), dtype=x1.dtype)
    x[0::3] = x1
    x[1::3] = x2
    x[2::3] = x3
    grid = torchvision.utils.make_grid(x, ncol)
    grid = torch.permute(grid, (1,2,0))
    return grid
