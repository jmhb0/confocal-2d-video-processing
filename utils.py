import matplotlib.pyplot as plt
import matplotlib.animation as animation
from aicsimageio import AICSImage
import pipeline_utils as pu

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

def get_img_and_seg(df_fnames, PATH_RESULTS, idx='WT-1-1', SEG_RUN_NAME="feb2", ):
    """
    Get the image from its location according to `fnames` and its segmentation
    under `SEG_RUN_NAME`

    Output has shape (T,1,Y,X)
    """
    row = df_fnames.loc[idx]
    SEG_RUN_NAME="feb2"
    fname_img = row['path_file']
    DIR_results = PATH_RESULTS / row['folder']
    fname_results = (DIR_results / row['fname'])

    img_obj=AICSImage(fname_img)
    img=img_obj.data
    seg_obj=AICSImage(fname_results)
    seg=seg_obj.data
    return img, seg, img_obj

def make_img_and_seg_grid(img, seg, channel='mito', sz=200, ac_norm=[0,17]):
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


