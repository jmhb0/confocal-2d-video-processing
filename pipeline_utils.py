from aicsimageio import AICSImage
from skimage import measure
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings
import numpy as np
from scipy.stats import norm
import re
import itertools
import torch
import yaml
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
import util_es_smooth_2d 
import importlib
importlib.reload(util_es_smooth_2d)


def read_czi_img(fname, return_aics_object=False, 
    order_channels=['golgi', 'er','lyso', 'mito', 'peroxy'], frame=None):
    """
    TODO: this function sucks. Fix it.
    Read in CZI image, and convert to standard shape: TCYX. 
    Reorder the channels using the codes `order_channels`. The CZI file must 
    have channel names that softly match these labels. 
    order_channels=['golgi', 'er','lyso', 'mito', 'perox']
    Args: 
        frame: single frame to pull out. If None or "all" then get all frames.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_aics = AICSImage(fname)
        if frame is None or frame=="all":
            img = img_aics.get_image_data("TCYX", S=0, Z=0)
        else:
            img = img_aics.get_image_data("CYX", S=0, Z=0, T=frame)
    
    ### code to reorder the channels if they are out of order 
    channels = img_aics.get_channel_names()
    if "T PMT" in channels:
        channels.remove("T PMT")

    assert len(order_channels) <= len(channels), "wrong number of in channels requested"
    channel_lookup = dict(zip(order_channels, range(len(order_channels))))
    re_string = r"(?i)(" + "|".join(order_channels) + ")*"
    
    indx_reorder = np.zeros(len(channels), dtype='int')-1
    for i, channel in enumerate(channels):
        match = re.match(re_string, channel)
        order_channel = match.groups()[0]
        if match is None or order_channel is None:
            raise ValueError(f"CZI channel list could not be matched to "\
                             "order_channels value {channels}")
        new_indx = channel_lookup[order_channel.lower()]
        if i in indx_reorder:
            raise ValueError(f"Repeated channel names in CZI file")
        indx_reorder[new_indx] = i

    img = img[...,indx_reorder,:,:]
    
    if return_aics_object: 
        return img, img_aics
    else:
        return img

def get_channel_idx_lookup(channel_names):
    """
    Given list of channel names, return a dict that looksup the 
    channels that I care about with their canonical names.
    """
    ch_lookup = {}
    # cananocial ordering should put nuclei last
    # put 'er' earlier in the list to peroxy, else it will get confused
    orgs = ['lyso','mito','golgi','peroxy','er','bodipy','residuals','nuclei']
    for i, ch in enumerate(channel_names):
        for org in orgs:
            if org in ch.lower():
                ch_lookup[org]=i
                break
    return ch_lookup

def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data: \
        self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)    


def get_objects_from_segmentation_2d(img, seg, min_pixels_per_object=0):
    """
    Given an original image, `img`, and an instance segmentation mask,
    `seg` with the same shape, pull out lists of objects, where connection is 
    defined wrt 1-connectivity (see skimage.measure.connectivity)
    Returns:
        objects (lst<np.array>): list of 2d object. The surrounding stuff
            to the object is retained. 
        objects_mased (lst<np.array>): list of 2d objects. The surrounding stuff
            is maked out.
        max_dim (lst<int>): list where the ith elemnt is the largest dimnension
            of the array in objects[i].
    Each entry in `objects` and objects_masked` may have different shapes. To 
    get them to be consistent, call `centr_img_to_size`
    """
    # get distinct objects
    labels = measure.label(seg, connectivity=1)

    # remove objects that are smaller than X pixels
    label_vals, cnts = np.unique(labels, return_counts=True)
    label_vals[cnts<min_pixels_per_object]=0

    # check unique labels again
    label_vals, cnts = np.unique(label_vals, return_counts=True)

    # Iterate through labels to get the distinct objects
    objects = []    # variable length patches
    objects_masked = []
    coords = []     # store the first (x,y) coordinate of this bounding box
    for l in label_vals[1:]:
        # create the image mask from the larger mask
        mask = np.zeros(seg.shape)
        mask[labels==l]=1
        img_object = mask*img

        # get object bounds
        yones, xones = np.where(mask==1)
        ymin, ymax = yones.min(), yones.max()
        xmin, xmax = xones.min(), xones.max()
        max_dim=max(ymax-ymin, xmax-xmin)

        # get bounding box around the object, and mask out other stuff
        y_slc = slice(ymin,ymax+1)
        x_slc = slice(xmin,xmax+1)
        img_cropped = img[y_slc, x_slc]
        mask_cropped = mask[y_slc, x_slc]
        img_cropped_and_masked =img_cropped*mask_cropped

        objects.append(img_cropped)
        objects_masked.append(img_cropped_and_masked)
        coords.append([ymin,xmin])

    return objects_masked, objects, coords 

def wrapper_timelapse_get_objects_from_segmentation_2d(img, seg, min_pixels_per_object=0):
    """
    Wrapper around `get_objects_from_segmentation_2d` that calls it for each frame
    """
    pass

def normalize_0_1(img_in, display_range=[0,1], inplace=False):
    """
    Input: numpy array 
    Send image to range [0,1], then truncate to display range. 
    E.g. if display_range=[0,0.5], this function first rescales all images 
    to range [0,1], then all pixels >0.5 are sent to 0.5.
    """
    assert len(display_range)==2
    assert display_range[0]<display_range[1]
    for d in display_range: assert 0<= d <= 1

    img = img_in.copy() if inplace else img_in
    img = (img-img.min()) / (img.max()-img.min())
    img[img<display_range[0]]=display_range[0]
    img[img>display_range[1]]=display_range[1]
    return img


def center_img_to_size(objects_masked_lst, original_img, coords
                    , img_dim=64, verbose=1):
    """
    Take a list of variable-shaped 2d arrays that contain an object, assuming
    that the object occupy the entire image (the bbox is the whole image).
    Put img inside a patch that is shape (img_dim,img_dim) and center.
    Return
        objects (tensor): shape (bs,y,x). bs is the number of elements
            in objects_lst minus those that are filtered for being too big
        scenes (tensor): same shape as objects. Shows same scene but without
            any masking
    """
    assert len(original_img.shape)==2, "Assume 2d image input"
    # filter objects that are too big for the chosen size
    n_objects_before = len(objects_masked_lst)
    filt = [True if np.max(np.shape(o))<=img_dim else False 
                        for o in objects_masked_lst]
    objects_masked_lst = list(itertools.compress(objects_masked_lst, filt))
    coords = list(itertools.compress(coords, filt))
    assert len(objects_masked_lst)==len(coords)
    n_objects = len(objects_masked_lst)
    if verbose>=1:
        print(f"Filtered {n_objects_before-n_objects} objects for being too big")
        print(f"{n_objects} objects found.\nFitting to {img_dim}X{img_dim} frame and centering")

    # create results Tensor
    objects_masked = torch.zeros(n_objects, img_dim, img_dim)
    scenes = torch.zeros(n_objects, img_dim, img_dim)
    # Centre object and put it in the tensor
    for i in range(len(objects_masked_lst)):
        yrange, xrange = objects_masked_lst[i].shape
        ystart, xstart = (img_dim-yrange)//2, (img_dim-xrange)//2
        y_slc = slice(ystart, ystart+yrange)
        x_slc = slice(xstart, xstart+xrange)

        # put the masked object in the center of an empty image
        obj_centred_masked = torch.zeros(img_dim, img_dim)
        obj_centred_masked[y_slc, x_slc] = torch.Tensor(objects_masked_lst[i])
        objects_masked[i] = obj_centred_masked

        # get the full scene around the object from the original image
        ymin, xmin = coords[i]
        ymin -= ystart 
        xmin -= xstart
        y_slc = slice(max(0,ymin), ymin+img_dim)
        x_slc = slice(max(xmin,0), xmin+img_dim)
        scene = torch.Tensor(original_img[y_slc, x_slc])
        # the complicated next line is to hanle the case that scene is not dim
        # img_dim x img_dim because the slice exceeded the image
        scenes[i,0:scene.shape[0], 0:scene.shape[1]] = scene

    return objects_masked, scenes, coords

def _get_range_for_img_norm(img, scaling_param):
    """
    Copied from the definition of `intensity_normalization`
    https://github.com/AllenCell/aics-segmentation/blob/main/aicssegmentation/core/pre_processing_utils.py
    """
    if len(scaling_param)==1:
        strech_min = img.min()
        strech_max = scaling_param[0]
    if len(scaling_param)==2:
        m, s = norm.fit(img.flat)
        strech_min = max(m - scaling_param[0] * s, img.min())
        strech_max = min(m + scaling_param[1] * s, img.max())
    return strech_min, strech_max

def plot_sample_segments_and_scenes(objects, scenes, coords=None, fill_rows=True, 
                                    nrows=10,ncols=8, show_seg_masks=False, show_coords=True):
    """
    TODO: could refactor using make_grid.
    Args:
        fill_rows: if True, then choose the number of rows such that all the objects
            and scenes are included. 
    """
    ## Constants for creating a plt.subplots grid with little spacek:w
    left  = 0.0    # the left side of the subplots of the figure
    right = 1.0    # the right side of the subplots of the figure
    bottom = 0.0   # the bottom of the subplots of the figure
    top = 1.0      # the top of the subplots of the figure
    wspace = 0.03  # the amount of width reserved for blank space between subplots
    hspace = 0.03  # the amount of height reserved for white space between subplots

    if show_coords: assert coords is not None, "must provide `coords "\
                                    "or set `show_coords=False`."
    figsize=np.array([ncols,nrows])*2
    f, axs = plt.subplots(nrows,ncols,figsize=figsize)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for i in range(nrows):
        for j in (range(ncols//2)):
            # plot scene
            axs[i, j*2].imshow(scenes[i*(ncols//2)+j], cmap='gray')
            axs[i, j*2].set_axis_off()
            # plot objects
            axs[i, j*2+1].imshow(objects[i*(ncols//2)+j], cmap='gray')
            axs[i, j*2+1].set_axis_off()
            if show_coords:
                axs[i, j*2+1].text(0,2,str(coords[i*(ncols//2)+j]), color='w')

    return f, axs

def plot_seg_distinct_objects(seg, return_img=False, figsize=(15,15)):
    """
    Take a segmentation and plot distinct objects in distinct colors. 
    This is to easily differentiate objects that are close (or touching) but
    are distinct objects according to the segmentation. 
    
    Args: 
        seg: Integer image with shape is TYX with T==1, or YX. 
            0 is background. If all other values are 1's, 
            then distinct objects are connected components (defined with 
            1-connectivity). If other values are integers, then each integer is
            a distinct object. It's assumed that this form is the output of 
            skimage.measure.label. 
            Shape is ()
        return_img: if True, return only the array where the label values are 
            replaced with a float in range [0,1] that can be used by plt.imshow
            to do the plotting
    """
    seg1 = seg.copy()
    if len(seg1.shape)==2: seg1 = seg[None]
    elif len(seg1.shape)==2: assert seg1.shape[0]==1
    else: raise ValueError("Wrong shape")
    
    if len(np.unique(seg))==2:
        labels = measure.label(seg1, connectivity=1)
    else: 
        labels = seg1
    cmap = plt.cm.nipy_spectral

    labels_max = (labels.max())
    # generate random numbers in the range [0.1,1] because we will map objects 
    # randomly to the range of plt.cm.nipy_spectral, and the start of the range is black
    rand_mapping =  np.random.rand(labels_max) * 0.9 +0.1
    label_color_mapping = dict(zip(np.arange(1,labels_max+1), rand_mapping))
    label_color_mapping[0] = 0.0
    seg1 = np.vectorize(label_color_mapping.get)(labels)
    if return_img: 
        return seg1
    
    f, axs = plt.subplots(figsize=figsize)
    plt.imshow(seg1[0], cmap=plt.cm.nipy_spectral)
    plt.close()
    
    return f, axs


def compare_segmentation_windows(img_in, seg, window_sz=200, stride=200, min_pixels=10, 
       ac_norm_params=None, mm_norm_params=None, gs_smooth_params=None, es_smooth_params=None,
        color_instances=True, figsize_scale=10, verbose=1):
    """
    View all the zoomed in siubsections of the original image, 'windows'm and 
    compare their segmentations. The input is the original image, and the seg
    mask (either binary or instance-level). It scans over the whole image, and 
    only plots windows with >window_sz pixels that are part of an object. Also
    returns 2 original images that mask out the areas not plotted because they 
    have no objects.
    Recommended: control the contrast/brightness of the original image using 
    `display_range`, since features usually aren't visible with the default 
    contrast. Use, for example, ImageJ to view the image. 
    
    It's allowed to have both Gaussian and ES smoothing parameters passed in. 
    
    The img_in should not get modified in place.

    Args:
        img_in: np.array shape YX, that is the original image.
        seg: np.array shape YX segmentation image, where 0 is background and 
            distinct integers 1,2,3 etc are distinct objects. If binary (0,1) 
            then we plot a black-wite seg mask. If more than one class, we plot
            each object with a distinct colour by assigning each object a 
            random colour in the colormap plt.cm.nipy_spectral. 
        window_sz: int, num pixels that is the width and height of the image. 
        stride: int, how far to move the window along y and x for each new img. 
        min_pixels: int, if number of pixels in window smaller, don't plot.
        ac_norm_params: list[list[float,float]], for example [[0,30],[0,10]].
        mm_norm_oarams: list[list[float]], for example [[1000],[20000]]
        figsize_scale: int, multiplier to make the figure bigger. The aspect
            ratio is fixed. 

    Returns:
        f: matplotlib figure where col 0 is original image and col 1 is the seg
            mask. There is a row per window that has num_pixels>min_pixels that 
            are non-background.                     
        noshow_mask: np.array with same shape as img where windows that are 
            included in `f` are masked out by being set to display_range[1], the 
            max intensity. Users can run plt.imshow(), we can view the parts of 
            the image not included in `f` if we want. 
        show_mask: np.array with same shape as img. It is the inverse of 
            `noshow_mask`, so only windows in `f` are included. 
    """ 
    pass # deleted the old implementation. Keeping it around to copy the documentation


def _windowing_create_imgs_contrast_smooth_levels(img_in, ac_norm_params=None, mm_norm_params=None
                                        , es_smooth_params=None, gs_smooth_params=None
                                      , verbose=1):
    """
    Util function for `compare_segmentation_windows`. Create a list, where each element is an 
    image that will be used in one of the display columns. These will be due to contrast 
    adjustment, and/or smoothing. Create a second list that is the title for this plot. 
    
    If no norm or smooth params are input (None) then return an empty lists. If norm is 
    included, but no smoothing, create one image where the smoothing is Gaussian with sigma
    of 0.1 (effectively no smoothing). If smoothing is included, but no norming, create AC
    norm of [0,50] which is effectively no norming. 
    
    We generate an image for each combo of contrast setting and smoothness setting. 
    """
    img_tmp = img_in.copy()
    min_pixel_original, max_pixel_original = img_in.min(), img_in.max()
    if verbose: 
        print(f"Image range [{min_pixel_original},{max_pixel_original}]")

    # choose the norm parameters for contrast images. If None, choose a very large AC norm thing 
    # that is effectively no bound
    norm_params = (ac_norm_params if ac_norm_params else []) \
                + (mm_norm_params if mm_norm_params else []) 

    # likewise, choose the smoothing parameters. If None choose gaussian parameters [1]
    smooth_params = (gs_smooth_params if gs_smooth_params else []) \
                  + (es_smooth_params if es_smooth_params else [])
    
    # handle the cases where there are missing params (explained in docstring)
    if len(norm_params)+len(smooth_params)==0:
        return [], []
    if len(norm_params)==0: norm_params.append([0,50])   # effectively no adjustment
    if len(smooth_params)==0: smooth_params.append(0.1) # Effectively no smoothing

    # create the images list 
    imgs_contrast_smooth_levels, imgs_labels = [], []
    if verbose: print(f"Creating range of ac or mm norm images {norm_params}")
    if verbose: print(f"Creating range of smoothed images {smooth_params}")
        
    for i, norm_param in enumerate(norm_params):
        img_norm = img_in.copy()
        img_norm = intensity_normalization(img_norm, scaling_param=norm_param)
        norm_label_code="MM" if len(norm_param)==1 else "AC"
        for j, smooth_param in enumerate(smooth_params):
            # copy img_norm unless it's the last one to smooth, then we don't need to. This saves expensive copies
            if j==len(smooth_params)-1: img_smooth = img_norm
            else: img_smooth = img_norm.copy()

            # do the smoothing, depending on whether it's gaussian or edge-preserving
            if type(smooth_param) in [float, int]: 
                smoooth_label_code = "GS"
                img_smooth = image_smoothing_gaussian_slice_by_slice(img_smooth, smooth_param)
            elif type(smooth_param) is dict: 
                smoooth_label_code = "ES"
                img_smooth = util_es_smooth_2d.anisodiff(img_smooth, **smooth_param)
            else: 
                raise ValueError("Invalid smoothing value")

            # save the images and the labels 
            imgs_contrast_smooth_levels.append(img_smooth)
            label = f"PreProcess: Norm {norm_label_code}: {norm_param} -- smooth {smoooth_label_code}: {smooth_param}"
            imgs_labels.append(label)
            
        if verbose: 
            min_pixel, max_pixel = _get_range_for_img_norm(img_norm, scaling_param=norm_param)
            print(f"\tImg norm {norm_param}, truncates range [{min_pixel:.2f},{max_pixel:.2f}]")
    return imgs_contrast_smooth_levels, imgs_labels

def _windowing_prepare_comparison_images(img, seg, ac_norm_params=None, mm_norm_params=None,
                              gs_smooth_params=None, es_smooth_params=None, 
                              imgs_inter=None, inter_img_param_title=True, verbose=1):
    """
    Create 2 sets of comparison images, that can then be plotted in a grid. 
    1st set (row 1) will be original image, then a set of preprocessing images
    2nd set (row 2) will be the intermediate values of the image. 
    Args:
        Preprocessing:
            
    """
    imgs_preprocess, labels_set1 = _windowing_create_imgs_contrast_smooth_levels(
                        img_in=img, ac_norm_params=ac_norm_params,mm_norm_params=mm_norm_params, 
                        gs_smooth_params=gs_smooth_params, es_smooth_params=es_smooth_params, 
                        verbose=verbose)
    
#     for im in imgs_preprocess:
#         print(im.shape)

    # set 1 is the original, the preprocessing, and the segmentation mask
    imgs_set1 = imgs_preprocess
    imgs_set1.insert(0, img)
    imgs_set1.insert(1, seg)
    labels_set1.insert(0, 'original')
    labels_set1.insert(1,'segmentation')

    # prepare set 2 which is the intermediate values
    imgs_set2, labels_set2 = [], []
    if imgs_inter:
        for func_name, _ in imgs_inter.items():
            imgs_set2.append(imgs_inter[func_name]['img'])
            label = f"**Inter: {func_name}"
            if inter_img_param_title: label += f", params: {imgs_inter[func_name]['params']}"
            labels_set2.append(label)
    else: pass

    return imgs_set1, labels_set1, imgs_set2, labels_set2

def _windowing_get_included_slices(img_in, seg, window_sz=200, stride=200, min_pixels=30):
    """
    Util func to get the slices of the seg mask that we are going to include in analysis,
    ignoring those images that are almost entirely black. 
    
    Iterate over the windows as specified by window_sz and stride, recoring those slices 
    that we will include. 
    """
    assert seg.ndim==2 and img_in.ndim==2
    noshow_mask,  show_mask = img_in.copy(), img_in.copy()
    noshow_slices, show_slices = [], []
    # get some important stats
    min_pixel_original, max_pixel_original = img_in.min(), img_in.max()
    n_labels = len(np.unique(seg))  # check if binary mask or instance seg labels.
    ylen, xlen = seg.shape
   
    # iterate over all the windows. Update the masks, and save the slices that 
    # plotting later. 
    xsteps = (xlen-stride) // stride + 2
    ysteps = (ylen-stride) // stride + 2
    for y in range(xsteps):
        for x in range(xsteps):
            slicex = slice(x*stride, (x+1)*stride)
            slicey = slice(y*stride, (y+1)*stride)
            seg_tmp = seg[slicey, slicex]
            nonzero_pix = seg_tmp[seg_tmp!=0].shape[0]
            if nonzero_pix>=min_pixels:
                show_slices.append([slicey,slicex])
                noshow_mask[slicey, slicex] = max_pixel_original
            else: 
                noshow_slices.append([slicey,slicex])
                show_mask[slicey, slicex] = max_pixel_original

    return show_slices, noshow_slices, noshow_mask, show_mask

def _windowing_display_windows(imgs_set1, labels_set1, imgs_set2, labels_set2,
                            slices, max_cols=4, figsize_mult=7, do_plot_dividing_lines=True):
    """
    Display the evaluation plots, taking imgs_set1, and imgs_set2
    """
    n_slices, n_set1, n_set2 = len(slices), len(imgs_set1), len(imgs_set2)
    nrows_set1 = n_set1 // max_cols + (n_set1%max_cols!=0)
    nrows_set2 = n_set2 // max_cols + (n_set2%max_cols!=0)
    nrows_p_slice = nrows_set1+nrows_set2

    nrows = n_slices*nrows_p_slice
    ncols_set1, ncols_set2 = min(n_set1, max_cols), min(n_set2, max_cols)
    ncols_max = max(ncols_set1, ncols_set2)

    ## Constants for creating a plt.subplots grid with little spacek:
    left, right  = 0.0, 1.0    # the left & right side of the subplots of the figure
    bottom, top = 0.0, 1.0   # the bottom & top of the subplots of the figure
    wspace, hspace = 0.03, 0.2  # width & height reserved for blank space between subplots

    figsize = figsize_mult*(np.array([ncols_max, nrows]))
    f = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for k in range(n_slices):
        slicey, slicex = slices[k]

        # do set 1
        for i, img in enumerate(imgs_set1):
            plt.subplot(nrows, ncols_set1, (k*nrows_p_slice*ncols_set1)+(i+1))
            # plot seg mask stuff
            if i==1:
                seg_tmp = plot_seg_distinct_objects(img[0,slicey, slicex], return_img=True)[0]
                plt.imshow(seg_tmp, cmap=plt.cm.nipy_spectral)
                plt.title("segmentation")
            # plot non-seg-mask stuff
            else:
                plt.imshow(img[0,slicey,slicex], cmap='gray')
                plt.title(labels_set1[i])
            # title the original
            if i==0:
                plt.title(f"Original: {slices[k]}")

        # do set 2
        for i, img in enumerate(imgs_set2):
            plt.subplot(nrows, ncols_set2, ((k*nrows_p_slice+nrows_set1)*ncols_set2)+(i+1))
            title = labels_set2[i]
            # special case of spot or filament filte
            if 'F2_precut' in title:
                plt.imshow(imgs_set2[i][0,slicey,slicex], cmap='gray', vmin=0, vmax=1)
                plt.colorbar()
            if 'S2_precut' in title:
                plt.imshow(imgs_set2[i][0,slicey,slicex], cmap='gray', vmin=0, vmax=0.4)
                plt.colorbar()
            else:
                plt.imshow(imgs_set2[i][0,slicey,slicex], cmap='gray')
            plt.title(labels_set2[i])
    
    if do_plot_dividing_lines:
        ys = np.linspace(0,1,n_slices+1, endpoint=True)[1:-1]
        for y in ys:
            line = plt.Line2D([0,1],[y,y], transform=f.transFigure, color="black")
            f.add_artist(line)


    plt.close()
    return f

def compare_seg_windows(img, seg, imgs_inter=None, window_sz=200, stride=200, min_pixels=30, 
                        ac_norm_params=None, mm_norm_params=None, gs_smooth_params=None, es_smooth_params=None,
                        max_display_cols=4, figsize_mult=7, do_plot_dividing_lines=True,
                        verbose=0):
    """
    Main function for doing a scan over the windows of the image, by calling a series of 
    util functions
    
    Args:
        imgs_inter: OrderedDict of intermediate images with their labels.
    """
    assert img.ndim==3 and seg.ndim==3
    
    imgs_set1, labels_set1, imgs_set2, labels_set2 \
                = _windowing_prepare_comparison_images(img, seg, 
                                    ac_norm_params=ac_norm_params, mm_norm_params=mm_norm_params,
                                    gs_smooth_params=gs_smooth_params, es_smooth_params=es_smooth_params,
                                    imgs_inter = imgs_inter, verbose=verbose)

    show_slices, noshow_slices, noshow_mask, show_mask \
            = _windowing_get_included_slices(img[0], seg[0], window_sz=window_sz, stride=stride, min_pixels=min_pixels)

    f = _windowing_display_windows(imgs_set1, labels_set1, imgs_set2, labels_set2,max_cols=max_display_cols,
                                slices=show_slices, figsize_mult=figsize_mult, do_plot_dividing_lines=do_plot_dividing_lines) 

    return f, show_mask, noshow_mask


