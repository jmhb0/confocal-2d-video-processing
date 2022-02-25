import itertools
import skimage
import torch
import numpy as np
import pipeline_utils as pu
from aicsimageio import AICSImage
from aicssegmentation.core.pre_processing_utils import intensity_normalization
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from skimage import measure
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_objects_from_segmentation(img, seg, min_pixels_per_object=0, connectivity=1, verbose=0):
    """
    Given an original image, `img`, and an instance segmentation mask,
    `seg` with the same shape, get ob.
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
    labels = skimage.measure.label(seg, connectivity=connectivity)

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
        # put the mask back to range [0,1]
        img_cropped_and_masked[img_cropped_and_masked>0]=1

        objects.append(img_cropped)
        objects_masked.append(img_cropped_and_masked)
        coords.append([ymin,xmin])

    coords = torch.IntTensor(coords) # other objects cannot be Tensors - different dims
    return objects_masked, objects, coords 

def center_img_to_size(idx, fname_image, fname_seg, objects_masked_lst, original_img, seg_img, coords
                    , img_dim=64, verbose=1, filter_big_things=0):
    """
    """
    # identify big objects
    n_objects = len(objects_masked_lst)
    idxs_big_objects =  np.array([True if np.max(np.shape(o))>img_dim else False for o in objects_masked_lst])
    n_big_objects = np.sum(idxs_big_objects)

    # if filtering big stuff, remove these parts from the original list
    if filter_big_things:
        n_objects = n_objects-n_big_objects
        objects_masked_lst = list(itertools.compress(objects_masked_lst, filt))
        coords = coords[filt]
        assert len(objects_masked_lst)==len(coords)
        idxs_big_objects = np.zeros(n_objects) # all False

    # arrays to hold objects
    objects_masked = torch.zeros(n_objects, img_dim, img_dim)
    scenes = torch.zeros(n_objects, img_dim, img_dim)
    scenes_segmented = torch.zeros(n_objects, img_dim, img_dim)
    all_metadata=[]

    print(n_objects)
    for i in range(n_objects):
        metadata=dict()
        metadata['idx']=idx
        metadata['fname_image']=fname_image
        metadata['fname_seg']=fname_seg

        # coordinate positions
        ymin, xmin = coords[i]
        metadata['coords']=coords[i]

        # size of the cropped object
        ysz, xsz = objects_masked_lst[i].shape
        metadata['shape_original'] = objects_masked_lst[i].shape
        metadata['is_big'] = idxs_big_objects[i]

        # We'll place the cropped image inside a fixed size image shape (img_dim, img_dim)
        # Find the slices where the crop will be placed

        # first the regular case
        if not idxs_big_objects[i]:
            ystart, xstart = (img_dim-ysz)//2, (img_dim-xsz)//2
            y_slc = slice(ystart, ystart+ysz)
            x_slc = slice(xstart, xstart+xsz)
            objects_masked[i, y_slc, x_slc] = torch.Tensor(objects_masked_lst[i])

            # the scene parameters
            xmin -= xstart
            ymin -= ystart

        # now the too-big object case
        else:
            # for big objects, pull out the scene
            y_slc_big_scene, x_slc_big_scene = slice(ymin, ymin+ysz), slice(xmin, xmin+xsz)
            big_obj_scene = original_img[y_slc_big_scene, x_slc_big_scene]

            # y too big, x fine
            if ysz>img_dim and xsz<=img_dim:
                # the ystart equation is inverted
                ystart, xstart = (ysz-img_dim)//2, (img_dim-xsz)//2 # one or both will be negative
                y_slc_obj = slice(ystart, ystart+img_dim)
                x_slc = slice(xstart, xstart+xsz)
                objects_masked[i, :, x_slc] = torch.Tensor(objects_masked_lst[i][y_slc_obj,:])

                # scene params
                xmin -= xstart
                ymin += ystart

            # x too big, y fine
            elif ysz<=img_dim and xsz>img_dim:
                ystart, xstart = (img_dim-ysz)//2, (xsz-img_dim)//2 # one or both will be negative
                y_slc = slice(ystart, ystart+ysz)
                x_slc_obj = slice(xstart, xstart+img_dim)
                objects_masked[i, y_slc, :] = torch.Tensor(objects_masked_lst[i][:,x_slc_obj])
                xmin += xstart
                ymin -= ystart

            elif ysz>img_dim and xsz>img_dim:
                ystart, xstart = (ysz-img_dim)//2, (xsz-img_dim)//2 # one or both will be negative
                y_slc_obj = slice(ystart, ystart+img_dim)
                x_slc_obj = slice(xstart, xstart+img_dim)
                objects_masked[i, :, :] = torch.Tensor(objects_masked_lst[i][y_slc_obj,x_slc_obj])
                xmin += xstart
                ymin += ystart

            else:
                raise ValueError("implementation error")

            metadata['big_obj_data'] = [
                i,
                objects_masked_lst[i],
                big_obj_scene,
                coords[i],
            ]

        # produce the scene
        y_slc = slice(max(0,ymin), ymin+img_dim)
        x_slc = slice(max(xmin,0), xmin+img_dim)
        scene = torch.Tensor(original_img[y_slc, x_slc].astype(np.float16))
        scene_seg = torch.Tensor(seg_img[y_slc, x_slc].astype(np.float16))
        scenes[i, 0:scene.shape[0], 0:scene.shape[1]] = scene
        scenes_segmented[i, 0:scene.shape[0], 0:scene.shape[1]] = scene_seg
        
        all_metadata.append(metadata)


    objects_masked = objects_masked.unsqueeze(1)
    scenes = scenes.unsqueeze(1)
    scenes_segmented = scenes_segmented.unsqueeze(1)

    return objects_masked, scenes, scenes_segmented, all_metadata

    """
    # the old version of this funciton **
    Take a list of variable-shaped 2d arrays that contain an object, assuming
    that the object occupy the entire image (the bbox is the whole image).
    Put img inside a patch that is shape (img_dim,img_dim) and center.
    Return
        objects (tensor): shape (bs,y,x). bs is the number of elements
            in objects_lst minus those that are filtered for being too big
        scenes (tensor): same shape as objects. Shows same scene but without
            any masking. Do not do any pixel scaling. 
    """
    """
    # filter objects that are too big for the chosen size
    n_objects_before = len(objects_masked_lst)
    idxs_big_objects =  [True if np.max(np.shape(o))<=img_dim else False 
            for o in objects_masked_lst]
    print(f"{len(idxs_big_objects)} objects that are too big")
    filt = [True if np.max(np.shape(o))<=img_dim else False 
                        for o in objects_masked_lst]
    objects_masked_lst = list(itertools.compress(objects_masked_lst, filt))
    coords = list(itertools.compress(coords, filt))
    assert len(objects_masked_lst)==len(coords)
    n_objects = len(objects_masked_lst)
    n_too_big = n_objects_before-n_objects
    if verbose>=1:
        print(f"Filtered {n_too_big} objects for being too big")
        print(f"{n_objects} objects found.\nFitting to {img_dim}X{img_dim} frame and centering")

    # create results Tensor
    n_objects=len(objects_masked_lst)
    objects_masked = torch.zeros(n_objects, img_dim, img_dim)
    scenes = torch.zeros(n_objects, img_dim, img_dim)

    # Centre object and put it in the tensor
    for i in range(n_objects):
        yrange, xrange = objects_masked_lst[i].shape

        # handle the regular case 'object is not too big'
        if not idxs_big_objects[i]:
            ystart, xstart = (img_dim-yrange)//2, (img_dim-xrange)//2
            y_slc = slice(ystart, ystart+yrange)
            x_slc = slice(xstart, xstart+xrange)

            # put the masked object in the center of an empty image
            obj_centred_masked = torch.zeros(img_dim, img_dim)
            obj_centred_masked[y_slc, x_slc] = torch.Tensor(objects_masked_lst[i])
            objects_masked[i] = obj_centred_masked
        else: 
            # in this case the image is bigger than 
            pass


        # get the full scene around the object from the original image
        ymin, xmin = coords[i]
        ymin -= ystart 
        xmin -= xstart
        y_slc = slice(max(0,ymin), ymin+img_dim)
        x_slc = slice(max(xmin,0), xmin+img_dim)
        scene = torch.Tensor(original_img[y_slc, x_slc].astype(np.float16))

        # the complicated next line is to hanle the case that scene is not dim
        # img_dim x img_dim because the slice exceeded the image
        scenes[i,0:scene.shape[0], 0:scene.shape[1]] = scene
    
    coords = torch.Tensor(coords)
    return objects_masked[:,None], scenes[:,None], coords, n_too_big
    """  

def build_dataset_objects_and_scenes(df_fnames, PATH_RESULTS, channel="mito", frame=0, verbose=1, filter_big_things=0):
    """
    Buid arrays that will be the dataset of segmented objects. 

    Calls: 
        get_objects_from_segmentation (this module)
        sp.center_img_to_size (this modules)
    Args:
        df_fnames (pd.DataFrame) having index 'idx', and keys `path_seg`, `path_file`, `folder`
    """
    all_objects, all_scenes, all_scenes_segmented, all_metadata = [],[],[],[]
    if verbose:
        print(f"Working on {len(df_fnames)} objects")
    for num, (i, row) in enumerate(df_fnames.iterrows()):
        idx = row.name
        if verbose: print(idx)

        DIR_seg = PATH_RESULTS / row['folder']
        fname_seg = row['path_seg']
        fname_image = row['path_file']

        seg_obj=AICSImage(fname_seg)
        img_obj=AICSImage(fname_image)
        ch_idx_lookup = pu.get_channel_idx_lookup(img_obj.channel_names)
        ch_idx = ch_idx_lookup[channel]

        img_ch=img_obj.get_image_data("TZYX",C=ch_idx)
        seg_ch=seg_obj.data

        seg_ch=seg_ch[:,ch_idx]

        objects_masked_original, objects, coords = get_objects_from_segmentation(img_ch[frame,0], seg_ch[frame,0],
                                                                       connectivity=1, verbose=verbose)
        objects_masked, scenes, scenes_segmented, metadata = center_img_to_size(idx, fname_image, fname_seg, objects_masked_original, original_img=img_ch[frame,0], seg_img=seg_ch[frame,0],
                                                                         coords=coords, img_dim=64, verbose=1, filter_big_things=filter_big_things,)

        labels = [idx]*len(objects_masked)

        all_objects.append(objects_masked)
        all_scenes.append(scenes)
        all_scenes_segmented.append(scenes_segmented)
        all_metadata.append(metadata)

        if verbose: print(f"\t{len(objects_masked)} objects")

    all_objects=torch.cat(all_objects)
    all_scenes=torch.cat(all_scenes)
    all_scenes_segmented=torch.cat(all_scenes_segmented)
    all_metadata = all_metadata # pass

    return all_objects, all_scenes, all_scenes_segmented, all_metadata

def whole_cell_combine_channels(fname, img, merge_channels=['lyso','mito','golgi','peroxy','er',]):
    """ 
    Given an image with file `fname` and image array `img`, get the channels 
    specified in `merge_channels`, do AC normalisation ad then take the mean over them. 
    This is  a subroutine for generating a whole-cell segmentation mask. 
    Args 
        fname (str): filename of the czi file. Is needed to find the channel lookup list. 
        img (np.array): shape (T,C,Y,X). By default we take the first frame, T=0. 
        merge_chanells (List[str]); list of channels to merge. 
    Returns:
        img_sum (np.array): shape (1,Y,X). 
    """
    assert img.ndim==4 
    img_obj = AICSImage(fname)
    ch_idx_lookup = pu.get_channel_idx_lookup(img_obj.channel_names)
    # get the channel idxs we want to merge
    ch_idxs = [ch_idx_lookup[ch] for ch in merge_channels]
    # put the images to merge in a list
    img_set = [img[[0],ch_idx] for ch_idx in ch_idxs]
    # autocontrast normalize the images 
    img_set = [intensity_normalization(img, [0,20]) for img in img_set]
    # sum the images 
    img_sum = np.mean( np.concatenate([im for im in img_set]), axis=0, keepdims=True)
    return img_sum 
def whole_cell_smooth_and_mask(img_sum, sigma=15, threshold_sensitivity=5):
    """
    """
    img_sum_blurred=image_smoothing_gaussian_slice_by_slice(img_sum, sigma=sigma)
    upper = img_sum_blurred.max()    
    mask = np.zeros_like(img_sum_blurred)
    mask[img_sum_blurred>upper/threshold_sensitivity]=1
    return img_sum_blurred, mask
def whole_cell_choose_central_big_cell(mask, threhold_cell_obj_size=10000):
    """
    Given a mask, pick a single object to be the final cell mask. 
    It separates the object according to connected components, filters out 
    all objects smaller than threhold_cell_obj_size pixels, and then chooses 
    the object with the centroid closest to the object center 
    Args: 
        mask (np.array): shape (1,H,W)
    Returns: 
        mask_cell
    """
    assert mask.ndim==3
    label_img = measure.label(mask[0])

    uniq_labels, label_sizes = np.unique(label_img, return_counts=True)
    # get the labels of the objects that are above the threshold
    big_objects = uniq_labels[label_sizes>=threhold_cell_obj_size][1:] # the [1:] part removes the background 
    # update the label image to only include 'big objects' 
    label_img[~np.isin(label_img, big_objects)]=0
    # get the centroids of all the labels that are not 0 
    # (Bug risk: the index of the res objects is not the same index as what's in the label_img array #
    res=measure.regionprops(label_img)
    centroids = np.array([r['centroid'] for r in res])
    # compute distance of the things to the img center 
    img_center = np.array(label_img.shape)//2
    dists = pairwise_distances(img_center[None,:], centroids)
    # identify the object closest to the center 
    argmin = np.argmin(dists) # indexes the 
    idx_object = big_objects[argmin]

    # mask out the one final cell object 
    mask_cell = label_img.copy()
    mask_cell[label_img!=idx_object]=0

    return mask_cell[None,:]

def whole_cell_segment(fname, img, sigma=10, threshold_sensitivity=10, threhold_cell_obj_size=10000,
                      merge_channels = ['lyso','mito','golgi','peroxy','er']):
    """
    Given a multichannel single-frame image, identify the segmentatiom mask of the central cell.
    Also return the intermediate results.
    Args:
        fname (str):  filename
        img (np.array): shape (C,H,W), multichannel image.
    Returns:
        img_sum (np.array): shape (1,H,W) summed image over the channels identified in merge_channels.
        img_sum_blurred (np.array): shape (1,H,W) summed image after Gaussian blur
        mask (np.array): shape (1,H,W) seg mask after gaussian blur.
        mask_cell (np.array): shape (1,H,W) mask with only the central cell.
    """
    img_sum=whole_cell_combine_channels(fname, img, merge_channels=merge_channels)
    img_sum_blurred, mask = whole_cell_smooth_and_mask(img_sum, sigma=sigma, threshold_sensitivity=threshold_sensitivity)
    mask_cell = whole_cell_choose_central_big_cell(mask, threhold_cell_obj_size=10000)
    return img_sum, img_sum_blurred , mask, mask_cell

def whole_cell_segment_eval(*args, title="", **kwargs):
    """
    Wrapper around `whole_cell_segment` that also plots the results for eval purposes.
    """
    img_sum, img_sum_blurred , mask, mask_cell = whole_cell_segment(*args, **kwargs)
    f, axs = plt.subplots(1,4, figsize=(45,9))
    axs[0].imshow(img_sum[0], cmap='gray')
    axs[0].set(title=title)
    axs[1].imshow(img_sum_blurred[0], cmap='gray')
    axs[2].imshow(mask[0], cmap='gray')
    axs[3].imshow(mask_cell[0], cmap='gray')
    plt.close()
    return f, (img_sum, img_sum_blurred , mask, mask_cell)

def whole_cell_segment_eval_make_pdf(idxs, fnames, fname_eval):
    """
    For a list  of fnames, seg

    Note; right now it just takes the default params for whole_cell_segment()
    so this function would have to be updated to fix that if the params were to 
    be changed. 
    """
    print(f"Putting pdf in {fname_eval}")
    print(f"{len(idxs)} files")
    with PdfPages(fname_eval) as pdf:
        for i in range(len(idxs)):
            # get the indx, fname, and the image
            idx=idxs[i]
            print(idx, end=", ")
            fname=fnames[i]
            img_obj = AICSImage(fname)
            img = img_obj.data

            # get the central frame only
            frame = img.shape[0]//2
            img = img[[frame],:,0]

            # plot out the segmentation
            f, _ =whole_cell_segment_eval(fname, img, title=idx)
            pdf.savefig(f)
