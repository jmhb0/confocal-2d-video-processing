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

def get_objects_from_segmentation(img, seg, ch_idx, frame, 
        min_pixels_per_object=0, connectivity=1, verbose=0):
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
    img = img[frame, ch_idx, 0]
    seg = seg[frame, ch_idx, 0]
    assert img.ndim==2 and seg.ndim==2

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
        if l>1000:
            raise ValueError(f"More than {l} objects counted. Probably an error")
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


def center_img_to_size(idx, fname_image, fname_seg, objects_masked_lst, coords,  
        original_img, seg_img, obj_ch_idx, frame, img_dim=64, verbose=1, filter_big_things=0):
    """
    obj_ch_idx is the index of the image that is centered
    """
    # check the seg mask really is 1s and zeros only
    assert np.all((seg_img==0)|(seg_img==1)), "Seg masks should be 0s and 1s"

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

    for i in range(n_objects):
        metadata=dict()
        metadata['idx']=idx
        metadata['fname_image']=fname_image
        metadata['fname_seg']=fname_seg

        # size of the cropped object
        ysz, xsz = objects_masked_lst[i].shape
        metadata['shape_original'] = objects_masked_lst[i].shape
        metadata['is_big'] = idxs_big_objects[i]
        metadata['area'] = np.sum(objects_masked_lst[i])

        # coordinate positions
        ymin, xmin = coords[i]
        ymin, xmin = ymin.item(), xmin.item()
        metadata['coords']=[ymin, xmin]
        metadata['coords_center']=[ymin+ysz//2, xmin+xsz//2]

        ### We'll place the cropped image inside a fixed size image shape (img_dim, img_dim)
        ### Find the slices where the crop will be placed

        # first the regular case: the object fits inside img_dim
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
            big_obj_scene = original_img[frame, obj_ch_idx, 0, y_slc_big_scene, x_slc_big_scene]

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
        scene = torch.Tensor(original_img[frame, obj_ch_idx, 0, y_slc, x_slc].astype(np.float16))
        seg_img_scene=seg_img[frame, obj_ch_idx, 0]
        scene_seg = torch.Tensor(seg_img[frame,obj_ch_idx,0,y_slc, x_slc].astype(np.float16))
        scenes[i, 0:scene.shape[0], 0:scene.shape[1]] = scene
        scenes_segmented[i, 0:scene.shape[0], 0:scene.shape[1]] = scene_seg
        
        # add metadata for this object to the list
        all_metadata.append(metadata)


    objects_masked = objects_masked.unsqueeze(1)
    scenes = scenes.unsqueeze(1)
    scenes_segmented = scenes_segmented.unsqueeze(1)

    return objects_masked, scenes, scenes_segmented, all_metadata


def get_img_channel_contexts(img, seg, frame, save_ch_idxs, coords_center, save_img=1, img_dim=128):
    """
    Given an array of coordinate centers, pull out the image that is centered at those 
    coordinates with size ìmg_dim` from the image `img` and segmentation `seg`. 
    Use the 
    
    Args:
        save_ch_idxs (list): indexes in the channel dimension to save. 
        save_img (bool): if True then also save the original image

    Returns: 
        object_contexts (torch.Tensor)

    """
    assert img.ndim==5 and seg.ndim==5
    assert img.shape==seg.shape
    ymax, xmax = img.shape[-2:]
    n_objs=len(coords_center)
    n_channels=len(save_ch_idxs)
    contexts_seg = torch.zeros(n_objs, n_channels, img_dim, img_dim)
    contexts_img = torch.zeros(n_objs, n_channels, img_dim, img_dim)
    
    for i in range(len(coords_center)):
        yc, xc = coords_center[i]

        y_slc = slice(max(0,yc-img_dim//2),  yc+img_dim//2)
        x_slc = slice(max(0,xc-img_dim//2),  xc+img_dim//2)
        
        cutout_seg = torch.Tensor(seg[frame,save_ch_idxs,0,y_slc,x_slc].astype(np.int16))

        # the cutout seg may actually be smaller if the slice goes beyond the center. Get those params
        # (this part doesn't apply for a majority of objects - only for objects near image border)
        y_actual, x_actual = cutout_seg.shape[-2:]
        if y_actual==img_dim:
            y_slc_left=slice(0,img_dim)
        else: 
            y_slc_left = slice((img_dim-y_actual)//2, img_dim - (img_dim-y_actual+1)//2 )
        if x_actual==img_dim:
            x_slc_left=slice(0,img_dim)
        else:
            x_slc_left = slice((img_dim-x_actual)//2, img_dim - (img_dim-x_actual+1)//2 )

        contexts_seg[i, :, y_slc_left, x_slc_left] = cutout_seg
        
        if save_img:
            cutout_img = torch.Tensor(img[frame,save_ch_idxs,0,y_slc,x_slc].astype(np.float32))
            contexts_img[i, :, y_slc_left, x_slc_left] = cutout_img

    return contexts_seg, contexts_img 

def build_dataset_objects_and_scenes(df_fnames, PATH_RESULTS, channel="mito", frame=0, 
        verbose=1, filter_big_things=0, return_all_channel_segs=0,
        context_kwargs=dict(do=1, img_dim=128, save_img=1,
            save_channels=['mito', 'lyso', 'golgi', 'peroxy', 'er', 'nuclei'])
        ):
    """
    Buid arrays that will be the dataset of segmented objects. 

    Calls: 
        get_objects_from_segmentation (this module)
        sp.center_img_to_size (this modules)
    Args:
        df_fnames (pd.DataFrame) having index 'idx', and keys `path_seg`, `path_file`, `folder`
    """
    all_objects, all_scenes, all_scenes_segmented, all_metadata, all_contexts_seg, all_contexts_img = [],[],[],[],[],[]
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

        #img_ch=img_obj.get_image_data("TZYX",C=ch_idx)
        img = img_obj.data
        seg=seg_obj.data

        # pull out separate objects from the segmentation mask
        objects_masked_original, objects, coords = get_objects_from_segmentation(img, seg, ch_idx, frame, connectivity=1, verbose=verbose)
        # create dataset of objects with size img_dim that have the centered
        objects_masked, scenes, scenes_segmented, metadata = center_img_to_size(idx, fname_image, fname_seg, objects_masked_original,
                                obj_ch_idx=ch_idx, frame=frame, original_img=img, seg_img=seg, coords=coords, img_dim=64, verbose=1,
                                filter_big_things=filter_big_things,)

        # get the surrouding channels
        if context_kwargs['do']:
            save_ch_idxs = [ch_idx_lookup[k] for k in context_kwargs['save_channels']
                                        if k in ch_idx_lookup.keys()]
            coords_center = [d['coords_center'] for d in metadata]
            contexts_seg, contexts_img = get_img_channel_contexts(img, seg, frame, save_ch_idxs, coords_center,
                            save_img=context_kwargs['save_img'], img_dim=context_kwargs['img_dim'])

        # append this iterations objects
        labels = [idx]*len(objects_masked)

        all_objects.append(objects_masked)
        all_scenes.append(scenes)
        all_scenes_segmented.append(scenes_segmented)
        all_metadata.extend(metadata)
        all_contexts_seg.append(contexts_seg)
        all_contexts_img.append(contexts_img)
        if verbose: print(f"\t{len(objects_masked)} objects")

    all_objects=torch.cat(all_objects)
    all_scenes=torch.cat(all_scenes)
    all_contexts_seg=torch.cat(all_contexts_seg)
    all_contexts_img=torch.cat(all_contexts_img)
    all_scenes_segmented=torch.cat(all_scenes_segmented)
    all_metadata = all_metadata # pass

    return all_objects, all_scenes, all_scenes_segmented, all_metadata, all_contexts_seg, all_contexts_img

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
