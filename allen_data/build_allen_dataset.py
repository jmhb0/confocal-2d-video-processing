from pathlib import Path
from aicsimageio import AICSImage
import os
import numpy as np
import torch
import tqdm
from skimage.measure import regionprops
import matplotlib.pyplot as plt

def check_centroid_is_centered(img, dims=2, verbose=0, tol=0.51):
    """
    Given an image that is 2d or 3d (no channels) check that the
    centroid (measured by skimage.measure.regionprops) is within 0.5
    (pixel tolerance) of the image center.
    """
    assert img.ndim==dims
    midpoint = np.array(img.shape)/2
    centroid = regionprops(np.array(img))[0].centroid
    gap = np.absolute(midpoint-centroid)
    if verbose:
        print(f"midpoint {midpoint}\ncentroid {centroid}\ngap {gap}")
    return np.all(gap<tol)

def fov_to_3d_cell_nucleus_masks(dir_fovs = "/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/fovs",
                                 dir_segs_3d = "/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/cell_nucleus_masks_3d",
                                 first_n=None,
                                 fname_meta_df='/pasteur/u/jmhb/confocal-2d-video-processing/allen_data/meta_df.csv'):
    """
    `dir_fovs` is a directory with output of running `download_allen_data.download_cell_nucleus_seg_fovs()`
    which has segmented fields-of-view of cells and nuclei, and  with filename 
    for example, fov_seg_11270.tiff, meaning FOVId 11270. 
    This method creates a list of cropped 3d cell and nuclei objects in the FOV
    and saves them in `dir_segs_3d` with filename `seg_crop_11270.sav` (for example)
    which can be loaded with `torch.load(fname)`.

    Args:
    first_n (int): if debugging, set to 1 or 2 to only process the first 1 or 2 fov's.

    Side effects:
    ret (list): ret[i] is cell crop i, with cell_id ret[i][0] and 
        ret[i][1] is the array (channel,zslice,Y,X). Channel 0 is nucleus, channel
        1 is cell.
    It create a `ret` list for each fov file and saves it to `dir_segs_3d`.

    """
    if fname_meta_df is None:
        meta_df = pkg["metadata.csv"]()
    else:
        meta_df = pd.read_csv(fname_meta_df)

    fov_fns= [os.path.join(dir_fovs, f) for f in os.listdir(dir_fovs)]
    sorted(fov_fns)
    if first_n is not None:
        fov_fns = fov_fns[:first_n]

    for i, fov_fn in enumerate(tqdm.tqdm(fov_fns)):
        img_aics = AICSImage(fov_fn)
        img = img_aics.get_image_data('CZYX', T=0)

        # get the info about the cell
        fov_id = Path(fov_fn).stem.split("_")[-1]
        cells = meta_df.query(f"FOVId=={fov_id}")

        # for each cell get the cell and nucleus channels cropped
        all_crops = []
        for (i, row) in cells.iterrows():
            # get the seg masks for this particular cell
            mask_indx = row.this_cell_index
            nucleus_mask = (img[0]==mask_indx)
            cell_mask = (img[1]==mask_indx)

            # the crop regions are the coords that capture the cell. Nucleus
            # will use the same coords.
            coords = np.argwhere(cell_mask)
            l0, l1, l2 = coords.min(0)
            u0, u1, u2 = coords.max(0)
            s0, s1, s2 = slice(l0,u0+1), slice(l1,u1+1), slice(l2,u2+1)

            # create a new image to hold the cell and nucleus - 2 channel
            crop = np.zeros_like(img[[0,1], s0, s1, s2])
            crop[0] = nucleus_mask[s0, s1, s2]
            crop[1] = cell_mask[s0, s1, s2]

            # save a tuple of CellId and crop
            all_crops.append(( row.CellId, crop ))

        # save crops for this fov
        torch.save(all_crops, os.path.join(dir_segs_3d, f"seg_crop_{fov_id}.sav"))

def get_3d_crops_project(dir_segs_3d = "/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/cell_nucleus_masks_3d",
                         f_project_out="/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/cell-nucleus-2d-projections.sav"
                        , first_n=None):
    """
    Get the cropped objects and create a 2d representation. There are 2 methods and we ave them both.
    1. index the slice with the greatest nucleus area.
    2. max intensity projection.

    dir_segs_3d (Path): the directory holding the lists of cropped objects.
    f_project_out (Path): the filename to save the result

    Side-effect:
    Save output to `f_project_out`. Which is a 3-element tuple. Els 0 and 1 are lists of the 2d projection
    of the cropped object: el 0 is the slice. El 1 is the max intensity projection. El 2 a list of cell ids.
    """

    fnames = [ os.path.join(dir_segs_3d, f) for f in os.listdir(dir_segs_3d) ]

    if first_n is not None:
        fnames = fnames[:first_n]

    all_cell_ids, all_slice_max, all_max_projection = [], [], []

    for f in tqdm.tqdm(fnames):
        # get the crops for this fov
        all_crops = torch.load(f)

        # get the 2d representation: cell nucleus slice and max intensity projection separately.
        for i in range(len(all_crops)):
            cell_id, crop = all_crops[i]
            all_cell_ids.append(cell_id)

            # choose the slice with largest nucleus area
            max_nucleus_idx = np.argmax(crop.sum(-1).sum(-1)[0])
            all_slice_max.append(crop[:,max_nucleus_idx])

            # alternatively do max projection
            all_max_projection.append(crop.max(1))

    torch.save((all_cell_ids, all_slice_max, all_max_projection), f_project_out)

def center_img(img, dims=2, by_channel=1, method='center_mass',
       pad=200, verify_new_centroid=False, assert_centroid=1, verbose=0):
    """
    Args: 
    img: if dims=2 must have shape (C,Y,X). If dims=3, shape (C,Z,Y,X)
    dims (int): 2 if 2d, or 3 if 3d. The image will also have a channel dim. 
    by_channel (int): center the object in this channel, and shift 
        the other channels with it. 
    method (str): one of ('center_mass','bbox'), probably center_mass is better. 
    verify_new_centroid (bool): for debugging

    Note: if using method='center_mass', centering might cause part of the image to 
    leave the frame. That's bad, so we add padding, leading to the side-effect: 
    Side effect: 
        Add lots of padding (pad=200 default) and then assert that nothing went 
        wrong by calling check_centroid_is_centered(img[by_channel]).
        Then one should separately call `crop_img_evenly`. This is easier than 
        padding by the right amount
    """
    assert np.array_equal(np.unique(img), np.array([0,1])), "seg array must be 0s and 1s"
    assert img.ndim==dims+1, f"wrong number of dimensions for dims={dims}"

    # padding 
    pad_args=tuple([(0,0)] + [(pad,pad) for _ in range(dims)]) 
    img = np.pad(img, pad_args)

    # pick out the image channel that we base centering on 
    img_to_center= img[by_channel] 

    # bbox lower and upper bounds
    coords = np.where(img_to_center)
    lower_bounds = np.array([c.min() for c in coords])
    upper_bounds = np.array([c.max() for c in coords])

    if method=='center_mass':
        properties=regionprops(img_to_center)
        centroid = np.array(properties[0].centroid)
    elif method=='bbox':
        centroid = lower_bounds+(upper_bounds-lower_bounds)/2

    ## shift the whole image to put centroid in the middle 
    # first compute how much we need to shift
    shape = np.array(img.shape[1:])
    midpoint = shape/2
    shift = (midpoint-centroid).round().astype(np.int64)

    # now roll
    roll_axis = (1,2) if dims==2 else (1,2,3)
    img = np.roll(img, shift, axis=roll_axis)
    
    assert check_centroid_is_centered(img[by_channel], verbose=verbose)
    img=_crop_img_evenly(img, dims=2, by_channel=by_channel, verbose=verbose)
    assert check_centroid_is_centered(img[by_channel], verbose=verbose)
    
    return img

def _crop_img_evenly(img, dims=2, by_channel=1, assert_centroid=1, verbose=0):
    """
    Input: an image whose center of mass in channel `by_channel` is at the image
    center. 

    Crop an image by the same amount in axis 1 or 2 or 3. So if removing 
    pixels in x axis, then remove the same to the left or to the right. 
    This means the center of mass remains at the image center. 

    Args: 
    """
    # image params
    shape = np.array(img.shape[1:])
    coords = np.where(img[by_channel])
    lower_bounds = np.min(coords, axis=1)
    upper_bounds = np.max(coords, axis=1)
    obj_sizes = upper_bounds - lower_bounds
    # gap padding 
    lower_gap = lower_bounds
    upper_gap = shape-upper_bounds
    remove_pad = np.minimum(lower_gap, upper_gap) # even padding on either side - keep centroid

    # do slicing 
    slcs =  tuple( 
        [slice(0,len(img))] +\
        [slice(remove_pad[i]-1, shape[i]-remove_pad[i]+1) for i in range(dims)] 
    )
    img=img[slcs]

    if assert_centroid:
        is_centered = check_centroid_is_centered(img[by_channel], verbose=verbose)
        assert is_centered
    return img 

def put_centered_imgs_to_standard_sz(all_slices_centered, all_cell_ids, sz=512,
                                     dims=2, by_channel=1, keep_too_big_cells=1):
    """
    Args:
    all_slices_centered (list<np.array>): output of running something like
        `all_slices_centered = [bad.center_img(img, dims=2, by_channel=1) for img in all_slice_max]`
    dims (int): One of (2,3) for 2d or 3d.
    """ 
    n=len(all_slices_centered) 
    c=all_slices_centered[0].shape[0]
    assert all_slices_centered[0].ndim==1+dims
    
    data=np.zeros((n,c,sz,sz), dtype=np.uint8)
    data_cell_ids = np.zeros(n, dtype=np.int64)
    target_shape = np.array([sz]*dims)

    del_idxs=[]
    for i in range(n):
        big_flag=0
        img=all_slices_centered[i]
        shape = np.array(img.shape[1:])

        # if image is too big for requested size
        if not np.all(shape < sz):
            # don't add it. record the index to delete from array later
            if not keep_too_big_cells:
                del_idxs.append(i)
                continue
            # do add it: crop down the long axes (evenly on bothe sides)
            else:
                gap = shape-target_shape
                gap[gap<0]=0
                start = gap//2
                slcs = [slice(0,len(img))] + [slice(gap[i], gap[i]+sz) for i in range(len(gap))]
                img=img[tuple(slcs)]
                big_flag=1

        # put image in center of data array
        shape=img.shape[1:]
        gap = (target_shape - shape)//2
        slcs = [i, slice(0,len(img))] + [slice(gap[i], gap[i]+shape[i]) for i in range(len(gap))]

        # copy the data
        data[tuple(slcs)] = img.copy()
        data_cell_ids[i] = all_cell_ids[i]

        # if we have to do a crop, then the centroid of the remaining image will have moved, so we'd
        # fail this test. But we actually prefer not to move it (if you think about it)
        if not big_flag:
            assert check_centroid_is_centered(data[i,by_channel], tol=2, verbose=0)

    # if we ignored some cells, then delete their data
    if not keep_too_big_cells:
        keep_idxs = ~np.isin(np.arange(n), del_idxs)
        data = data[keep_idxs]
        data_cell_ids = data_cell_ids[keep_idxs]

    return data, data_cell_ids

def combine_cell_nuclei_one_img(data):
    """
    Given output that is coming from `put_centered_imgs_to_standard_sz`,
    combine the 2d cell+nuclei onto one image.
    Input data image shape (n,2,y,x) where the 2 channels are nuclei (0) and 
    cell (1). 
    Create a new data array with shape (n,1,y,x) so that cells have value 2 and 
    nucelei value 1. If a pixel has both, then it will be 1 (the nucleus).
    """
    n, _, y, x = data.shape
    nucleus_mask, cell_mask = (data[:,[0]]==1), (data[:,[1]]==1)
    joint = np.zeros((n,1,y,x))
    joint[cell_mask]=2
    joint[nucleus_mask]=1
    
    if type(data)==torch.Tensor:
        joint=torch.from_numpy(joint)

    return joint

def build_dataset(data, data_cell_ids, meta_df, M0_only=0, do_plot=0, 
        data_samples_plot=None, resize=64):
    """
    Build the dataset by filtering for M0 if required.
    Optionally make a plot of what the cells look like.

    Optionally resize.
    Args 
    data (np.ndarray): 
    data_cell_ids (np.ndarray): 
    """
    from torchvision.transforms import Resize
    meta_cells = meta_df.set_index('CellId').loc[data_cell_ids]
    cell_stages = meta_cells['cell_stage'].values

    if do_plot:
        from torchvision.utils import make_grid
        nrows=7
        stages, cnts = np.unique(cell_stages, return_counts=True)
        f, axs = plt.subplots(len(stages)//2, 2, figsize=(10,15))
        axs = axs.flatten()
        for i, stage in enumerate(stages):
            idxs = (cell_stages==stage)
            samples = torch.from_numpy(data_samples_plot)[idxs][:nrows**2]
            grid = make_grid(samples, nrows)[0]
            axs[i].imshow(grid, cmap='gray')
            axs[i].set(title=f"{stage}, {idxs.sum()}")
            axs[i].set_axis_off()
        plt.close()
    else:
        f=None

    # filter if required
    if M0_only:
        idxs = (cell_stages=="M0")
    else:
        idxs=np.arange(len(cell_stages))
    data_filtered = torch.from_numpy(data)[idxs]
    data_cell_ids_filtered = torch.from_numpy(data_cell_ids[idxs])
    cell_stages_filtered = cell_stages[idxs]

    if resize is not None: 
        data_filtered = Resize(resize)(data_filtered)

    return (data_filtered, data_cell_ids_filtered, cell_stages_filtered), f


