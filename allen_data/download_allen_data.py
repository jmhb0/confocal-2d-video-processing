import quilt3
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.io import imread, imsave
from pathlib import Path
import numpy as np
import pandas as pd
import os
import torch
import tqdm

def save_lite_meta_df(f_in='/pasteur/u/jmhb/confocal-2d-video-processing/allen_data/metadata.csv?versionId=C6BdJMGP_rt96VhTS5LFUqmY9Y.2aGmh',
        f_out='/pasteur/u/jmhb/confocal-2d-video-processing/allen_data/meta_df.csv',
        ):
    """
    Load the full metadata csv and chop out the dim reduction metadata to save time loading 
    """
    meta_df = pd.read_csv(f_in)
    meta_df_lite = meta_df[meta_df.columns[:47]]
    meta_df_lite.to_csv(f_out)


def download_raw_img_and_segs(structure_name="TOMM20",
         save_path="/pasteur/data/allen_single_cell_image_dataset/tmp",
         num_files=None, 
         fname_meta_df=None,
         ):
    """
    Download cell data and segmentation from Allen cell databse
    Based on tutorials at https://github.com/AllenCell/quilt-data-access-tutorials
    With dataset at https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset
    And paper at  https://www.biorxiv.org/content/10.1101/2020.12.08.415562v1.full.pdf

    Args
        structure_name (str): structure name stored in the quilt package, referring 
            to the fluorophore of a particular type. "TOMM20" is mitochondria. Can
            read the options off of the `meta_df` code that is demonstrated here.
        num_files (int): limit number of files to download (about 200MB each).
            If None, then no limit
        fname_meta_df (str): if None, then assume that the meta_df can be pulled fropm 
            the quilt pkg. Otherwise read from a file path. This is because the quilt pkg
            was erroring 
            CSV from https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset
            
    Structure names can be looked up in table 1 of the linked paper. 
        TOMM20 - mitochondria
        LAMP1  - lysosome 
        SLC25A17  - peroxisome 
        SEC61B and ATP2A2 - both types of ER with different staining stategies I guess. 
        ST6GAL1 - Golgi (I think)
    """
    print("Getting Quilt package")
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", "s3://allencell")
    print("Getting medata.csv")
    if fname_meta_df is None: 
        meta_df = pkg["metadata.csv"]()
    else: 
        meta_df = pd.read_csv(fname_meta_df)
    ## prepare list of data we'll download 
    meta_samples = meta_df.query(f"structure_name=='{structure_name}'")
    print(f"Size before dropping duplicates {len(meta_samples)}")
    # collapse the data table based on FOVId (not sure what this means)
    meta_samples = meta_samples.drop_duplicates(subset="FOVId")
    print(f"Size after dropping duplicates {len(meta_samples)}")
    meta_samples = meta_samples.reset_index()

    ## set up paths for the raw data, segmentation, and strucutre
    if save_path is None: raise ValueError()
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    raw_path = save_path / Path("raw_image")
    raw_path.mkdir(exist_ok=True)
    structure_path = save_path / Path("structure")
    structure_path.mkdir(exist_ok=True)
    seg_path = save_path / Path("structure_segmentation")
    seg_path.mkdir(exist_ok=True)

    ## Do download 
    if num_files is None:
        num_files=len(meta_samples)

    for row in meta_samples.itertuples():
        if row.Index >num_files:
            break# early stop if reached count
        print(f"Downloading {row.Index} of {num_files} downloads")

        # download
        subdir_name = row.fov_path.split("/")[0]
        file_name = row.fov_path.split("/")[1]
        print(f"Getting {row.FOVId}_original.tiff and linked seg and structure files")

        local_fn = raw_path / f"{row.FOVId}_original.tiff"
        pkg[subdir_name][file_name].fetch(local_fn)

        # extract the structure channel
        structure_fn = structure_path / f"{row.FOVId}.tiff"
        """
        reader = AICSImage(local_fn)
        with OmeTiffWriter(structure_fn) as writer:
            writer.save(
                reader.get_image_data("ZYX", C=row.ChannelNumberStruct, S=0, T=0),
                dimension_order='ZYX'
            )
        """

        # fetch structure segmentation
        subdir_name = row.struct_seg_path.split("/")[0]
        file_name = row.struct_seg_path.split("/")[1]

        seg_fn = seg_path / f"{row.FOVId}_segmentation.tiff"
        pkg[subdir_name][file_name].fetch(seg_fn)

def save_img_and_seg(save_path="/pasteur/data/allen_single_cell_image_dataset/mitochondria_samples"):
    """
    Expecting data files in a directory structure created by `download_raw_img_and_segs`
    Creates a new directory under `save_path` called `structure_segmentation`.
    
    This gets the images, choosese the channel with the image, and saves the segmentation. Tkes 
    one Z from the zstack. 
    """
    ## Create results dir if it doesn't already
    new_save_path=Path(f"{save_path}/sample-imgs-and-segs")
    new_save_path.mkdir(exist_ok=True)

    ## Get file list
    raw_img_files=os.listdir(f"{save_path}/raw_image")
    raw_img_files.sort()
    print(f"Found {len(raw_img_files)} raw image files")
    seg_files=os.listdir(f"{save_path}/structure_segmentation")
    seg_files.sort()
    print(f"Found {len(seg_files)} segmentation image files\n")
    file_prefixes = [s.split('_')[0] for s in seg_files]

    for i, f_pr in enumerate(file_prefixes):
        print('-'*80)
        fname_img=f"{save_path}/raw_image/{f_pr}_original.tiff"
        fname_seg=f"{save_path}/structure_segmentation/{f_pr}_segmentation.tiff"
        if fname_img.split('/')[-1] not in raw_img_files:
            print(f"{fname_img} not found, skipping")
            continue
        if fname_seg.split('/')[-1] not in seg_files:
            print(f"{fname_seg} not found, skipping")
            continue

        print(f"Reading original     {fname_img}")
        print(f"Reading segmentaiton {fname_seg}")

        img=imread(fname_img)
        seg=imread(fname_seg)
        print(img.shape, seg.shape)

        if img.shape[3]==4:
            img=img.swapaxes(1,3).swapaxes(2,3)
            channel=1
        elif img.shape[1]==7:
            channel=3
        else:
            print(f"**Unrecognized raw data for {fname_img}, {fname_seg}. Skipping")
            continue

        if img.shape[0]!=seg.shape[0] or img.shape[2:]!=seg.shape[1:]:
            print(f"Shape mismatch for {fname_img}, {fname_seg}. Skipping")
            print(f"Shapes are {img.shape}, {seg.shape}\n")
            continue

        frames=img.shape[0]
        frame_indx=frames//2
        fname_new_img=f"{new_save_path}/{f_pr}_img.tif"
        fname_new_seg=f"{new_save_path}/{f_pr}_seg.tif"
        print(f"Saving image at        {fname_new_img}")
        print(f"Saving segmentation at {fname_new_seg}")
        imsave(fname_new_img, img[frame_indx,channel])
        imsave(fname_new_seg, seg[frame_indx])


        import tqdm
        
def download_cell_nucleus_seg_fovs(dir_fovs="/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/fovs"):
    """
    Download all the fields of view in the dataset (there are 18186)
    Each FOV has multiple cells. 
    """
    
    ## supply an fname for the metadata if it's being held locally
    fname_meta_df = "/pasteur/u/jmhb/confocal-2d-video-processing/allen_data/metadata.csv?versionId=C6BdJMGP_rt96VhTS5LFUqmY9Y.2aGmh"

    ##### get the metadata 
    print("Getting Quilt package")
    if "pkg" not in locals():
        pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", "s3://allencell")

    print("Getting medata.csv")
    if fname_meta_df is None: 
        print("Loading metadata dataframe from the `pkg`")
        meta_df = pkg["metadata.csv"]()
    else: 
        print(f"Loading metadata from {fname_meta_df}")
        meta_df = pd.read_csv(fname_meta_df)
        
    ##### do the download 
    # each row of meta_df is a cell, so the same fov has multiple rows. Just get 1. 
    samples = meta_df.groupby("FOVId", group_keys=False)
    samples = samples.apply(pd.DataFrame.sample, n=1)
    samples.sort_values("FOVId")["FOVId"]

    for i, (_, row) in enumerate(samples.iterrows()):
        print(i)
        subdir_name = row.fov_seg_path.split("/")[0]
        file_name = row.fov_seg_path.split("/")[1]
        local_fn = os.path.join(dir_fovs, f"fov_seg_{row.FOVId}.tiff")
        pkg[subdir_name][file_name].fetch(local_fn)

from pathlib import Path
from aicsimageio import AICSImage
import os
import numpy as np
import torch
import tqdm

def fov_to_3d_cell_nucleus_masks(dir_fovs = "/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/fovs",
                                 dir_segs_3d = "/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/cell_nucleus_masks_3d",
                                 first_n=None,
                                 fname_meta_df='/pasteur/u/jmhb/confocal-2d-video-processing/allen_data/meta_df.csv'):
    """
    Given a path containing images of fovs,
    first_n (int): if debugging, set to 1 or 2 to only process the first 1 or 2 fov's.
    """
    if fname_meta_df is None:
        meta_df = pkg["metadata.csv"]()
    else:
        meta_df = pd.read_csv(fname_meta_df)

    fov_fns= [os.path.join(dir_fovs, f) for f in os.listdir(dir_fovs)]
    sorted(fov_fns)
    for i, fov_fn in enumerate(tqdm.tqdm(fov_fns)):
        if first_n is not None and i>=first_n:
            break
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

