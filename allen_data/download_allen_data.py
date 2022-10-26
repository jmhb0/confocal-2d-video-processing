import quilt3
import psutil
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
    Download cell data and segmentation from Allen cell databse for structures (organelles / proteins)
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

def get_structure_single_cells(dir_data_out = "/pasteur/data/allen_single_cell_image_dataset/mito_samples/mito_single_cells", 
                         sturcure_name="TOMM20"):
    # below, all the instances of "structure" mean some organelle depending on "structure_name"

    # data directories
    d_structures = "/pasteur/data/allen_single_cell_image_dataset/mito_samples/structure_segmentation"
    d_structure_raw = "/pasteur/data/allen_single_cell_image_dataset/mito_samples/raw_image"
    d_fov_segs = "/pasteur/data/allen_single_cell_image_dataset/cell-nucleus-segs/fovs"

    fname_meta_df='/pasteur/u/jmhb/confocal-2d-video-processing/allen_data/meta_df.csv'
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", "s3://allencell")

    ## packages
    # print("Getting Quilt package")
    print("Getting metadata.csv")
    if fname_meta_df is None: 
        pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", "s3://allencell")
        meta_df = pkg["metadata.csv"]()
    else: 
        meta_df = pd.read_csv(fname_meta_df)

    ## filter for this structure 

    def norm_0_1(x):
        l,u = np.min(x), np.max(x)
        return (x-l) / (u-l)

    df_structure=meta_df.query(f"structure_name=='{structure_name}'")
    print(f"Unique FOVS with this structure {structure_name} is: {len(df_structure.FOVId.unique())}")
    print(f"Unique Cellids is {len(df_structure.CellId.unique())}")
    # get existing files 
    fs_m = os.listdir(d_structures)
    fs_f = os.listdir(d_fov_segs)
    ids_m = np.array([f.split("_")[0] for f in fs_m])
    ids_f = np.array([f.split(".")[0].split("_")[-1] for f in fs_f])
    print("number of fov ids in mito", ids_m.shape, "in fovs", ids_f.shape)
    print("pcnt of mitos found in fovs", np.isin(ids_m, ids_f).sum() / len(ids_m))
    i=0
    import tqdm
    t = tqdm.tqdm(ids_m)
    # iterate over fovids 
    for fovid in t:
        t.set_description(f"i: {i} Memory usage: {psutil.virtual_memory().percent} %")
        f_structure_seg = os.path.join(d_structures, f"{fovid}_segmentation.tiff")
        f_structure_raw = os.path.join(d_structure_raw, f"{fovid}_original.tiff")
        f_cell_seg = os.path.join(d_fov_segs, f"fov_seg_{fovid}.tiff")

        img_structure_seg = imread(f_structure_seg)
        img_structure_raw = imread(f_structure_raw)
        img_cell_seg = imread(f_cell_seg)

        ### df_this is the cells for this FOV
        df_this = meta_df.query(f"FOVId=={fovid}")
        all_cells, all_cellids = [], []

        # iterate over cells in the fov
        for i, (_, row) in enumerate(df_this.iterrows()):
            t.set_description(f"i: {i} Memory usage: {psutil.virtual_memory().percent}%, cpu {psutil.cpu_percent()}")
            # cell and nucleus mask
            cell_mask = np.zeros_like(img_structure_seg, np.uint8)
            nucleus_mask = np.zeros_like(img_structure_seg, np.uint8)
            nucleus_mask[img_cell_seg[:,:,:,0]==row.this_cell_index]=1
            cell_mask[img_cell_seg[:,:,:,1]==row.this_cell_index]=1

            # crop based on cell mask 
            coords = np.argwhere(cell_mask)
            l0, l1, l2 = coords.min(0)
            u0, u1, u2 = coords.max(0)
            s0, s1, s2 = slice(l0,u0+1), slice(l1,u1+1), slice(l2,u2+1)

            ## out img to hold everything 
            out_img = np.zeros((u0-l0+1, u1-l1+1, u2-l2+1, 6), np.float32)

            ## segmentations 
            # add cell and nucleus mask to channels 0 and 1
            out_img[:,:,:,0] = nucleus_mask[s0, s1, s2].copy()
            out_img[:,:,:,1] = cell_mask[s0, s1, s2].copy()
            # mito seg 
            out_img[:,:,:,2] = (cell_mask*img_structure_seg/255)[s0, s1, s2].copy()

            ## raw  
            # cell, nucleus, and structure are all in img_structure raw.
            # there are 2 different images that may appear, one with 
            # shape (z,c,y,x) with c=7, and one with shape (z,y,x,c) with c=4. 
            if img_structure_raw.shape[1]==7:
                nucleus_raw = (nucleus_mask*norm_0_1(img_structure_raw[:,5]))[s0, s1, s2] 
                cell_raw = (cell_mask*norm_0_1(img_structure_raw[:,1]))[s0, s1, s2] 
                structure_raw = (cell_mask*norm_0_1(img_structure_raw[:,3]))[s0, s1, s2] 
            elif img_structure_raw.shape[-1]==4:
                nucleus_raw = (nucleus_mask*norm_0_1(img_structure_raw[:,:,:,2]))[s0, s1, s2] 
                cell_raw = (cell_mask*norm_0_1(img_structure_raw[:,:,:,0]))[s0, s1, s2] 
                structure_raw = (cell_mask*norm_0_1(img_structure_raw[:,:,:,1]))[s0, s1, s2] 
            else:
                raise ValueError(f"unknwon format for img structure raw shape {img_structure_raw.shape}")    
            out_img[:,:,:,3] = nucleus_raw
            out_img[:,:,:,4] = cell_raw
            out_img[:,:,:,5] = structure_raw

            ## append cell to the list
            all_cells.append(out_img)
            all_cellids.append(row.CellId)

        f_out = os.path.join(dir_data_out, f"{fovid}_cell_structure.sav")
        np.save(f_out, [all_cells, all_cellids]) 


