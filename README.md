The set of recommended segmentation algorithms:

The first is the notebooks, the second is the 'wrapper' that is used for the actual segmentation in the AICS datasets. 
- https://github.com/AllenCell/aics-segmentation/tree/main/lookup_table_demo
- https://github.com/AllenCell/aics-segmentation/tree/main/aicssegmentation/structure_wrapper


The channel-testing folder is for notebooks for testing different hyperparams on particular images step-by-step in the pipeline, or for whole-pipeline results for sets of images.

## config
Home directory `config.yaml` should have project root_dir. 

Put a piece of analysis in its own folder in `analyes`. 


## Config files 
- AC: auto-contrast normalization 
- MM: min-max normalization
- G2: 2d gaussian smoothing
- S: size filter

Params: list of parameters defined by making a new line and then indenting
Passing any value that is not indented means no function args, e.g. 
WD: None

Or branch means we create 2 separate seg masks using funcs in branch1, branch2, .... 
and then combine the final mask result using logical OR. 

## `pipeline_utils.py`

## `seg_eval.py`
Functions for doing single-frame segmentations and producing pdfs. 



# Standard workflow example 
Example analysis in `analyses/snr/jan22-neural-diff`:
- Notebooks to display segmentation results and choose parameters: `seg-mito.ipynb`, `seg-nucleus.ipynb` etc. Each one will have a string structured like a yaml config that determines the method. 
- Once you have the config for each organelle, put it in `configs/` folder (see exmaple)
- To do the segmentation, run `do-segmentations.py`.
-- A lot of parameters need to be set here: the location of the data folder; location of a csv containing a column called `path_file` that is the relative path (from the data folder) to the images to segment; that csv will be saved to `df_fnames` and there is code to choose a subset of this data to actually segment; location of the config file for segmentation and only those channels in the config will get segmented; the `SEG_RUN_NAME` will create a folder in the data home directory that mirrors the structure of the data directory but replacing the image with a segmented version; some code for choosing a subset. 
-- The script reads the data assuming the dimensions are TCZYX. If the flag `DO_CELL_MASK` is True, then  a whole-cell mask is estimated at each timestep and applied to each segmentation. 
-- Segmentation is done for each timestep and channel independently. 
- In `seg-image-mito-dataset-jun6.ipynb` we build the 'mito inference dataset'. 
-- First choose anhoutput data directory in `DIR_OUT`. 
-- Run `get_imgs_from_timelapse_segmentation` which has parameters inside the function: it samples timeseries data with some frequency set by the array `ts`; it only copies the channels from the list `chs` in the order its listed. Those images are put into a list and saved in torch Tensor format as `imgs-whole-seg-samples.sav` and a list of metadata for each image is saved to `meta-whole-seg-samples.sav` which saves the cell index, timestamp, and pixel dimensions.
-- Run `extract_mito_from_img_slices()` which extracts individual mito from each sampled image and puts each into a centered image with the same dimensions (parameters need to be set to choose those dimensions). Outputs a Tensor of images in `all_crops` and metadata list of the same length in `all_meta` which records which cell image it should be linked to.
-- There's a final panel which saves everything in `DIR_out`. 
- The data folder created from the last notebook can be copied to the other repo `steerable-vae`. Then update the file `data/props_datasets` and the function `get_datasets_loaders_and_metadata ` and update it to point at the new folder. In that repo there are notebooks in `results_organelle_interaction` called `neighbour_measure_funcs.ipynb` which can use the data in this format to measure per-mitochondria rates of contact.




