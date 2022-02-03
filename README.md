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

