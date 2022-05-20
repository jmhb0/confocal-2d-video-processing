## Building the Allen single cell datasets 

Download all FOVS:
```
import download_allen_data as dad
import build_allen_dataset as bad

dir_fovs="/path/to/fov/dir/"
bad.download_cell_nucleus_seg_fovs(dir_fovs)
```