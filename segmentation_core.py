## Core methods for taking images/timelapses -> segmentations 
import pipeline_utils as pu

from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_2d, dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.pre_processing_utils import suggest_normalization_param
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.utils import topology_preserving_thinning
from util_es_smooth_2d import anisodiff as edge_preserving_smoothing_2d

import numpy as np
import yaml
import importlib
from collections import OrderedDict
from skimage import morphology, feature, segmentation
from scipy import ndimage as ndi
import warnings

def load_segmentation_functions(fname='seg-func-map.yaml'):
    """
    Segmentation functions are represented as codes (e.g F2 is filament filter)
    so that we can specify an imaging pipeline as a list of codes and their 
    params, e.g. in `seg_params.yaml`.
    
    The corresponding package and function names are in a yaml file, by default
    in ${PROJECT_DIR}/img-pipeline/seg-func-map.yaml
    
    This method opens the config/map file, stores it as a dict, and loads the 
    function so that the F2 function can be loaded by doing: 
        seg_funcs = load_segmentation_functions(fname='seg-func-map.yaml')
        f2_function = seg_funcs['F2']['f']
    """
    with open(fname) as file:
        seg_funcs = yaml.load(file, Loader=yaml.FullLoader)

    for k, v in seg_funcs.items():
        package = importlib.import_module(v['package'])
        func = getattr(package, v['func'])
        v['f'] = func

    return seg_funcs

def load_segmentation_config_from_file(fname=None, seg_funcs=None, verbose=1):
    yaml_link = open(fname)
    return load_segmentation_config(yaml_link, seg_funcs=seg_funcs, verbose=verbose, fname=fname)

def load_segmentation_config(yaml_in=None,  seg_funcs=None, verbose=0, fname=None):
    """ 
    Load in the segmentation config file, check that the parameter names are valid
    for each functions, and set parameters to defaults where applicable. 

    Returns a dictionary that can be passed to segmentation_core.seg_2d.
    Args: 
        yaml_str: string that holds valid YAML, OR a file descriptor. 
    """
    if seg_funcs is None: raise ValueError("Must supply a seg function dictionary like in "\
                                      "seg-func-map.yaml")
    pu.setup_yaml()    # import stuff as an ordered dict instead of dict
    seg_config = yaml.load(yaml_in, Loader=yaml.FullLoader)
    
    # Outermost loop: is to access all single function calls, and check/update params 
    # according to the util func `_check_func_params_for_defaults`.
    # Loop through each channel, then each function, handling the case of an `or_branch`
    # which requires some extra stuff
    for channel, _ in seg_config.items():
        func_param_dict = seg_config[channel]['seg']['pipeline']
        # the following operates in place
        for func_name, params in func_param_dict.items():
            # handle the branching case
            if func_name=='or_branch':
                for branch_name, branch_val in params.items():
                    for func_name, branch_params in branch_val.items():
                        seg_config[channel]['seg']['pipeline']['or_branch'][branch_name][func_name] \
                          = _check_func_params_for_defaults(func_name, branch_params, seg_funcs, verbose,
                                                        fname, channel)
            
            # the regular non-branched case
            else: 
                seg_config[channel]['seg']['pipeline'][func_name]\
                    = _check_func_params_for_defaults(func_name, params, seg_funcs, verbose,
                                                        fname, channel)

    return seg_config 

def _check_func_params_for_defaults(func_name, params, seg_funcs, verbose=0, 
                                    config_fname=None, channel=None):
    """
    Util for taking a function name `func_name` and parameter dict `params`:
        (i) if a param is not included in params.keys() or if it is included 
            but it's listed as a string "none", then replace the param value
            with the default defined in `seg_funcs`.
        (ii) if a param is defined that is not a real parameter for the method
            (so not in the defaults of `seg_funcs`) then raise an error.
 
    Args:
        func_name: str that is a code from the valid list (e.g.'S2')
        params: usually it's a dictionary where key: param name (e.g. 'S2_param')
            and val: param value (e.g. [[1.2,0.1]]). If it's not a dict, then it 
            will be replaced by an empty dict, then it will be replaced by default
            values
    """
    if type(params) is dict: 
        # Go through each config parameter. Check that it's a valid parmeter.
        # If value is "None", then replace with the default from seg_funcs.
        for param_name, param_val in params.items():
            if param_name not in seg_funcs[func_name]['params_default'].keys():
                raise ValueError(f"Invalid argument name {param_name} in the config file {config_fname}")
            if str(param_val).lower()=="none":
                params[param_name] = seg_funcs[func_name]['params_default'][param_name]
                if verbose: 
                    print(f"INFO: param is None. Channel {channel}, func {func_name}, param {param_name}"\
                          f", config {config_fname} is None. Adding default")
    else: 
        params = {}
        if verbose: print('INFO: No parameters passed to {func_name}', func_name)

    # Go through list of default params. If it's not included in the 
    # config, then add the default parameter
    default_params = seg_funcs[func_name]['params_default']

    if type(default_params) is dict: 
        for default_param_name, default_param_val in default_params.items():
            if default_param_name not in params.keys():
                params[default_param_name] = default_param_val
                if verbose: 
                    print(f"INFO: missing param. Channel {channel}, func {func_name}, param {default_param_name}"\
                          f", config {config_fname}. Adding default")
    else: 
        if verbose: print('skipping default')

    return params

def seg_2d(img_in, seg_config, seg_funcs, ch_name='mito', inplace=False,
          save_inters=False, mask=None):
    """ 
    Input is YX
    Args: 
        img_in: np.array image with shape YX
        seg_pipeline: dict with keys 
        save_inters: is save_intermediates. Either True, False or list of 
            function names, e.g. ['AC','GS','F2','S']. Saving intermediates 
            does image copies, so only keep this True if it's important.
    TODO: (maybe). Embed the intermediate img values in the config dictionary object. 
        For this to make sense, I would have to update the config structure. Currently 
        where there is a key for the function name, and a value is k-v pairs of parameters, 
        we would need a key that is func name, then under that value, we could have 
        `img` and then a key of `params` (so the params go one level down). 
        This has the disadvantage that interpreting the dictionary is more complicated. 
    """
    # do checking of image properties and whether to inplace
    # inplace = seg_config[ch_name]['seg']['ops']['inplace']
    img = img_in.copy() if not inplace else img_in

    if mask is not None:
        assert img.shape==mask.shape
        img = img*mask

    channel_name_list = ['mito','peroxy','lyso','er','golgi']
    assert ch_name in channel_name_list
    assert img.ndim==2, "Input shape must be YX"
    img = np.expand_dims(img, 0) # since aics seg assumes TYX input

    
    # prepare document for 'save_inters'
    inter_imgs = OrderedDict()
    
    # loop over list of functions and apply them
    func_sequence = seg_config[ch_name]['seg']['pipeline']
    for func_name, func_params in func_sequence.items():
        # handle the branch case where we create parallel images and then do 
        # logical or to combine them
        if func_name == "or_branch":
            img_list = []
            for branch_name, branch_val in func_params.items():
                img_tmp = img.copy()
                for branch_func_name, branch_params in branch_val.items():
                    
                    img_tmp, inter_imgs = _apply_func_to_img(img_tmp, branch_func_name, branch_params, seg_funcs, save_inters, inter_imgs)
#                     inter_imgs,  = _optionally_save_img(branch_func_name, img, inter_imgs, save_inters)
                img_list.append(img_tmp)
            img = np.logical_or(*img_list)
            inter_imgs = _optionally_save_img('logical_or_out', {}, img, inter_imgs, save_inters)
        # handle the regular case of no branching 
        else:                
            img, inter_imgs = _apply_func_to_img(img, func_name, func_params, seg_funcs, save_inters, inter_imgs)
#             inter_imgs = _optionally_save_img(func_name, func_params, img, inter_imgs, save_inters)

    if save_inters: return img, inter_imgs
    else: return img

    
def seg_2d_timelapse_by_channel(img_in, *args, ch_name=None, inplace=False, **kwargs):
    """
    Input is TYX
    Wrapper for seg_2d. Apply the same thing to each channel

    """
    if ch_name is None: raise ValueError()
    assert img_in.ndim==3, "Input shape must be CYX"
    img = img_in.copy() if not inplace else img_in
    for i in range(len(img)):
        img[[i]] = seg_2d(img[i], *args, ch_name=ch_name, **kwargs, inplace=inplace)
    return img

def seg_2d_timelapse(img_in, *args, order_channels=['golgi', 'er','lyso', 'mito', 'peroxy'],
                    verbose=0, inplace=True , **kwargs):
    """
    Input is TCYX
    """
    assert img_in.ndim==4, "Input shape must be TCYX"
    img = img_in.copy() if not inplace else img_in
    
    for ch_indx, ch_name in enumerate(order_channels):
        if verbose: print(f"Processing channel {ch_name}" )
        img[:,ch_indx] = seg_2d_timelapse_by_channel(img[:,ch_indx], ch_name=ch_name,
                                                     inplace=inplace, *args, **kwargs)
    return img

def _apply_func_to_img(img, func_name, func_params, seg_funcs, save_inters, inter_imgs):
    """
    Apply a function to an image given a string that is the function 
    name (where the corresonding function is in `seg_funcs`), with the 
    parameters defined in the dictionary `func_params`.
    Args:
        `seg_funcs`: dict of function names mapped to function objects
        `save_inters`: bool or list; same as in `seg_2d` func definition. 
    """
    func = seg_funcs[func_name]['f']
    ## TODO: this is an unpleasant hack to deal with the fact that the interface 
    # takes different dimensions. 
#     if func_name == 'ES_aics':
#         print("Doing ES_aics ", img.shape)
#         img = img[0]
#         img = func(img, **func_params)
#         img = np.expand_dims(img, 0)
#     else:
#         img = func(img, **func_params)

    # handle special case of saving inter_imgs
    inter_imgs = _handle_intermediate_filters_out_of_seg_flow(img, func_name, func_params, save_inters, inter_imgs)
    # do the function 
    img = func(img, **func_params)
    # save the updated image if relevant
    _optionally_save_img(func_name, func_params, img, inter_imgs, save_inters)
    
    return img, inter_imgs

def _optionally_save_img(func_name, func_params, img, inter_imgs, save_inters):
    """
    util to for seg_2d to add k-v pair of func_names & images. `save_inters` is
    either True, False, or a list of func_names. 
        """
    if save_inters==True or (type(save_inters) is list and func_name in save_inters):
        assert func_name not in inter_imgs, "Cannot duplicate keys"
        inter_imgs[func_name] = {}
        inter_imgs[func_name]['img'] = img.copy()
        inter_imgs[func_name]['params'] = func_params
    else: pass
    return inter_imgs

def _handle_intermediate_filters_out_of_seg_flow(img, func_name, func_params, save_inters, inter_imgs):
    """
    Util called by `_apply_func_to_img`, which handles the special case where we want to 
    record intermediate images for evaluation, and we want to call the F2 or S2 filters,
    but before their cutoff. This adds a new img to inter_imgs if F2 or S2 are called but
    does nothing else. 
    """
    if save_inters==True or (type(save_inters) is list and func_name in save_inters):
        # handle case where we want the pre-cutoff value
        if func_name in ('F2', 'S2'):
            img_tmp = img.copy()
            if func_name=='F2': 
                img_tmp = filamend_2d_pre_thresholding(img_tmp[0], **func_params)                
                img_tmp = np.expand_dims(img_tmp, 0)
            elif func_name=='S2': 
                img_tmp = dot_2d_slice_by_slice_wrapper_pre_thresholding(img_tmp, **func_params)
            else: raise ValueError()
            
                
            new_func_name = func_name+'_precut'
            assert new_func_name not in inter_imgs, "Cannot duplicate keys"
            inter_imgs[new_func_name] = {}
            inter_imgs[new_func_name]['img'] = img_tmp
            inter_imgs[new_func_name]['params'] = func_params

    return inter_imgs

### Seg core utils
def _2d_seg_input_ops(img_in, inplace=False):
    """
    Standard operations done to the input for every 2d segmentation 
    function. Used by functions `seg_(mito|peroxy|lyso|er)_2d`
    """
    assert len(img_in.shape)==2, "Input shape must be YX"
    img = img_in.copy() if not inplace else img_in
    img = np.expand_dims(img, 0) # since aics seg assumes TYX input
    return img

def watershed_2d_distance_transform(seg, footprint=np.ones((3,3)), testing=False):
    """
    Watershed based segmentation on a distance transform map. 
    The input is a segmentation mask proposal, where occluded objects should
    be separated. 
    This is intended for spherical occluded objects, like peroxisomes. 
    """
    assert len(seg.shape)==3
    seg = seg[0]
    assert seg.dtype=='bool' or len(np.unique(seg))==2

    distance = ndi.distance_transform_edt(seg)
    distance0 = distance.copy()

    basins_coords = feature.peak_local_max(distance, footprint=footprint, labels=seg)
    basins = np.zeros(distance.shape, dtype=bool)
    basins[tuple(basins_coords.T)] = True
    markers, _ = ndi.label(basins)
    labels = segmentation.watershed(-distance, markers, mask=seg)

    labels, distance = np.expand_dims(labels, 0), np.expand_dims(distance0, 0)
    if testing: return labels, distance
    else: return labels

def normalize_0_1(img_in, display_range=[0,1], inplace=False):
    """
    Send image to range [0,1], then truncate do display range. 
    E.g. if display_range=[0,0.5], this function first rescales all images 
    to range [0,1], then all pixels >0.5 are sent to 0.5
    """
    assert len(display_range)==2
    assert display_range[0]<display_range[1]
    for d in display_range: assert 0<= d <= 1

    img = img_in.copy() if inplace else img_in
    img = (img-img.min()) / (img.max()-img.min())
    img[img<display_range[0]]=display_range[0]
    img[img>display_range[1]]=display_range[1]
    return img


def compute_vesselness2D(eigen2, tau):
    """backend for computing 2D filament filter"""

    Lambda3 = copy.copy(eigen2)
    Lambda3[np.logical_and(Lambda3 < 0, Lambda3 >= (tau * Lambda3.min()))] = tau * Lambda3.min()

    response = np.multiply(np.square(eigen2), np.abs(Lambda3 - eigen2))
    response = divide_nonzero(27 * response, np.power(2 * np.abs(eigen2) + np.abs(Lambda3 - eigen2), 3))

    response[np.less(eigen2, 0.5 * Lambda3)] = 1
    response[eigen2 >= 0] = 0
    response[np.isinf(response)] = 0

    return response


##### Modified aicsSegmentation functions 
from aicssegmentation.core.hessian import absolute_3d_hessian_eigenvalues
from scipy.ndimage.filters import gaussian_laplace
from aicssegmentation.core.utils import divide_nonzero
import copy 
from scipy import ndimage as ndi

def filamend_2d_pre_thresholding(struct_img, f2_param):
    """
    A copy of aicssegmentation.core.vessel.filament_2d_wrapper
    https://github.com/AllenCell/aics-segmentation/blob/main/aicssegmentation/core/vessel.py
    Except without doing the final thresholding step. 
    The code is copied exactly, with the relevant lines commented out.
    """
    # bw = np.zeros(struct_img.shape, dtype=bool)

    if len(struct_img.shape) == 2:
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]
            eigenvalues = absolute_3d_hessian_eigenvalues(struct_img, sigma=sigma, scale=True, whiteonblack=True)
            responce = compute_vesselness2D(eigenvalues[1], tau=1)
            # bw = np.logical_or(bw, responce > f2_param[fid][1])
    elif len(struct_img.shape) == 3:
        mip = np.amax(struct_img, axis=0)
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]

            res = np.zeros_like(struct_img)
            for zz in range(struct_img.shape[0]):
                tmp = np.concatenate((struct_img[zz, :, :], mip), axis=1)
                eigenvalues = absolute_3d_hessian_eigenvalues(tmp, sigma=sigma, scale=True, whiteonblack=True)
                responce = compute_vesselness2D(eigenvalues[1], tau=1)
                res[zz, :, : struct_img.shape[2] - 3] = responce[:, : struct_img.shape[2] - 3]
            # bw = np.logical_or(bw, res > f2_param[fid][1])
    return responce

def dot_2d_pre_thresholding(struct_img, s2_param):
    """
    A copy of aicssegmentation.core.seg_dot.dot_2d
    https://github.com/AllenCell/aics-segmentation/blob/main/aicssegmentation/core/vessel.py
    Except without doing the final thresholding step. 
    The code is copied exactly, with the relevant lines commented out.
    """
    pass

def dot_2d_slice_by_slice_wrapper_pre_thresholding(struct_img, s2_param):
    """wrapper for 2D spot filter on 3D image slice by slice
    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d numpy array, usually the image after smoothing
    s2_param: List
        [[scale_1, cutoff_1], [scale_2, cutoff_2], ....], e.g. [[1, 0.1]]
        or [[1, 0.12], [3,0.1]]: scale_x is set based on the estimated radius
        of your target dots. For example, if visually the diameter of the
        dots is usually 3~4 pixels, then you may want to set scale_x as 1
        or something near 1 (like 1.25). Multiple scales can be used, if
        you have dots of very different sizes. cutoff_x is a threshold
        applied on the actual filter reponse to get the binary result.
        Smaller cutoff_x may yielf more dots and fatter segmentation,
        while larger cutoff_x could be less permisive and yield less
        dots and slimmer segmentation.
    """
    #bw = np.zeros(struct_img.shape, dtype=bool)
    for fid in range(len(s2_param)):
        log_sigma = s2_param[fid][0]
        responce = np.zeros_like(struct_img)
        for zz in range(struct_img.shape[0]):
            responce[zz, :, :] = -1 * (log_sigma ** 2) * ndi.filters.gaussian_laplace(struct_img[zz, :, :], log_sigma)
        #bw = np.logical_or(bw, responce > s2_param[fid][1])
    return responce
