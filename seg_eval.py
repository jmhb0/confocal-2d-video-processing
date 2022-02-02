""" 
Functions for doing single-frame segmentations and producing pdfs 
for evaluation. 
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import segmentation_core as sc 
import pipeline_utils as pu
from matplotlib.backends.backend_pdf import PdfPages

def create_pdf_eval_seg_channels(img_in, fname_save, seg_funcs, seg_config, 
            order_channels, frame=0, do_pdf_txt=True,
            ac_norm_params= [[0,20],[0,13]], gs_smooth_params = [1,2.5],                   
            show_whole_seg=True, show_window_masks=True, show_windows=True,
            figsize_mult=7, max_display_cols=4, do_plot_dividing_lines=5,
            window_szs = dict(mito=200, peroxy=200, lyso=200, golgi=200, er=200)):
    """
    Produce a pdf of single-frame segmentation evaluation plots for a multiple 
    channels. 
    Args: 
        img_in: img with shape TCYX
    """
    assert img_in.ndim==4, "Shape must be TCYX"
    img_in = img_in[frame]
    with PdfPages(fname_save) as pdf:
        for i, channel in enumerate(order_channels):
            if do_pdf_txt:
                f = plt.figure(); plt.axis('off')
                plt.text(0.5,0.5,channel,ha='center',va='center', 
                         fontsize='xx-large', fontstretch='expanded')
                pdf.savefig(f)
            
            # do segmentation and get intermediates
            img = img_in[[i]]
            seg, imgs_inter = sc.seg_2d(img[0], seg_config, seg_funcs, 
                            ch_name=channel, inplace=False, save_inters=True)

            # show original & its segmentation
            if show_whole_seg:
                f, axs = plt.subplots(1,2,figsize=(20,20))
                axs[0].imshow(img[0], cmap='gray'); axs[0].set(title="Original")
                axs[1].imshow(seg[0], cmap='gray'); axs[1].set(title="Semantic Segmentation")
                pdf.savefig(f, bbox_inches='tight'
                            , dpi=400
                           )

            # get the windowed plot and the mask plot
#             window_sz = window_szs[channel]
            f, show_mask, noshow_mask = pu.compare_seg_windows(img, seg, imgs_inter=imgs_inter,
                                    max_display_cols=max_display_cols, figsize_mult=figsize_mult,
#                                     window_sz=window_sz, stride=window_sz,
                                    do_plot_dividing_lines=do_plot_dividing_lines,
                                    ac_norm_params=ac_norm_params,gs_smooth_params=gs_smooth_params, 
                                    es_smooth_params=None)

            # save the masked window part if applicable 
            if show_window_masks:
                f_tmp, axs = plt.subplots(1,2, figsize=(20,20))
                axs[0].imshow(noshow_mask, cmap='gray');    axs[0].set(title="Ignored windows")
                axs[1].imshow(show_mask, cmap='gray')  ;    axs[1].set(title="Included widnows")
                pdf.savefig(f_tmp, bbox_inches='tight')
                
            # finally save the windowed part that was computed in `pu.compare_seg_windows` 
            pdf.savefig(f, bbox_inches='tight')
            plt.close('all')
            
def process_snr_files(fname_files, fname_seg_func_map, dir_configs, dir_data_a, dir_data_b,
                     dir_result_summaries, order_channels=['mito','peroxy','lyso', 'golgi', 'er'],
                     ac_norm_params=[[0,20],[0,6]], gs_smooth_params=[1,4], frame=0, verbose=1,):
    """
    Calls `create_pdf_eval_seg_channels`
    Takes a csv holding a list of files indexed by "A0", "A1" etc, and a directory
    for config files named "Ao-config.yaml" .... Does single-frame segmentations and 
    creates the evaluation images for them
    """
    df_files = pd.read_csv(fname_files)
    seg_funcs = sc.load_segmentation_functions(fname_seg_func_map)
    for i, row in df_files.iterrows():
        # handle data getting 
        index = row['index']
        data_dir = dir_data_a if 'A' in index else dir_data_b
        fname_base = row['name']
        fname_img = f"{data_dir}/{fname_base}.czi"
        fname_save = f"{dir_result_summaries}/{index}-{fname_base}.pdf"        
        if verbose: print(f"Doing {index} with fname {fname_img}")

        # load the image
        img_in, aics_object = pu.read_czi_img(fname_img, return_aics_object=True
                                              , order_channels=order_channels)
        # load the config 
        seg_config = sc.load_segmentation_config_from_file(f"{dir_configs}/{index}-config.yaml", 
                        seg_funcs=seg_funcs, verbose=0)

        create_pdf_eval_seg_channels(img_in, fname_save, seg_funcs, seg_config, 
                order_channels, frame=frame, do_pdf_txt=True,
                ac_norm_params= ac_norm_params, gs_smooth_params = gs_smooth_params,                   
                show_whole_seg=True, show_window_masks=True, show_windows=True,
                figsize_mult=7, max_display_cols=4, do_plot_dividing_lines=5,
                window_szs = dict(mito=200, peroxy=200, lyso=100, golgi=200, er=100))

def do_june_snr_evaluations():
    fname_files = "fname-lookup.csv"
    dir_data_a = "/Users/jamesburgess/vae-bio-image/data/raw_data/ips-snr-tests/hiPSCs_setA"
    dir_data_b = "/Users/jamesburgess/vae-bio-image/data/raw_data/ips-snr-tests/hiPSCs_setB"
    dir_img_pipeline = "/Users/jamesburgess/vae-bio-image/img-pipeline"
    dir_configs = f"{dir_img_pipeline}/snr-tests-june/configs"
    dir_result_summaries = f"{dir_img_pipeline}/snr-tests-june/results_evaluate_frames"
    fname_seg_func_map = f"{dir_img_pipeline}/seg-func-map.yaml"
    fname_files = f"{dir_img_pipeline}/snr-tests-june/fname-lookup.csv"

    frame=0
    verbose=1
    ac_norm_params=[[0,20],[0,6]]
    gs_smooth_params=[1,2.5]
    order_channels=['mito','peroxy','lyso', 'golgi', 'er']

    process_snr_files(fname_files, fname_seg_func_map, dir_configs, dir_data_a, dir_data_b,
                      dir_result_summaries, frame=frame, verbose=verbose, order_channels=order_channels,
                    gs_smooth_params=gs_smooth_params, ac_norm_params=ac_norm_params)
