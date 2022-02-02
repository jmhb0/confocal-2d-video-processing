import numpy as np
from skimage import measure 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
import importlib
from sort import sort

def mask_to_bbox_detections_2d(img, default_score=1):
    """
    Get bounding box coordinates around distinct objects in a 2d image. 

    Args:
        imgs: 2d np.array. If 0's and 1's (binary mask), then assume its semantic
            seg mask. Otherwise I assume (without checking) that img is in the 
            form output from skimage.measure.label()
        default_score: the detection score that is passed in to SORT.
    Returns:
        detections: for N objects, is shape (n,5) with the format:
            [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            This is what SORT expects in SORT.update() method
    """
    assert img.ndim==2
    if len(np.unique(img))==2:
        labels = measure.label(img, connectivity=1)
    else:
        labels = img

    labels_cnt = labels.max()
    detections = np.zeros((labels_cnt,5), dtype=np.uint16)
    for i in range(1, labels_cnt+1):
        obj_coords = np.argwhere(labels==i)
        x1, y1 = obj_coords[:,1].min(), obj_coords[:,0].min()
        x2, y2 = obj_coords[:,1].max(), obj_coords[:,0].max()
        detections[i-1] = np.array([x1,y1,x2,y2,default_score])
    return detections

def mask_to_bbox_detections_4d(img, verbose=0):
    """
    Call mask_to_bbox_detections_2d for each frame/channel. 
    Return as an OrderedDict `all_detections` so that all_detections[3,2] is 
    the bounding box list (the output from mask_to_bbox_detections_2d) for 
    frame 3, and channel 2. 
    """
    all_detections = OrderedDict()
    assert img.ndim == 4 
    frames, channels = img.shape[:2]

    for ch in range(channels):
        if verbose: print(f"\t channel {ch}")
        all_detections[ch] = OrderedDict()
        for fr in range(frames):
            if verbose>1: print(f"\t\t frame {fr}")
            detections = mask_to_bbox_detections_2d(img[fr,ch])
            all_detections[ch][fr] = detections
    return all_detections

def plot_img_with_bbox(img, bboxes, figsize=(9,9), bbox_offset=-1,
            bbox_kwargs=dict(linewidth=1, edgecolor='r', facecolor='none')
                       ):
    assert img.ndim==2
    f, axs = plt.subplots(figsize=figsize)
    plt.imshow(img, cmap='gray')
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i,:4]
        rect = patches.Rectangle((x1+bbox_offset, y1+bbox_offset), 
                (x2-x1), (y2-y1), **bbox_kwargs)
        axs.add_patch(rect)
    plt.close()
    return f

def run_sort_tracking(all_detections, ch, max_age=4, min_hits=1, iou_threshold=0.3):
    """
    Run SORT detections for a single channel given the detections/bbox info
    Args:
        all_detections: dict, output of `mask_to_bbox_detections_4d`
        ch: channel indexed in the keys of all_detections
        max_age: Maximum number of frames to keep alive a track without
            associated detections
        min_hits: Minimum number of associated detections before track is
            initialised
        iou_threshold: minimum IOU for match.
    """
    importlib.reload(sort)
    mot_tracker = None
    sort_tracks = OrderedDict()
    mot_tracker = sort.Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    n_frames = max(list(all_detections[0].keys()))
    for frame in range(n_frames):
        detections = all_detections[ch][frame]
        sort_tracks[frame] = mot_tracker.update(detections)

    return sort_tracks

def get_map_to_indx_color(sort_tracks, cmap=plt.cm.nipy_spectral):
    """
    Assign a random color from `cmap` to each indexed object found in sort_tracks
    as a dictionary. This is useful so that nearby bounding boxes are likely to
    have different colors, so we can keep track of a tracklet over time in
    a video.
    Args
        sort_tracks: dict, output of run_sort_tracking.
        cmap: colormap
    Returns:
        map_indx_to_color: dict with keys from range(1,max_indx) where max_indx
            is the highest object index in sort_tracks (which is the last column
            of the arrays that are the values of sort_tracks. The color is a 4-
            element array RGBA value.
    """
    # find the max indx
    max_tracklet_id = 0
    for v in sort_tracks.values():
        this_max = int(v[:,-1].max())
        max_tracklet_id = max(this_max, max_tracklet_id)

    # generate random numbers in the range [0.1,1] because we will map objects
    # randomly to the range of plt.cm.nipy_spectral, and the start of the range is black
    rand_mapping =  np.random.rand(max_tracklet_id) * 0.9 +0.1
    map_indx_to_color = dict(zip(np.arange(1,max_tracklet_id+1), cmap(rand_mapping)))
    return map_indx_to_color

def animate_imgs(imgs, figsize=(12,12), cmap='gray', fname_save=None,
                 animation_kwargs=dict(interval=200, repeat_delay=1000, blit=True)):
    """
    Usage:
            ani = animate_imgs(imgs, figsize=(8,8))
            from IPython.display import HTML
            HTML(ani.to_jshtml())

    Args
        imgs: array of images to pass to plt.imshow with cm
        interval: int, time (ms) between frames
    """
    frames = [] # for storing the generated images
    old_figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = figsize
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = old_figsize

    for i in range(n_frames):
        frames.append([plt.imshow(imgs[i], cmap=cmap,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, **animation_kwargs)
    if fname_save is not None:
        ani.save(fname_save)
    plt.close()
    return ani

def _draw_bboxes_to_axis(ax, sort_tracks, frame, map_indx_to_color,
                        bbox_kwargs=dict(linewidth=1, facecolor='none')):
    """
    Draw bounding boxes onto an axis object. 
    The bounding boxes array is sort_tracks[ch][frame]
    Args
        ax: the axis object on which to draw the bouding boxes
        sort: dict of arrays, where each array is in the format used by SORT.
        map_indx_to_color: dict mapping indx to color in RGBA. The indx is the
            last column of the arrays in bboxes.
        bbox_kwargs: dict to be passed directly to matplotlib.patches.Rectangle        
    Returns: 
        artists: lists of artist objects (the thing returned by ax.add_patch)
    """
    bboxes = sort_tracks[frame]
    artists=[]
    for i in range(len(bboxes)):
        x1, y1, x2, y2, indx = bboxes[i,:]
        color = map_indx_to_color[indx]
        bbox_kwargs['edgecolor']=color
        rect = patches.Rectangle((x1, y1), (x2-x1), (y2-y1), **bbox_kwargs)
        artists.append(ax.add_patch(rect))
    return artists

def normalize_0_1(img):
    l, u = img.min(), img.max()
    return (img-l)/(u-l)
left, right  = 0.0, 1.0    # the left side of the subplots of the figure
bottom, top = 0.0, 1.0   # the bottom of the subplots of the figure
wspace, hspace = 0.03, 0.03  # the amount of width reserved for blank space between subplots

def do_animation_wtracking_woriginal(img, img_original=None, sort_tracks=None, n_frames=10,
                                    fname_save=None, whitespace_width=10, figsize_mult=10,
                                    img_original_kwargs=dict(AC=[0,20], G2=1.5),
                                    animation_kwargs=dict(interval=300, repeat_delay=1000, blit=True),
                                    bbox_kwargs=dict(linewidth=2.5, facecolor='none')):
    """
    Args
        img: the segmentation image for a single channel with shape (frames,1,Y,X).
        img_original: the original image to display next to the segmentation.
            Should be same shape as img.
            If None, then don't display original image.
        sort_tracks: output of `run_sort_tracking` used for assigning bounding
            boxes of detected & linked objects.
            If None, then don't display bboxes.
    """
    assert img.shape==img_original.shape
    imgs=list(img[:,0,:,:])

    # make sure we have bboxes for all the frames that we want
    if sort_tracks: n_frames = min(n_frames, max(list(sort_tracks.keys())))

    if img_original is not None:
        # preprocessing the original
        img_original = ol.normalize_0_1(img_original)
        AC, G2 = img_original_kwargs.get('AC',None), img_original_kwargs.get('G2',None)
        if AC: img_original = intensity_normalization(img_original, AC)
        if G2: img_original = image_smoothing_gaussian_slice_by_slice(img_original, G2)

        # update imgs to put the original image next to the seg mask for each frame
        imgs_original=list(img_original[:,0,:,:])
        whitespace = np.ones((img_0.shape[1], whitespace_width))*0.5
        imgs = [np.hstack((imgs[i], whitespace, imgs_original[i]))
                for i in range(len(imgs))]

    # do plotting frames
    figsize=np.array([2,1])*figsize_mult
    fig, axs = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    frames=[]
    for i in range(n_frames):
        seg_img = plt.imshow(imgs[i], cmap='gray',animated=True)
        if sort_tracks:
            bbox_artists = ol._draw_bboxes_to_axis(axs, sort_tracks, i, map_indx_to_color, bbox_kwargs=bbox_kwargs)
            frames.append([seg_img]+bbox_artists)
        else:
            frames.append([seg_img])

    ani = animation.ArtistAnimation(fig, frames, **animation_kwargs)
    if fname_save: ani.save(fname_save)
    plt.close()
    return ani
