import cv2
import numpy as np
from numpy import genfromtxt

__all__ = ['load_raw_labels', 'resize_labels', 'get_frames_from_idxs']

def load_raw_labels(labels_arr, pose_algo, likelihood_thresh=0.9):
    """Load labels and build masks from a variety of standardized source files.

    This function currently supports the loading of csv and h5 files output by DeepLabCut (DLC) and
    Deep Graph Pose (DGP).

    Parameters
    ----------
    file_path : :obj:`str`
        absolute file path of label file
    pose_algo : :obj:`str`
        'dlc' | 'dgp'
    likelihood_thresh : :obj:`float`
        likelihood threshold used to define masks; any labels/timepoints with a likelihood below
        this value will be set to NaN and the corresponding masks file with have a 0

    Returns
    -------
    :obj:`tuple`
        - (array-like): labels, all x-values first, then all y-values
        - (array-like): masks; 1s correspond to good values, 0s correspond to bad values

    """
    if pose_algo == 'dlc' or pose_algo == 'dgp':
        labels_tmp = labels_arr.to_numpy().astype('float') # get rid of headers, etc.
        xvals = labels_tmp[:, 0::3]
        yvals = labels_tmp[:, 1::3]
        likes = labels_tmp[:, 2::3]
        labels = np.hstack([xvals, yvals])
        likes = np.hstack([likes, likes])
        masks = 1.0 * (likes >= likelihood_thresh)
        labels[masks != 1] = np.nan
    elif pose_algo == 'dpk':
        raise NotImplementedError
    elif pose_algo == 'leap':
        raise NotImplementedError
    else:
        raise NotImplementedError('the pose algorithm "%s" is currently unsupported' % pose_algo)

    return labels, masks


def resize_labels(labels, xpix_new, ypix_new, xpix_old, ypix_old):
    """Update label values to reflect scale of corresponding images.

    Parameters
    ----------
    labels : :obj:`array-like`
        np.ndarray of shape (n_time, 2 * n_labels); for a given row, all x-values come first,
        followed by all y-values
    xpix_new : :obj:`int`
        xpix of new images
    ypix_new : :obj:`int`
        ypix of new images
    xpix_old : :obj:`int`
        xpix of original images
    ypix_old : :obj:`int`
        ypix of original images


    Returns
    -------
    array-like
        resized label values

    """
    if xpix_new is None or ypix_new is None:
        return labels
    else:
        n_labels = labels.shape[1] // 2
        old = np.array([xpix_old] * n_labels + [ypix_old] * n_labels)
        new = np.array([xpix_new] * n_labels + [ypix_new] * n_labels)
        labels_scale = (labels / old) * new
        return labels_scale


def get_frames_from_idxs(cap, idxs):
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    if n_frames == 0:
        return None
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print('warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames

