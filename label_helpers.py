import cv2
import numpy as np
from numpy import genfromtxt

__all__ = ['load_raw_labels', 'resize_labels', 'get_frames_from_idxs', 'get_highest_me_trials']

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


def get_highest_me_trials(markers_2d, batch_size, n_batches):
    """Find trials with highest motion energy to help with batch selection.

    Parameters
    ----------
    markers_2d : dict
        keys are camera names; vals are themselves dicts with marker names; those vals are arrays
        of shape (n_timepoints, 2), i.e.
        >> points_2d['left']['paw_l'].shape
        >> (100, 2)
    batch_size : int
        number of contiguous time points per batch
    n_batches : int
        total number of batches to add to hdf5

    Returns
    -------
    array-like
        trial indices of the `n_batches` trials with highest motion energy (sorted low to high)
    """

    # just use paws to compute motion energy
    if isinstance(markers_2d, dict):
        vll = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['left']['paw_l'], axis=0)])
        vlr = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['left']['paw_r'], axis=0)])
        vrr = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['right']['paw_r'], axis=0)])
        vrl = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['right']['paw_l'], axis=0)])
        me_all = np.abs(np.hstack([vll, vlr, vrr, vrl]))
    else:
        me_all = np.abs(
            np.vstack([np.zeros((1, markers_2d.shape[1])), np.diff(markers_2d, axis=0)]))

    n_total_frames = me_all.shape[0]
    n_trials = int(np.ceil(n_total_frames / batch_size))
    assert n_trials >= batch_size

    total_me = np.zeros(n_trials)
    for trial in range(n_trials):
        trial_beg = trial * batch_size
        trial_end = (trial + 1) * batch_size
        total_me[trial] = np.nanmean(me_all[trial_beg:trial_end])

    total_me[np.isnan(total_me)] = -100  # nans get pushed to end of sorted array
    sorted_me_idxs = np.argsort(total_me)
    best_trials = sorted_me_idxs[-n_batches:]

    return best_trials

