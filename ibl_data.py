import cv2
import h5py
import numpy as np
import os
import argparse
import brainbox.io.one as bbone
from numpy import genfromtxt
from one.api import ONE
from brainbox.processing import bincount2D

def build_hdf5_for_decoding(
        save_file, video_file, spikes, trial_data=None, labels=None, pose_algo=None, xpix=None,
        ypix=None, label_likelihood_thresh=0.9, zscore=True):
    """Build Behavenet-style HDF5 file from video file and optional label file.

    This function provides a basic example for how to convert raw video and label files into the
    processed version required by Behavenet. In doing so no additional assumptions are made about
    a possible trial structure; equally-sized batches are created. For more complex data, users
    will need to adapt this function to suit their own needs.

    Parameters
    ----------
    save_file : :obj:`str`
        absolute file path of new HDF5 file; the directory does not need to be created beforehand
    video_file : :obj:`str`
        absolute file path of the video (.mp4, .avi)
    label_file : :obj:`str`, optional
        absolute file path of the labels; current formats include DLC/DGP csv or h5 files
    trial_data : :obj:'numpy ndarray', optional
        array of trial beg. and end times; numpy array - found as _ibl_trial.intervals.npy
    pose_algo : :obj:`str`, optional
        'dlc' | 'dgp'
    batch_size : :obj:`int`, optional
        uniform batch size of data if no trial file
    xpix : :obj:`int`, optional
        if not None, video frames will be reshaped before storing in the HDF5
    ypix : :obj:`int`, optional
        if not None, video frames will be reshaped before storing in the HDF5
    label_likelihood_thresh : :obj:`float`, optional
        likelihood threshold used to define masks; any labels/timepoints with a likelihood below
        this value will be set to NaN
    zscore : :obj:`bool`, optional
        individually z-score each label before saving in the HDF5

    """

    # load video capture
    video_cap = cv2.VideoCapture(video_file)
    n_total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    xpix_og = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ypix_og = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # load labels
    if labels is not None:
        labels, masks = load_raw_labels(
            labels, pose_algo=pose_algo, likelihood_thresh=label_likelihood_thresh)
        # error check
        n_total_labels = labels.shape[0]
        assert n_total_frames == n_total_labels, 'Number of frames does not match number of labels'

    # assign trial information based on trial file or uniform batch size
    if trial_data is not None:
        trial_info = trial_data
        n_trials = len(trial_info)
    else: 
        n_trials = int(np.ceil(n_total_frames / batch_size))
    trials = np.arange(n_trials)

    timestamps = np.arange(n_total_frames)

    # compute z-score params
    if labels is not None and zscore:
        means = np.nanmean(labels, axis=0)
        stds = np.nanstd(labels, axis=0)
    else:
        means = None
        stds = None

    # create directory for hdf5 if it doesn't already exist
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    with h5py.File(save_file, 'w', libver='latest', swmr=True) as f:

        # single write multi-read
        f.swmr_mode = True

        # create image group
        group_i = f.create_group('images')

        if labels is not None:
            # create labels group (z-scored)
            group_ls = f.create_group('labels_sc')

            # create label mask group
            group_m = f.create_group('labels_masks')

            # create labels group (not z-scored, but downsampled if necessary)
            group_l = f.create_group('labels')

        if spikes is not None: 
            # create neural group 
            group_n = f.create_group('neural')

        # create a dataset for each trial within groups
        for tr_idx, trial in enumerate(trials):

            # find video timestamps during this trial
            trial_beg = trial_info[trial][0] * fps
            trial_end = trial_info[trial][1] * fps

            ts_idxs = np.where((timestamps >= trial_beg) & (timestamps < trial_end))[0]

            # ----------------------------------------------------------------------------
            # image data
            # ----------------------------------------------------------------------------
            # collect from video capture, downsample
            frames_tmp = get_frames_from_idxs(video_cap, ts_idxs)
            if xpix is not None and ypix is not None:
                # Nones to add batch/channel dims
                frames_tmp = [cv2.resize(f[0], (xpix, ypix))[None, None, ...] for f in frames_tmp]
            else:
                frames_tmp = [f[None, ...] for f in frames_tmp]
            group_i.create_dataset(
                'trial_%04i' % tr_idx, data=np.vstack(frames_tmp), dtype='uint8')

            # ----------------------------------------------------------------------------
            # neural data
            # ----------------------------------------------------------------------------
            group_n.create_dataset(
                'trial_%04i' % tr_idx, data=spikes, dtype='uint8')

            # ----------------------------------------------------------------------------
            # label data
            # ----------------------------------------------------------------------------
            if labels is not None:
                # label masks
                group_m.create_dataset('trial_%04i' % tr_idx, data=masks[ts_idxs], dtype='float32')

                # label data (zscored, masked)
                # labels_tmp = (labels[ts_idxs] - means) / stds
                # labels_tmp[masks[ts_idxs] == 0] = 0  # pytorch doesn't play well with nans
                # assert ~np.any(np.isnan(labels_tmp))
                # group_ls.create_dataset('trial_%04i' % tr_idx, data=labels_tmp, dtype='float32')

                # label data (non-zscored, masked)
                labels_tmp = labels[ts_idxs]
                labels_tmp = resize_labels(labels_tmp, xpix, ypix, xpix_og, ypix_og)
                labels_tmp[masks[ts_idxs] == 0] = 0
                group_l.create_dataset('trial_%04i' % tr_idx, data=labels_tmp, dtype='float32')


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


def get_spike_data_per_trial(times, clusters, intervals, binsize=0.02):
    """Select spiking data for specified interval on each trial.
    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    interval_begs : array-like
        beginning of each interval in seconds
    interval_ends : array-like
        end of each interval in seconds
    binsize : float
        width of each bin in seconds; default 20 ms
    Returns
    -------
    tuple
        - (list): time in seconds for each trial
        - (list): data for each trial of shape (n_clusters, n_bins)
    """

    interval_begs = intervals[:, 0]
    interval_ends = intervals[:, 1]
    n_trials = len(interval_begs)
    n_bins = int((interval_ends[0] - interval_begs[0]) / binsize) + 1
    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = np.zeros((n_trials, n_clusters_in_region, n_bins))
    spike_times_list = []
    for tr, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        # just get spikes for this region/trial
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)

        # update data block
        binned_spikes[tr, idxs_tmp, :] += binned_spikes_tmp[:, :n_bins]
        spike_times_list.append(t_idxs[:n_bins])

    return spike_times_list, binned_spikes


def get_spike_trial_data(times, clusters, intervals, binsize=0.02):
    """Select spiking data for specified interval on each trial.
    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    interval_begs : array-like
        beginning of each interval in seconds
    interval_ends : array-like
        end of each interval in seconds
    binsize : float
        width of each bin in seconds; default 20 ms
    Returns
    -------
    tuple
        - (list): time in seconds for each trial
        - (list): data for each trial of shape (n_clusters, n_bins)
    """

    n_trials = len(intervals)
    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = []
    spike_times_list = []
    for tr, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        # just get spikes for this region/trial
        idxs_t = (times >= t_beg) & (times < t_end)
        temp_n_bins = int((t_end - t_beg) / binsize) + 1
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        binned_spikes_tmp = np.zeros((temp_n_bins, n_clusters_in_region))
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            bincount_spikes, t_idxs, cluster_idxs = bincount2D(
                clust_curr, times_curr, ybin=binsize, ylim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
            binned_spikes_tmp[:, idxs_tmp] += bincount_spikes

        # update data block
        binned_spikes.append(binned_spikes_tmp)
        spike_times_list.append(t_idxs)

    return spike_times_list, np.asarray(binned_spikes)


def main(save_dir, eid):
    # create directory for raw data if it doesn't already exist
    if not os.path.exists(os.path.dirname(save_dir + '/raw_data')):
        os.makedirs(os.path.dirname(save_dir + '/raw_data'))

    # initialize one object to retrieve data
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', cache_dir=save_dir + '/raw_data',
          password='international', silent=True)

    # get datasets (currently only using left cam data)
    try:
        cam_data = one.load_dataset(eid, f'raw_video_data/_iblrig_leftCamera.raw.mp4')
    except:
        print('raw video data not available')
        return

    # get trial dataset if it exists
    try:
        trial_data = one.load_dataset(eid, f'alf/_ibl_trials.intervals.npy')
    except:
        print('trial data not available')
        trial_data = None

    # get dlc label data if it exists
    try:
        label_data = one.load_dataset(eid, f'alf/_ibl_leftCamera.dlc.pqt')
    except:
        print('dlc data not available')
        label_data = None

    # Get spikes dict 
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
    spike_times = spikes['probe00']['times']
    spike_clusters = spikes['probe00']['clusters']

    spike_times_list, binned_spikes = get_spike_data_per_trial(spike_times, spike_clusters, trial_data, float(1000/60))

    build_hdf5_for_decoding(save_dir + '/data.hdf5', str(cam_data), binned_spikes, trial_data, label_data, 'dlc', xpix=160, ypix=128)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("IBL data retrieval, processing, and saving")
    parser.add_argument("-dir", "--save_dir", help="The absolute directory path where to save the raw data and HDF5 file - do not include / at the end of the path (directory does not need to exist)", type=str)
    parser.add_argument("-e", "--eid", help="experiment id for IBL data to retrieve.", type=str)
    args = parser.parse_args()

    # retreiving params
    save_dir = args.save_dir
    eid = args.eid

    main(save_dir, eid)





