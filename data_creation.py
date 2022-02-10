import cv2
import h5py
import numpy as np
import os
import argparse
import brainbox.io.one as bbone
from numpy import genfromtxt
from one.api import ONE

from label_helpers import load_raw_labels
from label_helpers import resize_labels
from label_helpers import get_frames_from_idxs
from label_helpers import get_highest_me_trials
from neural_helpers import get_spike_trial_data

def build_hdf5_for_decoding(
        save_file, video_file, spikes, batch_size=None, num_batches=None, trial_data=None, labels=None, pose_algo=None, xpix=None,
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
        trials = np.arange(n_trials)
    elif trial_data is None and num_batches != None and batch_size != None:
        # get the num_batches highest motion energy trials - give x & y pix of left and right paws 
        trials = get_highest_me_trials(labels[:, np.asarray([5, 6, 16, 17])], batch_size, num_batches)
    else:
        n_trials = int(np.ceil(n_total_frames / batch_size))
        trials = np.arange(n_trials)

    if spikes is not None: 
        spike_times = spikes[0]
        spike_clusters = spikes[1]
        spike_times_list, binned_spikes = 
            get_spike_trial_data(spike_times, spike_clusters, trial_info, float(1/60))

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
            if trial_data is not None:
                trial_beg = trial_info[trial][0] * fps
                trial_end = trial_info[trial][1] * fps
            else:
                trial_beg = trial * batch_size
                trial_end = (trial + 1) * batch_size
            
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
            spike_times_list, binned_spikes = 
                get_spike_trial_data(spike_times, spike_clusters, [[trial_beg, trial_end]], float(1/60))
            group_n.create_dataset(
                'trial_%04i' % tr_idx, data=binned_spikes[0], dtype='uint8')

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


def main(save_dir, eid, xpix, ypix, batch_size, num_batches):
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

    build_hdf5_for_decoding(save_dir + '/data.hdf5', str(cam_data), [spike_times, spike_clusters], batch_size, 
        num_batches, labels=label_data, pose_algo='dlc', xpix=xpix, ypix=ypix)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("IBL data retrieval, processing, and saving")
    parser.add_argument("-dir", "--save_dir", 
        help="The absolute directory path where to save the raw data and HDF5 file - do not include / at the end of the path (directory does not need to exist)", type=str)
    parser.add_argument("-e", "--eid", 
        help="experiment id for IBL data to retrieve.", type=str)
    parser.add_argument("-x", "--x_pix", 
        help="The x resolution to which the video will be downsampled - ex. 160", type=int)
    parser.add_argument("-y", "--y_pix", 
        help="The y resolution to which the video will be downsampled - ex. 128", type=int)
    parser.add_argument("-bs", "--batch_size", 
        help="The number of frames to include in a single trial", type=int)
    parser.add_argument("-nb", "--num_batches", 
        help="The number of trials used train the model", type=int)
    args = parser.parse_args()

    # retreiving params
    save_dir = args.save_dir
    eid = args.eid
    xpix = args.x_pix
    ypix = args.y_pix
    batch_size = args.batch_size != None ? args.batch_size : None
    num_batches = args.num_batches != None ? args.num_batches : None

    main(save_dir, eid, xpix, ypix, batch_size, num_batches)





