import numpy as np

__all__ = ['bincount2D', 'get_spike_trial_data', 'get_binsize']


def get_spike_trial_data(times, clusters, intervals, binsize=0.02):
    """Select spiking data for specified interval on each trial.
    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    intervals : array-like
        beginning and end of each interval in seconds
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
    for tr, (t_beg, t_end) in enumerate(intervals):
        # just get spikes for this region/trial
        idxs_t = (times >= t_beg) & (times < t_end)
        temp_n_bins = int((t_end - t_beg) / binsize) + 1
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((temp_n_bins, n_clusters_in_region))
            t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            bincount_spikes, cluster_idxs, t_idxs = bincount2D(
                clust_curr, times_curr, ybin=binsize, ylim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
            binned_spikes_tmp = np.zeros((bincount_spikes.shape[0], n_clusters_in_region))
            binned_spikes_tmp[:, idxs_tmp] += bincount_spikes

        # update data block
        binned_spikes.append(binned_spikes_tmp)
        spike_times_list.append(t_idxs)

    return spike_times_list, binned_spikes


def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None):
    """
    Computes a 2D histogram by aggregating values in a 2D array.

    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin:
        scalar: bin size along 2nd dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param ybin:
        scalar: bin size along 1st dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    # if no bounds provided, use min/max of vectors
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    def _get_scale_and_indices(v, bin, lim):
        # if bin is a nonzero scalar, this is a bin size: create scale and indices
        if np.isscalar(bin) and bin != 0:
            # to match the number of cam frames
            scale_beg = np.ceil(round(lim[0]/bin, 3))
            scale_end = np.ceil(lim[1]/bin)
            scale = np.arange(scale_beg, scale_end)
            ind = (np.floor((v - lim[0]) / bin)).astype(np.int64)
        # if bin == 0, aggregate over unique values
        else:
            scale, ind = np.unique(v, return_inverse=True)
        return scale, ind

    xscale, xind = _get_scale_and_indices(x, xbin, xlim)
    yscale, yind = _get_scale_and_indices(y, ybin, ylim)
    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx), mode='clip')
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)

    # if a set of specific values is requested output an array matching the scale dimensions
    if not np.isscalar(xbin) and xbin.size > 1:
        _, iout, ir = np.intersect1d(xbin, xscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ny, xbin.size))
        r[:, iout] = _r[:, ir]
        xscale = xbin

    if not np.isscalar(ybin) and ybin.size > 1:
        _, iout, ir = np.intersect1d(ybin, yscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ybin.size, r.shape[1]))
        r[iout, :] = _r[ir, :]
        yscale = ybin

    return r, xscale, yscale


def get_binsize(str_times):
        times_arr = []
    for i in range(len(str_times)):
        # get the timestamp without day information
        time_str = str_times[i][11:27]
        # hour
        time = int(time_str[0:2]) * 3600
        # minutes
        time += int(time_str[3:5]) * 60
        # seconds
        time += float(time_str[6:16])
        
        times_arr.append(time)
    assert len(times_arr) == len(str_times)

    return np.median(np.diff(times_arr))
