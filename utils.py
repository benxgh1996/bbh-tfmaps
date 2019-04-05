import numpy as np
import pycbc


# @param confMat (array): The rows represent the actual classes,
#   while the columns represent the predicted classes.
# @param attNames (array_like<str_like>): A collection of class names.
def printConfMat(mat, attNames):
    print "Rows (Actual), Columns (Predicted)"
    print "=" * 50
    maxLen = 0
    minWidth = 5
    for s in attNames:
        if len(s) > maxLen:
            maxLen = len(s)
    if maxLen < minWidth:
        maxLen = minWidth
    w = maxLen + 4
    print "{:{width}}".format("", width=w),
    for att in attNames:
        print "{:>{width}}".format(att, width=w),
    print

    for i, att in enumerate(attNames):
        print "{:>{width}}".format(att, width=w),
        for j in range(len(attNames)):
            print "{:{width}}".format(mat[i][j], width=w),
        print
    print "=" * 50


# Extract the waveform name from its file PATH
# wvpath is a string of form xx/xx/xxxx.h5
# An example wavePath is "GT0577". Notice that the ".h5"
# extension will be removed.
def get_wvname(wvpath):
    wvname = wvpath.split("/")[-1]
    wvname = wvname.split(".")[0]
    return wvname


# This function select points in wplane that are necessary
# for identifying the double chirp feature. Essentially, we
# want to select only the points near the given middle time.
# Besides, this function also returns the selected parts of
# samples_times and wfreqs.

# We want our selected points to have a height and width of
# exponentials of 2. So 64 * 64 might be a good choice.

# The types of the 3 parameters below are natural types as
# a result from waveform generation and tf decomposition.
# wplane (numpy.ndarray)
# sample_times (pycbc.types.array.Array)
# wfreqs (numpy.ndarray)

# x-axis represents the frequency axis, while y represents time.
# xnum: number of points to be selected along the x-axis.
# ynum: number of points to be selected along the y-axis.

# left_window must be a negative number.
# left_window: the left time window of the interesting region.

# right_window: the right time window of the interesting region.

# freq_window: the total number of interesting frequency points,
# over which we will downsample into xnum points.
# The entire frequency window will be (0, 0 + freq_window - 1).
# Note that the freq_window only refers to index rather than
# actual frequency values. So the number, 0, in the above
# description means the index for the first freq value in wfreqs.
# The default frequency window is carefully selected to be 505
# because 505=63*8+1. Therefore, 505 is able to give 64 selected
# points exactly.
def select_wplane(wplane, wfreqs, sample_times, mid_t=0,
                  xnum=64, ynum=64,
                  left_t_window=-0.05, right_t_window=0.05,
                 freq_window=505):
    # Should try the best to make the passed-in freq_window work
    # exactly with xnum.
    assert (freq_window - 1) % (xnum - 1) == 0, "Freq window not" \
        + " divided evenly into number of selected freq points."

    if isinstance(sample_times, pycbc.types.array.Array):
        sample_times = np.array(sample_times)
        # sample_times = sample_times.numpy()

    # Since the left and right time windows might not be exact multiples
    # of the time step and we cannot directly control the number of
    # time points included within this window, at this stage we do not
    # know the number of time points included within this window.
    # However, we should still expect this window to be relatively small
    # compared to the entire array of sample_times. Therefore we should
    # still feel free to select an arbitrary number of points within this
    # time window.

    # Locating the left and right interesting indices for sample_times.
    left_t_idx = np.searchsorted(sample_times, mid_t+left_t_window)
    right_t_idx = np.searchsorted(sample_times, mid_t+right_t_window)

    # Obtaining an interesting region within wplane.
    # To emphasize, xl, xr, yl, yr are all indices, rather than physical
    # values.
    # I am naming these variables to mimic the style in function
    # select_points(arr, region, xnum, ynum).
    xl, xr = 0, 0 + freq_window - 1
    yl, yr = left_t_idx, right_t_idx
    reg = (xl, xr, yl, yr)

    # Obtaining the selected points within wplane.
    wplane_sel, right_borders = select_points(wplane, reg, xnum, ynum)
    xr_new, yr_new = right_borders

    # Obtaining the selected parts of sample_times and wfreqs.
    x_step = (xr_new - 0) / (xnum - 1)
    y_step = (yr_new - yl) / (ynum - 1)
    assert xl + x_step * (xnum - 1) == xr_new
    assert yl + y_step * (ynum - 1) == yr_new
    freqs_sel = wfreqs[0:xr_new+1:x_step]
    times_sel = sample_times[yl:yr_new+1:y_step]

    return wplane_sel, freqs_sel, times_sel


# Select points from a 2-D array. The indices are selected with equal spacing
# with regard to both the x-axis and y-axis.
# arr: the 2-D array.

# region: a tuple (x_left, x_right, y_left, y_right) indicating the
# region only within which point selection will happen. In here,
# x_left, x_right, y_left, and y_right are all only indices of arr for
# points in that region.

# xnum: the number of x-indices to be selected.
# ynum: the number of y-indices to be selected.
# Neither xnum, ynum can be 1.
def select_points(arr, region, xnum, ynum):
    xl, xr, yl, yr = region

    # These are the initial number of x and y points that are of interst.
    xlen = xr - xl + 1
    ylen = yr - yl + 1

    # We want to slightly increase (or not) xlen and ylen so that
    # these numbers can be seen as *---*---*---* where * represents
    # the selected points, and - represents the unselected points
    # in the interesting region.

    # The minus signs are due to the *---*---* structure.
    # The plus sign is due to adding in the last *.
    xlen = rndup_dvd(xlen-1, xnum-1) + 1
    ylen = rndup_dvd(ylen-1, ynum-1) + 1

    # Need to calculate number of *--- in each periodic group.
    xstep = (xlen - 1) / (xnum - 1)
    ystep = (ylen - 1) / (ynum - 1)
    assert xlen - 1 == (xnum - 1) * xstep
    assert ylen - 1 == (ynum - 1) * ystep

    # Move the interesting right borders to the "right" to accommodate
    # the initial interesting number of points. Notice that xr and yr
    # mark indices that are indeed to be selected.
    # These right-border indices also need to be returned because
    # we will need them to select points from the times and frequencies
    # arrays that act as independent variables for wplane, i.e. arr in here.
    xr = xl + xlen - 1
    yr = yl + ylen - 1

    region_sel = arr[xl:xr+1:xstep, yl:yr+1:ystep]
    return region_sel, (xr, yr)


# Increase an int (dvd) by a minimum amount so that
# dvd can be divided exactly by another int (dvr).
def rndup_dvd(dvd, dvr):
    rmd = dvd % dvr
    if rmd == 0:
        return dvd
    else:
        return dvd + (dvr - rmd)


# Converts a numerical angular value to string.
# This is useful when quoting the angular value in the name of a plot.
# Argument ang must be in radian.
def ang_to_str(ang, to_deg=True, dec=2, show_pi=True):
    PI = np.pi
    if to_deg:
        return "{:.{prec}f}dg".format(ang*180/PI, prec=dec)
    elif show_pi:
        return "{:.{prec}f}pi".format(ang/PI, prec=dec)
    else:
        return "{:.{prec}f}".format(ang, prec=dec)


# Assuming the elements of arr are in increasing order.
# Want to find the lower limit (inclusive) and the upper limit
# (exclusive) of the indices of the
# argument array whose corresponding values arr[i] having low < arr[i] < upp.
# Return a tuple (low_idx, upp_idx), where range(low_idx, upp_idx) will
# give desried range of indices.
def get_idx_limits(arr, low, upp):
    if low >= upp:
        raise RuntimeError("Lower value not less than upper value.")

    found_low = False
    found_upp = False

    for i in range(len(arr)):
        if arr[i] > low:
            low_idx = i
            found_low = True
            break
    if found_low == False:
        raise RuntimeError("Lower value too large.")
    if arr[low_idx] >= upp:
        raise RuntimeError("Window between lower and upper limits too small.")

    for i in range(low_idx+1, len(arr)):
        if arr[i] >= upp:
            upp_idx = i
            found_upp = True
            break
    if found_upp == False:
        upp_idx = len(arr)

    return (low_idx, upp_idx)



def print_sorted_dict(dict):
    for key in sorted(dict):
        print key, ": ", dict[key]

# Obtain the multi-dimensional array elements that lie between the
# lower and upper bounds.
# Returned data is a dictionary of {idx: value}
def test_array_value(arr, lower, upper):
    dimen = len(arr.shape)
    idx_list = _search_values(arr, dimen-1, lower, upper)
    res = _get_values(arr, idx_list)
    return res

# Returns a list of indices of array elements that lie between the
# lower and upper bounds.
def _search_values(sub_array, curr_depth, lower, upper):
    # curr_depth = 0 means reaching the last index of an array.
    # So, curr_depth = dimen-1 means the first
    # index layer of an array.
    idx_list = []

    if curr_depth == 0:
        for i in range(len(sub_array)):
            if lower < sub_array[i] < upper:
                idx_list.append([i])
        return idx_list

    for i in range(len(sub_array)):
        rt_idx_list = _search_values(sub_array[i], curr_depth-1, lower, upper)
        for rt_idx in rt_idx_list:
            idx_list.append([i]+rt_idx)

    return idx_list

# Obtain the element value in the arr corresponding to idx_list
def _get_values(arr, idx_list):
    res = {}
    for idx in idx_list:
        val = reduce(lambda x, y: x[y], idx, arr)
        res[tuple(idx)] = val
    return res


# Return the summed-over-time frequency spectrum over a certain time window.
# The time windows is exclusive of both the low_time and upp_time.
def get_t_sum_spectrum(low_time, upp_time, t_range, f_range, tfmap):
    t_start, t_end = get_idx_limits(t_range, low_time, upp_time)
    f_spec = np.empty(len(f_range))

#     print t_range[t_start: t_end]

#     print len(f_spec)
    for i in range(len(f_range)):
#         print i
#         print tfmap[i, t_start:t_end]
        f_spec[i] = tfmap[i, t_start:t_end].sum()

    return f_spec


# Obtain the waveform name, say GT0577, from the waveform PATH,
# say "/waves/GT0577.h5".
# Note that this only works for UNIX file systems where "/" is used as
# separators in file PATH.
def get_waveform_name(numrel_data):
    name = numrel_data.split("/")[-1]
    name = name.split(".")[0]
    return name
