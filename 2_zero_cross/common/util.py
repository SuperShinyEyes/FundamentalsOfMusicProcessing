# util.py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import wave
import sys
import os
import os.path

import IPython.display as ipd

PI = np.pi


def load_wav(filepath, t_start = 0, t_end = sys.maxint) :
    """Load a wave file, which must be 22050Hz and 16bit and must be either
    mono or stereo. 
    Inputs:
        filepath: audio file
        t_start, t_end:  (optional) subrange of file to load (in seconds)
    Returns:
        a numpy floating-point array with a range of [-1, 1]
    """
    
    wf = wave.open(filepath)
    num_channels, sampwidth, fs, end, comptype, compname = wf.getparams()
    
    # for now, we will only accept 16 bit files at 22k
    assert(sampwidth == 2)
    assert(fs == 22050)

    # start frame, end frame, and duration in frames
    f_start = int(t_start * fs)
    f_end = min(int(t_end * fs), end)
    frames = f_end - f_start

    wf.setpos(f_start)
    raw_bytes = wf.readframes(frames)

    # convert raw data to numpy array, assuming int16 arrangement
    samples = np.fromstring(raw_bytes, dtype = np.int16)

    # convert from integer type to floating point, and scale to [-1, 1]
    samples = samples.astype(np.float)
    samples *= (1 / 32768.0)

    if num_channels == 1:
        return samples

    elif num_channels == 2:
        return 0.5 * (samples[0::2] + samples[1::2])

    else:
        raise('Can only handle mono or stereo wave files')

def save_wav(channels, fs, filepath) :
    """Interleave channels and write out wave file as 16bit audio.
    Inputs:
        channels: a tuple or list of np.arrays. Or can be a single np.array in which case this will be a mono file.
                  format of np.array is floating [-1, 1]
        fs: sampling rate
        filepath: output filepath
    """

    if type(channels) == tuple or type(channels) == list:
        num_channels = len(channels)
    else:
        num_channels = 1
        channels = [channels]

    length = min ([len(c) for c in channels])
    data = np.empty(length*num_channels, np.float)

    # interleave channels:
    for n in range(num_channels):
        data[n::num_channels] = channels[n][:length]

    data *= 32768.0
    data = data.astype(np.int16)
    data = data.tostring()

    wf = wave.open(filepath, 'w')
    wf.setnchannels(num_channels)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(data)


def load_annotations(filepath) :
    '''Load annotations from a tab-separated text file where each line is one annotation and all annotations are numbers
    Input:
        filepath: annotation file
    Return:
        np.array (MxN) with annotation data
    '''
    lines = open(filepath).readlines()
    return np.array([float(l.split('\t')[0]) for l in lines])

def write_annotations(data, filepath) :
    f = open(filepath, 'w')
    for d in data:
        f.write('%f\n' % d)

def get_directory_files(dirpath, file_ext = None):
    '''Return all files in a directory
    Inputs:
        dirpath: directory name
        file_ext: (optional) only return files ending with that extension.
    '''
    files = os.listdir(dirpath)
    return [os.path.join(dirpath, f) for f in files if file_ext == None or f.endswith(file_ext)]

def plot_and_listen(filepath, len_t = 0) :
    """Plot the audio waveform and create an audio listening widget.
    Inputs:
        filepath: audio file
        len_t: (optional) only load the first len_t seconds of audio.
    Returns:
        IPython.display.Audio object for listening
    """
    if len_t != 0:
        x = load_wav(filepath, 0, len_t)
    else:
        x = load_wav(filepath)
    fs = 22050
    t = np.arange(len(x)) / float(fs)
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("time (secs)")
    plt.show()
    return ipd.Audio(x, rate=fs)

def plot_fft_and_listen(filepath, raw_axis = False) :
    """Plot the audio waveform and create an audio listening widget.
    Inputs:
        filepath: audio file
        raw_axis: (optional)
    Returns:
        IPython.display.Audio object for listening
    """
    fs = 22050
    x = load_wav(filepath)
    x_ft = np.abs(np.fft.fft(x))

    time = np.arange(len(x),dtype=np.float) / fs
    freq = np.arange(len(x_ft), dtype=np.float) / len(x_ft) * fs

    if raw_axis:
        print 'sample rate:', fs
        print 'N: ', len(x)

    plt.figure()
    plt.subplot(2,1,1)
    if raw_axis:
        plt.plot(x)
        plt.xlabel('n')
        plt.ylabel('$x(n)$')
    else:
        plt.plot(time, x)
        plt.xlabel('time')

    plt.subplot(2,1,2)
    if raw_axis:
        plt.plot(x_ft)
        plt.xlabel('k')
        plt.ylabel('$|X(k)|$')
        plt.xlim(0, 3000*len(x) / fs)
    else:
        plt.plot(freq, x_ft)
        plt.xlim(0, 3000)
        plt.xlabel('Frequency (Hz)')

    return ipd.Audio(x, rate=fs)




def find_peaks(x, thresh = 0.2) :
    ''' finds peaks in 1D vector.
    Inputs:
        x: input vector
        thresh: relative threshold value. Discard peak whose value is lower than (thresh * max_peak_value).
    Output:
        np.array of index values of peaks
    '''

    x0 = x[:-2]   # x
    x1 = x[1:-1]  # x shifted by 1
    x2 = x[2:]    # x shifted by 2

    peak_bools = np.logical_and(x0 < x1, x1 > x2) # where x1 is higher than surroundings
    values = x1[peak_bools]                       # list of all peak values

    # find a threshold relative to the highest peak
    th = np.max(values) * thresh
    
    # filter out values that are below th
    peak_bools = np.logical_and(peak_bools, x1 > th)

    peaks = np.nonzero( peak_bools )[0] + 1       # get indexes of peaks, shift by 1
    return peaks


def find_highest_peaks(x, N) :
    ''' finds largest peaks in 1D vector.
    Inputs:
        x: input vector
        N: number of peaks to return
    Output:
        np.array of index values of peaks
    '''

    peaks = find_peaks(x)
    vis = [(x[i], i) for i in peaks]
    vis.sort(reverse=True)
    return [x[1] for x in vis[:N]]


def plot_spectrogram(spec, cmap=None, colorbar=True) :
    mag_spec = abs(spec)
    maxval = np.max(mag_spec)
    minval = .1
    plt.imshow(mag_spec, origin='lower', interpolation='nearest', aspect='auto', 
        norm=LogNorm(vmin=minval, vmax=maxval), cmap=cmap)
    if colorbar:
        plt.colorbar()


def plot_two_chromas(c1, c2, cmap = 'Greys'):
    '''plot two chromagrams with subplots(2,1,1) and (2,1,2). Ensure that vmin and vmax are the same
    for both chromagrams'''

    plt.subplot(2,1,1)
    _min = 0.5 * ( np.min(c1) + np.min(c2) )
    _max = 0.5 * ( np.max(c1) + np.max(c2) )
    plt.imshow(c1, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap, vmin=_min, vmax=_max)
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(c2, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap, vmin=_min, vmax=_max)
    plt.colorbar()

