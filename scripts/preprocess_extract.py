import os
from attrs import fields
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats, interpolate
import numpy as np

# 1. Define the direct path (omit .hea and .dat)
file_path = "/Users/monicamihailescu/Desktop/Documents/PPG Summer Research 2026/ppg_annotation_tool/data/raw/p00/p000052/3238451_0005_0_1"

# Load the record
record = wfdb.rdrecord(file_path)

# plot the record to screen
wfdb.plot_wfdb(record=record, title='Example signals')

# Extract the signal data and channel names
signals = record.p_signal
channels = record.sig_name
#print(channels, signals.shape[0]) # rows [0]: 30s at 125Hz = 3750 samples; cols [1] : 3 channels (PPG, Lead II, RESP)

# Apply butterworth bandpass filter to the PPG signal (channel 1)
ppg_signal = signals[:, 0] 
fs = 125  # Sampling frequency from the record
lowcut = 0.4  
highcut = 5.0   

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design the filter coefficients (b, a)
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply the filter forward and backward (zero phase shift)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

filtered_ppg = butter_bandpass_filter(ppg_signal, lowcut, highcut, fs)

# Plot the original and filtered PPG signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ppg_signal, label='Original PPG Signal')
plt.title('Original PPG Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(filtered_ppg, label='Filtered PPG Signal', color='orange')
plt.title('Filtered PPG Signal (Bandpass 0.4-5 Hz)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.tight_layout()
#plt.show()

#time = [i / fs for i in range(len(ppg_signal))]
time = np.arange(len(ppg_signal)) / fs
#plot 10s window of the filtered signal to see changes 
start_sample = 0
end_sample = int(10 * fs) # 10 seconds
plt.figure(figsize=(12, 6))
plt.plot(time[start_sample:end_sample], filtered_ppg[start_sample:end_sample], label='Filtered PPG Signal')
plt.title('Filtered PPG Signal (Bandpass 0.4-5 Hz) - First 10s')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
#plt.show()  

## Normalize the filtered signal (z-score normalization)
normalized_ppg = (filtered_ppg - np.mean(filtered_ppg)) / np.std(filtered_ppg) ##Is it supposed to look like this??

# Plot the normalized PPG signal
plt.figure(figsize=(12, 6))
plt.plot(time, normalized_ppg, label='Normalized Filtered PPG Signal', color='green')
plt.title('Normalized Filtered PPG Signal (Bandpass 0.4-5 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude') 
#plt.show()     

# Check that the normalized signal has mean 0 and std dev 1
print(f"New Mean: {np.mean(normalized_ppg):.4f}")  
print(f"New Std Dev: {np.std(normalized_ppg):.4f}")

#------------- Feature Extraction --------------
from scipy.signal import find_peaks
# Find peaks and troughs with physiological constraints
# distance=36 samples ensures beats aren't closer than ~210 bpm
# height=0.5 ensures we only catch true systolic peaks well above the mean
ppg_peaks, properties = signal.find_peaks(normalized_ppg, distance=36, height=0.5)
ppg_troughs, _ = signal.find_peaks(-normalized_ppg, distance=36, height=-0.5)
ppg_hr = len(ppg_peaks) * (60 / (len(normalized_ppg) / fs))  # Calculate heart rate in bpm
print(f"\nEstimated Heart Rate: {ppg_hr:.0f} bpm")

# compare with ground truth heart rate from ecg channel (channel 2)
ecg_signal = signals[:, 1]
ecg_peaks, _ = signal.find_peaks(ecg_signal, distance=36, height=1)
ecg_hr = len(ecg_peaks) * (60 / (len(ecg_signal) / fs))
print(f"Ground Truth Heart Rate from ECG: {ecg_hr:.0f} bpm")  
if abs(ppg_hr - ecg_hr) >= 2:
    print("Warning: Estimated PPG heart rate differs from ECG ground truth by 2 bpm or more.")
#plot the detected peaks and troughson the normalized PPG signal
plt.plot(time, normalized_ppg, label='Normalized Filtered PPG Signal', color='green')
plt.title('Normalized Filtered PPG Signal with Detected Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.plot(time[ppg_peaks], normalized_ppg[ppg_peaks], 'ro', label='Detected Peaks')
plt.plot(time[ppg_troughs], normalized_ppg[ppg_troughs], 'bx', label='Detected Troughs')
#plt.show()

# 1. Skewness of the normalized PPG signal
from scipy.stats import skew
ppg_skewness = skew(normalized_ppg)
print(f"\n1. PPG Signal Skewness: {ppg_skewness:.3f}")

# 2. Interbeat interval (IBI) Stability in seconds
ibi = np.diff(ppg_peaks) / fs
# calculate overall stability score (SDNN)
ibi_stability = np.std(ibi)
print(f"2. IBI Stability (SDNN): {ibi_stability:.3f} s")

# 3. Template Correlation - compare to median ppg signals from all beats
min_bpm = 30
max_bpm = 220

min_samples = int(fs * 60 / max_bpm)  # = 34 samples
max_samples = int(fs * 60 / min_bpm)  # = 250 samples
# --- a. extract segments using troughs 
raw_beats = []
for i in range(len(ppg_troughs) - 1):
    start = ppg_troughs[i]
    end = ppg_troughs[i + 1]
    beat = normalized_ppg[start:end]
    if min_samples < len(beat) < max_samples:  # filter out too short/noisy beats
        raw_beats.append(beat)

# --- b. interpolate each beat to fixed length
fixed_len = 100
resampled_beats = []
for beat in raw_beats:
    x_old = np.linspace(0, 1, len(beat))
    x_new = np.linspace(0, 1, fixed_len)
    f = interpolate.interp1d(x_old, beat, kind='linear')
    resampled_beats.append(f(x_new))

if len(raw_beats) == 0:
    print("3. Template Correlation: No valid beats found")
else:
    resampled_beats = np.array(resampled_beats)
# --- c. build median template
    template = np.median(resampled_beats, axis=0)
# --- d. calculate pearson correlation of each beat to template
    correlations = np.array([stats.pearsonr(beat, template).statistic for beat in resampled_beats])
    template_correlation_mean = np.mean(correlations) 
    print(f"3. Template Correlation (mean r): {template_correlation_mean:.3f}")
    print(f"   Beats used: {len(raw_beats)}, Beats rejected: {len(ppg_troughs)-1 - len(raw_beats)}")

print(correlations)

# 4. Perfusion Index (PI) - AC/DC ratio
ac_component = np.mean(ppg_signal[ppg_peaks]) - np.mean(ppg_signal[ppg_troughs])  # AC = peak - trough
dc_component = np.mean(ppg_signal)  # DC = mean of the signal
if dc_component != 0:
    perfusion_index = ac_component / dc_component
    print(f"4. Perfusion Index (PI): {perfusion_index:.3f}")
else:
    print("4. Perfusion Index (PI): DC component is zero, cannot calculate PI.")    

# 5. SNR 
# SNR = power in the PPG frequency band / power outside it (noise)
freqs, psd = signal.welch(normalized_ppg, fs=fs, nperseg=512) # estimates PSD - power at each frequency, 
# nperseg is the number of samples per segment. 30s * 125Hz = 3750 samples, so with 512, frequwncy resolution = fs/nperseg = 0.244 Hz
# ppg signal band is 0.4-5 Hz, so 512 should be a good power of 2 to use. 

signal_band = (freqs >= lowcut) & (freqs <= highcut)
noise_band = ~signal_band

signal_power = np.trapezoid(psd[signal_band], freqs[signal_band])  # integrates the area under the PSD curve within a frequency band (total power in that band)
noise_power = np.trapezoid(psd[noise_band], freqs[noise_band])

snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf #calculate in DB 
print(f"5. SNR: {snr:.2f} dB")