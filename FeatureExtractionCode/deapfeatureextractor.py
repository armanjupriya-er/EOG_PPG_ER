import csv
import os
import pickle

import numpy as np
from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch
from scipy.integrate import simps


EEG_CHANNELS = {1: 'FP1', 2: 'AF3', 3: 'F3', 4: 'F7', 5: 'FC5', 6: 'FC1', 7: 'C3', 8: 'T7', 9: 'CP5', 10: 'CP1',
                11: 'P3', 12: 'P7', 13: 'PO3', 14: 'O1', 15: 'Oz', 16: 'Pz', 17: 'Fp2', 18: 'AF4', 19: 'Fz', 20: 'F4',
                21: 'F8', 22: 'FC6', 23: 'FC2', 24: 'Cz', 25: 'C4', 26: 'T8', 27: 'CP6', 28: 'CP2', 29: 'P4', 30: 'P8',
                31: 'PO4', 32: 'O2', 33: 'hEOG', 34: 'vEOG', 35: 'zEMG', 36: 'tEMG', 37: 'GSR', 38: 'Respiration belt',
                39: 'Plethysmograph', 40: 'Temperature', }
# PreProcess_Channel_Order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
#                             'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
#                             'P4', 'P8', 'PO4', 'O2', 'hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Respiration belt',
#                             'Plethysmograph', 'Temperature', ]
Features_data_file = r"C:\Users\armanjupriya\Downloads\DEAP_Features.csv"

FEATURES = ['Trial', 'Subject', 'FP1_HjorthActivity', 'FP1_HjorthComplexity', 'ALPHA_FP1_Welch_PSD',
            'BETA_FP1_Welch_PSD', 'GAMMA_FP1_Welch_PSD', 'THETA_FP1_Welch_PSD', 'DELTA_FP1_Welch_PSD',
            'AF3_HjorthActivity', 'AF3_HjorthComplexity', 'ALPHA_AF3_Welch_PSD', 'BETA_AF3_Welch_PSD',
            'GAMMA_AF3_Welch_PSD', 'THETA_AF3_Welch_PSD', 'DELTA_AF3_Welch_PSD', 'F3_HjorthActivity',
            'F3_HjorthComplexity', 'ALPHA_F3_Welch_PSD', 'BETA_F3_Welch_PSD', 'GAMMA_F3_Welch_PSD',
            'THETA_F3_Welch_PSD',
            'DELTA_F3_Welch_PSD', 'F7_HjorthActivity', 'F7_HjorthComplexity', 'ALPHA_F7_Welch_PSD', 'BETA_F7_Welch_PSD',
            'GAMMA_F7_Welch_PSD', 'THETA_F7_Welch_PSD', 'DELTA_F7_Welch_PSD', 'FC5_HjorthActivity',
            'FC5_HjorthComplexity', 'ALPHA_FC5_Welch_PSD', 'BETA_FC5_Welch_PSD', 'GAMMA_FC5_Welch_PSD',
            'THETA_FC5_Welch_PSD', 'DELTA_FC5_Welch_PSD', 'FC1_HjorthActivity', 'FC1_HjorthComplexity',
            'ALPHA_FC1_Welch_PSD', 'BETA_FC1_Welch_PSD', 'GAMMA_FC1_Welch_PSD', 'THETA_FC1_Welch_PSD',
            'DELTA_FC1_Welch_PSD', 'C3_HjorthActivity', 'C3_HjorthComplexity', 'ALPHA_C3_Welch_PSD',
            'BETA_C3_Welch_PSD',
            'GAMMA_C3_Welch_PSD', 'THETA_C3_Welch_PSD', 'DELTA_C3_Welch_PSD', 'T7_HjorthActivity',
            'T7_HjorthComplexity',
            'ALPHA_T7_Welch_PSD', 'BETA_T7_Welch_PSD', 'GAMMA_T7_Welch_PSD', 'THETA_T7_Welch_PSD', 'DELTA_T7_Welch_PSD',
            'CP5_HjorthActivity', 'CP5_HjorthComplexity', 'ALPHA_CP5_Welch_PSD', 'BETA_CP5_Welch_PSD',
            'GAMMA_CP5_Welch_PSD', 'THETA_CP5_Welch_PSD', 'DELTA_CP5_Welch_PSD', 'CP1_HjorthActivity',
            'CP1_HjorthComplexity', 'ALPHA_CP1_Welch_PSD', 'BETA_CP1_Welch_PSD', 'GAMMA_CP1_Welch_PSD',
            'THETA_CP1_Welch_PSD', 'DELTA_CP1_Welch_PSD', 'P3_HjorthActivity', 'P3_HjorthComplexity',
            'ALPHA_P3_Welch_PSD', 'BETA_P3_Welch_PSD', 'GAMMA_P3_Welch_PSD', 'THETA_P3_Welch_PSD', 'DELTA_P3_Welch_PSD',
            'P7_HjorthActivity', 'P7_HjorthComplexity', 'ALPHA_P7_Welch_PSD', 'BETA_P7_Welch_PSD', 'GAMMA_P7_Welch_PSD',
            'THETA_P7_Welch_PSD', 'DELTA_P7_Welch_PSD', 'PO3_HjorthActivity', 'PO3_HjorthComplexity',
            'ALPHA_PO3_Welch_PSD', 'BETA_PO3_Welch_PSD', 'GAMMA_PO3_Welch_PSD', 'THETA_PO3_Welch_PSD',
            'DELTA_PO3_Welch_PSD', 'O1_HjorthActivity', 'O1_HjorthComplexity', 'ALPHA_O1_Welch_PSD',
            'BETA_O1_Welch_PSD',
            'GAMMA_O1_Welch_PSD', 'THETA_O1_Welch_PSD', 'DELTA_O1_Welch_PSD', 'Oz_HjorthActivity',
            'Oz_HjorthComplexity',
            'ALPHA_Oz_Welch_PSD', 'BETA_Oz_Welch_PSD', 'GAMMA_Oz_Welch_PSD', 'THETA_Oz_Welch_PSD', 'DELTA_Oz_Welch_PSD',
            'Pz_HjorthActivity', 'Pz_HjorthComplexity', 'ALPHA_Pz_Welch_PSD', 'BETA_Pz_Welch_PSD', 'GAMMA_Pz_Welch_PSD',
            'THETA_Pz_Welch_PSD', 'DELTA_Pz_Welch_PSD', 'Fp2_HjorthActivity', 'Fp2_HjorthComplexity',
            'ALPHA_Fp2_Welch_PSD', 'BETA_Fp2_Welch_PSD', 'GAMMA_Fp2_Welch_PSD', 'THETA_Fp2_Welch_PSD',
            'DELTA_Fp2_Welch_PSD', 'AF4_HjorthActivity', 'AF4_HjorthComplexity', 'ALPHA_AF4_Welch_PSD',
            'BETA_AF4_Welch_PSD', 'GAMMA_AF4_Welch_PSD', 'THETA_AF4_Welch_PSD', 'DELTA_AF4_Welch_PSD',
            'Fz_HjorthActivity', 'Fz_HjorthComplexity', 'ALPHA_Fz_Welch_PSD', 'BETA_Fz_Welch_PSD', 'GAMMA_Fz_Welch_PSD',
            'THETA_Fz_Welch_PSD', 'DELTA_Fz_Welch_PSD', 'F4_HjorthActivity', 'F4_HjorthComplexity',
            'ALPHA_F4_Welch_PSD',
            'BETA_F4_Welch_PSD', 'GAMMA_F4_Welch_PSD', 'THETA_F4_Welch_PSD', 'DELTA_F4_Welch_PSD', 'F8_HjorthActivity',
            'F8_HjorthComplexity', 'ALPHA_F8_Welch_PSD', 'BETA_F8_Welch_PSD', 'GAMMA_F8_Welch_PSD',
            'THETA_F8_Welch_PSD',
            'DELTA_F8_Welch_PSD', 'FC6_HjorthActivity', 'FC6_HjorthComplexity', 'ALPHA_FC6_Welch_PSD',
            'BETA_FC6_Welch_PSD', 'GAMMA_FC6_Welch_PSD', 'THETA_FC6_Welch_PSD', 'DELTA_FC6_Welch_PSD',
            'FC2_HjorthActivity', 'FC2_HjorthComplexity', 'ALPHA_FC2_Welch_PSD', 'BETA_FC2_Welch_PSD',
            'GAMMA_FC2_Welch_PSD', 'THETA_FC2_Welch_PSD', 'DELTA_FC2_Welch_PSD', 'Cz_HjorthActivity',
            'Cz_HjorthComplexity', 'ALPHA_Cz_Welch_PSD', 'BETA_Cz_Welch_PSD', 'GAMMA_Cz_Welch_PSD',
            'THETA_Cz_Welch_PSD',
            'DELTA_Cz_Welch_PSD', 'C4_HjorthActivity', 'C4_HjorthComplexity', 'ALPHA_C4_Welch_PSD', 'BETA_C4_Welch_PSD',
            'GAMMA_C4_Welch_PSD', 'THETA_C4_Welch_PSD', 'DELTA_C4_Welch_PSD', 'T8_HjorthActivity',
            'T8_HjorthComplexity',
            'ALPHA_T8_Welch_PSD', 'BETA_T8_Welch_PSD', 'GAMMA_T8_Welch_PSD', 'THETA_T8_Welch_PSD', 'DELTA_T8_Welch_PSD',
            'CP6_HjorthActivity', 'CP6_HjorthComplexity', 'ALPHA_CP6_Welch_PSD', 'BETA_CP6_Welch_PSD',
            'GAMMA_CP6_Welch_PSD', 'THETA_CP6_Welch_PSD', 'DELTA_CP6_Welch_PSD', 'CP2_HjorthActivity',
            'CP2_HjorthComplexity', 'ALPHA_CP2_Welch_PSD', 'BETA_CP2_Welch_PSD', 'GAMMA_CP2_Welch_PSD',
            'THETA_CP2_Welch_PSD', 'DELTA_CP2_Welch_PSD', 'P4_HjorthActivity', 'P4_HjorthComplexity',
            'ALPHA_P4_Welch_PSD', 'BETA_P4_Welch_PSD', 'GAMMA_P4_Welch_PSD', 'THETA_P4_Welch_PSD', 'DELTA_P4_Welch_PSD',
            'P8_HjorthActivity', 'P8_HjorthComplexity', 'ALPHA_P8_Welch_PSD', 'BETA_P8_Welch_PSD', 'GAMMA_P8_Welch_PSD',
            'THETA_P8_Welch_PSD', 'DELTA_P8_Welch_PSD', 'PO4_HjorthActivity', 'PO4_HjorthComplexity',
            'ALPHA_PO4_Welch_PSD', 'BETA_PO4_Welch_PSD', 'GAMMA_PO4_Welch_PSD', 'THETA_PO4_Welch_PSD',
            'DELTA_PO4_Welch_PSD', 'O2_HjorthActivity', 'O2_HjorthComplexity', 'ALPHA_O2_Welch_PSD',
            'BETA_O2_Welch_PSD',
            'GAMMA_O2_Welch_PSD', 'THETA_O2_Welch_PSD', 'DELTA_O2_Welch_PSD', 'hEOG_HjorthActivity',
            'hEOG_HjorthComplexity', 'ALPHA_hEOG_Welch_PSD', 'BETA_hEOG_Welch_PSD', 'GAMMA_hEOG_Welch_PSD',
            'THETA_hEOG_Welch_PSD', 'DELTA_hEOG_Welch_PSD', 'vEOG_HjorthActivity', 'vEOG_HjorthComplexity',
            'ALPHA_vEOG_Welch_PSD', 'BETA_vEOG_Welch_PSD', 'GAMMA_vEOG_Welch_PSD', 'THETA_vEOG_Welch_PSD',
            'DELTA_vEOG_Welch_PSD', 'zEMG_HjorthActivity', 'zEMG_HjorthComplexity', 'ALPHA_zEMG_Welch_PSD',
            'BETA_zEMG_Welch_PSD', 'GAMMA_zEMG_Welch_PSD', 'THETA_zEMG_Welch_PSD', 'DELTA_zEMG_Welch_PSD',
            'tEMG_HjorthActivity', 'tEMG_HjorthComplexity', 'ALPHA_tEMG_Welch_PSD', 'BETA_tEMG_Welch_PSD',
            'GAMMA_tEMG_Welch_PSD', 'THETA_tEMG_Welch_PSD', 'DELTA_tEMG_Welch_PSD', 'GSR_HjorthActivity',
            'GSR_HjorthComplexity', 'ALPHA_GSR_Welch_PSD', 'BETA_GSR_Welch_PSD', 'GAMMA_GSR_Welch_PSD',
            'THETA_GSR_Welch_PSD', 'DELTA_GSR_Welch_PSD', 'Respiration belt_HjorthActivity',
            'Respiration belt_HjorthComplexity', 'ALPHA_Respiration belt_Welch_PSD', 'BETA_Respiration belt_Welch_PSD',
            'GAMMA_Respiration belt_Welch_PSD', 'THETA_Respiration belt_Welch_PSD', 'DELTA_Respiration belt_Welch_PSD',
            'Plethysmograph_HjorthActivity', 'Plethysmograph_HjorthComplexity', 'ALPHA_Plethysmograph_Welch_PSD',
            'BETA_Plethysmograph_Welch_PSD', 'GAMMA_Plethysmograph_Welch_PSD', 'THETA_Plethysmograph_Welch_PSD',
            'DELTA_Plethysmograph_Welch_PSD', 'Temperature_HjorthActivity', 'Temperature_HjorthComplexity',
            'ALPHA_Temperature_Welch_PSD', 'BETA_Temperature_Welch_PSD', 'GAMMA_Temperature_Welch_PSD',
            'THETA_Temperature_Welch_PSD', 'DELTA_Temperature_Welch_PSD', 'valence', 'arousal', 'dominance', 'liking',
            'valence_class', 'dominance_class', 'arousal_class', 'liking_class', 'valence_arousal_class']

FREQUENCY_BANDS = {'ALPHA': (8, 12), 'BETA': (12, 30), 'GAMMA': (30, 45), 'THETA': (4, 8), 'DELTA': (0, 4)}
EOG_ARTIFACTS_CHANNELS = []
PHYSIOLOGICAL_CHANNELS = {33: 'hEOG', 34: 'vEOG', 35: 'zEMG', 36: 'tEMG', 37: 'GSR', 38: 'Respiration belt',
                          39: 'Plethysmograph', 40: 'Temperature'}
PSD_METHOD = {1: "WELCH", 2: "MULTI_TAPER"}
SAMPLING_FREQUENCY = 512
DOWN_SAMPLED_FREQUENCY = 256
WINDOW_SIZE_IN_SECS = 2
CLASS_CROSS_OVER = 5


def isnumpyarray(func):
    def function_wrapper(wave_data):
        if not (type(wave_data) == np.ndarray):
            raise IncorrectDataType("Input needs to be of numpy.ndarray type")
        return func(wave_data)

    return function_wrapper


class HjorthFeatures:
    def __init__(self):
        pass

    @staticmethod
    @isnumpyarray
    def activity(wave_data):
        return np.var(wave_data)

    @staticmethod
    @isnumpyarray
    def time_dertivative(wave_data):
        return np.gradient(wave_data)

    @staticmethod
    @isnumpyarray
    def mobility(wave_data):
        return np.sqrt(np.var(HjorthFeatures.time_dertivative(wave_data)) / np.var(wave_data))

    @staticmethod
    @isnumpyarray
    def complexity(wave_data):
        wave_data = np.array(wave_data)
        return HjorthFeatures.mobility(HjorthFeatures.time_dertivative(wave_data)) / HjorthFeatures.mobility(wave_data)


def load_data(file_name):
    x = pickle.load(open(file_name, 'rb'), encoding="bytes")
    labels = x[b'labels']
    eeg_data = x[b'data']
    return eeg_data, labels



def get_channel_wise_data(complete_data):
    channel_wise_data = {}
    for i in range(1, 41):
        channel_name = EEG_CHANNELS[i]
        channel_data = complete_data[i - 1]
        # print("channel_name:", channel_name, len(channel_data))
        channel_wise_data[channel_name] = channel_data
    return channel_wise_data


def get_frequency_bands_for_channel(channel_eeg_data, channel_name):
    data = {}
    fft_vals = np.fft.rfft(channel_eeg_data)
    fft_freqs = np.fft.rfftfreq(len(channel_eeg_data), 1.0 / DOWN_SAMPLED_FREQUENCY)
    # print("channel_freq_data:", fft_freqs, len(fft_vals))
    for frequency_band in FREQUENCY_BANDS:
        freq_ix = np.where((fft_freqs >= FREQUENCY_BANDS[frequency_band][0]) &
                           (fft_freqs <= FREQUENCY_BANDS[frequency_band][1]))
        # print("channel_freq_data:", freq_ix)
        channel_freq_data = fft_vals[freq_ix]
        # print("channel_freq_data:", len(channel_freq_data))
        data[channel_name + "_" + frequency_band] = channel_freq_data
    return data


def compute_psd(data, method=1):
    if data.any():
        if method == 1:
            freqs, psd = welch(data, fs=DOWN_SAMPLED_FREQUENCY, nperseg=DOWN_SAMPLED_FREQUENCY * WINDOW_SIZE_IN_SECS)
        elif method == 2:
            psd, freqs = psd_array_multitaper(data, DOWN_SAMPLED_FREQUENCY, adaptive=True,
                                              normalization='full', verbose=0)
        return freqs, psd


def compute_features(complete_data, trial=None, subject=None):
    data = {'Trial': trial, 'Subject': subject}
    channel_data = get_channel_wise_data(complete_data)
    temp_data = {}
    for ch in channel_data:
        print("***********************Processing Channel '", ch, "' for subject ", subject, " [Trial No: ", trial, "]",
              "****************************")
        temp_data[ch + "_Welch_PSD"] = compute_psd(channel_data[ch], 1)
        data[ch + "_HjorthMobility"] = HjorthFeatures.mobility(channel_data[ch])
        data[ch + "_HjorthActivity"] = HjorthFeatures.activity(channel_data[ch])
        data[ch + "_HjorthComplexity"] = HjorthFeatures.complexity(channel_data[ch])
        for freq_band in FREQUENCY_BANDS:
            band = FREQUENCY_BANDS[freq_band]
            welchfreqs, welchpsd = temp_data[ch + "_Welch_PSD"]
            welchfreq_res = welchfreqs[1] - welchfreqs[0]
            welchidx_band = np.logical_and(welchfreqs >= band[0], welchfreqs <= band[1])
            data[freq_band + "_" + ch + "_Welch_PSD"] = simps(welchpsd[welchidx_band], dx=welchfreq_res)
    print(data.keys())
    return data


def get_class(value):
    if value <= CLASS_CROSS_OVER:
        return 0
    return 1


def get_four_classes(valence, arousal):
    if valence <= CLASS_CROSS_OVER and arousal <= CLASS_CROSS_OVER:
        return 0
    elif valence > CLASS_CROSS_OVER and arousal <= CLASS_CROSS_OVER:
        return 1
    elif valence <= CLASS_CROSS_OVER and arousal > CLASS_CROSS_OVER:
        return 2
    elif valence > CLASS_CROSS_OVER and arousal > CLASS_CROSS_OVER:
        return 3
    else:
        return 4


def write_data_for_subject(file_name, subject=None):
    eeg_data, labels = load_data(file_name)
    for i in range(0, 40):
        label = labels[i]
        # {'valence': label[0], 'arousal': label[1], 'dominance': label[2], 'liking': label[3]}
        feature_data = compute_features(complete_data=eeg_data[i], trial=i + 1, subject=subject)
        feature_data.update({'valence': label[0], 'arousal': label[1],
                             'dominance': label[2], 'liking': label[3],
                             'valence_class': get_class(label[0]), 'arousal_class': get_class(label[1]),
                             'dominance_class': get_class(label[2]), 'liking_class': get_class(label[3]),
                             'valence_arousal_class': get_four_classes(valence=label[0], arousal=label[1])})
        with open(Features_data_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FEATURES, extrasaction='ignore')
            writer.writerow(feature_data)
            csvfile.flush()


def get_features_for_data_set(data_set_path):
    files = []
    if os.path.isdir(data_set_path):
        files = os.listdir(data_set_path)
    elif os.path.isfile(data_set_path):
        files = [data_set_path]
    for file_name in files:
        subject = file_name.split(".")[0]
        full_path = os.path.join(data_set_path, file_name)
        print("Processing:", full_path)
        write_data_for_subject(full_path, subject=subject)


def write_headers():
    with open(Features_data_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FEATURES)
        writer.writeheader()
        csvfile.flush()


if __name__ == "__main__":
    write_headers()
    data_set_path = r"F:\DEAP\data_preprocessed_python"
    #data_set_path = r"F:\DEAP\data_preprocessed_python\s01.dat"
    get_features_for_data_set(data_set_path)
