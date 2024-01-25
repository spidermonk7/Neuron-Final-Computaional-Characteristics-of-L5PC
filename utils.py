from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Load the template file
import random
from time import time
from preprocess import *
import torch
import os
import argparse
from L5PC_cell import HPC


"""
This is a file with some utility functions that are used in the main file.
"""


# A function for generate spike trian based on some specific distribution
def generaete_spike_train(distribution, train_numbers, spike_numbser, seed=100, start_time=100, end_time=600, use_scale = True, scale = 75):
    """
    the result is the list of spikes, each element in the list is a spike train
    they should be like: [[start_time1, start_time2, ...], [start_time1, start_time2, ...], ...]

    """
    # set seed first
    np.random.seed(seed)

    spike_train_set = []
    for i in range(train_numbers):
        if distribution == 'uniform':
            spike_train = np.random.uniform(start_time, end_time, spike_numbser)
        elif distribution == 'normal':
            mu = (end_time + start_time)/2
            sigma = (mu - start_time)/2
            print(f"mu: {mu}, sigma:{sigma}")
            spike_train = np.random.normal(mu, sigma, spike_numbser)
        elif distribution == 'laplacian':
            mu = (end_time + start_time)/2
            sigma = (mu - start_time)/3
            if use_scale:
                spike_train = np.random.laplace(mu, scale, spike_numbser) 
            else:
                spike_train = np.random.laplace(mu, sigma, spike_numbser)
        
        else:
            raise ValueError('Distribution error: only support uniform, normal and poisson distribution')
        
        # make sure the spike train is all positive and in range
        # spike_train = np.abs(spike_train)
        spike_train = np.clip(spike_train, start_time, end_time)
        spike_train_set.append(spike_train)
        # sort the spike train
        spike_train_set[i].sort()

    return spike_train_set



# A function for mapping the spike train to the cell
def map_method(cell_apical_num, cell_basal_num, spike_train_set, seed):
    positions = cell_apical_num + cell_basal_num
    np.random.seed(seed)
    apical_spike_dic = {}
    basal_spike_dic = {}
    # randomly map the spike train to the cell
    # first, we need to make sure the spike train is not empty

    apical_spike_num = 0
    basal_spike_num = 0

    if len(spike_train_set) == 0:
        raise ValueError('Spike train set is empty')
    # print("the len of spike train set: ", len(spike_train_set))
    # print("the spike trianset: ", spike_train_set)
    for i in range(len(spike_train_set)):
        # randomly choose a position
        position = np.random.randint(0, positions)
        if position < cell_apical_num:
            # apical
            apical_spike_num += 1
            if position in apical_spike_dic:
                apical_spike_dic[position].append(spike_train_set[i])
            else:
                apical_spike_dic[position] = [spike_train_set[i]]
        else:
            # basal
            position = position - cell_apical_num
            basal_spike_num += 1
            if position in basal_spike_dic:
                basal_spike_dic[position].append(spike_train_set[i])
            else:
                basal_spike_dic[position] = [spike_train_set[i]]
    # print(f"the len apical: {apical_spike_num}, the len basal: {basal_spike_num}")
    return apical_spike_dic, apical_spike_num, basal_spike_dic, basal_spike_num


# A function for adding noise
def add_noise(apical_spike_dic, apical_spike_num, basal_spike_dic, basal_spike_num, noise_distribution, location_distribution, noise_level, seed):
    """
    noise_distribution: 'uniform', 'normal', 'poisson', 'laplacian'
    location_distribution: spike/noise [x, y]
    """
    # check valid
    if noise_distribution not in ['uniform', 'normal', 'poisson', 'laplacian']:
        raise ValueError('Noise distribution error: only support uniform, normal, poisson and laplacian')

    # set seed first
    np.random.seed(seed)
    # print(f"the len apical: {(apical_spike_num)}, the len basal: {(basal_spike_num)}")
    apic_noise_num = apical_spike_num * location_distribution[0]
    basal_noise_num = basal_spike_num * location_distribution[1]
    # print(f"the len apical: {(apic_noise_num)}, the len basal: {(basal_noise_num)}, location distribution: {location_distribution}")

    cnt_apic = 0
    cnt_basal = 0


    noised_apical_spike_dic = {}
    noised_basal_spike_dic = {}
    # generate noise according to distribution and add them to the spike train
    for key in apical_spike_dic:
        for spike in range(len(apical_spike_dic[key])):
            leng = len(apical_spike_dic[key][spike])
            # generate noise
            if noise_distribution == 'uniform':
                noise = np.random.uniform(-noise_level/2, noise_level/2, leng)
            elif noise_distribution == 'normal':
                noise = np.random.normal(0, noise_level/4, leng)
            elif noise_distribution == 'laplacian':
                noise = np.random.laplace(0, noise_level/6, leng)

            to_be_appended = apical_spike_dic[key][spike]
            if cnt_apic < apic_noise_num:
                to_be_appended = to_be_appended + noise
                cnt_apic += 1
            if key in noised_apical_spike_dic:
                noised_apical_spike_dic[key].append(to_be_appended)
            else:
                noised_apical_spike_dic[key] = []
                noised_apical_spike_dic[key].append(to_be_appended)
        
    print(location_distribution, apic_noise_num, basal_noise_num)
    for key in basal_spike_dic:
        for spike in range(len(basal_spike_dic[key])):
            leng = len(basal_spike_dic[key][spike])
            # generate noise
            if noise_distribution == 'uniform':
                noise = np.random.uniform(-noise_level/2, noise_level/2, leng)
            elif noise_distribution == 'normal':
                noise = np.random.normal(0, noise_level/4, leng)
            elif noise_distribution == 'laplacian':
                noise = np.random.laplace(0, noise_level/6, leng)
            
            to_be_appended = basal_spike_dic[key][spike]
            if cnt_basal < basal_noise_num:
                to_be_appended = to_be_appended + noise
                cnt_basal += 1
            if key in noised_basal_spike_dic.keys():
                noised_basal_spike_dic[key].append(to_be_appended)
            else:
                noised_basal_spike_dic[key] = []
                noised_basal_spike_dic[key].append(to_be_appended)


    return noised_apical_spike_dic, noised_basal_spike_dic


# A function for calculating the robustness error
def robustness_error(path_base, path_noise, window_size, window_num):
    """
    path_base: the path of the base spike train
    path_noise: the path of the noise spike train
    """
    # load the spike train
    spike_train_base = np.load(path_base)
    spike_train_noise = np.load(path_noise)

    # calculate the robustness error
    mse = np.sum((spike_train_base - spike_train_noise)**2) / len(spike_train_base)

    # calculate the spiking rate error
        
    # Set a threshold for spike detection (e.g., -20 mV)
    threshold = -40 # You may need to adjust this based on your data

    # Detect spikes
    spikes_base = np.where(np.diff((spike_train_base > threshold).astype(int)) == 1)[0]

    spikes_noise = np.where(np.diff((spike_train_noise > threshold).astype(int)) == 1)[0]

    # print(f"base spikes: {len(spikes_base)}, noise spikes: {len(spikes_noise)}")


    # Calculate the spike rate error
    spike_rate_error = np.abs(len(spikes_base) - len(spikes_noise)) 


    # calculate the window error:
    window_error = []
    for i in range(window_num):
        # randomly select a window with size window_size
        start = np.random.randint(0, len(spike_train_base) - window_size)
        end = start + window_size
        # calculate the error
        spike_window_base = np.where(np.diff((spike_train_base[start:end] > threshold).astype(int)) == 1)[0]
        spike_window_noise = np.where(np.diff((spike_train_noise[start:end] > threshold).astype(int)) == 1)[0]

        # print(f"{i}: base spikes: {len(spike_window_base)}, noise spikes: {len(spike_window_noise)}")
        # print(f"{i}: base spikes: {spike_window_base}, noise spikes: {spike_window_noise}")
        if len(spike_window_base) == 0 and len(spike_window_noise) == 0:
            continue
        window_error.append(np.abs(len(spike_window_base) - len(spike_window_noise)))

    window_error = np.array(window_error)
    window_error = np.sum(window_error)

    return mse, spike_rate_error, window_error





if __name__ == '__main__':
    # mse, spike_rate_error, window_error = robustness_error('./v1.npy', './v2.npy', 200, 10000)
    # print(mse, spike_rate_error, window_error)
    spike_train_set = generaete_spike_train('normal', 100, 10, seed=100, start_time=0, end_time=500)

    # map the spike train to the cell
    apical_spike_dic, apical_spike_num, basal_spike_dic, basal_spike_num = map_method(101, 83, spike_train_set, 100)

    # add noise based on distribution
    apical_spike_dic11, basal_spike_dic11 = add_noise(apical_spike_dic, apical_spike_num, basal_spike_dic, basal_spike_num,\
                                                                    'normal', (1, 1), 0.1, 100)
    
    apical_spike_dic10, basal_spike_dic10 = add_noise(apical_spike_dic, apical_spike_num, basal_spike_dic, basal_spike_num,\
                                                                    'normal', (0, 0), 0.1, 100)
    
    for key1, key2 in zip(apical_spike_dic10, apical_spike_dic11):
        a = apical_spike_dic10[key1]
        b = apical_spike_dic11[key2]
        for a1, b1 in zip(a, b):
            print(a1==b1)
    
    print('-'*20)


    for key1, key2 in zip(basal_spike_dic10, basal_spike_dic11):
        a = basal_spike_dic10[key1]
        b = basal_spike_dic11[key2]

        for a1, b1 in zip(a, b):
            print(a1==b1)