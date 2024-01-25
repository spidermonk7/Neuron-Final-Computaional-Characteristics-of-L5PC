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
from utils import *
from tqdm import tqdm


"""
This is a file for the main running function
"""
store_path = './'


# you should: build up a cell, add synapses, add spike train, run the simulation, get the voltage
# and then plot the voltage

def load_args():
    parser = argparse.ArgumentParser(description='Training a L5PC cell')
    parser.add_argument('--seed', type=int, default=200, help='random seed')
    parser.add_argument('--start_time', type=int, default=150, help='start time of the spike train')
    parser.add_argument('--end_time', type=int, default=600, help='end time of the spike train')
    parser.add_argument('--spike_train_distribution', type=str, default='normal', help='the distribution of the spike train')
    parser.add_argument('--morpho', type=int, default=1, help='the cell morpho to use')
    parser.add_argument('--train_numbers', type=int, default=50, help='the number of spike train')
    parser.add_argument('--spike_numbers', type=int, default=10, help='the number of spikes in each train')
    parser.add_argument('--noise_distribution', type=str, default='uniform', help='the distribution of the noise')
    parser.add_argument('--noise_level', type=float, default=1, help='the level of the noise')
    parser.add_argument('--apic_weight', type=float, default=0.1, help='the weight of the synapse')
    parser.add_argument('--basal_weight', type=float, default=0.1, help='the weight of the synapse')
    parser.add_argument('--scale', type=float, default=75, help='the scale of the spike train')

    return parser.parse_args()



def main(curve_path, args):
    if not os.path.exists(curve_path):
        os.makedirs(curve_path)

    # set seed first
    np.random.seed(args.seed)
    # build a spike train
    spike_train_set = generaete_spike_train(args.spike_train_distribution, args.train_numbers, args.spike_numbers, seed=args.seed,\
                                            start_time=args.start_time, end_time=args.end_time, use_scale=True, scale=args.scale)

    # add noise based on distribution
    for location_distribution in [(0, 0), (0, 1), (1, 0), (1, 1)]:
         # load the cell
        cell = HPC(store_path, morphologies=args.morpho)
        print(f"handleing {location_distribution}")
        # map the spike train to the cell
        apical_num = len(cell.HCell.apic)
        basal_num = len(cell.HCell.dend)
        apical_spike_dic, apical_spike_num, basal_spike_dic, basal_spike_num = map_method(apical_num, basal_num, spike_train_set, args.seed)
        noise_distribution = args.noise_distribution
        # print(noise_distribution)

        ori_apical = 0
        for a in apical_spike_dic.keys():
            spikes = apical_spike_dic[a]
            for spike in spikes:
                ori_apical += len(spike)

        print("origin", ori_apical)






        noised_apical_spike_dic, noised_dend_spike_dic = add_noise(apical_spike_dic, apical_spike_num, basal_spike_dic, basal_spike_num,\
                                                                     noise_distribution, location_distribution, args.noise_level, args.seed)
        # print(location_distribution, noised_apical_spike_dic)


        actua_apical_spike_dic = {}
        actua_basal_spike_dic = {}
        remained_apical = 0
        remained_basal = 0
        for key in noised_apical_spike_dic.keys():
            for spike_train in range(len(noised_apical_spike_dic[key])):
                spikes = []
                for spike_time in noised_apical_spike_dic[key][spike_train]:
                    if spike_time <= args.end_time and spike_time>=args.start_time:
                        remained_apical +=1
                        spikes.append(cell.get_spike_train(1, 1, spike_time))

                if key in actua_apical_spike_dic.keys():
                    actua_apical_spike_dic[key].append(spikes)
                else:
                    actua_apical_spike_dic[key] = [spikes]


        # print(f"location distroibution:{location_distribution}", actua_apical_spike_dic)
        for key in actua_apical_spike_dic.keys():
            position = cell.HCell.apic[key](0.5)
            for spike_trains in actua_apical_spike_dic[key]:
                for spike in spike_trains:
                    # print("spike train is", spike_train)
                    syn, stim, nc = cell.add_spike_train(position, spike, args.apic_weight)


        
        for key in noised_dend_spike_dic.keys():
            for spike_train in range(len(noised_dend_spike_dic[key])):
                spikes = []
                for spike_time in noised_dend_spike_dic[key][spike_train]:
                    if spike_time <= args.end_time and spike_time>=args.start_time:
                        remained_basal+=1
                        spikes.append(cell.get_spike_train(1, 1, spike_time))

                if key in actua_basal_spike_dic.keys():
                    actua_basal_spike_dic[key].append(spikes)
                else:
                    actua_basal_spike_dic[key] = [spikes]

        for key in actua_basal_spike_dic.keys():
            position = cell.HCell.dend[key](0.5)
            for spike_trains in actua_basal_spike_dic[key]:
                for spike in spike_trains:
                    # print("spike train is", spike_train)
                    syn, stim, nc = cell.add_spike_train(position, spike, args.basal_weight)

        print(apical_spike_num*args.spike_numbers, basal_spike_num*args.spike_numbers)
        print(remained_apical, remained_basal)


        # run the simulation
        vec_t = h.Vector()
        vec_v = h.Vector()
        vec_t.record(h._ref_t)
        vec_v.record(cell.HCell.soma[0](0.5)._ref_v)

        h.run()

        vec_v.to_python()
        vec_t.to_python()

        # save them to npy file
        np.save(curve_path + f'v{location_distribution}_level{args.noise_level}.npy', vec_v)
        np.save(curve_path + f't{location_distribution}_level{args.noise_level}.npy', vec_t)
        plt.plot(vec_t, vec_v, label=f'apical: {location_distribution[0]}, basal: {location_distribution[1]}')
    plt.legend()

    plt.savefig(curve_path + f'result{args.noise_level}.png')
    # plt.show()
    plt.close()


def calculate_robustloss(curve_paths, noise_level):
    """
    calculate the robust loss of the curve
    """
    path_base = curve_paths
    path_nonoise = path_base + 'v(0, 0).npy'
    path_apic_noise = path_base + 'v(1, 0).npy'
    path_basal_noise = path_base + 'v(0, 1).npy'
    path_all_noise = path_base + 'v(1, 1).npy'

    apic_mse, apc_rate, apic_window = robustness_error(path_nonoise, path_apic_noise, 100, 1000)
    basal_mse, basal_rate, basal_window = robustness_error(path_nonoise, path_basal_noise, 100, 1000)
    all_mse, all_rate, all_window = robustness_error(path_nonoise, path_all_noise, 100, 1000)
    


    print(f"apic mse: {apic_mse}, apic rate: {apc_rate}, apic window: {apic_window}")
    print(f"basal mse: {basal_mse}, basal rate: {basal_rate}, basal window: {basal_window}")
    print(f"all mse: {all_mse}, all rate: {all_rate}, all window: {all_window}")




        

# write a main_run function for each experiments
# let's start with the experiments of the distribution of noise among different compartments

def initail_h():
    h.steps_per_ms = 10
    h.dt = 1.0 / h.steps_per_ms
    h.celsius = 37
    h.v_init = -86
    h.tstop = 600


if __name__ == '__main__':
   
    store_path = './'
    args = load_args()

    initail_h()

    noise_levels = [1, 5, 10, 15, 20, 30, 40, 50, 65, 80, 95, 110, 130, 150, 170, 200]

    for noise_distribution in ['uniform', 'normal', 'laplacian']:
        args.noise_distribution = noise_distribution
        for seed in [11, 22, 33, 44, 55]:
            np.random.seed(seed)
            for spike_train_distribution in ['laplacian']:
                args.spike_train_distribution = spike_train_distribution
                for scale in [5, 10, 15, 20]:
                    args.scale = scale
                    for noise_level in tqdm(noise_levels):
                        args.noise_level = noise_level
                        print(f"handling: {noise_level}, {noise_distribution}, {spike_train_distribution}, {seed}")
                        main(curve_path=f'./new_results/seed{seed}_{spike_train_distribution}_{noise_distribution}_scale{scale}/', args = args)
    
                    