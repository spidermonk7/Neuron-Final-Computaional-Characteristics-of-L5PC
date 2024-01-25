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



def draw_curve(curve_paths,save_path, noise_levels, save=False, window_size=100, calculate_all = False):
    apical_mses = []
    apical_rates = []
    apical_windows = []
    
    basal_mses = []
    basal_rates = []
    basal_windows = []

    all_mses = []
    all_rates = []
    all_windows = []

    for noise_level in noise_levels:
        # args.noise_level = noise_level
        # print(f"noise level is {noise_level}")
        # initail_h()
        # main(curve_path='./new_results/normal_spike_uniform_noise', args = args)
        path_base = curve_paths
        path_nonoise = path_base + f'/v(0, 0)_level{noise_level}.npy'
        path_apic_noise = path_base + f'/v(1, 0)_level{noise_level}.npy'
        path_basal_noise = path_base + f'/v(0, 1)_level{noise_level}.npy'
        path_all = path_base + f'/v(1, 1)_level{noise_level}.npy'
        mse, rate, window = robustness_error(path_nonoise, path_all, window_size, 5000)
        apical_mse, apical_rate, apical_window = robustness_error(path_nonoise, path_apic_noise, window_size, 5000)
        basal_mse, basal_rate, basal_window = robustness_error(path_nonoise, path_basal_noise, window_size, 5000)

        apical_mses.append(apical_mse)
        apical_rates.append(apical_rate)
        apical_windows.append(apical_window)

        basal_mses.append(basal_mse)
        basal_rates.append(basal_rate)
        basal_windows.append(basal_window)

        all_mses.append(mse)
        all_rates.append(rate)
        all_windows.append(window)

    apical_mses = np.array(apical_mses)
    apical_rates = np.array(apical_rates)
    apical_windows = np.array(apical_windows)

    basal_mses = np.array(basal_mses)
    basal_rates = np.array(basal_rates)
    basal_windows = np.array(basal_windows)

    all_mses = np.array(all_mses)
    all_rates = np.array(all_rates)
    all_windows = np.array(all_windows)


    if save:
        # plot those 3 curve pairs respectively in 3 subfig
        plt.figure()


        plt.subplot(3, 1, 1)
        plt.plot(apical_mses, label='apical')
        plt.plot(basal_mses, label='basal')

        plt.subplot(3, 1, 2)
        plt.plot(apical_rates, label='apical')
        plt.plot(basal_rates, label='basal')

        plt.subplot(3, 1, 3)
        plt.plot(apical_windows, label='apical')
        plt.plot(basal_windows, label='basal')

        title = curve_paths.split('/')[-1]
        plt.suptitle(title)


        plt.legend()

        plt.savefig(save_path + title + '.png')

        plt.close()

    if calculate_all:
        return all_mses, all_rates, all_windows

    # plt.show()
    return apical_mses, apical_rates, apical_windows, basal_mses, basal_rates, basal_windows

def draw_curves_mean(curve_paths,save_path, noise_levels, task='apical_basal', window_size=50):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if task == 'apical_basal':
        result_dic = {}
        for noise_distribution in ['uniform', 'normal', 'laplacian']:
            for spike_distribution in ['uniform', 'normal', 'laplacian']:
                apical_mses = []
                apical_windows = []

                basal_mses = []
                basal_windows = []
                for seed in tqdm([11, 22, 33, 44, 55]):
                    path = f'./new_results/seed{seed}_{spike_distribution}_{noise_distribution}'
                    apical_mse, apical_rate, apical_window, basal_mse, basal_rate, basal_window = draw_curve(path, save_path, noise_levels, window_size=window_size)
                    apical_mses.append(apical_mse)
                    apical_windows.append(apical_window)
                    apical_windows.append(apical_window)
                    

                    basal_mses.append(basal_mse)
                    basal_windows.append(basal_window)


                apical_mses = np.array(apical_mses).mean(axis=0)
                apical_windows = np.array(apical_windows).mean(axis=0)

                basal_mses = np.array(basal_mses).mean(axis=0)
                basal_windows = np.array(basal_windows).mean(axis=0)

                result_dic[f'{spike_distribution}_{noise_distribution}'] = {
                    'mean_mse': np.mean(basal_mse - apical_mses),
                    'max_mse': np.max(basal_mses - apical_mses),
                    'mean_window': np.mean(basal_windows - apical_windows),
                    'max_window': np.max(basal_windows - apical_windows)
                }
                # plot them: only mse and window
                plt.figure()
                plt.subplot(2, 1, 1)
                plt.plot(apical_mses, label='apical')
                plt.plot(basal_mses, label='basal')
                
                plt.title('mse')

                plt.subplot(2, 1, 2)
                plt.plot(apical_windows, label='apical')
                plt.plot(basal_windows, label='basal')

                title = f'{spike_distribution}_{noise_distribution}'
                plt.suptitle(title)

                plt.legend()
                plt.savefig(save_path + title + f'{window_size}.png')
                plt.close()


    elif task == 'distribution':
        for noise_distribution in ['uniform', 'normal', 'laplacian']:
            dic_rec = {}
            result_dic = {}
            for spike_distribution in ['uniform', 'normal', 'laplacian']:
                apical_mses = []
                apical_windows = []

                basal_mses = []
                basal_windows = []

                all_mses = []
                all_windows = []
                for seed in tqdm([11, 22, 33, 44, 55]):
                    path = f'./new_results/seed{seed}_{spike_distribution}_{noise_distribution}'
                    all_mse, all_rate, all_window = draw_curve(path, save_path, noise_levels, calculate_all=True)
                    apical_mses.append(apical_mse)
                    apical_windows.append(apical_window)

                    basal_mses.append(basal_mse)
                    basal_windows.append(basal_window)

                    all_mses.append(all_mse)
                    all_windows.append(all_window)


                apical_mses = np.array(apical_mses).mean(axis=0)
                apical_windows = np.array(apical_windows).mean(axis=0)

                basal_mses = np.array(basal_mses).mean(axis=0)
                basal_windows = np.array(basal_windows).mean(axis=0)

                all_mses = np.array(all_mses).mean(axis=0)
                all_windows = np.array(all_windows).mean(axis=0)

                dic_rec[f'{spike_distribution}_{noise_distribution}'] = [apical_mses, apical_windows, basal_mses, basal_windows, all_mses, all_windows]
                result_dic[f'{spike_distribution}_{noise_distribution}'] = {
                    'apical_mse': np.mean(apical_mses),
                    'apical_window': np.mean(apical_windows),
                    'basal_mse': np.mean(basal_mses),
                    'basal_window': np.mean(basal_windows)
                }

            # plot them
            plt.figure()
            plt.subplot(2, 1, 1)
            for key in dic_rec:
                plt.plot(dic_rec[key][0], label=key)
            plt.title('apical mse')

            plt.subplot(2, 1, 2)
            for key in dic_rec:
                plt.plot(dic_rec[key][1], label=key)
            plt.title('apical window')
            plt.legend()

            plt.savefig(save_path + f'noise{noise_distribution}_apical.png')
            plt.clf()

            plt.figure()
            plt.subplot(2, 1, 1)
            for key in dic_rec:
                plt.plot(dic_rec[key][2], label=key)
            plt.title('basal mse')

            plt.subplot(2, 1, 2)
            for key in dic_rec:
                plt.plot(dic_rec[key][3], label=key)
            plt.title('basal window')
            plt.legend()

            plt.savefig(save_path + f'noise{noise_distribution}_basal.png')
            plt.clf()

            plt.close()
            # plt.figure()
            # plt.subplot(2, 1, 1)
            # for key in dic_rec:
            #     plt.plot(dic_rec[key][5], label=key)
            # plt.title('all mse')

            # plt.subplot(2, 1, 2)
            # for key in dic_rec:
            #     plt.plot(dic_rec[key][6], label=key)
            # plt.title('all window')

    elif task == 'laplacian':
        # plot the mse and window curve for each scale
        scales = [5, 10, 15, 20, 25, 50, 75]
        spike_distribution = 'laplacian'
        noise_distributions = ['laplacian', 'normal', 'uniform']
        for noise_distribution in noise_distributions:
            plt.figure()
            dic_rec = {}
            result_dic = {}
            print(noise_distribution)
            for scale in scales:
                # print(f"handleing scale {scale}")
                all_mses = []
                all_windows = []
                for seed in [11, 22, 33, 44, 55]:
                    path = f'./new_results/seed{seed}_{spike_distribution}_{noise_distribution}_scale{scale}'
                    all_mse, all_rate, all_window = draw_curve(path, save_path, noise_levels, calculate_all=True)
    
                    all_mses.append(all_mse)
                    all_windows.append(all_window)

                all_mses = np.array(all_mses).mean(axis=0)
                all_windows = np.array(all_windows).mean(axis=0)

                dic_rec[f'{spike_distribution}_{noise_distribution}'] = [all_mses, all_windows]
                result_dic[f'{spike_distribution}_{noise_distribution}'] = {
                    'all_mse': np.mean(all_mses),
                    'all_window': np.mean(all_windows),
                }
                print(f"MSE: {np.mean(all_mses)}, AWE: {np.mean(all_windows)}")
                # plot them

                plt.subplot(2, 1, 1)
                plt.plot(all_mses, label=f'scale: {scale}')
                plt.title('MSE')

                plt.subplot(2, 1, 2)
                plt.plot(all_windows, label=f'scale: {scale}')
                plt.title('AWE')
                plt.legend()

            plt.savefig(save_path + f'spike{spike_distribution}_noise{noise_distribution}_multi_scale01.png')
            plt.clf()



    elif task == 'weight':
        result_dic = {}
        apical_weight_list = [0.01, 0.05, 0.1, 0.5, 1]
        basal_weight_list = [0.1]
        # plot the mse and window curve for each weight
        noise_distribution = 'normal'
        spike_distribution = 'normal'
        plt.figure()
        for apical_weight in apical_weight_list:
            apical_mses = []
            apical_windows = []

            basal_mses = []
            basal_windows = []
            for basal_weight in basal_weight_list:
               
                for seed in [11, 22, 33, 44, 55]:
                    path = f'./new_results/seed{seed}_{spike_distribution}_{noise_distribution}_{apical_weight}_{basal_weight}'
                    apical_mse, apical_rate, apical_window, basal_mse, basal_rate, basal_window = draw_curve(path, save_path, noise_levels)
                    apical_mses.append(apical_mse)
                    apical_windows.append(apical_window)

                    basal_mses.append(basal_mse)
                    basal_windows.append(basal_window)

                apical_mses = np.array(apical_mses).mean(axis=0)
                apical_windows = np.array(apical_windows).mean(axis=0)

                basal_mses = np.array(basal_mses).mean(axis=0)
                basal_windows = np.array(basal_windows).mean(axis=0)

 
            plt.subplot(2, 1, 1)
            plt.plot(apical_windows, label=f'({apical_weight}, 0.1)')
            plt.ylabel('AWE')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(basal_windows)
            plt.ylabel('AWE')
            print(f"apical weight: {apical_weight}, apical AWE: {np.mean(apical_windows)}, basal AWE: {np.mean(basal_windows)}")



        title = f'weight_{apical_weight}_0.1'
        # plt.suptitle(title)


        plt.savefig(save_path + title + 'pure_all.png')
    
    
    else:
        raise NotImplementedError
    
    
    return result_dic




def draw_curve_spike_distribution(noise_levels):
    noise_distribution = 'uniform'
    spike_distributions = ['normal', 'laplacian', 'uniform']

    # plot the curve for each spike distribution, 4 distribution of mse on the same figure
    plt.figure()
    apical_mse_normal = []
    apical_mse_laplacian = []
    apical_mse_uniform = []

    basal_mse_normal = []
    basal_mse_laplacian = []
    basal_mse_uniform = []

    for spike_distribution in spike_distributions:
        for seed in [1, 22]:
            path = f'./new_results/seed{seed}_{spike_distribution}_spike_{noise_distribution}_noise'
            apical_mse, apical_rates, apical_windows, basal_mse, basal_rates, basal_windows = draw_curve(path, noise_levels)
            if spike_distribution == 'normal':
                apical_mse_normal.append(apical_mse)
                basal_mse_normal.append(basal_mse)
            elif spike_distribution == 'laplacian':
                apical_mse_laplacian.append(apical_mse)
                basal_mse_laplacian.append(basal_mse)
            elif spike_distribution == 'uniform':
                apical_mse_uniform.append(apical_mse)
                basal_mse_uniform.append(basal_mse)
        
    apical_mse_normal = np.array(apical_mse_normal)
    apical_mse_laplacian = np.array(apical_mse_laplacian)
    apical_mse_uniform = np.array(apical_mse_uniform)

    basal_mse_normal = np.array(basal_mse_normal)
    basal_mse_laplacian = np.array(basal_mse_laplacian)
    basal_mse_uniform = np.array(basal_mse_uniform)

    # calculate the average of 3 seed
    apical_mse_normal = np.mean(apical_mse_normal, axis=0)
    apical_mse_laplacian = np.mean(apical_mse_laplacian, axis=0)
    apical_mse_uniform = np.mean(apical_mse_uniform, axis=0)

    basal_mse_normal = np.mean(basal_mse_normal, axis=0)
    basal_mse_laplacian = np.mean(basal_mse_laplacian, axis=0)
    basal_mse_uniform = np.mean(basal_mse_uniform, axis=0)

    # plot the those curves: mse, rate, window
    plt.subplot(3, 1, 1)
    plt.plot(apical_mse_normal, label='normal')
    plt.plot(apical_mse_laplacian, label='laplacian')
    plt.plot(apical_mse_uniform, label='uniform')
    plt.title('apical mse')

    plt.subplot(3, 1, 2)
    plt.plot(basal_mse_normal, label='normal')
    plt.plot(basal_mse_laplacian, label='laplacian')
    plt.plot(basal_mse_uniform, label='uniform')
    plt.title('basal mse')

    plt.legend()
    plt.show()


def draw_laplacian_curve(noise_levels):
    for seed in [11, 22, 33]:
        for scale in [2, 3, 4, 5]:
            
            path = f'./new_results/seed{seed}_laplacian_{scale}'
            apicalmse, apicalrate, apicalwindow, basalmse, basalrate, basalwindow = draw_curve(path, noise_levels)

           
            plt.subplot(3, 1, 1)
            plt.plot(apicalmse, label=f'scale: {scale}')
            plt.title('mse')

            plt.subplot(3, 1, 2)
            plt.plot(apicalrate, label=f'scale: {scale}')
            plt.title('rate')

            plt.subplot(3, 1, 3)
            plt.plot(apicalwindow, label=f'scale: {scale}')
            plt.title('window')

            # plt.subplot(3, 1, 1)
            # plt.plot(basalmse, label=f'scale: {scale}')
            # plt.title('mse')

            # plt.subplot(3, 1, 2)
            # plt.plot(basalrate, label=f'scale: {scale}')
            # plt.title('rate')

            # plt.subplot(3, 1, 3)
            # plt.plot(basalwindow, label=f'scale: {scale}')
            # plt.title('window')

        uniformpath = f'./new_results/seed{seed}_uniform_0'
        uniapimse, uniapirate, uniapiwindow, unibasalmse, unibasalrate, unibasalwindow = draw_curve(uniformpath, noise_levels)

        plt.subplot(3, 1, 1)
        plt.plot(uniapimse, label=f'uniform')
        plt.title('mse')

        plt.subplot(3, 1, 2)
        plt.plot(uniapirate, label=f'uniform')
        plt.title('rate')

        plt.subplot(3, 1, 3)
        plt.plot(uniapiwindow, label=f'uniform')
        plt.title('window')


        plt.legend()
        plt.show()
        plt.close()



if __name__ == '__main__':
    save_path = './weights/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    t = 2.554867498
    print('%.2f'%t)

    task = 'laplacian'
    noise_levels = [1, 5, 10, 15, 20, 30, 40, 50, 65, 80, 95, 110, 130, 150, 170, 200]
    noise_levels = [1, 5, 10, 15, 20, 30, 40, 50, 65, 80, 95, 110]
    draw_curves_mean('./new_results', save_path, noise_levels, task=task)
    print(task)

    # # store result to a csv file
    # with open(save_path + 'result.csv', 'w') as f:
    #     f.write('spike distribution, noise distribution, mean mse, max mse, mean window, max window\n')
    #     for key in result_dic:
    #         f.write(f'{key}, {result_dic[key]["mean_mse"]}, {result_dic[key]["max_mse"]}, {result_dic[key]["mean_window"]}, {result_dic[key]["max_window"]}\n')
