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

path = './'



def initail_h():
    h.steps_per_ms = 10
    h.dt = 1.0 / h.steps_per_ms
    h.celsius = 37
    h.v_init = -86
    h.tstop = 300

class HPC:
    def __init__(self, filepath, morphologies=1):
        self.morphologies = morphologies
        self.HCell = self.setup_cell(filepath)
        self.E_PAS = -86
        self.CM = 0.44
        self.RM = 48300
        self.RA = 261.97
        self.rng = np.random.default_rng(100)
        self.synlist = []
        self.conlist = []
        self.back_synlist = []
        self.head_list_by_seg = []
        self.neck_list_by_seg = []
        #"../Morphs/2013_03_06_cell11_1125_H41_06.ASC"
   
    def setup_cell(self, filepath):
            if self.morphologies not in [1, 2, 3]:
                raise ValueError(f"Morphologies error: file cell{self.morphologies}.asc not found")
            
            h.load_file("import3d.hoc")
            morphology_file= filepath + f'/morphologies/cell{self.morphologies}.asc'
            biophys_file = filepath + '/models/L5PCbiophys1.hoc'
            hoc_file = filepath + '/models/L5PCtemplate.hoc'
            h.load_file("import3d.hoc")
            h.load_file(biophys_file)
            h.load_file(hoc_file)

            L5PC = h.L5PCtemplate(morphology_file)
            return L5PC

    def add_spike_train(self, position, stim, syn_weight):
        """
        Adds a spike train to a specific location on the cell.

        :param target_location: Tuple indicating the section and the relative position in that section (section_name, position)
        :param spike_interval: Interval between spikes in ms
        :param spike_number: Total number of spikes in the train
        :param syn_weight: Synaptic weight
        """

        # Create a synapse at the target location
        syn = h.ExpSyn(position)
        self.synlist.append(syn)

        # Connect the spike train to the synapse
        nc = h.NetCon(stim, syn)
        nc.weight[0] = syn_weight
        self.conlist.append(nc)

        return syn, stim, nc

    def get_spike_train(self, spike_interval, spike_number, start_time=200):
        stim = h.NetStim()
        stim.number = spike_number
        stim.interval = spike_interval
        stim.start = start_time
        return stim



def test_load_cell(morphologies=1):
    cell = HPC(path, morphologies=morphologies)
    stim = cell.get_spike_train(50, 1000)
    syn, stim, nc = cell.add_spike_train(cell.HCell.soma[0](0.5), stim, 0.1)
    # h.tstop = 200

    vec_t = h.Vector()
    vec_v = h.Vector()
    

    vec_t.record(h._ref_t)
    vec_v.record(cell.HCell.soma[0](0.5)._ref_v)

    h.run()
    print(syn.e)
    print(syn.tau)
    print(nc.weight[0])
    print(nc.delay)

    vec_v.to_python()
    vec_t.to_python()

    # save them to npy file
    np.save('./v1.npy', vec_v)
    np.save('./t1.npy', vec_t)
    plt.plot(vec_t, vec_v)

    plt.show()







if __name__=='__main__':
    initail_h()
    test_load_cell(morphologies=1)