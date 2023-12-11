#!/usr/bin/env python3

import os
import cairosvg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import stim
matplotlib.get_backend()


def createcircuit(code_size, rounds, clifford_noise=0.0, reset_flip_prob=0.0, measure_flip_prob=0.0, round_data_noise=0.0):
    circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", 
            rounds=rounds, distance=code_size, 
            after_clifford_depolarization=clifford_noise,
            after_reset_flip_probability=reset_flip_prob,
            before_measure_flip_probability=measure_flip_prob,
            before_round_data_depolarization=round_data_noise)
    return circuit


def customreshape(samples, code_size):
    M = code_size+1
    images = []
    for l in range(samples.shape[0]): 
        t = np.zeros((M,M), dtype=np.float32)
        t[0,1:-1:2] = samples[l,0:int((code_size-1)/2)]
        k = 1
        l = 1 
        st = int((code_size-1)/2)
        for i in range(code_size-1):
            t[l,k:k+code_size] = samples[l,st:st+code_size]
            st = st+code_size
            if l%2 == 1:
                k = 0
            else:
                k = 1
            l += 1
        t[M-1,2:-1:2] = samples[l,st:]
        images.append(t)
    return images

def sampledata(circuit, code_size, numsnaps, rounds): 
    converter = circuit.compile_m2d_converter()
    sampler = circuit.compile_sampler()
    snapshots = []
    labels = []
    logz = []
    samps = [] 
    num_x_stabilizers = (int(code_size/2)+1)*(code_size-1)
    num_z_stabilizers = num_x_stabilizers
    num_stabilizers = num_x_stabilizers + num_z_stabilizers
    start = num_z_stabilizers + (rounds-2)*num_stabilizers
    start1 = rounds*num_stabilizers
    res = numsnaps
    while len(snapshots) < numsnaps:
        samples = sampler.sample(shots=res)
        lsamples, obs = converter.convert(measurements=samples, separate_observables=True)
        #lsamples, obs = sampler.sample(shots=res, separate_observables=True)
        lsamples = lsamples[:, start:start+num_stabilizers]
        logsamples = samples[:, start1:start1+code_size]
        print(lsamples)
        non_empty_indices = (np.sum(lsamples, axis=1)!=0)
        #print(non_empty_indices)
        reshapedsnapshots = customreshape(lsamples[non_empty_indices, :], code_size)
        snapshots.extend(reshapedsnapshots)
        labels.extend(obs[non_empty_indices, :].astype(np.uint8))
        logz.extend(logsamples[non_empty_indices, :].astype(np.uint8))
        samps.extend(samples[non_empty_indices, :].astype(np.uint8))
        res = numsnaps - len(snapshots)
    return [np.stack(snapshots, axis=0), np.stack(labels, axis=0), np.stack(logz, axis=0), np.stack(samps, axis=0)]

def printcircuit(circuit, visual='timeline', directory='junk'):
    os.makedirs(directory, exist_ok=True)
    if visual == 'nothing':
        print(circuit)
        return
    if visual == 'simple':
        print(circuit.diagram())
        return
    if visual == 'timeline':
        ckttype = 'timeline-svg'
    if visual == 'timeslice':
        ckttype = 'timeslice-svg'
    if os.path.exists("{}/out.svg".format(directory)):
        os.remove("{}/out.svg".format(directory))
    with open("{}/out.svg".format(directory), 'w') as fout:
        for l in str(circuit.diagram(ckttype)).split('\n'):
            fout.write(l)
    cairosvg.svg2png(url="{}/out.svg".format(directory), write_to="{}/out.png".format(directory))
    imageobj = plt.imread("{}/out.png".format(directory))
    plt.imshow(imageobj)
    plt.show()
    return

#def customreshape(vlist, code_size):
#    M = code_size+1
#    t = np.zeros((M,M))
#    t[0,1:-1:2] = vlist[0:int((code_size-1)/2)]
#    k = 1
#    l = 1 
#    st = int((code_size-1)/2)
#    for i in range(code_size-1):
#        t[l,k:k+code_size] = vlist[st:st+code_size]
#        st = st+code_size
#        if l%2 == 1:
#            k = 0
#        else:
#            k = 1
#        l += 1
#    t[M-1,2:-1:2] = vlist[st:]
#    return t

if __name__ == '__main__':
    #ckt = createcircuit(5, 3, reset_flip_prob=0.1, round_data_noise=0.2)
    ckt = createcircuit(5, 3, round_data_noise=0.2)
    a, b, l, s = sampledata(ckt, 5, 20, 3)
    #printcircuit(ckt, visual='timeline')
