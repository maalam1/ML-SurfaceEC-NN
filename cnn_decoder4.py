#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torch
import stim
import os

class preparecktandsample(object):
    def __init__(self, code_size, rounds=2):
        self.rounds = rounds
        self.code_size = code_size 
        self.code_type = "surface_code:rotated_memory_z"
        self.num_x_stabilizers = (int(self.code_size/2)+1)*(self.code_size-1)
        self.num_z_stabilizers = self.num_x_stabilizers
        self.num_stabilizers = self.num_x_stabilizers + self.num_z_stabilizers

    def setnoisevectors(self, probvector, \
                        clifford_noise=False, \
                        reset_flip_prob=False, \
                        measure_flip_prob=False, \
                        round_data_noise=False):
        self.clifford_noise = []
        self.reset_flip_prob = []
        self.measure_flip_prob = []
        self.round_data_noise = []
        self.probvector = probvector
        for p in probvector:
            if not clifford_noise:
                self.clifford_noise.append(p)
            else:
                self.clifford_nose.append(0.0)
            if not reset_flip_prob:
                self.reset_flip_prob.append(p)
            else:
                self.reset_flip_prob.append(0.0)
            if not measure_flip_prob:
                self.measure_flip_prob.append(p)
            else:
                self.measure_flip_prob.append(0.0)
            if not round_data_noise:
                self.round_data_noise.append(p)
            else:
                self.round_data_noise.append(0.0)

    def createcircuits(self):
        self.circuits = []
        for i,v in enumerate(self.probvector):
            ckt = stim.Circuit.generated(
                    self.code_type, 
                    rounds=self.rounds, distance=self.code_size, 
                    after_clifford_depolarization=self.clifford_noise[i],
                    after_reset_flip_probability=self.reset_flip_prob[i],
                    before_measure_flip_probability=self.measure_flip_prob[i],
                    before_round_data_depolarization=self.round_data_noise[i])
            self.circuits.append(ckt)

    def customreshape(self, samples):
        M = self.code_size+1
        images = []
        for kk in range(samples.shape[0]):
            t = np.zeros((M,M), dtype=np.float32)
            t[0,1:-1:2] = samples[kk,0:int((self.code_size-1)/2)]
            k = 1
            l = 1 
            st = int((self.code_size-1)/2)
            for i in range(self.code_size-1):
                t[l,k:k+self.code_size] = samples[kk,st:st+self.code_size]
                st = st+self.code_size
                if l%2 == 1:
                    k = 0
                else:
                    k = 1
                l += 1
            t[M-1,2:-1:2] = samples[kk,st:]
            padn = int((24-(self.code_size+1)*2)/2)
            images.append(
                np.pad(
                    np.kron(t, np.ones((2,2))).astype(np.float32),
                    pad_width=[(padn,padn),(padn,padn)], 
                    mode='constant'
                    )
                )
        return images

    def sampledata(self, numdata):
        self.createcircuits()
        numsnaps = int(numdata/len(self.circuits))
        snapshots = []
        labels = []
        for circuit in self.circuits:
            sampler = circuit.compile_detector_sampler()
            start = self.num_z_stabilizers + (self.rounds-2)*self.num_stabilizers
            for i in range(2):
                snapshots_one_class = []
                labels_one_class = []
                while len(snapshots_one_class) < numsnaps/2:
                    res = int(numsnaps/2)
                    lsamples, obs = sampler.sample(shots = res, separate_observables = True)
                    lsamples = lsamples[np.where(obs==i)[0]]
                    obs = obs[np.where(obs==i)[0]]
                    if lsamples.shape[0] < 1: 
                        continue
                    lsamples = lsamples[:, start:start+self.num_stabilizers]
                    non_empty_indices = (np.sum(lsamples, axis=1)!=0)
                    reshapedsnapshots = self.customreshape(lsamples[non_empty_indices, :])
                    snapshots_one_class.extend(reshapedsnapshots)
                    labels_one_class.extend(obs[non_empty_indices, :].astype(np.uint8))
                    res = numsnaps - len(snapshots)
                snapshots.extend(snapshots_one_class)
                labels.extend(labels_one_class)
        X = np.stack(snapshots, axis=0)
        y =  np.stack(labels, axis=0)
        perm = np.random.permutation(X.shape[0])
        return [X[perm], y[perm]]

class CustomTensorDataset(Dataset):
    def __init__(self, dataset, transform_list=None):
        [data_X, data_y] = dataset
        X_tensor, y_tensor = torch.tensor(data_X), torch.tensor(data_y)
        tensors = (X_tensor, y_tensor)
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors 
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transforms:
            x = self.transforms(x)
        y = self.tensors[1][index]
        return torch.unsqueeze(x, 0), y.float()

    def __len__(self):
        return self.tensors[0].size(0)

class Netconv5(nn.Module):
    def __init__(self): 
        super(Netconv5, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 3)
        self.conv2 = nn.Conv2d(24, 24, 3)
        self.conv3 = nn.Conv2d(24, 16, 2)
        self.mpool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(16*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.mpool(x)
        x = x.view(-1, 16*6*6)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

class Netconv(nn.Module):
    def __init__(self, code_distance, num_outchannels, filter_size): 
        super(Netconv, self).__init__()
        self.code_distance = code_distance
        self.num_outchannels = num_outchannels
        self.filter_size = filter_size
        oc = self.num_outchannels
        fz = self.filter_size
        self.conv1 = nn.Conv2d(1, oc, fz)
        self.conv2 = nn.Conv2d(oc, oc*2, fz)
        self.conv3 = nn.Conv2d(oc*2, oc*4, fz)
        if fz == 3:
            self.mpool = nn.MaxPool2d(6, 6)
            fv = oc*4*3*3
        elif fz == 5:
            self.mpool = nn.MaxPool2d(4, 4)
            fv = oc*4*3*3
        elif fz == 7:
            self.mpool = nn.MaxPool2d(2, 2)
            fv = oc*4*3*3
        self.fv = fv
        if self.code_distance == 5:
            self.fc1 = nn.Linear(fv, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 1)
        elif self.code_distance == 7:
            self.fc1 = nn.Linear(fv, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 128)
            self.fc4 = nn.Linear(128, 1)
        elif self.code_distance == 9:
            self.fc1 = nn.Linear(fv, 2048)
            self.fc2 = nn.Linear(2048, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc4 = nn.Linear(512, 128)
            self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        print(x.shape)
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print(x.shape)
        x = torch.relu(self.conv3(x))
        print(x.shape)
        x = self.mpool(x)
        x = x.view(-1, self.fv)
        if self.code_distance == 5:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.code_distance == 7:
            print(x.shape)
            x = torch.relu(self.fc1(x))
            print(x.shape)
            x = torch.relu(self.fc2(x))
            print(x.shape)
            x = torch.relu(self.fc3(x))
            print(x.shape)
            x = self.fc4(x)
        elif self.code_distance == 9:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
        print(x.shape)
        x = torch.sigmoid(x)
        return x

class Netlin1(nn.Module):
    def __init__(self, sizefirst): 
        super(Netlin1, self).__init__()
        self.linlayers = nn.ModuleList()
        self.linlayers.append(nn.Linear(1*24*24,sizefirst))
        p1 = sizefirst
        p2 = int(p1/2)
        while p2 >= 128:
            self.linlayers.append(nn.Linear(p1,p2))
            p1 = p2
            p2 = int(p2/2)
        self.linlayers.append(nn.Linear(128,1))

    def forward(self, x):
        x = x.view(-1, 24*24)
        for i, l in enumerate(self.linlayers):
            if i == len(self.linlayers)-1:
                break
            x = torch.relu(self.linlayers[i](x))
            x = torch.relu(x)
        x = self.linlayers[-1](x)
        x = torch.sigmoid(x)
        return x

class Netlin2(nn.Module):
    def __init__(self, sizefirst): 
        super(Netlin2, self).__init__()
        self.linlayers = nn.ModuleList()
        self.linlayers.append(nn.Linear(1*24*24,sizefirst))
        p1 = sizefirst
        p2 = int(p1/2)
        self.linlayers.append(nn.Linear(p1,128))
        self.linlayers.append(nn.Linear(128,1))

    def forward(self, x):
        x = x.view(-1, 24*24)
        for i, l in enumerate(self.linlayers):
            if i == len(self.linlayers)-1:
                break
            x = torch.relu(self.linlayers[i](x))
            x = torch.relu(x)
        x = self.linlayers[-1](x)
        x = torch.sigmoid(x)
        return x

def trainmodel(model, code_distance, probs, training_instances, testloader, optimizer, criterion, numepochs, device, batchsize=32, valinterval=1):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    num = 0
    num_images = 0
    epochelapsed = 0
    samplingobj = preparecktandsample(code_distance)
    samplingobj.setnoisevectors(probs, round_data_noise=True)

    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0
    epochcounter = 0
    for epoch in range(numepochs):
        trainset = CustomTensorDataset(dataset = samplingobj.sampledata(training_instances))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += ((outputs>0.5) == labels).sum().item()
            num_images += labels.size(0)
            print("\r images processed: {} and epoch {}".format(num_images, epoch+1), end='')

        epochcounter += 1
        if epochcounter == valinterval:
            train_loss /= (len(trainloader)*epochcounter)
            train_loss_history.append(train_loss)
            train_acc /= (len(trainloader.dataset)*epochcounter)
            train_acc_history.append(train_acc)
        
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += ((outputs > 0.5) == labels).sum().item()
            val_loss /= len(testloader)
            val_loss_history.append(val_loss)
            val_acc /= len(testloader.dataset)
            val_acc_history.append(val_acc)

            epochcounter = 0
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

    print("")
    return [train_loss_history, train_acc_history, val_loss_history, val_acc_history]

def sweeplayercount(code_distance, ntype="lin1", directory="denselayer" ):
    os.makedirs(directory, exist_ok=True)
    fname1 = os.path.join(directory, "loss_accuracy_{}_{}.csv".format(code_distance, ntype))
    fname2 = os.path.join(directory, "loss_accuracy_{}_{}.png".format(code_distance, ntype))
    probs = [0.001, 0.005, 0.01, 0.1]
    training_data_size = len(probs)*10000
    test_data_size = len(probs)*1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 30
    samplingobjtest = preparecktandsample(code_distance)
    samplingobjtest.setnoisevectors(probs, round_data_noise=True)
    testset = CustomTensorDataset(dataset = samplingobjtest.sampledata(test_data_size)) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    criterion = nn.BCELoss()
    sizes = [256, 512, 1024, 2048, 2048*2, 2048*2*2]
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    fout1 = open(fname1, 'w')
    fout1.write("SIZE,TRAIN-LOSS,VAL-LOSS,TRAIN-ACCURACY,VAL-ACCURACY\n")
    for size in sizes:
        net = Netlin1(size)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        [y1, y2, y3, y4] = trainmodel(net, code_distance, probs, \
                   training_data_size, testloader, \
                   optimizer, criterion, epochs, device)
        g1.append(y1[-1])
        g2.append(y2[-1])
        g3.append(y3[-1])
        g4.append(y4[-1])
        fout1.write("{},{},{},{},{}\n".format(size, y1[-1], y3[-1], y2[-1], y4[-1]))
    fout1.close()
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(sizes, g1, label='train loss')
    axes[0].plot(sizes, g3, label='val loss')
    axes[0].set_xlabel("Size")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(sizes, g2, label='train Accuracy')
    axes[1].plot(sizes, g4, label='val Accuracy')
    axes[1].set_xlabel("Size")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.savefig(fname2)
    plt.show()


def sweepconvchannel(code_distance, directory="convelayer" ):
    os.makedirs(directory, exist_ok=True)
    fname1 = os.path.join(directory, "loss_accuracy_vs_outchannel_{}.csv".format(code_distance))
    fname2 = os.path.join(directory, "loss_accuracy_vs_outchannel_{}.png".format(code_distance))
    probs = [0.001, 0.005, 0.01, 0.1]
    training_data_size = len(probs)*10000
    test_data_size = len(probs)*1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 30
    samplingobjtest = preparecktandsample(code_distance)
    samplingobjtest.setnoisevectors(probs, round_data_noise=True)
    testset = CustomTensorDataset(dataset = samplingobjtest.sampledata(test_data_size)) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    criterion = nn.BCELoss()
    num_channels = [12, 24, 48]
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    fout1 = open(fname1, 'w')
    fout1.write("NUM_CHANNELS,TRAIN-LOSS,VAL-LOSS,TRAIN-ACCURACY,VAL-ACCURACY\n")
    for nc in num_channels:
        net = Netconv(code_distance, nc, 5)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        [y1, y2, y3, y4] = trainmodel(net, code_distance, probs, \
                   training_data_size, testloader, \
                   optimizer, criterion, epochs, device)
        g1.append(y1[-1])
        g2.append(y2[-1])
        g3.append(y3[-1])
        g4.append(y4[-1])
        fout1.write("{},{},{},{},{}\n".format(nc, y1[-1], y3[-1], y2[-1], y4[-1]))
    fout1.close()
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(num_channels, g1, label='train loss')
    axes[0].plot(num_channels, g3, label='val loss')
    axes[0].set_xlabel("Num Channels")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(num_channels, g2, label='train Accuracy')
    axes[1].plot(num_channels, g4, label='val Accuracy')
    axes[1].set_xlabel("Num Channels")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.savefig(fname2)
    plt.show()

def sweepconvfiltersize(code_distance, directory="convelayer" ):
    os.makedirs(directory, exist_ok=True)
    fname1 = os.path.join(directory, "loss_accuracy_vs_filtersizes_{}.csv".format(code_distance))
    fname2 = os.path.join(directory, "loss_accuracy_vs_filtersizes_{}.png".format(code_distance))
    probs = [0.001, 0.005, 0.01, 0.1]
    training_data_size = len(probs)*10000
    test_data_size = len(probs)*1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 20
    samplingobjtest = preparecktandsample(code_distance)
    samplingobjtest.setnoisevectors(probs, round_data_noise=True)
    testset = CustomTensorDataset(dataset = samplingobjtest.sampledata(test_data_size)) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    criterion = nn.BCELoss()
    filter_sizes = [3, 5, 7]
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    fout1 = open(fname1, 'w')
    fout1.write("NUM_CHANNELS,TRAIN-LOSS,VAL-LOSS,TRAIN-ACCURACY,VAL-ACCURACY\n")
    for fz in filter_sizes:
        net = Netconv(code_distance, 24, fz)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        [y1, y2, y3, y4] = trainmodel(net, code_distance, probs, \
                   training_data_size, testloader, \
                   optimizer, criterion, epochs, device)
        g1.append(y1[-1])
        g2.append(y2[-1])
        g3.append(y3[-1])
        g4.append(y4[-1])
        fout1.write("{},{},{},{},{}\n".format(fz, y1[-1], y3[-1], y2[-1], y4[-1]))
    fout1.close()
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(filter_sizes, g1, label='train loss')
    axes[0].plot(filter_sizes, g3, label='val loss')
    axes[0].set_xlabel("Filter sizes")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(filter_sizes, g2, label='train Accuracy')
    axes[1].plot(filter_sizes, g4, label='val Accuracy')
    axes[1].set_xlabel("Filter sizes")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.savefig(fname2)
    plt.show()

def sweepbatch(code_distance, directory="batchsize" ):
    os.makedirs(directory, exist_ok=True)
    fname1 = os.path.join(directory, "loss_accuracy_vs_batchsizes_{}.csv".format(code_distance))
    fname2 = os.path.join(directory, "loss_accuracy_vs_batchrsizes_{}.png".format(code_distance))
    probs = [0.001, 0.005, 0.01, 0.1]
    training_data_size = len(probs)*10000
    test_data_size = len(probs)*1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 20
    samplingobjtest = preparecktandsample(code_distance)
    samplingobjtest.setnoisevectors(probs, round_data_noise=True)
    testset = CustomTensorDataset(dataset = samplingobjtest.sampledata(test_data_size)) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    criterion = nn.BCELoss()
    batch_sizes = [16, 24, 32, 48]
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    fout1 = open(fname1, 'w')
    fout1.write("BATCH_SIZES,TRAIN-LOSS,VAL-LOSS,TRAIN-ACCURACY,VAL-ACCURACY\n")
    for bs in batch_sizes:
        net = Netconv(code_distance, 24, 7)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        [y1, y2, y3, y4] = trainmodel(net, code_distance, probs, \
                   training_data_size, testloader, \
                   optimizer, criterion, epochs, device, batchsize=bs)
        g1.append(y1[-1])
        g2.append(y2[-1])
        g3.append(y3[-1])
        g4.append(y4[-1])
        fout1.write("{},{},{},{},{}\n".format(bs, y1[-1], y3[-1], y2[-1], y4[-1]))
    fout1.close()
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(batch_sizes, g1, label='train loss')
    axes[0].plot(batch_sizes, g3, label='val loss')
    axes[0].set_xlabel("Batch sizes")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(batch_sizes, g2, label='train Accuracy')
    axes[1].plot(batch_sizes, g4, label='val Accuracy')
    axes[1].set_xlabel("Batch sizes")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.savefig(fname2)
    plt.show()

def sweeplr(code_distance, directory="Learnrate" ):
    os.makedirs(directory, exist_ok=True)
    fname1 = os.path.join(directory, "loss_accuracy_vs_lr_{}.csv".format(code_distance))
    fname2 = os.path.join(directory, "loss_accuracy_vs_lr_{}.png".format(code_distance))
    probs = [0.001, 0.005, 0.01, 0.1]
    training_data_size = len(probs)*10000
    test_data_size = len(probs)*1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 20
    samplingobjtest = preparecktandsample(code_distance)
    samplingobjtest.setnoisevectors(probs, round_data_noise=True)
    testset = CustomTensorDataset(dataset = samplingobjtest.sampledata(test_data_size)) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    criterion = nn.BCELoss()
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    fout1 = open(fname1, 'w')
    fout1.write("LR,TRAIN-LOSS,VAL-LOSS,TRAIN-ACCURACY,VAL-ACCURACY\n")
    for lrv in learning_rates:
        net = Netconv(code_distance, 24, 7)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=lrv)
        [y1, y2, y3, y4] = trainmodel(net, code_distance, probs, \
                   training_data_size, testloader, \
                   optimizer, criterion, epochs, device, batchsize=48)
        g1.append(y1[-1])
        g2.append(y2[-1])
        g3.append(y3[-1])
        g4.append(y4[-1])
        fout1.write("{},{},{},{},{}\n".format(lrv, y1[-1], y3[-1], y2[-1], y4[-1]))
    fout1.close()
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(learning_rates, g1, label='train loss')
    axes[0].plot(learning_rates, g3, label='val loss')
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(learning_rates, g2, label='train Accuracy')
    axes[1].plot(learning_rates, g4, label='val Accuracy')
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.savefig(fname2)
    plt.show()

if __name__ == '__main__':
    sweepconvchannel(7)
    #sweeplayercount(7)
