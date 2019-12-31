from statsmodels.robust import mad
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pywt
import json
import keras.backend as K
import tensorflow as tf
import math as mt
import time
import keras  
from numba import jit
import BaselineWanderRemoval as bwr
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, BatchNormalization,Flatten, Dropout, Lambda, Add, add, AveragePooling1D, Reshape
from keras.models import Model
import pickle

def Openf(link):
    # function for opening some json files
    with open (link,'r') as f:
        data = json.load(f)
        return data



def healthy(diagnos):
    is_heathy =True
    axis_ok = False
    rythm_ok = False
    for key in diagnos.keys():
        if key == 'electric_axis_normal':
            if diagnos[key] == True:
                axis_ok = True
                continue
        if key == 'regular_normosystole':
            if diagnos[key] == True:
                rythm_ok = True
                continue
        if diagnos[key] == True:
            is_heathy = False
            break
    return axis_ok and rythm_ok and is_heathy


@jit
def fix_line(entry):
    FREQUENCY_OF_DATASET = 500
    return bwr.fix_baseline_wander(entry, FREQUENCY_OF_DATASET)


def wavelet_smooth(X, wavelet="db4", level=1, title=None):
    coeff = pywt.wavedec(X, wavelet, mode="per")
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(X)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    y = pywt.waverec(coeff, wavelet, mode="per")
    return y

@jit
def find_peaks_div(x, scope_max=10, scope_null=50):
    lenght = x.shape[0]
    y0 = np.zeros(lenght)
    y1 = np.zeros(lenght)
    y2 = np.zeros(lenght)
    y3 = np.zeros(lenght)

    for i in range(lenght-2):
        y0[i+2] = mt.fabs(x[i+2]-x[i])
    for i in range(lenght-4):
        y1[i+4] = mt.fabs(x[i+4]-2*x[i+2]+ x[i])
    for i in range(lenght-4):
        y2[i+4] = 1.3*y0[i+4]+1.1*y1[i+4]
    for i in range(lenght-4-7):
        for k in range(7):
            y3[i] += y2[i+4-k]
        y3[i] /= 8
    max_idx = []
    curr_max = max(y3)
    curr_argmax = np.argmax(y3)
    true_argmax = np.argmax(x[max(0,curr_argmax-scope_max):min(curr_argmax+scope_max,lenght)])

    max_idx.append(max(0, curr_argmax-scope_max) + true_argmax)
    y3[max(0,curr_argmax-scope_null):min(curr_argmax+scope_null,lenght)] *= 0

    prev_max = curr_max
    curr_max = max(y3)

    while (prev_max - curr_max) < (prev_max / 4.0):
        curr_argmax = np.argmax(y3)
        true_argmax = np.argmax(x[max(0,curr_argmax-scope_max):min(curr_argmax+scope_max,lenght)])
        max_idx.append(max(0, curr_argmax-scope_max) + true_argmax)
        y3[max(0,curr_argmax-scope_null):min(curr_argmax+scope_null,lenght)] *= 0
        prev_max = curr_max
        curr_max = max(y3)
    return max_idx


def generator_QRS_complexes(data, batch_size, cycle_lenght = 250, seed = 10):
    otvedenie = 'i'
    rd.seed(seed) # set a seed
    while True:
        RES = []
        count = 0
        while(count <batch_size):
            case_id = str(rd.sample(data.keys(), 1)[0])
            diagnos = data[case_id]['StructuredDiagnosisDoc']
            leads = data[case_id]["Leads"] # take random a patient
            signal = fix_line(leads[otvedenie]["Signal"]) # take a signal and aply fix line
            # signal = leads[otvedenie]["Signal"]
            x_fltr = wavelet_smooth(signal, wavelet="db4", level=1, title=None)
            peaks_idx = find_peaks_div(x_fltr)
            peaks_idx.sort()
            if len(peaks_idx)>=3:
                peaks_idx = peaks_idx[1:-1] # remove edgest peaks
            else:
                continue # make one else iteration
            peak = rd.sample(peaks_idx, 1)[0] # take a random peak
            while (peak-cycle_lenght<0) or (peak+cycle_lenght>len(signal)): # resave the peak if index out of range
                peak = rd.sample(peaks_idx, 1)[0] # take a random peak
            res = signal[peak-cycle_lenght:peak+cycle_lenght]
            RES.append(res)
            count +=1
        RES = np.array(RES)
        RES = np.reshape(RES, (count, cycle_lenght*2, 1 ))
        yield (RES, RES)



def visualize_latent_space(start_true,end_true,start, end, decoder, count_of_step, size_of_data):
    from matplotlib.animation import FuncAnimation
    interpol_arr = []
    direction = (end-start)/count_of_step
    RES = []
    for i in range(count_of_step+1):
        tmp = start + i*direction
        RES.append(tmp)
    RES = np.array(RES) # this is our batch

    final_out = decoder.predict(RES)

    fig = plt.figure(3)
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(i):
        x = np.arange(0,size_of_data)
        y = final_out[i].reshape(size_of_data)
        ax1.clear()
        ax1.plot(x, start_true)
        ax1.plot(x, end_true)
        ax1.plot(x, y, color = 'k')
        ax1.legend(['ecg A', 'ecg B', 'latent point'], loc='upper left')
        interpol_arr.append(y)
        print('signal was added')
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title("iteration "+ str(i)+ "/"+ str(count_of_step+1))
    anim = FuncAnimation(fig, animate,frames=count_of_step+1, interval=30)
    anim.save('animation_1.gif', writer='imagemagick', fps=60)
    with open('interpol_array.pickle', 'wb') as q:
        pickle.dump(interpol_arr, q)
    plt.show()

def visualize_learning(history, graph1, graph2 ):
    plt.plot(history.history[graph1])
    plt.plot(history.history[graph2])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

class Reconstr_on_each_epoch(keras.callbacks.Callback):
    def __init__(self, signal_reconst):
        self.signal = signal_reconst # shape is (2,2000,1)
    def on_train_begin(self, logs={}):
        self.arr = []

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.signal)[0]
        self.arr.append(output)


def create_encoder(input_for_encoder, latent_dim):
    '''create an encoder'''
    x = Conv1D(30, 100, activation='relu', padding='same')(input_for_encoder)
    x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(20, 100, activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(20, 30, activation='relu', padding='same')(x)
    x =  MaxPooling1D(2, padding='same')(x)
    x = Conv1D(15, 20, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    shape = K.int_shape(x) # save this shape for purposes of reconstruction 
    x = Flatten()(x)
    encoded = Dense(latent_dim, )(x) # закодированный вектор
    return (Model(input_for_encoder, encoded), shape) # create a model 

def create_decoder(input_for_decoder, shape):
    '''create a decoder'''
    x = Dense(shape[1]*shape[2])(input_for_decoder)
    x = Reshape((shape[1] ,shape[2]))(x) 
    x = UpSampling1D(2)(x)
    x = Conv1D(15, 20, activation = 'relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(20, 30, activation = 'relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(20, 100, activation = 'relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 100, padding='same')(x)
    return Model(input_for_decoder, decoded) # create a model

