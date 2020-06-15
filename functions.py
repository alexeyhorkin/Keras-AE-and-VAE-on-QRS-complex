import json
import keras
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import random
import keras.backend as K
import tensorflow as tf
import math as mt
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, BatchNormalization,Flatten, Dropout, Lambda, Add, add, AveragePooling1D, Reshape
from keras.models import Model
import BaselineWanderRemoval as bwr
import pickle

dataset_path="data_2033\\data_2033.json" #файл с датасетом
def Openf(link):
    # function for opening some json files
    with open (link,'r') as f:
        data = json.load(f)
        return data


def dell_some_ecg(data, arr):
    for i in arr:
        del data[i]

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


def fix_line(entry):
    FREQUENCY_OF_DATASET = 500
    return bwr.fix_baseline_wander(entry, FREQUENCY_OF_DATASET)

def norm_ecg(signal, mean = 0, sigma = 1):
    signal = np.array(signal)
    return (signal - mean)/sigma    

def modify_dataset_on_healthy(data , mean =0 , sigma=1):
    lead_name = 'i'
    wrong_array = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        diag = data[case_id]['StructuredDiagnosisDoc']
        if healthy(diag):
            new_entry = leads[lead_name]['Signal']
            data[case_id]['Leads'][lead_name]['Signal'] = norm_ecg( fix_line(new_entry), mean, sigma)  # modify
        else:
            # если пациент не зворов, то удаляем его из датасета
            wrong_array.append(case_id)
    return wrong_array

def modify_dataset_on_sick(data, mean=0, sigma=1):
    lead_name = 'i'
    wrong_array = []
    for case_id in data.keys():
        diag = data[case_id]['StructuredDiagnosisDoc']
        if healthy(diag): # если здоров, то надо удалить
            wrong_array.append(case_id)
        else:
            # выровнить изолинию
            new_entry = data[case_id]['Leads'][lead_name]["Signal"]
            data[case_id]['Leads'][lead_name]['Signal'] = norm_ecg( fix_line(new_entry), mean, sigma ) # modify
    return wrong_array
    
def modify_dataset(data , mean =0, sigma=1):
    lead_name = 'i'
    wrong_array = []
    for case_id in data.keys():
        new_entry = data[case_id]['Leads'][lead_name]["Signal"]
        data[case_id]['Leads'][lead_name]['Signal'] = norm_ecg( fix_line(new_entry), mean, sigma ) # modify



def Create_data_generator(data,count_augmentation_data,count_of_diffrent_signals,size_of_data,flag,flag2):
    # generator for learning data with augmentation
    rd.seed(10)
    length_of_move = 10 # step of moving
    SIZE = 5000
    procent_train = 0.8

    count_of_train = int(len(data.keys())*procent_train)
    data_train = { k: data[k] for k in list(data.keys())[:count_of_train]} 
    data_test =  { k: data[k] for k in list(data.keys())[count_of_train:]}

    def modify(data_train, data_test, flag):
        if flag == "healthy":
            if count_of_train/procent_train > 200:
                mean, sigma = 6.66376, 117.68305393157674
            else:
                mean, sigma = 28.449627848101258, 107.07395896277849
            wrong_dataset1 = modify_dataset_on_healthy(data_train)
            wrong_dataset2 = modify_dataset_on_healthy(data_test)
            dell_some_ecg(data_train, wrong_dataset1)
            dell_some_ecg(data_test, wrong_dataset2)
        elif flag == "is_not_healthy":
            if count_of_train/procent_train > 200:
                mean, sigma = 23.730822824974414, 147.70848142124848
            else:
                mean, sigma = 25.09567675675674, 136.8313677231403
            wrong_dataset1 = modify_dataset_on_sick(data_train)
            wrong_dataset2 = modify_dataset_on_sick(data_test)
            dell_some_ecg(data_train, wrong_dataset1)
            dell_some_ecg(data_test, wrong_dataset2)
        elif flag == None:
            if count_of_train/procent_train > 200:
                mean, sigma = 23.914190063944915, 146.34316902498435
            else:
                mean, sigma = 25.963282999999986, 135.52340330312663
            
            modify_dataset(data_train)
            modify_dataset(data_test)

    modify(data_train, data_test , flag2) 

    if flag == "train":
        DATA = data_train
        print("size of dataset train is ", len(data_train))
    elif flag == "test":
        DATA = data_test
        print("size of dataset test is ", len(data_test))

    while True:
        RES = []
        count = 0
        for i in range(count_of_diffrent_signals):
            case_id = str(rd.sample(DATA.keys(), 1)[0])
            leads = DATA[case_id]["Leads"] # take random a patient
            diagnos = DATA[case_id]['StructuredDiagnosisDoc']
            # print("diag is - ", diagnos)
            # otvedenie = str(rd.sample(leads.keys(),1)[0]) # random
            otvedenie = 'i' # special
            # otvedenie = list(leads.keys())[0] # special v2
            signal = leads[otvedenie]["Signal"] # take a signal 
            start = rd.randint(0,SIZE - size_of_data-count_augmentation_data*length_of_move)
            for x in range(count_augmentation_data+1): #делаем срезы по каждому пациенту
                res = signal[start+x*length_of_move : start+x*length_of_move + size_of_data] # make a slice
                RES.append(res) # add resault in batch
                count +=1
        RES = np.array(RES)
        RES = np.reshape(RES, (count,size_of_data,1))
        # yield (RES,None)
        yield (RES,RES)

def Create_data_generator_for_model(data,size_of_data,flag):
    # generator for learning data with augmentation
    my_generator = Create_data_generator(data = data,
                                    count_augmentation_data =0,
                                    count_of_diffrent_signals=2,
                                    size_of_data= size_of_data)
    while(True):
        data1 = next(my_generator)
        labels = np.array([0,0])
        yield (data1[0],  [data1[1], labels] ) 

def Create_data_for_classificator(data_good_ekg, data_false_ekg,size_of_batch,size_of_data,flag):
    #generator for classificator bad and good ekg
    # 1 - bad ecg
    # 0 - good ecg
    rd.seed(6)
    length_of_move = 10 # step of moving
    procent_train = 0.8
    count_of_train_false = int(len(data_false_ekg.keys())*procent_train)
    count_of_train_true = int(len(data_good_ekg.keys())*procent_train)
    data_train_false_ekg = { k: data_false_ekg[k] for k in list(data_false_ekg.keys())[:count_of_train_false]} 
    data_test_false_ekg =  { k: data_false_ekg[k] for k in list(data_false_ekg.keys())[count_of_train_false:]}
    data_train_good_ekg = { k: data_good_ekg[k] for k in list(data_good_ekg.keys())[:count_of_train_true]} 
    data_test_good_ekg =  { k: data_good_ekg[k] for k in list(data_good_ekg.keys())[count_of_train_true:]}

    SIZE1 = len(data_train_false_ekg[0])
    SIZE2 = 5000 
    if flag == "train":
        DATA_bad = data_train_false_ekg
        DATA_good = data_train_good_ekg 
    elif flag == "test":
        DATA_bad = data_test_false_ekg
        DATA_good = data_test_good_ekg
    while True:
        RES = []
        labels = []
        count = 0
        for i in range(size_of_batch):
            start1, start2   =   ( rd.randint(0,SIZE1 - size_of_data) , rd.randint(0,SIZE2 - size_of_data) ) 
            index1 , index2  =   ( rd.sample(DATA_bad.keys(),1)[0] , rd.sample(DATA_good.keys(), 1)[0]  ) 
            otvedenie        =   rd.sample(    DATA_good[index2]["Leads"].keys() ,1)[0]
            signal1, signal2 =   ( DATA_bad[index1], DATA_good[index2]["Leads"][otvedenie]["Signal"] )
            res1 , res2      =   ( signal1[start1 : start1 + size_of_data] , signal2[start2 : start2 + size_of_data] ) 
            RES.append(res1) # add a label 1
            RES.append(res2) # add a label 0
            labels.append([1])
            labels.append([0])
            count+=2 
        RES = np.array(RES)
        RES = np.reshape(RES, (count,size_of_data,1))
        labels = np.array(labels)
        yield (RES,labels)

def Print_EKG(signal):
    size = len(signal)
    plt.plot(range(size),signal)
    plt.xlabel(r'$time$' ,  fontsize=15, horizontalalignment='right' , x=1)
    plt.ylabel(r'$value$',  fontsize=15, horizontalalignment='right',  y=1)

def visualize_latent_space(start, end, decoder, count_of_step, size_of_data):
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
        ax1.plot(x, final_out[0])
        ax1.plot(x, final_out[-1])
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




def Get_bad_EKG(ecg1,ecg2,ecg3,decoder):
    alpha1 = random.normalvariate(0.5,0.2)
    alpha2 = random.normalvariate(0.5,0.2)
    alpha3 = random.normalvariate(0.5,0.2)
    data1 = alpha1*ecg1 + (1-alpha1)*ecg2
    data2 = alpha2*ecg1 + (1-alpha2)*ecg3
    data3 = alpha3*ecg2 + (1-alpha3)*ecg3
    tmp = np.array([data1,data2,data3])
    return decoder.predict(tmp)

def Generate_bad_data(count, path, encoder , decoder, data, size_of_data):
    RES = {}
    generator = Create_data_generator(data = data,
                                        count_augmentation_data =0,
                                        count_of_diffrent_signals=3,
                                        size_of_data= size_of_data) # создаём генератор

    index = 0
    for i in range(count):
        data_for_denerator = next(generator)[0]
        ecg1, ecg2, ecg3 = encoder.predict(data_for_denerator)
        bad = Get_bad_EKG(ecg1, ecg2, ecg3, decoder)
        bad =  np.reshape(bad, (3,size_of_data))
        for j in range(3):
            RES[index] = bad[j]
            index = index+1
            print(str(index) + "/"+ str(count*3) + " saved" )

    np.save(path+'.npy', RES)        



def Load_bad_EKG(path):
    return np.load(path).item()

def visualize_learning(history, graph1, graph2 ):
    plt.plot(history.history[graph1])
    plt.plot(history.history[graph2])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

def visualize_learning_model(history, graph1,graph2):
    plt.subplot(2,1,1)
    # (x^2+x^2)/2 = x^2 -> x =sqrt(x^2)
    plt.plot([mt.sqrt(i) for i in history.history[graph1]])
    plt.legend(['classificator'], loc='upper left')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.subplot(2,1,2)
    plt.plot(history.history[graph2])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['autoencoder'], loc='upper left')
    plt.grid()
    plt.show()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lern_rate = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lern_rate.append( K.eval(self.model.optimizer.lr) ) 

class Reconstr_on_each_epoch(keras.callbacks.Callback):
    def __init__(self, signal_reconst):
        self.signal = signal_reconst # shape is (2,2000,1)
    def on_train_begin(self, logs={}):
        self.arr = []

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.signal)[0]
        self.arr.append(output)


###################################################
## Functions for create some models
##################################################

def create_classificator(input_for_classificator):
    drop_out_rate = 0.3
    x = Conv1D(15, 100, activation=LeakyReLU(alpha = 0.2), padding='same')(input_for_classificator)
    x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(10, 80,activation=LeakyReLU(alpha = 0.2), padding='same')(x)
    x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(20, 30, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(30, activation='sigmoid')(x)
    x = Dropout(drop_out_rate)(x)
    x = Dense(15, activation='sigmoid')(x)
    x = Dropout(drop_out_rate)(x)
    out = Dense(1, activation='sigmoid')(x) 
    return Model(input_for_classificator, out) 

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


def create_big_model( input_for_model, encoder, decoder, classificator):
    x = encoder(input_for_model)
    x1 = Lambda( lambda x: K.slice(x, (0, 0, 0), (1, -1, -1)))(x)
    x2 = Lambda( lambda x: K.slice(x, (1, 0, 0), (1, -1, -1)))(x)
    center = Lambda (lambda x: x*0.5)( Add()([x1, x2]) ) 
    psevdo_ecg = decoder(center)
    out_of_classificator = classificator(psevdo_ecg)
    out_of_decoder = decoder(x)
    return  Model(inputs=input_for_model,outputs=[out_of_decoder, out_of_classificator])
