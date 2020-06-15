from functions import *
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, BatchNormalization,Flatten, Dropout, Lambda, Add, add
from keras.models import Model
from matplotlib.animation import FuncAnimation
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
import pickle
# ecg_data_200
# data_2033
dataset_path="..\\..\\data_2033\\ecg_data_200.json" #файл с датасетом
data = Openf(dataset_path) # Open data file
print("file was opened")

###############################################
## Building a model
###############################################
SIZE = 5000
size_of_data = 2000
count_of_batch = 35
learning_rate = 0.1
epochs = 10
count_of_diffrent_signals = 10
count_augmentation_data = 4
batch_size = count_of_diffrent_signals * (count_augmentation_data + 1)
latent_dim = 200
mode_of_learning = None


# define inputs placeholders for all models
input_for_encoder = Input(batch_shape=(None,size_of_data,1) )
input_for_decoder = Input(batch_shape=(None ,latent_dim), name = 'input_for_decoder' )


# # work with the whole model
encoder, shape = create_encoder(input_for_encoder, latent_dim)
decoder = create_decoder(input_for_decoder, shape) 
encoder.summary()
decoder.summary()


out = decoder ( encoder(input_for_encoder) )
autoencoder  = Model( input_for_encoder, out ) # create autoencoder
autoencoder.summary()


def my_loss(x,decoded):
	return K.mean((out- input_for_encoder)**2)

generator_train = Create_data_generator(data,count_augmentation_data,count_of_diffrent_signals,size_of_data,'train', mode_of_learning)
generator_test = Create_data_generator(data,count_augmentation_data,count_of_diffrent_signals,size_of_data,'test', mode_of_learning) 

next(generator_train)
next(generator_train)

autoencoder = load_model('autoencoder_model_200_150epoch.h5', custom_objects={'my_loss': my_loss})
print("Weights have been loaded")

decoder = load_model('decoder_model_200_150epoch.h5', custom_objects={'my_loss': my_loss})
print("Weights have been loaded")

encoder = load_model('encoder_model_200_150epoch.h5', custom_objects={'my_loss': my_loss})
print("Weights have been loaded")





/


# next(generator_train)
# # print reconstruction
# arr_of_seed = [1,2,3,10]
# rd.seed(35)
# for i, seed in enumerate(arr_of_seed):
# 	plt.subplot(2,2,i+1)
# 	data1 = next(generator_train)[0]
# 	plt.plot(np.reshape(data1[0], (size_of_data)))
# 	data_out = autoencoder.predict(data1)
# 	plt.plot(np.reshape(data_out[0], (size_of_data)))
# 	plt.legend(['Input', 'Output'], loc='upper left')
# plt.show()


# def visual(signal_for_reconst, reconstr_arr, frames, size_of_data):
#     from matplotlib.animation import FuncAnimation
#     fig = plt.figure(3)
#     ax1 = fig.add_subplot(1, 1, 1)
#     def animate(i):
#         x = np.arange(0,size_of_data)
#         y = reconstr_arr[i].reshape(size_of_data)
#         ax1.clear()
#         ax1.plot(signal_for_reconst.reshape(size_of_data))
#         ax1.plot(x, y)
#         plt.xlabel('time')
#         plt.ylabel('signal')
#         plt.title("Epoch =  "+ str(i+1))
#         plt.legend(['signal', 'output from nn'], loc='upper left')
#     anim = FuncAnimation(fig, animate,frames=frames, interval=100)
#     anim.save('animation_reconstr.gif', writer='imagemagick', fps=60)
#     plt.show()


# with open( 'reconstr_200.pickle', 'rb') as f:
#     loaded_data = pickle.load(f)

# reconstr_signal , reconstr  = loaded_data

# plt.plot(np.reshape(reconstr[9], (size_of_data)), color = 'lightgray')
# plt.plot(np.reshape(reconstr[29], (size_of_data)), color = 'gray')
# plt.plot( np.reshape(reconstr_signal[0], size_of_data), color = 'orange')
# plt.plot(np.reshape(reconstr[-1], (size_of_data)), color = 'black')
# plt.legend(['epochs = 10', 'epochs = 30', 'Input','epochs = 150'], loc='upper left')
# plt.ylabel('value')
# plt.xlabel('time')
# plt.grid()
# plt.show()





# # visual(reconstr_signal, reconstr, np.shape(reconstr)[0], np.shape(reconstr_signal)[1])

# generator_train = Create_data_generator(data,0,2,size_of_data,'test', 'healthy')
# next(generator_train)
# rd.seed(1)
# data2 = next(generator_train)[0]
# plt.plot(np.reshape(data2[0], (size_of_data)))
# plt.plot(np.reshape(data2[1], (size_of_data)))
# plt.legend([r'$first \ \ ecg \ \ signal$', r'$second \ \ ecg \ \ signal$'], loc='upper left', fontsize=12)
# plt.xlabel(r'$time$', fontsize=15)
# plt.ylabel(r'$value$' , fontsize=15)
# plt.show()
# start, end = encoder.predict(data2)
# visualize_latent_space(start,end,decoder,100,size_of_data)





# with open( 'interpol_array.pickle', 'rb') as f:
#     interpol_array = pickle.load(f)
# print(np.shape(interpol_array))

# plt.title('Interpolation')
# plt.plot(interpol_array[10], color = 'mediumaquamarine', linestyle = '--')
# plt.plot(interpol_array[20], color = 'mediumseagreen', linestyle = '--')
# plt.plot(interpol_array[52], color = 'g', linestyle = '--')	
# plt.plot(interpol_array[-10], color = 'lightgreen', linestyle = '--')
# plt.plot(interpol_array[-20], color = 'lawngreen', linestyle = '--')
# plt.plot(np.reshape(data2[0], (size_of_data)), color = 'dodgerblue')
# plt.plot(np.reshape(data2[1], (size_of_data)), color = 'orange')
# plt.legend(['post-start' ,'pre-middle ', 'middle' ,'post-middle', 'pre-end', 'ecg A', 'ecg B'], loc='upper left')
# plt.ylabel('value')
# plt.xlabel('time')
# plt.show()





with open( 'statistic_of_learning_150epoch.pickle', 'rb') as f:
    history = pickle.load(f)
plt.plot(history['loss'] , color = 'b')
plt.plot(history['val_loss'], color = 'gold')
plt.legend(['Train', 'Test'], loc='upper left')
plt.plot(history['loss'] , 'b-o')
plt.plot(history['val_loss'], 'o', color = 'gold')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.grid()
plt.show()

 