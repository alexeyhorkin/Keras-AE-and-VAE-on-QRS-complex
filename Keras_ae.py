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
epochs = 1
count_of_diffrent_signals = 10
count_augmentation_data = 4
batch_size = count_of_diffrent_signals * (count_augmentation_data + 1)
latent_dim = 200
mode_of_learning = None


# define inputs placeholders for all models
input_for_encoder = Input(batch_shape=(None,size_of_data,1) )
input_for_decoder = Input(batch_shape=(None ,latent_dim), name = 'input_for_decoder' )


# work with the whole model
encoder, shape = create_encoder(input_for_encoder, latent_dim)
decoder = create_decoder(input_for_decoder, shape) 
encoder.summary()
decoder.summary()


out = decoder ( encoder(input_for_encoder) )
autoencoder  = Model( input_for_encoder, out ) # create autoencoder
autoencoder.summary()


def my_loss(x,decoded):
	return K.mean((out- input_for_encoder)**2)

optimizer = keras.optimizers.Adadelta(learning_rate=0.1, rho=0.01)
autoencoder.compile(optimizer=optimizer, loss = 'mse') 		

generator_train = Create_data_generator(data,count_augmentation_data,count_of_diffrent_signals,size_of_data,'train', mode_of_learning)
generator_test = Create_data_generator(data,count_augmentation_data,count_of_diffrent_signals,size_of_data,'test', mode_of_learning) 


# my_history = LossHistory()

##################################
rd.seed(35)
signal_reconst = next(generator_train)[0][0]
plt.plot(np.reshape(signal_reconst, (size_of_data)))
plt.show()
signal_reconst = np.array([signal_reconst])
reconst = Reconstr_on_each_epoch(signal_reconst)
###################################

rd.seed(10) # установить зерно на оубчение
history = autoencoder.fit_generator(generator_train, validation_data = generator_test, validation_steps =10,
                    steps_per_epoch=count_of_batch, epochs=epochs , callbacks = [reconst])

autoencoder.save("autoencoder_model_200_150epoch.h5")
encoder.save("encoder_model_200_150epoch.h5")
decoder.save("decoder_model_200_150epoch.h5")
visualize_learning(history, 'loss', 'val_loss') # print loss history 


# save learning history and reconstr
loss = history.history['loss']
val_loss = history.history['val_loss']
statistic = {'loss': loss , 'val_loss' : val_loss}
# with open( 'statistic_of_learning_150epoch.pickle', 'wb') as f:
# 	pickle.dump(statistic, f)

# with open('reconstr_150epoch.pickle', 'wb') as z:
# 	pickle.dump([signal_reconst, reconst.arr], z)


# print reconstruction
arr_of_seed = [1,2,3,10]
rd.seed(10)
for i, seed in enumerate(arr_of_seed):
	plt.subplot(2,2,i+1)
	data1 = next(generator_train)[0]
	plt.plot(np.reshape(data1[0], (size_of_data)))
	data_out = autoencoder.predict(data1)
	plt.plot(np.reshape(data_out[0], (size_of_data)))
	plt.legend(['Input', 'Output'], loc='upper left')
plt.show()

# print interpolation
generator_train = Create_data_generator(data,0,2,size_of_data,'test', 'healthy')
next(generator_train)
rd.seed(1)
data2 = next(generator_train)[0]
plt.plot(np.reshape(data2[0], (size_of_data)))
plt.plot(np.reshape(data2[1], (size_of_data)))
plt.legend([r'$first \ \ ecg \ \ signal$', r'$second \ \ ecg \ \ signal$'], loc='upper left', fontsize=12)
plt.xlabel(r'$time$', fontsize=15)
plt.ylabel(r'$value$' , fontsize=15)
plt.show()
start, end = encoder.predict(data2)
visualize_latent_space(start,end,decoder,100,size_of_data)




