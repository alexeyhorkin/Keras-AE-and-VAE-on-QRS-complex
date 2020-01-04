from matplotlib.animation import FuncAnimation
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from Functions import *
from keras.models import load_model
import os
# ecg_data_200
# data_2033
# data_1078

# dataset_path="..\\DATA\\data_1078.json" #файл с датасетом
# data = Openf(dataset_path) # Open data file
# print("file was opened")
# create data test and train
# data_train = { k: data[k] for k in list(data.keys())[:count_of_train]} 
# data_test =  { k: data[k] for k in list(data.keys())[count_of_train:]}

# generator_train = generator_QRS_complexes(data_train,batch_size, cycle_lenght)
# generator_test = generator_QRS_complexes(data_test,batch_size, cycle_lenght)

# open data
with open('DATA_big.pickle', 'rb') as z:
	data = pickle.load(z)


###############################################
## Building a model
###############################################
procent_train = 0.8
SIZE = 5000
cycle_lenght = 256
size_of_data = cycle_lenght*2
count_of_batch = 20
learning_rate = 0.1
epochs = 100
batch_size = 100
latent_dim = 30


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


# create data test and train
count_of_train = int(len(data)*procent_train)
data_train = data[:count_of_train]
data_test =  data[count_of_train:]
generator_train = generator_QRS_complexes_on_big_data(data_train,batch_size, cycle_lenght)
generator_test = generator_QRS_complexes_on_big_data(data_test,batch_size, cycle_lenght)




# for visualization reconstraction per epoches
##################################
rd.seed(35)
signal_reconst = next(generator_train)[0][0]
# plt.plot(np.reshape(signal_reconst, (size_of_data)))
# plt.show()
signal_reconst = np.array([signal_reconst])
reconst = Reconstr_on_each_epoch(signal_reconst)
###################################

history = autoencoder.fit_generator(generator_train, validation_data = generator_test, validation_steps =count_of_batch,
                    steps_per_epoch=count_of_batch, epochs=epochs , callbacks = [reconst])


folder_path = 'save model AE/'
if not os.path.exists(os.path.dirname(folder_path)):
	os.makedirs(folder_path)
autoencoder.save(folder_path + "autoencoder_model.h5")
encoder.save(folder_path + "encoder_model.h5")
decoder.save(folder_path + "decoder_model.h5")
visualize_learning(history, 'loss', 'val_loss') # print loss history 



folder_path_for_hist = 'history of learning AE/'
if not os.path.exists(os.path.dirname(folder_path_for_hist)):
	os.makedirs(folder_path_for_hist)
# save learning history and reconstr
loss = history.history['loss']
val_loss = history.history['val_loss']
statistic = {'loss': loss , 'val_loss' : val_loss}

with open(folder_path_for_hist +'statistic_of_learning.pickle', 'wb') as f:
	pickle.dump(statistic, f)

with open(folder_path_for_hist + 'reconstr.pickle', 'wb') as z:
	pickle.dump([signal_reconst, reconst.arr], z)


# print reconstruction
rd.seed(10)
for i in range(1,10):
	plt.subplot(3,3,i)
	data1 = next(generator_train)[0]
	plt.plot(np.reshape(data1[0], (size_of_data)))
	data_out = autoencoder.predict(data1)
	plt.plot(np.reshape(data_out[0], (size_of_data)))
	plt.legend(['Input', 'Output'], loc='upper left')
plt.show()



# print interpolation
rd.seed(33)
for i in range(20):
	postfix_for_file = str(i)
	data2 = next(generator_train)[0]
	start_true = np.reshape(data2[0], (size_of_data))
	end_true = np.reshape(data2[1], (size_of_data))
	# show 2 ecg signals for interpolation
	# plt.plot(start_true)
	# plt.plot(end_true)
	# plt.legend([r'$first \ \ ecg \ \ signal$', r'$second \ \ ecg \ \ signal$'], loc='upper left', fontsize=12)
	# plt.xlabel(r'$time$', fontsize=15)
	# plt.ylabel(r'$value$' , fontsize=15)
	# plt.show()
	start, end = encoder.predict(data2)[:2]
	visualize_latent_space(start_true, end_true, start,end,decoder,100,size_of_data, folder_path_for_hist, postfix_for_file)




