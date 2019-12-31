from matplotlib.animation import FuncAnimation
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from cycle_generator import *
# ecg_data_200
# data_2033
dataset_path="..\\data_2033\\ecg_data_200.json" #файл с датасетом
data = Openf(dataset_path) # Open data file
print("file was opened")

###############################################
## Building a model
###############################################
procent_train = 0.8
SIZE = 5000
cycle_lenght = 256
size_of_data = cycle_lenght*2
count_of_batch = 35
learning_rate = 0.1
epochs = 2
batch_size = 40
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
count_of_train = int(len(data.keys())*procent_train)
data_train = { k: data[k] for k in list(data.keys())[:count_of_train]} 
data_test =  { k: data[k] for k in list(data.keys())[count_of_train:]}

generator_train = generator_QRS_complexes(data_train,batch_size, cycle_lenght)
generator_test = generator_QRS_complexes(data_test,batch_size, cycle_lenght)


# for visualization reconstraction per epoches
##################################
rd.seed(35)
signal_reconst = next(generator_train)[0][0]
plt.plot(np.reshape(signal_reconst, (size_of_data)))
plt.show()
signal_reconst = np.array([signal_reconst])
reconst = Reconstr_on_each_epoch(signal_reconst)
###################################

history = autoencoder.fit_generator(generator_train, validation_data = generator_test, validation_steps =count_of_batch,
                    steps_per_epoch=count_of_batch, epochs=epochs , callbacks = [reconst])

autoencoder.save("autoencoder_model_200_150epoch.h5")
encoder.save("encoder_model_200_150epoch.h5")
decoder.save("decoder_model_200_150epoch.h5")
visualize_learning(history, 'loss', 'val_loss') # print loss history 


# save learning history and reconstr
loss = history.history['loss']
val_loss = history.history['val_loss']
statistic = {'loss': loss , 'val_loss' : val_loss}
with open( 'statistic_of_learning_150epoch.pickle', 'wb') as f:
	pickle.dump(statistic, f)

with open('reconstr_150epoch.pickle', 'wb') as z:
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
rd.seed(1)
data2 = next(generator_train)[0]
start_true = np.reshape(data2[0], (size_of_data))
end_true = np.reshape(data2[1], (size_of_data))
plt.plot(start_true)
plt.plot(end_true)
plt.legend([r'$first \ \ ecg \ \ signal$', r'$second \ \ ecg \ \ signal$'], loc='upper left', fontsize=12)
plt.xlabel(r'$time$', fontsize=15)
plt.ylabel(r'$value$' , fontsize=15)
plt.show()
start, end = encoder.predict(data2)[:2]
visualize_latent_space(start_true, end_true, start,end,decoder,100,size_of_data)



