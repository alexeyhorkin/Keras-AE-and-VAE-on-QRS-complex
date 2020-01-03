from Functions import *
from keras.models import Model
from matplotlib.animation import FuncAnimation
from tensorflow.python.keras.callbacks import TensorBoard , ModelCheckpoint
from time import time
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.losses import mse
import pickle
import os

# ecg_data_200
# data_2033
# data_1078
dataset_path="..\\data_2033\\data_1078.json" #файл с датасетом
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
input_for_encoder = Input(batch_shape=(None ,size_of_data,1), name = 'input_for_encoder' )
latent_inputs = Input(batch_shape=(None ,latent_dim), name = 'input_for_decoder' )

# build models
encoder, decoder, vae , z_mean, z_log_var, outputs  = get_vae(input_for_encoder, latent_dim, latent_inputs)

# print summary
encoder.summary()
decoder.summary()
vae.summary()

vae_loss = Get_loss(z_mean, z_log_var, size_of_data, input_for_encoder, outputs)

def get_loss(x, decoded):
    return get_KL_loss(x, decoded) + get_RL(x,decoded)
def get_KL_loss(x, decoded):
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)
def get_RL(x, decoded):
    reconstruction_loss = K.mean((outputs - input_for_encoder) ** 2, axis=(1, 2))
    reconstruction_loss*=size_of_data
    return K.mean(reconstruction_loss)


vae.compile(optimizer=keras.optimizers.Adadelta(learning_rate=0.1, rho=0.01),
            loss = get_loss , metrics = [get_KL_loss, get_RL] )
# vae.compile(optimizer = 'rmsprop')


# define data for train and test and create generators for 
count_of_train = int(len(data.keys())*procent_train)
data_train = { k: data[k] for k in list(data.keys())[:count_of_train]} 
data_test =  { k: data[k] for k in list(data.keys())[count_of_train:]}
generator_train = generator_QRS_complexes(data_train,batch_size, cycle_lenght)
generator_test = generator_QRS_complexes(data_test,batch_size, cycle_lenght)

######################################
rd.seed(35)
signal_reconst = next(generator_train)[0][0]
# plt.plot(np.reshape(signal_reconst, (size_of_data)))
# plt.show()
signal_reconst = np.array([signal_reconst])
reconst = Reconstr_on_each_epoch(signal_reconst)
######################################
history = vae.fit_generator(generator_train, steps_per_epoch=count_of_batch, epochs=epochs,
            callbacks = [reconst],  validation_data = generator_test, validation_steps =count_of_batch) 



folder_path = 'save model VAE/'
if not os.path.exists(os.path.dirname(folder_path)):
    os.makedirs(folder_path)
vae.save(folder_path + "vae.h5") # save the model
encoder.save(folder_path + "encoder.h5") # save the model
decoder.save(folder_path + "decoder.h5") # save the model

folder_path_for_hist = 'history of learning VAE/'
if not os.path.exists(os.path.dirname(folder_path_for_hist)):
    os.makedirs(folder_path_for_hist)

# save learning history and reconstr
loss = history.history['loss']
kl_loss = history.history['get_KL_loss']
val_loss = history.history['val_loss']
val_kl_loss = history.history['val_get_KL_loss']
statistic = {'loss': loss, 'kl_loss':kl_loss, 'val_loss':val_loss, 'val_kl_loss':val_kl_loss}

with open(folder_path_for_hist +'statistic_of_learning_vae.pickle', 'wb') as f:
    pickle.dump(statistic, f)
with open(folder_path_for_hist + 'reconstr_vae.pickle', 'wb') as z:
    pickle.dump([signal_reconst, reconst.arr], z)


# print reconstruction
rd.seed(10)
for i in range(1,10):
    plt.subplot(3,3,i)
    data1 = next(generator_train)[0]
    plt.plot(np.reshape(data1[0], (size_of_data)))
    data_out = vae.predict(data1)
    plt.plot(np.reshape(data_out[0], (size_of_data)))
    plt.legend(['Input', 'Output'], loc='upper left')
plt.show()

# print loses history
plt.title("Losses")
plt.subplot(2,1,1)
plt.plot(loss)
plt.plot(val_loss) 
plt.legend(['Reconstruction loss train', 'Reconstruction loss test'], loc='upper left')
plt.subplot(2,1,2)
plt.plot(kl_loss)
plt.plot(val_kl_loss)
plt.legend(['KL loss train', 'KL loss train'], loc='upper left')
plt.show()



# visual interpolation
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
start, end = encoder.predict(data2)[2][:2]
visualize_latent_space(start_true, end_true, start,end,decoder,100,size_of_data, folder_path_for_hist)


