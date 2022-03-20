import h5py 

# Load data.
with h5py.File('/content/drive/MyDrive/KalmanNetProject/meas000.data','r') as hdf:
    ls = list(hdf.keys())
    print(ls)
    data = hdf.get('fetalSignal')
    data = np.array(data)
    data = data.reshape(6,810000)
    print(data.shape)
    peaks = hdf.get('fRpeaks')
    peaks = np.array(peaks)
    peaks = peaks.reshape(1,1631)
    print(peaks.shape)

# Reshape the data, tarin and test.
input_length = 1024
number_inputs = 791

ecg = data[:,0:input_length*number_inputs]
ecg_train = ecg[0:5,:]
print('Ecg train shape')
print(ecg_train.shape)
ecg_train = ecg_train.reshape(5*number_inputs,input_length,1)
print(ecg_train.shape)

ecg_test = ecg[5:6,:]
print('Ecg test shape')
print(ecg_test.shape)
ecg_test = ecg_test.reshape(1*number_inputs,input_length)
print(ecg_test.shape)

# Training ....
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 1

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(1024, 1),
        conv_filters=(40, 20, 20, 20, 40),
        conv_kernels=(16, 16, 16, 16, 16),
        conv_strides=(2, 2, 2, 2, 2),
        latent_space_dim= 100
    )
    autoencoder.summary()
    print('Before Compile')
    autoencoder.compile(learning_rate)
    print('Compiled')
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder

autoencoder = train(ecg_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
autoencoder.save("model")