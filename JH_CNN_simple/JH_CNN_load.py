import os
import sys
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
import simplejson as simplejson
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.saving import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
#
# load json and create model  # https://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras/43263973
# referred Inherited Geek's code to load/save model and weights.
json_file = open('quasiAlexNet_2020_02_21_half15m.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
CNN_load = model_from_json(loaded_model_json)

# load weights into new model
CNN_load.load_weights('quasiAlexNet_2020_02_21_half15w.h5')
print("Loaded model from disk")

# CNN_load = Sequential()
# CNN_load = load_model('variables/quasiAlexNet_2020_02_21_half15_a.h5')
CNN_load.summary()

LDB_array = np.load(file='LDB_half_array_15.npy')
LDB_ohe = np.load(file='LDB_half_ohe_15.npy')

X_tr, X_ts, Y_tr, Y_ts = train_test_split(LDB_array, LDB_ohe, test_size=0.2)

print(f'X_tr.shape = {X_tr.shape}')  # (2423, 64, 64, 3)
print(f'Y_tr.shape = {Y_tr.shape}')  # (2423, 10)
print(f'X_ts.shape = {X_ts.shape}')  # (606, 64, 64, 3)
print(f'Y_ts.shape = {Y_ts.shape}')  # (606, 10)

#####################################################

early_stop = EarlyStopping(monitor='accuracy', verbose=1, patience=5)

model_check_point = ModelCheckpoint(filepath='variables/quasiAlexNet_2020_02_21_half15_a.h5', monitor='accuracy',
                                    verbose=1,
                                    save_best_only=True)

CNN_optimizer = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-7)  # we are going to use Adam for optimizing.

CNN_load.compile(loss='binary_crossentropy', optimizer=CNN_optimizer, metrics=['accuracy'])

#####################################################

# RUN CONVOLUTIONAL NEURAL NETWORK -> Won't it take too long time??

fitting = CNN_load.fit(x=X_tr, y=Y_tr, batch_size=100, epochs=20, verbose=1, callbacks=[early_stop, model_check_point],
                       validation_data=(X_ts, Y_ts), workers=0, use_multiprocessing=True)

# serialize model to JSON
model_json = CNN_load.to_json()
with open('quasiAlexNet_2020_02_21_half15.json', "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

# serialize weights to HDF5
CNN_load.save_weights('quasiAlexNet_2020_02_21_half15.h5')
print("Saved model to disk")

evaluation = CNN_load.evaluate(X_ts, Y_ts)
print(f'Test loss: {evaluation[0]}, accuracy: {evaluation[1]}')

# CNN_load.save('variables/quasiAlexNet_2020_02_21_half15_a.h5')
#####################################################

# Draw graphs about train and test loss/accuracy. refer from mnist_cnn.py.
# LOSS GRAPH

grp_train_loss = fitting.history['loss']
grp_test_loss = fitting.history['val_loss']

x = range(len(grp_train_loss))
plt.plot(x, grp_train_loss, marker='.', color='red', label='Train loss')
plt.plot(x, grp_test_loss, marker='.', color='blue', label='Test loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#####################################################
# ACCURACY GRAPH

grp_train_acc = fitting.history['accuracy']
grp_test_acc = fitting.history['val_accuracy']

plt.plot(x, grp_train_acc, marker='.', c='red', label='Train Acc.')
plt.plot(x, grp_test_acc, marker='.', c='blue', label='Test Acc.')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
