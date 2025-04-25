import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
print("")
import title_screen
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import random 
from scipy.fft import fft, fftfreq, fftshift
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tqdm import tqdm 
import time
import ast
import tensorflow.keras.backend as K
print("No time evolution")
print("")


def rmsd(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

file_name = "EXPERIMENT_TRAINING_SET_10K_5.csv"
training_df = pd.read_csv(file_name)
print("Training set: " + file_name)

x_coupled = training_df["Coupled"]
y = training_df["FID"]
x_input = []
y_output = []

print("Parsing CSV file:")
for j in tqdm(range(len(x_coupled))):
    x_coupled[j] = ast.literal_eval(x_coupled[j])
    y[j] = ast.literal_eval(y[j])

#This loop converts our training data from the form [R1+I1, R2+I2, R3+I3] to [R1, R2, R3, I1, I2, I3]
#This is because the model cannot work with complex numbers, 
#hence each complex value must be split into real and imaginary components.
print("Converting complex FIDs to real components:")
for i in tqdm(range(len(x_coupled))):
    x_coupled_real_add = []
    x_coupled_imag_add = []
    y_real_add = []
    y_imag_add = []
    for j in range(len(x_coupled[i])):
        x_coupled_real_add.append(x_coupled[i][j].real)
        x_coupled_imag_add.append(x_coupled[i][j].imag)
        y_real_add.append(y[i][j].real)
        y_imag_add.append(y[i][j].imag)
    x_final_add = np.concatenate((x_coupled_real_add, x_coupled_imag_add), axis=0)
    y_final_add = np.concatenate((y_real_add, y_imag_add), axis=0)
    x_input.append(x_final_add)
    y_output.append(y_final_add)

x_input = np.array(x_input)
y_output = np.array(y_output)
x_input = x_input.reshape(-1, 1024)
y_output = y_output.reshape(-1, 1024)


print("Training model:")
print("")

#Model architecture:
layer_size = 1024

input_layer_x = Input(shape=(x_input.shape[1],))

x_branch = Dense(layer_size, activation="relu")(input_layer_x)
hidden1 = Dense(layer_size, activation="relu")(x_branch)
hidden2 = Dense(layer_size, activation="relu")(hidden1)
hidden3 = Dense(layer_size, activation="relu")(hidden2)
hidden4 = Dense(layer_size, activation="relu")(hidden3)
hidden5 = Dense(layer_size, activation="relu")(hidden4)
output_layer = Dense(1024)(hidden5)
#Model architecture^


model = Model(inputs=[input_layer_x], outputs=output_layer)
model.compile(optimizer=Adam(), loss="mse", metrics=[rmsd])
print("Number of parameters: " + str(model.count_params()))

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1) #Slows the learning rate when the decrease in loss slows down
early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1) #Stops training when convergence is reached.

start_time = time.time()
history = model.fit(
    [x_input], 
    y_output, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[lr_scheduler, early_stopping])
end_time = time.time()

print("")
print("**************************************************************")
print("Training complete")
training_time = end_time - start_time
print("Training time: " + str(round(training_time, 2)) + " seconds")
print("")

loss_history = history.history["loss"]
log_loss = np.log(loss_history)
min_loss = min(loss_history)
rmsd_history = history.history["rmsd"]
min_rmsd = min(rmsd_history)
print("Final training loss: " + str(round(min_loss, 5)))
print("")
print("Change in training loss: " + str(round(loss_history[0]-min_loss, 5)))
print("")
print("Loss decrease: " + str(round(((loss_history[0]-min_loss)/loss_history[0])*100, 5)) + "%")
print("")
print("Final RMSD: " + str(round(min_rmsd, 5)))
print("")
val_loss_history = history.history["val_loss"]
loss_fig = plt.figure(figsize=(15, 6))
loss_fig.canvas.manager.set_window_title("Change in loss")
plt.plot(loss_history, color="red", label="Training loss")
plt.plot(val_loss_history, color="blue", label="Validation loss")
plt.plot(rmsd_history, color="green", label="RMSD")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Change in loss")
plt.show()
loss_history = history.history["val_loss"]
min_loss = min(loss_history)
log_loss = np.log(loss_history)
rmsd_history = history.history["val_rmsd"]
min_rmsd = min(rmsd_history)
print("Final Loss: " + str(round(min_loss, 5)))
print("")
print("Change in loss: " + str(round(loss_history[0]-min_loss, 5)))
print("")
print("Loss decrease: " + str(round(((loss_history[0]-min_loss)/loss_history[0])*100, 5)) + "%")
print("")
print("Final RMSD: " + str(round(min_rmsd, 5)))
print("")
loss_loss = history.history["loss"]
loss_fig = plt.figure(figsize=(15, 6))
loss_fig.canvas.manager.set_window_title("Change in loss")
plt.plot(loss_history, color="red", label="val_loss")
plt.plot(loss_loss, color="blue", label="loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Change in loss")
plt.show()


print("")
print("")
model.summary()
print("")
print("")

