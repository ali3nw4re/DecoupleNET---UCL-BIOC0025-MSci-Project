import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
print("")
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import random 
from scipy.fft import fft, fftfreq, fftshift
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import tensorflow.keras.backend as K

class minuk_gen_FID:
    #Spectra parameters
    R2_upper = 60.
    R2_lower = 5.
    A_mean = 1.
    A_SD = 0.5
    points = 512
    tpi = 2 * np.pi
    magnet_strength_hz = 800 * 10**6 #In Hz
    reference_frequency = magnet_strength_hz * (1/4) #Because C13 NMR
    ppm_upper = 65            
    ppm_lower = 50
    sweep_width = (ppm_upper - ppm_lower) * (reference_frequency / 10**6) # = 3000
    duration = points / sweep_width # = 0.1706 
    J_upper = 40
    J_lower = 28
    max_nuclei = 5
    min_nuclei = 1 
    time_period = 0.0035
    noise_magnitude = 0.1

    def normal_A(self):
        return random.normal(loc=self.A_mean, scale=self.A_SD)
    
    def __init__(self):
        J = 0
        A = 1
        omega = 0
        couple_degree = 0
        number_of_nuclei = random.randint(self.min_nuclei, self.max_nuclei)
        full_FID = np.zeros(self.points, dtype="complex_")
        full_coupled_FID = np.zeros(self.points, dtype="complex_")
        full_coupled_T_FID = np.zeros(self.points, dtype="complex_")
        for i in range(number_of_nuclei):    
            omega = random.uniform(-self.sweep_width/2, self.sweep_width/2)
            t = np.linspace(0, self.duration, self.points)
            A = self.normal_A()
            while A > 2 or A < 0:
                A = self.normal_A()
            R2 = random.uniform(self.R2_lower, self.R2_upper)
            FID = A * np.exp(1j * t * omega - R2 * t)
            FID[0] = FID[0] / 2
            coupled_FID = FID
            coupled_T_FID = FID
            couple_degree = random.randint(0, 3)
            J = random.uniform(self.J_lower, self.J_upper)
            coupled_FID = FID * (np.cos(np.pi * J * t))**couple_degree
            t += self.time_period
            coupled_T_FID = FID * (np.cos(np.pi * J * t))**couple_degree
            noise_real = random.normal(loc=0, scale=self.noise_magnitude, size=self.points)
            noise_imag = random.normal(loc=0, scale=self.noise_magnitude, size=self.points)
            noise = noise_real + 1j * noise_imag
            full_FID = full_FID + FID
            full_coupled_FID = full_coupled_FID + coupled_FID + noise
            full_coupled_T_FID = full_coupled_T_FID + coupled_T_FID + noise

        self.out_full_FID = full_FID
        self.out_coupled_FID = full_coupled_FID
        self.out_coupled_T_FID = full_coupled_T_FID
        self.out_omega = omega
        self.out_couple_degree = couple_degree
        self.out_J = J
        self.out_A = A
        self.out_no_nuclei = number_of_nuclei

#Function to convert an array in the form [R1, R2, R3, I1, I2, I3] to [R1+I1, R2+I2, R3+I3]
def output_parse(array):
    final_prediction = []
    transpose = int(len(array)/2)
    for i in range(transpose):
        real = array[i]
        imag = array[i+transpose]
        add = complex(real, imag)
        final_prediction.append(add)
    return final_prediction

def gen_FID():
    return minuk_gen_FID()

def rmsd(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

verif_number = input("How many verification examples would you like? ")
for i in range(int(verif_number)):
    print("********************************* Verification example " + str(i + 1) + " *********************************")
    m = gen_FID()
    full_FID = m.out_full_FID
    full_coupled_FID = m.out_coupled_FID
    full_coupled_T_FID = m.out_coupled_T_FID

    spectrum = fftshift(fft(full_FID))
    spectrum2 = fftshift(fft(full_coupled_FID))
    spectrum3 = fftshift(fft(full_coupled_T_FID))
    x_axis = fftshift(fftfreq(minuk_gen_FID.points, minuk_gen_FID.duration / minuk_gen_FID.points))
    ppm_center = (minuk_gen_FID.ppm_lower + minuk_gen_FID.ppm_upper) / 2
    ppm = ((x_axis) / (minuk_gen_FID.reference_frequency / 10**6)) + ppm_center

    verify_no_nuclei = m.out_no_nuclei
    verify_J = m.out_J
    verify_A = m.out_A
    verify_omega = m.out_omega
    verify_coupling_degree = m.out_couple_degree

    print("Number of nuclei / peaks: " + str(verify_no_nuclei))
    print("Parameters for the last peak on the spectrum:")
    print("J: " + str(verify_J))
    print("A: " + str(verify_A))
    print("Omega: " + str(verify_omega))
    print("Coupling degree: " + str(verify_coupling_degree))
    print("")
    print("")
    print("INPUT: ")
    print("Peak PPM: " + str(ppm[spectrum.argmax()]))

    #1D TRAIN PREDICTION
    pre_trained_model = "saved_model/DecoupleNET_1D_10k.keras"
    print("")
    print("Loading model: " + str(pre_trained_model))
    model = tf.keras.models.load_model(pre_trained_model, custom_objects={"rmsd": rmsd})

    model_input = []
    for j in range(len(full_coupled_FID)):
        model_input.append(full_coupled_FID[j].real)
    for k in range((len(full_coupled_FID))):
        model_input.append(full_coupled_FID[k].imag)
    model_input = np.array(model_input)
    model_input = model_input.reshape(1, -1)
    prediction = model.predict([model_input])
    output = []
    for n in range(len(prediction[0])):
        output.append(prediction[0][n])
    final_output = output_parse(output)
    final_spectrum = fftshift(fft(final_output))

    print("1D Train PREDICTION: ")
    print("Predicted peak PPM: "  + str(ppm[final_spectrum.argmax()]))
    print("Difference between peaks: " + str(np.abs((ppm[final_spectrum.argmax()]) - (ppm[spectrum.argmax()]))))

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle("1D Train prediction", fontsize=16)
    fig.canvas.manager.set_window_title("DecoupleNET prediction")
    axs[0][0].plot(np.real(ppm), np.real(spectrum2))
    axs[0][0].set_title("Coupled (Input)")
    axs[0][0].set_xlabel("PPM")
    axs[0][1].plot(np.real(ppm), np.real(spectrum3))
    axs[0][1].set_title("Coupled + T (Input)")
    axs[0][1].set_xlabel("PPM")
    axs[0][2].plot(np.real(ppm), np.real(spectrum), color="green")
    axs[0][2].set_title("Decoupled (Target)")
    axs[0][2].set_xlabel("PPM")
    axs[1][0].plot(np.real(ppm), np.real(spectrum), label="Decoupled", color="green")
    axs[1][0].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
    axs[1][0].set_title("Coupled + Decoupled (Overlaid)")
    axs[1][0].set_xlabel("PPM")
    axs[1][0].legend()
    axs[1][1].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
    axs[1][1].plot(np.real(ppm), np.real(spectrum3), label="Coupled + T")
    axs[1][1].set_title("Coupled + Coupled+T (Overlaid)")
    axs[1][1].set_xlabel("PPM")
    axs[1][1].legend()
    axs[1][2].plot(np.real(ppm), np.real(final_spectrum), color="red")
    axs[1][2].set_title("Prediction")
    axs[1][2].set_xlabel("PPM")
    for row in axs:
        for ax in row:
            ax.invert_xaxis()
            ax.spines["top"].set_visible(False)   
            ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("")
    print("")
    print("")
    print("")

    #DecoupleNET PREDICTION
    pre_trained_model = "saved_model/DecoupleNET_10k.keras"
    print("")
    print("Loading model: " + str(pre_trained_model))
    model = tf.keras.models.load_model(pre_trained_model, custom_objects={"rmsd": rmsd})

    model_input_t = []
    for l in range((len(full_coupled_T_FID))):
        model_input_t.append(full_coupled_T_FID[l].real)
    for m in range((len(full_coupled_T_FID))):
        model_input_t.append(full_coupled_T_FID[m].imag)
    model_input_t = np.array(model_input_t)
    model_input_t = model_input_t.reshape(1, -1)
    prediction = model.predict([model_input, model_input_t])
    output = []
    for n in range(len(prediction[0])):
        output.append(prediction[0][n])
    final_output = output_parse(output)
    final_spectrum = fftshift(fft(final_output))

    print("DecoupleNET PREDICTION: ")
    print("Predicted peak PPM: "  + str(ppm[final_spectrum.argmax()]))
    print("Difference between peaks: " + str(np.abs((ppm[final_spectrum.argmax()]) - (ppm[spectrum.argmax()]))))

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle("DecoupleNET prediction", fontsize=16)
    fig.canvas.manager.set_window_title("DecoupleNET prediction")
    axs[0][0].plot(np.real(ppm), np.real(spectrum2))
    axs[0][0].set_title("Coupled (Input)")
    axs[0][0].set_xlabel("13C PPM")
    axs[0][1].plot(np.real(ppm), np.real(spectrum3))
    axs[0][1].set_title("Coupled + T (Input)")
    axs[0][1].set_xlabel("13C PPM")
    axs[0][2].plot(np.real(ppm), np.real(spectrum), color="green")
    axs[0][2].set_title("Decoupled (Target)")
    axs[0][2].set_xlabel("13C PPM")
    axs[1][0].plot(np.real(ppm), np.real(spectrum), label="Decoupled", color="green")
    axs[1][0].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
    axs[1][0].set_title("Coupled + Decoupled (Overlaid)")
    axs[1][0].set_xlabel("13C PPM")
    axs[1][0].legend()
    axs[1][1].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
    axs[1][1].plot(np.real(ppm), np.real(spectrum3), label="Coupled + T")
    axs[1][1].set_title("Coupled + Coupled+T (Overlaid)")
    axs[1][1].set_xlabel("13C PPM")
    axs[1][1].legend()
    axs[1][2].plot(np.real(ppm), np.real(final_spectrum), color="red")
    axs[1][2].set_title("Prediction")
    axs[1][2].set_xlabel("13C PPM")
    for row in axs:
        for ax in row:
            ax.invert_xaxis()
            ax.spines["top"].set_visible(False)   
            ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("")
    print("")
    print("")
    print("")

    #DecoupleNET_conv PREDICTION
    pre_trained_model = "saved_model/DecoupleNET_CONV_10k.keras"
    print("")
    print("Loading model: " + str(pre_trained_model))
    model = tf.keras.models.load_model(pre_trained_model, custom_objects={"rmsd": rmsd})

    prediction = model.predict([model_input, model_input_t])
    prediction = tf.reshape(prediction, [-1])
    output = np.array(prediction)
    final_output = output_parse(output)
    final_spectrum = fftshift(fft(final_output))

    print("DecoupleNET_conv PREDICTION: ")
    print("Predicted peak PPM: "  + str(ppm[final_spectrum.argmax()]))
    print("Difference between peaks: " + str(np.abs((ppm[final_spectrum.argmax()]) - (ppm[spectrum.argmax()]))))

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle("DecoupleNET_conv prediction", fontsize=16)
    fig.canvas.manager.set_window_title("DecoupleNET prediction")
    axs[0][0].plot(np.real(ppm), np.real(spectrum2))
    axs[0][0].set_title("Coupled (Input)")
    axs[0][0].set_xlabel("13C PPM")
    axs[0][1].plot(np.real(ppm), np.real(spectrum3))
    axs[0][1].set_title("Coupled + T (Input)")
    axs[0][1].set_xlabel("13C PPM")
    axs[0][2].plot(np.real(ppm), np.real(spectrum), color="green")
    axs[0][2].set_title("Decoupled (Target)")
    axs[0][2].set_xlabel("13C PPM")
    axs[1][0].plot(np.real(ppm), np.real(spectrum), label="Decoupled", color="green")
    axs[1][0].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
    axs[1][0].set_title("Coupled + Decoupled (Overlaid)")
    axs[1][0].set_xlabel("13C PPM")
    axs[1][0].legend()
    axs[1][1].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
    axs[1][1].plot(np.real(ppm), np.real(spectrum3), label="Coupled + T")
    axs[1][1].set_title("Coupled + Coupled+T (Overlaid)")
    axs[1][1].set_xlabel("13C PPM")
    axs[1][1].legend()
    axs[1][2].plot(np.real(ppm), np.real(final_spectrum), color="red")
    axs[1][2].set_title("Prediction")
    axs[1][2].set_xlabel("13C PPM")
    for row in axs:
        for ax in row:
            ax.invert_xaxis()
            ax.spines["top"].set_visible(False)   
            ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
        