import pandas as pd
import numpy as np 
from numpy import random 
from tqdm import tqdm 

print("")

class minuk_gen_FID:
    #Spectra parameters
    R2_upper = 60.
    R2_lower = 5.
    A_mean = 1.
    A_SD = 0.5
    points = 512
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


def gen_FID():
    return minuk_gen_FID()

number_of_spectra = int(input("How many spectra do you want in the training dataset? "))
training_df = pd.DataFrame(columns=["Coupled", "Coupled+T", "FID"])
print("")
print("Generating training data:")
for i in tqdm(range(number_of_spectra)):
    m = gen_FID()
    full_FID = m.out_full_FID
    full_coupled_FID = m.out_coupled_FID
    full_coupled_T_FID = m.out_coupled_T_FID
    training_df = training_df.append({"Coupled" : list(full_coupled_FID), "Coupled+T" : list(full_coupled_T_FID), "FID" : list(full_FID)}, ignore_index=True)
print("Training data generated. " + str(int(len(training_df.index))) + " examples generated.")
print("")

training_df.to_csv("EXPERIMENT_TRAINING_SET_10K_5.csv", index=False)
