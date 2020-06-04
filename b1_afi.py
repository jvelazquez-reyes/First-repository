##AFI (Actual Flip-Angle Imaging) Method

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual

alpha_nom = (60*np.pi/180);
flip_angles = np.arange(0,100,10);
T1 = 426;
TE = 2.8; #[ms]

TR1 = 30;
TR2 = 150;

#Longitudinal magnetization
Mz1 = 1000*((1 - np.exp(-TR2/T1) + (1 - np.exp(-TR1/T1))*np.exp(-TR2/T1)*np.cos(flip_angles*np.pi/180))/(1 - np.exp(-TR1/T1)*np.exp(-TR2/T1)*np.cos(flip_angles*np.pi/180)*np.cos(flip_angles*np.pi/180)));
Mz2 = 1000*((1 - np.exp(-TR1/T1) + (1 - np.exp(-TR2/T1))*np.exp(-TR1/T1)*np.cos(flip_angles*np.pi/180))/(1 - np.exp(-TR1/T1)*np.exp(-TR2/T1)*np.cos(flip_angles*np.pi/180)*np.cos(flip_angles*np.pi/180)));

S1 = Mz1*0.8**np.sin(flip_angles*np.pi/180);
S2 = Mz2*0.8**np.sin(flip_angles*np.pi/180);

#Normalizing at 20°
S1_norm = S1/S1[1];
S2_norm = S2/S1[1];

#Signal intensity vs angle, ranging from 10 to 100 degrees
#Parameters are later passed to the function varying TR and T1 separately
plt.plot(flip_angles, Mz1)
plt.plot(flip_angles, Mz2)
plt.xlabel('Flip angles '  r'($\theta$)')
plt.ylabel('Signal intensities')
plt.ylim(0,1000)
plt.show()


r = S2/S1;
plt.plot(flip_angles, r)
plt.xlabel('r '  r'($\theta$)')
plt.ylabel('Signal intensities')
plt.ylim(0.2,1)
plt.show()

actual_fa = np.arccos((r*TR2/TR1 - 1)/(TR2/TR1 - r))*180/np.pi;
plt.plot(flip_angles, actual_fa)
plt.xlabel('Nominal flip angle '  r'($\theta$)')
plt.ylabel('Actual flip angle ' r'($\theta$)')
plt.show()

#Gaussian noise
noise_r = np.random.normal(0,0.1,r.shape);
noise_S2 = S2*np.random.normal(0,10,S2.shape);
#noise_r = noise_S2/noise_S1;
noise_actual_fa = np.arccos((noise_r*TR2/TR1 - 1)/(TR2/TR1 - noise_r))*180/np.pi;
plt.plot(flip_angles, noise_actual_fa)
plt.xlabel('Nominal flip angle '  r'($\theta$)')
plt.ylabel('Actual flip angle ' r'($\theta$)')
plt.show()
