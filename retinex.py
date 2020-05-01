import cv2
import numpy as np

img = cv2.imread('arg.jpg')
I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

Ir = I[:, :, 0] #R
Ig = I[:, :, 1] #G
Ib = I[:, :, 2] #B

#set the required parameters
G = 192
b = -30 
alpha = 125 
beta = 46
Ir_double=float(Ir) 
Ig_double=float(Ig) 
Ib_double=float(Ib)

#Set the Gaussian parameter
sigma_1=15   #Three Gaussian Kernels
sigma_2=80 
sigma_3=250 

x, y = np.meshgrid(((-(Ir.shape[1]-1)/2) : (Ir.shape[1]/2)),((-(Ir.shape[0]-1)/2) : (Ir.shape[0]/2)))
gauss_1 = np.exp(-(np.power(x,2)+np.power(y,2))/(2*sigma_1*sigma_1))
Gauss_1= gauss_1 / np.sum(np.ravel(gauss_1))
gauss_2 = np.exp(-(np.power(x,2)+np.power(y,2))/(2*sigma_2*sigma_2))
Gauss_2= gauss_2 / np.sum(np.ravel(gauss_2))
gauss_3 = np.exp(-(np.power(x,2)+np.power(y,2))/(2*sigma_3*sigma_3))
Gauss_3= gauss_3 / np.sum(np.ravel(gauss_3))

#Operates on R component
#MSR Section
Ir_log= np.log(Ir_double+1);  #Converts an image to a logarithm field 
f_Ir= np.fft.fft2(Ir_double);  #The image is Fourier transformed and converted to the frequency domain

#sigma = 15 processing results
fgauss = np.fft.fft2(Gauss_1, Ir.shape[0], Ir.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rr = np.fft.ifft2(np.multiply(fgauss, f_Ir))
min1 = np.min(Rr)
Rr_log= np.log(Rr - min1+1)
Rr1=Ir_log-Rr_log

#sigma = 80
fgauss = np.fft.fft2(Gauss_2, Ir.shape[0], Ir.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rr = np.fft.ifft2(np.multiply(fgauss, f_Ir))
min1 = np.min(Rr)
Rr_log= np.log(Rr - min1+1)
Rr2=Ir_log-Rr_log

#sigma = 250
fgauss = np.fft.fft2(Gauss_3, Ir.shape[0], Ir.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rr = np.fft.ifft2(np.multiply(fgauss, f_Ir))
min1 = np.min(Rr)
Rr_log= np.log(Rr - min1+1)
Rr3=Ir_log-Rr_log

Rr=0.33*Rr1+0.34*Rr2+0.33*Rr3 #Weighted summation 
MSR1 = Rr
SSR1 = Rr2
#calculate CR
CRr = beta*(np.log(alpha*Ir_double+1)-np.log(Ir_double+Ig_double+Ib_double+1));

#SSR
min1 = np.min(SSR1) 
max1 = np.max(SSR1)
SSR1 = np.uint8(255*(SSR1-min1)/(max1-min1))

#MSR
min1 = np.min(MSR1) 
max1 = np.max(MSR1) 
MSR1 = np.uint8(255*(MSR1-min1)/(max1-min1));

#MSRCR
Rr = G*(np.multiply(CRr,(Rr+b))); 
min1 = np.min(Rr)); 
max1 = np.max(Rr)); 
Rr_final = np.uint8(255*(Rr-min1)/(max1-min1)); 

#Operates on G component
#MSR Section
Ig_log= np.log(Ig_double+1);  #Converts an image to a logarithm field 
f_Ig= np.fft.fft2(Ig_double);  #The image is Fourier transformed and converted to the frequency domain

#sigma = 15 processing results
fgauss = np.fft.fft2(Gauss_1, Ig.shape[0], Ig.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rg = np.fft.ifft2(np.multiply(fgauss, f_Ig))
min2 = np.min(Rg)
Rg_log= np.log(Rg - min2+1)
Rg1=Ig_log-Rg_log

#sigma = 80
fgauss = np.fft.fft2(Gauss_2, Ig.shape[0], Ig.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rg = np.fft.ifft2(np.multiply(fgauss, f_Ig))
min2 = np.min(Rg)
Rg_log= np.log(Rg - min2+1)
Rg2=Ig_log-Rg_log

#sigma = 250
fgauss = np.fft.fft2(Gauss_3, Ig.shape[0], Ig.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rg = np.fft.ifft2(np.multiply(fgauss, f_Ig))
min2 = np.min(Rg)
Rg_log= np.log(Rg - min2+1)
Rg3=Ig_log-Rg_log

Rg=0.33*Rg1+0.34*Rg2+0.33*Rg3 #Weighted summation 
MSR2 = Rg
SSR2 = Rg2
#calculate CR
CRg = beta*(np.log(alpha*Ig_double+1)-np.log(Ir_double+Ig_double+Ib_double+1));

#SSR
min2 = np.min(SSR2) 
max2 = np.max(SSR2)
SSR2 = np.uint8(255*(SSR2-min2)/(max2-min2))

#MSR
min2 = np.min(MSR2) 
max2 = np.max(MSR2) 
MSR2 = np.uint8(255*(MSR2-min2)/(max2-min2));

#MSRCR
Rg = G*(np.multiply(CRg,(Rg+b))); 
min2 = np.min(Rg)); 
max2 = np.max(Rg)); 
Rg_final = np.uint8(255*(Rg-min2)/(max2-min2)); 

#Operates on B component
#MSR Section
Ib_log= np.log(Ib_double+1);  #Converts an image to a logarithm field 
f_Ib= np.fft.fft2(Ib_double);  #The image is Fourier transformed and converted to the frequency domain

#sigma = 15 processing results
fgauss = np.fft.fft2(Gauss_1, Ib.shape[0], Ib.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rb = np.fft.ifft2(np.multiply(fgauss, f_Ib))
min3 = np.min(Rb)
Rb_log= np.log(Rb - min3+1)
Rb1=Ib_log-Rb_log

#sigma = 80
fgauss = np.fft.fft2(Gauss_2, Ib.shape[0], Ib.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rb = np.fft.ifft2(np.multiply(fgauss, f_Ib))
min3 = np.min(Rb)
Rb_log= np.log(Rb - min3+1)
Rb2=Ib_log-Rb_log

#sigma = 250
fgauss = np.fft.fft2(Gauss_3, Ib.shape[0], Ib.shape[1])
fgauss = np.fft.fftshift(fgauss)
Rb = np.fft.ifft2(np.multiply(fgauss, f_Ib))
min3 = np.min(Rb)
Rb_log= np.log(Rb - min3+1)
Rb3=Ib_log-Rb_log

Rb=0.33*Rb1+0.34*Rb2+0.33*Rb3 #Weighted summation 
MSR3 = Rb
SSR3 = Rb2
#calculate CR
CRb = beta*(np.log(alpha*Ib_double+1)-np.log(Ir_double+Ig_double+Ib_double+1));

#SSR
min3 = np.min(SSR3) 
max3 = np.max(SSR3)
SSR3 = np.uint8(255*(SSR3-min3)/(max3-min3))

#MSR
min3 = np.min(MSR3) 
max3 = np.max(MSR3) 
MSR3 = np.uint8(255*(MSR3-min3)/(max3-min3));

#MSRCR
Rb = G*(np.multiply(CRb,(Rb+b))); 
min3 = np.min(Rb)); 
max3 = np.max(Rb)); 
Rb_final = np.uint8(255*(Rb-min3)/(max3-min3)); 

#MSRCP