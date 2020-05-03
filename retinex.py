from __future__ import division
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
#Ir_double=float(Ir) 
Ir_double = np.asfarray(Ir, float)
Ig_double = np.asfarray(Ig, float)
Ib_double = np.asfarray(Ib, float)
#Ig_double=float(Ig) 
#Ib_double=float(Ib)

#Set the Gaussian parameter
sigma_1=15   #Three Gaussian Kernels
sigma_2=80 
sigma_3=250 

X = np.arange(-(Ir.shape[1]-1)/2 , (Ir.shape[1]/2))
Y = np.arange(-(Ir.shape[0]-1)/2 , (Ir.shape[0]/2))
x, y = np.meshgrid(X, Y)
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
fgauss = np.fft.fft2(Gauss_1, (Ir.shape[0], Ir.shape[1]))
fgauss = np.fft.fftshift(fgauss)
Rr = np.fft.ifft2(np.multiply(fgauss, f_Ir))
min1 = np.min(Rr)
Rr_log= np.log(Rr - min1+1)
Rr1=Ir_log-Rr_log

#sigma = 80
fgauss = np.fft.fft2(Gauss_2, (Ir.shape[0], Ir.shape[1]))
fgauss = np.fft.fftshift(fgauss)
Rr = np.fft.ifft2(np.multiply(fgauss, f_Ir))
min1 = np.min(Rr)
Rr_log= np.log(Rr - min1+1)
Rr2=Ir_log-Rr_log

#sigma = 250
fgauss = np.fft.fft2(Gauss_3, (Ir.shape[0], Ir.shape[1]))
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
MSR1 = np.uint8(255*(MSR1-min1)/(max1-min1))

#MSRCR
Rr = G*(np.multiply(CRr,(Rr+b))) 
min1 = np.min(Rr)
max1 = np.max(Rr)
Rr_final = np.uint8(255*(Rr-min1)/(max1-min1))

#Operates on G component
#MSR Section
Ig_log= np.log(Ig_double+1);  #Converts an image to a logarithm field 
f_Ig= np.fft.fft2(Ig_double);  #The image is Fourier transformed and converted to the frequency domain

#sigma = 15 processing results
fgauss = np.fft.fft2(Gauss_1, (Ig.shape[0], Ig.shape[1]))
fgauss = np.fft.fftshift(fgauss)
Rg = np.fft.ifft2(np.multiply(fgauss, f_Ig))
min2 = np.min(Rg)
Rg_log= np.log(Rg - min2+1)
Rg1=Ig_log-Rg_log

#sigma = 80
fgauss = np.fft.fft2(Gauss_2, (Ig.shape[0], Ig.shape[1]))
fgauss = np.fft.fftshift(fgauss)
Rg = np.fft.ifft2(np.multiply(fgauss, f_Ig))
min2 = np.min(Rg)
Rg_log= np.log(Rg - min2+1)
Rg2=Ig_log-Rg_log

#sigma = 250
fgauss = np.fft.fft2(Gauss_3, (Ig.shape[0], Ig.shape[1]))
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
MSR2 = np.uint8(255*(MSR2-min2)/(max2-min2))

#MSRCR
Rg = G*(np.multiply(CRg,(Rg+b)))
min2 = np.min(Rg)
max2 = np.max(Rg)
Rg_final = np.uint8(255*(Rg-min2)/(max2-min2))

#Operates on B component
#MSR Section
Ib_log= np.log(Ib_double+1);  #Converts an image to a logarithm field 
f_Ib= np.fft.fft2(Ib_double);  #The image is Fourier transformed and converted to the frequency domain

#sigma = 15 processing results
fgauss = np.fft.fft2(Gauss_1, (Ib.shape[0], Ib.shape[1]))
fgauss = np.fft.fftshift(fgauss)
Rb = np.fft.ifft2(np.multiply(fgauss, f_Ib))
min3 = np.min(Rb)
Rb_log= np.log(Rb - min3+1)
Rb1=Ib_log-Rb_log

#sigma = 80
fgauss = np.fft.fft2(Gauss_2, (Ib.shape[0], Ib.shape[1]))
fgauss = np.fft.fftshift(fgauss)
Rb = np.fft.ifft2(np.multiply(fgauss, f_Ib))
min3 = np.min(Rb)
Rb_log= np.log(Rb - min3+1)
Rb2=Ib_log-Rb_log

#sigma = 250
fgauss = np.fft.fft2(Gauss_3, (Ib.shape[0], Ib.shape[1]))
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
MSR3 = np.uint8(255*(MSR3-min3)/(max3-min3))

#MSRCR
Rb = G*(np.multiply(CRb,(Rb+b)))
min3 = np.min(Rb)
max3 = np.max(Rb)
Rb_final = np.uint8(255*(Rb-min3)/(max3-min3))

#MSRCP
Int = (Ir_double + Ig_double + Ib_double) / 3.0
#print(Int)
Int_log = np.log(Int+1);  #Converts an image to a logarithm field
f_Int=np.fft.fft2(Int_log) #The image is Fourier transformed and converted to the frequency domain 

#sigma = 15 processing results
fgauss=np.fft.fft2(Gauss_1, (Int.shape[0],Int.shape[1])) 
fgauss=np.fft.fftshift(fgauss)  #Move the center of the frequency domain to zero 
RInt=np.fft.ifft2(np.multiply(fgauss,f_Int)) #After convoluting, transform back into the airspace 
min1=np.min(RInt)
RInt_log= RInt - min1+1
RInt1=Int_log-RInt_log

#sigma=80 
fgauss=np.fft.fft2(Gauss_2, (Int.shape[0],Int.shape[1]))
fgauss=np.fft.fftshift(fgauss)  #Move the center of the frequency domain to zero 
RInt=np.fft.ifft2(np.multiply(fgauss,f_Int)) #After convoluting, transform back into the airspace 
min1=np.min(RInt) 
RInt_log= RInt - min1+1; 
RInt2=Int_log-RInt_log;  

 #sigma=250 
fgauss=np.fft.fft2(Gauss_3, (Int.shape[0],Int.shape[1]))
fgauss=np.fft.fftshift(fgauss)  #Move the center of the frequency domain to zero 
RInt=np.fft.ifft2(np.multiply(fgauss,f_Int)) #After convoluting, transform back into the airspace 
min1=np.min(RInt)
RInt_log= RInt - min1+1; 
RInt3=Int_log-RInt_log; 

RInt=0.33*RInt1+0.34*RInt2+0.33*RInt3;   #Weighted summation

minInt = np.min(RInt)
maxInt = np.max(RInt)
Int1 = np.uint8(255*(RInt-minInt)/(maxInt-minInt))

MSRCPr = np.zeros((I.shape[0], I.shape[1]))
MSRCPg = np.zeros((I.shape[0], I.shape[1]))
MSRCPb = np.zeros((I.shape[0], I.shape[1]))

for ii in range(I.shape[0]):
    for jj in range(I.shape[1]):
        C = max(Ig_double[ii][jj], Ib_double[ii][jj])
        B = max(Ir_double[ii][jj], C)
        A = min(255.0 / B, Int1[ii][jj] / Int[ii][jj])
        MSRCPr[ii][jj] = A * Ir_double[ii][jj]
        MSRCPg[ii][jj] = A * Ig_double[ii][jj]
        MSRCPb[ii][jj] = A * Ib_double[ii][jj]

minInt = np.min(MSRCPr) 
maxInt = np.max(MSRCPr) 
MSRCPr = np.uint8(255*(MSRCPr-minInt)/(maxInt-minInt))

minInt = np.min(MSRCPg)
maxInt = np.max(MSRCPg) 
MSRCPg = np.uint8(255*(MSRCPg-minInt)/(maxInt-minInt))

minInt = np.min(MSRCPb)
maxInt = np.max(MSRCPb) 
MSRCPb = np.uint8(255*(MSRCPb-minInt)/(maxInt-minInt))

ssr = np.dstack((SSR1,SSR2,SSR3))
msr = np.dstack((MSR1,MSR2,MSR3))
msrcr = np.dstack((Rr_final,Rg_final,Rb_final)) #Combine the three-channel image 
MSRCP = np.dstack((MSRCPr, MSRCPg, MSRCPb))

cv2.imwrite( 'Original.jpg', I )
cv2.imwrite( 'SSR.jpg', ssr )
cv2.imwrite( 'MSR.jpg', msr )
cv2.imwrite( 'MSRCR.jpg', msrcr )
cv2.imwrite( 'MSRCP.jpg', MSRCP )
cv2.waitKey(0) 