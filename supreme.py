import numpy as np
from PIL import Image

def setfreq(px_size_m, img_size_m):
    # setfreq() is a function theat does not accept any inputs, rather it sets the frequency scale of the fourier
    #transformed image allowing the user to make comparasions between
    #the frequencies of fourier transformed images 
    
    import numpy as np
    
    dx_m = px_size_m  #pixel size in meter
    Dx_m = img_size_m      #image full size
    x_m = np.linspace(-Dx_m/2, Dx_m/2, Dx_m/dx_m + 1)
    real_axis = x_m
    fs = 1/(real_axis[1]-real_axis[0])
    Nfft=len(real_axis)
            
    df = fs/Nfft
    f_cpm = np.linspace(0,(fs-df),Nfft) - (fs-np.mod(Nfft,2)*df)/2
    df_cpm = f_cpm[1] - f_cpm[0]  #pixel size in meter
    Df_cpm = f_cpm[-1] - f_cpm[0]      #image full size

    variables = [dx_m, Dx_m, f_cpm, px_size_m, x_m, Nfft, df_cpm, Df_cpm]
    return variables

def spkim(path, image_num, plot = True):
    '''
    spkim(path, image_num) This function takes image path and speckle number as an input argouments and plots the image 
    on pixel-sized axes. Also, the function will pair each "spk_num" with the corresponding image in a dictionary "im_dic"
     Note: the "path" must be typed in qoutation marks, and "image_num" must be an intger or a float. The function uses
    image number to lable each speckle image correctly in case of the function is used to process multiple images.
    '''
    from PIL import Image 
    import matplotlib.pyplot as plt
    
    #(dx_m, Dx_m, f_cpm, px_size_m, x_m, Nfft, df_cpm, Df_cpm) = sp.setfreq(1*10**-6, 1e-3)
    dx_m = 1e-6
    Dx_m = 1e-3
    #FIX ME!!!!!!!!!!
    
    y = Image.open(path , "r" ) 
    im_dic = {('spk_'+ str(image_num)): y} # This line pairs the "spk_num" to imported image and return it in dictionary
    
    
    if plot:
        extent = (-Dx_m/2 * 1e6, +Dx_m/2 * 1e6, -Dx_m/2 * 1e6, +Dx_m/2 * 1e6) 
        plt.imshow(y, extent = extent, cmap = 'inferno')
        plt.xlabel('Horizontal Position [um]')
        plt.ylabel('Vertical Positon [um]')
        plt.title('Speckel ' + str(image_num))
        plt.colorbar()
        plt.show()
    return im_dic

def ftim(spk_num):
    # ftim(spk_num) is a function that: First, it converts imput image into a nympy array. Second, it preforms
    # a fourier transformation. Lastly, it plots the transformation on frequncy scale.
    # Note: Input of this function should be in the form of an image file that has been already imported. If "spkim()" function
    # was used prior to "ftim()", then input can be optained using the keys of the global dictionary im_dic. 
    #EX: ftim(im_dic['spk_1'])
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    data_1 = np.asarray(spk_num) # Converts the image into a numpy array
    Y1 = np.fft.fftshift(np.fft.fft2(data_1))  # it rearranges the fourier transform to make it symetrical about zero

    df_cpm = f_cpm[1] - f_cpm[0]  #pixel size in meter
    Df_cpm = f_cpm[-1] - f_cpm[0]      #image full size
    extent = (-Df_cpm/2 * 1e-6, +Df_cpm/2 * 1e-6, -Df_cpm/2 * 1e-6, +Df_cpm/2 * 1e-6) 

    plt.imshow(np.abs(Y1)**0.1, extent = extent)
    plt.xlabel('Freq[/um]')
    plt.ylabel('Freq[/um]')
    plt.title('PSD '+ list(im_dic.keys())[list(im_dic.values()).index(spk_num)] + ' Magnitude') # adds the dictionary(spk_num)
                                                                                                    #key to plot title
    plt.show()
    return Y1, data_1

def read(filepath)
    return np.asarray(Image.open(filepath))

def display(x,y,img):
    Dx = x[-1]-x[0]
    Dy = y[-1]-y[0]
    extent = (-Dx/2, +D/2 * 1e6, -Dy/2 * 1e6, +Dy/2 * 1e6)
    plt.imshow(img,extent=extent)
    plt.show()

def ft(img):
    return np.fft.fftshift(np.fft.fft2(img))

def fs(real_axis):
    fs = 1/(real_axis[1]-real_axis[0]) # Setting the unit of the frequncy scale to 1/um (when i do the calculations i get 1/mm)
    Nfft=len(real_axis)
            
    df = fs/Nfft
    f_cpm = np.linspace(0,(fs-df),Nfft) - (fs-np.mod(Nfft,2)*df)/2 # Do not understand the calculations here.
    return f_cpm

def circsum(IMG):
    return np.sum(warp_polar(np.abs(IMG)**2), 0)


