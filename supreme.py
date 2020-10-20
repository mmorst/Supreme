import numpy as np
from skimage.transform import warp, warp_polar
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

def read(filepath):
    return np.asarray(Image.open(filepath))

def display(x,y,img):
    Dx = x[-1]-x[0]
    Dy = y[-1]-y[0]
    extent = (-Dx/2, +D/2 * 1e6, -Dy/2 * 1e6, +Dy/2 * 1e6)
    plt.imshow(img,extent=extent)
    plt.show()

def circsum(IMG):
    return np.sum(warp_polar(np.abs(IMG)**2), 0)

def ft(t):
    return np.fft.fftshift( np.fft.fft2(np.fft.ifftshift(t)))

def ift(t):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(t)))

def fs(t):
    # works with Matlab, breaks with python
    #return (np.arange(0,1/(t[1]-t[0]),1/((t[1]-t[0])*len(t)))) - (1/(t[1]-t[0])-np.mod(len(t),2)*1/((t[1]-t[0])*len(t)))/2
    
    # from https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
    N = len(t)
    df_cpm = 1/(t[-1]-t[0])
    f_cpm = np.fft.fftshift(np.array([df_cpm*n if n<N/2 else df_cpm*(n-N) for n in range(N)]))
    return f_cpm

def propTF(E_in,L_m,lambda_m,z_m):
    #get input field array size
    (Nx, Ny)=np.shape(E_in); 
    dx=L_m/Nx; #sample interval

    #(dx<lambda.*z/L)


    fx = fs(np.arange(Nx)*dx);
    if Ny>2:
        fy = fs(np.arange(Ny)*dx);
    else:
        fy = 0;

    [FX,FY]= np.meshgrid(fx,fy);

    H=np.exp(-1j*np.pi*lambda_m*z_m*(FX**2+FY**2))

    E_out = ft(ft(E_in)*H);
    
    return E_out

def gaussian(x_px, mean_px, fwhm_px):
    '''
    GAUSSIAN Returns a 1D gaussian
    gaussian(x_px, mean_px, fwhm_px)
    :param x_px: numpy array
    :param mean_px: double
    :param fwhm_px: double
    :return: numpy array
    '''
    sigma_x = fwhm_px/(2*np.sqrt(2*np.log(2)));
    return np.exp(-((x_px-mean_px)/(np.sqrt(2)*sigma_x))**2)
        
def generate_speckle(dx_m, Dx_m, z_m, fhwm_m=500e-6, lambda_m=1e-6, fc_cpm =1e6, wfe_w=1/20):
    '''GENERATE_SPECKLE
    generate_speckle(dx_m, Dx_m, z_m, lambda_m=1e-6, fc_cpm =1e6, wfe_w=1/20)
    :param dx_m: double
    :param Dx_m: double
    :param z_m: double
    :param lambda_m: double
    :param fc_cpm: double
    :param wfe_w: double
    :return: numpy array
    '''

    # define spatial axis
    x_m = np.linspace(-Dx_m/2,Dx_m/2,int(np.ceil(Dx_m/dx_m)))
    N = len(x_m);

    (X_m,Y_m) = np.meshgrid(x_m,x_m);
    # define frequency axis
    f_cpm = fs(x_m);
    (Fx, Fy) = np.meshgrid(f_cpm,f_cpm);

    # frequency filter
    GFILT = gaussian(Fx, 0, fc_cpm)*gaussian(Fy, 0, fc_cpm);

    # generation of speckle and propagations
    noise = np.random.randn(N,N)
    noise_filt = np.abs(ift(ft(noise)*GFILT));
    noise_filt = noise_filt/np.std(noise_filt);

    # generate laser beam
    E0 = gaussian(X_m, 0, fhwm_m)*gaussian(Y_m, 0, fhwm_m);
    # add speckle to beam
    E1 = E0*np.exp(1j*2*np.pi*noise_filt*wfe_w);

    # propagation
    E2 = propTF(E1,Dx_m,lambda_m,z_m);
    I = np.abs(E2)**2
    return I
 
def azimuthal_avg(data_2d):
    '''
    AZIMUTHAL_AVG Azimuthal average of a 2D square image
    azimuthal_avg(data_2d)
    :param data_2d: square 2D numpy array
    :return: numpy 1D array
    '''
    Np = int(np.floor((len(data_2d)+1)/2))
    azimuthal_avg = np.sum(warp_polar(data_2d), 0)
    return azimuthal_avg[0:(Np-1)]

#imshow(x_m, I, zoom=1.0)