import cv2
import numpy as np

def to_img(matrix):
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255

def fourierTransformGray(image, normalize = True, output_form = 'log'): #TO USE WITH AI
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(imageGray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    if output_form == 'linear':
        output = (cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) # linear amplitude output
    elif output_form == 'log':
        output = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) # logarithmic amplitude output
    elif output_form == 'complex':
        output = dft_shift # linear complex output
    if normalize:
        output = to_img(output)
    return output

def unitCycleKernel(rad, size): #TO USE WITH AI
    rx, ry = size/2, size/2
    x, y = np.indices((size, size))
    return ((np.hypot(rx - x, ry - y) - rad) < 0.5).astype(int)

def reverseUnitCycleKernel(rad, size): #TO USE WITH AI
    rx, ry = size/2, size/2
    x, y = np.indices((size, size))
    return ((np.hypot(rx - x, ry - y) - rad) > 0.5).astype(int)

def gaussianKernel(matrixSize, sig=1.):#TO USE WITH AI
    ax = np.linspace(-(matrixSize - 1) / 2., (matrixSize - 1) / 2., matrixSize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def reverseGaussian(matrixSize, sig=1.,amplitude=0):#TO USE WITH AI
    gauss=gaussianKernel(matrixSize, sig)
    mx = np.amax(gauss)
    output = np.ones([matrixSize, matrixSize])*mx-gauss
    if(amplitude!=0):
        output = output*(amplitude/mx)
    return output
#
# ker = reverseUnitCycleKernel(20 * np.pi, 256)
# img = cv2.imread('lena.jpg')
# fft = fourierTransformGray(img)
# cv2.imwrite('dupa.jpg', ker * fft)