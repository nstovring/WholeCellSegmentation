import cv2
import numpy as np

from PIL import Image, ImageEnhance
import numpy as np
from PIL import ImageFilter
import colorsys
import os
from skimage.filters import gabor, gaussian
from IPython.display import display
from matplotlib.pyplot import imshow
from pywt import dwt2
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Gabor():
    def gabor(self,img_in):
        image = Image.open(img_in).convert('RGB')
        image_size = image.size
        print(image_size)
        pixels = np.asarray(image, dtype="int32")
        energy_density = Gabor.get_energy_density(pixels)
        # get fixed bandwidth using energy density
        bandwidth = abs(0.4 * energy_density - 0.5)

        magnitude_dict = {}
        for theta in np.arange(0, np.pi, np.pi / 6):
            for freq in np.array([1.4142135623730951, 2.414213562373095, 2.8284271247461903, 3.414213562373095]):
                filt_real, filt_imag = gabor(image, frequency=freq, bandwidth=bandwidth, theta=theta)
                # get magnitude response
                magnitude = Gabor.get_magnitude([filt_real, filt_imag])
                ''' uncomment the lines below to visualize each magnitude response '''
                # im = Image.fromarray(magnitude.reshape(image_size)).convert('L')
                # display(im)
                magnitude_dict[(theta, freq)] = magnitude.reshape(image.size)
        # apply gaussian smoothing
        gabor_mag = []
        for key, values in magnitude_dict.items():
            # the value of sigma is chosen to be half of the applied frequency
            sigma = 0.5 * key[1]
            smoothed = gaussian(values, sigma=sigma)
            gabor_mag.append(smoothed)
        gabor_mag = np.array(gabor_mag)

        # reshape so that we can apply PCA
        value = gabor_mag.reshape((-1, image_size[0] * image_size[1]))

        # get dimensionally reduced image
        pcaed = Gabor.apply_pca(value.T).astype(np.uint8)
        result = pcaed.reshape((image_size[0], image_size[1]))
        result_im = Image.fromarray(result, mode='L')
        display(result_im)

    def get_image_energy(pixels):
        """
        :param pixels: image array
        :return: Energy content of the image
        """
        _, (cH, cV, cD) = dwt2(pixels.T, 'db1')
        energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / pixels.size
        return energy

    def get_energy_density(pixels):
        """
        :param pixels: image array
        :param size: size of the image
        :return: Energy density of the image based on its size
        """
        energy = Gabor.get_image_energy(pixels)
        energy_density = energy / (pixels.shape[0] * pixels.shape[1])
        return round(energy_density * 100, 5)  # multiplying by 100 because the values are very small

    def get_magnitude(response):
        """
        :param response: original gabor response in the form: [real_part, imag_part]
        :return: the magnitude response for the input gabor response
        """
        magnitude = np.array([np.sqrt(response[0][i][j] ** 2 + response[1][i][j] ** 2)
                              for i in range(len(response[0])) for j in range(len(response[0][i]))])
        return magnitude

    def apply_pca(array):
        """
        :param array: array of shape pXd
        :return: reduced and transformed array of shape dX1
        """
        # apply dimensionality reduction to the input array
        standardized_data = StandardScaler().fit_transform(array)
        pca = PCA(n_components=1)
        pca.fit(standardized_data)
        transformed_data = pca.transform(standardized_data)
        return transformed_data

class Utilty():
    def histogram_equalization(img_in):
    # segregate color streams
        #image = cv2.imread(img_in)
        #b,g, r = img_in
        b,g,r =  cv2.split(img_in)
        h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
        cdf_b = np.cumsum(h_b)
        cdf_g = np.cumsum(h_g)
        cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
        cdf_m_b = np.ma.masked_equal(cdf_b,0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')

        cdf_m_g = np.ma.masked_equal(cdf_g,0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
        cdf_m_r = np.ma.masked_equal(cdf_r,0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
    # merge the images in the three channels
        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]

        img_out = cv2.merge((img_b, img_g, img_r))
    # validation
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))
        #print(equ)
        #cv2.imwrite('output_name.png', equ)
        return img_out