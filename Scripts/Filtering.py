import numpy as np
import cv2
import  matplotlib.pyplot as plt
import os

def make_kernel():
    """
    returns a kernel, of a gaborFilter
    """

    sigma = 3  # Large sigma on small features will fully miss the features.
    theta = 3 * np.pi / 4 # /4 shows horizontal 3/4 shows other horizontal. Try other contributions
    lamda = 1 * np.pi / 4  # 1/4 works best for angled.
    gamma = 0.75  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
    # Value of 1, spherical may not be ideal as it picks up features from other regions.
    phi = 0  # Phase offset. I leave it to 0.
    kernel = cv2.getGaborKernel((50, 50), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    return kernel



def make_kernels():
    """
    return multiple gaborFilters
    """
    filters = []
    lamda = 1 * np.pi / 4
    phi = 0
    gamma = 0.78
    for sigma in (3,5):
        for x in range(1,3):
            print(x)
            for y in np.arange(.5, 1., .24):
                kernel = cv2.getGaborKernel((50, 50), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                filters.append(kernel)
                print("sigma: "+str(sigma),"theta " + str(theta),"gamma "+str(gamma))

    return filters

