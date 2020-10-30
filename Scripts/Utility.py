
import numpy as np
import cv2
import  matplotlib.pyplot as plt
import os

root_dir = os.path.abspath("../")
class Utility2():
    def __init__(self):
        print("in init")

    def loadImages(self, path):

        images = []
        imagePaths = os.listdir(".."+path)
        for x in imagePaths:
            img = cv2.imread(root_dir + path + "/" + x)
            images.append(img)
        return images

    #def loadImages(self, path):
    #    transmitted = []
    #    channels = []
    #    imagePaths = os.listdir(".." + path)
#
    #    y = 0
    #    for x in imagePaths:
    #        img = cv2.imread(root_dir + path + "/" + x)
    #        y += 1
    #        if y is 5:
    #            y = 0
    #            transmitted.append(img)
    #        else:
    #            channels.append(img)
#
    #    return transmitted, channels

    def processImage(self, image):
        eqImages = []
        smoothImages = []

        filters, filterValues = self.generate_gabor_filters()

        for x in range(len(filters)):
            fimg = cv2.filter2D(image, cv2.CV_8UC3, filters[x])  # img_eq

            n = 13;  # where n*n is the size of filter
            smoothed_image = cv2.medianBlur(fimg, n)
            smoothImages.append(smoothed_image)
            mg_eq = self.histogram_equalization(smoothed_image)

            eqImages.append(mg_eq)
            # final_image= cv2.hconcat([final_image,fimg])

        final_image = self.h_concatenate_images(eqImages)
        return eqImages, smoothImages, final_image, filterValues

    def medianBlurImages(self, images, kernelSize=None):
        output = []
        n = 13
        if kernelSize is not None:
            n = kernelSize

        for x in images:
            out = cv2.medianBlur(x, n)
            output.append(out)

        return output

    def load_and_process_image(self, path):
        images = []
        smoothImages = []

        #load image
        img = cv2.imread(root_dir+path)
        images.append(img)
        # histogram equalization on image
        #img_eq = histogram_equalization(img)
        #final_image = cv2.hconcat([img,img_eq])
        #images.append(img_eq)
        # apply gabor filters on equalized image
        filters, filterValues = self.generate_gabor_filters()

        for x in range(len(filters)):
            fimg = cv2.filter2D(img,cv2.CV_8UC3,filters[x]) #img_eq

            n=13; #where n*n is the size of filter
            smoothed_image = cv2.medianBlur(fimg, n)
            smoothImages.append(smoothed_image)

            mg_eq = self.histogram_equalization(smoothed_image)
            images.append(mg_eq)
            #final_image= cv2.hconcat([final_image,fimg])

        final_image = self.h_concatenate_images(images)
        return images, smoothImages, final_image, filterValues

    def applyGaborOnImages(self, images):
        output = []
        filters, filterValues = self.generate_gabor_filters()

        for x in range(len(images)):
            for y in range(len(filters)):
                _img = cv2.filter2D([x], cv2.CV_8UC3, filters[y])  # img_eq
                output.append(_img)
        return output

    def histogram_equalization(self, img_in):
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
       #equ_b = cv2.equalizeHist(b)
       #equ_g = cv2.equalizeHist(g)
       #equ_r = cv2.equalizeHist(r)
       #equ = cv2.merge((equ_b, equ_g, equ_r))
        #print(equ)
        #cv2.imwrite('output_name.png', equ)
        return img_out
        #cv2.imshow(str(img),img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def histogramImages(self, images):
        output = []
        for x in range(len(images)):
            _img = self.histogram_equalization(images[x])
            output.append(_img)
        return output

    def generate_gabor_filters(self):

        k = (50,50)
        sigma = 3
        theta = 1*np.pi/4
        lamda = 1*np.pi/4  #1/4 works best for angled.
        #gamma = .78 #.5 and .78 yields results
        gammArr = [.1,0.25,0.5,1.0]
        phi = 0
        filters = []
        values = []
        for x in range(1,4,2):
            theta = x*np.pi/4
            for y in gammArr:
                gamma = y
                kernel = cv2.getGaborKernel(k, sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                #kernel = cv2.resize(kernel, (32,32))
                value = ("Sigma", sigma, "Theta", theta, "Gamma", gamma)
                values.append(value)
                filters.append(kernel)
                #print(sigma,theta, gamma)

        return filters, values
    def h_concatenate_images(self,imgArr):
        _img = imgArr[0]
        for x in range(len(imgArr)-1):
            _img = cv2.hconcat([_img,imgArr[x+1]])
        return _img
    def v_concatenate_images(self,imgArr):
        _img = imgArr[0]
        for x in range(len(imgArr)-1):
            _img = cv2.vconcat([_img,imgArr[x+1]])
        return _img
    def pairs_from_array(self,arr1,arr2):
        out=[]
        arr1 = arr1[1:]
        for x in range(len(arr1)):
            im1 = arr1[x]
            im2 = arr2[x]
            _img = cv2.hconcat([im1,im2])
            out.append(_img)
        return out
    def show_filters(self):
        filters = self.generate_gabor_filters()
        return self.h_concatenate_images(filters)

    def Test(self):
        print("HUH?")

    def showImages(self, images, titles=None, scale=None):
        if titles is not None:
            for x in range(len(titles)):
                if scale is not None:
                    cv2.imshow(titles[x], cv2.resize(images[x], scale))
                else:
                    cv2.imshow(titles[x], images[x])
        else:
            for x in range(len(images)):
                if scale is not None:
                    cv2.imshow(str(x), cv2.resize(images[x], scale))
                else:
                    cv2.imshow(str(x), images[x])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def writeImages(self, images, prefix=None, titles=None, scale=None, path=None):
        if path is not None:
            os.chdir(path)

        if prefix is None:
            prefix = ""

        if titles is not None:
            for x in range(len(titles)):
                if scale is not None:
                    cv2.imwrite(prefix + titles[x] + ".jpg", cv2.resize(images[x], scale))
                else:
                    cv2.imwrite(prefix + titles[x] + ".jpg", images[x])
        else:
            for x in range(len(images)):
                if scale is not None:
                    cv2.imwrite(prefix + str(x) + ".jpg", cv2.resize(images[x], scale))
                else:
                    cv2.imwrite(prefix + str(x) + ".jpg", images[x])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mergeImages(self, images, index1, index2, ratio):
        output = []
        for x in images:
            out = cv2.addWeighted(x[index1], ratio, x[index2], ratio, 0)
            output.append(out)
        return output

    def pairMerge(self, images):
        output = images[0]
        for x in range(len(images) - 1):
            output = cv2.addWeighted(output, 0.5, images[x+1], 0.5, 0)
        return output


    def morphologyImages(self, images, kernel=None, iterations=None):
        i = 15
        if iterations is not None:
            i = iterations

        k = np.ones((3, 3), np.uint8)
        if kernel is not None:
            k = kernel

        output = []

        for x in images:
            out = cv2.morphologyEx(x, cv2.MORPH_OPEN, k, iterations=i)
            output.append(out)

        return output

    def dilateImages(self, images, kernel=None, iterations=None):
        i = 10
        if iterations is not None:
            i = iterations

        k = np.ones((3, 3), np.uint8)
        if kernel is not None:
            k = kernel

        output = []

        for x in images:
            out = cv2.dilate(x, k, iterations=i)
            output.append(out)

        return output

    def grayscaleImages(self, images):
        output = []

        for x in images:
            out = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            output.append(out)

        return output

    def thresholdImages(self, images, minval=0, maxval=255, invert=None):
        thresh = []
        output = []
        for x in images:
            ret, thresh = cv2.threshold(x, minval, maxval, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            if invert is not None:
                thresh = 255 - thresh
            output.append(thresh)

        return output

    def waterShedImages(self, images, processed=True):
        output = []
        markersArr = []
        thresholded = images
        if processed is False:
            grayscaled = self.grayscaleImages(images)
            thresholded = self.thresholdImages(grayscaled)

        openings = self.morphologyImages(thresholded)
        #ut.showImages(thresholded, scale=(500, 500))

        sure_bgs = self.dilateImages(openings)

        for x in range(len(images)):
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(openings[x], cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
            ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bgs[x], sure_fg)

            ret, markers = cv2.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            markers = cv2.watershed(images[x], markers)
            markersArr.append(markers)
            images[x][markers == -1] = [255, 0, 0]
            output.append(images[x])

        return output, markersArr

    def kmeansOutImages(self, images, kVal):
        output = []
        for x in images:
            Z = x.reshape((-1,3))
            # convert to np.float32
            Z = np.float32(Z)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = kVal
            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape(x.shape)
            output.append(res2)
        return output

