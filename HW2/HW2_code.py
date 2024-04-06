from PIL import Image 
import numpy as np 
import math

#Part 1

#1-1
def boxfilter(n):
    assert n%2!=0, "Dimension must be odd"
    return np.ones((n,n)) / (n*n)

#1-2
def gauss1d(sigma):
    #Calculate the filter length(rounded up to the next odd integer)
    len = sigma * 6
    len = math.ceil(len)

    if(len%2==0):
        len=len+1 # if len is even, make it odd
    
    #generated an array of values of distance from center
    x = np.arange(len//2 * -1, len//2 + 1,1)
    #compute gaussian value
    gaussian = np.exp(-1*x*x/(2*sigma*sigma))
    #normalize the filter so that the sum of values is 1
    return gaussian / np.sum(gaussian)
    

#1-3

def gauss2d(sigma):
    #compute 1-dimension gaussian filter
    gaussian_1d = gauss1d(sigma)
    #cross product of 1-dimension gaussian filter
    gaussian_2d = np.outer(gaussian_1d,gaussian_1d)
    #normailze the filter so that the sum of values is 1
    return gaussian_2d / np.sum(gaussian_2d)

#1-4
#a
def convolve2d(array, filter):
    #assume that row size and column size of filter is same

    #make empty image
    result = np.zeros_like(array)
    #get filter size(for calculating padding size)
    filter_row_size = filter.shape[0]
    #calculating padding size
    pad_size = int((filter_row_size-1)/2)
    #applying padding to array(image)
    pad_array = np.pad(array, ((pad_size,pad_size),(pad_size,pad_size)), 'constant', constant_values=0)

    #flip filter based on the axis=0 and axis=1(for convolution)
    flip_filter = np.flip(filter)
    
    #perform convolution
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            result[i,j] = np.sum(pad_array[i:i+filter_row_size,j:j+filter_row_size]*flip_filter)

    return result
#b
def gaussconvolve2d(array,sigma):
    #make gaussian filter using sigma
    filter = gauss2d(sigma)
    #perform convolution
    return convolve2d(array,filter)

#c

#open image
im = Image.open('images/3b_tiger.bmp')
#make image gray scale
im_gray = im.convert('L')
#convert image to array
im_array = np.asarray(im_gray)

#apply gaussian filter(sigma=3)
convolved = gaussconvolve2d(im_array,3)
# clamping the pixels on (0,255)
convolved = np.clip(convolved, 0, 255, out=convolved)
convolved = convolved.astype(np.uint8)
#d
# show original image, show filtered image, and save
im.show()
convolved_image = Image.fromarray(convolved)
convolved_image.show()
convolved_image.save("part1_result.bmp","bmp")





#part2

#apply gauss filter to RGB image(RGB array)
def gaussconvolved2dRGB(array,sigma):
    #numpy array that store the result image array
    low_frequency_result = np.zeros_like(array)
    # filter each of the three color channels (RGB) separately
    for i in range(3):
        low_frequency_result[:,:,i] = gaussconvolve2d(array[:,:,i],sigma)
        low_frequency_result[:,:,i] = np.clip(low_frequency_result[:,:,i], 0, 255)
    return low_frequency_result

#2-1
#open image(for making low frequency image)
im2 = Image.open('images/3b_tiger.bmp')

#make array from image
im2_array = np.asarray(im2)

low_frequency_result = gaussconvolved2dRGB(im2_array,4)

# convert result array to image, show and save
low_frequency_img = Image.fromarray(low_frequency_result)
low_frequency_img.show()
low_frequency_img.save("part2-1_result.bmp","bmp")



#2-2
#open image(for making high frequency image)
im3 = Image.open('images/3a_lion.bmp')

#make array from image
im3_array = np.asarray(im3)

#for making high frequency image, we should make low frequency image
low_frequency = gaussconvolved2dRGB(im3_array,4)

#make array that store high frequency image
high_frequency_result = np.asarray(im3_array)

#high frequency image is original image - low frequency image
high_frequency_result = high_frequency_result - low_frequency

# convert result array to image, show and store
#values of above high_frequency array are zero-means negative values. so we should add 128 for visualizing
high_frequency_img = Image.fromarray(high_frequency_result + 128)
high_frequency_img.show()
high_frequency_img.save("part2-2_result.bmp","bmp")

#2-3
#making hybrid image. hybrid image is low frequency image + high frequency image
hybrid_array = low_frequency_result + high_frequency_result
hybrid_array =  np.clip(hybrid_array, 0, 255)
hybrid_image = Image.fromarray(hybrid_array)
hybrid_image.show()
hybrid_image.save("part2-3_result.bmp","bmp")


