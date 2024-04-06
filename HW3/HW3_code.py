from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
    #Calculate the filter length(rounded up to the next odd integer)
    len = sigma * 6
    len = math.ceil(len)

    if(len%2==0):
        len=len+1 # if len is even, make it odd
    
    #generated an array of values of distance from center
    x = np.arange(len//2 * -1, len//2 + 1,1,dtype=np.float32)
    #compute gaussian value
    gaussian = np.exp(-1*x*x/(2*sigma*sigma),dtype=np.float32)
    #normalize the filter so that the sum of values is 1
    return gaussian / np.sum(gaussian)

def gauss2d(sigma):
    # implement
    #compute 1-dimension gaussian filter
    gaussian_1d = gauss1d(sigma)
    #cross product of 1-dimension gaussian filter
    gaussian_2d = np.outer(gaussian_1d,gaussian_1d)
    #normailze the filter so that the sum of values is 1
    return gaussian_2d / np.sum(gaussian_2d)

def convolve2d(array, filter):
    #assume that row size and column size of filter is same

    #make empty image
    result = np.zeros_like(array,dtype=np.float32)
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
            result[i,j] = np.sum(pad_array[i:i+filter_row_size,j:j+filter_row_size]*flip_filter,dtype=np.float32)

    return result

def gaussconvolve2d(array,sigma):
    # implement
    #make gaussian filter using sigma
    filter = gauss2d(sigma)
    #perform convolution
    return convolve2d(array,filter)

def reduce_noise(img):
    """ Return the gray scale gaussian filtered image with sigma=1.6
    Args:
        img: RGB image. Numpy array of shape (H, W, 3).
    Returns:
        res: gray scale gaussian filtered image (H, W).
    """
    #implement
    img = img.convert('L')
    res = np.asarray(img,dtype=np.float32)
    res = gaussconvolve2d(res,1.6)
    return res

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    #implement 
    s_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
    s_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float32)

    convolve_x = convolve2d(img,s_x)
    convolve_y = convolve2d(img,s_y)

    G = np.hypot(convolve_x,convolve_y)
    G = G/G.max()*255
    theta = np.arctan2(convolve_y, convolve_x,dtype=np.float32)

    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    deg = np.abs(np.rad2deg(theta))
    size = G.shape

    res = np.zeros_like(G,dtype=np.float32)
    
    # 22.5~67.5 : 45
    # 67.5~112.5 : 90
    # 112.5~157.5 : 135
    #else : 0
    nearMax=0
    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1):
            nearMax=0
            if 22.5<=deg[i,j]<67.5:
                nearMax = max(G[i-1,j-1],G[i+1,j+1])
            elif 67.5<=deg[i,j]<112.5:
                nearMax = max(G[i-1,j],G[i+1,j])
            elif 112.5<=deg[i,j]<157.5:
                nearMax = max(G[i+1,j-1],G[i-1,j+1])
            else:
                nearMax = max(G[i,j+1],G[i,j-1])

            if G[i,j]>=nearMax:
                res[i,j] = G[i,j]
    res = res/res.max() * 255
    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    #implement     

    max = np.max(img)
    min = np.min(img[img>0])
    
    diff = max-min
    t_high = min + diff * 0.15
    t_low = min + diff * 0.03

    size = img.shape

    res = np.zeros_like(img,dtype=np.float32)

    res = np.where(img>=t_low,80,0)
    res = np.where(img>=t_high,255,res)

    

    return res

def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    #implement 

    res = np.zeros_like(img,dtype=np.float32)

    size = img.shape
    visited = []
    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1):
            if img[i,j] == 255 and (i,j) not in visited:
                dfs(img,res,i,j,visited)

    return res

def main():
    RGB_img = Image.open('./iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    
    Image.fromarray(g.astype('uint8')).save('./iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('./iguana_sobel_theta.bmp', 'BMP')

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('./iguana_non_max_suppression.bmp', 'BMP')

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('./iguana_double_thresholding.bmp', 'BMP')

    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('./iguana_hysteresis.bmp', 'BMP')
    

if __name__ == '__main__':
    main()