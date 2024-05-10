import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation(-PI to PI)
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START

    largest_set = [] #The set that makes the most matches 
    largest_cnt = 0 #count of most matches
    for i in range(10):
        temp_cnt=0 #count of matches with selected_pair
        selected_pair = matched_pairs[random.randint(0,len(matched_pairs)-1)] #select one pair randomly

        i = selected_pair[0] # index of keypoints1 (from selected match)
        j = selected_pair[1] # index of keypoints2 (from selected match)

        # check that the change of orientation between the two keypoints of selected match
        selected_orientation = keypoints2[j][3]-keypoints1[i][3]

        if(selected_orientation >= 2 * math.pi):
            selected_orientation = selected_orientation - (2 * math.pi)

        # check that the change of scale between the two keypoints of selected match
        selected_scale = float(keypoints2[j][2])/keypoints1[i][2]

        # Check all matches and count only those similar to selected matches
        similar_pair = []
        for pair in matched_pairs:
            m = pair[0] # index of keypoints1
            n = pair[1] # index of keypoints2

            #check that the change of orientation between the two keypoints of a match
            orientation = keypoints2[n][3]-keypoints1[m][3] 
            if(orientation >= 2*math.pi):
               orientation = orientation - (2*math.pi)

            # check that the change of scale between the two keypoints of a match
            scale = float(keypoints2[n][2])/keypoints1[m][2]

            # calculate orientation difference of selected match and a match
            orientation_diff = math.fabs(orientation-selected_orientation)
            if orientation_diff >= 2 * math.pi:
                orientation_diff = orientation_diff - (2*math.pi)
            
            # check diffrences are within agreements(threshold)
            if orientation_diff < orient_agreement * math.pi/180.0 and ((selected_scale * (1.0 - scale_agreement))<scale<((selected_scale * (1.0 + scale_agreement)))):
                # if within agreements
                similar_pair.append(pair)
                temp_cnt=temp_cnt+1
        if largest_cnt < temp_cnt:
            largest_set= similar_pair # if find largest, then largest set is selected pair.
            largest_cnt=temp_cnt

    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    
    y1=descriptors1.shape[0] # get number of keypoint in image1
    y2=descriptors2.shape[0] # get number of keypoint in image2

    #descriptors1[i] means the descriptor of ith keypoint in image1
    #descriptors2[j] means the descriptor of jth keypoint in image2

    temp = np.zeros(y2)
    matched_pairs=[]

    for i in range(y1):
        for j in range(y2):
            #temp[j] is the angle between descriptors1[i] and descriptors2[j]
            #dot product of descriptors1[i] and descriptors2[j] equals to |descriptors1[i]||descriptors2[j]|cosθ
            #since descriptors have unit length, so we don't have to divide dot product into length of descriptors
            temp[j] = math.acos(np.dot(descriptors1[i],descriptors2[j]))

        compare=sorted(range(len(temp)),key=lambda k:temp[k]) # for finding index of nearest neighbor and second nearest neighbor
        #compare[0] is the index of the nearest neighbour, compare[1] is the index of second nearest neighbor. smallest angle is the nearest neighbor.
        #if the (nearest neighbour)/(second nearest neighbour) is small, then angle of the nearest neighbor is much smaller than angle of the second nearest neighbor
        if temp[compare[0]]/temp[compare[1]] < threshold: 
            matched_pairs.append([i,compare[0]]) #append [i,index of the nearest neighbor of descriptors1[i] in descriptors2]

    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START

    #convert the 2d points to homogeneous coordinate
    #xy_points is like [[1,2],
    #                   [3,4],
    #                   [5,6]
    #                    ...]
    #in pad_width=((0,0),(0,1)), (0,0) means padding top 0, padding bottom 0. (0,1) means padding left 0, padding right 1
    hcs = np.pad(xy_points,pad_width=((0,0),(0,1)),mode='constant',constant_values=1)

    #perform matrix mutliplication
    results = np.matmul(h,hcs.transpose())

    #results is like [[1,2,3,...] x
    #                 [4,5,6,...] y
    #                 [7,8,9,...] z]
    #before dividing by z values, if z is 0, change z to 1e-10
    z = results[2,:]
    z[z==0]=1e-10
    results[2,:]=z

    #divide by z coordinate
    results = results / z

    #remove z coordinate and transpose
    xy_points_out = results[:-1,:]
    xy_points_out = xy_points_out.transpose()
    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    #xy_src, xy_ref is like [[1,2],[3,4],[5,6]...]
    max_inliers=-1
    h=None

    n = xy_src.shape[0] # get number of matches
    SAMPLE_CNT=4 #number of sample
    for i in range(num_iter):
        sample_indices = np.random.choice(n,SAMPLE_CNT,replace=False) # choose sample indices
        src_sample = xy_src[sample_indices] # get src sample
        ref_sample = xy_ref[sample_indices] # get ref sample

        # find homography
        # Ah=0, A.T Ah=λh, choose smallest λ and corresponding h(then h is the homography matrix)
        # [v,λ] = eig(A.T A), if λ1<λ2...n then x=v1
        #ch = cv2.findHomography(src_sample,ref_sample)[0] # we can not use opencv for finding homography

        A = []
        for i in range(SAMPLE_CNT):
            # A = [[x,y,1,0,0,0,-x'x,-x'y,-x'],
            #      [0,0,0,x,y,1,-y'x,-y'y,-y']
            #       ...                      ]
            x = src_sample[i][0]
            y = src_sample[i][1]
            x_prime = ref_sample[i][0]
            y_prime = ref_sample[i][1]

            A.append([x,y,1,0,0,0,-1*x_prime * x, -1*x_prime*y,-1*x_prime])
            A.append([0,0,0,x,y,1,-1*y_prime*x,-1*y_prime*y,-1*y_prime])
        
        A=np.array(A,dtype=float)
        eig_pair = np.linalg.eig(np.matmul(A.transpose(),A)) #get eig(A.T A)
        smallestValueIndex = eig_pair[0].argmin() # get index of smallest eigen value
        ch=eig_pair[1][:,smallestValueIndex] # get eigenvector corresponding with the smallest eigenvalue
        ch=ch/ch[-1] # make h22 to 1
        ch=ch.reshape((3,3)) # change shape to (3,3)
        
        
        xy_res = KeypointProjection(xy_src,ch) #using KeyointProjection function, find points that result of projection xy_src with calculated homography
        distance = np.sqrt(np.sum((xy_res-xy_ref)**2,axis=1)) # get distance from calculated point and real point
        inlier = np.sum(distance<tol) # distance<tol is like [0,1,0,1,1,0...] so np.sum(distance<tol) is the number of ref points that distance from res is smaller than tol

        if inlier>max_inliers:
            # if we find max, change answer to above homography(ch)
            h=ch
            max_inliers=inlier
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
