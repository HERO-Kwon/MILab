import cv2
import numpy as np
import os
import re

img_path = 'D:\Matlab_Drive\Data\WIFI\Image\Signature\\'
img_list = os.listdir(img_path)
save_path = 'D:\Matlab_Drive\Data\WIFI\Image\Signature_processed\\'

for img_sign in img_list:
    img_name = re.search('IMG_\d+',img_sign).group(0)
    # Read the image
    img = cv2.imread(img_path+img_sign, 0)

    # Rotate
    if img.shape[0] < img.shape[1]:
        center=tuple(np.array(img.shape[0:2])/2)
        rot_mat = cv2.getRotationMatrix2D(center,270,1.0)
        img = cv2.warpAffine(img, rot_mat, img.shape[0:2],flags=cv2.INTER_LINEAR)

    #elode,dilate
    kernel = np.ones((3,3),np.uint8)
    e = cv2.erode(img,kernel,iterations = 2)  
    d = cv2.dilate(e,kernel,iterations = 1)


    # Thresholding the image
    (thresh, img_bin) = cv2.threshold(d, 128, 255,cv2.THRESH_BINARY|     cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(d,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,15,15)
    th3 = cv2.adaptiveThreshold(d,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,15,15)
    # Invert the image
    #img_bin = 255-img_bin 
    img_bin = 255-th2
    #cv2.imwrite("Image_bin2.jpg",255-th2)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//80
    
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    #cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    #cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite("img_final_bin.jpg",img_final_bin)

    # Find contours for image, which will detect all the boxes
    im2, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
    
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
    
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
    
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
    
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)
    
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")
    idx = 0
    ori_shape = img_bin.shape
    bias_x = 20 # to remove box line
    bias_y = 17 # to remove box line
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        if (w > 300 and h > 150) and 15*w*h < ori_shape[0]*ori_shape[1]:
            idx += 1
            new_img = img[y+bias_y:y+h-bias_y, x+bias_x:x+w-bias_x]
            resized_img = cv2.resize(new_img,(625,375))
            cv2.imwrite(save_path+img_name+'_'+str(idx) + '.png', resized_img)