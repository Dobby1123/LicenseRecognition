import numpy as np
import cv2 as cv 
import util

def get_candidate_plates(plate_image):
    blured_image = cv.GaussianBlur(plate_image, (5, 5), 0)
    gray_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray_image, cv.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    grad_image = cv.addWeighted(abs_grad_x, 1, 0, 0, 0)
    ret, threshold_image = cv.threshold(grad_image, 0, 255, cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
    morphology_image = cv.morphologyEx(threshold_image, cv.MORPH_CLOSE, kernel)
        
    ### 执行筛选、矫正和尺寸调整
    contours, _ = cv.findContours(morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    verified_plates = []
    for i in np.arange(len(contours)):
        if  util.verify_plate_sizes(contours[i]):
            output_image = util.rotate_plate_image(contours[i], plate_image)
            output_image = util.unify_plate_image(output_image)
            verified_plates.append(output_image)

    return verified_plates