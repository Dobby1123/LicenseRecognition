import numpy as np
import cv2 as cv
import charseperator

plate_file_path = "images/candidate_plate.jpg"
candidate_plate_image = cv.imread(plate_file_path)

candidate_chars = charseperator.get_candidate_chars(candidate_plate_image)
for char in candidate_chars:
    cv.imshow("", char)
    cv.waitKey()

cv.destroyAllWindows()