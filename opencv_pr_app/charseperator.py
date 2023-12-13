import numpy as np
import cv2 as cv

def get_candidate_chars(candidate_plate_image):
    # 灰度化和二值化该区域
    gray_image = cv.cvtColor(candidate_plate_image, cv.COLOR_BGR2GRAY)
    ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    # 去掉外部白色边框，以免查找轮廓时仅找到外框
    offsetX = 3
    offsetY = 5
    offset_region = binary_image[offsetY:-offsetY, offsetX:-offsetX]
    working_region = np.copy(offset_region)

    # 仅将汉字字符所在区域模糊化，使得左右结构或上下结构的汉字不会被识别成多个不同的轮廓
    chinese_char_max_width = working_region.shape[1] // 8;		# 假设汉字最大宽度为整个车牌宽度的1/8
    chinese_char_region = working_region[:, 0:chinese_char_max_width]
    cv.GaussianBlur(chinese_char_region, (9, 9), 0, dst=chinese_char_region)            # 采用In-Place平滑处理

    # 在工作区中查找轮廓
    char_contours, _ = cv.findContours(working_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(candidate_plate_image, char_contours, -1, (0, 0, 255))
    # cv.imshow("", candidate_plate_image)
    # cv.waitKey()
    # 以下假设字符的最小宽度是车牌宽度的1/40（考虑到I这样的字符），高度是80%
    min_width = working_region.shape[1] // 40
    min_height = working_region.shape[0] * 7 // 10
    valid_char_regions = []
    for i in np.arange(len(char_contours)):
        x, y, w, h = cv.boundingRect(char_contours[i])
        #字符高度和宽度必须满足条件
        if h >= min_height and w >= min_width:       
            # 这里采用offset_region而不是working_region，是因为：
            # (1)二者区块相同
            # (2)working_region被上面汉字模糊化处理过，会导致提取的汉字区域模糊
            valid_char_regions.append((x, offset_region[y:y+h, x:x+w]))     
    # 按照轮廓的x坐标从左到右排序
    sorted_regions = sorted(valid_char_regions, key=lambda region:region[0])

    # 包装到一个list中返回
    candidate_char_images = []
    for i in np.arange(len(sorted_regions)):
        candidate_char_images.append(sorted_regions[i][1])

    return candidate_char_images