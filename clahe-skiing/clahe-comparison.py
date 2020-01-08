import cv2

def coloredCLAHE(image_path, gridsize, clipLimit):
    img_bgr = cv2.imread(image_path) # read in image
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB) # convert to YCrCb
    split_ycrcb = cv2.split(img_ycrcb) # split into channels

    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridsize,gridsize))
    split_ycrcb[0] = clahe.apply(split_ycrcb[0]) # apply CLAHE to Y channel

    img_ycrcb = cv2.merge(split_ycrcb) # merge channels back together
    img_bgr_clahe = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR) # # convert back to bgr

    return img_bgr_clahe


if __name__ == "__main__":
    # filepaths
    image_path = "ski-hill.png"

    # CLAHE parameters
    gridsize = 8
    clipLimit = 2.0

    # read in original image
    original = cv2.imread(image_path)

    # enhance image with CLAHE
    enhanced = coloredCLAHE(image_path=image_path, gridsize=gridsize, clipLimit=clipLimit)

    # display images
    cv2.imshow("Original", original)
    cv2.imshow("CLAHE Enhanced", enhanced)
    cv2.waitKey(0)
