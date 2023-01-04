import cv2
import numpy as np
import zhang_suen_algorithm_functions as zsf
import b_spline_functions as bsf


# Read image file
image_name = 'death.png'
img = cv2.imread(image_name, 2)

# Initialize resizing parameters
scale_percent = 20 # percent of original size (choose for hand, runner)
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# Binarize image: 0 = black, 255 = white
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Scale colors: 0 = black, 1 = white
bw_img = bw_img/255
rows, cols = np.shape(bw_img)

# Skeletonize Image
bw_img = zsf.skeletonize(bw_img)

# Assign branch labels
branches = zsf.assign_branch_labels(bw_img)

# Find all branches and split them
B_set = zsf.split_all_branches(bw_img, branches)

# Plot final result
bsf.plot_bspline_result(B_set, image_name, bw_img, dpi=200)








