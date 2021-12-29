import copy

import matplotlib.pyplot as plt
import numpy as np
import cv2
import q1_funcs

#### Question 1 : Hough Transform ####
import q1_funcs

input_paths = ["inputs/im01.jpg", "inputs/im02.jpg"]
output_paths = ["outputs/res01.jpg", "outputs/res02.jpg", "outputs/res03-hough-space.jpg"
    , "outputs/res04-hough-space.jpg", "outputs/res05-lines.jpg", "outputs/res06-lines.jpg",
                "outputs/res07-chess.jpg", "outputs/res08-chess.jpg",
                "outputs/res09-corners.jpg", "outputs/res10-corners.jpg"]

''' first image '''
im1 = cv2.imread(input_paths[0])
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

'''  detecting edges '''
img_edge_1 = cv2.Canny(im1, 350, 350)

number_of_good_points_limit = 3

# showing and saving
plt.imshow(img_edge_1, cmap="gray")
plt.savefig(output_paths[0])
plt.show()

'''  generating accumulator array '''
max_distance = int(np.ceil(np.sqrt(im1.shape[0] ** 2 + im1.shape[1] ** 2)))
rho_range = np.linspace(-int(max_distance), int(max_distance), max_distance * 2)
theta_range = np.linspace(0, 179, 180)
threshold = 100

accumulator_array1 = q1_funcs.generate_accumulator_array(img_edge_1, rho_range, theta_range)
plt.imshow(accumulator_array1)
plt.savefig(output_paths[2])
plt.show()

''' drawing lines '''
img_lines_1 = im1.copy()
lines1 = q1_funcs.draw_lines(accumulator_array1, img_lines_1, threshold, int(rho_range.size / 2))
q1_funcs.filter_lines_by_cross_points(img_lines_1, lines1)
plt.imshow(img_lines_1)
plt.savefig(output_paths[4])
plt.show()

''' removing irrelevant lines '''
img_valid_lines1 = copy.deepcopy(im1)
valid_lines = q1_funcs.filter_lines_by_cross_points(im1, lines1, number_of_good_points_limit)
q1_funcs.draw_lines_with_slope(img_valid_lines1, valid_lines, color=(0, 0, 255))
plt.imshow(img_valid_lines1)
plt.savefig(output_paths[6])
plt.show()

''' drawing corners '''
img_corners1 = im1.copy()
q1_funcs.draw_corners(img_corners1, valid_lines)
plt.imshow(img_corners1)
plt.savefig(output_paths[8])
plt.show()
''' second image '''

im2 = cv2.imread(input_paths[1])
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

''' detecting edges '''
img_edge_2 = cv2.Canny(im2, 350, 350)

plt.imshow(img_edge_2, cmap='gray')
plt.savefig(output_paths[1])
plt.show()

'''  generating accumulator array '''
max_distance = int(np.ceil(np.sqrt(im2.shape[0] ** 2 + im2.shape[1] ** 2)))
rho_range = np.linspace(-int(max_distance), int(max_distance), max_distance * 2)

accumulator_array2 = q1_funcs.generate_accumulator_array(img_edge_2, rho_range, theta_range)
plt.imshow(accumulator_array2)
plt.savefig(output_paths[3])
plt.show()

''' drawing lines '''
img_lines_2 = im2.copy()
lines2 = q1_funcs.draw_lines(accumulator_array2, img_lines_2, threshold, int(rho_range.size / 2))
q1_funcs.filter_lines_by_cross_points(img_lines_2, lines2)
plt.imshow(img_lines_2)
plt.savefig(output_paths[5])
plt.show()

''' removing irrelevant lines '''
img_valid_lines2 = copy.deepcopy(im2)
valid_lines2 = q1_funcs.filter_lines_by_cross_points(im2, lines2, number_of_good_points_limit)
q1_funcs.draw_lines_with_slope(img_valid_lines2, valid_lines2, color=(0, 0, 255))
plt.imshow(img_valid_lines2)
plt.savefig(output_paths[7])
plt.show()

''' drawing corners '''
img_corners2 = im2.copy()
q1_funcs.draw_corners(img_corners2, valid_lines2)
plt.imshow(img_corners2)
plt.savefig(output_paths[9])
plt.show()
