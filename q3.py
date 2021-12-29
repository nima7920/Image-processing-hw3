import numpy as np
import cv2
import q3_funcs

input_paths = ["inputs/im03-1.jpg", "inputs/im04.jpg"]

output_paths = ["outputs/res15.jpg", "outputs/res16.jpg"]

img_birds = cv2.imread(input_paths[0])
img_swimmer = cv2.imread(input_paths[1])

''' removing birds  '''

result_birds = img_birds.copy()
patch_size = (6, 6)
# two bottom birds
A1 = img_birds[700:950, 820:980, :].copy()
B1 = img_birds[720:950, 480:780, :].copy()
result1 = q3_funcs.perform_patch_match(A1, B1, patch_size, 4)
result_birds[700:950, 820:980, :] = result1

A2 = img_birds[610:740, 1130:1250, :].copy()
B2 = img_birds[550:650, 1110:1155, :].copy()
result2 = q3_funcs.perform_patch_match(A2, B2, patch_size, 4)
result_birds[610:740, 1130:1250, :] = result2

A3_1_2 = result_birds[120:175, 325:381, :].copy()
B3_1 = img_birds[140:240, 330:390, :].copy()
result3_1_2 = q3_funcs.perform_patch_match(A3_1_2, B3_1, patch_size, 4)
result_birds[120:175, 325:381, :] = result3_1_2

A3_1_1 = result_birds[65:175, 325:381, :].copy()
B3_1 = img_birds[140:240, 330:390, :].copy()
result3_1_1 = q3_funcs.perform_patch_match(A3_1_1, B3_1, patch_size, 4)
result_birds[65:175, 325:381, :] = result3_1_1


A3_2 = result_birds[65:175, 375:441, :].copy()
B3_2 = img_birds[80:165, 220:305, :].copy()
result3_2 = q3_funcs.perform_patch_match(A3_2, B3_2, patch_size, 4)
result_birds[65:175, 375:441, :] = result3_2

A3_3 = result_birds[65:175, 435:491, :].copy()
B3_3 = img_birds[120:240, 425:480, :].copy()
result3_3 = q3_funcs.perform_patch_match(A3_3, B3_3, patch_size, 4)
result_birds[65:175, 435:491, :] = result3_3

A3_4 = result_birds[65:175, 485:551, :].copy()
B3_4 = img_birds[140:270, 460:530, :].copy()
result3_4 = q3_funcs.perform_patch_match(A3_4, B3_4, patch_size, 4)
result_birds[65:175, 485:551, :] = result3_4

cv2.imwrite(output_paths[0], result_birds)

''' removing swimmer '''
patch_size = (6, 6)
A = img_swimmer[680:1200 + patch_size[0], 740:960 + patch_size[1], :].copy()
B = img_swimmer[1200:, :, :].copy()
offsets = q3_funcs.initialize_random(A, B, patch_size)
result = q3_funcs.perform_patch_match(A, B, patch_size, 4)
img_swimmer[680:1200 + patch_size[0], 740:960 + patch_size[1], :] = result
cv2.imwrite(output_paths[1], img_swimmer)
