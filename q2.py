import cv2
import q2_funcs

input_paths = ["inputs/texture-new.jpg", "inputs/texture08.jpg",
               "inputs/texture6.jpg", "inputs/texture03.jpg"]
output_paths = ["outputs/res11.jpg", "outputs/res12.jpg",
                "outputs/res13.jpg", "outputs/res14.jpg"]

result_shape = (2500, 2500, 3)
patch_size = (200, 200)
overlap = 100

for i in range(len(input_paths)):
    print(i)
    print(input_paths[i])
    texture = cv2.imread(input_paths[i])
    print(texture.shape)
    result = q2_funcs.apply_texture_synthesis(texture, result_shape, patch_size, overlap)
    cv2.imwrite(output_paths[i], result)
