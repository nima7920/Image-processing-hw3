import numpy as np
import cv2
import random


def get_random_patch(texture, patch_size):
    x_max, y_max = texture.shape[0] - patch_size[0] - 1, texture.shape[1] - patch_size[1] - 1
    x, y = random.randint(0, x_max), random.randint(0, y_max)
    patch = texture[x:x + patch_size[0], y:y + patch_size[1], :].copy()
    return patch


def find_patch_from_template(texture, template):
    search_result = cv2.matchTemplate(texture, template, cv2.TM_CCORR_NORMED)
    threshold = 0.95
    while threshold > 0.5:
        match_points_x, match_points_y = np.where(search_result > threshold)
        if match_points_x.size > 0:
            i = random.randint(0, match_points_x.size - 1)
            x, y = match_points_x[i], match_points_y[i]
            return x, y
        threshold -= 0.05


def find_path(weights):
    result = np.zeros((weights.shape[0], weights.shape[1], 3))
    index = np.where(weights[-1, :] == np.min(weights[-1, :]))[0][0]
    result[-1, 0:index, :] = 1
    for i in reversed(range(weights.shape[0] - 1)):
        if index == 0:
            if weights[i, 0] > weights[i, 1]:
                index = 1
                result[i, 0, :] = 1
        elif index == weights.shape[1] - 1:
            if weights[i, index] > weights[i, index - 1]:
                index = index - 1
            result[i, 0:index, :] = 1
        else:
            t = min(weights[i, index - 1], weights[i, index], weights[i, index + 1])
            if weights[i, index - 1] == t:
                index = index - 1
            elif weights[i, index + 1] == t:
                index = index + 1
            result[i, 0:index, :] = 1
    return result


def find_min_cut(template1, template2):
    difference = np.sum(np.square(template1 - template2), axis=2, keepdims=True).reshape(
        (template1.shape[0], template1.shape[1]))
    weights = np.zeros(difference.shape)
    weights[0, :] = difference[0, :].copy()
    for j in range(1, difference.shape[0]):
        for i in range(difference.shape[1]):
            if i == 0:
                weights[j, i] = difference[j, i] + min(weights[j - 1, i], weights[j - 1, i + 1])
            elif i == difference.shape[1] - 1:
                weights[j, i] = difference[j, i] + min(weights[j - 1, i], weights[j - 1, i - 1])
            else:
                weights[j, i] = difference[j, i] + min(weights[j - 1, i], weights[j - 1, i + 1], weights[j - 1, i - 1])
    result = find_path(weights)
    return result


def first_row_synthesis(img, texture, patch_size, overlap):
    first_patch = get_random_patch(texture, patch_size)
    img[0:patch_size[0], 0:patch_size[1], :] = first_patch
    number_of_patches = int((img.shape[1] - patch_size[1]) / (patch_size[1] - overlap))
    cut_from_end = patch_size[1]
    for i in range(number_of_patches):
        template_x = (i + 1) * (patch_size[0] - overlap)
        template = img[0:patch_size[1], template_x:template_x + overlap, :].copy()
        x, y = find_patch_from_template(texture[:, 0:-cut_from_end, :], template)
        template2 = texture[x:x + patch_size[0], y:y + overlap, :].copy()
        min_cut = find_min_cut(template, template2)
        final_template = (min_cut * template + (1 - min_cut) * template2).copy()
        img[0:patch_size[0], template_x:template_x + overlap, :] = final_template.copy()
        patch = texture[x:x + patch_size[0], y + overlap:y + patch_size[1], :].copy()
        img[0:patch_size[0], template_x + overlap:template_x + patch_size[0], :] = patch.copy()


'''   functions for synthesising other rows '''


def find_patch_from_L_template(texture, template, mask):
    search_result = cv2.matchTemplate(texture, template, cv2.TM_CCORR_NORMED, mask=mask)
    threshold = 0.95
    while threshold > 0.5:
        match_points_x, match_points_y = np.where(search_result > threshold)
        if match_points_x.size > 0:
            i = random.randint(0, match_points_x.size - 1)
            x, y = match_points_x[i], match_points_y[i]
            return x, y
        threshold -= 0.05


def find_L_min_cut(template1, template2, overlap):
    result = np.zeros(template1.shape)
    top_min_cut = find_min_cut(template1[0:overlap, :, :].transpose(1, 0, 2),
                               template2[0:overlap, :, :].transpose(1, 0, 2)).transpose(1, 0, 2)
    left_min_cut = find_min_cut(template1[:, 0:overlap, :], template2[:, 0:overlap, :])
    result[0:overlap, :, :] = np.logical_or(result[0:overlap, :, :], top_min_cut)
    result[:, 0:overlap, :] = np.logical_or(result[:, 0:overlap, :], left_min_cut)
    return result


def synthesis_row_i(img, texture, patch_size, overlap, i):
    x = i * (patch_size[0] - overlap)
    template = img[x:x + overlap, 0:patch_size[1], :].copy()
    x1, y1 = find_patch_from_template(texture[0:-patch_size[0], :, :], template)
    template2 = texture[x1:x1 + overlap, y1:y1 + patch_size[1], :].copy()
    min_cut = find_min_cut(template.transpose(1, 0, 2), template2.transpose(1, 0, 2)).transpose(1, 0, 2)
    final_template = (min_cut * template + (1 - min_cut) * template2).copy()
    img[x:x + overlap, 0:patch_size[1], :] = final_template.copy()
    patch = texture[x1 + overlap:x1 + patch_size[0], y1:y1 + patch_size[1], :].copy()
    print(patch.shape)
    img[x + overlap:x + patch_size[0], 0:patch_size[1], :] = patch.copy()
    ''' synthesising the rest of the row '''
    mask = np.zeros((patch_size[0], patch_size[1], 3), dtype='uint8')
    mask[0:overlap, :, :] = 1
    mask[:, 0:overlap, :] = 1
    number_of_patches = int((img.shape[1] - patch_size[1]) / (patch_size[1] - overlap))
    for j in range(number_of_patches):
        template_x = (j + 1) * (patch_size[0] - overlap)
        template = img[x:x + patch_size[1], template_x:template_x + patch_size[0], :].copy()
        x1, y1 = find_patch_from_L_template(texture[0:-patch_size[0], 0:-patch_size[0], :], template, mask)
        template2 = texture[x1:x1 + patch_size[1], y1:y1 + patch_size[0], :].copy()
        min_cut = find_L_min_cut(template, template2, overlap)
        patch = (min_cut * template + (1 - min_cut) * template2).copy()
        img[x:x + patch_size[0], template_x:template_x + patch_size[1], :] = patch.copy()


def next_rows_synthesis(img, texture, patch_size, overlap):
    num_of_iterations = int(img.shape[0] / (patch_size[0] - overlap) - 1)
    for i in range(1, num_of_iterations):
        synthesis_row_i(img, texture, patch_size, overlap, i)


def apply_texture_synthesis(texture, result_shape, patch_size, overlap):
    result = np.zeros(result_shape, dtype='uint8')
    first_row_synthesis(result, texture, patch_size, overlap)
    next_rows_synthesis(result, texture, patch_size, overlap)
    final_result = np.zeros((result_shape[0], result_shape[1] + texture.shape[1] + overlap, result_shape[2]),
                            dtype='uint8')
    final_result[:, :result_shape[1], :] = result
    final_result[:texture.shape[0], result_shape[1] + overlap:, :] = texture
    return final_result
