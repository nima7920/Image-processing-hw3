import random

import matplotlib.pyplot as plt
import numpy as np
import cv2
import q2_funcs as q2

''' patch match method for image completion '''


def initialize_random(A, B, patch_size):
    A_dims, B_dims = A.shape, B.shape
    offsets_x = np.zeros((A_dims[0] - patch_size[0], A_dims[1] - patch_size[1]), dtype='int16')
    offsets_y = np.zeros((A_dims[0] - patch_size[0], A_dims[1] - patch_size[1]), dtype='int16')
    for i in range(A_dims[0] - patch_size[0]):
        for j in range(A_dims[1] - patch_size[1]):
            offsets_x[i, j] = np.random.randint(-i, B_dims[0] - i - patch_size[0] - 1)
            offsets_y[i, j] = np.random.randint(-j, B_dims[1] - j - patch_size[1] - 1)
    offsets = np.zeros((A_dims[0] - patch_size[0], A_dims[1] - patch_size[1], 2), dtype='int16')
    offsets[:, :, 0] = offsets_x.copy()
    offsets[:, :, 1] = offsets_y.copy()
    return offsets


def find_difference(A, B, offsets, patch_size, x, y):
    patch_A = A[x:x + patch_size[0], y:y + patch_size[1], :].copy()
    patch_B = B[x + int(offsets[x, y, 0]):x + int(offsets[x, y, 0]) + patch_size[0],
              y + int(offsets[x, y, 1]):y + int(offsets[x, y, 1]) + patch_size[1], :].copy()
    ssd = np.sum(np.square(patch_A - patch_B))
    return ssd


def get_random_patch(B, x_min, x_max, y_min, y_max, patch_size):
    high_x, high_y = x_max - patch_size[0], y_max - patch_size[1]
    x, y = np.random.randint(x_min, high_x - 1), np.random.randint(y_min, high_y - 1)
    return x, y


def random_search(A, B, offsets, patch_size, x, y, d, alpha):
    patch_A = A[x:x + patch_size[0], y:y + patch_size[1], :].copy()
    x_b, y_b = x + offsets[x, y, 0], y + offsets[x, y, 1]
    offset_final = offsets[x, y].copy()
    d_min = d
    # parameters indicating window :
    x_min, x_max = 0, B.shape[0] - 1
    y_min, y_max = 0, B.shape[1] - 1
    while (x_max - x_min) > 2 * patch_size[0] and (y_max - y_min) > 2 * patch_size[1]:
        x_r, y_r = get_random_patch(B, x_min, x_max, y_min, y_max, patch_size)
        random_patch = B[x_r:x_r + patch_size[0], y_r:y_r + patch_size[1], :].copy()
        ssd = np.sum(np.square(patch_A - random_patch))
        if ssd < d_min:
            d_min = ssd
            offset_final = (x_r - x, y_r - y)
        # updating window
        x_min, x_max = x_min + int((x_b - x_min) / alpha), x_max - int((x_max - x_min - patch_size[0]) / alpha)
        y_min, y_max = y_min + int((y_b - y_min) / alpha), y_max - int((y_max - y_min - patch_size[1]) / alpha)
    return offset_final


def update_value(A, B, offsets, patch_size, x, y):
    values_x = offsets[max(0, x - int(patch_size[0] / 2) + 1):x + 1, max(0, y - int(patch_size[1] / 2) + 1): y + 1,
               0] + x
    values_y = offsets[max(0, x - int(patch_size[0] / 2) + 1):x + 1, max(0, y - int(patch_size[1] / 2) + 1): y + 1,
               1] + y
    values_x = values_x.reshape(-1).astype('int16')
    values_y = values_y.reshape(-1).astype('int16')
    value = np.array(
        [np.sum(B[values_x, values_y, 0]), np.sum(B[values_x, values_y, 1]), np.sum(B[values_x, values_y, 2])]) / (
                values_x.shape[0])
    A[x, y] = value
    # deviate = int(patch_size[0] / 2)
    # A[x + deviate, y + deviate] = value
    # A[x, y] = B[x + offsets[x, y, 0], y + offsets[x, y, 1]].copy()


def propagate_odd(A, B, offsets, patch_size, alpha=2):
    m, n = A.shape[0] - patch_size[0], A.shape[1] - patch_size[1]
    for i in range(m):
        for j in range(n):
            if i == 0:  # on the first row
                if j > 0:
                    d1, d2 = find_difference(A, B, offsets, patch_size, i, j), find_difference(A, B, offsets,
                                                                                               patch_size, i, j - 1)
                    d = min(d1, d2)
                    if d == d2:
                        if offsets[i, j - 1, 1] + j < B.shape[1] - patch_size[1]:
                            offsets[i, j] = (offsets[i, j - 1, 0], offsets[i, j - 1, 1])
                        else:
                            offsets[i, j] = (offsets[i, j - 1, 0], offsets[i, j - 1, 1] - 1)
                    offset_final = random_search(A, B, offsets, patch_size, i, j, d, alpha)
                    offsets[i, j] = offset_final
                    # update_value(A, B, offsets, patch_size, i, j)
            else:
                if j > 0:  # inside the matrix
                    d1 = find_difference(A, B, offsets, patch_size, i, j)
                    d2 = find_difference(A, B, offsets, patch_size, i - 1, j)
                    d3 = find_difference(A, B, offsets, patch_size, i, j - 1)
                    d = min(d1, d2, d3)
                    if d == d2:
                        if offsets[i - 1, j, 0] + i < B.shape[0] - patch_size[0]:
                            offsets[i, j] = (offsets[i - 1, j, 0], offsets[i - 1, j, 1])
                        else:
                            offsets[i, j] = (offsets[i - 1, j, 0] - 1, offsets[i - 1, j, 1])
                    elif d == d3:
                        if offsets[i, j - 1, 1] + j < B.shape[1] - patch_size[1]:
                            offsets[i, j] = (offsets[i, j - 1, 0], offsets[i, j - 1, 1])
                        else:
                            offsets[i, j] = (offsets[i, j - 1, 0], offsets[i, j - 1, 1] - 1)
                else:  # on the first column
                    d1, d2 = find_difference(A, B, offsets, patch_size, i, j), find_difference(A, B, offsets,
                                                                                               patch_size, i - 1, j)
                    d = min(d1, d2)
                    if d == d2:
                        if offsets[i - 1, j, 0] + i < B.shape[0] - patch_size[0]:
                            offsets[i, j] = (offsets[i - 1, j, 0], offsets[i - 1, j, 1])
                        else:
                            offsets[i, j] = (offsets[i - 1, j, 0] - 1, offsets[i - 1, j, 1])
                offset_final = random_search(A, B, offsets, patch_size, i, j, d, alpha)
                offsets[i, j] = offset_final
                # update_value(A, B, offsets, patch_size, i, j)


def propagate_even(A, B, offsets, patch_size, alpha=2):
    m, n = A.shape[0] - patch_size[0], A.shape[1] - patch_size[1]
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            if i == m - 1:  # on the last row
                if j < n - 1:
                    d1, d2 = find_difference(A, B, offsets, patch_size, i, j), find_difference(A, B, offsets,
                                                                                               patch_size, i, j + 1)
                    d = min(d1, d2)
                    if d == d2:
                        if offsets[i, j + 1, 1] + j >= 0:
                            offsets[i, j] = (offsets[i, j + 1, 0], offsets[i, j + 1, 1])
                        else:
                            offsets[i, j] = (offsets[i, j + 1, 0], offsets[i, j + 1, 1] + 1)
                    offset_final = random_search(A, B, offsets, patch_size, i, j, d, alpha)
                    offsets[i, j] = offset_final
                    # update_value(A, B, offsets, patch_size, i, j)
            else:
                if j < n - 1:  # inside the matrix
                    d1 = find_difference(A, B, offsets, patch_size, i, j)
                    d2 = find_difference(A, B, offsets, patch_size, i + 1, j)
                    d3 = find_difference(A, B, offsets, patch_size, i, j + 1)
                    d = min(d1, d2, d3)
                    if d == d2:
                        if offsets[i + 1, j, 0] + i >= 0:
                            offsets[i, j] = (offsets[i + 1, j, 0], offsets[i + 1, j, 1])
                        else:
                            offsets[i, j] = (offsets[i + 1, j, 0] + 1, offsets[i + 1, j, 1])
                    elif d == d3:
                        if offsets[i, j + 1, 1] + j >= 0:
                            offsets[i, j] = (offsets[i, j + 1, 0], offsets[i, j + 1, 1])
                        else:
                            offsets[i, j] = (offsets[i, j + 1, 0], offsets[i, j + 1, 1] + 1)
                else:  # on the last column
                    d1, d2 = find_difference(A, B, offsets, patch_size, i, j), find_difference(A, B, offsets,
                                                                                               patch_size, i + 1, j)
                    d = min(d1, d2)
                    if d == d2:
                        if offsets[i + 1, j, 0] + i >= 0:
                            offsets[i, j] = (offsets[i + 1, j, 0], offsets[i + 1, j, 1])
                        else:
                            offsets[i, j] = (offsets[i + 1, j, 0] + 1, offsets[i + 1, j, 1])
                offset_final = random_search(A, B, offsets, patch_size, i, j, d, alpha)
                offsets[i, j] = offset_final
                # update_value(A, B, offsets, patch_size, i, j)


def generate_image(A, patch_size, offsets, B):
    m, n = offsets.shape[0], offsets.shape[1]
    for i in range(m):
        for j in range(n):
            update_value(A, B, offsets, patch_size, i, j)


def perform_patch_match(A, B, patch_size, iteration=4):
    offsets = initialize_random(A, B, patch_size)
    for i in range(iteration):
        propagate_odd(A, B, offsets, patch_size)
        generate_image(A, patch_size, offsets, B)

        propagate_even(A, B, offsets, patch_size)
        generate_image(A, patch_size, offsets, B)
    return A


''' functions for image completion using texture synthesis '''


def synthesis_row_i(img, texture, patch_size, overlap, i):
    x = i * (patch_size[0] - overlap)
    mask = np.zeros((patch_size[0], patch_size[1], 3), dtype='uint8')
    mask[0:overlap, :, :] = 1
    mask[:, 0:overlap, :] = 1
    number_of_patches = int((img.shape[1] - patch_size[1]) / (patch_size[1] - overlap))
    for j in range(number_of_patches + 1):
        template_x = j * (patch_size[1] - overlap)
        template = img[x:x + patch_size[0], template_x:template_x + patch_size[1], :].copy()
        x1, y1 = q2.find_patch_from_L_template(texture[0:-patch_size[0], 0:-patch_size[1], :], template, mask)
        template2 = texture[x1:x1 + patch_size[1], y1:y1 + patch_size[0], :].copy()
        min_cut = q2.find_L_min_cut(template, template2, overlap)
        patch = (min_cut * template + (1 - min_cut) * template2).copy()
        img[x:x + patch_size[0], template_x:template_x + patch_size[1], :] = patch.copy()


def complete_image_with_texture(img, texture, patch_size, overlap):
    num_of_iterations = int(img.shape[0] / (patch_size[0] - overlap)) - 1
    for i in range(num_of_iterations):
        synthesis_row_i(img, texture, patch_size, overlap, i)
