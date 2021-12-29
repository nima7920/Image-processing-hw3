import numpy as np
import cv2


def generate_accumulator_array(img_edge, rho_range, theta_range):
    result = np.zeros((rho_range.size, theta_range.size))
    offset = int(rho_range.size / 2)
    mask = np.where(img_edge > 0)
    for i in range(mask[0].size):
        y, x = mask[0][i], mask[1][i]
        for theta in theta_range:
            theta_rad = np.deg2rad(theta)
            rho = np.round(x * np.cos(theta_rad) + y * np.sin(theta_rad)) + offset
            result[int(rho), int(theta)] += 1

    return result


def draw_line_with_slope(img, m, b, color=(255, 0, 0)):
    height, width = img[:, :, 0].shape
    x1, x2, y1, y2 = int(-b / m), int((height - b) / m), int(b), int(m * width + b)
    if 0 <= x1 <= width and 0 <= y1 <= height:
        cv2.line(img, (x1, 0), (0, y1), color, 3)
    elif 0 <= x1 <= width and 0 <= y2 <= height:
        cv2.line(img, (x1, 0), (width, y2), color, 3)
    elif 0 <= x2 <= width and 0 <= y1 <= height:
        cv2.line(img, (x2, height), (0, y1), color, 3)
    elif 0 <= x2 <= width and 0 <= y2 <= height:
        cv2.line(img, (x2, height), (width, y2), color, 3)
    elif 0 <= x1 <= width and 0 <= x2 <= width:
        cv2.line(img, (x1, 0), (x2, height), color, 3)
    elif 0 <= y1 <= height and 0 <= y2 <= height:
        cv2.line(img, (0, y1), (width, y2), color, 3)


def draw_line(img, rho, theta):
    theta_rad = np.deg2rad(theta)
    sin, cos = np.sin(theta_rad), np.cos(theta_rad)
    if sin == 0:
        sin += 0.00000001
    if cos == 0:
        cos += 0.00000001
    m, b = -cos / sin, rho / sin
    draw_line_with_slope(img, m, b)
    return [m, b]


def draw_lines(accumulator_array, img, threshold, rho_offset):
    mask = np.where(accumulator_array > threshold)
    lines = np.zeros((2, 0))
    min_rho_diff, min_theta_diff = 30, 45
    prev_rho, prev_theta = int(mask[0][0]), int(mask[1][0])
    for i in range(mask[0].size):
        rho, theta = int(mask[0][i]), int(mask[1][i])
        if theta > 2 and np.abs(theta - 90) > 2:
            if np.abs(rho - prev_rho) > min_rho_diff or np.abs(theta - prev_theta) > min_theta_diff:
                [m, b] = draw_line(img, rho - rho_offset, theta)
                lines = np.c_[lines, [m, b]]
                prev_rho, prev_theta = rho, theta
    return lines


def find_cross(m1, b1, m2, b2):
    x = int((b2 - b1) / (m1 - m2))
    y = int(m1 * x + b1)
    return y, x


def is_colors_valid(color_l, color_r, color_u, color_d):
    dif1, dif2, dif3 = np.abs(int(color_l) - int(color_d)), np.abs(int(color_l) - int(color_r)), np.abs(
        int(color_l) - int(color_u))
    dif4, dif5, dif6 = np.abs(int(color_r) - int(color_d)), np.abs(int(color_r) - int(color_u)), np.abs(
        int(color_u) - int(color_d))
    if (dif2 < 50 or dif6 < 50) and (dif4 > 300 or dif5 > 300):
        return True

    return False


def is_point_valid(img, x, y, m):
    height, width = img.shape
    offset = 15
    if offset <= x < width - offset and offset <= y < height - offset:
        xl, xr, xu, xd = x - offset, x + offset, x, x
        yl, yr, yu, yd = y, y, y - offset, y + offset
        color_l, color_r, color_u, color_d = img[yl, xl], img[yr, xr], img[yu, xu], img[yd, xd]
        print(x,y)
        return is_colors_valid(color_l, color_r, color_u, color_d)

    return False


def is_line_valid(img_gray, m, b, lines, limit=8):
    counter = 0
    for i in range(lines.shape[1]):
        m2, b2 = lines[0, i], lines[1, i]
        if np.abs(m - m2) > 0.3:
            y, x = find_cross(m, b, m2, b2)
            if is_point_valid(img_gray, x, y, m):
                counter += 1
    if counter >= limit:
        return True
    return False


def filter_lines_by_cross_points(img, lines, limit=8):
    valid_lines = np.zeros((2, 0))
    img_gray = np.sum(img, axis=2, keepdims=True).reshape((img.shape[0], img.shape[1])).astype('int')
    print(img_gray.shape)
    for i in range(lines.shape[1]):
        m, b = lines[0, i], lines[1, i]
        if is_line_valid(img_gray, m, b, lines, limit):
            valid_lines = np.c_[valid_lines, [m, b]]

    return valid_lines


def draw_lines_with_slope(img, lines, color=(255, 0, 0)):
    for i in range(lines.shape[1]):
        m, b = lines[0, i], lines[1, i]
        draw_line_with_slope(img, m, b, color)


def draw_corners(img, valid_lines):
    for i in range(valid_lines.shape[1]):
        m1, b1 = valid_lines[0, i], valid_lines[1, i]
        for j in range(valid_lines.shape[1]):
            m2, b2 = valid_lines[0, j], valid_lines[1, j]
            if np.abs(m1 - m2) > 0.1:
                y, x = find_cross(m1, b1, m2, b2)
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
