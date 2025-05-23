# processing.py
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import (
    reconstruction, remove_small_objects, binary_opening, disk, binary_erosion
)

from ultralytics import YOLO

def load_and_crop_digit_region(image_path, model_path):
    model = YOLO(model_path, task="detect")
    original_img = cv2.imread(image_path)
    orig_h, orig_w = original_img.shape[:2]
    results = model.predict(source=image_path, save=False, imgsz=160)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        box_data = []
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
            xyxy = box.xyxy[0].cpu().numpy()
            box_data.append((conf, cls, xyxy))
        box_data.sort(reverse=True, key=lambda x: x[0])

        if len(box_data) > 0:
            _, _, box = box_data[0]
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(orig_w - 1, x1))
            x2 = max(0, min(orig_w, x2))
            y1 = max(0, min(orig_h - 1, y1))
            y2 = max(0, min(orig_h, y2))
            return original_img[y1:y2, x1:x2]
    return original_img

def rotate_image(cropped_display):
    edges = cv2.Canny(cropped_display, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    angles = []

    if lines is not None:
        for rho, theta in lines[:, 0]:
            deg = np.degrees(theta)
            if abs(deg - 90) < 15:
                angles.append(deg - 90)
    if len(angles) > 0:
        median_angle = round(np.median(angles) * 2) / 2.0
        (h, w) = cropped_display.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(cropped_display, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cropped_display.copy()

def enhance_image(gray_display):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray_display)

def smart_trim(image, threshold=100, max_trim=60, white_ratio=0.80, lookahead=30):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top, bottom, left, right = 0, h, 0, w

    for y in range(max_trim):
        row = gray[y, :]
        if np.sum(row > threshold) / w < white_ratio:
            if np.mean(np.sum(gray[y+1:y+1+lookahead, :] > threshold, axis=1) / w) < white_ratio:
                break
        top += 1

    for y in range(h - 1, h - max_trim - 1, -1):
        row = gray[y, :]
        if np.sum(row > threshold) / w < white_ratio:
            if np.mean(np.sum(gray[max(0, y-lookahead):y, :] > threshold, axis=1) / w) < white_ratio:
                break
        bottom -= 1

    for x in range(max_trim):
        col = gray[:, x]
        if np.sum(col > threshold) / h < white_ratio:
            if np.mean(np.sum(gray[:, x+1:x+1+lookahead] > threshold, axis=0) / h) < white_ratio:
                break
        left += 1

    for x in range(w - 1, w - max_trim - 1, -1):
        col = gray[:, x]
        if np.sum(col > threshold) / h < white_ratio:
            if np.mean(np.sum(gray[:, max(0, x-lookahead):x] > threshold, axis=0) / h) < white_ratio:
                break
        right -= 1

    return image[top:bottom, left:right] if right > left and bottom > top else image

def normalize_height(image, target_height=100):
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

def binarize_and_clean(normalized):
    blurred_bg = cv2.GaussianBlur(normalized, (35, 35), 0)
    flattened = cv2.subtract(normalized, blurred_bg)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
    t_strict = threshold_sauvola(flattened, window_size=25, k=-0.1)
    t_soft = threshold_sauvola(flattened, window_size=25, k=0.5)

    marker = (flattened > t_strict)
    mask = (flattened > t_soft)
    marker = remove_small_objects(marker, min_size=150)
    marker = binary_opening(marker, disk(2))
    marker = binary_opening(marker, disk(1))
    marker = binary_erosion(marker, disk(1))
    marker = remove_small_objects(marker, min_size=50)

    mask = remove_small_objects(mask, min_size=150)
    mask = binary_opening(mask, disk(1))
    recon = reconstruction(seed=marker, mask=mask, method='dilation')
    opened = binary_opening(recon, disk(1))
    cleaned = remove_small_objects(opened, min_size=150)
    return (cleaned.astype(np.uint8)) * 255

def split_into_digits(final_result, n_digits=5):
    h, w = final_result.shape
    digit_width = w // n_digits
    digit_regions = []

    for i in range(n_digits):
        x_start = i * digit_width
        x_end = (i + 1) * digit_width if i < n_digits - 1 else w
        digit_crop = final_result[:, x_start:x_end]
        digit_regions.append(digit_crop)

    return digit_regions

def extract_preserved_digit(original_segment, min_y=10, y_margin=20, max_y_ratio=0.85):
    original_bin = (original_segment > 0).astype(np.uint8) * 255

    def remove_horizontal_lines(img, line_min_width=20):
        inverted = 255 - img
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
        detect_horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        return cv2.subtract(img, detect_horizontal)

    segment = remove_horizontal_lines(original_bin)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(segment, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hoogte = original_bin.shape[0]
    max_y_abs = int(hoogte * max_y_ratio)
    valid_contours = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        midden = y + h // 2
        area = cv2.contourArea(cnt)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        if min_y <= midden <= max_y_abs and h > 10 and w > 5 and extent > 0.2 and 0.1 < w / h < 3.0:
            valid_contours.append(cnt)

    if not valid_contours and contours:
        valid_contours = [max(contours, key=cv2.contourArea)]
    if not valid_contours:
        return np.zeros_like(original_bin)

    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    y_start = max(0, y - y_margin)
    y_end = min(original_bin.shape[0], y + h + y_margin)
    x_start = max(0, x - 5)
    x_end = min(original_bin.shape[1], x + w + 5)

    mask = np.zeros_like(original_bin)
    mask[y_start:y_end, x_start:x_end] = 1
    final_digit = (original_bin * mask).astype(np.uint8)

    contours_final, _ = cv2.findContours(final_digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_final:
        return np.zeros_like(original_bin)

    best_contour = max(contours_final, key=cv2.contourArea)
    result = np.zeros_like(original_bin)
    cv2.drawContours(result, [best_contour], -1, 255, thickness=cv2.FILLED)
    return np.where((result == 255) & (original_bin == 255), 255, 0).astype(np.uint8)
