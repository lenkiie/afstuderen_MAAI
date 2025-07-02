from skimage.filters import threshold_sauvola
from skimage.morphology import reconstruction, remove_small_objects, binary_opening, disk, binary_erosion
import numpy as np
import tensorflow as tf
import cv2
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import networkx as nx
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import skeletonize, binary_dilation
from skimage.transform import resize
from torch_geometric.utils import dense_to_sparse
import networkx as nx
from torch_geometric.utils import remove_self_loops
import cv2
import warnings
import torch
from torch_geometric.nn import VGAE
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge
import matplotlib.pyplot as plt 
import os


warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.abspath(__file__))  # map van het script
model_path = os.path.join(project_root, "models", "detectie_model.tflite")
image_paths = [
    os.path.join(project_root, "data", "new_image1.jpg")
]


n_digits = 6
# Detectie model
################# DETECTIE (zonder YOLO) ###############
for image_path in image_paths:
    # === LAAD MODEL ===
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    # === LAAD EN VERWERK AFBEELDING ===
    original_img = cv2.imread(image_path)
    orig_h, orig_w = original_img.shape[:2]
    print(f"[DEBUG] Originele afbeelding: breedte={orig_w}, hoogte={orig_h}")

    image_resized = cv2.resize(original_img, (input_width, input_height))
    input_tensor = np.expand_dims(image_resized.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # === VERWERK OUTPUT ===
    output = interpreter.get_tensor(output_details[0]['index'])  # [1, 5, 525] of vergelijkbaar
    output = np.squeeze(output).T  # [525, 5]

    detections = []
    threshold = 0.2

    for row in output:
        xc, yc, w, h, conf = row
        if conf >= threshold:
            x1 = int((xc - w / 2) * orig_w)
            y1 = int((yc - h / 2) * orig_h)
            x2 = int((xc + w / 2) * orig_w)
            y2 = int((yc + h / 2) * orig_h)

            x1, x2 = max(0, x1), min(orig_w, x2)
            y1, y2 = max(0, y1), min(orig_h, y2)
            detections.append((conf, (x1, y1, x2, y2)))

    # === SELECTEER BESTE DETECTIE ===
    if detections:
        detections.sort(reverse=True, key=lambda x: x[0])
        _, (x1, y1, x2, y2) = detections[0]
        cropped_display = original_img[y1:y2, x1:x2]
        plt.imshow(cv2.cvtColor(cropped_display, cv2.COLOR_BGR2RGB))
        plt.title("Beste gecropt display")
        plt.axis("off")
        plt.show()
    else:
        print("Geen objecten gedetecteerd.")
        cropped_display = original_img.copy()
        


    ############## ROTEREN ##############################3
    import numpy as np

    # === Canny edge detection ===
    edges = cv2.Canny(cropped_display, 50, 150, apertureSize=3)

    # === Hough Line Transform ===
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
        rotated_display = cv2.warpAffine(cropped_display, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
    else:
        print("Geen rotatie nodig.")
        rotated_display = cropped_display.copy()

    ################################# GRIJS ######################
    gray_display = cv2.cvtColor(rotated_display, cv2.COLOR_BGR2GRAY)


    ######################### RUIS ETC. ###################
    # Verwijder ruis & verhoog contrast
    # blurred = cv2.medianBlur(gray_display, 3)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_display)


    ########################### TRIM ##################################
    def smart_trim(image, threshold=200, max_trim=50, white_ratio=0.90, lookahead=5):
        """
        Trim witte randen met pixelanalyse per rij/kolom en lookahead om vals stoppen te vermijden.
        """
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        top, bottom = 0, h
        left, right = 0, w

        # Bovenkant
        for y in range(max_trim):
            row = gray[y, :]
            ratio = np.sum(row > threshold) / w
            if ratio < white_ratio:
                lookahead_rows = gray[y+1:y+1+lookahead, :]
                lookahead_ratios = np.mean(np.sum(lookahead_rows > threshold, axis=1) / w)
                if lookahead_ratios < white_ratio:
                    break
            top += 1

        # Onderkant
        for y in range(h - 1, h - max_trim - 1, -1):
            row = gray[y, :]
            ratio = np.sum(row > threshold) / w
            if ratio < white_ratio:
                lookahead_rows = gray[max(0, y-lookahead):y, :]
                lookahead_ratios = np.mean(np.sum(lookahead_rows > threshold, axis=1) / w)
                if lookahead_ratios < white_ratio:
                    break
            bottom -= 1

        # Linkerkant
        for x in range(max_trim):
            col = gray[:, x]
            ratio = np.sum(col > threshold) / h
            if ratio < white_ratio:
                lookahead_cols = gray[:, x+1:x+1+lookahead]
                lookahead_ratios = np.mean(np.sum(lookahead_cols > threshold, axis=0) / h)
                if lookahead_ratios < white_ratio:
                    break
            left += 1

        # Rechterkant
        for x in range(w - 1, w - max_trim - 1, -1):
            col = gray[:, x]
            ratio = np.sum(col > threshold) / h
            if ratio < white_ratio:
                lookahead_cols = gray[:, max(0, x-lookahead):x]
                lookahead_ratios = np.mean(np.sum(lookahead_cols > threshold, axis=0) / h)
                if lookahead_ratios < white_ratio:
                    break
            right -= 1

        if right > left and bottom > top:
            return image[top:bottom, left:right]
        else:
            print("Ongeldige smart crop — origineel teruggegeven.")
            return image

    trimmed = smart_trim(enhanced, threshold=100, max_trim=60, white_ratio=0.80, lookahead=30)


    #################################### NORMALISEREN ########################
    def normalize_height(image, target_height=100):
        h, w = image.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        return resized

    normalized = normalize_height(trimmed, target_height=100)



    # === Parameters ===
    window = 25
    k_strict = -0.1
    k_soft = 0.5
    min_marker_size = 150
    min_final_size = 150
    blur_size = 35

    # === 1. Flatten achtergrond ===
    blurred_bg = cv2.GaussianBlur(normalized, (blur_size, blur_size), 0)

    flattened = cv2.subtract(normalized, blurred_bg)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)




    # === 2. Bereken thresholds ===
    t_strict = threshold_sauvola(flattened, window_size=window, k=k_strict)
    t_soft = threshold_sauvola(flattened, window_size=window, k=k_soft)

    # === 3. Binariseer ===
    marker_raw = (flattened > t_strict)
    mask_raw = (flattened > t_soft)

    # === 4. Filter marker ===
    marker = remove_small_objects(marker_raw, min_size=min_marker_size)
    marker = binary_opening(marker, disk(2))
    marker = binary_opening(marker, disk(1))
    marker = binary_erosion(marker, disk(1))
    # marker = binary_erosion(marker, disk(1))
    # optioneel
    marker = remove_small_objects(marker, min_size=50)
    # === 5. Filter mask ===
    mask = remove_small_objects(mask_raw, min_size=150)
    mask = binary_opening(mask, disk(1))

    # === 6. Reconstructie ===
    recon = reconstruction(seed=marker, mask=mask, method='dilation')

    # === 7. Morph. opening ===
    opened = binary_opening(recon, disk(1))

    # === 8. Laatste ruisfiltering ===
    cleaned = remove_small_objects(opened, min_size=min_final_size)

    # === 9. Final naar uint8 ===
    final_result = (cleaned.astype(np.uint8)) * 255


    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Functie om horizontale lijnen te verwijderen
    def remove_horizontal_lines(img, line_min_width=20):
        inverted = 255 - img
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
        detect_horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        cleaned = cv2.subtract(img, detect_horizontal)
        return cleaned

    # Functie om segment te reinigen
    def clean_digit_segment(segment):
        segment = segment.copy()
        if segment.max() <= 1:
            segment = (segment * 255).astype(np.uint8)

        segment = remove_horizontal_lines(segment)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(segment, cv2.MORPH_OPEN, kernel, iterations=1)
        return cleaned

    # Functie om hoofdcomponent te extraheren
    def extract_main_component_filtered(crop_bin, y_margin=20, min_y=10):
        crop_bin = (crop_bin > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            if y >= min_y and extent > 0.2 and 0.1 < w/h < 3.0:
                valid_contours.append(cnt)

        if not valid_contours and contours:
            valid_contours = [max(contours, key=cv2.contourArea)]

        if not valid_contours:
            return np.zeros_like(crop_bin)

        main_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        y_min, y_max = y, y + h

        output = np.zeros_like(crop_bin)
        for cnt in valid_contours:
            _, y_cnt, _, h_cnt = cv2.boundingRect(cnt)
            if (y_cnt >= y_min - y_margin) and (y_cnt + h_cnt <= y_max + y_margin):
                cv2.drawContours(output, [cnt], -1, 255, thickness=cv2.FILLED)

        return output

    # Functie om uiteindelijke cijfer te extraheren
    def extract_preserved_digit(original_segment, min_y=10, y_margin=20, max_y_ratio=0.85):
        original_bin = (original_segment > 0).astype(np.uint8) * 255
        cleaned = clean_digit_segment(original_bin)
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
            if min_y <= midden <= max_y_abs and h > 10 and w > 5 and extent > 0.2 and 0.1 < w/h < 3.0:
                valid_contours.append(cnt)

        if not valid_contours and contours:
            valid_contours = [max(contours, key=cv2.contourArea)]

        if not valid_contours:
            return np.zeros_like(original_bin)

        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        mask = np.zeros_like(original_bin)
        y_start = max(0, y - y_margin)
        y_end = min(original_bin.shape[0], y + h + y_margin)
        x_start = max(0, x - 5)
        x_end = min(original_bin.shape[1], x + w + 5)
        mask[y_start:y_end, x_start:x_end] = 1

        final_digit = (original_bin * mask).astype(np.uint8)

        contours_final, _ = cv2.findContours(final_digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_final:
            return np.zeros_like(original_bin)

        best_contour = max(contours_final, key=cv2.contourArea)
        result = np.zeros_like(original_bin)
        cv2.drawContours(result, [best_contour], -1, 255, thickness=cv2.FILLED)

        return np.where((result == 255) & (original_bin == 255), 255, 0).astype(np.uint8)

    # =========================== HOOFDPROCESSING ===========================



    h, w = final_result.shape

    digit_width = w // n_digits

    digit_regions = []
    for i in range(n_digits):
        x_start = i * digit_width
        x_end = (i + 1) * digit_width if i < n_digits - 1 else w
        digit_crop = final_result[:, x_start:x_end]
        digit_regions.append(digit_crop)


    # Schoonmaken en extraheren
    final_digits = []
    for i, region in enumerate(digit_regions):
        final_digit = extract_preserved_digit(region)
        final_digits.append(final_digit)



    goede_cijfers = []
    slechte_cijfers = []
    slechte_cijfer_indexen = []
    slechte_cijfer_info = []  # (index, oude_predictie, oude_confidence)
    voorspelde_reeks = []

    # Laad TFLite model
    model_path = os.path.join("models", "final_model_full_train.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # interpreter = tf.lite.Interpreter(model_path=r"C:\Users\lenka\OneDrive\Documenten\Afstuderen Master\pipeline\final_model_full_train.tflite")
    # interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Verwerk input afbeeldingen
    resized_digits = [cv2.resize(d, (64, 64), interpolation=cv2.INTER_AREA) for d in final_digits]
    input_digits = np.stack(resized_digits).astype(np.float32) / 255.0
    input_digits = input_digits[..., np.newaxis]  # (N, 64, 64, 1)

    confidence_threshold = 0.80

    for i, digit in enumerate(input_digits):
        input_data = np.expand_dims(digit, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        prediction = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        if confidence >= confidence_threshold:
            goede_cijfers.append(resized_digits[i])
            voorspelde_reeks.append((i, str(prediction)))
            print(f"[✓] Cijfer {i+1}: {prediction} ({confidence:.2%})")
        else:
            slechte_cijfers.append(resized_digits[i])
            slechte_cijfer_indexen.append(i)
            slechte_cijfer_info.append((i, prediction, confidence))  # bewaar oude
            print(f"[✗] Cijfer {i+1}: {prediction} ({confidence:.2%})")

        

    voorspelde_reeks = []

    for i, digit in enumerate(input_digits):
        input_data = np.expand_dims(digit, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        prediction = int(np.argmax(output_data))
        voorspelde_reeks.append(str(prediction))
        print(f"Cijfer {i+1}: {prediction}")

    cijfer_string = ''.join(voorspelde_reeks)

    import csv
    import os

    csv_path = "resultaten.csv"
    header = ["zonder_reconstructie", "met_reconstructie"]
    met_reconstructie = ""  # geen reconstructie dus leeg

    bestand_bestaat = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        if not bestand_bestaat:
            writer.writerow(header)
        writer.writerow([cijfer_string, met_reconstructie])

    print(f"Output: {cijfer_string}")
