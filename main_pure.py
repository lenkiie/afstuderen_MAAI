# main.py

from processing_pure import (
    load_and_crop_digit_region,
    rotate_image,
    enhance_image,
    smart_trim,
    normalize_height,
    binarize_and_clean,
    split_into_digits,
    extract_preserved_digit
)
from classificatie_pure import (
    classify_with_tflite,
    classify_with_gnn
)
import cv2

if __name__ == "__main__":
    image_path = "data/test1.jpg"
    model_path = "models/detectie_model.tflite"

    # Stap 1: Detectie en croppen
    cropped = load_and_crop_digit_region(image_path, model_path)

    # Stap 2: Roteren
    rotated = rotate_image(cropped)

    # Stap 3: Verwerking
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    trimmed = smart_trim(enhanced)
    normalized = normalize_height(trimmed)
    cleaned = binarize_and_clean(normalized)

    # Stap 4: Splits in cijfers en extraheren
    digit_imgs = split_into_digits(cleaned)
    final_digits = [extract_preserved_digit(d) for d in digit_imgs]

    # Stap 5: Classificatie
    voorspelde_reeks = classify_with_tflite(final_digits)
    voorspelde_reeks = classify_with_gnn(final_digits, voorspelde_reeks)

    cijfer_string = ''.join([d for _, d in sorted(voorspelde_reeks, key=lambda x: x[0])])
    print(f"Output: {cijfer_string}")
