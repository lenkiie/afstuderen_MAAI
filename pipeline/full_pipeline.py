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
import networkx as nx
from torch_geometric.utils import remove_self_loops
from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings("ignore")





model_path = r"C:\Users\lenka\OneDrive\Documenten\Afstuderen Master\pipeline\detectie_model.tflite"
image_path = r"C:\Users\lenka\OneDrive\Documenten\Afstuderen Master\detectie model\UFPR-AMR Dataset\training\meter0217.jpg"

# Detectie model
################# DETECTIE ###############

# Laad model
model = YOLO(model_path, task="detect")

# Laad originele afbeelding
original_img = cv2.imread(image_path)
orig_h, orig_w = original_img.shape[:2]

# Predictie
results = model.predict(source=image_path, save=False, imgsz=160)

# Begin van verbeterde logica (deze vervangt je oude box-handling)
boxes = results[0].boxes

if boxes is not None and len(boxes) > 0:
    # Zet alle boxen en hun confidence in lijst
    box_data = []
    for box in boxes:
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
        xyxy = box.xyxy[0].cpu().numpy()
        box_data.append((conf, cls, xyxy))

    # Sorteer op confidence (hoogste eerst)
    box_data.sort(reverse=True, key=lambda x: x[0])

    if len(box_data) > 0:
        _, _, box = box_data[0]
        x1, y1, x2, y2 = map(int, box)

        # Clamp de box binnen afbeelding
        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w, x2))
        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h, y2))

        cropped_display = original_img[y1:y2, x1:x2]

    else:
        print("Geen geschikte boxen gevonden.")
else:
    print("Geen objecten gedetecteerd.")

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
        print("⚠️ Ongeldige smart crop — origineel teruggegeven.")
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
n_digits = 5

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
voorspelde_reeks = []  # lijst van tuples (index, voorspeld cijfer)

# Laad TFLite model
interpreter = tf.lite.Interpreter(model_path="classificatie_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Verwerk input afbeeldingen
resized_digits = [cv2.resize(d, (64, 64), interpolation=cv2.INTER_AREA) for d in final_digits]  # of final_digits
input_digits = np.stack(resized_digits).astype(np.float32) / 255.0
input_digits = input_digits[..., np.newaxis]  # (N, 64, 64, 1)

confidence_threshold = 0.90

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
        print(f"[✗] Cijfer {i+1}: {prediction} ({confidence:.2%}) → door naar GNN")



# ----------------------------
# Preprocessing helpers
# ----------------------------

def resize_with_padding(img, target_size=64, inner_size=60):
    h, w = img.shape
    scale = min(inner_size / h, inner_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = resize(img, (new_h, new_w), mode='constant', anti_aliasing=True)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(resized, ((top, bottom), (left, right)), mode='constant', constant_values=0)

def preprocess_image(img, target_size=64, inner_size=60):
    img = img / 255.0 if img.max() > 1 else img
    threshold = threshold_otsu(img)
    binary = img > threshold

    # Crop
    rows, cols = np.any(binary, axis=1), np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    margin = 1
    rmin = max(rmin - margin, 0)
    rmax = min(rmax + margin, img.shape[0] - 1)
    cmin = max(cmin - margin, 0)
    cmax = min(cmax + margin, img.shape[1] - 1)
    cropped = img[rmin:rmax+1, cmin:cmax+1]

    # Resize en binarisatie
    resized = resize_with_padding(cropped, target_size, inner_size)
    smoothed = gaussian(resized, sigma=1.0)
    binary = smoothed > threshold_otsu(smoothed)

    # Skeletonisatie
    skeleton1 = skeletonize(binary)
    dilated = binary_dilation(skeleton1)
    skeleton = skeletonize(dilated)
    return skeleton


def skeleton_to_graph(skeleton):
    coords = np.argwhere(skeleton)
    G = nx.Graph()
    for y, x in coords:
        G.add_node((x, y))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                    if skeleton[ny, nx_]:
                        G.add_edge((x, y), (nx_, ny))
    return G

def project_graph_to_grid(G, grid_size=8, image_size=64):
    cell_size = image_size // grid_size
    G_projected = nx.Graph()
    point_to_cell = {}
    added_edges = set()

    for x, y in G.nodes:
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        cell = (cx, cy)
        point_to_cell[(x, y)] = cell
        G_projected.add_node(cell)

    for u, v in G.edges:
        cu = point_to_cell.get(u)
        cv = point_to_cell.get(v)
        if cu and cv and cu != cv:
            edge = tuple(sorted((cu, cv)))
            if edge not in added_edges:
                G_projected.add_edge(*edge)
                added_edges.add(edge)

    return G_projected

def graph_to_pyg_data(G, grid_size=8, label=None):
    num_nodes = grid_size * grid_size
    node_index_map = { (x, y): y * grid_size + x for x in range(grid_size) for y in range(grid_size) }

    # Alleen identity-matrix als features (one-hot vector per knoop)
    features = []
    for y in range(grid_size):
        for x in range(grid_size):
            idx = y * grid_size + x
            one_hot = torch.zeros(num_nodes)
            one_hot[idx] = 1.0
            features.append(one_hot)

    x = torch.stack(features)  # Vorm: [64, 64]

    edge_index = []
    for (u, v) in G.edges:
        uid = node_index_map.get(u)
        vid = node_index_map.get(v)
        if uid is not None and vid is not None:
            edge_index.append([uid, vid])
            edge_index.append([vid, uid])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data

class PyGSkeletonDataset(Dataset):
    def __init__(self, images, labels=None, grid_size=8):
        self.images = images
        self.labels = labels
        self.grid_size = grid_size
        self.graphs = [self._image_to_graph(img, i) for i, img in enumerate(images)]

    def _image_to_graph(self, img, idx):
        skeleton = preprocess_image(img)
        G = skeleton_to_graph(skeleton)
        G_proj = project_graph_to_grid(G, grid_size=self.grid_size)
        label = self.labels[idx] if self.labels is not None else None
        return graph_to_pyg_data(G_proj, grid_size=self.grid_size, label=label)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reconstruct_edges(data, model, threshold=0.5):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)

    # Genereer alle mogelijke nodeparen in grid
    num_nodes = data.num_nodes
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
    full_edge_index = torch.stack([row.flatten(), col.flatten()], dim=0).to(device)

    # Decode scores (edge probabilities)
    scores = model.decode(z, full_edge_index).sigmoid()

    # Houd alleen edges boven drempel
    keep = scores > threshold
    edge_index_reconstructed = full_edge_index[:, keep]

    return edge_index_reconstructed.cpu()


def zijn_directe_buren(u, v):
    """Controleer of twee knopen directe buren zijn op een 8x8 raster, inclusief diagonalen."""
    x1, y1 = u % 8, u // 8
    x2, y2 = v % 8, v // 8
    return max(abs(x1 - x2), abs(y1 - y2)) == 1 and (u != v)


def filter_directe_buren(edge_index):
    """Behoud alleen edges tussen directe buren."""
    filtered_edges = []
    for u, v in edge_index.t().tolist():
        if zijn_directe_buren(u, v):
            filtered_edges.append((u, v))
    return torch.tensor(filtered_edges).t()


reconstructie_model = torch.jit.load(r"C:\Users\lenka\OneDrive\Documenten\Afstuderen Master\pipeline\vgae_scripted.pt", map_location="cpu")
reconstructie_model.eval()


def edge_index_to_image(edge_index, image_size=64, grid_size=8, line_thickness=3):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    cell_size = image_size // grid_size

    positions = {i: (int((i % grid_size + 0.5) * cell_size),
                     int((i // grid_size + 0.5) * cell_size))
                 for i in range(grid_size * grid_size)}

    for u, v in edge_index.t().tolist():
        pt1 = positions[u]
        pt2 = positions[v]
        cv2.line(img, pt1, pt2, color=255, thickness=line_thickness)

    return img

for idx, cijfer in enumerate(slechte_cijfers):
    print(f"\n--- GNN verwerking cijfer {idx+1} ---")

    # GNN-preprocessing
    # skeleton = preprocess_image(cijfer)
    skeleton = skeleton = preprocess_image(cijfer, inner_size=48)
    G = skeleton_to_graph(skeleton)
    G_proj = project_graph_to_grid(G, grid_size=8)
    # G_proj = project_graph_to_grid(G, grid_size=8, image_size=64)

    data = graph_to_pyg_data(G_proj, grid_size=8).to(device)

    # Reconstruct edges
    reconstructed_edges = reconstruct_edges(data, reconstructie_model, threshold=0.8)
    reconstructed_edges, _ = remove_self_loops(reconstructed_edges)
    reconstructed_edges = filter_directe_buren(reconstructed_edges)

    digit_img = edge_index_to_image(reconstructed_edges)

    # Naar CNN sturen
    cnn_input = digit_img.astype(np.float32) / 255.0
    cnn_input = cnn_input.reshape(1, 64, 64, 1)
    interpreter.set_tensor(input_details[0]['index'], cnn_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    # Bewaar met originele index
    originele_index = slechte_cijfer_indexen[idx]
    voorspelde_reeks.append((originele_index, str(prediction)))

    print(f"→ GNN → CNN voorspelling: {prediction} (zekerheid: {confidence:.2%})")


# Sorteer op oorspronkelijke index en toon de reeks
voorspelde_reeks.sort(key=lambda x: x[0])
cijfer_string = ''.join([digit for _, digit in voorspelde_reeks])

print(f"Output: {cijfer_string}")
