# classificatie.py
import numpy as np
import cv2
import tensorflow as tf
import torch
# from torch_geometric.utils import remove_self_loops
from gnn_utils_pure import (
    preprocess_image,
    skeleton_to_graph,
    project_graph_to_grid,
    graph_to_pyg_data,
    reconstruct_edges,
    filter_directe_buren,
    edge_index_to_image,
    remove_self_loops,
    normalize_adjacency,
    edge_index_to_adj,

)

interpreter = tf.lite.Interpreter(model_path="models/classificatie_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
reconstructie_model = torch.jit.load("models/vgae_pure_scripted.pt", map_location="cpu")
reconstructie_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify_with_tflite(final_digits, confidence_threshold=0.90):
    resized_digits = [cv2.resize(d, (64, 64), interpolation=cv2.INTER_AREA) for d in final_digits]
    input_digits = np.stack(resized_digits).astype(np.float32) / 255.0
    input_digits = input_digits[..., np.newaxis]

    voorspelde_reeks = []
    global slechte_cijfers, slechte_cijfer_indexen
    slechte_cijfers = []
    slechte_cijfer_indexen = []

    for i, digit in enumerate(input_digits):
        input_data = np.expand_dims(digit, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        prediction = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        if confidence >= confidence_threshold:
            voorspelde_reeks.append((i, str(prediction)))
            print(f"[✓] Cijfer {i+1}: {prediction} ({confidence:.2%})")
        else:
            slechte_cijfers.append(resized_digits[i])
            slechte_cijfer_indexen.append(i)
            print(f"[✗] Cijfer {i+1}: {prediction} ({confidence:.2%}) → door naar GNN")

    return voorspelde_reeks

def classify_with_gnn(final_digits, voorspelde_reeks):
    for idx, cijfer in enumerate(slechte_cijfers):
        print(f"\n--- GNN verwerking cijfer {idx+1} ---")
        skeleton = preprocess_image(cijfer)
        G = skeleton_to_graph(skeleton)
        G_proj = project_graph_to_grid(G, grid_size=8)
        data = graph_to_pyg_data(G_proj, grid_size=8).to(device)

        reconstructed_edges = reconstruct_edges(data, reconstructie_model, threshold=0.8)
        reconstructed_edges, _ = remove_self_loops(reconstructed_edges)
        reconstructed_edges = filter_directe_buren(reconstructed_edges)

        digit_img = edge_index_to_image(reconstructed_edges)
        cnn_input = digit_img.astype(np.float32) / 255.0
        cnn_input = cnn_input.reshape(1, 64, 64, 1)

        interpreter.set_tensor(input_details[0]['index'], cnn_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        prediction = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        originele_index = slechte_cijfer_indexen[idx]
        voorspelde_reeks.append((originele_index, str(prediction)))

        print(f"→ GNN → CNN voorspelling: {prediction} (zekerheid: {confidence:.2%})")

    return voorspelde_reeks
