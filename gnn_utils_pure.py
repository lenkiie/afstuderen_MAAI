# gnn_utils.py
import numpy as np
import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Data
import networkx as nx
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import skeletonize, binary_dilation
from skimage.transform import resize
import cv2

class SimpleData:
    def __init__(self, x, edge_index, y=None):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.num_nodes = x.shape[0]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self


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
    rows, cols = np.any(binary, axis=1), np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    margin = 1
    rmin = max(rmin - margin, 0)
    rmax = min(rmax + margin, img.shape[0] - 1)
    cmin = max(cmin - margin, 0)
    cmax = min(cmax + margin, img.shape[1] - 1)
    cropped = img[rmin:rmax+1, cmin:cmax+1]
    resized = resize_with_padding(cropped, target_size, inner_size)
    smoothed = gaussian(resized, sigma=1.0)
    binary = smoothed > threshold_otsu(smoothed)
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
    features = []
    for y in range(grid_size):
        for x in range(grid_size):
            idx = y * grid_size + x
            one_hot = torch.zeros(num_nodes)
            one_hot[idx] = 1.0
            features.append(one_hot)
    x = torch.stack(features)
    edge_index = []
    for (u, v) in G.edges:
        uid = node_index_map.get(u)
        vid = node_index_map.get(v)
        if uid is not None and vid is not None:
            edge_index.append([uid, vid])
            edge_index.append([vid, uid])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # data = Data(x=x, edge_index=edge_index)
    data = SimpleData(x=x, edge_index=edge_index)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data

def reconstruct_edges(data, model, threshold=0.5):
    model.eval()
    data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    num_nodes = data.num_nodes
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
    full_edge_index = torch.stack([row.flatten(), col.flatten()], dim=0).to(data.x.device)
    scores = model.decode(z, full_edge_index).sigmoid()
    keep = scores > threshold
    edge_index_reconstructed = full_edge_index[:, keep]
    return edge_index_reconstructed.cpu()

def zijn_directe_buren(u, v):
    x1, y1 = u % 8, u // 8
    x2, y2 = v % 8, v // 8
    return max(abs(x1 - x2), abs(y1 - y2)) == 1 and (u != v)

def filter_directe_buren(edge_index):
    filtered_edges = []
    for u, v in edge_index.t().tolist():
        if zijn_directe_buren(u, v):
            filtered_edges.append((u, v))
    return torch.tensor(filtered_edges).t()

def remove_self_loops(edge_index):
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask], None

def edge_index_to_image(edge_index, image_size=64, grid_size=8, line_thickness=3):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    cell_size = image_size // grid_size
    positions = {i: (int((i % grid_size + 0.5) * cell_size), int((i // grid_size + 0.5) * cell_size)) for i in range(grid_size * grid_size)}
    for u, v in edge_index.t().tolist():
        pt1 = positions[u]
        pt2 = positions[v]
        cv2.line(img, pt1, pt2, color=255, thickness=line_thickness)
    return img

def normalize_adjacency(adj):
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    deg = adj.sum(1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


# --- Converteer edge_index naar dense adjacency ---
def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def reconstruct_edges(data, model, threshold=0.9):
    model.eval()

    # Genormaliseerde adjacency matrix
    adj = edge_index_to_adj(data.edge_index, num_nodes=data.num_nodes)
    adj_norm = normalize_adjacency(adj)

    # Genereer alle mogelijke edges
    num_nodes = data.num_nodes
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
    full_edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)

    with torch.no_grad():
        scores = model(data.x, adj_norm, full_edge_index)
        probs = torch.sigmoid(scores)

    keep = probs > threshold
    edge_index_reconstructed = full_edge_index[:, keep]
    return edge_index_reconstructed.cpu()
