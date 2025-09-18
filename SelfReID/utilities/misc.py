import os
import csv
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, jaccard_score, pairwise_distances
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

def save_log_csv(log_dict:dict, path:str):
    reset_dirs(path)
    with open(os.path.join(path, "score_log.csv"), 'w') as f:
        w = csv.writer(f)
        w.writerow(log_dict.keys())
        w.writerow(log_dict.values())

def reset_dirs(path:str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def compute_ssim(img1, img2):
    img1 = img1.convert("L").resize((128, 128))
    img2 = img2.convert("L").resize((128, 128))
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return ssim(arr1, arr2)

def save_cluster_csv(labels, paths, output_csv="dbscan_results.csv"):
    assert len(labels) == len(paths), "Labels and paths must be the same length!"
    
    df = pd.DataFrame({
        'label': labels,
        'path': paths
    })

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} entries to {output_csv}")

def compute_cluster_purity_ssim(cluster_indices, image_paths):
    imgs = [Image.open(image_paths[i]).convert("RGB") for i in cluster_indices]
    anchor = imgs[0]
    similarities = [compute_ssim(anchor, img) for img in imgs]
    return np.mean(similarities)

def ClusterMetrics(embeddings, image_paths, labels):
    cluster_purities = []
    unique_labels = [label for label in set(labels) if label != -1]

    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) > 1:
            purity = compute_cluster_purity_ssim(cluster_indices, image_paths)
            cluster_purities.append(purity)

    overall_purity = np.mean(cluster_purities)

    # Step 2: Filter out noise points (label == -1)
    valid_indices = labels != -1
    filtered_embeddings = embeddings[valid_indices]
    filtered_labels = labels[valid_indices]

    if len(set(filtered_labels)) < 2:
        print("Not enough clusters formed to evaluate clustering quality.")
        return None

    # Step 3: Compute scores
    silhouette = silhouette_score(filtered_embeddings, filtered_labels)
    db_index = davies_bouldin_score(filtered_embeddings, filtered_labels)
    ch_index = calinski_harabasz_score(filtered_embeddings, filtered_labels)
    dunn_index = compute_dunn_index(filtered_embeddings, filtered_labels)

    return {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Index": ch_index,
        "Num Clusters": len(set(filtered_labels)),
        "Num Noise Points": np.sum(labels == -1),
        "Overall visual cluster purity (SSIM-based)": overall_purity,
        "Dunn Index": dunn_index,
    }
    # df.to_csv(save_path, index=False)

def compute_centroid_drift(embedding, label, all_embeddings, all_labels):
    if label == -1:
        return None
    cluster_points = all_embeddings[all_labels == label]
    if len(cluster_points) < 2:
        return None
    centroid = cluster_points.mean(axis=0)
    return euclidean(embedding, centroid)

def local_cluster_agreement(umap_2d, labels, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(umap_2d)
    neighbors = nn.kneighbors(umap_2d, return_distance=False)[:, 1:]
    scores = []
    for i, nbrs in enumerate(neighbors):
        if labels[i] == -1:
            scores.append(None)
        else:
            same_cluster = [labels[j] == labels[i] for j in nbrs]
            scores.append(np.mean(same_cluster))
    return scores

def compute_jaccard_per_point(old_labels, new_labels):
    scores = []
    for i in range(len(old_labels)):
        if old_labels[i] == -1 or new_labels[i] == -1:
            scores.append(None)
        else:
            a = (old_labels == old_labels[i]).astype(int)
            b = (new_labels == new_labels[i]).astype(int)
            score = jaccard_score(a, b)
            scores.append(score)
    return scores

def ClusteringDrifts(embeddings, labels_v1, labels_v2, paths, umap_embd, save_path):
    drift_scores = [
        compute_centroid_drift(embeddings[i], labels_v2[i], embeddings, labels_v2)
        for i in range(len(embeddings))
    ]
    jaccard_scores = compute_jaccard_per_point(labels_v1, labels_v2)
    agreement_scores = local_cluster_agreement(umap_embd, labels_v2)

    df = pd.DataFrame({
        "index": list(range(len(embeddings))),
        "image_path": paths,
        "cluster_id": labels_v2,
        "centroid_drift": drift_scores,
        "local_agreement": agreement_scores,
        "jaccard_consistency": jaccard_scores,
    })

    df.to_csv(save_path, index=False)
    # print("Saved cluster drift metrics to cluster_drift_metrics.csv")

def compute_dunn_index(data, labels):
    unique_clusters = np.unique(labels)
    num_clusters = len(unique_clusters)

    if num_clusters < 2:
        raise ValueError("Dunn Index requires at least two clusters.")

    # Calculate inter-cluster distances
    inter_cluster_distances = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            cluster_i = data[labels == unique_clusters[i]]
            cluster_j = data[labels == unique_clusters[j]]
            dist = pairwise_distances(cluster_i, cluster_j).min()
            inter_cluster_distances.append(dist)

    # Calculate intra-cluster distances
    intra_cluster_distances = []
    for cluster in unique_clusters:
        cluster_points = data[labels == cluster]
        if len(cluster_points) > 1:
            dist = pairwise_distances(cluster_points).max()
            intra_cluster_distances.append(dist)

    # Compute Dunn Index
    min_inter_cluster_distance = min(inter_cluster_distances)
    max_intra_cluster_distance = max(intra_cluster_distances)
    dunn = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn

# Hungarian-matched accuracy
def hungarian_match_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize
    return cm[row_ind, col_ind].sum() / cm.sum()

def evaluate_clustering_with_gt(y_pred, y_true):
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter out -1 (noise) from y_pred
    valid_mask = y_pred != -1
    y_true_filtered = y_true[valid_mask]
    y_pred_filtered = y_pred[valid_mask]

    if y_true_filtered.size == 0:
        return None

    # Compute clustering metrics
    ari = adjusted_rand_score(y_true_filtered, y_pred_filtered)
    ami = adjusted_mutual_info_score(y_true_filtered, y_pred_filtered)
    nmi = normalized_mutual_info_score(y_true_filtered, y_pred_filtered)
    h_acc = hungarian_match_accuracy(y_true_filtered, y_pred_filtered)

    return {
        "adjusted_rand_index": ari,
        "adjusted_mutual_info": ami,
        "normalized_mutual_info": nmi,
        "hungarian_accuracy": h_acc,
        "num_data_points": len(y_true_filtered),
        "num_predicted_clusters": len(set(y_pred_filtered)),
        "num_true_classes": len(set(y_true_filtered)),
        "noise_ratio": 1.0 - len(y_true_filtered) / len(y_pred)
    }
