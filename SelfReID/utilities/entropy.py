import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Refresh folder if not exist
def reset_dir(path:str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Extract timestamp from path, "./<root>/<timestamp>/<mask>.jpeg"
def find_timestamp(path:str) -> int:
    return int(path.split("/")[-2])

# Generate cluster_id:[images] dictionary for plotting histograms
def convert_from_csv(paths:list, cluster_ids:list) -> dict:
    result = {}
    assert len(paths) == len(cluster_ids)
    for path, cluster_id in zip(paths, cluster_ids):
        if cluster_id not in result.keys():
            result.update({cluster_id:[path]})
        else:
            result[cluster_id].append(path)
    return result

# Plot histogram from a dictionary index, and save it in destined path
def plot_histogram(timestamp_ints:list, cluster_id:int, dest:str):
    reset_dir(dest)
    plt.hist(timestamp_ints, bins=np.arange(min(x_item), max(x_item) + 1, 1), edgecolor='black')
    plt.xticks(np.arange(min(x_item), max(x_item) + 1, int((max(x_item) + 1)*0.1)))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Add titles and labels
    plt.title(f'Histogram of Cluster ID: {str(cluster_id).zfill(3)}')
    plt.xlabel('Timestamp ID')
    plt.ylabel('Frequency')

    # Save the plot
    plt.savefig(os.path.join(dest, "histogram.png"))
    plt.close()

# Sort and save images into destined folder, FORMAT: <root>/<cluster_id>/items/<items>.jpeg
def sort_by_dict(result:dict, dest:str):
    reset_dir(dest)
    for key, value in result.items():
        cluster_name = str(key).zfill(3) if key != -1 else "Noise"
        cluster_folder = os.path.join(dest, cluster_name)
        cluster_folder_item = os.path.join(cluster_folder, "items")
        reset_dir(cluster_folder_item)
        timestamp_dict = {}
        for image in value:
            timestamp = find_timestamp(image)
            if timestamp not in timestamp_dict.keys():
                timestamp_dict.update({timestamp:1})
            else:
                timestamp_dict[timestamp] += 1
            shutil.copy(image, os.path.join(cluster_folder_item, f"from_{timestamp}_{timestamp_dict[timestamp]}.jpeg"))

def cluster_entropy(target_list:list, N:int, noise_penalty:float=1.0, noise_power:float=2.0) -> float:
    """
    Computes entropy-like score without needing the reference list explicitly.

    Parameters:
    - target_list: List[int], values in [-1, N-1]
    - N: int, number of expected unique values (0 to N-1)
    - noise_penalty: float, base penalty per -1
    - noise_power: float > 1, controls penalty growth for noise

    Returns:
    - float, entropy-like penalty score

    Notes:
    - Currently using abs(value - 1), L1-norm values, may adjust to L2-norm
    - Weighted error addition over noise(-1), may raise the weight to punish the noises.
    """
    counter = Counter(target_list)
    score = 0

    # Deviation from ideal frequency of 1 for values 0 to N-1
    for k in range(N):
        f_k = counter.get(k, 0)
        score += abs(f_k - 1)

    # Nonlinear penalty for -1 noise
    f_noise = counter.get(-1, 0)
    score += noise_penalty * (f_noise ** noise_power)

    return score

def sum_entropy(csv_root:str, num_frames:int) -> float:

    df = pd.read_csv(csv_root, usecols=['image_path', 'cluster_id'])

    image_paths = df['image_path'].tolist()
    cluster_ids = df['cluster_id'].tolist()

    entry = convert_from_csv(image_paths, cluster_ids)
    sum_entropy = 0.0
    # sort_by_dict(entry, dest=dest)
    for key, value in entry.items():
        cluster_name = str(key).zfill(3) if key != -1 else "Noise"
        x_item = [find_timestamp(item) for item in value]
        if key == -1:
            continue
        entropy = cluster_entropy(x_item, N=num_frames, noise_penalty=0)
        sum_entropy += entropy

    return sum_entropy

def plot_dunn_index(source:str):
    """
    Plot Dunn Index in respect to Epoch, which is found within <source>/drift_metrics_<epoch>.csv
    """
    csv_items = [item for item in os.listdir(source) if item.endswith(".csv")]
    epochs = [int(item.split("_")[-1][:-4]) for item in csv_items]
    dunn_indexes = []
    for csv_item in csv_items:
        df = pd.read_csv(os.path.join(source, csv_item), usecols=['dunn_index'])
        dunn_indexes.append(df['dunn_index'].tolist()[0])

    plt.plot(epochs, dunn_indexes)
    plt.title("Dunn Index with respect to Training Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Dunn Index")

    plt.show()
