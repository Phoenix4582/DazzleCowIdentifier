import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, mutual_info_score

def KNNClusterPerformance(embeddings, labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(embeddings, labels.ravel())
    predictions = neigh.predict(embeddings)

    total = len(labels.ravel())
    # How many were correct?
    correct = (predictions == labels.ravel()).sum()
    acc = (float(correct) / total) * 100
    acc_score = neigh.score(embeddings, labels) * 100
    precision, recall, f1_score = additional_metrics(predictions, labels.ravel())
    silhouette = silhouette_score(embeddings, labels)
    variance_ratio_criterion = calinski_harabasz_score(embeddings, labels)
    db_score = davies_bouldin_score(embeddings, labels)

    return {
        "acc": acc,
        "acc_score": acc_score,
        "silhouette": silhouette,
        "vrc": variance_ratio_criterion,
        "db_score": db_score,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
    return acc, acc_score, silhouette, variance_ratio_criterion, db_score, precision, recall, f1_score


def KNNClusterConsistency(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh_train = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)
    # neigh_test = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh_train.fit(train_embeddings, train_labels)
    # neigh_test.fit(train_embeddings, train_labels.ravel())

    lbls_pred = neigh_train.predict(test_embeddings)
    # print(test_labels)
    # print(lbls_pred)
    # print(lbls_pred == test_labels.ravel())
    # lbls_test = neigh_test.predict(test_embeddings)

    # col_choice = hungarian_algorithm(train_embeddings, test_embeddings, metric='cosine_distance')

    # lbls_pred_choice = np.array([lbls_pred[choice] for choice in col_choice])
    # assert lbls_pred.shape == lbls_pred_choice.shape

    total = len(test_labels.ravel())

    # How many were correct?
    correct = (lbls_pred == test_labels.ravel()).sum()
    # correct_choice = (lbls_pred_choice == test_labels.ravel()).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100
    accuracy_score = neigh_train.score(test_embeddings, test_labels) * 100

    # Extra metrics
    ari = adjusted_rand_score(test_labels, lbls_pred)
    mutual_info = mutual_info_score(test_labels, lbls_pred)

    return {
        "acc": accuracy,
        "acc_score": accuracy_score,
        "ari": ari,
        "mutual_info": mutual_info,
    }

def additional_metrics(preds, targets, average='weighted'):
    num_classes = len(np.unique(targets))
    # labels = [f"Cow {id+1}" for id in range(num_classes)]
    labels = [id for id in range(1, num_classes+1)]
    precision = precision_score(preds, targets.ravel(), labels=labels, average=average)
    recall = recall_score(preds, targets.ravel(), labels=labels, average=average)
    f1 = f1_score(preds, targets.ravel(), labels=labels, average=average)
    return precision, recall, f1

def KNNProbe(embeddings, gt_labels, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    indices = nbrs.kneighbors(embeddings, return_distance=False)

    correct = 0
    for i in range(len(embeddings)):
        neighbor_ids = indices[i][1:]  # skip self
        neighbor_labels = gt_labels[neighbor_ids]
        majority_label = Counter(neighbor_labels).most_common(1)[0][0]
        if gt_labels[i] == majority_label:
            correct += 1

    return correct / len(embeddings)
