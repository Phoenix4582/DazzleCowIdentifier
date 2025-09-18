import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pytorch_metric_learning import miners, losses
import torchvision.transforms as T
import torchvision
from torchvision.models import *
import lightning as L
import csv
from sklearn.cluster import DBSCAN, HDBSCAN
import joblib
from itertools import product

from sklearn.preprocessing import normalize
from utilities.plots import init_umap, scatter
from utilities.misc import ClusterMetrics, ClusteringDrifts, evaluate_clustering_with_gt, save_cluster_csv
from utilities.entropy import sum_entropy
from utilities.knn import KNNClusterConsistency, KNNClusterPerformance, KNNProbe

class SelfReIDModel(L.LightningModule):
    def __init__(self,
                 backbone: str = 'ResNet18',
                 hidden_dims: int = 64,
                 lossname: str = 'NTXentLoss()',
                 lr: float = 0.001,
                 imsize: int = 128,
                 mining: bool = True,
                 augment: bool = True,
                 save_path: str = "outputs/name",
                 ):
        super().__init__()
        self.lossname = lossname
        self.lr = lr
        self.imsize = imsize
        self.mining = mining
        self.augment = augment
        self.save_path = save_path

        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()

        # Dictionaries of embeddings and (pseudo-)labels
        self.embed = {}
        self.embed['train'] = []
        self.embed['val'] = []
        self.embed['test'] = []
        self.embed['predict'] = []
        # self.embed['test_base'] = []
        # self.embed['test_target'] = []

        # Placeholder for calculating clustering driftness
        self.labels = []

        self.paths = {}
        self.paths['train'] = []
        self.paths['val'] = []
        self.paths['test'] = []
        self.paths['predict'] = []

        # self.paths['test_base'] = []
        # self.paths['test_target'] = []
        if augment:
            self.augtrn = T.Compose([T.RandomResizedCrop(self.imsize, scale=(0.95, 1.0), ratio=(0.95,1.05)),
                                     T.Resize(self.imsize),
                                     T.RandomPerspective(distortion_scale=0.5, p=0.5),
                                     T.ElasticTransform(alpha=100.0),
                                     T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                     T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                     T.RandomGrayscale(p=0.2),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.save_hyperparameters()
        ntnme  = backbone+'_Weights.DEFAULT'
        ldr = f"self.net = {backbone.lower()}(weights='{ntnme}')"
        print(ldr)
        exec(ldr)
        self.net.fc = nn.Sequential(self.net.fc, nn.ReLU(), nn.Linear(1000, self.hparams.hidden_dims))

        # initial empty embeddings
        self.umap = init_umap()
        self.embed_ump = {}
        self.embed_ump['train'] = None
        self.embed_ump['val'] = None
        self.embed_ump['test'] = None

    def doloss(self, batch, mode='train'):
        embeddings = self.net(batch[0])
        # test_labels = [int(item.split("/")[-2]) for item in batch[1]]
        test_paths = [item.split("/")[-1] for item in batch[1]]
        test_labels = [int(item.split("_")[-1][:-5]) for item in test_paths]

        if mode != 'test' and mode != 'predict':
            labels = torch.arange(len(batch[0]))
        else:
            labels = torch.tensor(test_labels, dtype=torch.int32)
        # labels = batch[1]

        self.embed[mode].append(embeddings.clone().detach().cpu())
        self.paths[mode] += batch[1]
        # assert len(test_labels) == len(batch[0]), f"{embeddings.shape}: {len(test_labels)} vs {len(batch[0])}"

        lf = eval(f'losses.{self.lossname}')
        hard_pairs = None
        labels = labels.reshape(-1)
        if self.augment and mode == "train":
            augembedd = self.net(self.augtrn(batch[0]))
            embeddings = torch.cat([embeddings, augembedd])
            labels = torch.cat([labels, labels])
        if self.mining and mode == "train":
            miner = miners.MultiSimilarityMiner()
            hard_pairs = miner(embeddings, labels)

        if hard_pairs is not None:
            nll = lf(embeddings, labels, hard_pairs)
        else:
            nll = lf(embeddings, labels)

        if not mode == 'predict':
            self.log(mode + "_loss", nll, prog_bar=True, on_step=False, on_epoch=True)

        if mode == 'train':
            return {"loss": nll}

        return {f"{mode}_loss": nll}

    def training_step(self, batch, batch_idx):
        return self.doloss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.doloss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.doloss(batch, mode='test')

    def predict_step(self, batch, batch_idx):
        return self.doloss(batch, mode='predict')

    def _epoch_end(self, mode='train'):
        embd = torch.cat(self.embed[mode]).numpy()
        paths = self.paths[mode]
        test_paths = [item.split("/")[-1] for item in paths]
        gt_labels = [int(item.split("_")[-1][:-5]) for item in test_paths]

        # For cosine metrics, normalise embeddings in l2-norm
        embd = normalize(embd, norm='l2', axis=1)

        # Default DBSCAN, need to tune eps and min_samples
        # clustering = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(embd)

        # HDBSCAN, automatically finds the best clusters
        if mode in ['val', 'test', 'predict']:
            cluster_sizes = [10, 20, 30, 40, 50, 60]
            num_min_samples = [3, 5, 7, 10, 15, 20]
            best_ari = float('-inf')
            assumed_labels = None
            best_heuristics = None
            hdbscan_heuristics = list(product(cluster_sizes, num_min_samples))
            for heuristic in hdbscan_heuristics:
                min_cluster_size, min_samples = heuristic
                if min_samples > min_cluster_size:
                    continue
                clustering = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,         # controls conservativeness; set equal to or smaller than min_cluster_size
                        metric='cosine',
                        cluster_selection_epsilon=0.1,
                        ).fit(embd)
                prod_labels = clustering.labels_
                ari = evaluate_clustering_with_gt(prod_labels, gt_labels)['adjusted_rand_index']
                if ari > best_ari:
                    assumed_labels = prod_labels
                    best_ari = ari
                    best_heuristics = heuristic

            print(f"Best heuristics: {best_heuristics}")

        if mode in ['test', 'predict']:
            mode_save = True if mode == 'test' else False
            # self.umap = joblib.load(os.path.join(self.save_path, 'umap_model.joblib')) # Unsure if reloading umap during training is necessary
            self.umap.fit(embd)
            self.embed_ump[mode] = self.umap.transform(embd)
            scatter(self.embed_ump[mode], labels=np.array(gt_labels), title=f"Test Epoch UMAP Visualisation", filename=os.path.join(self.save_path, 'visualisations', mode, f'{mode}_umap_gt_labelled.png'))
            scatter(self.embed_ump[mode], labels=np.array(gt_labels), title=f"Test Epoch UMAP Visualisation", filename=os.path.join(self.save_path, 'visualisations', mode, f'{mode}_umap_gt_labelled.pdf'))
            print("\n"+"-"*20 + "KNN Metrics" + "-"*20)
            print(f"{mode.title()} Mode: KNN Probe Accuracy: {KNNProbe(embd, np.array(gt_labels))}")
            label_log_entry = evaluate_clustering_with_gt(assumed_labels, gt_labels)
            print("-"*20 + "Label Metrics" + "-"*20)
            self.show_and_save_log(label_log_entry, mode=mode, save=mode_save)
            print("-"*20 + "Clustering Metrics" + "-"*20)
            result_entry = ClusterMetrics(embd, paths, assumed_labels)

            self.show_and_save_log(result_entry, mode=mode, save=mode_save)

            save_cluster_csv(assumed_labels, paths, output_csv=os.path.join(self.save_path, 'visualisations', mode, f'cluster.csv'))
        if mode == 'val':
            if self.current_epoch == 0:
                self.umap.fit(embd)
                self.embed_ump[mode] = embd
                # joblib.dump(self.umap, os.path.join(self.save_path, 'umap_model.joblib'))
            else:
                self.embed_ump[mode] = self.umap.transform(embd)
            # clustering = DBSCAN(eps=0.5, min_samples=5).fit(self.embed_ump[mode]) # May need to be adjusted to raw embeddings
            if self.current_epoch == 0:
                self.labels = assumed_labels

            # knn_cp_entry = KNNClusterPerformance(embd, np.array(gt_labels))
            # knn_cc_entry = KNNClusterConsistency(embd, np.array(gt_labels), embd, assumed_labels)
            print("\n"+"-"*20 + "KNN Metrics" + "-"*20)
            print(f"{mode.title()} Mode: KNN Probe Accuracy: {KNNProbe(embd, np.array(gt_labels))}")
            # self.show_and_save_log(knn_cp_entry, mode=mode, save=False)
            # self.show_and_save_log(knn_cc_entry, mode=mode, save=False)

            label_log_entry = evaluate_clustering_with_gt(assumed_labels, gt_labels)
            print("-"*20 + "Label Metrics" + "-"*20)

            self.show_and_save_log(label_log_entry, mode=mode)
            # if label_log_entry is not None:
            #     for k, v in label_log_entry.items():
            #         print(f"{mode.title()} Mode: {k}: {v}")
            #         self.log(mode+"_"+k, v, prog_bar=False, on_step=False, on_epoch=True)
            print("-"*20 + "Clustering Metrics" + "-"*20)

            scatter(self.embed_ump[mode], labels=assumed_labels, title=f"Epoch {self.current_epoch+1} training UMAP Visualisation", filename=os.path.join(self.save_path, 'visualisations', mode, f'umap_{str(self.current_epoch).zfill(5)}.png'))
            scatter(self.embed_ump[mode], labels=assumed_labels, title=f"Epoch {self.current_epoch+1} training UMAP Visualisation", filename=os.path.join(self.save_path, 'visualisations', mode, f'umap_{str(self.current_epoch).zfill(5)}.pdf'))            
            scatter(self.embed_ump[mode], labels=np.array(gt_labels), title=f"Epoch {self.current_epoch+1} GT UMAP Visualisation", filename=os.path.join(self.save_path, 'visualisations', mode, f'umap_gt_{str(self.current_epoch).zfill(5)}.png'))
            scatter(self.embed_ump[mode], labels=np.array(gt_labels), title=f"Epoch {self.current_epoch+1} GT UMAP Visualisation", filename=os.path.join(self.save_path, 'visualisations', mode, f'umap_gt_{str(self.current_epoch).zfill(5)}.pdf'))
            # result_entry = ClusterMetrics(self.embed_ump[mode], paths, assumed_labels)
            result_entry = ClusterMetrics(embd, paths, assumed_labels)
            ClusteringDrifts(embd, self.labels, assumed_labels, paths, self.embed_ump[mode], save_path=os.path.join(self.save_path, 'visualisations', mode, f'drift_metrics_{str(self.current_epoch).zfill(5)}.csv'))
            self.labels = assumed_labels
            self.show_and_save_log(result_entry, mode=mode)
            # if result_entry is not None:
            #     for k, v in result_entry.items():
            #         print(f"{mode.title()} Mode: {k}: {v}")
            #         self.log(mode+"_"+k, v, prog_bar=False, on_step=False, on_epoch=True)
            total_entropy = sum_entropy(csv_root = os.path.join(self.save_path, 'visualisations', mode, f'drift_metrics_{str(self.current_epoch).zfill(5)}.csv'), num_frames=1500)
            print(f"{mode.title()} Mode: Total Entropy: {int(total_entropy)}")
            self.log(mode+"_SumEntropy", int(total_entropy), prog_bar=False, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        self._epoch_end(mode='train')

    def on_validation_epoch_end(self):
        self._epoch_end(mode='val')

    def on_test_epoch_end(self):
        self._epoch_end(mode='test')

    def on_predict_epoch_end(self):
        self._epoch_end(mode='predict')

    def _epoch_start(self, mode):
        self.embed[mode].clear()
        self.paths[mode].clear()
        self.embed[mode]  = []
        self.paths[mode] = []

    def on_train_epoch_start(self):
        self._epoch_start('train')

    def on_validation_epoch_start(self):
        self._epoch_start('val')

    def on_test_epoch_start(self):
        # pass
        # print('\nStart of test_epoch\n')
        self._epoch_start('test')

    def on_predict_epoch_start(self):
        # pass
        # print('\nStart of predict_epoch\n')
        self._epoch_start('predict')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def show_and_save_log(self, entry, mode, save=True):
        if entry is not None:
            for k, v in entry.items():
                print(f"{mode.title()} Mode: {k}: {v}")
                if save:
                    self.log(mode+"_"+k, v, prog_bar=False, on_step=False, on_epoch=True)
