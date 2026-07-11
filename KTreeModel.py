import os
import numpy as np 
from sklearn.cluster import KMeans
import time

class KTree:
    
    def __init__(self):
        self.val_frames = np.load("val_frames_with_window_features.npy", allow_pickle=True)
        self.test_frames = np.load("test_frames_with_window_features.npy", allow_pickle=True)

        self.train_x = np.load("X.npy", allow_pickle=True)
        self.train_y = np.load("Y.npy", allow_pickle=True)

    
    def create_tree(self , k, d):
        start = time.time()

        k_means_models = []
        cluster_label = np.zeros((len(self.train_x),d))
        for i in range(d):
            if(i == 0):
                tree = KMeans(n_clusters=k, random_state=42)
                y_pred = tree.fit_predict(self.train_x)
                k_means_models.append(tree)
                cluster_label[:,i] = y_pred
            else:
                models = []
                for i in range(k):
                    tree = KMeans(n_clusters=k, random_state=42)
                    mask = self.train_x == i
                    y_pred = tree.fit_predict(mask)
                    cluster_label[:,i] = y_pred
                    models.append(tree)
                k_means_models.append(models)
        
        elapsed_time = time.time() - start

        return k_means_models, cluster_label, elapsed_time
                            



            



