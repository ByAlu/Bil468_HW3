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

        k_means_models = [np.empty(k**i, dtype=object) for i in range(d)]
        cluster_label = np.zeros((len(self.train_x),d))
     
        self.create_tree_recursive(k, d, 0, 0, 0, [], k_means_models, cluster_label)

        elapsed_time = time.time() - start

        return k_means_models, cluster_label, elapsed_time

    def create_tree_recursive(self, k, d, parent_model_index, model_index, d_val, masks, k_means_models, cluster_label):
        if d_val >= d:
            return

        if(d_val == 0):
            tree = KMeans(n_clusters=k, random_state=42)
            y_pred = tree.fit_predict(self.train_x)
            k_means_models[0][0] = tree
            cluster_label[:,0] = y_pred
            for i in range(k):
                mask = y_pred == i
                self.create_tree_recursive(k, d, 0, i, 1, [mask], k_means_models, cluster_label)
        
        else:
            indices = np.arange(len(self.train_x))
            for mask in masks:
                indices = indices[mask]
            
            tree = KMeans(n_clusters=k, random_state=42)
            y_pred = tree.fit_predict(self.train_x[indices])
            cluster_label[indices,d_val] = y_pred

            tree_index = parent_model_index * k + model_index
            k_means_models[d_val][tree_index] = tree
            for i in range(k):
                mask = y_pred == i
                new_masks = masks + [mask]
                self.create_tree_recursive(k, d, tree_index, i, d_val + 1, new_masks, k_means_models, cluster_label)





    # Araba için +1, diğerleri için -1. Total değer sıfırdan büyükse araba var deriz
    def find_class_value(self, x, k_means_models, cluster_Label):
        y = self.train_y
        



    def validate(k_means_models, cluster_Label):
        s = 9

                            



            



