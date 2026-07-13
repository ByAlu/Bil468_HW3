import numpy as np 
from sklearn.cluster import KMeans
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class KTree:
    
    def __init__(self):
        self.val_frames = np.load("val_frames_with_window_features.npy", allow_pickle=True)
        self.test_frames = np.load("test_frames_with_window_features.npy", allow_pickle=True)

        self.train_x = np.load("X.npy", allow_pickle=True)
        self.train_y = np.load("Y.npy", allow_pickle=True)

    
    def create_tree(self , k, d):

        k_means_models = [np.empty(k**i, dtype=object) for i in range(d+1)]
        cluster_label = np.zeros((len(self.train_x),d))
     
        self.create_tree_recursive(k, d, 0, 0, 0, [], k_means_models, cluster_label)


        self.k_means_models = k_means_models
        self.cluster_label = cluster_label


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

            if len(indices) < k:
                paren_model = k_means_models[d_val-1][parent_model_index]
                x = self.train_x[indices]
                y_pred = paren_model.predict(x)
                y_val = self.train_y[indices]
                mask = y_val == 0
                y_val[mask] = -1
                class_val = sum(y_val)
                if class_val >= 0:
                    class_val = 1
                else:
                    class_val = 0

                k_means_models[d_val][parent_model_index*k + y_pred[0]] =  class_val
                return


            leaf_model = False
            if d_val == d-1:
                leaf_model = True 

            
            
            tree = KMeans(n_clusters=k, random_state=42)
            y_pred = tree.fit_predict(self.train_x[indices])
            cluster_label[indices,d_val] = y_pred

            tree_index = parent_model_index * k + model_index
            k_means_models[d_val][tree_index] = tree

            if leaf_model:
                for i in range(k):
                    mask = y_pred == i
                    new_indices = indices[mask]
                    y_val = self.train_y[new_indices]
                    mask_val = y_val == 0
                    y_val[mask_val] = -1
                    class_val = sum(y_val)
                    if class_val >= 0:
                        class_val = 1
                    else:
                        class_val = 0
                    k_means_models[d_val+1][tree_index*k + i] =  class_val
                return

                
            for i in range(k):
                mask = y_pred == i
                new_masks = masks + [mask]
                self.create_tree_recursive(k, d, tree_index, i, d_val + 1, new_masks, k_means_models, cluster_label)





    # Araba için +1, diğerleri için -1. Total değer sıfırdan büyükse araba var deriz
    def find_class_value(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            return [0]
        if X.ndim == 1:
            X = X.reshape(1, -1)

        depth = self.cluster_label.shape[1]
        k = len(self.k_means_models[1])

        results = np.full(len(X), -2, dtype=int)

        node_indices = np.zeros(len(X), dtype=int)

        active = np.ones(len(X), dtype=bool)

        for level in range(depth + 1):
            active_samples = np.flatnonzero(active)

            if len(active_samples) == 0:
                break

            current_nodes = node_indices[active_samples]

            # Aynı düğümdeki örnekleri birlikte işle
            for node_index in np.unique(current_nodes):
                sample_indices = active_samples[current_nodes == node_index]

                model = self.k_means_models[level][node_index]

                if isinstance(model, np.ndarray) and model.size == 1:
                    model = model.item()

                if isinstance(model, KMeans):
                    predictions = model.predict(X[sample_indices])

                    node_indices[sample_indices] = (
                        node_index * k + predictions
                    )
                else:
                    results[sample_indices] = int(model)
                    active[sample_indices] = False

        return results
        
        
        

        



    def validate_frames(self):
        y_true = []
        y_pred = []
        from tqdm import tqdm

        for frame in tqdm(self.val_frames):
            for i in range (2,6):
                features = frame[i]["features"]
                labels = frame[i]["labels"]

                y_true.extend(labels)

                for i,label in enumerate(labels):
                    label_features = features[i]
                    win_predict = self.find_class_value(label_features)
                    mask = win_predict == 0
                    win_predict[mask] = -1                    
                    class_val = sum(win_predict)
                    if class_val >= 0:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return accuracy, precision, recall, f1, cm




                            



            



