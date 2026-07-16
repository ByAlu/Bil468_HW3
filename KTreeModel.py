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

        self.val_x = np.load("val_X.npy", allow_pickle=True)
        self.val_y = np.load("val_Y.npy", allow_pickle=True)

        zeros = np.count_nonzero(self.train_y == 0)
        ones  = np.count_nonzero(self.train_y == 1)
             
        self.label_ratio = zeros // ones


    
    def create_tree(self , k, d):

        k_means_models = [np.empty(k**i, dtype=object) for i in range(d+1)]
        cluster_label = np.zeros((len(self.train_x),d))
     
        self.create_tree_recursive(k, d, 0, 0, 0, [], k_means_models, cluster_label)


        self.k_means_models = k_means_models
        self.cluster_label = cluster_label
        self.k = k
        self.d = d


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
                # mask = y_val == 1
                # y_val[mask] = self.label_ratio
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
                    # mask = y_val == 1
                    # y_val[mask] = self.label_ratio
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
        
        if X is None or len(X) == 0:
            return np.array([0])

        X = np.asarray(X)

        number_of_samples = len(X)
        results = np.zeros(number_of_samples)

        # Her örneğin mevcut düğüm indeksi
        node_indices = np.zeros(number_of_samples, dtype=np.int64)

        # Henüz yaprak sınıfına ulaşmamış örnekler
        active = np.ones(number_of_samples, dtype=bool)

        number_of_levels = len(self.k_means_models)

        for level in range(number_of_levels):
            if not np.any(active):
                break

            active_indices = np.flatnonzero(active)
            active_nodes = node_indices[active_indices]

            # Aynı düğümde bulunan örnekleri birlikte işle
            for node_index in np.unique(active_nodes):
                sample_indices = active_indices[
                    active_nodes == node_index
                ]

                model = self.k_means_models[level][node_index]

                if isinstance(model, KMeans):
                    predictions = model.predict(X[sample_indices])

                    node_indices[sample_indices] = (
                        node_index * self.k + predictions
                    )
                else:
                    results[sample_indices] = int(model)
                    active[sample_indices] = False

        return results
        
        

    def validate_frames(self):
        y_true = []
        y_pred = []

        from tqdm import tqdm
        for i in tqdm(range(self.val_frames.shape[0])):
            features = self.val_frames[i][0]
            label_true = self.val_frames[i][1]

            y_true.append(label_true)

            win_predict = self.find_class_value(features)
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
    

    def get_cluster_centers(self):
        centers = []
        labels = []

        for d_val in range(self.d):
            for k_val in range(self.k**d_val):
                model = self.k_means_models[d_val][k_val]
                if not isinstance(model, KMeans):
                    continue
                
                children = self.k_means_models[d_val + 1][k_val * self.k:(k_val + 1) * self.k]

                if isinstance(children[0], KMeans):
                    continue

                centers.extend(model.cluster_centers_)
                labels.extend(children)

        return centers, labels
    

    def test(self):
        y_true = []
        y_pred = []

        from tqdm import tqdm
        for i in tqdm(range(self.test_frames.shape[0])):
            features = self.test_frames[i][0]
            label_true = self.test_frames[i][1]

            y_true.append(label_true)

            win_predict = self.find_class_value(features)
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
    

    def photo_test(self):
        import cv2
        test_frames = np.load("test_frames.npy", allow_pickle=True)
        sift = cv2.SIFT_create()
        windows = [16, 32, 64, 128]
        
        for frame in test_frames:
            path = frame[0]
            image = cv2.imread("images/"+path)
            for window in windows:
                stride = window//2
                found = False

                for y in range(0, 128 - window + 1, stride):
                    for x in range(0, 128 - window + 1, stride):
                        patch = image[y:y + window, x:x + window]
                        _, feats = sift.detectAndCompute(patch,None)
                        class_val = self.find_class_value(feats)
                        mask = class_val == 0
                        class_val[mask] = -1
                        class_val = sum(class_val)                        
                        if class_val>=0: #sadece ilk bulunanı çiz
                            cv2.rectangle(image, (x, y), (x + window, y + window), (0,255,0), 2)
            image = cv2.resize(image,(480,480))
            cv2.imshow("image", image)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                return 

            
    def deneme(self):
        y_pred = self.find_class_value(self.val_x)

        y_true = self.val_y

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return accuracy, precision, recall, f1, cm

    





                            



            



