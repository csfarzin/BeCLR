import torch
import tqdm
import numpy as np
from model import Model


class Bmeans():
    def __init__(self, model, dataloader, n_clusters=2, max_iter=300, device=torch.device('cuda'))
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        #self.n_init = n_init
        self.dataloader = dataloader
        self.model = model
        self.device = device
        
        feature_bank = []
        label_bank = []
        X = []
        for data, target in tqdm(
            data_loader, desc=f'{bcolors.OKBLUE}Features extracting and Labels{bcolors.ENDC}'):
            _, out = self.model(data.to(self.device))
            feature_bank.append(out.detach().cpu())
            X.append(data.detach().cpu())
            label_bank.append(target)
        # [N, k]
        self.feature_bank = torch.cat(feature_bank, dim=0)
        self.label_bank = torch.cat(label_bank, dim=0)
        self.X = torch.cat(X, dim=0)
        self.X_shape = self.X.shape

    def initialize_random_centroids(self):
        centroids = torch.zeros(self.n_clusters, self.X_shape[1], self.X_shape[2], self.X_shape[3])

        for k in range(self.n_clusters):
            centroid = self.X[np.random.choice(range(self.X_shape[0]))]
            centroids[k] = centroid

        return centroids        
      
    def to_centers_dist(self, centroid)
        feature_centroids = self.model(centroid.cuda())
        feature_centroids = feature_centroids.detach().cpu()
        # clac bregman of centers
        #[n_clusters, 1]
        mf = torch.max(feature_centroids, dim=1)
        # clac bregman of each sample
        # [N, 1]
        mfb = torch.max(self.feature_bank, dim=1)
        
        indx_max_centroids = mf[1]
        value_max_centroids = mf[0].reshape(-1, 1)
        
        indx_max_feature_bank = mfb[1]
        value_max_feature_bank = mfb[0].reshape(-1, 1)      
        # [N, n_clusters]
        dist_matrix = value_max_feature_bank - self.feature_bank[:, indx_max_centers]
        
        return dist_matrix
    
    def create_clusters(self, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.n_clusters)]
        
        dist_matrix = self.to_centers_dist(centroid)
        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(dist_matrix):
            closest_centroid = torch.argmin(point)
            clusters[closest_centroid].append(point_idx)

        return clusters
    
    def calculate_new_centroids(self, clusters):
        centroids = torch.zeros(self.n_clusters, self.X_shape[1], self.X_shape[2], self.X_shape[3])
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(self.X[cluster], axis=0)
            centroids[idx] = new_centroid

        return centroids   
    
    def predict_cluster(self, clusters):
        y_pred = torch.zeros(self.X_shape[0])

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred
     
    def fit(self):
        centroids = self.initialize_random_centroids()

        for _ in range(self.max_iter):
            clusters = self.create_clusters(centroids)

            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters)

            diff = centroids - previous_centroids

            if not diff.any():
                print("Termination criterion satisfied")
                break
                
        # Get label predictions
        y_pred = self.predict_cluster(clusters)

        return y_pred


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'runs/May21_05-41-07_cifar10.de/resnet18_0.001_128_0.1_200_512_500_model.pth'
model = Model(base_model="resnet18",
                 fc_dim=128,
                 k_subs=100,
                 layer_sizes=[90, 1],
                 use_bn=False,
                 dr_rate=0.2)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.to(device)