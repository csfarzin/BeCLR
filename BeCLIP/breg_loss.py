import torch
import torch.nn as nn


class BregmanLoss(nn.Module):
    def __init__(self, batch_size, temperature, length_scale):
        super(BregmanLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.length_scale = length_scale

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
        #self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def b_sim(self, features):
        mm = torch.max(features, dim=1)
        indx_max_features = mm[1]
        max_features = mm[0].reshape(-1, 1)
        
        # Compute the number of active subnets in one batch
        eye = torch.eye(features.shape[1])
        one = eye[indx_max_features]
        num_max = torch.sum(one, dim=0)
        
        dist_matrix = max_features - features[:, indx_max_features]
        

        sigma = torch.tensor([self.length_scale]).to(features.device)
        sigma = 2 * torch.pow(sigma, 2)
        sim_matrix = torch.exp(torch.div(-dist_matrix, sigma))
        
        return sim_matrix, num_max

    def forward(self, out_a, out_b):
        
        N = 2 * self.batch_size

        features = torch.cat((out_a, out_b), dim=0)
        
        ###################################################
        ### Computing Similarity Matrix ###################
        sim_matrix, num_max = self.b_sim(features)
        sim_matrix = sim_matrix / self.temperature
        ###################################################
        #sim_matrix = self.similarity_f(out.unsqueeze(1), out.unsqueeze(0)) / self.temperature

        pos_ab = torch.diag(sim_matrix, self.batch_size)
        pos_ba = torch.diag(sim_matrix, -self.batch_size)

        positives = torch.cat((pos_ab, pos_ba), dim=0).reshape(N, 1)
        negatives = sim_matrix[self.mask].reshape(N, -1)

        labels = torch.zeros(N, dtype=torch.long).to(features.device)
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss, num_max
