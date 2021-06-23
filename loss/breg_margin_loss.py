import torch
import torch.nn


class BregMarginLoss(torch.nn.Module):

    def __init__(self, batch_size, margin=1, beta=0.5):
        super(BregMarginLoss, self).__init__()
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(batch_size)
        self.margin = margin
        self.beta = beta
        
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
        
    def criterion(self, dist):
        pos = dist[:, 0].reshape(-1, 1)
        mdist = self.margin - dist[:, 1:]
        neg = torch.clamp(mdist, min=0.0)
        
        #print(pos)
        #print(torch.sum(torch.pow(neg, 2), dim=1).reshape(-1, 1)/(dist.size()[0]))
        #loss = self.beta * ((dist.size()[1] - 1) * pos) + \
        #(1 - self.beta) * (torch.sum(torch.pow(neg, 2), dim=1).reshape(-1, 1))
        loss = pos + torch.sum(torch.pow(neg, 2), dim=1).reshape(-1, 1)
        loss = torch.sum(loss) / 2 #/ (dist.size()[0])
        
        return loss
        
    def berg_div(self, features):
        mm = torch.max(features, dim=1)
        indx_max_features = mm[1]
        max_features = mm[0].reshape(-1, 1)
        
        # Compute the number of active subnets in one batch
        eye = torch.eye(features.shape[1])
        one = eye[indx_max_features]
        num_max = torch.sum(one, dim=0)
        
        dist_matrix = max_features - features[:, indx_max_features]
        
        return dist_matrix, num_max
        
    def forward(self, out_a, out_b):
        N = 2 * self.batch_size
        features = torch.cat((out_a, out_b), dim=0)
        ###################################################
        ### Computing Distance Matrix ###################
        dist_matrix, num_max = self.berg_div(features)
        ###################################################
        pos_ab = torch.diag(dist_matrix, self.batch_size)
        pos_ba = torch.diag(dist_matrix, -self.batch_size)

        positives = torch.cat((pos_ab, pos_ba), dim=0).reshape(N, 1)
        negatives = dist_matrix[self.mask].reshape(N, -1)
        #######################################################
        ### New loss
        N_neg = []
        for row in negatives:
            N_neg.append(row[torch.randint(0, len(row), (1,))])
            
        negatives = torch.cat(N_neg, dim=0).reshape(N, -1)
        #negatives = negatives.reshape(-1, 1)
        #negatives, negatives_indices = negatives.topk(k=10*N, largest=False, dim=0)
        #negatives = negatives.reshape(N, -1)
        #######################################################
        #labels = torch.zeros(N, dtype=torch.long).to(features.device)
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits)
        
        return loss, num_max