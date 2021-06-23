import torch
import torch.nn as nn


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def forward(self, out_a, out_b):
        
        N = 2 * self.batch_size

        out = torch.cat((out_a, out_b), dim=0)
        
        ###################################################
        ### Computing Similarity Matrix ###################
        sim_matrix = self.similarity_f(out.unsqueeze(1), out.unsqueeze(0)) / self.temperature
        ###################################################
        

        pos_ab = torch.diag(sim_matrix, self.batch_size)
        pos_ba = torch.diag(sim_matrix, -self.batch_size)

        positives = torch.cat((pos_ab, pos_ba), dim=0).reshape(N, 1)
        negatives = sim_matrix[self.mask].reshape(N, -1)
        
        #######################################################
        ### New loss
        #negatives = negatives.reshape(-1, 1)
        #negatives, negatives_indices = negatives.topk(k=(N-10)*N, largest=False, dim=0)
        #negatives = negatives.reshape(N, -1)
        #######################################################

        labels = torch.zeros(N, dtype=torch.long).to(out.device)
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss