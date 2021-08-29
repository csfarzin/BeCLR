import pandas as pd
import torch
from tqdm import tqdm
from loss.breg_loss import BregmanLoss
from loss.nt_xent import NT_Xent
from loss.breg_margin_loss import BregMarginLoss


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Trainer():
    def __init__(self, model,
                 optimizer,
                 scheduler,
                 temperature,
                 num_cls,
                 epochs,
                 device):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temperature = temperature
        self.num_cls = num_cls
        self.epochs = epochs
        self.device = device
        self.mixed_loss = True

    # train for one epoch to learn unique features
    def train(self, data_loader, epoch):
        self.model.train()
        batch_size = data_loader.batch_size
        #bloss = BregMarginLoss(batch_size)
        bloss = BregmanLoss(batch_size, self.temperature)
        nt_xent = NT_Xent(batch_size, self.temperature)
        
        total_loss, total_num, tot_max, train_bar = 0.0, 0, 0, tqdm(data_loader)
        tot_bloss, tot_nt_xent = 0.0, 0.0
        num_max = torch.tensor([0])
        for [aug_1, aug_2], target in train_bar:
            aug_1, aug_2 = aug_1.to(self.device), aug_2.to(self.device)
            feature_1, out_1 = self.model(aug_1)
            feature_2, out_2 = self.model(aug_2)

            # compute loss
            loss, num_max = bloss(out_1, out_2)
            tot_bloss += loss.item() * batch_size
            if self.mixed_loss:
                loss1 = nt_xent(feature_1, feature_2)
                tot_nt_xent += loss1.item() * batch_size
                loss = loss + loss1
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            tot_max += num_max
            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description(
                '{}Train{} {}Epoch:{} [{}/{}] {}Loss:{}  {:.4f} {}Active Subs:{} [{}/{}]'
                .format(
                    bcolors.OKCYAN, bcolors.ENDC,
                    bcolors.WARNING, bcolors.ENDC,
                    epoch,
                    self.epochs,
                    bcolors.WARNING, bcolors.ENDC,
                    total_loss / total_num,
                    bcolors.WARNING, bcolors.ENDC,
                    len(torch.where(tot_max>10)[0]),
                    tot_max.shape[0]))
            
        # warmup with nt_xent loss for the first 50 epochs
        #if epoch >= 100:
        self.scheduler.step()

        return (total_loss/total_num,
                tot_bloss/total_num,
                tot_nt_xent/total_num,
                self.scheduler.get_last_lr()[0])
    
    
    def bregman_sim(self, feature, feature_bank):
        # [B, 1]
        mf = torch.max(feature, dim=1)
        # [N, 1]
        mfb = torch.max(feature_bank, dim=1)
        indx_max_feature_bank = mfb[1]
        max_feature = mf[0].reshape(-1, 1)
        # [B, N]
        dist_matrix = max_feature - feature[:, indx_max_feature_bank]
        # Computing Similarity from Bregman distance
        sigma = torch.tensor([1.]).to(self.device)
        sigma = 2 * torch.pow(sigma, 2)
        sim_matrix = torch.exp(torch.div(-dist_matrix, sigma))
        
        return sim_matrix
        
    # test for one epoch, use weighted knn to find the most similar images' label to assign the test image
    def test(self, memory_data_loader, test_data_loader, k_nn, epoch):
        self.model.eval()
        total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
        
        with torch.no_grad():
            # generate feature bank
            
            for [data, _], target in tqdm(memory_data_loader,
                                        desc=f'{bcolors.OKBLUE}Feature extracting{bcolors.ENDC}'):
                feature, out = self.model(data.to(self.device))
                feature_bank.append(out)
                feature_labels.append(target)
            # [N, D]
            feature_bank = torch.cat(feature_bank, dim=0)
            feature_labels = torch.cat(feature_labels, dim=0).long().to(self.device)
            # [N]
            
            #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=self.device)
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(test_data_loader)
            for [data, _], target in test_bar:
                data, target = data.to(self.device), target.to(self.device)
                feature, out = self.model(data)

                total_num += data.size(0)
                # compute bregman similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = self.bregman_sim(out, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k_nn, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1),
                                          dim=-1,
                                          index=sim_indices)
                sim_weight = (sim_weight / self.temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * k_nn, self.num_cls, device=self.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(
                    data.size(0), -1, self.num_cls) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum(
                    (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum(
                    (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                
                test_bar.set_description(
                    '{}Test{}  {}Epoch:{} [{}/{}] {}Acc@1: {}{:.2f}% {}Acc@5: {}{:.2f}%'.format(
                    bcolors.OKCYAN, bcolors.ENDC,
                    bcolors.WARNING, bcolors.ENDC,
                    epoch,
                    self.epochs,
                    bcolors.WARNING, bcolors.ENDC,
                    (total_top1 / total_num) * 100,
                    bcolors.WARNING, bcolors.ENDC,
                    (total_top5 / total_num) * 100))

        return (total_top1 / total_num) * 100, (total_top5 / total_num) * 100