import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50


class Model(nn.Module):
    
    def __init__(self,
                 base_model="resnet18",
                 fc_dim=128,
                 k_subs=10,
                 layer_sizes=[64, 1],
                 use_bn=False,
                 dr_rate=0.2):
        super(Model, self).__init__()
        
        imagenet = True
        resnet_dict = {"resnet18": resnet18(num_classes=fc_dim),
                       "resnet50": resnet50(num_classes=fc_dim)}
        self.model = resnet_dict[base_model]
        dim_mlp = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.model.fc)

        
        # k subnetworks for bregman
        self.subnets = nn.ModuleList()
        
        for k_idx in range(k_subs):
            fc = nn.Sequential()
            
            for i, (in_size, out_size) in enumerate(zip([fc_dim] + layer_sizes[:-1], layer_sizes)):
                if i + 1 < len(layer_sizes):
                    fc.add_module(
                        name="fc_{:d}_{:d}".format(k_idx, i),
                        module=nn.Linear(in_size, out_size))
                    
                    if use_bn:
                        fc.add_module(
                            name="bn_{:d}_{:d}".format(k_idx, i),
                            module=nn.BatchNorm1d(out_size))
                        
                    fc.add_module(
                        name="relu_{:d}_{:d}".format(k_idx, i),
                        module=nn.ReLU())
                    
                    fc.add_module(
                        name="dp_{:d}_{:d}".format(k_idx, i),
                        module=nn.Dropout(p=dr_rate))

                else:
                    fc.add_module(
                        name="output_{:d}".format(k_idx),
                        module=nn.Linear(in_size, out_size))
                    
                    #fc.add_module(
                    #    name="output_A_{:d}".format(k_idx),
                    #    module=nn.Sigmoid())
                
            self.subnets.append(fc)
            
    def forward(self, x):
        fc_out = self.model(x)
        
        out = []
        for subnet in self.subnets:
            out.append(subnet(fc_out))
        
        out = torch.cat(out, -1)
        #F.normalize(feature, dim=-1)
        return fc_out, out
