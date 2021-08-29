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
        
        skin = False
        resnet_dict = {"resnet18": resnet18(num_classes=fc_dim),
                       "resnet50": resnet50(num_classes=fc_dim)}
        model = resnet_dict[base_model]
        dim_mlp = model.fc.in_features
        
        self.f = []
        
        if skin:
            for name, module in model.named_children():
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        else:
            for name, module in model.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)

        
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                               nn.BatchNorm1d(dim_mlp),
                               nn.ReLU(inplace=True),
                               nn.Linear(dim_mlp, fc_dim, bias=True))
        
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
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        fc_out = self.g(feature)
        fc_out = F.normalize(fc_out, dim=-1)
        
        out = []
        for subnet in self.subnets:
            out.append(subnet(fc_out))
        
        out = torch.cat(out, -1)
        #F.normalize(feature, dim=-1)
        return fc_out, out
