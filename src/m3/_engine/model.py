import torch
import torch.nn as nn
from torch import exp, log
from torch import linalg as LA
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .util import poe
from .nn import MLP

class LinBnDrop(nn.Sequential):
    """Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"""
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

class Encoder(nn.Module):
    """Flexible encoder for multi-modal data"""
    def __init__(self, nfeatures=[10703, 192], hidden_features=[185, 15], z_dim=128):
        super().__init__()
        assert len(nfeatures) == len(hidden_features), "Length mismatch between nfeatures and hidden_features"

        self.n_modalities = len(nfeatures)
        self.feature_splits = nfeatures  
        self.encoders_mean = nn.ModuleList([LinBnDrop(nf, hf, p=0.2, act=nn.ReLU()) for nf, hf in zip(nfeatures, hidden_features)])
        self.encoders_var = nn.ModuleList([LinBnDrop(nf, hf, p=0.2, act=nn.ReLU()) for nf, hf in zip(nfeatures, hidden_features)])

    def forward(self, x):
        splits = torch.split(x, self.feature_splits, dim=1)
        means = []
        vars_ = []
        for i in range(self.n_modalities):
            means.append(self.encoders_mean[i](splits[i]))
            vars_.append(self.encoders_var[i](splits[i]))
        return means, vars_

class batch_Encoder(nn.Module):
    """Encoder for CITE-seq data"""
    def __init__(self, nfeatures_modality=2, z_dim=2):
        super().__init__()
        #self.encoder1 = nn.Linear(nfeatures_modality, z_dim)
        #self.encoder2 = nn.Linear(nfeatures_modality, z_dim)

        self.norm = "ln"
        self.drop = 0.2
        self.encoder1 = MLP(
            [nfeatures_modality] + [16] + [z_dim],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )
        self.encoder2 = MLP(
            [nfeatures_modality] + [16] + [z_dim],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )
    def forward(self, b):
        b_mean = self.encoder1(b)
        b_var = self.encoder2(b)
        return b_mean, b_var

class Decoder(nn.Module):
    """Flexible decoder for multi-modal data"""
    def __init__(self, nfeatures=[10703, 192], z_dim=128, norm="ln", drop=0.2):
        super().__init__()
        self.n_modalities = len(nfeatures)
        self.nfeatures = nfeatures
        self.norm = norm
        self.drop = drop

        self.decoders = nn.ModuleList([
            MLP(
                [z_dim] + [100] + [nfeat],
                hid_norm=self.norm,
                hid_drop=self.drop
            ) for nfeat in nfeatures
        ])

    def forward(self, x):
        decoded = [decoder(x) for decoder in self.decoders]
        return torch.cat(decoded, dim=1)
        
class batch_Decoder(nn.Module):
    """Encoder for CITE-seq data"""
    def __init__(self, nfeatures=2, z_dim=2):
        super().__init__()
        #self.decoder = nn.Linear(z_dim, nfeatures)

        self.norm = "ln"
        self.drop = 0.2
        self.decoder = MLP(
            [z_dim] + [16] + [nfeatures],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

    def forward(self, b):
        b = self.decoder(b)
        return b

class M3_model(nn.Module):
    def __init__(self, nfeatures=[10703, 192], hidden_features=[30, 30], 
                 z_dim=30, cty_classify_dim=50, batch_classify_dim=2, 
                 condition_dim=[2,2]):
        super().__init__()
        self.norm = "ln"
        self.drop = 0.2

        self.encoder = Encoder(nfeatures=nfeatures, hidden_features=hidden_features, z_dim=z_dim)
        self.decoder = Decoder(nfeatures=nfeatures, z_dim=z_dim)

        self.batch_encoder = batch_Encoder(nfeatures_modality=batch_classify_dim, z_dim=z_dim)
        self.batch_decoder = batch_Decoder(nfeatures=batch_classify_dim, z_dim=2)

        self.cty1_classify = MLP(
            [z_dim - 2 - len(condition_dim)*2] + [16] + [cty_classify_dim],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

        self.batch1_classify = MLP(
            [z_dim - 2] + [128,64] + [batch_classify_dim],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

        self.condition_classifiers = nn.ModuleList([
            MLP(
                [2] + [128,64] + [dim],
                hid_norm=self.norm,
                hid_drop=self.drop,
            ) for dim in condition_dim
        ])

        self.batch_classify_dim = batch_classify_dim
        self.condition_dim = condition_dim

    def forward(self, x, b, mask):
        mu_list, logvar_list = self.encoder(x)

        b = b.long()
        b = F.one_hot(b, num_classes=self.batch_classify_dim).float()
        batch_mu, batch_logvar = self.batch_encoder(b)

        mu, logvar = poe(mu_list + [batch_mu], logvar_list + [batch_logvar], mask)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_embedding = z[:, :z.shape[1] - 2 - 2*len(self.condition_dim)]
        z_conds = []
        start = z.shape[1] - 2*len(self.condition_dim) - 2
        for dim in self.condition_dim:
            end = start + 2
            z_conds.append(z[:, start:end])
            start = end
        z_batch = z[:, -2:]

        x_recon = self.decoder(z)
        x_batch_recon = self.batch_decoder(z_batch)

        cla_cty1 = self.cty1_classify(z_embedding)
        cla_batch1 = self.batch1_classify(torch.cat([z_embedding] + z_conds, dim=1))
        cla_conditions = [classifier(zc) for classifier, zc in zip(self.condition_classifiers, z_conds)]

        return x_recon, x_batch_recon, z_embedding, z_batch, z_conds, cla_cty1, cla_batch1, cla_conditions, mu, logvar


class M3_model_wo_condition(nn.Module):
    def __init__(self, nfeatures=[10703, 192], hidden_features=[30, 30], 
                 z_dim=30, cty_classify_dim=50, batch_classify_dim=2, 
                 condition_dim=[2,2]):
        super().__init__()
        self.norm = "ln"
        self.drop = 0.2

        self.encoder = Encoder(nfeatures=nfeatures, hidden_features=hidden_features, z_dim=z_dim)
        self.decoder = Decoder(nfeatures=nfeatures, z_dim=z_dim)

        self.batch_encoder = batch_Encoder(nfeatures_modality=batch_classify_dim, z_dim=z_dim)
        self.batch_decoder = batch_Decoder(nfeatures=batch_classify_dim, z_dim=2)


        self.batch1_classify = MLP(
            [z_dim - 2] + [128, 64] + [batch_classify_dim],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )


        self.batch_classify_dim = batch_classify_dim
        self.condition_dim = condition_dim

    def forward(self, x, b, mask):
        mu_list, logvar_list = self.encoder(x)

        b = b.long()
        b = F.one_hot(b, num_classes=self.batch_classify_dim).float()
        batch_mu, batch_logvar = self.batch_encoder(b)

        mu, logvar = poe(mu_list + [batch_mu], logvar_list + [batch_logvar], mask)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_embedding = z[:, :z.shape[1] - 2]
        z_conds = []
        start = z.shape[1] - 2
        z_batch = z[:, -2:]

        x_recon = self.decoder(z)
        x_batch_recon = self.batch_decoder(z_batch)

        cla_cty1 = 0
        cla_batch1 = self.batch1_classify(torch.cat([z_embedding] + z_conds, dim=1))
        cla_conditions = 0

        return x_recon, x_batch_recon, z_embedding, z_batch, z_conds, cla_cty1, cla_batch1, cla_conditions, mu, logvar
