import torch
from torch import nn
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout, Activation
class MlpDecoder(torch.nn.Module):
    """
    MLP decoder
    return drug-disease pair predictions
    """

    def __init__(self, input_dim):
        super(MlpDecoder, self).__init__()
        self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim * 2), int(input_dim)),
                                   nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim), int(input_dim // 2)),
                                   nn.ReLU())
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, drug_feature, disease_feature):
        pair_feature = torch.cat([drug_feature, disease_feature], dim=1)
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        outputs = self.mlp_3(embedding_2)
        return outputs
# class DenseDecoder(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(DenseDecoder, self).__init__()
#         self.densenet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim,400),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.Linear(400, 300),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.Linear(300, 2),
#             torch.nn.Sigmoid(),
#         )
#     def forward(self, feature):
#         outputs = self.densenet(feature)
#         return outputs
class DenseDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(DenseDecoder, self).__init__()
        self.densenet = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 400),
            nn.BatchNorm1d(400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 2),
            torch.nn.Sigmoid(),
        )
    def forward(self, feature):
        outputs = self.densenet(feature)
        return outputs

class FulconDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(FulconDecoder, self).__init__()
        self.fullynet = torch.nn.Sequential(
            torch.nn.Linear(input_dim,128),
            torch.nn.Dropout(p=0.3),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(p=0.3),
            nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Sigmoid(),
        )
    def forward(self, feature):
        outputs = self.fullynet(feature)
        return outputs
class multi_class_FulconDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(multi_class_FulconDecoder, self).__init__()
        self.fullynet = torch.nn.Sequential(
            torch.nn.Linear(input_dim,256),
            # torch.nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            # torch.nn.Dropout(p=0.3),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 65),
            # torch.nn.Dropout(p=0.3),
            torch.nn.Sigmoid(),
        )
    def forward(self, feature):
        outputs = self.fullynet(feature)
        return outputs
class FulconDecoder1(torch.nn.Module):
    def __init__(self, input_dim):
        super(FulconDecoder1, self).__init__()
        self.fullynet = torch.nn.Sequential(
            torch.nn.Linear(input_dim,128),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1),
            torch.nn.Sigmoid(),
        )
    def forward(self, feature):
        outputs = self.fullynet(feature)
        return outputs
# class FulconDecoder1(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(FulconDecoder1, self).__init__()
#         self.fullynet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim,512),
#             nn.BatchNorm1d(512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 64),
#             nn.BatchNorm1d(64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, 2),
#             torch.nn.Sigmoid(),
#         )
#     def forward(self, feature):
#         outputs = self.fullynet(feature)
#         return outputs
# class FulconDecoder(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(FulconDecoder, self).__init__()
#         self.fullynet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 400),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.Linear(400, 200),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.Linear(200, 2),
#             torch.nn.Sigmoid(),
#         )
#     def forward(self, feature):
#         outputs = self.fullynet(feature)
#         return outputs

