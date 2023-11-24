from decoder import *
import torch
import torch.nn.functional as F
import numpy as np
class FusionNet(torch.nn.Module):
    def __init__(self, node_feature,hid1,hid2,hid3):
        super(FusionNet, self).__init__()
        self.encoder = FusionFea(node_feature,hid1,hid2)
        self.decoder = FulconDecoder(hid3)

    def forward(self, x,indices,sim_emb,output,row,col):
        drug_feature = self.encoder(x,indices,output,sim_emb)
        # print(drug_feature.shape)
        feature = torch.cat([drug_feature[row, :], drug_feature[col, :]], dim=1)
        pre_score = self.decoder(feature)
        return pre_score
class FusionFea(torch.nn.Module):
    def __init__(self, node_feature,hid1,hid2):
        super(FusionFea, self).__init__()
        self.encoder_1 = DD_Encoder(node_feature,hid1,hid2)
        self.encoder_2 = Graph_Atte_Encoder1()
    def forward(self, x, edge_index,data,sim_emb):
        drug_feature0 = self.encoder_1(x, edge_index)
        drug_feature0 = F.normalize(drug_feature0)
        # print(drug_feature0.shape)
        # drug_feature1 = self.encoder_2(data,x)
        drug_feature1 = self.encoder_2(data, drug_feature0)
        # drug_feature1 = F.normalize(drug_feature1)
        # print(data.shape)
        # print(x.shape)
        # print(drug_feature1.shape)
        # drug_feature = torch.cat((drug_feature0,drug_feature1),dim=1)
        # drug_feature = F.normalize(drug_feature,dim=0)
        # print(drug_feature.shape)
        return drug_feature1
class DrugFea(torch.nn.Module):
    def __init__(self):
        super(DrugFea, self).__init__()
        self.encoder_2 = Graph_Encoder()
    def forward(self,data):
        drug_feature = self.encoder_2(data)
        return drug_feature

class NetFea(torch.nn.Module):
    def __init__(self, node_feature,hid1,hid2):
        super(NetFea, self).__init__()
        self.encoder_1 = DD_Encoder(node_feature,hid1,hid2)

    def forward(self, x, edge_index):
        drug_feature = self.encoder_1(x, edge_index)
        drug_feature = F.normalize(drug_feature,dim=0)
        return drug_feature