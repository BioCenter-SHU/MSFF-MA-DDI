import numpy as np
import csv
import torch
from torch import nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Query(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Query,self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim,hidden_dim)
    def forward(self,x):
        x = self.fc(x)
        return x
class Key(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Key,self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim,hidden_dim)
    def forward(self,x):
        x = self.fc(x)
        return x
class Value(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Value,self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim,hidden_dim)
    def forward(self,x):
        x = self.fc(x)
        return x
class AttentionBlock(torch.nn.Module):
    def __init__(self,sim_dim,smi_dim,atte_dim):
        super(AttentionBlock,self).__init__()
        self.query = Query(sim_dim,atte_dim)
        self.key = Key(smi_dim,atte_dim)
        self.value = Value(smi_dim,atte_dim)
        self.scale = torch.sqrt(torch.FloatTensor([atte_dim])).float()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,sim_emb,smi_emb):
        sim_Q = self.query(sim_emb)

        smi_K = self.key(smi_emb)

        smi_V = self.value(smi_emb)

        m = torch.matmul(sim_Q,smi_K.T).float()

        m = m/self.scale.float()
        attention = torch.softmax(m, dim=-1)
        # print(attention)
        # print(attention.shape)


        smi_atte_emb = torch.matmul(attention, smi_V)
        smi_atte_emb = self.dropout(smi_atte_emb)

        return smi_atte_emb,attention

class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self,sim_dim,smi_dim,atte_dim,n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(sim_dim,smi_dim,atte_dim))
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads*atte_dim,atte_dim)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,sim_emb,smi_emb):

        attentions = []
        a_emb = []
        modular_output = []
        for h in self.heads:
            smi_atte_emb, attention = h(sim_emb,smi_emb)
            # print(attention.shape)
            a_emb.append(smi_atte_emb)
            attention = attention.detach().numpy()
            attentions.append([attention])
        a_heads_emb = torch.cat(a_emb,dim=-1)
        smile_emb = self.fc(a_heads_emb)
        attention_ave = np.mean(attentions,axis=0)
        # print(attention_ave[0].shape)
        return smile_emb,attention_ave