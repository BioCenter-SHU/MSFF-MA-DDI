import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from MultiHeadAttention import *
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
def label_smiles(line, smi_ch_ind, length):
    X = np.zeros(length,dtype=np.int64())
    for i, ch in enumerate(line[:length]):
        X[i] = smi_ch_ind[ch]
    return X

# class PositionalEncoding(nn.Module):
#     """位置编码"""
#     #num+hiddens:向量长度  max_len:序列最大长度
#     def __init__(self, num_hiddens, dropout, max_len):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.P = torch.zeros((max_len, num_hiddens))
#         X = torch.arange(max_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(max_len, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
#
#         self.P[:, 0::2] = torch.sin(X)   #::2意为指定步长为2 为[start_index : end_index : step]省略end_index的写法
#         self.P[:, 1::2] = torch.cos(X)
#
#     def forward(self, X):
#         X = X + self.P[:X.shape[0], :].to(X.device)
#         return self.dropout(X)
# def label_smiles(line, smi_ch_ind, length):
#     X = np.zeros(length,dtype=np.int64())
#     for i, ch in enumerate(line[:length]):
#         X[i] = smi_ch_ind[ch]
#     return X
def PositionalEncoding(num_hiddens, max_len,embedding):
    P = torch.zeros((max_len, num_hiddens))
    X = torch.arange(max_len, dtype=torch.float32).reshape(
        -1, 1) / torch.pow(max_len, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
    P[:, 0::2] = torch.sin(X)
    P[:, 1::2] = torch.cos(X)
    embedding = embedding + P[:embedding.shape[0], :]
    return embedding

class cnn_selfatte_encoder1(torch.nn.Module):
    def __init__(self):
        super(cnn_selfatte_encoder1, self).__init__()
        self.dim = 64
        self.conv = 40
        self.CHARISOSMILEN = 64
        self.compound_max = 200
        self.drug_MAX_LENGH = 100
        self.drug_kernel = [4, 6, 8]
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        # self.pos_embedding = PositionalEncoding(64, 0)
        self.smi_dim = 64
        self.atte_dim = 64
        self.atte_heads = 8
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(
            self.drug_MAX_LENGH - self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3)

        self.attention = MultiHeadAttentionBlock(self.smi_dim, self.smi_dim, self.atte_dim, self.atte_heads)

    def forward(self, output):

        embdding_all = []
        data = output
        ids = len(data)
        for i in range(ids):

            drugstr = data[i]
            # print(drugstr)
            l = len(drugstr)
            # print(l)
            drugstr = torch.from_numpy(label_smiles(drugstr, CHARISOSMISET, l))
            drug_embed = self.drug_embed(drugstr)
            # print(drug_embed.shape)
            # print("?????????????")

            drug_embed = PositionalEncoding(64,l,drug_embed)
            # print(drug_embed.shape)
            # print("?????????????")
            drug_embed, atte = self.attention(drug_embed, drug_embed)
            # print(drug_embed.shape)
            if l >= 100:
                drug_embed = drug_embed[:100]
            else:
                temp = np.zeros((100 - l, 64))
                temp = torch.tensor(temp).long()
                drug_embed = torch.cat((drug_embed, temp), dim=0)
            # print(drug_embed.shape)

            drug_embed = drug_embed.unsqueeze(0)
            # print(drug_embed.shape)
            drug_embed = drug_embed.permute(0, 2, 1)
            cnn_embedding = self.Drug_CNNs(drug_embed)
            # print(cnn_embedding.shape)
            cnn_embedding = cnn_embedding.permute(0, 2, 1)
            # print(cnn_embedding.shape)
            # print("?????????????????????????")
            cnn_embedding = cnn_embedding.squeeze(0)
            # print(cnn_embedding.shape)
            # print("?????????????????????????")
            batch = np.zeros((cnn_embedding.shape[0],cnn_embedding.shape[1]))
            batch = torch.tensor(batch).long()
            cnn_embedding = gmp(cnn_embedding,batch)

            # cnn_embedding = self.Drug_max_pool(cnn_embedding).squeeze(2)
            # print(cnn_embedding.shape)
            # print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{")
            embdding_all.append(cnn_embedding)
        modular_feature = torch.cat(tuple(embdding_all))
        # modular_feature = nn.Dropout(0.3)(modular_feature)
        # modular_feature,atte = self.attention(modular_feature,modular_feature)
        return modular_feature
