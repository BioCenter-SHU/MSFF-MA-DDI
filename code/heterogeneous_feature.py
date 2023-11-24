import numpy as np
import matplotlib
import torch
matplotlib.use('agg')
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib
from tensorflow.python.keras.layers import Dropout, Activation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc,average_precision_score
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.metrics import accuracy_score
from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import label_binarize
from torch_geometric.nn.models import GAE,InnerProductDecoder
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
import os
import pandas as pd
from sklearn.model_selection import KFold
os.environ['KERAS_BACKEND'] = 'tensorflow'
from scipy.sparse import csr_matrix
from utils import *
from encoder import *

hid1 = 256
hid2 = 128
# 单个药物网络
hid3 = 170
# 两个
# hid3 = 512
droprate = 0.5
event_num = 65
node_feature = 548
def DNN():
    train_input = Input(shape=(1024,), name='Inputlayer')
    train_in = Dense(512, activation='relu')(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(256, activation='relu')(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(event_num)(train_in)
    out = Activation('softmax')(train_in)
    model = Model(inputs=train_input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[3] = f1_score(y_test, pred_type, average='macro')
    result_all[4] = precision_score(y_test, pred_type, average='macro')
    result_all[5] = recall_score(y_test, pred_type, average='macro')
    return result_all
class DenseDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(DenseDecoder, self).__init__()
        self.fullynet = torch.nn.Sequential(
            torch.nn.Linear(input_dim,512),
            # 526
            torch.nn.Dropout(p=0.1),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(p=0.1),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 65),
            torch.nn.Dropout(p=0.1),
            torch.nn.Sigmoid(),
        )
    def forward(self, feature):
        outputs = self.fullynet(feature)
        return outputs
def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
class AE(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(AE, self).__init__()
        self.vector_size = vector_size
        self.drop_out_rating = 0.1
        self.l1 = torch.nn.Linear(self.vector_size, 1024)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.l2 = torch.nn.Linear(1024, len_after_AE)
        self.l3 = torch.nn.Linear(len_after_AE, 1024)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.l4 = torch.nn.Linear(1024, self.vector_size)
        self.dr = torch.nn.Dropout(self.drop_out_rating)
        self.ac = torch.nn.ReLU()
    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.l2(X)
        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))
        X_AE = self.l4(X_AE)
        return X,X_AE
def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1
    return index_all_class
def save_result(result_type, result):
    with open(result_type + '_' + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0
def prepare_data1(fold,num_cross_val):
    drug = pd.read_csv('../data/drug572.csv')
    event = pd.read_csv('../data/event.csv')
    # print(drug)
    smiles = []
    with open("../data/smile572.txt") as f:
        for line in f:
            line = line.rstrip()
            smiles.append(line)
    # print(smiles)
    drug_smile_dict = dict([(drug_id, id) for drug_id, id in zip(drug['id'], drug['index'])])
    # print(drug_smile_dict)
    index1 = []
    index2 = []
    index_pair = []
    for i in event['id1']:
        index1.append(drug_smile_dict[i])
    for j in event['id2']:
        index2.append(drug_smile_dict[j])
    for i in range(0, len(index1)):
        index_pair.append([index1[i], index2[i]])
    label = np.loadtxt("../data/type572.txt", dtype=float, delimiter=" ")
    # print(label.shape)
    train_label = np.array([x for i, x in enumerate(label) if i % num_cross_val != fold])
    test_label = np.array([x for i, x in enumerate(label) if i % num_cross_val == fold])
    train_index = np.array([x for i, x in enumerate(index_pair) if i % num_cross_val != fold])
    test_index = np.array([x for i, x in enumerate(index_pair) if i % num_cross_val == fold])
    return smiles,train_index,test_index,train_label,test_label,index_pair,label

num_cross_val = 5
max_auc = 0.1
seed = 0
smile_embedding = np.loadtxt("../data/smile_embedding.txt", dtype=float, delimiter=" ")
smile_embedding = torch.tensor(smile_embedding, dtype=torch.float32)
target_embedding= np.loadtxt("../data/target_embedding.txt", dtype=float, delimiter=" ")
target_embedding = torch.tensor(target_embedding, dtype=torch.float32)
enzyme_embedding= np.loadtxt("../data/enzyme_embedding.txt", dtype=float, delimiter=" ")
enzyme_embedding = torch.tensor(enzyme_embedding, dtype=torch.float32)
drugsim_feature = torch.cat((smile_embedding,target_embedding,enzyme_embedding), dim=1)
print(drugsim_feature.shape)
len_after_AE = 512
model = GAE(AE(1716),  DenseDecoder(1024))
for fold in range(5):
    smiles,train_index,test_index,train_label,test_label,index_pair,label= prepare_data1(fold, num_cross_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)
    for epoch in range(800):
        print(epoch)
        model.train()
        optimizer.zero_grad()
        embedding,AE = model.encoder(drugsim_feature)
        # print(embedding.shape)
        drug1_train = train_index[:,0]
        drug2_train = train_index[:,1]
        drug1_emb_train = embedding[drug1_train, :]
        drug2_emb_train = embedding[drug2_train, :]
        drug_data = torch.cat((drug1_emb_train, drug2_emb_train), 1)
        pred_score = model.decoder(drug_data)
        l_pred_score = pred_score
        pred_score = pred_score.detach().numpy()
        pred_type = np.argmax(pred_score,axis=1)
        train_label = torch.LongTensor(train_label)
        criterion = nn.CrossEntropyLoss()
        criterion1 = torch.nn.MSELoss()
        loss = criterion(l_pred_score, train_label) + criterion1(drugsim_feature, AE)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.tolist())
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.zeros((0, 65), dtype=float)
        y_true = np.hstack((y_true, train_label))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
        all_eval_type = 6
        result_all = np.zeros((all_eval_type, 1), dtype=float)
        y_test = y_true
        y_one_hot = label_binarize(y_test, np.arange(65))
        # print(y_one_hot)
        pred_one_hot = label_binarize(pred_type, np.arange(65))
        # print(pred_one_hot.shape)
        result_all[0] = accuracy_score(y_test, pred_type)
        result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
        result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
        result_all[3] = f1_score(y_test, pred_type, average='macro')
        result_all[4] = precision_score(y_test, pred_type, average='macro')
        result_all[5] = recall_score(y_test, pred_type, average='macro')
        print('训练集')
        print(result_all)
        model.eval()
        drug1_test = test_index[:, 0]
        drug2_test = test_index[:, 1]
        drug1_emb_test = embedding[drug1_test, :]
        drug2_emb_test = embedding[drug2_test, :]
        drug_data = torch.cat((drug1_emb_test, drug2_emb_test), 1)
        pred_score = model.decoder(drug_data)
        l_pred_score = pred_score
        pred_score = pred_score.detach().numpy()
        pred_type = np.argmax(pred_score, axis=1)
        test_label = torch.LongTensor(test_label)
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.zeros((0, 65), dtype=float)
        y_true = np.hstack((y_true, test_label))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
        all_eval_type = 6
        result_all = np.zeros((all_eval_type, 1), dtype=float)
        y_test = y_true
        y_one_hot = label_binarize(y_test, np.arange(65))
        # print(y_one_hot)
        pred_one_hot = label_binarize(pred_type, np.arange(65))
        # print(pred_one_hot.shape)
        result_all[0] = accuracy_score(y_test, pred_type)
        result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
        result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
        result_all[3] = f1_score(y_test, pred_type, average='macro')
        result_all[4] = precision_score(y_test, pred_type, average='macro')
        result_all[5] = recall_score(y_test, pred_type, average='macro')
        print('测试集')
        print(result_all)
        if result_all[2] > max_auc:
            max_auc = result_all[2]
            heterogeneous_embedding = embedding
            heterogeneous_embedding = heterogeneous_embedding.detach().numpy()
            np.savetxt("hetbranch_embedding.txt",  heterogeneous_embedding, fmt="%6.4f")
    print("Optimization Finished!")
####=====================deep learning model to predict =============================
    hetbranch_embedding = np.loadtxt("hetbranch_embedding.txt", dtype=float, delimiter=" ")
    hetbranch_embedding = torch.tensor((hetbranch_embedding))
    print(hetbranch_embedding.shape)
    index_pair = np.array(index_pair)
    drug1 = index_pair[:, 0]
    drug2 = index_pair[:, 1]
    drug1_emb = hetbranch_embedding[drug1, :]
    drug2_emb = hetbranch_embedding[drug2, :]
    drug_data = torch.cat((drug1_emb, drug2_emb), 1)

    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label, event_num, seed, num_cross_val)
    train_index = np.where(index_all_class != fold)
    test_index = np.where(index_all_class == fold)
    pred = np.zeros((len(test_index[0]), event_num), dtype=float)
    x_train = drug_data[train_index]
    x_train = x_train.detach().numpy()
    x_test = drug_data[test_index]
    x_test = x_test.detach().numpy()
    y_train = label[train_index]
    y_train_one_hot = np.array(y_train)
    y_train_one_hot = (np.arange(y_train_one_hot.max() + 1) == y_train[:, None]).astype(dtype='float32')
    y_test = label[test_index]
    # one-hot encoding
    y_test_one_hot = np.array(y_test)
    y_test_one_hot = (np.arange(y_test_one_hot.max() + 1) == y_test[:, None]).astype(dtype='float32')
    # print(type(x_train))
    # print(type(x_test))
    # print(type(y_train))
    # print(type(y_test))
    dnn = DNN()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    dnn.fit(x_train, y_train_one_hot, batch_size=128, epochs=100, validation_data=(x_test, y_test_one_hot),
            callbacks=[early_stopping])
    pred += dnn.predict(x_test)
    pred_score = pred
    pred_type = np.argmax(pred_score, axis=1)
    y_true = np.hstack((y_true, y_test))
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, pred_score))
    result_all = evaluate(y_pred, y_score, y_true, event_num)
    print('最后')
    print(result_all)
    save_result('heterogeneous_result', result_all)
