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
from model import *
from utils import *
from encoder import *
event_num = 65
droprate = 0.3
vector_size = 572
def DNN():
    train_input = Input(shape=(1344,), name='Inputlayer')
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
def cross_validation(feature, label,drugA,drugB,event_num):
    cross_ver_tim = 5
    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])

    # # cro val
    # temp_drug1 = [[] for i in range(event_num)]
    # temp_drug2 = [[] for i in range(event_num)]
    # for i in range(len(label)):
    #     temp_drug1[label[i]].append(drugA[i])
    #     temp_drug2[label[i]].append(drugB[i])
    # drug_cro_dict = {}
    # for i in range(event_num):
    #     for j in range(len(temp_drug1[i])):
    #         drug_cro_dict[temp_drug1[i][j]] = j % cross_ver_tim
    #         drug_cro_dict[temp_drug2[i][j]] = j % cross_ver_tim
    # train_drug = [[] for i in range(cross_ver_tim)]
    # test_drug = [[] for i in range(cross_ver_tim)]
    # for i in range(cross_ver_tim):
    #     for dr_key in drug_cro_dict.keys():
    #         if drug_cro_dict[dr_key] == i:
    #             test_drug[i].append(dr_key)
    #         else:
    #             train_drug[i].append(dr_key)
    #
    # for cross_ver in range(1):
    #     X_train = []
    #     X_test = []
    #     y_train = []
    #     y_test = []
    drug = pd.read_csv('../data/drug572.csv')
    drug_cro_dict = dict([(drug_id, id) for drug_id, id in zip(drug['name'], drug['index'])])
    print(drug_cro_dict)
    train_drug = [[] for i in range(cross_ver_tim)]
    test_drug = [[] for i in range(cross_ver_tim)]
    for i in range(cross_ver_tim):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] % cross_ver_tim == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)
        print(len(train_drug[i]))
        print(len(test_drug[i]))
    for cross_ver in range(1):
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(len(drugA)):
            if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                X_train.append(feature[i])
                y_train.append(label[i])

            if (drugA[i] not in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                X_test.append(feature[i])
                y_test.append(label[i])

            if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] not in np.array(train_drug[cross_ver])):
                X_test.append(feature[i])
                y_test.append(label[i])

        print("train len", len(y_train))
        print("test len", len(y_test))
        pred = np.zeros((len(y_test), event_num), dtype=float)
        x_train = np.array(X_train)
        print(x_train.shape)
        x_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        # np.savetxt("task2_test_embedding.txt", x_test, fmt="%6.4f")
        # np.savetxt("task2_test_label.txt", y_test, fmt="%6.4f")
        # one-hot encoding
        y_train_one_hot = y_train
        y_train_one_hot = (np.arange(64 + 1) == y_train[:, None]).astype(dtype='float32')
        # one-hot encoding
        y_test_one_hot = y_test
        y_test_one_hot = (np.arange(64 + 1) == y_test[:, None]).astype(dtype='float32')
        dnn = DNN()
        # print(x_train)
        # print("???????????????????????")
        # print(x_test)
        # print("???????????????????????")
        # print(y_train_one_hot)
        # print("???????????????????????")
        # print(y_test_one_hot)
        # print("???????????????????????")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        dnn.fit(x_train, y_train_one_hot, batch_size=32, epochs=100, validation_data=(x_test, y_test_one_hot),
                callbacks=[early_stopping])
        pred += dnn.predict(x_test)
        pred_score = pred
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
    return y_pred, y_score, y_true
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
    return smiles,index_pair,label
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
sequ_hete_embedding = np.loadtxt("sequ_hete_embedding.txt",dtype=float, delimiter=" ")
sequ_hete_embedding = torch.tensor(sequ_hete_embedding)
print(sequ_hete_embedding.shape)
num_cross_val = 5
event_num = 65
event = pd.read_csv("../data/event.csv")
# print(event)
drugA = event['name1']
# print(drugA)
drugB = event['name2']
for fold in range(5):
    smiles,index_pair,label= prepare_data1(fold, num_cross_val)
    index_pair = np.array(index_pair)
    drug1 = index_pair[:, 0]
    drug2 = index_pair[:, 1]
    drug1_emb = sequ_hete_embedding[drug1, :]
    drug2_emb = sequ_hete_embedding[drug2, :]
    drug_data = torch.cat((drug1_emb, drug2_emb), 1)
    drug_data = np.array(drug_data)
    y_pred, y_score, y_true = cross_validation(drug_data, label, drugA, drugB, event_num)
    result_all = evaluate(y_pred, y_score, y_true, event_num)
    print('最后')
    print(result_all)
    save_result('task A_result', result_all)