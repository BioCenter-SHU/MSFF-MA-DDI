# MSFF-MA-DDI
This is the repository of the work "MSFF-MA-DDI: Multi-Source Feature Fusion with Multiple Attention blocks for predicting Drug-Drug Interaction events"

## 1.dependency
The packages required for this work can be found in "requirements.txt".

## 2.data
The dataset used in this work can be found in the "data" folder, including 572 drugs (drug572.csv), 37264 pairs of associated events (event.csv), 65 types corresponding to the events (type572.txt), similarity features (smile_embedding.txt, target_embedding.txt, enzyme_embedding.txt), SMILES sequences of the drugs (smile572.txt).

## 3.code
First, use python sequence_feature.py to get drug sequence features and use python heterogeneous_feature.py to get drug heterogeneous features.
Second, use python multi_source fusion.py to train and test the model under the warm start scenario.
Third, if you want to run task A, use python task_A.py. If you want to run task B, use python task_B.py.
