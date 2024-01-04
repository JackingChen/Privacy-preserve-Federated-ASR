# predict AD using (masked) embeddings
import pandas as pd
import numpy as np
from numpy import nan
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from utils import ID2Label
import os
import pickle
def trainSVM(x_train, y_train, x_test, y_test, df_test, title, outdir="./saves/results/SVM"):
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    
    svm = SVC() #class_weight='balanced')
    svm.fit(x_train_std, y_train["dementia_labels"].values)
    
    pred = svm.predict(x_test_std)
    true = y_test["dementia_labels"].values
    """
    # utt-wise results
    cm = confusion_matrix(true, pred)
    
    # save results
    df = pd.read_csv("./saves/results/SVM/results.csv")                           # read in previous results
    new_row = {'model': title + " utt-wise",
               'ACC':accuracy_score(true, pred), 'BACC':balanced_accuracy_score(true, pred), 'F1':f1_score(true, pred),
               'Sens':recall_score(true, pred), 'Spec':cm[0,0]/(cm[0,0]+cm[0,1]), 'UAR': recall_score(true, pred, average='macro')}
                                                                                  # result to save
    df2 = pd.DataFrame([new_row])
    df3 = pd.concat((df, df2), axis = 0)                                          # append row
    df3.to_csv("./saves/results/SVM/results.csv", index=False)
    """
    sorted_dict = {}                                                              # sort based on spk id
    for idx, i in enumerate(df_test.index.tolist()): 
        id_part = df_test['path'][i].split('_')                                   # split file name   
        if id_part[1] == 'PAR':                                                   # predict only on participant
            if id_part[0] not in sorted_dict.keys():                              # new spk
                sorted_dict[id_part[0]] = [pred[idx]]                             # add values to this spk
            else:
                sorted_dict[id_part[0]].append(pred[idx])                         # append to existing list

    true = []                                                                     # ground truth
    pred = []                                                                     # prediction
    for spkid in sorted_dict.keys():                                              # for each spk
        true_label = ID2Label(spkid + '_PAR')                                     # get his/her label
        true.append(true_label)                                                   # add to list

        vote = sum(sorted_dict[spkid]) / len(sorted_dict[spkid])                  # average result of predictions
        if vote > 0.5:                                                            # over half of the pred is AD
            pred.append(1)                                                        # view as AD
        else:
            pred.append(0)                                                        # view as HC
    
    cm = confusion_matrix(true, pred)
    
    # save results
    df = pd.read_csv(f"{outdir}/results.csv")                           # read in previous results
    new_row = {'model': title + " spkid-wise",
               'ACC':accuracy_score(true, pred), 'BACC':balanced_accuracy_score(true, pred), 'F1':f1_score(true, pred),
               'Sens':recall_score(true, pred), 'Spec':cm[0,0]/(cm[0,0]+cm[0,1]), 'UAR': recall_score(true, pred, average='macro')}
                                                                                  # result to save
    df2 = pd.DataFrame([new_row])
    df3 = pd.concat((df, df2), axis = 0)                                          # append row
    print("The result: ", df2)
    df3.to_csv(f"{outdir}/results.csv", index=False)


def df2xy_masked(df_data, feat_col="masked_hidden_states"):
    # re = df_train.hidden_states * df_train.lm_mask
    re = df_data.hidden_states.copy() * df_data.lm_mask.copy()
    # re = df_data.hidden_states.copy()
    for idx, i in enumerate(df_data.index.tolist()):
        re[i] = pooling_func(re[i], axis=0)# 平均成(1, hidden_size)
        #  = data[0]                                                      # 轉成(hidden_size)
        #print(re[i].shape)0
        print("\r"+ str(idx+1), end="")
    print(" ")
    x_train = pd.DataFrame(re, columns=[feat_col])[feat_col].tolist() # masked_hidden_states to list
    y_train = pd.DataFrame(df_data["dementia_labels"])
    return x_train,y_train
parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model_name', type=str, default="data2vec-audio-large-960h", help="name of the desired model, ex: FSM_indiv_AMLoss_5_lmFSM")
parser.add_argument('-INV', '--INV', action='store_true', default=False, help="True: train w/ INV")
parser.add_argument('-sq', '--squeeze', type=str, default="min", help="way to squeeze hidden_states, 'mean', 'min', and 'max'")
parser.add_argument('-Audio_dataIn', '--Audio_dataIn_dir', type=str, default="/home/FedASR/dacs/centralized/saves/results/", help="")
parser.add_argument('-Lexical_dataIn', '--Lexical_dataIn_dir', type=str, default="/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings", help="")
parser.add_argument('-rsltOut', '--rsltOut_dir', type=str, default="/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/results/SVM/", help="")
parser.add_argument('--mode', default='text', help="True: train w/ INV")
args = parser.parse_args()
sqz = args.squeeze
mode=args.mode

Lexical_dataIn_dir=args.Lexical_dataIn_dir
model_name=args.model_name


# 注意df_text系列data是session level的
df_text_train = pd.read_pickle(f"{Lexical_dataIn_dir}/train.pkl")
df_text_test = pd.read_pickle(f"{Lexical_dataIn_dir}/test.pkl")

Str2Func={
    "mean":np.mean,
    "min":np.min,
    "max":np.max,
    "median":np.median,
}
pooling_func=Str2Func[sqz]

suffix='.pkl'
if not os.path.exists(args.rsltOut_dir):
    os.makedirs(args.rsltOut_dir)
# load in train / test data for certain model
# df_train = pd.read_csv(f"{args.Audio_dataIn_dir}" + args.model_name + "_train.csv")
# df_train = pd.read_csv(f"{args.Audio_dataIn_dir}" + args.model_name + ".csv") 
# df_test = pd.read_csv(f"{args.Audio_dataIn_dir}" + args.model_name + ".csv")
# training_pkl=f"{args.Audio_dataIn_dir}" + args.model_name + '_train' + suffix
# testing_pkl=f"{args.Audio_dataIn_dir}" + args.model_name + '' + suffix

# Audio的data寫死在這邊
training_pkl="/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h_train.pkl"
testing_pkl="/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h.pkl"
with open(training_pkl, "rb") as f:
    df_train = pickle.load(f)
with open(testing_pkl, "rb") as f:
    df_test = pickle.load(f)






if not args.INV:
    print("Train w/ PAR only...")
    df_train = df_train[df_train.path.str.contains("PAR")]                        # train w/ PAR only
    print("Remaining training samples: ", len(df_train))                          # show # of utt left
    df_test = df_test[df_test.path.str.contains("PAR")]                           # train w/ PAR only
    print("Remaining testing data: ", len(df_test))                               # show # of utt left

def df_fusion_2xy(df_data, df_text_data, main_feat_col="hidden_states",assist_feat_col='Embedding'):
    # 在 df_data 中新增 'session' 欄位
    df_data['session'] = df_data['path'].str.split('_').str[0]
    # 迭代 df_train，尋找相應的 'session' 並將 'embedding' 添加到 'embedding' 欄位中
    for index, row in df_data.iterrows():
        session = row['session']
        matching_row = df_text_data[df_text_data['session'] == session]
        
        audio_embedding=pooling_func(row['hidden_states'], axis=0)
        if not matching_row.empty: 
            text_embedding = np.array(matching_row[assist_feat_col].values[0])
            fusion_embedding=np.concatenate([audio_embedding,text_embedding],axis=0)

            df_data.at[index, 'hidden_states'] = fusion_embedding
    re = df_data[main_feat_col].copy()
    x_train = pd.DataFrame(re, columns=[main_feat_col])[main_feat_col].tolist()
    y_train = pd.DataFrame(df_data["dementia_labels"])
    return x_train,y_train
########################
def df2xy(df_data, feat_col="hidden_states"):
    re = df_data[feat_col].copy()
    for idx, i in enumerate(df_data.index.tolist()):
        re[i] = pooling_func(re[i], axis=0)# 平均成(1, hidden_size)
        #  = data[0]                                                      # 轉成(hidden_size)
        #print(re[i].shape)0
        print("\r"+ str(idx+1), end="")
    print(" ")
    x_train = pd.DataFrame(re, columns=[feat_col])[feat_col].tolist() # masked_hidden_states to list
    y_train = pd.DataFrame(df_data["dementia_labels"])
    return x_train,y_train
def df_text2xy(df_data, df_text_data,  main_feat_col="hidden_states",assist_feat_col='Embedding'):
    # 在 df_data 中新增 'session' 欄位
    df_data['session'] = df_data['path'].str.split('_').str[0]
    # 迭代 df_train，尋找相應的 'session' 並將 'embedding' 添加到 'embedding' 欄位中
    for index, row in df_data.iterrows():
        session = row['session']
        matching_row = df_text_data[df_text_data['session'] == session]
        
        # audio_embedding=pooling_func(row['hidden_states'], axis=0)
        if not matching_row.empty: 
            text_embedding = np.array(matching_row[assist_feat_col].values[0])
            # fusion_embedding=np.concatenate([audio_embedding,text_embedding],axis=0)
            # text_embedding.shape
            df_data.at[index, main_feat_col] = text_embedding
    re = df_data[main_feat_col].copy()
    x_train = pd.DataFrame(re, columns=[main_feat_col])[main_feat_col].tolist()
    y_train = pd.DataFrame(df_data["dementia_labels"])
    return x_train,y_train

Only_text=True
if mode=='fusion':
    x_train,y_train=df_fusion_2xy(df_train, df_text_train, main_feat_col="hidden_states",assist_feat_col='Embedding')
    x_test,y_test=df_fusion_2xy(df_test, df_text_test, main_feat_col="hidden_states",assist_feat_col='Embedding')
elif mode=='text':
    x_train,y_train=df_text2xy(df_train,df_text_train, main_feat_col="hidden_states",assist_feat_col='Embedding')
    x_test,y_test=df_text2xy(df_test,df_text_test, main_feat_col="hidden_states",assist_feat_col='Embedding')
else:
    x_train,y_train=df2xy(df_train, feat_col="hidden_states")
    x_test,y_test=df2xy(df_test, feat_col="hidden_states")

# Train SVM start here
title=args.model_name + "_ori_INV_" + str(args.INV) + "_" + sqz
outdir=args.rsltOut_dir

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

svm = SVC() #class_weight='balanced')
svm.fit(x_train_std, y_train["dementia_labels"].values)

pred = svm.predict(x_test_std)
true = y_test["dementia_labels"].values
"""
# utt-wise results
cm = confusion_matrix(true, pred)

# save results
df = pd.read_csv("./saves/results/SVM/results.csv")                           # read in previous results
new_row = {'model': title + " utt-wise",
            'ACC':accuracy_score(true, pred), 'BACC':balanced_accuracy_score(true, pred), 'F1':f1_score(true, pred),
            'Sens':recall_score(true, pred), 'Spec':cm[0,0]/(cm[0,0]+cm[0,1]), 'UAR': recall_score(true, pred, average='macro')}
                                                                                # result to save
df2 = pd.DataFrame([new_row])
df3 = pd.concat((df, df2), axis = 0)                                          # append row
df3.to_csv("./saves/results/SVM/results.csv", index=False)
"""
sorted_dict = {}                                                              # sort based on spk id
for idx, i in enumerate(df_test.index.tolist()): 
    id_part = df_test['path'][i].split('_')                                   # split file name   
    if id_part[1] == 'PAR':                                                   # predict only on participant
        if id_part[0] not in sorted_dict.keys():                              # new spk
            sorted_dict[id_part[0]] = [pred[idx]]                             # add values to this spk
        else:
            sorted_dict[id_part[0]].append(pred[idx])                         # append to existing list

true = []                                                                     # ground truth
pred = []                                                                     # prediction
for spkid in sorted_dict.keys():                                              # for each spk
    true_label = ID2Label(spkid + '_PAR')                                     # get his/her label
    true.append(true_label)                                                   # add to list

    vote = sum(sorted_dict[spkid]) / len(sorted_dict[spkid])                  # average result of predictions
    if vote > 0.5:                                                            # over half of the pred is AD
        pred.append(1)                                                        # view as AD
    else:
        pred.append(0)                                                        # view as HC

cm = confusion_matrix(true, pred)
if not os.path.exists(outdir):
    os.makedirs(outdir)
# save results
prev_savedFile=f"{outdir}/results.csv"
if not os.path.exists(prev_savedFile):
    df=pd.DataFrame()
else:
    df=pd.read_csv(prev_savedFile)  # read in previous results
new_row = {'model': title + " spkid-wise",
            'ACC':accuracy_score(true, pred), 'BACC':balanced_accuracy_score(true, pred), 'F1':f1_score(true, pred),
            'Sens':recall_score(true, pred), 'Spec':cm[0,0]/(cm[0,0]+cm[0,1]), 'UAR': recall_score(true, pred, average='macro')}
                                                                                # result to save
df2 = pd.DataFrame([new_row])
print("The result: ", df2)
df3 = pd.concat((df, df2), axis = 0)                                          # append row
df3.to_csv(f"{outdir}/results.csv", index=False)