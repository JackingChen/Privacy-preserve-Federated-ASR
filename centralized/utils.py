import argparse
import os
import pandas as pd
from datasets import Dataset, load_from_disk
import librosa
import scipy.io.wavfile
import numpy as np
# =======================================
# 有一些common的function丟到這邊
# =======================================
parser = argparse.ArgumentParser()
#parser.add_argument('-model', '--model_path', type=str, default="./saves/wav2vec2-base-960h_GRL_0.5", help="Where the model is saved")
parser.add_argument('-opt', '--optimizer', type=str, default="adamw_hf", help="The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor")
parser.add_argument('-MGN', '--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
parser.add_argument('-sr', '--sampl_rate', type=float, default=16000, help="librosa read smping rate")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning rate")
parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--AudioLoadFunc', default='librosa', help="用scipy function好像可以比較快")
args = parser.parse_args(args=[])


# def csv2dataset(audio_path = '{}/clips/'.format(args.root_dir),
#                 csv_path = '{}/mid_csv/test.csv'.format(args.root_dir)):
#     stored = "./dataset/" + csv_path.split("/")[-1].split(".")[0]
#     if (os.path.exists(stored)):
#         #print("loaded")
#         return load_from_disk(stored)
        
#     data = pd.read_csv(csv_path)                                                # read desired csv
#     dataset = Dataset.from_pandas(data)                                     # turn into class dataset
    
#     # initialize a dictionary
#     my_dict = {}
#     my_dict["path"] = []                                                   # path to audio
#     my_dict["array"] = []                                                  # waveform in array
#     my_dict["text"] = []                                                   # ground truth transcript

#     i = 1
#     for file_path in dataset['path']:                                           # for all files
#         if dataset['sentence'][i-1] != None:                               # only the non-empty transcript
#             if args.AudioLoadFunc == 'librosa':
#                 sig, s = librosa.load('{0}/{1}'.format(audio_path,file_path), sr=args.sampl_rate, dtype='float32')  # read audio w/ 16k sr
#             else:
#                 s, sig = scipy.io.wavfile.read('{0}/{1}'.format(audio_path,file_path))
#                 sig=librosa.util.normalize(sig)
#             my_dict["path"].append(file_path)                                   # add path
#             my_dict["array"].append(sig)                                   # add audio wave
#             my_dict["text"].append(dataset['sentence'][i-1].upper())       # transcript to uppercase
#         print(i, end="\r")                                                 # print progress
#         i += 1
#     print("There're ", len(my_dict["path"]), " non-empty files.")

#     result_dataset = Dataset.from_dict(my_dict)

#     return result_dataset
def ID2Label(ID,
            spk2label = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/test_dic.npy", allow_pickle=True).tolist()):
    name = ID.split("_")                                                    #  from file name to spkID
    if (name[1] == 'INV'):                                                  # interviewer is CC
        label = 0
    else:                                                                   # for participant
        label = spk2label[name[0]]                                          # label according to look-up table
    return label                                                            # return dementia label for this file

def csv2dataset(audio_path = '{}/clips/'.format(args.root_dir),
                csv_path = '{}/mid_csv/test.csv'.format(args.root_dir)):
    stored = "./dataset/" + csv_path.split("/")[-1].split(".")[0]
    if (os.path.exists(stored)):
        print("Load data from local...")
        return load_from_disk(stored)
 
    data = pd.read_csv(csv_path)                                                # read desired csv
    dataset = Dataset.from_pandas(data)                                     # turn into class dataset
    
    # initialize a dictionary
    my_dict = {}
    my_dict["path"] = []                                                    # path to audio
    my_dict["array"] = []                                                   # waveform in array
    my_dict["text"] = []                                                    # ground truth transcript
    my_dict["dementia_labels"] = []

    spk2label=np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/test_dic.npy", allow_pickle=True).tolist()

    i = 1
    for file_path in dataset['path']:                                            # for all files
        if dataset['sentence'][i-1] != None:                                # only the non-empty transcript
            if args.AudioLoadFunc == 'librosa':
                sig, s = librosa.load('{0}/{1}'.format(audio_path,file_path), sr=args.sampl_rate, dtype='float32')  # read audio w/ 16k sr
            else:
                s, sig = scipy.io.wavfile.read('{0}/{1}'.format(audio_path,file_path))
                sig=librosa.util.normalize(sig)
            if len(sig) > 1600:                                             # get rid of audio that's too short
                my_dict["path"].append(file_path)                                # add path
                my_dict["array"].append(sig)                                # add audio wave
                my_dict["text"].append(dataset['sentence'][i-1].upper())    # transcript to uppercase
                my_dict["dementia_labels"].append(ID2Label(ID=file_path,
                                                           spk2label=spk2label))
        print(i, end="\r")                                                  # print progress
        i += 1
    print("There're ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)                                     # save for later use
    
    return result_dataset

def WriteResult(result,Save_path):
    df_results=pd.DataFrame([result['text'],result['pred_str']], index=['GroundTruth','PredStr']).T
    df_results.to_csv('{}/Result.csv'.format(Save_path))
    print("Writing results to {}".format(Save_path))


