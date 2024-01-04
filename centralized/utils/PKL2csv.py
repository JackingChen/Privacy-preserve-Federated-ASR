import pickle
import pandas as pd
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_prompt_No0/test.pkl', help="[text | pred_str]")
args = parser.parse_args()

file_path=args.file_path
filename=os.path.basename(file_path)
output_root=os.path.dirname(file_path)
# 讀取 pickle 檔案
with open(file_path, 'rb') as file:
    data = pickle.load(file)


# 現在 'data' 變數中包含了你在 pickle 檔案中儲存的資料
# 將資料轉換成 DataFrame
df = pd.DataFrame(data)
# 僅保留 'path', 'text', 'dementia_labels', 'pred_str' 欄位
selected_columns = ['path', 'text', 'dementia_labels', 'pred_str',"Summary"]
df_selected = df[selected_columns]


# 存成 CSV 檔案
output_csv_path = f"{output_root}/{filename.replace('pkl','csv')}"
df_selected.to_csv(output_csv_path, index=False)
