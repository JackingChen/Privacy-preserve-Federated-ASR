# Main Structure
Git clone from this [GitHub](https://github.com/AshwinRJ/Federated-Learning-PyTorch), modify to our model

# Train client at the same time
Use python multiprocessing spawn, like [here](https://biicgitlab.ee.nthu.edu.tw/jack/dacs/-/blob/main/federated/Debug_multiprocess_loadmdl.ipynb)

# Client Split
- public: 54 PAR (50% of all training set)
- client 1: 27 PAR (25% of all training set) with 13 AD and 14 HC
- client 2: 27 PAR (25% of all training set) with 14 AD + 13 HC

# Training Process
1. Global train ASR
    - from `facebook/data2vec-audio-large-960h`
    - to `./save/data2vec-audio-large-960h_new1_recall_finetune_global/final/`


2. Global train AD classifier
    - from `./save/data2vec-audio-large-960h_new1_recall_finetune_global/final/`
    - to `./save/data2vec-audio-large-960h_new1_recall_global/final/`


3. FL train ASR
    - from `./save/data2vec-audio-large-960h_new1_recall_global/final/`
    - （`./save/data2vec-audio-large-960h_new1_recall_finetune_clientXXX_roundXXX/final/` in the middle）
    - to `./save/data2vec-audio-large-960h_new1_recall_FLASR_global/final/`


4. FL train AD
    - from `./save/data2vec-audio-large-960h_new1_recall_finetune_clientN_roundXXX/final/` for client N
    - （`./save/data2vec-audio-large-960h_new1_recall_clientXXX_roundXXX/final/` in the middle）
    - to `./save/data2vec-audio-large-960h_new1_recall_FLAD_global/final/`


5. Global train toggling network
    - from `./save/data2vec-audio-large-960h_new1_recall_FLAD_global/final/`
    - to `./save/data2vec-audio-large-960h_new2_recall_global/final/`


6. FL train toggling network
    - from `./save/data2vec-audio-large-960h_new1_recall_clientN_roundXXX/final/` for client N
        with global toggling network weight from `./save/data2vec-audio-large-960h_new2_recall_global/final/`
    - （`./save/data2vec-audio-large-960h_new2_recall_clientXXX_roundXXX/final/` in the middle）
    - to `./save/data2vec-audio-large-960h_new2_recall_final_global/final/`

