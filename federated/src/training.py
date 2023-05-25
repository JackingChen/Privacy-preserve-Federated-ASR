from options import args_parser
from update import test_inference, ASRLocalUpdate, get_model_weight
from utils import get_dataset, average_weights, exp_details
from ASRLocalUpdate_Multitask import ASRLocalUpdate_multi

def client_train(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, 
                 test_dataset, idx, epoch, global_weights=None):                      # train function for each client, train from model in model_in_path 
                                                                                      #                                                                    + "_global/final/"
                                                                                      #                                                                    + "_client" + str(idx) + "_round" + str(args.epochs-1) + "/final/"
                                                                                      # or model in last round
                                                                                      # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round)
    #logger = SummaryWriter('../logs/' + model_out_path + "_client" + str(idx) + "_round" + str(epoch) ) 
                                                                                      # current training process
    # BUILD MODEL for every process
    if epoch == 0:                                                                    # start from global model
        if args.STAGE == 0:                                                           # train ASR
            model_in_path = model_in_path_root + "_global/final/"                     # using global trained ASR + AD
        else:                                                                         # train AD or toggling network
            model_in_path = model_in_path_root + "_client" + str(idx) + "_round" + str(args.epochs-1) + "/final/"
                                                                                      # using final local model from last training
    else:                                                                             # start from previous local model
        model_in_path = model_out_path + "_client" + str(idx) + "_round" + str(epoch-1) + "/final/"
    
    local_model = ASRLocalUpdate(args=args, dataset_supervised=train_dataset_supervised, dataset_unsupervised=train_dataset_unsupervised,
                                 global_test_dataset=test_dataset, client_id=idx, model_in_path=model_in_path, model_out_path=model_out_path)
                                                                                      # initial dataset of current client

    w, loss = local_model.update_weights(global_weights=global_weights, global_round=epoch) 
                                                                                      # from model_in_path model, update certain part using given weight
    

    return w, loss

def centralized_training(args, model_in_path, model_out_path, train_dataset, 
                         test_dataset, epoch, client_id="public"):                    # train function for global models, train from model in model_in_path
                                                                                      # final result in model_out_path + "_global/final"
    #logger = SummaryWriter('../logs/' + model_out_path + "_global")                   # current training process

    local_model = ASRLocalUpdate(args=args, dataset=train_dataset,
                        global_test_dataset=test_dataset, client_id=client_id, 
                        model_in_path=model_in_path, model_out_path=model_out_path)   # initial public dataset
    
    w, loss = local_model.update_weights(global_weights=None, global_round=epoch)     # from model_in_path to train

    return w, loss

def unsupervised_client_train(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised,
                              test_dataset, idx, epoch, fully_unsupervised, init_head, global_weights=None):  
                                                                                      # train function for each unsupervised client, train from model in model_in_path 
                                                                                      #                                                                    + "_global/final/"
                                                                                      #                                                                    + "_client" + str(idx) + "_round" + str(args.epochs-1) + ("_unsuper") + "/final/"
                                                                                      # or model in last round
                                                                                      # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round) + ("_unsuper")
    # BUILD MODEL for every process
    if epoch == 0:                                                                    # start from global model
        init_head = 1                                                                 # train from model w/ single lm_head, init_head needed  
        if args.STAGE == 0:                                                           # train ASR
            model_in_path = model_in_path_root + "_global/final/"                     # using global trained ASR + AD
                                                            
    else:                                                                             # start from previous local model
        if fully_unsupervised:
            model_in_path = model_out_path + "_client" + str(idx) + "_round" + str(epoch-1) + "_unsuper/final/"
        else:
            model_in_path = model_out_path + "_client" + str(idx) + "_round" + str(epoch-1) + "/final/"
            init_head = 1                                                             # train from model w/ single lm_head, init_head needed  

    

    local_model = ASRLocalUpdate_multi(args=args, dataset_supervised=train_dataset_supervised, dataset_unsupervised=train_dataset_unsupervised,
                                       global_test_dataset=test_dataset, client_id=idx, model_in_path=model_in_path, model_out_path=model_out_path)
                                                                                      # initial dataset of current client
    w, loss = local_model.update_weights_adapted(global_weights=global_weights, global_round=epoch, init_head=init_head, fully_unsupervised=fully_unsupervised) 
                                                                                      # from model_in_path model, update certain part using given weight
    return w, loss
