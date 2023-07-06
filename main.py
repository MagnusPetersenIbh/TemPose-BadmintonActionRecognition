import argparse
import numpy as np
import torch 
import torchvision
from torchvision import transforms
import os
import glob
import pandas as pd
import pickle
from models.TemPoseII import TemPose_TF #,TemPose-NF
import yaml
from data_tools.DataUtils import PoseData_OL,RandomScaling,RandomFlip,RandomTranslation,select_trainingtest,filterOL,one_hot_ol
#import wandb
def main(args):


    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize wandb project
    # wandb.init(
    #     project=args.project,
    #     tags=args.tags,
    #     name=args.run_name,
    #     config=config  # Pass the loaded configuration as hyperparameters
    # )

    # Extract model configuration from the loaded dictionary
    model_config = config["model"]

    #if args.hidden_dim is not None:
    #    model_config['head_dim'] = args.hidden_dim
    #if args.T_layers is not None:
    #    model_config['depth_t'] = args.T_layers
    #if args.N_layers is not None:
    #    model_config['depth_n'] = args.N_layers
        
    datasetname = config['dataset']['name']
    modeltype = model_config['model_name']
    match = config['dataset']['match']
    time_steps = model_config['sequence_length']
    num_people = model_config['num_people']

    if datasetname=="OL":
        poses = pd.read_pickle('Data/2d_poses.pkl')
        position = pd.read_pickle('Data/2d_positions.pkl')
        shuttle = pd.read_pickle('Data/2d_shuttle-position.pkl')
        labels = pd.read_pickle('Data/2d_labels.pkl')
        if config['dataset']['run'] == 'train-val':   
            train,val = select_trainingtest(poses,labels,position,shuttle,match)
            y_train = one_hot_ol(train[1])
            y_val = one_hot_ol(val[1])
            X_train,y_train,pos_train,shut_train = filterOL(train[0],y_train,train[2],train[3])
            X_val,y_val,pos_val,shut_val = filterOL(val[0],y_val,val[2],val[3])
            transform = transforms.Compose([RandomTranslation()])
            train_dataset = PoseData_OL(X_train,y_train,pos_train,shut_train,len_max=time_steps,modelname=modeltype,concat=True,interpolation=True,transform=transform)
            val_dataset = PoseData_OL(X_val,y_val,pos_val,shut_val,len_max=time_steps,modelname=modeltype,concat=True,interpolation=True,transform=None)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=int(config['dataset']['num_workers']/2))
            loader = (train_loader,val_loader)

        elif config['dataset']['run'] == 'train':
            train,_ = select_trainingtest(poses,labels,position,shuttle,match)
            y_train = one_hot_ol(train[1])
            X_train,y_train,pos_train,shut_train = filterOL(train[0],y_train,train[2],train[3])
            transform = transforms.Compose([RandomTranslation()])
            train_dataset = PoseData_OL(X_train,y_train,pos_train,shut_train,len_max=time_steps,modelname=modeltype,concat=True,interpolation=True,transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
            loader = train_loader
        else:
            print('Run command invalid!')
    else:
        #Implement your own dataset here
        print('Missing configuration for the following dataset')
    n_cls = model_config['output_dim']
    inp_dim = model_config['input_dim'] 
    d_t = model_config['depth_t']
    d_int = model_config['depth_n']
    d_l = model_config['model_dim']
    d_e = model_config['head_dim']
    drop = config['hyperparameters']['dropout']

    if modeltype == 'TF':
        model = TemPoseII_TF(poses_numbers=inp_dim, time_steps=time_steps,num_people=num_people,num_classes=n_cls,dim = d_l,depth=d_t,depth_int=d_int,dim_head=d_e,emb_dropout=drop,dataset=datasetname)
    elif modeltype == 'NF':
        model = TemPoseII_NF(poses_numbers=inp_dim, time_steps=time_steps,num_people=num_people,num_classes=n_cls,dim = d_l,depth=d_t,depth_int=d_int,dim_head=d_e,emb_dropout=drop,dataset=datasetname)
    else:
        print('Invalid model type')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = config['training']['run']

    train(model,loader,run,device,cfg=config)

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="PyTorch Model Configuration")

        # Add command-line arguments for wandb and the YAML configuration file
        #parser.add_argument("--T_layers", type=int, help="Number of temporal transformer layers")
        #parser.add_argument("--N_layers", type=int, help="Number of Interaction transformer layers")
        #parser.add_argument("--hidden_dim", type=int, help="Output dimension")
        #parser.add_argument("--project", type=str, required=True, help="Wandb project name")
        #parser.add_argument("--tags", nargs="+", help="Wandb tags")
        #parser.add_argument("--run_name", type=str, help="Optional wandb run name")
        parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

        args = parser.parse_args()
        main(args)

