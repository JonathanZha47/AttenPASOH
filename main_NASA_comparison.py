import torch
import torch.nn as nn
import numpy as np
import os
from utils.util import AverageMeter,get_logger
from Model.futureThoughts.Compare_Models import MLP,CNN
from Model.singleInput_MultiHeadAtt_PINN import SingleInputMultiHeadAtten
from Model.singleInput_Attention import Attention
from Model.CompareModels.BaselineModel import LR_Scheduler
from dataloader.NASAloader import NASAdata
from utils.util import eval_metrix
import argparse

class Trainer():
    def __init__(self,model,train_loader,valid_loader,test_loader,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.save_dir = args.save_folder
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))


        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.warmup_lr)
        self.scheduler = LR_Scheduler(optimizer=self.optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)
        
    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")


    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def train_one_epoch(self,epoch):
        self.model.train()
        self.loss_meter.reset()
        for (x,y) in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)
            loss = self.loss_func(y_pred,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_meter.update(loss.item())
        return self.loss_meter.avg

    def valid(self,epoch):
        self.model.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x,y) in self.valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                loss = self.loss_func(y_pred,y)
                self.loss_meter.update(loss.item())
                true_label.append(y)
                pred_label.append(y_pred.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return self.loss_meter.avg, mse.item()

    def test(self):
        self.model.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x,y) in self.test_loader:
                x = x.to(self.device)
                y_pred = self.model(x)

                true_label.append(y.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label

    def train(self):
        min_loss = 100
        early_stop = 0
        for epoch in range(1,self.epochs+1):
            early_stop += 1
            train_loss = self.train_one_epoch(epoch)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(epoch,current_lr,train_loss)
            self.logger.info(info)
            valid_loss, mse = self.valid(epoch)
            info = '[Valid] epoch:{}, MSE: {}'.format(epoch,mse)
            self.logger.info(info)
            if valid_loss < min_loss and self.test_loader is not None:
                min_loss = valid_loss
                true_label,pred_label = self.test()
                [MAE, MAPE, MSE, RMSE,Rsquare,L1,L2] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}, Rsquare: {:.6f}, L1 error:{:.6f}, L2 error:{:.6f}'.format(MSE, MAE, MAPE, RMSE,Rsquare,L1,L2)
                self.logger.info(info)
                early_stop = 0
            if early_stop > 20:
                break
        self.clear_logger()


def load_model(args):
    if args.model == 'MLP':
        model = MLP()
    elif args.model == 'CNN':
        model = CNN()
    elif args.model == 'attention':
        model = Attention()
        #model = SingleInputMultiHeadAtten(input_dim=49, d_model=64, n_heads=8, n_layers=4, d_ff=256, d_k=8, d_v=8, dropout=0.1,seq_length=10)
    return model

def load_data(args,small_sample=None):
    root = 'data/NASA data'
    data_list  = []
    for batch in ['Batch1','Batch2','Batch3','Batch4','Batch5','Batch6','Batch7','Batch9']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for f in files:
            data_list.append(os.path.join(batch_root, f))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    
    data = NASAdata(root=root, args=args)
    loader = data.read_all(specific_path_list=data_list)
    
    dataloader = {'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}
    
    return dataloader


def get_args():
    parser = argparse.ArgumentParser('The parameters of Comparision methods')
    parser.add_argument('--model',type=str,default='CNN',choices=['MLP','CNN','attention'])
    parser.add_argument('--dataset',type=str,default='NASA',choices=['XJTU','HUST','MIT','TJU','NASA'])
    parser.add_argument('--normalization_method',type=str, default='min-max', help='min-max,z-score')

    # XJTU data
    parser.add_argument('--xjtu_batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])

    # TJU data
    parser.add_argument('--in_same_batch',type=bool,default=True)
    parser.add_argument('--tju_batch',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--tju_train_batch',type=int,default=-1, choices=[-1,0,1,2])
    parser.add_argument('--tju_test_batch',type=int,default=-1, choices=[-1,0,1,2])

    # NASA data
    parser.add_argument('--nasa_batch',type=str,default='Batch1',choices=['Batch1','Batch2','Batch3','Batch4','Batch5','Batch6','Batch7','Batch9'])

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')


    parser.add_argument('--save_folder',type=str,default='./results of reviewer/')
    parser.add_argument('--log_dir',type=str,default='logging.txt')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--seq_length',type=int,default=10,help='seq_length')
    parser.add_argument('--stride',type=int,default=1,help='stride')

    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    return args


if __name__ == '__main__':
    args = get_args()
    nasa_batch_names = ['Batch1','Batch2','Batch3','Batch4','Batch5','Batch6','Batch7','Batch9']
    # tju_batch = [0,1,2] select model: MLP or CNN
    for i in range(len(nasa_batch_names)):
        setattr(args,'nasa_batch',nasa_batch_names[i])
        # setattr(args,'tju_batch',tju_batch[i])
        for e in range(10):
            setattr(args,'save_folder',os.path.join('./results of reviewer/',f'{args.dataset}-{args.model} results/{i}-{i}/Experiment{e+1}'))
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)

            model = load_model(args)
            data_loader = load_data(args)
            trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
            trainer.train()








