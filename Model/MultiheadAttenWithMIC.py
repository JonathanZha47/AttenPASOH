import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import sys
sys.path.append('/Users/jonathanzha/Desktop/Meta-Learning-PINN-for-SOH')
from utils.util import AverageMeter,get_logger,eval_metrix
import torch
torch.autograd.set_detect_anomaly(True)
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from minepy import MINE  # MIC implementation
import scipy.stats as stats
from Model.Modules.LRscheduler import LR_Scheduler
from Model.Modules.MLP import MLP

class MICPearsonAttention(nn.Module):
    def __init__(self, seq_length, d_model, n_heads, d_key, d_value, dropout=0.1):
        super(MICPearsonAttention, self).__init__()
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.query = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.key = nn.Linear(d_model, n_heads* d_key, bias=False)
        self.value = nn.Linear(d_model, n_heads * d_value, bias=False)
        self.fc = nn.Linear(n_heads * d_value, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.mine = MINE()

    def scaled_dot_product_attention(self, Q, K, V):
        """
        print("-------- here we are doing similarity calculation --------")
        # Calculate Pearson correlation coefficient
        pearson_corr = torch.tensor([
        stats.pearsonr(Q[i].detach().cpu().numpy().flatten(), K[i].detach().cpu().numpy().flatten())[0]
        for i in range(Q.shape[0])
    ], dtype=torch.float32).to(Q.device)
        print("pearson_corr shape: ", pearson_corr.shape)
        print("pearson_corr calculation fine")
    # Calculate Maximum Information Coefficient (MIC)
        mic_score = []
        for i in range(Q.shape[0]):
            q_flat = Q[i].detach().cpu().numpy().flatten()
            k_flat = K[i].detach().cpu().numpy().flatten()
            print(f"Q[{i}] values: {q_flat}, K[{i}] values: {k_flat}")
            mic_value = self.mine.compute_score(q_flat, k_flat)
            if mic_value is None:
                raise ValueError(f"MINE compute_score returned None for Q[{i}] and K[{i}]")
            mic_score.append(mic_value)
        print("mic_score shape: ", mic_score.shape)
        print("mic_score calculation fine")
        # Combine the two metrics
        combined_score = pearson_corr + mic_score
        """

        raw_weights = torch.matmul(Q, K.transpose(-2, -1)) 
        scale_factor = K.size(-1) ** 0.5
        scaled_weights = raw_weights / scale_factor    
        # Apply softmax to the combined scores
        attention_weights = F.softmax(scaled_weights, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def pearson_correlation(self, x, y):
        # Compute Pearson correlation in PyTorch
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        mean_y = torch.mean(y, dim=-1, keepdim=True)
        xm = x - mean_x
        ym = y - mean_y
        r_num = torch.sum(xm * ym, dim=-1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=-1) * torch.sum(ym ** 2, dim=-1))
        r = r_num / r_den
        return r
    def forward(self, query, key, value):
        batch_size = query.size(0)
        """
        print("batch size: ", batch_size)
        print("query shape: ", query.shape)
        print("key shape: ", key.shape)
        print("value shape: ", value.shape)
        """
        residual = query
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        """
        print("Q shape: ", Q.shape)
        print("K shape: ", K.shape)
        print("V shape: ", V.shape)
        """
        Q = Q.view(batch_size, self.seq_length, self.n_heads, self.d_key).transpose(1, 2)
        K = K.view(batch_size, self.seq_length, self.n_heads, self.d_key).transpose(1, 2)
        V = V.view(batch_size, self.seq_length, self.n_heads, self.d_value).transpose(1, 2)
        """
        print("Q shape after transpose: ", Q.shape)
        print("K shape after transpose: ", K.shape)
        print("V shape after transpose: ", V.shape)
        """
        # Calculate Pearson correlation and MIC
        pearson_scores = []

        for i in range(self.n_heads):
            Q_flat = Q[:, i, :, :]  # [batch_size, seq_length, d_key]
            K_flat = K[:, i, :, :]  # [batch_size, seq_length, d_key]

            # Pearson correlation
            pearson_corr = self.pearson_correlation(Q_flat, K_flat)  # [batch_size, seq_length]
            pearson_scores.append(pearson_corr)


        pearson_scores = torch.stack(pearson_scores, dim=1)  # [batch_size, n_heads, seq_length]   

        # Combine Pearson and MIC scores (e.g., sum or weighted sum)
        similarity_scores = pearson_scores  # [batch_size, n_heads, seq_length]

        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarity_scores, dim=-1).unsqueeze(-1)  # [128, 8, 49, 1]

        # Apply attention weights to V
        outputs = torch.matmul(attention_weights.transpose(-2, -1), V)  # [128, 8, 1, 64]
        outputs = outputs.squeeze(-2)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, self.seq_length, self.d_model)
        outputs = self.dropout(self.fc(outputs))
        outputs = self.layer_norm(residual + outputs)
        return outputs 

class EncoderLayer(nn.Module):
    def __init__(self, seq_length, d_model, n_heads, d_ff, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MICPearsonAttention(seq_length, d_model, n_heads, d_k, d_v, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, seq_length, d_model, n_heads, n_layers, d_ff, d_k, d_v, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(seq_length, d_model, n_heads, d_ff, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class SingleInputMultiHeadAtten(nn.Module):
    def __init__(self, args, input_dim=1, d_model=64, n_heads=8, n_layers=4, d_ff=256, d_k=8, d_v=8, dropout=0.1, seq_length = 49):
        super(SingleInputMultiHeadAtten, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.encoder = Encoder(input_dim, seq_length, d_model, n_heads, n_layers, d_ff, d_k, d_v, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        """
                # scheduler 相关
        parser.add_argument('--epochs', type=int, default=500, help='epoch')
        parser.add_argument('--early_stop', type=int, default=40, help='early stop')
        parser.add_argument('--base_lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
        parser.add_argument('--u_warmup_lr', type=float, default=0.002, help='warmup lr of solution u')
        parser.add_argument('--u_final_lr', type=float, default=0.0002, help='final lr of solution u')
        parser.add_argument('--F_warmup_lr', type=float, default=0.002, help='warmup lr of dynamical F')
        parser.add_argument('--F_final_lr', type=float, default=0.0002, help='final lr of dynamical F')
        parser.add_argument('--lan_warmup_lr', type=float, default=0.001, help='warmup lr of LAN')
        parser.add_argument('--lan_final_lr', type=float, default=0.0005, help='final lr of LAN')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.u_warmup_lr)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.u_warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.base_lr,
                                      final_lr=args.u_final_lr)
        
        self.loss_func = nn.MSELoss()
        self.loss_meter = AverageMeter()
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

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x,y) in enumerate(testloader):
                x = x.to(device)
                u = self.forward(x)
                true_label.append(y)
                pred_label.append(u.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x,y) in enumerate(validloader):
                x = x.to(device)
                u = self.forward(x)
                true_label.append(y)
                pred_label.append(u.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()
    
    def forward(self, x):
        encoded_x = self.encoder(x)
        pooled = encoded_x.mean(dim=1)
        output = self.decoder(pooled)
        return output
    
    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss_meter = AverageMeter()
        for iter,(x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            u = self.forward(x)
            # data loss
            loss = self.loss_func(u, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_meter.update(loss.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}".format(epoch,iter+1,loss))
        return loss_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE,Rsquare,L1,L2] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}, Rsquare: {:.6f}, L1 error:{:.6f}, L2 error:{:.6f}'.format(MSE, MAE, MAPE, RMSE,Rsquare,L1,L2)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'encoder':self.encoder.state_dict(),
                                   'decoder':self.decoder.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))

 
    

if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=500, help='epoch')
        parser.add_argument('--early_stop', type=int, default=40, help='early stop')
        parser.add_argument('--base_lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
        parser.add_argument('--u_warmup_lr', type=float, default=0.002, help='warmup lr of solution u')
        parser.add_argument('--u_final_lr', type=float, default=0.0002, help='final lr of solution u')
        parser.add_argument('--F_warmup_lr', type=float, default=0.002, help='warmup lr of dynamical F')
        parser.add_argument('--F_final_lr', type=float, default=0.0002, help='final lr of dynamical F')
        parser.add_argument('--lan_warmup_lr', type=float, default=0.001, help='warmup lr of LAN')
        parser.add_argument('--lan_final_lr', type=float, default=0.0005, help='final lr of LAN')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')

        parser.add_argument('--alpha', type=float, default=1, help='loss = alpha * l_data + beta * l_PDE + gamma * l_physics')
        parser.add_argument('--beta', type=float, default=1, help='loss = alpha * l_data + beta * l_PDE + gamma * l_physics')
        parser.add_argument('--gamma', type=float, default=1, help='loss = alpha * l_data + beta * l_PDE + gamma * l_physics')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        return parser.parse_args()