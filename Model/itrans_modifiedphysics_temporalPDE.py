import sys
sys.path.append('/Users/jonathanzha/Desktop/Meta-Learning-PINN-for-SOH')
from utils.util import AverageMeter, get_logger, eval_metrix
from iTransformer import iTransformer2D
import os


import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
    print("Cuda is available")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
from torch.autograd import grad
torch.autograd.set_detect_anomaly(True)
from Model.Modules.LRscheduler import LR_Scheduler
from Model.Modules.MLP import MLP
from Model.Modules.MLPsolutionu import Solution_u
from Model.Modules.attenSpatio_positionEmbedding import AttenSpatioEmbedding


class temporalPINN(nn.Module):
    def __init__(self, args):
        super(temporalPINN, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.alpha = nn.Parameter(torch.tensor(self.args.alpha))
        self.beta = nn.Parameter(torch.tensor(self.args.beta))
        self.gamma = nn.Parameter(torch.tensor(self.args.gamma))
        self.theta = nn.Parameter(torch.tensor(self.args.theta))
        self.seq_length = self.args.seq_length

        self.solution_u = Solution_u().to(device)
        self.dynamical_F = MLP(input_dim=35, output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               droupout=0.2).to(device)
        self.itransformer = iTransformer2D(
            num_variates = 18,
            num_time_tokens = 2,               # number of time tokens (patch size will be (look back length // num_time_tokens))
            lookback_len = 20,                  # the lookback length in the paper
            dim = 60,                          # model dimensions
            depth = 3,                          # depth
            heads = 8,                          # attention heads
            dim_head = 64,                      # head dimension
            pred_length = 20,     # can be one prediction, or many
            use_reversible_instance_norm = True # use reversible instance normalization
        ).to(device)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer_solutionu = torch.optim.Adam(self.solution_u.parameters(), lr=args.u_warmup_lr)
        self.optimizer_dynamical_F = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.F_warmup_lr)
        self.optimizer_itransformer = torch.optim.Adam(self.itransformer.parameters(), lr=args.F_warmup_lr)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer_solutionu,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.u_warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.base_lr,
                                      final_lr=args.u_final_lr)
        self.scheduler2 = LR_Scheduler(optimizer=self.optimizer_dynamical_F,
                                       warmup_epochs=args.warmup_epochs,
                                       warmup_lr=args.F_warmup_lr,
                                       num_epochs=args.epochs,
                                       base_lr=args.base_lr,
                                       final_lr=args.F_final_lr)
        self.scheduler3 = LR_Scheduler(optimizer=self.optimizer_itransformer,
                                       warmup_epochs=args.warmup_epochs,
                                       warmup_lr=args.F_warmup_lr,
                                       num_epochs=args.epochs,
                                       base_lr=args.base_lr,
                                       final_lr=args.F_final_lr)
        self.optimizer_weights = torch.optim.Adam([
            {'params': self.alpha},
            {'params': self.beta},
            {'params': self.gamma},
            {'params': self.theta}
        ], lr=0.001)


        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

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

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self, xt):
        return self.solution_u(xt)

    def Test(self, testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x, y) in enumerate(testloader):
                x = x.to(device)
                u = self.predict(x)
                true_label.append(y)
                pred_label.append(u.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label

    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x, y) in enumerate(validloader):
                x = x.to(device)
                u = self.predict(x)
                true_label.append(y)
                pred_label.append(u.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()
    
    def finite_difference(self, G, cycle):
        """
        Compute the finite difference approximation for ∂G/∂t.
        
        :param G: Tensor of shape [Batch_size, seq_length, 1], representing g values (e.g., SOH predictions).
        :param cycle: Tensor of shape [Batch_size, seq_length, 1], representing cycle numbers (e.g., charging cycle).
        :return: Tensor of finite difference approximations of ∂G/∂t, shape [Batch_size, seq_length - 1, 1].
        """
        # Remove the last dimension (1), so G and cycle become [Batch_size, seq_length]
        G = G.squeeze(-1)      # shape: [Batch_size, seq_length]
        cycle = cycle.squeeze(-1)  # shape: [Batch_size, seq_length]
        
        # Calculate finite differences for G (g) and cycle (t)
        delta_G = G[:, 1:] - G[:, :-1]         # g(i+1) - g(i), shape: [Batch_size, seq_length - 1]
        delta_t = cycle[:, 1:] - cycle[:, :-1] # t(i+1) - t(i), shape: [Batch_size, seq_length - 1]
        
        # Calculate the derivative ∂G/∂t by element-wise division
        dG_dt = delta_G / delta_t              # shape: [Batch_size, seq_length - 1]
        # Return the result with shape [Batch_size, seq_length - 1, 1]
        return dG_dt
    def resample_dg_dt(self, dg_dt, target_size):
        """
        Resample dg_dt to match the target size using linear interpolation.
        
        :param dg_dt: Tensor of shape [Batch_size, seq_length - 1] or similar
        :param target_size: The target sequence length (usually the length of u_t or u_x)
        :return: Resampled dg_dt tensor with the shape [Batch_size, target_size]
        """
        batch_size, seq_length = dg_dt.shape
        
        # Reshape dg_dt for interpolation: shape [Batch_size, 1, seq_length]
        dg_dt = dg_dt.unsqueeze(1)  # Add an extra dimension for interpolation
        
        # Perform linear interpolation to resize dg_dt to [Batch_size, 1, target_size]
        dg_dt_resampled = F.interpolate(dg_dt, size=target_size, mode='linear', align_corners=False)
        
        # Remove the extra dimension: shape [Batch_size, target_size]
        dg_dt_resampled = dg_dt_resampled.squeeze(1)
        
        return dg_dt_resampled
    def forward(self, xt, x_seq):
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:]
        u = self.solution_u(torch.cat((x, t), dim=1))
        u_t = grad(u.sum(), t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(), x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        outputsDict = self.itransformer(x_seq)
        outputs = outputsDict[20]
        G = outputs[:, :, -1]
        cycle = outputs[:, :, -2]
        dg_dt = self.finite_difference(G, cycle)
        # 重采样 dg_dt 使其与 u_t 的维度保持一致
        dg_dt_resampled = self.resample_dg_dt(dg_dt, target_size=u_t.shape[1])
        F = self.dynamical_F(torch.cat([xt, u, u_x, dg_dt_resampled], dim=1))
        # if seq_length = 3, then G is a 3*1 matrix
        # what we need is to construct the loss which is yt - yt-1 < 0 and yt-1 < t-2 < 0
        f = u_t - F
        return u, f, G

    def physics_loss(self, G: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute the Physics loss based on the difference between the trends (sign of differences) 
        between predicted sequence `G` and true sequence `y_seq`.

        :param G: Predicted sequence, shape [batch_size, seq_length, 1, 1]
        :param y_seq: True sequence, shape [batch_size, seq_length]
        :return: The computed physics loss.
        """
        # shape: [batch_size, seq_length]

        # Compute consecutive differences for G (predicted) and y_seq (true)
        g_diff = G[:, 1:] - G[:, :-1]  # shape: [batch_size, seq_length - 1]
        y_diff = y_seq[:, 1:] - y_seq[:, :-1]  # shape: [batch_size, seq_length - 1]

        # Apply the sign function to get the trend (1, 0, -1)
        g_sign = torch.sign(g_diff)  # shape: [batch_size, seq_length - 1]
        y_sign = torch.sign(y_diff)  # shape: [batch_size, seq_length - 1]

        # Compute the absolute difference between the trends of G and y_seq
        physics_loss_value = torch.abs(g_sign - y_sign).mean()

        return physics_loss_value

    def is_nan(self,tensor):
        return torch.isnan(tensor).any()

    def pad_or_trim_batch(self, x, y, x_seq, y_seq, target_batch_size):
        """
        Pad or trim batches to match the target batch size.
        
        :param x: Tensor for input x
        :param y: Tensor for labels y
        :param x_seq: Tensor for input x_seq
        :param y_seq: Tensor for labels y_seq
        :param target_batch_size: The desired batch size
        :return: Batches padded or trimmed to the target size
        """
        if x.size(0) < target_batch_size:
            # Pad x, y
            padding_size = target_batch_size - x.size(0)
            padding_x = torch.zeros((padding_size,) + x.shape[1:], device=x.device)
            padding_y = torch.zeros((padding_size,) + y.shape[1:], device=y.device)
            x = torch.cat([x, padding_x], dim=0)
            y = torch.cat([y, padding_y], dim=0)
        
        if x_seq.size(0) < target_batch_size:
            # Pad x_seq, y_seq
            padding_size = target_batch_size - x_seq.size(0)
            padding_x_seq = torch.zeros((padding_size,) + x_seq.shape[1:], device=x_seq.device)
            padding_y_seq = torch.zeros((padding_size,) + y_seq.shape[1:], device=y_seq.device)
            x_seq = torch.cat([x_seq, padding_x_seq], dim=0)
            y_seq = torch.cat([y_seq, padding_y_seq], dim=0)
        
        # Trim if necessary
        x = x[:target_batch_size]
        y = y[:target_batch_size]
        x_seq = x_seq[:target_batch_size]
        y_seq = y_seq[:target_batch_size]
        
        return x, y, x_seq, y_seq


    def train_one_epoch(self, epoch, dataloader, seqdataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        loss4_meter = AverageMeter()
        alpha_meter = AverageMeter()
        beta_meter = AverageMeter()
        theta_meter = AverageMeter()
        gamma_meter = AverageMeter()
        for (data, seqdata) in zip(dataloader, seqdataloader):
            self.optimizer_solutionu.zero_grad()
            self.optimizer_itransformer.zero_grad()
            self.optimizer_dynamical_F.zero_grad()
            self.optimizer_weights.zero_grad()

            x, y = data
            x_seq, y_seq = seqdata
            target_batch_size = min(x.size(0), x_seq.size(0))  # Set target batch size as the minimum of the two
            x, y, x_seq, y_seq = self.pad_or_trim_batch(x, y, x_seq, y_seq, target_batch_size)
            x, y = x.to(device), y.to(device)
            x_seq, y_seq = x_seq.to(device), y_seq.to(device)

            u, f, G= self.forward(x, x_seq)

            # data loss
            loss1 = self.loss_func(u, y)

            # PDE loss
            f_target = torch.zeros_like(f)
            loss2 = self.loss_func(f, f_target)
            # Physics loss
            loss3 = self.physics_loss(G, y_seq)
            
            # seq loss
            loss4 = self.loss_func(G, y_seq)
            # total loss
            loss_solution_u = self.alpha * loss1 + self.beta * loss2
            if self.is_nan(loss1) or self.is_nan(loss2):
                print(f"NaN detected in loss components at epoch {epoch}")
                print(f"loss1: {loss1.item()}, loss2: {loss2.item()}")
                return

            loss_solution_u = self.alpha * loss1 + self.beta * loss2

            if self.is_nan(loss_solution_u):
                print(f"NaN detected in combined loss at epoch {epoch}")
                return
            
            loss_itransformer = self.gamma * loss3 + self.theta * loss4
            
            loss_solution_u.backward(retain_graph=True)
            loss_itransformer.backward()
            self.optimizer_solutionu.step()
            self.optimizer_itransformer.step()
            self.optimizer_dynamical_F.step()
            self.optimizer_weights.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            loss4_meter.update(loss4.item())
            alpha_meter.update(self.alpha.item())
            beta_meter.update(self.beta.item())
            theta_meter.update(self.theta.item())
            gamma_meter.update(self.gamma.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            # if (iter+1) % 50 == 0:
            # print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}".format(epoch,iter+1,loss1,loss2))
        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, loss4_meter.avg, alpha_meter.avg, beta_meter.avg, gamma_meter.avg, theta_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None, seqtrainloader=None, seqvalidloader=None,
              seqtestloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.epochs + 1):
            early_stop += 1
            loss1, loss2, loss3, loss4, alpha, beta, gamma, theta = self.train_one_epoch(e, trainloader,seqtrainloader)
            current_lr_solutionu = self.scheduler.step()
            current_lr_itransformer = self.scheduler3.step()
            current_lr_dynamical_F = self.scheduler2.step()
            info = '[Train] epoch:{}, solution_u lr:{:.6f}, itransformer lr:{:.6f}, dynamical_F:{:.6f}, alpha:{:.4f} , beta:{:.4f}, gamma:{:.4f},theta:{:.4f} ' \
                   'total loss:{:.6f}'.format(e, current_lr_solutionu, current_lr_itransformer, current_lr_dynamical_F, alpha, beta, gamma, theta,
                                              self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3 + self.theta * loss4)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MSE, MAE, MAPE, RMSE, Rsquare, L1, L2] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}, Rsquare: {:.6f}, L1 error:{:.6f}, L2 error:{:.6f}'.format(
                    MSE, MAE, MAPE, RMSE, Rsquare, L1, L2)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
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
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))


if __name__ == "__main__":
    import argparse


    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=300, help='epoch')
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

        parser.add_argument('--alpha', type=float, default=1,
                            help='loss = alpha * l_data + beta * l_PDE + gamma * l_physics')
        parser.add_argument('--beta', type=float, default=1,
                            help='loss = alpha * l_data + beta * l_PDE + gamma * l_physics')
        parser.add_argument('--gamma', type=float, default=1,
                            help='loss = alpha * l_data + beta * l_PDE + gamma * l_physics')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        return parser.parse_args()

