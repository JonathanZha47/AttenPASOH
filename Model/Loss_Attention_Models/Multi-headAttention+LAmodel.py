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

"先定义多头缩放点积注意力计算"
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
class DualInputAttentionModel(nn.Module):
    def __init__(self, sim_input_dim, true_input_dim, output_dim, num_heads=3):
        super(DualInputAttentionModel, self).__init__()
        
        # Simulated data branch
        self.sim_fc = nn.Sequential(
            nn.Linear(sim_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # True data branch
        self.true_fc = nn.Sequential(
            nn.Linear(true_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 多头注意力机制
        self.attention = MultiHeadAttention(embed_dim=64, num_heads=num_heads)  # 将特征维度设置为64

        # Output layer
        self.fc_out = nn.Linear(32, output_dim)
    
    def forward(self, sim_data, true_data):
        # Process simulated data
        sim_features = self.sim_fc(sim_data)
        
        # Process true data
        true_features = self.true_fc(true_data)
        
        # Concatenate features
        combined_features = torch.cat((sim_features, true_features), dim=1).unsqueeze(1)
        
        # Apply multi-head attention
        attn_output, attn_weights = self.attention(combined_features)
        
        # Reduce dimensions by averaging the sequence length dimension
        attn_output = attn_output.mean(dim=0)
        
        # Final output
        output = self.fc_out(attn_output)
        
        return output, attn_weights

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)
    
class MultiLayerLinearLAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(MultiLayerLinearLAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,5),
            Sin(),
            nn.Linear(5,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)



class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        self.encoder = DualInputAttentionModel(sim_input_dim=17,true_input_dim=17,output_dim=32)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False, lan_lr_factor=None):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        self.lan_lr_factor = lan_lr_factor  # Factor by which LAN LR should be lower than F LR

        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
        self.num_epochs = 10000
        self.iter_per_epoch = iter_per_epoch

    def step(self):
        lr = self.lr_schedule[min(self.num_epochs * self.iter_per_epoch, len(self.lr_schedule) - 1)]
        if self.lan_lr_factor is not None:
            lr *= self.lan_lr_factor  # Scale the LR for LAN by the given factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        return self.current_lr


class LAPINN(nn.Module):
    def __init__(self,args):
        super(LAPINN, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.attention = DualInputAttentionModel(17, 17, 1).to(device)
        self.solution_u = Solution_u().to(device)
        self.dynamical_F = MLP(input_dim=99,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               droupout=0.2).to(device)
        self.lan_r = MultiLayerLinearLAN(1, args.LAN_hidden_dim, 1).to(device)
        self.lan_0 = MultiLayerLinearLAN(1, args.LAN_hidden_dim, 1).to(device)
        self.lan_b = MultiLayerLinearLAN(1, args.LAN_hidden_dim, 1).to(device)
        self.lan_l = MultiLayerLinearLAN(1, args.LAN_hidden_dim, 1).to(device)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        # optimizer related
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)
        self.optimizer_lan_0 = torch.optim.Adam(self.lan_0.parameters(), lr=args.lr_lan)
        self.optimizer_lan_b = torch.optim.Adam(self.lan_b.parameters(), lr=args.lr_lan)
        self.optimizer_lan_l = torch.optim.Adam(self.lan_l.parameters(), lr=args.lr_lan)
        self.optimizer_lan_r = torch.optim.Adam(self.lan_r.parameters(), lr=args.lr_lan)

        # Schedulers
        self.scheduler1 = LR_Scheduler(optimizer=self.optimizer1,
                                    warmup_epochs=args.warmup_epochs,
                                    warmup_lr=args.u_warmup_lr,
                                    num_epochs=args.epochs,
                                    base_lr=args.base_lr,
                                    final_lr=args.u_final_lr)

        self.scheduler2 = LR_Scheduler(optimizer=self.optimizer2,
                                    warmup_epochs=args.warmup_epochs,
                                    warmup_lr=args.F_warmup_lr,
                                    num_epochs=args.epochs,
                                    base_lr=args.base_lr,
                                    final_lr=args.F_final_lr)

        # Apply LAN LR factor to ensure LAN LR is always lower than F LR
        lan_lr_factor = 0.5  # Example: LAN LR is always half of F LR
        self.scheduler_lan_0 = LR_Scheduler(optimizer=self.optimizer_lan_0,
                                            warmup_epochs=args.warmup_epochs,
                                            warmup_lr=args.lan_warmup_lr,
                                            num_epochs=args.epochs,
                                            base_lr=args.base_lr,
                                            final_lr=args.lan_final_lr,
                                            lan_lr_factor=lan_lr_factor)

        self.scheduler_lan_b = LR_Scheduler(optimizer=self.optimizer_lan_b,
                                            warmup_epochs=args.warmup_epochs,
                                            warmup_lr=args.lan_warmup_lr,
                                            num_epochs=args.epochs,
                                            base_lr=args.base_lr,
                                            final_lr=args.lan_final_lr,
                                            lan_lr_factor=lan_lr_factor)

        self.scheduler_lan_l = LR_Scheduler(optimizer=self.optimizer_lan_l,
                                            warmup_epochs=args.warmup_epochs,
                                            warmup_lr=args.lan_warmup_lr,
                                            num_epochs=args.epochs,
                                            base_lr=args.base_lr,
                                            final_lr=args.lan_final_lr,
                                            lan_lr_factor=lan_lr_factor)

        self.scheduler_lan_r = LR_Scheduler(optimizer=self.optimizer_lan_r,
                                            warmup_epochs=args.warmup_epochs,
                                            warmup_lr=args.lan_warmup_lr,
                                            num_epochs=args.epochs,
                                            base_lr=args.base_lr,
                                            final_lr=args.lan_final_lr,
                                            lan_lr_factor=lan_lr_factor)
        


        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

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

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()
    
    def forward(self, xt):
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:]

        u = self.solution_u(torch.cat(x,t), dim=1)

        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True, allow_unused=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True, allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt, u, u_t, u_x], dim=1))

        f = u_t - F
        return u, f
    
    def calculate_se(self, u, y):
        return (u - y) ** 2

    def calculate_weighted_loss(self, se, lan):
        weights = lan(se.unsqueeze(-1)).squeeze()
        weighted_loss = (weights * se).mean()
        return torch.relu(weighted_loss)

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        loss4_meter = AverageMeter()
        true_label = []
        pred_label = []
        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            upred = self.predict(x1)
            pred_label.append(upred.cpu().detach().numpy())
            true_label.append(y1)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)

            se_l = self.calculate_se(u1, y1)
            se_0 = self.calculate_se(u2, y2)
            se_b = self.calculate_se(f1, torch.zeros_like(f1))
            se_r = self.calculate_se(f2, torch.zeros_like(f2))

            loss1_lan = self.calculate_weighted_loss(se_l, self.lan_l)
            loss2_lan = self.calculate_weighted_loss(se_0, self.lan_0)
            loss3_lan = self.calculate_weighted_loss(se_b, self.lan_b)
            loss4_lan = self.calculate_weighted_loss(se_r, self.lan_r)
             # Update LANs with gradient ascent
            self.optimizer_lan_0.zero_grad()
            self.optimizer_lan_b.zero_grad()
            self.optimizer_lan_l.zero_grad()
            self.optimizer_lan_r.zero_grad()

            (-loss1_lan).backward(retain_graph=True)
            (-loss2_lan).backward(retain_graph=True)
            (-loss3_lan).backward(retain_graph=True)
            (-loss4_lan).backward(retain_graph=True)

            self.optimizer_lan_0.step()
            self.optimizer_lan_b.step()
            self.optimizer_lan_l.step()
            self.optimizer_lan_r.step()

            # Update main network with gradient descent
            weighted_loss1 = self.calculate_weighted_loss(se_l, self.lan_l)
            weighted_loss2 = self.calculate_weighted_loss(se_0, self.lan_0)
            weighted_loss3 = self.calculate_weighted_loss(se_b, self.lan_b)
            weighted_loss4 = self.calculate_weighted_loss(se_r, self.lan_r)

            total_loss = weighted_loss1 + weighted_loss2 + weighted_loss3 + weighted_loss4

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            total_loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.solution_u.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.dynamical_F.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.lan_0.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.lan_b.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.lan_l.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.lan_r.parameters(), max_norm=1.0)

            loss1_meter.update(weighted_loss1.item())
            loss2_meter.update(weighted_loss2.item())
            loss3_meter.update(weighted_loss3.item())
            loss4_meter.update(weighted_loss4.item())

        return loss1_meter.avg, loss2_meter.avg,loss3_meter.avg,loss4_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3,loss4 = self.train_one_epoch(e,trainloader)
            current_lr1 = self.scheduler1.step()
            info = '[Train] epoch:{}, lr1:{:.6f}, total loss:{:.6f}'.format(
                e, current_lr1, loss1 + loss2 + loss3 + loss4 )
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
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
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

def evaluate_metrics(true_labels, pred_labels):
    return eval_metrix(pred_labels, true_labels)

def analyze_experiment(experiment_path):
    true_labels = np.load(os.path.join(experiment_path, 'true_label.npy'))
    pred_labels = np.load(os.path.join(experiment_path, 'pred_label.npy'))
    [MAE,MAPE,MSE,RMSE,AdjustRsquare,L1error,L2error] = evaluate_metrics(true_labels, pred_labels)
    return [MAE,MAPE,MSE,RMSE,AdjustRsquare,L1error,L2error]

def main(base_folder, output_path):
    results = []
    for batch in range(6):
        batch_folder = os.path.join(base_folder, f'{batch}-{batch}')
        for experiment in range(1, 11):
            experiment_folder = os.path.join(batch_folder, f'Experiment{experiment}')
            model_path = os.path.join(experiment_folder, 'model.pth')

            if os.path.exists(model_path):
                # Load the model and calculate metrics
                model = LAPINN(args)
                model.load_model(model_path)
                [MAE,MAPE,MSE,RMSE,AdjustRsquare,L1error,L2error] = analyze_experiment(experiment_folder)
                results.append((batch, experiment, MAE, MAPE, MSE, RMSE, AdjustRsquare, L1error, L2error))
     # Create DataFrame
    df = pd.DataFrame(results, columns=['Batch', 'Experiment', 'MAE', 'MAPE', 'MSE', 'RMSE', 'AdjustRsquare', 'L1error', 'L2error'])

    # Save to Excel
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=1, help='epoch')
        parser.add_argument('--base_lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--u_warmup_lr', type=float, default=0.0005, help='warmup lr of solution u')
        parser.add_argument('--u_final_lr', type=float, default=0.0001, help='final lr of solution u')
        parser.add_argument('--F_warmup_lr', type=float, default=0.001, help='warmup lr of dynamical F')
        parser.add_argument('--F_final_lr', type=float, default=0.0002, help='final lr of dynamical F')
        parser.add_argument('--lan_warmup_lr', type=float, default=0.0005, help='warmup lr of LAN')
        parser.add_argument('--lan_final_lr', type=float, default=0.0001, help='final lr of LAN')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')


        # model 相关
        ## attention
        parser.add_argument('--sim_input_dim', type=int, default=17, help='simulated data input dim')
        parser.add_argument('--true_input_dim', type=int, default=17, help='true data input dim')
        parser.add_argument('--attention_output_dim', type=int, default=32, help='output dim')

        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
        parser.add_argument('--LAN_hidden_dim', type=int, default=60, help='LAN hidden dim')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')   
        parser.add_argument('--results_output', type=str, default=None, help='Output Excel file')
        return parser.parse_args()


    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_folder = 'results of reviewer/XJTU(LAPINN) results4-L1-L2'
    results_output = 'results of reviewer/XJTU(LAPINN) results4-L1-L2.xlsx'
    setattr(args, "results_output", results_output)
    main(base_folder,args.results_output)




