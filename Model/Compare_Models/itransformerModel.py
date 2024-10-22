import torch
import torch.nn as nn
import torch.nn.functional as F
from iTransformer import iTransformer2D
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from utils.util import AverageMeter,get_logger,eval_metrix
import numpy as np


class itransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, args):
        super(itransformer, self).__init__()
        self.args = args
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
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)

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
        self.itransformer.load_state_dict(checkpoint['itransformer'])
        for param in self.parameters():
            param.requires_grad = True
    

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x,y) in enumerate(testloader):
                x = x.to(device)
                uDict = self.itransformer(x)
                u = uDict[1]
                u = u[:, :, -1]  # Shape becomes [512, 1]
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
                uDict = self.itransformer(x)
                u = uDict[1]
                u = u[:, :, -1]  # Shape becomes [512, 1]
                true_label.append(y)
                pred_label.append(u.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()



    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss_meter = AverageMeter()
        for iter,(x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            outputsDict = self.itransformer(x)
            outputs = outputsDict[20]
            outputs = outputs[:, :, -1]  # Shape becomes [512, 1]

            loss = self.loss_func(outputs,y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

        return loss_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss = self.train_one_epoch(e,trainloader)
            info = '[Train] epoch:{},  ' \
                   'total loss:{:.6f}'.format(e,loss)
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
                self.best_model = {'itransformer':self.itransformer.state_dict()}
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
        parser.add_argument('--epochs', type=int, default=1, help='epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
        parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
        parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

        parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
        parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        # basic config
        parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
        parser.add_argument('--model', type=str, required=True, default='iTransformer',
                            help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

        # data loader
        parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=20, help='start token length') # no longer needed in inverted Transformers
        parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')

        # model define
        parser.add_argument('--enc_in', type=int, default=17, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=32, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=1, help='output size') # applicable on arbitrary number of variates in inverted Transformers
        parser.add_argument('--d_model', type=int, default=17, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=60, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=3, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # iTransformer
        parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                            help='experiemnt name, options:[MTSF, partial_train]')
        parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
        parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
        parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
        parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
        parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
        parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                            'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
        return parser.parse_args()


