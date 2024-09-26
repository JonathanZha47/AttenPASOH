from dataloader.NASAloader import NASAdata
from Model.Loss_Attention_Models.LAPINN_NASA import LAPINN
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def main():
    args = get_args()
    batchs = ['Batch1','Batch2','Batch3','Batch4','Batch5','Batch6','Batch7','Batch9']
    for i in range(8):
        batch = batchs[i]
        setattr(args, 'batch', batch)
        for e in range(10):
            save_folder = 'results of reviewer/NASA(LAPINNwithAdjustedLR) results3/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_dir = 'logging.txt'
            setattr(args, "save_folder", save_folder)
            setattr(args, "log_dir", log_dir)

            dataloader = load_data(args)
            pinn = LAPINN(args)
            pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])

def small_sample():
    args = get_args()
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for n in [1,2,3,4]:
        for i in range(6):
            batch = batchs[i]
            setattr(args, 'batch', batch)
            setattr(args,'batch_size',128)
            setattr(args,'alpha',0.5)
            setattr(args,'beta',10)
            for e in range(10):
                save_folder = f'results/NASA results (small sample {n})/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)
                dataloader = load_data(args,small_sample=n)
                pinn = LAPINN(args)
                pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'],
                           testloader=dataloader['test'])

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for NASA dataset')
    parser.add_argument('--data', type=str, default='NASA', help='NASA, HUST, MIT, TJU')
    parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=1, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
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


    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--LAN_hidden_dim', type=int, default=60, help='LAN hidden dim')

    parser.add_argument('--log_dir', type=str, default='text log.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/NASA results', help='save folder')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

