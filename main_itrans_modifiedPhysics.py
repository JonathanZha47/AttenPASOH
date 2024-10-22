from dataloader.dualitransloader import XJTUdata
from Model.itrans_modifiedphysics_temporalPDE import temporalPINN
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def load_data_unseen_data(args,small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if args.batch in file:
            if '4' in file or '8' in file:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(noise_option=args.noise_option, noise_mean= args.noise_mean, noise_std = args.noise_std, specific_path_list=train_list)
    test_loader = data.read_all(noise_option=args.noise_option, noise_mean= args.noise_mean, noise_std = args.noise_std, specific_path_list=test_list)
    dataloader = {'temporal_train': train_loader['train_temporal'],
                  'temporal_valid': train_loader['valid_temporal'],
                  'temporal_test': test_loader['test_temporal'],
                  'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader

def load_data(args,small_sample=None):
    root = os.path.join('data', 'XJTU data')
    data = XJTUdata(root=root, args=args)
    files = os.listdir(root)
    data_list = []
    for file in files:
        if args.batch in file:
            data_list.append(os.path.join(root, file))
    if small_sample is not None:
        data_list = data_list[:small_sample]

    loader = data.read_all(noise_option=args.noise_option, noise_mean= args.noise_mean, noise_std = args.noise_std, specific_path_list=data_list)
    dataloader = {'temporal_train': loader['train_temporal'], 'temporal_valid': loader['valid_temporal'],
                  'temporal_test': loader['test_temporal'],
                  'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}
    return dataloader

def main():
    args = get_args()
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for i in range(6):
        batch = batchs[i]
        setattr(args, 'batch', batch)
        for e in range(10):
            save_folder = 'results of reviewer/XJTU(temporalPINN seq=20 without noise) results/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_dir = 'logging.txt'
            setattr(args, "save_folder", save_folder)
            setattr(args, "log_dir", log_dir)

            dataloader = load_data_unseen_data(args)
            tPINN = temporalPINN(args)
            tPINN.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'],
                        seqtrainloader=dataloader['temporal_train'],seqvalidloader=dataloader['temporal_valid'],seqtestloader=dataloader['temporal_test'])

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
                save_folder = f'results/XJTU results (small sample {n})/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)
                dataloader = load_data(args,small_sample=n)
                pinn = PINN(args)
                pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'],
                           testloader=dataloader['test'])

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for XJTU dataset')
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=1, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    # scheduler 相关

    parser.add_argument('--epochs', type=int, default=500, help='epoch')
    parser.add_argument('--early_stop', type=int, default=50, help='early stop')
    parser.add_argument('--base_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epoch')
    parser.add_argument('--u_warmup_lr', type=float, default=0.002, help='warmup lr of solution u')
    parser.add_argument('--u_final_lr', type=float, default=0.0001, help='final lr of solution u')
    parser.add_argument('--F_warmup_lr', type=float, default=0.002, help='warmup lr of dynamical F')
    parser.add_argument('--F_final_lr', type=float, default=0.0001, help='final lr of dynamical F')
    parser.add_argument('--lan_warmup_lr', type=float, default=0.001, help='warmup lr of LAN')
    parser.add_argument('--lan_final_lr', type=float, default=0.0005, help='final lr of LAN')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')

    # loss function related
    parser.add_argument('--alpha', type=float, default=25.0, help='alpha')
    parser.add_argument('--beta', type=float, default=25.0, help='beta')
    parser.add_argument('--gamma', type=float, default=25.0, help='gamma')
    parser.add_argument('--theta', type=float, default=25.0, help='theta')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--LAN_hidden_dim', type=int, default=60, help='LAN hidden dim')

    # solution_u architect 相关
    parser.add_argument('--u_input_dim', type=int, default=17, help='input dim of solution_u')
    parser.add_argument('--u_output_dim', type=int, default=32, help='output dim of solution_u')
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of solution_u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of solution_u')

    parser.add_argument('--log_dir', type=str, default='text log.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/NASA results', help='save folder')

    ## GRU related
    parser.add_argument('--seq_length', type=int, default=20, help='seq_length')

    # noise related
    parser.add_argument('--noise_option', action='store_true', default=True, help='whether to add noise into the dataset')
    parser.add_argument('--no_noise', action='store_false', dest='noise_option', help='disable noise in the dataset')

    parser.add_argument('--noise_mean',type=float,default=0,help='noise_mean')
    parser.add_argument('--noise_std',type=float,default=1,help='noise_std')

    return parser.parse_args()


if __name__ == '__main__':
    main()

