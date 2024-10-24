import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from utils.util import write_to_txt
import argparse

class DF():
    def __init__(self,args):
        self.normalization = True
        self.normalization_method = args.normalization_method # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self,df):
        '''
        :param df: DataFrame
        :return: DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def read_one_csv(self,file_name,nominal_capacity=None):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)

        ## this line add the 'cycle index' column to the very left-end of the dataframe
        df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))

        ## this line delete all the outlier rows that deviates from 3 std away from the mean 
        df = self.delete_3_sigma(df)

        # Reorder columns so that 'cycle index' is second-to-last and 'capacity' is last
        columns_order = [col for col in df.columns if col not in ['capacity', 'cycle index']]
        columns_order.append('cycle index')
        columns_order.append('capacity')
        df = df[columns_order]
        ## if we input the nominal_capacity to the function,
        ## then we will use the capacity/nominal_capacity to measure its state of health. If not, then just use capacity
        ## Also, here we normalized all the columns
        # Normalize the capacity column if nominal_capacity is provided
        if nominal_capacity is not None:
            df['capacity'] = df['capacity'] / nominal_capacity  # Normalize capacity column
            # Select all columns except the last one (which includes the normalized capacity)
            # Skip normalization for the 'capacity' column at index 14
            # Select all columns except 'cycle index'
            f_df = df.iloc[:,:-1]
            # Apply normalization only to the selected columns, excluding 'capacity' and 'cycle index'
            if self.normalization_method == 'min-max':
                min_val = f_df.min()
                max_val = f_df.max()
                f_df = 2 * (f_df - min_val) / (max_val - min_val) -1
            elif self.normalization_method == 'z-score':
                mean_val = f_df.mean()
                std_val = f_df.std()
                f_df = (f_df - mean_val) / std_val
            # Update the DataFrame with normalized features, excluding 'capacity' and 'cycle index'
            df.iloc[:,:-1] = f_df
        return df
    
    def load_one_battery(self,path,noise_option,noise_mean,noise_std,nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y
        :param path:
        :param nominal_capacity:
        :return:
        '''
        if noise_option:
            df = self.read_one_csv(path,nominal_capacity)
            x = df.iloc[:,:-1].values
            y = df.iloc[:,-1].values
            noise = np.random.normal(loc=noise_mean, scale=noise_std, size=x.shape)
            x_noisy = x + noise
            x_seq = df.values
            y_seq = df.iloc[:,-1].values
            return (x_noisy, y), (x_seq, y_seq)
        else:
            df = self.read_one_csv(path,nominal_capacity)
            x = df.iloc[:,:-1].values
            y = df.iloc[:,-1].values
            x_seq = df.values
            y_seq = df.iloc[:,-1].values
            return (x, y), (x_seq, y_seq)
        
    def create_sequence(self, x, y, seq_length):
        '''
        Create sequences from the full data with the length of `seq_length`.
        :param x: The full sequence from the CSV file.
        :param seq_length: Length of each sequence.
        :return: The input sequences (X) and the corresponding last element as the label (y_pred).
        '''
        sequences = []
        labels = []
        for i in range(len(x) - seq_length):
            seq = x[i:i + seq_length]
            label = seq[:,-1]  # Extract the full sequence.
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)
    
    def load_all_battery(self,path_list,nominal_capacity,noise_option,noise_mean,noise_std):
        '''
        Read multiple csv files, divide the data into X and Y, and then package it into a dataloader
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, used to calculate SOH
        :param batch_size: batch size
        :return: Dataloader
        '''
        X , Y , X_seq, Y_seq = [], [], [], []
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder,self.args.log_dir)
            write_to_txt(save_name,'data path:')
            write_to_txt(save_name,str(path_list))
        for path in path_list:
            (x,y),(x_seq, y_seq)= self.load_one_battery(path, nominal_capacity,noise_option,noise_mean,noise_std)
            # print(path)
            # print(x1.shape, x2.shape, y1.shape, y2.shape)
            X.append(x)
            Y.append(y)
            X_seq.append(x_seq)
            Y_seq.append(y_seq)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        X_seq = np.concatenate(X_seq, axis=0)
        Y_seq = np.concatenate(Y_seq, axis=0)
        # 创建基于历史cycle数据的序列
        seq_length = self.args.seq_length
        sequences, labels = self.create_sequence(X_seq, Y_seq, seq_length)
        # 基于历史cycle数据的序列划分训练集和测试集
        tensor_sequences = torch.from_numpy(sequences).float()
        tensor_labels = torch.from_numpy(labels).float()
        split = int(sequences.shape[0] * 0.8)
        train_sequences, test_sequences = tensor_sequences[:split], tensor_sequences[split:]
        train_labels, test_labels = tensor_labels[:split], tensor_labels[split:]

        #因为要捕捉时空关系，所以这里不要shuffle
        train_sequences, valid_sequences, train_labels, valid_labels = \
            train_test_split(train_sequences, train_labels, test_size=0.2, random_state=420)
        
        train_loader_temporal = DataLoader(TensorDataset(train_sequences, train_labels),
                                            batch_size=self.args.batch_size,
                                            shuffle=False)
        valid_loader_temporal = DataLoader(TensorDataset(valid_sequences, valid_labels),
                                            batch_size=self.args.batch_size,
                                            shuffle=False)
        test_loader_temporal = DataLoader(TensorDataset(test_sequences, test_labels),
                                            batch_size=self.args.batch_size,
                                            shuffle=False)


        # 普通的X，Y数据处理（x1 - xn) mapping y
        tensor_X = torch.from_numpy(X).float()
        tensor_Y = torch.from_numpy(Y).float().reshape(-1,1)
        # Condition 1
        # 1.1 划分训练集和测试集
        split = int(tensor_X.shape[0] * 0.8)
        train_X, test_X = tensor_X[:split], tensor_X[split:]
        train_Y, test_Y = tensor_Y[:split], tensor_Y[split:]

        # 1.2 划分训练集和验证集
        train_X, valid_X, train_Y, valid_Y = \
            train_test_split(train_X, train_Y, test_size=0.2, random_state=420)


        train_loader = DataLoader(TensorDataset(train_X, train_Y),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        
        valid_loader = DataLoader(TensorDataset(valid_X, valid_Y),
                                    batch_size=self.args.batch_size,
                                    shuffle=True)
        
        test_loader = DataLoader(TensorDataset(test_X, test_Y),
                                      batch_size=self.args.batch_size,
                                      shuffle=False)


        # Condition 2

        train_X, valid_X, train_Y, valid_Y = \
            train_test_split(tensor_X, tensor_Y, test_size=0.2, random_state=420)
        train_loader_2 = DataLoader(TensorDataset(train_X, train_Y),
                                    batch_size=self.args.batch_size,
                                    shuffle=True)
        valid_loader_2 = DataLoader(TensorDataset(valid_X, valid_Y),
                                    batch_size=self.args.batch_size,
                                    shuffle=True)
        
        # Condition 3
        test_loader_3 = DataLoader(TensorDataset(tensor_X, tensor_Y),
                                    batch_size=self.args.batch_size,
                                    shuffle=False)

        loader = {'train_temporal':train_loader_temporal, 'valid_temporal': valid_loader_temporal, 'test_temporal': test_loader_temporal,
                  'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2,'valid_2': valid_loader_2,
                  'test_3': test_loader_3}
        return loader


class NASAdata(DF):
    def __init__(self,root='../data/NASA data',args=None):
        super(NASAdata, self).__init__(args)
        self.root = root
        self.batchs = ['Batch1','Batch2','Batch3','Batch4','Batch5','Batch6','Batch7','Batch9']
        if self.normalization:
            self.nominal_capacities = 2.0
        else:
            self.nominal_capacities = None
        #print('-' * 20, 'TJU data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]; optional[1,2,3]
        :return: DataFrame
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        df = pd.DataFrame()
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacities[batch])

    def read_all(self,specific_path_list):
        '''
        读取所有csv文件,封装成dataloader
        English version: Read all csv files and encapsulate them into a dataloader
        :param self:
        :return: dict
        '''
        return self.load_all_battery(path_list=specific_path_list,nominal_capacity=self.nominal_capacities)
class XJTUdata(DF):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = os.listdir(root)
        try:
            self.variables = pd.read_csv(os.path.join(root, self.file_list[0]), encoding='utf-8').columns
        except UnicodeDecodeError:
            self.variables = pd.read_csv(os.path.join(root, self.file_list[0]), encoding='ISO-8859-1').columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size
        self.noise_mean = args.noise_mean
        self.noise_std = args.noise_std
        self.noise_option = args.noise_option

        if self.normalization:
            self.nominal_capacity = 2.0
        else:
            self.nominal_capacity = None
        #print('-'*20,'XJTU data','-'*20)

    def read_one_batch(self,batch='2C'):
        '''
        读取一个批次的csv文件,并把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version: Read a batch of csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :param batch: int or str:batch
        :return: dict
        '''
        if isinstance(batch,int):
            batch = self.batch_names[batch]
        assert batch in self.batch_names, 'batch must be in {}'.format(self.batch_names)
        file_list = []
        for i in range(self.num):
            if batch in self.file_list[i]:
                path = os.path.join(self.root,self.file_list[i])
                file_list.append(path)
        return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity,noise_option = self.noise_option,noise_mean=self.noise_mean,noise_std=self.noise_std)

    def read_all(self,noise_option,noise_mean,noise_std,specific_path_list=None):
        '''
        读取所有csv文件，并把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version: Read all csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for file in self.file_list:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list,nominal_capacity=self.nominal_capacity,noise_option=self.noise_option,noise_mean=self.noise_mean,noise_std=self.noise_std)
if __name__ == '__main__':

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--datasource',type=str,default='NASA',help='NASA, XJTU, HUST, MIT, TJU')
        parser.add_argument('--data',type=str,default='MIT',help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch',type=int,default=1,help='1,2,3')
        parser.add_argument('--batch_size',type=int,default=256,help='batch size')
        parser.add_argument('--seq_length',type=int,default=10,help='seq_length')
        parser.add_argument('--stride',type=int,default=1,help='stride')
        parser.add_argument('--normalization_method',type=str,default='z-score',help='min-max,z-score')
        parser.add_argument('--log_dir',type=str,default='test.txt',help='log dir')
        parser.add_argument('--num_ways',type=int,default=1,help='num_ways')
        parser.add_argument('--num_shots',type=int,default=5,help='num_shots')
        parser.add_argument('--train_test_split',type=int,default=2,help='train_test_split')
        parser.add_argument('--noise_mean',type=float,default=0,help='noise_mean')
           # noise related
        parser.add_argument('--noise_option', action='store_true', default=True, help='whether to add noise into the dataset')
        parser.add_argument('--no_noise', action='store_false', dest='noise_option', help='disable noise in the dataset')

        parser.add_argument('--noise_std',type=float,default=1,help='noise_std')
        parser.add_argument('--noise_option',type=bool,default=True,help='whether add the noise into the dataset')
        return parser.parse_args()

    args = get_args()