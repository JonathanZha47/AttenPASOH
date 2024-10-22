"""
    for this loader, we dont need to separate x1,y1,x2,y2 for different sequences

"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import random
import sys
sys.path.append('/Users/jonathanzha/Desktop/Meta-Learning-PINN-for-SOH')
from sklearn.model_selection import train_test_split
from utils.util import write_to_txt

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
        df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))

        df = self.delete_3_sigma(df)

        if nominal_capacity is not None:
            #print(f'nominal_capacity:{nominal_capacity}, capacity max:{df["capacity"].max()}',end=',')
            df['capacity'] = df['capacity']/nominal_capacity
            #print(f'SOH max:{df["capacity"].max()}')
            f_df = df.iloc[:,:-1]
            if self.normalization_method == 'min-max':
                f_df = 2*(f_df - f_df.min())/(f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean())/f_df.std()

            df.iloc[:,:-1] = f_df

        return df

    def load_one_battery(self, path, nominal_capacity=None):
        '''
        Read a CSV file, apply normalization if necessary, and return the full sequence.
        :param path: The file path for the CSV.
        :param nominal_capacity: The nominal capacity to normalize the 'capacity' column.
        :return: x (full sequence).
        '''
        df = self.read_one_csv(path, nominal_capacity)
        x = df.values  # Use all columns as part of the sequence (no more separating X and Y).
        y = df.iloc[:, -1].values   # The last column ('capacity')
        return x,y

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

    def load_all_battery(self, path_list, nominal_capacity):
        '''
        Load multiple CSV files, treat the full sequence as input, and prepare the dataloader.
        :param path_list: List of file paths to CSVs.
        :param nominal_capacity: Nominal capacity for normalization.
        :return: A dataloader ready for sequence-to-sequence learning.
        '''
        X = []
        Y = []

        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)
            write_to_txt(save_name, 'data path:')
            write_to_txt(save_name, str(path_list))

        # Load all battery data into sequences
        for path in path_list:
            x, y = self.load_one_battery(path, nominal_capacity)
            X.append(x)
            Y.append(y)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        seq_length = self.args.seq_length
        sequences, labels = self.create_sequence(X, Y, seq_length)
        tensor_X = torch.from_numpy(sequences).float()
        tensor_Y = torch.from_numpy(labels).float().view(-1, 1)

        split = int(tensor_X.shape[0] * 0.8)
        train_X, test_X = tensor_X[:split], tensor_X[split:]
        train_Y, test_Y = tensor_Y[:split], tensor_Y[split:]


        # 1.2 划分训练集和验证集
        train_X, valid_X, train_Y, valid_Y = \
            train_test_split(train_X, train_Y, test_size=0.2, random_state=420)

        train_loader = DataLoader(TensorDataset(train_X, train_Y),
                                  batch_size=self.args.batch_size,
                                  shuffle=False)
        valid_loader = DataLoader(TensorDataset(valid_X, valid_Y),
                                  batch_size=self.args.batch_size,
                                  shuffle=False)
        test_loader = DataLoader(TensorDataset(test_X, test_Y),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)

        # Condition 2
        train_X, valid_X, train_Y, valid_Y = \
            train_test_split(tensor_X, tensor_Y, test_size=0.2, random_state=420)
        
        train_loader_2 = DataLoader(TensorDataset(train_X, train_Y),
                                    batch_size=self.args.batch_size,
                                    shuffle=False)
        valid_loader_2 = DataLoader(TensorDataset(valid_X, valid_Y),
                                    batch_size=self.args.batch_size,
                                    shuffle=False)

        # Condition 3
        test_loader_3 = DataLoader(TensorDataset(tensor_X, tensor_Y),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)


        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2,'valid_2': valid_loader_2,
                  'test_3': test_loader_3}
        return loader


class XJTUdata(DF):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = os.listdir(root)
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size

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
        return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
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
            return self.load_all_battery(path_list=specific_path_list,nominal_capacity=self.nominal_capacity)

class XJTU_Meta(DF):
    def __init__(self,root,args):
        super(XJTU_Meta, self).__init__(args)
        self.root = root
        self.num_ways = args.num_ways
        self.num_shots = args.num_shots
        self.file_list = os.listdir(root)
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size
        self.method = args.train_test_split
    
    def get_data_samples(self):
        ## According to the num_ways and num_shots, fetch data samples
        ## each num_ways means a batch, eg: 1-way means randomly selected a batch
        ## each num_shots means number of csvs that we need to read eg: 1-shot means randomly selected a csv
        batch_index = random.sample(range(len(self.batch_names)), self.num_ways)
        sample_batch = []
        for i in batch_index:
            sample_batch.append(self.batch_names[i])

        selected_way_list = []
        for selected_batch in sample_batch:
            for i in range(self.num):
                if selected_batch in self.file_list[i]:
                    path = os.path.join(self.root, self.file_list[i])
                    selected_way_list.append(path)
        selected_shots = random.sample(selected_way_list, self.num_shots)
        unselected_shots = []
        for csv in selected_way_list:
            if csv not in selected_shots:
                unselected_shots.append(csv)
        return selected_shots, unselected_shots

    def read_all_selected_shots(self):
        selected_shots, unselected_shots = self.get_data_samples()
        return self.load_meta_battery_sample(selected_shots=selected_shots,test_shots=unselected_shots,method=self.method)

class HUSTdata(DF):
    def __init__(self,root='../data/HUST data',args=None):
        super(HUSTdata, self).__init__(args)
        self.root = root
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-'*20,'HUST data','-'*20)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件:如果指定了specific_path_list,则读取指定的文件；否则读取所有文件；
        把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version:
        Read all csv files.
        If specific_path_list is not None, read the specified file;
        otherwise read all files;
        :param self:
        :param specific_path:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            files = os.listdir(self.root)
            for file in files:
                path = os.path.join(self.root,file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class MITdata(DF):
    def __init__(self,root='../data/MIT data',args=None):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12','2017-06-30','2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-' * 20, 'MIT data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]
        :return: dict
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件。如果指定了specific_path_list,则读取指定的文件；否则读取所有文件；封装成dataloader
        English version:
        Read all csv files.
        If specific_path_list is not None, read the specified file; otherwise read all files;
        :param self:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root,batch)
                files = os.listdir(root)
                for file in files:
                    path = os.path.join(root,file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)

class TJUdata(DF):
    def __init__(self,root='../data/TJU data',args=None):
        super(TJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]
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
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)

class NASAdata(DF):
    def __init__(self,root='../data/NASA data',args=None):
        super(NASAdata, self).__init__(args)
        self.root = root
        self.batchs = ['Batch1','Batch2','Batch3','Batch4','Batch5','Batch6','Batch7','Batch9']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]
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
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)


if __name__ == '__main__':
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data',type=str,default='MIT',help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch',type=int,default=1,help='1,2,3')
        parser.add_argument('--batch_size',type=int,default=256,help='batch size')
        parser.add_argument('--normalization_method',type=str,default='z-score',help='min-max,z-score')
        parser.add_argument('--log_dir',type=str,default='test.txt',help='log dir')
        parser.add_argument('--num_ways',type=int,default=1,help='num_ways')
        parser.add_argument('--num_shots',type=int,default=5,help='num_shots')
        parser.add_argument('--train_test_split',type=int,default=2,help='train_test_split')
        parser.add_argument('--seq_length',type=int,default=3,help='seq_length')
        return parser.parse_args()

    args = get_args()

    # xjtu = XJTUdata(root='../data/XJTU data',args=args)
    # path = '../data/XJTU data/2C_battery-1.csv'
    # xjtu.read_one_batch('2C')
    # xjtu.read_all()
    #
    # hust = HUSTdata(args=args)
    # hust.read_all()
    #
    # tju = HUSTdata(args=args)
    # loader = tju.read_all()
    xjtu_meta = XJTU_Meta(root='data/XJTU data',args=args)
    xjtu_meta.get_data_samples()



