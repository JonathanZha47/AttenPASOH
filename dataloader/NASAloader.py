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
        if nominal_capacity is not None:
            #print(f'nominal_capacity:{nominal_capacity}, capacity max:{df["capacity"].max()}',end=',')
            df['capacity'] = df['capacity']/df['capacity'].max()
            #print(f'SOH max:{df["capacity"].max()}')
            f_df = df.iloc[:,:-1]
            if self.normalization_method == 'min-max':
                f_df = 2*(f_df - f_df.min())/(f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean())/f_df.std()

            df.iloc[:,:-1] = f_df
        
        return df
    
    def load_one_battery(self,path,nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y
        :param path:
        :param nominal_capacity:
        :return:
        '''
        df = self.read_one_csv(path, nominal_capacity)

        if df.columns[-1] == 'capacity':
            y = df.iloc[:, -1].values
            x = df.drop(df.columns[-1], axis=1).values
        else:
            raise ValueError("The column at index -1 is not 'capacity'. Please check the input data format.")
        return (x, y)
    
    def load_all_battery(self,path_list,nominal_capacity):
        '''
        Read multiple csv files, divide the data into X and Y, and then package it into a dataloader
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, used to calculate SOH
        :param batch_size: batch size
        :return: Dataloader
        '''
        X , Y = [], []
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder,self.args.log_dir)
            write_to_txt(save_name,'data path:')
            write_to_txt(save_name,str(path_list))
        for path in path_list:
            (x,y) = self.load_one_battery(path, nominal_capacity)
            # print(path)
            # print(x1.shape, x2.shape, y1.shape, y2.shape)
            X.append(x)
            Y.append(y)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

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

        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2,'valid_2': valid_loader_2,
                  'test_3': test_loader_3}
        return loader


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
        return parser.parse_args()

    args = get_args()