from sklearn import metrics
import numpy as np
import logging

def get_logger(log_name='log.txt'):
    logger = logging.getLogger('mylogger')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - function:%(funcName)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_name is not None:
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))
    AdjustRsquare = metrics.r2_score(true_label,pred_label)
    L1error = l1_relative_error(true_label,pred_label)
    L2error = l2_relative_error(true_label,pred_label)

    return [MSE,MAE,MAPE,RMSE,AdjustRsquare,L1error,L2error]

def adjusted_r2_score(y_true, y_pred, n, p):
    r2 = metrics.r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

def l1_relative_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def l2_relative_error(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2)) / np.sqrt(np.sum(y_true ** 2))


def write_to_txt(txt_name,txt):
    with open(txt_name,'a') as f:
        f.write(txt)
        f.write('\n')

if __name__ == '__main__':
    for i in range(3):
        logger_name = f'log_{i}.txt'
        if i == 1:
            logger_name = None
        logger = get_logger(log_name=logger_name)
        logger.info(f'time: {i}, This is a log info')
        logger.removeHandler(logger.handlers[0])
        logger.handlers.clear()

