import os
import pandas as pd
import numpy as np
import tqdm

from typing import Dict

import joblib
import torch.nn as nn
import torch
import pickle
import os
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm.notebook import tqdm


features = ['pre_since_opened',
 'pre_since_confirmed',
 'pre_pterm',
 'pre_fterm',
 'pre_till_pclose',
 'pre_till_fclose',
 'pre_loans_credit_limit',
 'pre_loans_next_pay_summ',
 'pre_loans_outstanding',
 'pre_loans_total_overdue',
 'pre_loans_max_overdue_sum',
 'pre_loans_credit_cost_rate',
 'pre_loans5',
 'pre_loans530',
 'pre_loans3060',
 'pre_loans6090',
 'pre_loans90',
 'is_zero_loans5',
 'is_zero_loans530',
 'is_zero_loans3060',
 'is_zero_loans6090',
 'is_zero_loans90',
 'pre_util',
 'pre_over2limit',
 'pre_maxover2limit',
 'is_zero_util',
 'is_zero_over2limit',
 'is_zero_maxover2limit',
 'enc_paym_0',
 'enc_paym_1',
 'enc_paym_2',
 'enc_paym_3',
 'enc_paym_4',
 'enc_paym_5',
 'enc_paym_6',
 'enc_paym_7',
 'enc_paym_8',
 'enc_paym_9',
 'enc_paym_10',
 'enc_paym_11',
 'enc_paym_12',
 'enc_paym_13',
 'enc_paym_14',
 'enc_paym_15',
 'enc_paym_16',
 'enc_paym_17',
 'enc_paym_18',
 'enc_paym_19',
 'enc_paym_20',
 'enc_paym_21',
 'enc_paym_22',
 'enc_paym_23',
 'enc_paym_24',
 'enc_loans_account_holder_type',
 'enc_loans_credit_status',
 'enc_loans_credit_type',
 'enc_loans_account_cur',
 'pclose_flag',
 'fclose_flag']

transaction_features = features

embedding_projection = {
  'pre_since_opened':(25, 10),
  'pre_since_confirmed':(22, 9),
  'pre_pterm':(22, 9),
  'pre_fterm':(21,9),
  'pre_till_pclose':(21, 9),
  'pre_till_fclose':(20, 8),
  'pre_loans_credit_limit':(25, 10),
  'pre_loans_next_pay_summ':(10, 6),
  'pre_loans_outstanding':(10,6),
  'pre_loans_total_overdue':(10, 6),
  'pre_loans_max_overdue_sum':(10, 6),
  'pre_loans_credit_cost_rate':(16, 8),
  'pre_loans5':(20, 9),
  'pre_loans530':(25, 10),
  'pre_loans3060':(15, 7),
  'pre_loans6090':(10, 6),
  'pre_loans90':(25, 10),
  'is_zero_loans5':(5, 3),
  'is_zero_loans530':(5, 3),
  'is_zero_loans3060':(5, 3),
  'is_zero_loans6090':(5, 3),
  'is_zero_loans90':(5, 3),
  'pre_util':(25, 10),
  'pre_over2limit':(25, 10),
  'pre_maxover2limit':(25, 10),
  'is_zero_util':(5, 3),
  'is_zero_over2limit':(5,3),
  'is_zero_maxover2limit':(5,3),
  'enc_paym_0': (10, 6),
  'enc_paym_1':(10,6),
  'enc_paym_2':(10,6),
  'enc_paym_3':(10,6),
  'enc_paym_4':(10,6),
  'enc_paym_5':(10,6),
  'enc_paym_6':(10,6),
  'enc_paym_7':(10,6),
  'enc_paym_8':(10,6),
  'enc_paym_9':(10,6),
  'enc_paym_10':(10,6),
  'enc_paym_11':(10,6),
  'enc_paym_12':(10,6),
  'enc_paym_13':(10,6),
  'enc_paym_14':(10,6),
  'enc_paym_15':(10,6),
  'enc_paym_16':(10,6),
  'enc_paym_17':(10,6),
  'enc_paym_18':(10,6),
  'enc_paym_19':(10,6),
  'enc_paym_20':(10,6),
  'enc_paym_21':(10,6),
  'enc_paym_22':(10,6),
  'enc_paym_23':(10,6),
  'enc_paym_24':(10,6),
  'enc_loans_account_holder_type':(18, 8),
  'enc_loans_credit_status':(18,8),
  'enc_loans_credit_type':(10, 6),
  'enc_loans_account_cur':(10, 6),
  'pclose_flag':(2, 1),
  'fclose_flag':(2, 1),}

def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                     num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    читает num_parts_to_read партиций, преобразует их к pd.DataFrame и возвращает
    :param path_to_dataset: путь до директории с партициями
    :param start_from: номер партиции, с которой начать чтение
    :param num_parts_to_read: количество партиций, которые требуется прочитать
    :param columns: список колонок, которые нужно прочитать из партиции
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset) 
                              if filename.startswith('train_data') or filename.startswith("test_data")])
    
    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm(chunks, desc="Reading dataset with pandas"):
        chunk = pd.read_parquet(chunk_path,columns=columns)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)


def batches_generator(list_of_paths, batch_size=32, shuffle=False, is_infinite=False,
                      verbose=False, device=None, output_format='torch', is_train=True):
    """
    функция для создания батчей на вход для нейронной сети для моделей на keras и pytorch.
    так же может использоваться как функция на стадии инференса
    :param list_of_paths: путь до директории с предобработанными последовательностями
    :param batch_size: размер батча
    :param shuffle: флаг, если True, то перемешивает list_of_paths и так же
    перемешивает последовательности внутри файла
    :param is_infinite: флаг, если True,  то создает бесконечный генератор батчей
    :param verbose: флаг, если True, то печатает текущий обрабатываемый файл
    :param device: device на который положить данные, если работа на торче
    :param output_format: допустимые варианты ['tf', 'torch']. Если 'torch', то возвращает словарь,
    где ключи - батчи из признаков, таргетов и app_id. Если 'tf', то возвращает картеж: лист input-ов
    для модели, и список таргетов.
    :param is_train: флаг, Если True, то для кераса вернет (X, y), где X - input-ы в модель, а y - таргеты, 
    если False, то в y будут app_id; для torch вернет словарь с ключами на device.
    :return: бачт из последовательностей и таргетов (или app_id)
    """
    while True:
        if shuffle:
            np.random.shuffle(list_of_paths)

        for path in list_of_paths:
            if verbose:
                print(f'reading {path}')

            with open(path, 'rb') as f:
                data = pickle.load(f)
            padded_sequences, targets = data['padded_sequences'], data['targets']
            app_ids = data['app_id']
            indices = np.arange(len(padded_sequences))

            if shuffle:
                np.random.shuffle(indices)
                padded_sequences = padded_sequences[indices]
                targets = targets[indices]
                app_ids = app_ids[indices]

            for idx in range(len(padded_sequences)):
                bucket = padded_sequences[idx]
                app_id = app_ids[idx]
                
                if is_train:
                    target = targets[idx]
                
                for jdx in range(0, len(bucket), batch_size):
                    batch_sequences = bucket[jdx: jdx + batch_size]
                    if is_train:
                        batch_targets = target[jdx: jdx + batch_size]
                    
                    batch_app_ids = app_id[jdx: jdx + batch_size]
                    
                    if output_format == 'tf':
                        batch_sequences = [batch_sequences[:, i] for i in
                                           range(len(transaction_features))]
                        
                        # append product as input to tf model
                        if is_train:
                            yield batch_sequences, batch_targets
                        else:
                             yield batch_sequences, batch_app_ids
                    else:
                        batch_sequences = [torch.LongTensor(batch_sequences[:, i]).to(device)
                                           for i in range(len(transaction_features))]
                        if is_train:
                            yield dict(transactions_features=batch_sequences,
                                       label=torch.LongTensor(batch_targets).to(device),
                                       app_id=batch_app_ids)
                        else:
                            yield dict(transactions_features=batch_sequences,
                                       app_id=batch_app_ids)
        if not is_infinite:
            break


def pad_sequence(array, max_len) -> np.array:
    """
    принимает список списков (array) и делает padding каждого вложенного списка до max_len
    :param array: список списков
    :param max_len: максимальная длина до которой нужно сделать padding
    :return: np.array после padding каждого вложенного списка до одинаковой длины
    """
    add_zeros = max_len - len(array[0])
    return np.array([list(x) + [0] * add_zeros for x in array])


def truncate(x, num_last_transactions=20):
    return x.values.transpose()[:, -num_last_transactions:].tolist()


def transform_transactions_to_sequences(transactions_frame: pd.DataFrame,
                                        num_last_transactions=20) -> pd.DataFrame:
    """
    принимает frame с транзакциями клиентов, сортирует транзакции по клиентам
    (внутри клиента сортирует транзакции по возрастанию), берет num_last_transactions танзакций,
    возвращает новый pd.DataFrame с двумя колонками: app_id и sequences.
    каждое значение в колонке sequences - это список списков.
    каждый список - значение одного конкретного признака во всех клиентских транзакциях.
    Всего признаков len(features), поэтому будет len(features) списков.
    Данная функция крайне полезна для подготовки датасета для работы с нейронными сетями.
    :param transactions_frame: фрейм с транзакциями клиентов
    :param num_last_transactions: количество транзакций клиента, которые будут рассмотрены
    :return: pd.DataFrame из двух колонок (app_id, sequences)
    """
    return transactions_frame \
        .sort_values(['id', 'rn']) \
        .groupby(['id'])[features] \
        .apply(lambda x: truncate(x, num_last_transactions=num_last_transactions)) \
        .reset_index().rename(columns={0: 'sequences'})

def create_padded_buckets(frame_of_sequences: pd.DataFrame,
                          save_to_file_path=None, has_target=True):
    """
    Функция реализует sequence_bucketing технику для обучения нейронных сетей.
    Принимает на вход frame_of_sequences (результат работы функции transform_transactions_to_sequences),
    словарь bucket_info, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding, далее группирует транзакции по бакетам (на основе длины), делает padding транзакций и сохраняет результат
    в pickle файл, если нужно
    :param frame_of_sequences: pd.DataFrame c транзакциями (результат применения transform_transactions_to_sequences)
    :param bucket_info: словарь, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding
    :param save_to_file_path: опциональный путь до файла, куда нужно сохранить результат
    :param has_target: флаг, есть ли в frame_of_sequences целевая переменная или нет. Если есть, то
    будет записано в результат
    :return: возвращает словарь с следюущими ключами (padded_sequences, targets, app_id, products)
    """
    #frame_of_sequences['bucket_idx'] = frame_of_sequences.sequence_length.map(bucket_info)
    padded_seq = []
    targets = []
    app_ids = []
    #products = []
    #print(frame_of_sequences)
    for i in tqdm(range(len(frame_of_sequences)), desc='Extracting buckets'):
        #print(frame_of_sequences.iloc[i])
        padded_sequences = pad_sequence(frame_of_sequences.iloc[i].sequences, 20)
        padded_sequences = np.array([np.array(x) for x in padded_sequences])
        padded_seq.append(padded_sequences)

        if has_target:
            targets.append(frame_of_sequences.iloc[i].flag)

        app_ids.append(frame_of_sequences.iloc[i].id)
        #products.append(bucket['product'].values)

    #frame_of_sequences.drop(columns=['bucket_idx'], inplace=True)

    dict_result = {
        'padded_sequences': np.array(padded_seq),
        'targets': np.array(targets) if targets else [],
        'app_id': np.array(app_ids),
    }

    if save_to_file_path:
        with open(save_to_file_path, 'wb') as f:
            pickle.dump(dict_result, f)
    return dict_result

# Функция для подготовки данных для нейросети
def create_buckets_from_transactions(path_to_dataset, save_to_path, frame_with_ids = None, 
                                     num_parts_to_preprocess_at_once: int = 1, 
                                     num_parts_total=12, has_target=False):
    block = 0
    for step in tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once), 
                                   desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once, 
                                                             verbose=True)
    
        seq = transform_transactions_to_sequences(transactions_frame)
        #seq['sequence_length'] = seq.sequences.apply(lambda x: len(x[1]))
        
        if frame_with_ids is not None:
            seq = seq.merge(frame_with_ids, on='id')

        block_as_str = str(block)
        if len(block_as_str) == 1:
            block_as_str = '00' + block_as_str
        else:
            block_as_str = '0' + block_as_str
            
        processed_fragment =  create_padded_buckets(seq, has_target=has_target, 
                                                    save_to_file_path=os.path.join(save_to_path, 
                                                                                   f'processed_chunk_{block_as_str}.pkl'))
        block += 1

