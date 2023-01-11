from utils import *
import numpy as np
import pandas as pd
import datetime
import json
import os
import shutil

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


output_dir = ['./data/process/', './data/sample/']
sample_rate = 0.05
for dir_ in output_dir:
    if os.path.isdir(dir_):
        shutil.rmtree(dir_)

    os.mkdir(dir_)

print("Read original data and preprocess...")
transactions = pd.read_csv("./data/original/transactions_train.csv.", dtype={"article_id": "str"})
customers = pd.read_csv("./data/original/customers.csv")
articles = pd.read_csv("./data/original/articles.csv", dtype={"article_id": "str"})

transactions['customer_id'] = customer_hex_id_to_int(transactions['customer_id'])
transactions.t_dat = pd.to_datetime(transactions.t_dat, format='%Y-%m-%d')
transactions['week'] = 104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7
transactions.article_id = article_id_str_to_int(transactions.article_id)
articles.article_id = article_id_str_to_int(articles.article_id)
transactions.week = transactions.week.astype('int8')
transactions.sales_channel_id = transactions.sales_channel_id.astype('int8')
transactions.price = transactions.price.astype('float32')
customers.customer_id = customer_hex_id_to_int(customers.customer_id)
for col in ['FN', 'Active', 'age']:
    customers[col].fillna(-1, inplace=True)
    customers[col] = customers[col].astype('int8')
customers.club_member_status = Categorize().fit_transform(customers[['club_member_status']]).club_member_status
customers.postal_code = Categorize().fit_transform(customers[['postal_code']]).postal_code
customers.fashion_news_frequency = Categorize().fit_transform(customers[['fashion_news_frequency']]).fashion_news_frequency
for col in articles.columns:
    if articles[col].dtype == 'object':
        articles[col] = Categorize().fit_transform(articles[[col]])[col]
for col in articles.columns:
    if articles[col].dtype == 'int64':
        articles[col] = articles[col].astype('int32')
transactions.sort_values(['t_dat', 'customer_id'], inplace=True)

print("Preprocess finish, saving parquet files...")
transactions.to_parquet(output_dir[0]+'transactions_train_process.parquet', index=False)
customers.to_parquet(output_dir[0]+'customers_train_process.parquet', index=False)
articles.to_parquet(output_dir[0]+'articles_train_process.parquet', index=False)

print("Now start sampling data for experiment speeding up")
customers_sample = customers.sample(frac=sample_rate, replace=False)
customers_sample_ids = set(customers_sample['customer_id'])
transactions_sample = transactions[transactions["customer_id"].isin(customers_sample_ids)]
articles_sample_ids = set(transactions_sample["article_id"])
articles_sample = articles[articles["article_id"].isin(articles_sample_ids)]

transactions_fe = transactions[~transactions["customer_id"].isin(customers_sample_ids)]
articles_fe = articles[~articles["article_id"].isin(articles_sample_ids)]
customers_fe = customers[~customers.customer_id.isin(customers_sample_ids)]

print("Save Sampling data and fe data")
customers_sample.to_parquet(output_dir[1] + f'customers_sample_{sample_rate}.parquet', index=False)
transactions_sample.to_parquet(output_dir[1] + f'transactions_train_sample_{sample_rate}.parquet', index=False)
articles_sample.to_parquet(output_dir[1] + f'articles_train_sample_{sample_rate}.parquet', index=False)

customers_fe.to_parquet(output_dir[1] + f'customers_fe_{sample_rate}.parquet', index=False)
transactions_fe.to_parquet(output_dir[1] + f'transactions_fe_{sample_rate}.parquet', index=False)
articles_fe.to_parquet(output_dir[1] + f'articles_fe_{sample_rate}.parquet', index=False)