import numpy as np
import pandas as pd
import matplotlib as plt
import random
import statistics
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_percentage_error
from lightfm.data import Dataset
from lightfm import LightFM
from tqdm.auto import tqdm
import datetime
from lightfm import LightFM # model
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split
import itertools
from sklearn. model_selection import train_test_split
import pickle 
import os
import zipfile
import shutil

wb_data = pd.read_csv("wb_school_task_1.csv")
wb_data.head()

wb_data['order_ts'] = pd.to_datetime(wb_data['order_ts'])

wb_data['month_day'] = wb_data['order_ts'].apply(lambda x: f"{x.month:02d}-{x.day:02d}")

monthly_orders = wb_data.groupby(['month_day']).agg({'user_id': 'nunique', 'item_id': 'count'}).reset_index()

monthly_orders.columns = ['День', 'Количество пользователей', 'Количество товаров']

wb_data['order_date'] = pd.to_datetime(wb_data['order_ts'])

wb_data['day_of_week'] = wb_data['order_ts'].dt.day_name()

daily_orders = wb_data.groupby('day_of_week').agg({'user_id': 'nunique', 'item_id': 'count'}).reset_index()

daily_orders.columns = ['День недели', 'Количество пользователей', 'Количество товаров']

plot1 = monthly_orders.plot(x = 'День',title='Количество уникальных пользователей в день', rot = 45, grid=True, figsize=(12,10) )

plot2 = daily_orders.plot(x = 'День недели',title='Количество товаров на каждый день', rot = 45, grid=True, figsize=(12,10) )

with open('fit_log.txt', 'w') as file:
    file.write(f'Файл с данными прочитан и обработан\n')

pic1 = plot1.get_figure()

pic2 = plot2.get_figure()

if  os.path.isdir("pictures"):
    shutil.rmtree("pictures")
if not os.path.isdir("pictures"):
    os.mkdir("pictures")

pic1.savefig('pictures/'+'month_changing'+'.png')

pic2.savefig('pictures/'+'daily_changing'+'.png')

dir = os.listdir('pictures')

archive = zipfile.ZipFile('pic.zip', 'w')
for file in dir:
    archive.write('pictures/' + file)
archive.close()

with open('fit_log.txt', 'w') as file:
    file.write(f'Первичный EDA в директории\n')

class MostPopularRecommender:
    def __init__(self):
        self.top_items = {}

    def fit(self, df):
        for month in df['order_ts'].dt.to_period('M').unique():
            month_str = str(month)
            top_items_month = df[df['order_ts'].dt.to_period('M') == month].groupby('item_id').size().nlargest(100).index.tolist()
            self.top_items[month_str] = top_items_month

    def predict(self, df):
        result = []
        for month in df['order_ts'].dt.to_period('M').unique():
            month_str = str(month)
            if month_str in self.top_items:
                result.append(random.sample(self.top_items[month_str],10))
        return result

def generate_lightfm_recs_mapper(model, item_ids, known_items, 
                                 user_features, item_features, N, 
                                 user_mapping, item_inv_mapping, 
                                 num_threads=1):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(user_id, item_ids, user_features=user_features, 
                             item_features=item_features, num_threads=num_threads)
        
        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        
        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]
    return _recs_mapper

def compute_metrics(df_true, df_pred, top_N, rank_col='rank'):
    result = {}
    test_recs = df_true.set_index(['user_id', 'item_id']).join(df_pred.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', rank_col])

    test_recs['users_item_count'] = test_recs.groupby(level='user_id')[rank_col].transform(np.size)
    test_recs['reciprocal_rank'] = (1 / test_recs[rank_col]).fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs[rank_col]
    
    users_count = test_recs.index.get_level_values('user_id').nunique()
    for k in range(1, top_N + 1):
        hit_k = f'hit@{k}'
        test_recs[hit_k] = test_recs[rank_col] <= k
        result[f'Precision@{k}'] = (test_recs[hit_k] / k).sum() / users_count
        result[f'Recall@{k}'] = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count

    result[f'MAP@{top_N}'] = (test_recs["cumulative_rank"] / test_recs["users_item_count"]).sum() / users_count
    result[f'MRR'] = test_recs.groupby(level='user_id')['reciprocal_rank'].max().mean()
    return pd.Series(result)

df = pd.read_csv("wb_school_task_1.csv")

baseline = df.copy()
baseline['order_ts'] = pd.to_datetime(baseline['order_ts'])

train, test = train_test_split(df, test_size= 0.16 , random_state= 0 )
train['order_ts'] = pd.to_datetime(train['order_ts'])
test['order_ts'] = pd.to_datetime(test['order_ts'])
test_idx = list(sorted(set(test.user_id)))

# Data for lightFM

df_fit = df.copy()

df_fit['target'] = 1

df_fit['order_ts'] = pd.to_datetime(df_fit['order_ts'])

df_fit['month'] = df_fit['order_ts'].dt.month

user_purchase_frequency = df_fit.groupby('user_id')['month'].nunique()

active_users = user_purchase_frequency[user_purchase_frequency >= 3].index

df_filtered = df_fit[df_fit['user_id'].isin(active_users)]

items_per_user = df_filtered.groupby(['user_id', 'month']).size().reset_index(name='num_items_bought')

items = items_per_user[(items_per_user['num_items_bought'] >= 5) & (items_per_user['num_items_bought'] <= 50)]

user_purchase_frequency = items.groupby('user_id')['month'].nunique()

active_users = user_purchase_frequency[user_purchase_frequency >= 3].index

items_res = items[items['user_id'].isin(active_users)]

usefull_users = items_res['user_id'].unique().tolist()

df_fit = df_fit[df_fit['user_id'].isin(usefull_users)]

df_fit['day_of_month'] = df_fit['order_ts'].dt.day

df_fit.drop(['month', 'day_of_month'], axis= 1 , inplace= True )

max_date = df_fit['order_ts'].max()

min_date = df_fit['order_ts'].min()

train = df_fit[(df_fit['order_ts'] < max_date - pd.Timedelta(days=7))]

test = df_fit[(df_fit['order_ts'] >= max_date - pd.Timedelta(days=7))]

with open('fit_log.txt', 'w') as file:
    file.write(f'Данные для обучения LightFM готовы\n')

dataset = Dataset()

dataset.fit(train['user_id'].unique(), train['item_id'].unique())

interactions_matrix, weights_matrix = dataset.build_interactions(
    zip(*train[['user_id', 'item_id', 'target']].values.T)
)

weights_matrix_csr = weights_matrix.tocsr()

lightfm_mapping = dataset.mapping()
lightfm_mapping = {
    'users_mapping': lightfm_mapping[0],
    'items_mapping': lightfm_mapping[2],
}

lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}

def sample_hyperparameters():
    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 50),
        }

def random_search(train_interactions, test_interactions, num_samples=50, num_threads=1):
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train_interactions, epochs=num_epochs, num_threads=num_threads)

        score = precision_at_k(model, test_interactions, train_interactions=train_interactions, k=12, num_threads=num_threads).mean()
        
        print(score)

        hyperparams["num_epochs"] = num_epochs

        yield (score, hyperparams, model)

optimized_dict={}


sparse_customer_article_train, sparse_customer_article_test = random_train_test_split(weights_matrix_csr, test_percentage=0.2, random_state=42)

(score, hyperparams, model) = max(random_search(train_interactions = sparse_customer_article_train, 
                                                test_interactions = sparse_customer_article_test, 
                                                num_threads = 4), key=lambda x: x[0])

optimized_dict['Amount_Spent'] = {'score': score, 
                                  'params': hyperparams}

with open('fit_log.txt', 'w') as file:
    file.write(f'Гиперпараметры подобраны и сохранены\n')

import pickle 


with open('optimized_dict.pkl', 'wb') as f:
    pickle.dump(optimized_dict, f)


lfm_model = LightFM(
    no_components=40,
    learning_schedule='adagrad',
    learning_rate=0.02290932417614553,
    item_alpha=5.3359141882696886e-09,
    user_alpha=2.0959657102774366e-08,
    loss='warp', 
    max_sampled=14, 
    random_state=42
)

#💪🏼 train model

with open('fit_log.txt', 'w') as file:
    file.write(f'Модель обучается\n')

num_epochs = 21

for _ in tqdm(range(num_epochs)):
    lfm_model.fit_partial(
        weights_matrix_csr
    )

pkl_filename = "model_upload.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lfm_model, file)

with open('fit_log.txt', 'w') as file:
    file.write(f'Модель сохранена\n')

all_cols = list(lightfm_mapping['items_mapping'].values())

top_N = 200

lfm_prediction = pd.DataFrame({
    'user_id': test['user_id'].unique()
})

known_items = train.groupby('user_id')['item_id'].apply(list).to_dict()

mapper = generate_lightfm_recs_mapper(
    lfm_model, 
    item_ids=all_cols, 
    known_items=known_items,
    N=top_N,
    user_features=None, 
    item_features=None, 
    user_mapping=lightfm_mapping['users_mapping'],
    item_inv_mapping=lightfm_mapping['items_inv_mapping'],
    num_threads=20
)

lfm_prediction['item_id'] = lfm_prediction['user_id'].map(mapper)
lfm_prediction = lfm_prediction.explode('item_id').reset_index(drop=True)
lfm_prediction['rank'] = lfm_prediction.groupby('user_id').cumcount() + 1

lfm_prediction.to_csv (r'lfm_data.csv', index= False )

with open('fit_log.txt', 'w') as file:
    file.write(f'LightFM предсказания сохранены\n')

lfm_metrics = compute_metrics(test[['user_id', 'item_id']],
                              lfm_prediction,
                              top_N=10)

with open('lfm_metrics_log.txt', 'w') as file:
    file.write(f'{lfm_metrics}\n')

with open('fit_log.txt', 'w') as file:
    file.write(f'Метрики модели сохранены\n')

recommender = MostPopularRecommender()

def predict(users, usefull_users):
    res = {}
    for i in users:
        if i in usefull_users:
            res[i] = list(lfm_prediction[lfm_prediction['user_id'] == i].iloc[:10].item_id)
        else:
            recommender.fit(train)
            test_preds = recommender.predict(test)
            res[i] = test_preds[0]
    return res



with open('user_idx.txt', 'r') as file:
    users = [line.strip() for line in file]

with open('fit_log.txt', 'w') as file:
    file.write(f'Считывание списка пользователей окончено\n')

a = predict(users, usefull_users)

with open('output.txt', 'w') as file:
    for key, value in a.items():
        file.write(f'{key}: {value}\n')

with open('fit_log.txt', 'w') as file:
    file.write(f'Рекомендации записаны в файл\n')


































