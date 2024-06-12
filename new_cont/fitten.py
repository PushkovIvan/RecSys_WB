import pandas as pd
import random
from sklearn.model_selection import train_test_split

lfm_prediction = pd.read_csv("lfm_data.csv")
baseline = pd.read_csv("wb_school_task_1.csv")


baseline['order_ts'] = pd.to_datetime(baseline['order_ts'])

train, test = train_test_split(baseline, test_size= 0.16 , random_state= 0 )
train['order_ts'] = pd.to_datetime(train['order_ts'])
test['order_ts'] = pd.to_datetime(test['order_ts'])
test_idx = list(sorted(set(test.user_id)))

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
    
def predict(users, usefull_users):
    res = {}
    recommender.fit(train)
    test_preds = recommender.predict(test)
    for i in users:
        if i in usefull_users:
            res[i] = list(lfm_prediction[lfm_prediction['user_id'] == i].iloc[:10].item_id)
        else:
            
            res[i] = test_preds[random.randint(0,2)]
    return res

recommender = MostPopularRecommender()

with open('user_idx.txt', 'r') as file:
    users = [line.strip() for line in file]


a = predict(users, (lfm_prediction['user_id'].unique()))

with open('output.txt', 'w') as file:
    for key, value in a.items():
        file.write(f'{key}: {value}\n')