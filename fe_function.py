
import pandas as pd
from tqdm.auto import tqdm
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import implicit 

fe_path = './data/fe/'

def generate_customer_gender():
    '''
        In original dataset,
        it's doesn't include gender columns,
        but we can infer the information by purchasing history
    '''
    train = pd.read_parquet('./data/process/transactions_train_process.parquet')
    articles = pd.read_csv('./data/original/articles.csv')


    mp = {'Ladieswear':1, 'Baby/Children':0.5, 'Menswear':0, 'Sport':0.5, 'Divided':0.5}
    articles["index_group_name"] = articles["index_group_name"].map(mp)
    articles = articles.rename(columns={"index_group_name": "gender"})
    merged = pd.merge(train, articles[["article_id", "gender"]], on=["article_id"], how='inner')
    g = merged.groupby('customer_id').gender.mean().reset_index()
    g["gender"] = g["gender"].apply(lambda x: 1 if x >= 0.5 else 0) # 1 is female; 0 is male
    g.to_parquet(fe_path + 'customer_sex.parquet')

def generate_collaborative_filter():
    '''
        Recommend articles to customer by collaborative filter with score,
        and similar article with score.
    '''
    iterations = 100
    factors = 128
    transactions = pd.read_parquet('./data/process/transactions_train_process.parquet')
    customer = pd.read_parquet('./data/process/customers_train_process.parquet')
    df = transactions.groupby(["customer_id", "article_id"]).t_dat.count().rename("rate").reset_index()
    df['customer_label'], customer_idx = pd.factorize(df['customer_id'])
    df['article_label'], article_idx = pd.factorize(df['article_id'])
    sparse_article_customer = sparse.csr_matrix((df['rate'].astype(float), (df['article_label'], df['customer_label'])))
    sparse_customer_article = sparse_article_customer.T.tocsr()
    model = implicit.als.AlternatingLeastSquares(factors = factors, regularization=0.05, iterations=iterations,)
    model.fit(sparse_customer_article)
    customer_list = []
    article_list = []
    score_list = []
    c_set = set(customer.customer_id.unique())
    N = 16
    for i in tqdm(range(len(customer_idx))):
        r = model.recommend(i, sparse_customer_article[i], N, filter_already_liked_items=False)
        customer_list += [customer_idx[i]] * N
        article_list += list(article_idx[r[0]])
        score_list += list(r[1])
    recommendation_df = pd.DataFrame({
        "customer_id": customer_list,
        "recommendation_article_id": article_list,
        "score": score_list
    })
    recommendation_df.to_parquet(fe_path + 'customer_recommendation_fully.parquet')
    
    K = 4
    similar_items = model.similar_items(df['article_label'].unique(), N=K)
    article_id_list = []
    recommendation_article_id_list = []
    score_list = []
    for i in tqdm(df['article_label'].unique()):
        article_id_list += [article_idx[i]] * K
        recommendation_article_id_list += list(article_idx[similar_items[0][i]])
        score_list += list(similar_items[1][i])
    recommendation_df = pd.DataFrame({
        "article_id": article_id_list,
        "recommendation_article_id": recommendation_article_id_list,
        "score": score_list
    })
    recommendation_df.to_parquet(fe_path + 'als.parquet')


def generate_hotness_articles():
    '''
        To generate hot articles for canditions groupby gender and age;
        If the customer never purchases, use only age.
    '''
    if not os.path.isfile('./data/fe/customer_sex.parquet'):
        raise OSError("create customer_gender first")

    K = 100
    transactions = pd.read_parquet('./data/sample/transactions_fe_0.05.parquet')
    customer_sex = pd.read_parquet(fe_path + 'customer_sex.parquet')
    customer = pd.read_parquet('./data/sample/customers_fe_0.05.parquet')

    transactions.drop_duplicates(["customer_id", "t_dat", "week"], inplace=True)
    transactions_with_gender = transactions \
        .merge(customer_sex, on=["customer_id"], how="left") \
        .merge(customer[["customer_id", "age"]], on=["customer_id"], how="left")
    
    transactions_with_gender["year"] = 2022
    transactions_with_gender["modify_year"] = transactions_with_gender["t_dat"].apply(lambda x: x.year)
    transactions_with_gender["age"] = transactions_with_gender["age"] - (transactions_with_gender["year"] - transactions_with_gender["modify_year"])
    bins = [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 999]
    labels = [i for i in range(len(bins)-1)]
    transactions_with_gender["age_group"] = pd.cut(transactions_with_gender["age"], bins=bins, labels=labels, right=False)
    
    group_rank = transactions_with_gender.groupby(["week", "gender", "age_group"])["article_id"].value_counts() \
                    .groupby(["week", "gender", "age_group"]).rank(method="min", ascending=False) \
                    .rename("bestseller_rank").astype('int')
    group_rank = group_rank[group_rank <= K]
    group_rank.reset_index().to_parquet(fe_path + 'group_rank.parquet')

    general_rank = transactions_with_gender.groupby(["week"])["article_id"].value_counts() \
        .groupby('week').rank(method='min', ascending=False) \
        .rename('bestseller_rank').astype('int')
    general_rank = general_rank[general_rank <= K]
    general_rank.reset_index().to_parquet(fe_path + 'general_rank.parquet') # for cold start customer (no gender information)

def generate_customer_features():
    '''
        Customer features
    '''
    if not os.path.isfile('./data/fe/customer_sex.parquet'):
        raise OSError("create customer_gender first")
    transactions = pd.read_parquet('./data/process/transactions_train_process.parquet')
    customer = pd.read_parquet('./data/process/customers_train_process.parquet')
    customer_sex = pd.read_parquet(fe_path + 'customer_sex.parquet')
    customer.age -= 2 ## to change to situlation , 2 = 2022 - 2020
    customer.loc[customer.age == -3, "age"] = customer.age.mean() ## -3 = -1 - 2
    history_transactions = transactions[transactions.week < 80]
    customer_with_avg_consumption = customer.merge(history_transactions.groupby(["customer_id"]).price.mean().rename("customer_avg_consumption"), 
               on=["customer_id"], how="left")
    customer_with_avg_consumption = customer_with_avg_consumption.merge(customer_sex, on=["customer_id"], how="left")
    bins =  [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 999]
    labels = [i for i in range(len(bins)-1)]
    customer_with_avg_consumption["age_group"] = pd.cut(customer_with_avg_consumption["age"], bins=bins, labels=labels, right=False)
    customer_with_avg_consumption["customer_avg_consumption"].fillna( \
        customer_with_avg_consumption.groupby(["gender", "age_group"])["customer_avg_consumption"].transform('mean'), inplace=True)
    customer_with_avg_consumption["customer_avg_consumption"].fillna( \
        customer_with_avg_consumption.groupby(["age_group"])["customer_avg_consumption"].transform('mean'), inplace=True)
    customer_with_avg_consumption.customer_avg_consumption.fillna(customer_with_avg_consumption.customer_avg_consumption.mean(), 
                                                              inplace=True)
    customer_with_avg_consumption = customer_with_avg_consumption.merge(transactions.groupby(["customer_id"]).sales_channel_id.mean().round(),
                                                                    on=["customer_id"], how="left")
    customer_with_avg_consumption["sales_channel_id"].fillna( \
        customer_with_avg_consumption.groupby(["gender", "age_group"])["sales_channel_id"].transform('mean').round(), inplace=True)
    tmp = history_transactions.groupby(["customer_id"]).article_id.nunique()
    q = history_transactions.groupby(["customer_id", "article_id"]).size()
    tmp2 = q[q > 1].reset_index().groupby(["customer_id"]).article_id.count().rename("dup_article")
    tmp = pd.merge(tmp.reset_index(), tmp2, on=["customer_id"], how="left")
    tmp.dup_article.fillna(0, inplace=True)
    tmp["dup_rate"] = tmp["dup_article"] / tmp["article_id"]
    customer_with_avg_consumption = customer_with_avg_consumption.merge(tmp[["customer_id", "dup_rate"]], on=["customer_id"], how="left")
    customer_with_avg_consumption.dup_rate.fillna(
        customer_with_avg_consumption.groupby(["gender", "age_group"])["dup_rate"].transform('mean').round(), inplace=True
    )
    customer_with_avg_consumption.to_parquet(fe_path + 'customer_features.parquet')

def generate_dynamic_article_df():
    '''
        Article dynamic features
    '''
    transactions = pd.read_parquet('./data/sample/transactions_fe_0.05.parquet')
    article_ = pd.read_parquet('./data/sample/articles_fe_0.05.parquet')
    # history_transactions = transactions[transactions.week < 80]
    merged = pd.merge(history_transactions, article_, on=["article_id"], how="left")
    weekly_mean = transactions.groupby(["week", "article_id"])["price"].mean().reset_index()
    weekly_counts = transactions.groupby(["week", "article_id"]).size().rename("weekly_sell_counts").reset_index()
    weekly_counts["rank"] = weekly_counts.groupby(["week"]).weekly_sell_counts.rank("dense", pct=True)
    week_df = pd.DataFrame({
        "week":[i for i in range(0, 105)]
    })

    selled_article_df = pd.DataFrame({
        "article_id": weekly_mean.article_id.unique()
    })
    week_df_fake = pd.DataFrame({
        "week":[i for i in range(-4, 105)]
    })

    dynamic_article_df = pd.merge(week_df, selled_article_df, how="cross")
    dynamic_article_df = pd.merge(dynamic_article_df, weekly_mean, on=["article_id", "week"], how='left')
    dynamic_article_df.price = dynamic_article_df.groupby(["week", "article_id"])["price"].ffill().bfill()
    dynamic_article_df = pd.merge(dynamic_article_df, weekly_counts, on=["article_id", "week"], how='left')
    dynamic_article_df.weekly_sell_counts.fillna(0, inplace=True)
    dynamic_article_df.to_parquet(fe_path + 'dynamic_article_df.parquet')

def generate_static_article_df():
    '''
        Article static features
    '''
    transactions = pd.read_parquet('./data/process/transactions_train_process.parquet')
    article_ = pd.read_parquet('./data/process/articles_train_process.parquet')
    history_transactions = transactions[transactions.week < 80]
    tmp = history_transactions.groupby(["customer_id", "article_id"]).size().rename("hold").reset_index()
    tmp["hold"] = tmp["hold"].apply(lambda x: 0 if x == 1 else 1)
    hold_counts = tmp.groupby(["article_id"]).count()
    duplicate_counts = tmp.groupby(["article_id"])["hold"].sum()
    duplicate_hold_df = pd.merge(duplicate_counts.reset_index(), hold_counts.reset_index(), on=["article_id"])
    duplicate_hold_df["duplicate_hold_rate"] = round(duplicate_hold_df.hold_x / duplicate_hold_df.hold_y, 4)
    static_article_df = pd.merge(duplicate_hold_df[["article_id", "duplicate_hold_rate"]],
                                transactions.groupby(["article_id"]).size().rename("totaly_sell_counts").reset_index(),
                                on=["article_id"], how="right")
    static_article_df.duplicate_hold_rate.fillna(static_article_df.duplicate_hold_rate.mean(), inplace=True)
    static_article_df = pd.merge(article_, static_article_df, on=["article_id"], how="inner")
    static_article_df.to_parquet(fe_path + 'static_article_df.parquet')