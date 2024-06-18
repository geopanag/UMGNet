import pandas as pd 
import os
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np




def prepare_RetailHero(train_ind_file: str = "uplift_train.csv", feature_file: str="clients.csv" , purchases_file:str="purchases.csv",
                               features_processed_file:str ="features_processed_v4.csv", features_final_file:str = "user_features_v4.csv",  
                               edge_index_file:str = "user_product_v4.csv",
                               age_filter:int=16) -> None:
    """
    Clean the data, transform the time stamps and the categorical variables to one-hot, transform the class, scale the features and save the data in a csv file
    RetailHero dataset from https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    """
    encoder = OneHotEncoder()
    ### The list of clients participating in the study, the treatment and the outcome
    train = pd.read_csv(train_ind_file).set_index("client_id")

    ### The features of clients (age, sex , coupon issue etc.)
    df_features = pd.read_csv(feature_file)

    df_features['first_redeem_date'] = pd.to_datetime(df_features['first_redeem_date'])
    df_features['first_issue_abs_time'] = (pd.to_datetime(df_features['first_issue_date'])- pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    df_features['first_redeem_abs_time'] = (pd.to_datetime(df_features['first_redeem_date']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    df_features['redeem_delay'] = (df_features['first_redeem_abs_time'] - df_features['first_issue_abs_time'])

    df_features = df_features[df_features['age'] > age_filter]
    df_features = df_features[ df_features['redeem_delay']>0]
    df_features = df_features.reset_index(drop=True)

    one_hot_encoded = encoder.fit_transform(df_features[["gender"]])
    one_hot_encoded_array = one_hot_encoded.toarray()
    encoded_categories = encoder.categories_ 

    df_encoded = pd.DataFrame(one_hot_encoded_array, columns=encoded_categories[0])
    df_features = df_features.drop("gender",axis=1)

    columns = list(df_features.columns) + list(encoded_categories[0])
    df_features = pd.concat([df_features, df_encoded], axis=1,ignore_index=True)
    df_features.columns = columns

    df_features = train.join(df_features.set_index("client_id"))

    df_features = df_features[~df_features.age.isna()]

    ### Use the purchase list to take the extra features and define the network
    purchases = pd.read_csv(purchases_file)
    purchases = purchases[['client_id','transaction_id','transaction_datetime','purchase_sum','store_id','product_id','product_quantity']]
    purchases['transaction_datetime'] = pd.to_datetime(purchases['transaction_datetime'])
    purchases['transaction_abs_time'] = (purchases['transaction_datetime']- pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    #purchases.set_index("client_id", inplace=True)

    # remove users that are not part of the experiment
    dictionary = dict(zip(df_features.index, df_features['first_redeem_date']))
    purchases['first_redeem_date'] = purchases['client_id'].map(dictionary)
    purchases = purchases[~purchases['first_redeem_date'].isna()]
    dictionary = dict(zip(df_features.index, df_features['treatment_flg']))
    purchases['treatment_flg'] = purchases['client_id'].map(dictionary)

    client_map = {j:i for i,j in enumerate(purchases.client_id.unique())}
    product_map = {j:i for i,j in enumerate(purchases.product_id.unique())}
    store_map = {j:i for i,j in enumerate(purchases.store_id.unique())}

    # separate before and after treatment redeem
    ind = (purchases['transaction_datetime'] < purchases['first_redeem_date'])
    purchases_before = purchases[ind]
    purchases_after = purchases[~ind]

    ## calculate metrics on average over client and transactions
    features_purchases_before = purchases_before.groupby('transaction_id').agg({'client_id':'first','purchase_sum': 'first','transaction_datetime':"first"}).reset_index() 
    features_purchases_before.columns = ["transaction_id","client_id","purchase_sum","transaction_datetime"]
    features_purchases_before = features_purchases_before.groupby('client_id').agg({'purchase_sum': 'mean','transaction_id':'count','transaction_datetime':['max','min']})
    features_purchases_before.columns = ["avg_money_before","total_count_before","last_purchase_before","first_purchase_before"]
    features_purchases_before['avg_count_before'] = features_purchases_before['total_count_before']/((features_purchases_before['last_purchase_before'] - features_purchases_before['first_purchase_before']).dt.days+1)
    features_purchases_before = features_purchases_before[['avg_money_before','avg_count_before']]

    labels_purchases_after= purchases_after.groupby('transaction_id').agg({'client_id':'first','purchase_sum': 'first','transaction_datetime':"first"}).reset_index() 
    labels_purchases_after.columns = ["transaction_id","client_id","purchase_sum","transaction_datetime"]
    labels_purchases_after = labels_purchases_after.groupby('client_id').agg({'purchase_sum': 'mean','transaction_id':'count','transaction_datetime':['max','min']})
    labels_purchases_after.columns = ["avg_money_after","total_count_after","last_purchase_after","first_purchase_after"]
    labels_purchases_after['avg_count_after'] = labels_purchases_after['total_count_after']/((labels_purchases_after['last_purchase_after'] - labels_purchases_after['first_purchase_after']).dt.days+1)
    labels_purchases_after = labels_purchases_after[['avg_money_after','avg_count_after']]

    purchases_before['client_id'] = purchases_before['client_id'].map(client_map)
    purchases_before['product_id'] = purchases_before['product_id'].map(product_map)
    purchases_before['store_id'] = purchases_before['store_id'].map(store_map)
    purchases_after['client_id'] = purchases_after['client_id'].map(client_map)
    purchases_after['product_id'] = purchases_after['product_id'].map(product_map)
    purchases_after['store_id'] = purchases_after['store_id'].map(store_map)

    purchases_before['label'] = 0
    purchases_after['label'] = 1

    #purchases_before.groupby(['client_id','product_id','label']).agg({'product_quantity':'sum'}).reset_index()
    purchases_before = purchases_before.groupby(["client_id","product_id","label"]).sum("product_quantity").reset_index()
    purchases_after = purchases_after.groupby(["client_id","product_id","label"]).sum("product_quantity").reset_index()

    purchase_processed = pd.concat([ purchases_before, purchases_after])
    purchase_processed = purchase_processed[['client_id','product_id','label','product_quantity']]
    purchase_processed.columns = ['user','product','T','weight']
    purchase_processed.to_csv(edge_index_file, index=False)

    degrees = purchase_processed[ (purchase_processed['T']==0) ].groupby("user").size().reset_index()
    degrees = dict(zip(degrees['user'], degrees[0]))
    
    weighted_degrees = purchase_processed[ (purchase_processed['T']==0) ].groupby("user").sum("weight").reset_index()
    #print(weighted_degrees.head(5))
    weighted_degrees = dict(zip(weighted_degrees['user'], weighted_degrees['weight']))

    
    ## add targets
    data = df_features.join(features_purchases_before).join(labels_purchases_after).fillna(0)

    data['avg_money_change'] = data['avg_money_after'] - data['avg_money_before']
    data['avg_count_change'] = data['avg_count_after'] - data['avg_count_before']
    data = data[data.index.isin(purchases.client_id.unique())].reset_index()
    
    data['client_id'] = data['client_id'].map(client_map)
    
    data['degree_before'] = data['client_id'].map(degrees).fillna(0)
    data['weighted_degree_before'] = data['client_id'].map(weighted_degrees).fillna(0)
    data.to_csv(features_processed_file, index=False)

    treatment = ['treatment_flg']
    labels = ['target','avg_money_change','avg_count_change','avg_money_after','avg_count_after']
    features = ['age','F','M','U','first_issue_abs_time','first_redeem_abs_time','redeem_delay','avg_money_before','avg_count_before','degree_before','weighted_degree_before']

    data = data[treatment + labels + features]

    data.to_csv(features_final_file, index=False)



def prepare_Movielens():

    lab = "X"
    #===== graph
    edges = pd.read_csv("ratings.csv")
    edges = edges[['movieId','userId','rating']]
    edges.columns = ['user','product','weight']

    # reduce to fit the user x user one hot encoding in memory
    gx = edges.groupby(['product'])['user'].count()
    chosen_products = gx[gx>200].reset_index()['product'] #gx.mean()
    edges = edges[edges['product'].isin(chosen_products)]

    # define treated and untreated
    ratings = edges.groupby('user')['weight'].mean()
    ratings = ratings.reset_index()
    user_map = {j:i for i,j in enumerate(ratings.user.unique())}
    ratings['t'] =ratings.weight>=ratings.weight.median()

    edges['T'] = edges.weight > ratings.weight.median()
    edges['T'] = edges['T'].astype(int)

    # derive the mappings
    product_map = {j:i for i,j in enumerate(edges['product'].unique())}

    edges['user'] = edges['user'].map(user_map)
    ratings['user'] = ratings['user'].map(user_map)
    edges['product'] = edges['product'].map(product_map)

    edges.to_csv("movielens_graph_filtered_"+lab+".csv",index=False)


    #===== treatment
    movies = pd.read_csv("movies.csv")

    movies['movieId'] = movies['movieId'].map(user_map)

    movies = movies[movies['movieId'].isin(edges.user.unique())]

    ratings['t'] = ratings['t'].astype(int)
    #ratings.to_csv("movielens_treatments.csv",index=False)
    dict_treatment = dict(zip(ratings['user'], ratings['t']))
    movies['t'] = movies['movieId'].map(dict_treatment)

    #===== features
    moviesd = np.expand_dims(movies['movieId'].values, axis=0).T
    treatmentd = np.expand_dims(movies['t'].values, axis=0).T
    print('making features')

    movies['sentence'] = " title: "+movies['title']+" genres:"+movies['genres']
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda') # use multilingual models for texts with non-english characters
    embeddings_lite = model.encode(movies['sentence'].values.tolist())
    #pd.DataFrame(np.hstack([moviesd,treatmentd,embeddings_lite])).to_csv("movielens_features_full.csv",index=False)

    pca = PCA(n_components=16) 
    embeddings_lite = pca.fit_transform(embeddings_lite)
    x = np.hstack([moviesd,treatmentd,embeddings_lite])
    x = pd.DataFrame(x).sort_values(0)
    x = pd.DataFrame(x.values[:,1:])

    x.to_csv("movielens_features_filtered_"+lab+".csv",index=False)
    x = x.values

    # making outcomes
    #mean = 0
    #std_dev = 5
    mean = 10
    std_dev =5 
    bound = 10
    # Number of samples to generate
    for run in range(5):
        np.random.seed(run)
        # Generate random Gaussian noise
        random_noise = np.random.normal(mean, std_dev, x.shape[0])
        #random_wf = np.random.uniform(bound, bound, size=x.shape[1]-1)
        #random_wt = np.random.uniform(bound, bound, size=1)
        random_wf = np.random.uniform(bound, 2*bound, size=x.shape[1]-1)
        random_wt = np.random.uniform(bound, 2*bound, size=1)
        #Y = x[:,0]*random_wt+x[:,1:].dot(random_wf)+random_noise
        #Y -= min(Y)
        Y = np.maximum(0,x[:,0]*random_wt+x[:,1:].dot(random_wf)+random_noise)
        pd.DataFrame(Y).to_csv(f'movielens_y_{run}_filtered_{lab}.csv',index=False)
    #pca = PCA(n_components=16) 
        



def main():
    os.chdir("../data/RetailHero")
    prepare_RetailHero()
    os.chdir("../data/Movielens25")
    prepare_Movielens()
  

if __name__ == '__main__':
    main()
