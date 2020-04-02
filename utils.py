'''
Helper functions for this project

'''
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import scipy.stats as stats
from ast import literal_eval
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# define the constant variable OFFER_TYPES
OFFER_TYPES = ['bogo', 'discount', 'informational']

# define the constant variable OFFER_NAMES and append to portolio
OFFER_NAMES = ['bogo_1', 'bogo_2', 'informational_1', 'bogo_3', 'discount_1', 'discount_2', 'discount_3',
            'informational_2', 'bogo_4', 'discount_4']

def proc_col_value(text):
    '''
    INPUT
    text - the text inside the 'value' column

    OUTPUT
    f - rename if the text has'offer id' to 'offer_id'
    text - otherwise remains unchanged

    Description
    rename 'offer id' to 'offer_id' in 'value' column
    '''
    f = {}
    for k, v in text.items():
        if k == 'offer id':
            f['offer_id'] = v
        else:
            return text
    return f



def proc_offer_status(df, offers):
    '''
    INPUT
    df - the dataframe of one customer all transactions group by offer type or name
    offers - either offer types or offer name

    OUTPUT
    d - the dictionary of the counts of [received, viewed, completed] fore each 3 offers type or
         10 offers names

    Descriptions:
    process each every offer for all records by a customer
    the order of offer status list as [received, viewed, completed]
    '''

    d = {}

    for offer in offers:
        if offer in df.index :

            completed = viewed = received = 0

            # check if the customer got any offer completed record
            if 'offer completed' in df.columns:
                if np.isnan(df.loc[offer, 'offer completed']):
                    completed = 0
                else:
                    completed = df.loc[offer, 'offer completed']
            else:
                completed = 0

            # check if the customer got any offer viewed record
            if 'offer viewed' in df.columns:
                if np.isnan(df.loc[offer, 'offer viewed']):
                    viewed = 0
                else:
                    viewed = df.loc[offer, 'offer viewed']
            else:
                viewed = 0

            # check if the customer got any offer received record
            if 'offer received' in df.columns:
                if np.isnan(df.loc[offer, 'offer received']):
                    received = 0
                else:
                    received = df.loc[offer, 'offer received']
            else:
                received = 0

            d[offer] = [received, viewed, completed]

        else:
            d[offer] = 0

    return d



def proc_offers_and_extends(col, df):
    '''
    INPUT
    col - the column name to process
    df - the merged the new transaction dataframe


    OUTPUT
    df_tmp - the new aggregates information of customers

    Description:
    process the column of the new transaction dataframe and aggregate information.
    use the two CONSTANT variables: OFFER_TYPES, OFFER_NAMES

    '''
    customers = list(df['person'].unique())
    tmp = {}
    cnt = 0
    for customer in tqdm(customers):
        #print(cnt)
        #print(customer)
        df_tmp = None
        df_tmp1 = None
        d2 = {}
        d3 = {}

        # get the aggregate information from new transcation dataframe
        d1 = df[df.person == customer][col].value_counts().to_dict()

        # check if the custmer resposed to any offer response
        lst = list(df.query("person == @customer")['event'])


        if 'offer received' in lst  or 'offer completed' in lst or 'off viewed' in lst:
            # get the cumtomer transcation record and group by offer name
            df_tmp = df.query("person == @customer").groupby('offer_name')[col].value_counts().unstack()
            df_tmp1 = df.query("person == @customer").groupby('offer_type')[col].value_counts().unstack()
            d2 = proc_offer_status(df_tmp, OFFER_NAMES)
            d3 = proc_offer_status(df_tmp1, OFFER_TYPES)
        else:
            # no any offer response
            for offer in OFFER_NAMES:
                d2[offer] = 0
            for offer in OFFER_TYPES:
                d3[offer] = 0

        tmp[customer] = {**d1, **d2, **d3}
        cnt += 1
        #break

    # transpost the dataframe to proper shape
    df_tmp = pd.DataFrame(tmp).T

    # reset the index to a column
    df_tmp.reset_index(level=0, inplace=True)


    return df_tmp



def proc_offer_state(x, state, from_csv=False):
    '''
    INPUT
    x - the cell of individual offer
    from_csv - boolean, check if the dataframe is read from csv file

    OUTPUT
    v - the value regarding to offer viewed count
    '''
    if from_csv:#if read value from csv file
        x = literal_eval(x)
    if x == 0:
        return 0
    if isinstance(x, list):
        if state == 'received':
            return x[0]
        if state == 'viewed':
            return x[1]
        if state == 'completed':
            return x[2]



def add_offer_status_features(offers, df, from_csv=False):
    '''
    INPUT
    offers - either offer names or offer type or both
    df - the target dataframe to expend columns
    from_csv - boolean, check if the dataframe is read from csv file

    OUTPUT
    df_target - the dataframe after adding extra features

    '''
    df_target = df.copy()

    # extend columns for each 10 offers or 3 kinds of offers or both
    for offer in offers:

        df1 = df2 = df3 = df_ = None
        df1 = pd.Series(df_target[offer].apply(proc_offer_state, args=('received',from_csv)), name=offer+"_received")
        df2 = pd.Series(df_target[offer].apply(proc_offer_state, args=('viewed',from_csv)), name=offer+"_viewed")
        if offer.find('informational') == -1:
            df3 = pd.Series(df_target[offer].apply(proc_offer_state, args=('completed',from_csv)), name=offer+"_completed")


        df_ = pd.concat([df1, df2, df3], axis=1)

        df_target = pd.concat([df_target, df_], axis=1)


    return df_target



def add_offer_rate_features(df):
    '''
    INPUT
    df - the target dataframe to expend columns
    from_csv - boolean, check if the dataframe is read from csv file

    OUTPUT
    df_target - the dataframe after adding extra features

    '''
    df_target = df.copy()

    # transaction_completed_ratio : transaction_count / offer_completed_total
    df_target['transaction_completed_ratio'] = df['transaction']/df['offer completed'].replace(0, np.nan)

    # offer_viewed_rate = offer_viewed_total / offer_received_total
    df_target['offer_viewed_rate'] = df['offer viewed']/df['offer received'].replace(0, np.nan)

    # offer_completed_received_rate = offer_completed_total / offer_received_total
    df_target['offer_completed_received_rate'] = df['offer completed']/df['offer received'].replace(0, np.nan)

    # offer_completed_viewed_rate = offer_completed_total / offer_viewed_total
    df_target['offer_completed_viewed_rate'] = df['offer completed']/df['offer viewed'].replace(0, np.nan)

    # for bogo_viewed_rate,  bogo_completed_received_rate, bogo_completed_viewed_rate
    # for discount_viewed_rate,  discount_completed_received_rate, discount_completed_viewed_rate
    for v in ['bogo', 'discount']:

        df_target[v+'_viewed_rate'] = df[v+'_received']/df[v+'_received'].replace(0, np.nan)
        df_target[v+'_completed_received_rate'] = df[v+'_completed']/df[v+'_received'].replace(0, np.nan)
        df_target[v+'_completed_viewed_rate'] = df[v+'_completed']/df[v+'_viewed'].replace(0, np.nan)



    # for informational_view_rate
    df_target['informational_viewed_rate'] = df['informational_received']/df['informational_received'].replace(0, np.nan)

    return df_target


def remove_outliers_col(col, df):
    '''
    INPUT
    col - the column intend to find outliers
    df - the source dataframe

    OUTPUT
    df - the dataframe after remove the outliers

    Description:
    using Tukey's Rule

    '''

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR_amount = q3 - q1
    max_value = q3 + 1.5*IQR_amount
    min_value = q1 - 1.5*IQR_amount

    print(min_value, max_value)

    df_ = df[(df[col] <= max_value) & (df[col] >= min_value)]

    return df_


def remove_outliers(df):
    '''
    INPUT
    df - the dataframe intended to remove outliers

    OUTPUT
    df_ - remove outliers - using Tukey's rule

    '''
    from collections import Counter

    mlist = []

    for feature in df.keys():

        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(df[feature], 25)

        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(df[feature], 75)

        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5*(Q3-Q1)

        dis = pd.DataFrame(df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))])

        # add all outliers' index to the list
        mlist = np.append(dis.index.values, mlist)


    # find 5 most occurrances in the outliers' index list (write into a dataframe in order to get index)
    outliers  =  pd.DataFrame(Counter(mlist).most_common(5))[0].tolist()

    # convert to integer (used by the index method for removal)
    outliers = [int(v) for v in outliers]
    print ("Following records are outliers for more than one feature:", outliers)

    # Remove the outliers, if any were specified
    df_ = df.drop(df.index[outliers]).reset_index(drop = True)

    return df_



def apply_feature_scaling(df):
    '''
    INPUT
    data - the dataframe intended to perform feature scaling - standardization

    Description:
    apply feature scaling by standardization
    '''
    # feature scaling
    scaled_d  = {}
    for each in df.columns:
        mean, std = df[each].mean(), df[each].std()
        scaled_d[each] = [mean, std]
        df[each] = (df[each] - mean)/std



def plot_weight(df, pca, ith):
    '''
    INPUT
    df - the dataframe
    pca - the fitted PCA of the dataframe
    ith - the ordered of PCA component

    OUTPUT
    ax - the barplot object

    Description
    plot weight for the i-th principal component to corresponding feature names
    the code is copied from open source
    '''
    sort_pca = sorted([(weight, label) for weight, label in zip(pca.components_[ith-1], df.columns)],
                      reverse=True)
    weights, features = zip(*sort_pca)
    weights, features = list(weights), list(features)
    fig, ax = plt.subplots(figsize=(10,20))
    ax = sns.barplot(weights, features)
    ax.set_title(f"Component {ith}")

    return ax




def finding_num_cluster(data, min_clusters=2, max_clusters=15):
    '''
    INPUT
    data - the dataframe in order to find number of clusters
    min_clusters - the minimum number of clusters
    max_clusters - the maximum number of clusters

    Desctiption:
    doing KMeans cluster and return silhouette score and model inertia
    based on that and draw a graph
    the code copied from open source
    '''
    silh = []
    inertia = []
    clusters = range(min_clusters,max_clusters)
    for n in tqdm(clusters):

        model = KMeans(n_clusters = n, random_state=1)
        preds = model.fit_predict(data)

        silhouette_avg = silhouette_score(data, preds)

        silh.append(silhouette_avg)
        inertia.append(model.inertia_)

    fig, (ax1,ax2) = plt.subplots(2,1, sharex=False, figsize=(8,7))
    ax1.plot(clusters, silh,marker="o")
    ax1.set_ylabel("Silhoutte Score")
    ax1.set_xlabel('number of clusters')
    ax2.plot(clusters, inertia, marker="o")
    ax2.set_ylabel("Inertia (SSE)")
    ax2.set_xlabel("number of clusters")
    plt.show()


def cal_statistic(df, K, cat='mean'):
    '''
    INPUT
    df - the source dataframe
    cat - either mean or standard deviation or variance

    OUTPUT
    stat - the dataframe features by statistic

    Description:
    calculate each features
    '''
    # get each feature' mean or var or std for each cluster
    cluser_statistic = {}
    print(cat)
    if cat == 'mean':
        for k in range(K):
            features = {}
            for col in df.columns:
                features[col] = df[df['Cluster'] == k][col].mean()

            cluser_statistic[k] = features

    if cat == 'std':
        for k in range(K):
            features = {}
            for col in df.columns:
                features[col] = df[df['Cluster'] == k][col].std()

            cluser_statistic[k] = features

    if cat == 'var':
        for k in range(K):
            features = {}
            for col in df.columns:
                features[col] = df[df['Cluster'] == k][col].var()

            cluser_statistic[k] = features

    # transform to pandas dataframe
    df_ = pd.DataFrame(cluser_statistic)
    print(df.shape)

    return df_

def compare_features(df, features, xlabel='cluster', ylabel='mean value', figsize=(10,6)):
    """
    INPUT
    df - the dataframe for plotting
    features - the target features
    xlabel - Axes object figure element
    ylable - Axes object figure element
    figsize - Axes object figure element

    Description
    plot the selected features for comparison
    """
    n_cols = len(features)
    rows = n_cols//2 + (n_cols % 2 >0)

    f, axs = plt.subplots(rows, 2, figsize=figsize)
    axs = axs.flatten()

    for i in range(n_cols):
        sns.barplot(df.columns, df.loc[features[i]], ax=axs[i])
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title('Feature : ' + features[i])

    plt.tight_layout()


def t_test(df1, df2):
    '''
    INPUT
    df1 - the dataframe, sample size by feature size
    df2 - the dataframe, sample size by feature size,

    OUTPUT
    t_score - the list of t-score of each feature
    p_value - the list of p-value of each feature

    Description:
    The 2 dataframe must has same feature size, the number of columns must be same and comparable.

    '''
    # check if comparable - the number of columns
    assert df1.shape[1] == df2.shape[1]

    # get the number of features
    fea = df1.shape[1]

    # get the sample size - the number of rows
    n1 = df1.shape[0]
    n2 = df2.shape[0]


    # get sample variance and sample mean
    var1 = []
    var2 = []
    mean1 = []
    mean2 = []
    for idx in range(fea):
        var1.append(df1.iloc[:,idx].var())
        var2.append(df2.iloc[:,idx].var())
        mean1.append(df1.iloc[:,idx].mean())
        mean2.append(df2.iloc[:,idx].mean())

    # get t-score and p-value
    t_score = []
    p_value = []
    for i in range(len(var1)):
        t = (mean2[i]-mean1[i])/np.sqrt(var1[i]/n1 + var2[i]/n2)
        t_score.append(t)

        # degrees of freedom
        df = n1 + n2 - 2

        # p-value after comparison with the t
        p = 1 - stats.t.cdf(t, df=df)
        p_value.append(p)

    return t_score, p_value














