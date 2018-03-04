# Using Recommender Systems

import  numpy as np

from lightfm.datasets import fetch_movielens #100k data points csv
from lightfm import LightFM


# get data and format it

data = fetch_movielens(min_rating=4.0)


#print training and test data
print(repr(data['train']))

print(repr(data['test']))

#create model

model = LightFM(loss='warp') #Weighted Approximate - Rank Pairwise

#training
'''
epoch means the training cycle , num_threads means multi processing

'''

model.fit(data['train'],epochs=30, num_threads=2)

def sample_recommendation(model,data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input

    for user_id in user_ids:

        #movies they like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they like
        scores =model.predict(user_id,np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("user %s" % user_id)
        print("  Known Positives :")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("     %s" % x)


sample_recommendation(model,data,[3,24,450])

