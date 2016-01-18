def sample_users_from_db(city_handles, size, rand=np.random.RandomState(111191)):
    per_city_size = int(size/len(city_handles))
    per_gender_size = per_city_size/2
    sample_users_id = []
    for city_h, city_name in city_handles:
        city_sample_users = []
        for g in ['m', 'f']:
            users = np.array([u['_id'] for u in city_h.find({'gender': g}, {'_id': 1})])
            rand.shuffle(users)
            city_sample_users.extend(users[:per_gender_size])
        city_users = []
        for u in city_h.find({'_id': {'$in': city_sample_users}}):
            u['location'] = city_name
            city_users.append(u)
        sample_users_id.append(city_users)
    return np.array(sample_users_id)

def get_corpus(tweet_handles, all_users):
    total_tweets = 0
    for city_users, tweet_h in zip(all_users, tweet_handles):
        for user in city_users:
            content = ""
            if 'tweets' in user.keys():
                tweets_from_db = tweet_h.aggregate([
                        {'$match': {'_id': {'$in': user['tweets']}}},
                        {'$project': {'text': 1}},
                        {'$group': {'_id': '_id', 'corpus': {'$push': '$text'}}}
                    ])
                corpus = list(tweets_from_db)[0]['corpus']
                total_tweets += len(corpus)
                content = " ".join(corpus)
            yield(content)
            del content
    print(total_tweets)

t0 = datetime.now()
n = 6000
city_handles = [(mongo.users.la, 'la'), (mongo.users.ny, 'ny')]
all_users = sample_users_from_db(cities, n)
delta = datetime.now()-t0
print("Time to sample %d users: %f seconds" % (n, delta.total_seconds()))

t0 = datetime.now()
vec = CountVectorizer(binary=True, min_df=100)
corpus = get_corpus(tweet_handles, all_users)
X = vec.fit_transform(c for c in corpus)
delta = datetime.now()-t0
print("Time to vectorize the corpus: %f seconds" % delta.total_seconds())
