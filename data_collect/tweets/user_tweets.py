#!/usr/bin/python3.4
from TwitterAPI import TwitterAPI, TwitterRestPager
import json
import sys
import pymongo
import time

def open_credentials(credentials_file):
    with open(credentials_file) as f:
        return json.load(f)

def print_usage(args):
    print("Usage: %s <credentials> <citycode>" % args[0])
    print("\nArguments:")
    print("\tcredentials\tis the path to the json file containing the credentials")
    print("\tcitycode\tcan only be ny or la")

def get_user_tweets(twitter_api, tweets_collection, users_collection, out_collection, uid):
    if out_collection.find({'_id': uid}).limit(1).count():
        return
    gender = users_collection.find({'_id': uid}).limit(1)[0]['value']['gender']
    user_out = {
        '_id': uid,
        'gender': gender
        }
    out_collection.insert(user_out)
    pager = TwitterRestPager(twitter_api, 'statuses/user_timeline', {'user_id': uid, 'count':200})
    for tw in pager.get_iterator():
        tw['_id'] = tw['id_str']
        tweets_collection.insert(tw)
        out_collection.update({'_id': uid}, {'$push': {'tweets': tw['_id']}})
    time.sleep(5)

if __name__== "__main__":
    if len(sys.argv) != 3:
        print_usage(sys.argv)
        sys.exit(1)

    mongo = pymongo.MongoClient()
    cred = open_credentials(sys.argv[1])
    api = TwitterAPI(**cred)
    citycode = sys.argv[2].lower()

    uids = mongo.users["gender_%s_m100_f220" % citycode].distinct("_id", {})
    tweets_collection = mongo.tweets["users_%s" % citycode]
    users_collection = mongo.users["gender_%s_m100_f220" % citycode]
    out_collection = mongo.users["%s" % citycode]

    for uid in uids:
        get_user_tweets(api, tweets_collection, users_collection, out_collection, uid)
