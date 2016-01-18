#!/usr/bin/python
from TwitterAPI import TwitterAPI
import json
import sys
import pymongo

class Coordinates:
    def __init__(self, longitude, latitude):
        self.longitude = longitude
        self.latitude = latitude

    def in_bbox(self, bbox):
        if self.latitude < bbox.sw.latitude:
            return False
        if self.latitude > bbox.ne.latitude:
            return False
        if self.longitude < bbox.sw.longitude:
            return False
        if self.longitude > bbox.ne.longitude:
            return False
        return True

    def __repr__(self):
        return "%f,%f" % (self.longitude, self.latitude)
    
class BoundingBox:
    def __init__(self, sw_coord, ne_coord):
        self.sw = sw_coord
        self.ne = ne_coord

    def __repr__(self):
        return "%s,%s" % (repr(self.sw), repr(self.ne))

    def get_tweets(self, credentials_path, collection_name):
        cred = open_credentials(credentials_path)
        mongo = pymongo.MongoClient()
        api = TwitterAPI(**cred)
        query = {'locations': self.__repr__()}
        r = api.request('statuses/filter', query)
        for item in r.get_iterator():
            item['_id'] = item['id']
            mongo.tweets[collection_name].update_one({'_id': item['_id']}, {'$set': item}, upsert=True)
    
        
def open_credentials(credentials_file):
    with open(credentials_file) as f:
        return json.load(f)

def print_usage(args):
    print "Usage: %s <credentials> <citycode>" % args[0]
    print "\nArguments:"
    print "\tcredentials\tis the path to the json file containing the credentials"
    print "\tcitycode\tcan only be ny or la"
    
if __name__ == '__main__':
    ny_bbox = BoundingBox(Coordinates(-74.259090, 40.491370),
                          Coordinates(-73.700272, 40.951256))
    la_bbox = BoundingBox(Coordinates(-118.668176,33.703692),
                          Coordinates(-118.155289,34.337306))
    bboxes = {
        'ny': ny_bbox,
        'la': la_bbox
    }
    
    if len(sys.argv) != 3 or sys.argv[2].lower() not in bboxes.keys():
        print_usage(sys.argv)
        sys.exit(1)
    bbox_name = sys.argv[2].lower()
    bbox = bboxes[bbox_name]
    bbox.get_tweets(sys.argv[1], 'location_%s' % bbox_name)
    
