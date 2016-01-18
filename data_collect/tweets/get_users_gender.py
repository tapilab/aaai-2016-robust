#!/usr/bin/python3.4
import sys
from pymongo import MongoClient
import urllib.request
from bson.code import Code
from bson.son import SON

def load_census(n_males=100, n_females=None):
    if n_females is None:
        n_females = n_males

    female_url = "http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first"
    male_url = "http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first"

    female_request = urllib.request.urlopen(female_url)
    male_request = urllib.request.urlopen(male_url)

    line2name = lambda x: x.decode('utf-8').split()[0].lower() if x else ''

    set_female = set([line2name(female_request.readline()) for _ in range(n_females)])
    set_male = set([line2name(male_request.readline()) for _ in range(n_males)])
    set_ambiguous = set_female & set_male
    set_female -= set_ambiguous
    set_male -= set_ambiguous

    return set_male, set_female, set_ambiguous


def main(args):
    n_male = int(args[2])
    n_female = int(args[3])
    set_male, set_female, set_ambiguous = load_census(n_male, n_female)
    mapper = Code("""
function () {
    function include(arr,obj) {
        return (arr.indexOf(obj) != -1);
    };
    var females = %s;
    var males = %s;
    var uid = this.user.id_str;
    var firstName = this.user.name.split(" ")[0].toLowerCase();
    var gender = "na";
    if (include(females, firstName)) {
        gender = "f";
    } else if (include(males, firstName)) {
        gender = "m";
    }
 
    var cond_followers = this.user.followers_count > 10 && this.user.followers_count < 1000;
    var cond_friends = this.user.friends_count > 10 && this.user.friends_count < 1000;
    var cond_statuses = this.user.statuses_count < 5000;
    value = {
        gender: gender,
        name: this.user.name,
        followers: this.user.followers_count,
        friends: this.user.friends_count,
        statuses: this.user.statuses_count,
        conds: {
           followers: cond_followers,
           friends: cond_friends,
           statuses: cond_statuses
        }
    };
    if (gender != "na" && cond_followers && cond_friends && cond_statuses)
        emit(uid, value);
};
""" % (list(set_female),
       list(set_male)))

    reducer = Code("""
function(uid, values) {
    return values[0];
};
""")
    city = args[1].lower()
    mongo = MongoClient()
    out = SON([("merge", "gender_%s_m%d_f%d" % (city, n_male, n_female)), ("db", "users")])
    mongo.tweets["location_%s" % city].map_reduce(mapper, reducer, out)

def print_usage(args):
    print("Usage: %s <citycode> <n_male> <n_female>")
    print("Arguments:\n\tcitycode\tcan be ny or la only")
    print("\tn_male\tnumber of men names to use")
    print("\tn_female\tnumber of women names to use")
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print_usage(sys.argv)
        sys.exit(1)
    main(sys.argv)
