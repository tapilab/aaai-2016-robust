# aaai-2016-robust

This github contains all the required code to replicate the experiments from the AAAI 2016 paper titled "Robust Text Classification in the Presence of Confounding Bias".

It is organized as follow:

```bash
.
├── data_collect # stores the script to collect the data
│   ├── hansard # scrapper to collect the data for the Canadian Parliament experiment
│   │   ├── hansard
│   │   │   ├── __init__.py
│   │   │   ├── items.py
│   │   │   ├── pipelines.py
│   │   │   ├── settings.py
│   │   │   └── spiders
│   │   │       ├── __init__.py
│   │   │       └── HansardSpider.py
│   │   └── scrapy.cfg
│   └── tweets
│       ├── geolocated_tweets.py
│       ├── get_users_gender.py
│       └── user_tweets.py
├── notebook
│   ├── __init__.py
│   ├── ba_c_study.py
│   ├── CanadianParliament.ipynb
│   ├── confound_plot.py
│   ├── DemographicTweets.ipynb
│   ├── find_confounder.py
│   ├── IMDb.ipynb
│   ├── imdb_alt.py
│   ├── imdb_confounding_experiments.py
│   ├── injecting_bias.py
│   ├── models.py
│   ├── most_changing_coef.py
│   ├── simpson_paradox.py
│   ├── top_terms_table.py
│   └── twitter_dataset2pickle.py
└── README.md

7 directories, 27 files
```
