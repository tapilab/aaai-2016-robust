# aaai-2016-robust

This github repository contains all the required code to replicate the experiments from the AAAI 2016 paper entitled "Robust Text Classification in the Presence of Confounding Bias".

## Dependencies

If you want to re-run the programs to collect the data from the Canadian Parliament, you will need [Python 2 and the Scrapy module](http://doc.scrapy.org/en/latest/faq.html#does-scrapy-work-with-python-3).

The rest of the repo uses Python 3 and the following Python modules:

- numpy
- scipy
- matplotlib
- scikit-learn
- bs4 (only for the Canadian Parliament experiments)

See requirements.txt for more details.

## Organization

Following is a description of the files that are available through this repo. The most important files are the iPython notebooks (\*.ipynb files), they use the other files to run the experiments and plot the results.

```bash
.
├── notebook # analysis folder where the iPython notebooks and the Python helper scripts are stored.
│   ├── CanadianParliament.ipynb # Canadian Parliament experiments.
│   ├── DemographicTweets.ipynb # Twitter experiments.
│   ├── IMDb.ipynb # IMDb experiments.
│   ├── __init__.py
│   ├── ba_c_study.py # script to evaluate the best C value for the back-door adjustment method.
│   ├── confound_plot.py # all functions used to plot the results of confounding experiments.
│   ├── imdb_confounding_experiments.py # IMDb experiments.
│   ├── injecting_bias.py # functions to inject confounding bias into a dataset.
│   ├── models.py # classification models (LR, SO, BA, etc).
│   ├── most_changing_coef.py # functions to compute and plot the change in coefficients.
│   ├── simpson_paradox.py # functions to compute and plot features that display Simpson's paradox.
│   └── top_terms_table.py # function to display the terms most correlated with the confounder.
├── data_collect # stores the script to collect the data
│   ├── hansard # scrapper to collect the data for the Canadian Parliament experiments.
│   │   ├── hansard
│   │   │   ├── __init__.py
│   │   │   ├── items.py
│   │   │   ├── pipelines.py
│   │   │   ├── settings.py
│   │   │   └── spiders
│   │   │       ├── __init__.py
│   │   │       └── HansardSpider.py
│   │   └── scrapy.cfg
│   └── tweets # scripts using the Twitter API to collect data for the Twitter experiments.
│       ├── geolocated_tweets.py
│       ├── get_users_gender.py
│       └── user_tweets.py
└── README.md # this help file.
```
