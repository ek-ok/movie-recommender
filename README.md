# Movie Recommender

### Overview
This project is to build 3 different recommendation engines.
- **Baseline**: this is our baseline to compare all the other recommendations against. It just takes mean ratings and recommends the most popular movies to everyone
- **Neighbor base**: this uses cosine similarity internally to recommend movies
- **ALS**: this uses Spark ALS implementation to recommend movies


### Structure
This project is broken down to a few components. Final results are summarized in `final_recommendation_results.ipynb`.
All the training was performed in `model_recommendation.ipynb` using algorithms in `source/model_recommender.py` files.
Data is stored in `data` dir and results are generated in `data/results` dir.

```bash
.
├── README.md
├── als_recommendation.ipynb
├── baseline_recommendation.ipynb
├── final_recommendation_results.ipynb
├── neighbor_base_recommendation.ipynb
├── data
│   ├── ml-20m
│   │   ├── README.txt
│   │   ├── genome-scores.csv
│   │   ├── genome-tags.csv
│   │   ├── links.csv
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   └── tags.csv
│   ├── results
│   │   ├── als_100000_distribution_rmse_movie.pkl
│   │   ├── als_100000_distribution_rmse_user.pkl
│   │   ├── als_100000_distribution_topk_user.pkl
│   │   ├── als_100000_hyperparameter_tuning_for_reg_param_rmse_.pkl
│   │   ├── als_150000_distribution_rmse_movie.pkl
│   │   ├── als_150000_distribution_rmse_user.pkl
│   │   ├── als_150000_distribution_topk_user.pkl
│   │   ├── als_150000_hyperparameter_tuning_for_reg_param_rmse_.pkl
│   │   ├── als_50000_distribution_rmse_movie.pkl
│   │   ├── als_50000_distribution_rmse_user.pkl
│   │   ├── als_50000_distribution_topk_user.pkl
│   │   └── als_50000_hyperparameter_tuning_for_reg_param_rmse_.pkl
├── environment.yml
└── source
    ├── als_recommender.py
    ├── baseline_recommender.py
    ├── neighbor_based_recommender.py
    └── utils.py
```

### Setup
Install Spark version 2.4.4 and then set up conda env with `conda env create -f environment.yml`
