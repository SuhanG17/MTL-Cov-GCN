#!/bin/sh

# # locwise
# python -u param_search_feature_remove.py \
# --region us \
# --data_file us_incidence_rate_with_index.csv \
# --data_type incidence_us \
# --search_type loc_wise \
# --num_supp_vars 1 > train_us_loc_randomforest_xgboost_lightgbm.log
# #2>&1 & echo $! > command.pid

# suppwise
python -u param_search_feature_remove.py \
--region us \
--data_file us_incidence_rate_with_index.csv \
--data_type incidence_us \
--search_type supp_wise \
--num_supp_vars 3 > train_us_supp_fat_hos.log
#2>&1 & echo $! > command.pid


# Arima
# python -u param_search_feature_remove.py \
# --region us \
# --data_file us_incidence_rate_with_index.csv \
# --data_type incidence_us \
# --search_type arima \
# --num_supp_vars 1 > train_us_arima.log
# #2>&1 & echo $! > command.pid