# predictive_models_soccer
Various models for predicting soccer match money lines

# Football Prediction Models with Machine Learning
Place Money Line bets (pick winning team) for any soccer match using historical data. <br>
`models.py` has 4 different prediction models <br>
`notebook.ipynb` workspace to test/validate each model


Results (2023 Season): 
- Poisson Model 1 (Distribution Avg): 54.079% accuracy
-  Poisson Model 2 (pmf for Point Spread): 51.518% 
- SVM Model: 45.198%
- Logistic Regression Model: 49.03%


Conclusion: All 4 models are better than picking the home team (45%) and randomly guessing (33%).

TODO:
  - Create [Machine Learning Playground](https://ml-playground.com/) to interact and vizualize results
  - More data (add features available before match like vegas odds, last 5 game averages, etc)






Donations directly fund our development efforts and cover essential hosting expenses.

<a href="https://buymeacoffee.com/michaelromerojr" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
