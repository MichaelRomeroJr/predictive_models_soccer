import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import poisson
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt # vizualize results

class Poisson_Model:
	def __init__(self, df):
		"""
			Poisson to calculate the most likely score-line of a match
				calculate the average number of goals each team is likely to score in that match. 
				This can be calculated by determining the “Attack Strength” and “Defence Strength” 
				for each team and comparing them.
				
				Attack Strength/Defence Strength depend on season average goals scored
				Returns Mean goals scored for home and away
			Args:
				df: pd.DataFrame of stats per match 
				home_team,away_team: team names of match to predict final score
		"""
		self.df = df

	def set_team_names(self,
			home_team, away_team
		):
		self.home_team = home_team
		self.away_team = away_team
		return

	def season_averages_stats(self, df):
		"""
		"""

		# season averages 
		season_number_of_games = self.df.shape[0] # each row is one game
		season_home_goals_scored = df['home_goals'].sum()
		season_away_goals_scored = df['away_goals'].sum()
		seasons_average_home_goals_scored_per_game =  season_home_goals_scored/season_number_of_games
		seasons_average_away_goals_scored_per_game =  season_away_goals_scored/season_number_of_games

		self.seasons_average_home_goals_scored_per_game = seasons_average_home_goals_scored_per_game
		self.seasons_average_away_goals_scored_per_game = seasons_average_away_goals_scored_per_game
		return


	def predict_goals(self):
		"""
			calculate the average number of goals each team is likely to score in that match
			Average goals scored as normalized aroudn 1
			Args:
				season averages stats:
					season_number_of_games
					season_away_goals_scored
			Calc home/away attack strength and defence strength
			mean value for goals 
			
			TODO: 
		"""
		df = self.df
		home_team, away_team = self.home_team, self.away_team

		#  home attack strength 
		"""
			Step - 1: Take the number of goals scored at home last season by the home team (Tottenham: 35) 
					and divide by the number of home games (35/19): 1.842.
			Step - 2: Divide this value by the season’s average home goals scored per game (1.842/1.492) to get an “Attack Strength” of 1.235.
		"""
		home_number_home_games = df[df['home_team'] == home_team].shape[0]
		home_goals_scored_total = df[df['home_team'] == home_team]['home_goals'].sum()
		home_average_home_goals_per_game = home_goals_scored_total / home_number_home_games
		home_attack_strength = home_average_home_goals_per_game / self.seasons_average_home_goals_scored_per_game
		# print(f"HOME TEAM'S home_attack_strength {home_attack_strength}")

		# away defence strength
		#     calc away team defence stength (number of goals given up while on road)
		"""
		Step - 1: Take the number of goals conceded away from home last season by the away team (Everton: 25) 
			and divide by the number of away games (25/19): 1.315.
		Step - 2: Divide this by the season’s average goals conceded by an away team per game (1.315/1.492) to get a “Defence Strength” of 0.881.
		"""
		away_number_away_games = df[df['away_team'] == away_team].shape[0]
		away_goals_allowed = df[df['away_team'] == away_team]['home_goals'].sum() # goals conceded by away team when away
		away_average_goals_conceded  = away_goals_allowed / away_number_away_games
		away_defence_strength = away_average_goals_conceded / self.seasons_average_home_goals_scored_per_game
		# print(f"AWAY TEAM'S away_defence_strength: {away_defence_strength}")

		# away attack strength
		away_team_number_away_games = df[df['away_team'] == away_team].shape[0]
		away_team_number_away_goals_scored = df[df['home_team'] == away_team]['home_goals'].sum()
		# away_team_number_away_goals_scored = df[df['away_team'] == away_team]['home_goals'].sum()
		away_attack_strength = away_team_number_away_goals_scored / self.seasons_average_away_goals_scored_per_game
		# print(f"AWAY attack strength: {away_attack_strength}")

		# home defence strength
		home_team_home_games_goals_conceded = df[df['home_team'] == home_team]['away_goals'].sum()
		home_average_goals_conceded = home_team_home_games_goals_conceded / home_number_home_games
		home_defence_strength = home_average_goals_conceded / self.seasons_average_home_goals_scored_per_game
		
		
		# predict home goals scored
		predict_home_goals = home_attack_strength*away_defence_strength*self.seasons_average_home_goals_scored_per_game
		# predict away goals scored
		predict_away_goals = away_attack_strength* home_defence_strength* self.seasons_average_away_goals_scored_per_game

		self.predict_home_goals = predict_home_goals
		self.predict_away_goals = predict_away_goals
		return [predict_home_goals, predict_away_goals]

	def poisson_pmfs(self, pred_home_goals,
			pred_away_goals
		):
		"""
		Probability mass function: is a function that gives the probability 
			that a discrete random variable is exactly equal to some value
			
			Check probably of event (goal scored) 0 through 6
			return dict of key: event x where is final score (i,j) for i,j=0,1,...6
				value: probablity of even occuring
		"""
		away_p_dist = [poisson.pmf(x, pred_away_goals) for x in range(0,6)]
		home_p_dist = [poisson.pmf(x, pred_home_goals) for x in range(0,6)]
		
		# key: final score (i,j) for i,j = 0,1,..6
		# value: probabilty of even (score) occuring
		pmfs = {} 
		for home_goals in range(0,6):
			for away_goals in range(0,6):
				score = f"{home_goals}_{away_goals}"
				pmfs[score] =  home_p_dist[home_goals] * away_p_dist[away_goals]
		self.pmfs = pmfs
		return 

	def predict_score(self):
		# key w/ max value
		v = list(self.pmfs.values())
		k = list(self.pmfs.keys())

		pred_correct_score = k[v.index(max(v))]
		correct_score_odds = self.pmfs[pred_correct_score]
		# print(f"Pred Outcome: {pred_correct_score}: {correct_score_odds}")
		
		self.pred_correct_score = pred_correct_score
		self.correct_score_odds = correct_score_odds
		return

	def vizualize_results(self, 
		pmfs={} # should create self.pmfs 
		):
	
		if self.pmfs:
			pmf_per_team = self.pmfs
		if pmfs:
			pmf_per_team = pmfs

		# key w/ max value
		v = list(pmf_per_team.values())
		k = list(pmf_per_team.keys())

		point_spread = k[v.index(max(v))]
		point_spread_probabilities = pmf_per_team[point_spread]
		print(f"Pred Outcome: {point_spread}: {point_spread_probabilities}")
		k = [i.replace('_','-') for i in k] # for readability

		width = 50
		height = 20
		plt.figure(figsize=(width, height)) 
		plt.xticks(fontsize=40, rotation=90)
		plt.yticks(fontsize=40)

		plt.plot(k, v, 'bo-', linewidth=2)

		font_size_ = 40
		plt.xlabel('Final Score ', fontsize=font_size_)
		plt.ylabel('Probability', fontsize=font_size_)
		plt.title('Likely Score of Match', fontsize=font_size_)
		return 
	
	def validation_by_season(
			data
		):
		"""
			script for checking PoissonModel() obj on each season
		"""
		p = Poisson_Model(df=data)

		# 2 Ways of Picking W-L.
		# Method 1: Poisson Average goals scored per team
		average_goals_scored_as_moneyline_accuracy_count = 0
		# Method 2: Poisson pmf of point spreads 
		point_spread_as_moneyline_accuracy_count = 0 # check if pointspread has correct winner
		point_spread_accuracy_count = 0 # check if we correctly predicted points spread

		# iterate through Games Data to check predictions (2 methods)
		for index,row in data.iterrows():
			# print(f"Match: {row['home_team']}v{row['away_team']} ({row['score']})")

			p.set_team_names(home_team=row['home_team'], away_team=row['away_team'], )
			p.season_averages_stats(df=data) # TODO: remove redunndancy

			# 2 ways of picking W-L
			# Average Goals scored per team
			predict_average_home_goals, predict_average_away_goals = p.predict_goals()
			# predict point spread of match
			p.poisson_pmfs(p.predict_home_goals,p.predict_away_goals)
			p.predict_score()
			p.pred_correct_score = p.pred_correct_score.replace('_','-')
			index = p.pred_correct_score.find('-')
			pred_home_moneyline_goals = int(p.pred_correct_score[:index])
			pred_away_moneyline_goals = int(p.pred_correct_score[index+1:])

			if row['home_goals'] > row['away_goals']: # Home Win
				# check average goals scored method (as moneyline and point spread)
				if predict_average_home_goals>predict_average_away_goals:
					average_goals_scored_as_moneyline_accuracy_count+=1
				# check point spread method
				if pred_home_moneyline_goals>pred_away_moneyline_goals:
					point_spread_as_moneyline_accuracy_count+=1
				# # correct point spread
				if (row['home_goals']==pred_home_moneyline_goals) and (row['away_goals']==pred_away_moneyline_goals):
					point_spread_accuracy_count+=1

			if row['home_goals'] < row['away_goals']: # Away Win
				if predict_average_home_goals<predict_average_away_goals:
					average_goals_scored_as_moneyline_accuracy_count+=1
				if pred_home_moneyline_goals<pred_away_moneyline_goals:
					point_spread_as_moneyline_accuracy_count+=1
				if (row['home_goals']==pred_home_moneyline_goals) and (row['away_goals']==pred_away_moneyline_goals):
					point_spread_accuracy_count+=1

			if row['home_goals'] == row['away_goals']: # Draw
				if predict_average_home_goals==predict_average_away_goals:
					average_goals_scored_as_moneyline_accuracy_count+=1
				if pred_home_moneyline_goals==pred_away_moneyline_goals:
					point_spread_as_moneyline_accuracy_count+=1
				if (row['home_goals']==pred_home_moneyline_goals) and (row['away_goals']==pred_away_moneyline_goals):
					point_spread_accuracy_count+=1
		
		display_digits = 6
		total_games = data.shape[0]
		poisson_average_as_ml_precision_ratio = average_goals_scored_as_moneyline_accuracy_count/total_games
		poisson_average_as_ml_precision_percent = str(poisson_average_as_ml_precision_ratio*100)[:display_digits]
		print(f"Poisson Average Goals scored to predict ML: {poisson_average_as_ml_precision_percent}%")

		poisson_pmf_pointspread_as_ml_precision_ratio = point_spread_as_moneyline_accuracy_count/total_games
		poisson_pmf_pointspread_as_ml_precision_percent = str(poisson_pmf_pointspread_as_ml_precision_ratio*100)[:display_digits]
		print(f"Poisson pmf pointspread to predict ML: {poisson_pmf_pointspread_as_ml_precision_percent}%") 

		poisson_pmf_pointspread_ratio = point_spread_accuracy_count/total_games
		poisson_pmf_pointspread_percent = str(poisson_pmf_pointspread_ratio*100)[:display_digits]
		print(f"Poisson pmf predicts exact match score: {poisson_pmf_pointspread_percent}%")  
		results = {
			"Poisson_Average_Goals_as_Money_Line": poisson_average_as_ml_precision_ratio,
			"Poisson_Point_Spread_as_Money_Line": poisson_pmf_pointspread_as_ml_precision_ratio,
			"Poisson_Point_Spread_Predicts_Score": poisson_pmf_pointspread_ratio
		}  
		return results


class SVM_Model:
	def __init__(self, df):
		"""
		"""
		self.df = df
		
	def train_test_split(self, independent_features, dependent_features):
		# features
		x = self.df.loc[:, independent_features]
		y = self.df.loc[:, dependent_features]
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.30)
		return

	def train_model(self):
		svc_predict = SVC()
		self.svc_predict = svc_predict.fit(self.x_train, self.y_train)
		return self.svc_predict
	
	def validate(self,
			test_data={}
		):

		test_data = self.x_test
		accuracy_count=0
		for index,row in test_data[:].iterrows():
			try: 
				match = self.df[(self.df['home_team_id']==row['home_team_id']) & (self.df['away_team_id']==row['away_team_id'])]
				home_team = match['home_team'].values[0]
				away_team = match['away_team'].values[0]
				home_goals = match['home_goals'].values[0]
				away_goals = match['away_goals'].values[0]

				pred_ = self.svc_predict.predict([test_data.loc[index]])[0]
				if pred_==0: # away win 
					if home_goals < away_goals:
						accuracy_count+=1
				if pred_==1: # draw
					if home_goals == away_goals:
						accuracy_count+=1
				if pred_ ==2: # home win
					if home_goals > away_goals:
						accuracy_count+=1

			except Exception as e: 
				print(f"err: {row}")
				print(e)

		# print(f"{accuracy_count}/{test_data.shape[0]}")
		precision_ratio = accuracy_count / test_data.shape[0]
		print(f"SVM predicts correct moneyline: {str(precision_ratio)[:5]}%")

		results = {
			"SVM_Money_Line_Accuracy": precision_ratio
		}
		return results
	

class LogisticRegressionModel:
	"""
		TODO: 
			- Add function to predict outcome of single match
			- Find different log loss / gradient descent alg
				so results have less variance 
	"""
	def __init__(self, df):
		"""
		"""
		self.df = df
		self.clf = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
						   intercept_scaling=1, class_weight='balanced', random_state=None,
						   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0)
		
	def train_test_split(self,
			independent_features,
			dependent_features
		):
		
		# features
		x = self.df.loc[:, independent_features]
		y = self.df.loc[:, dependent_features]

		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.30) 
		return
		
	def train(self
		):
		self.clf.fit(self.x_train, np.ravel(self.y_train.values))
		return

	def predict_x_test(self
		):
		self.y_pred = self.clf.predict_proba(self.x_test)
		return 

	def validate_x_test(self
		):
		index_count=0 # count each for of x_test, that's index to corresponding prediction y_pred
		accuracy_count=0
		for index,row in self.x_test.iterrows():
			match = self.df[(self.df['home_team_id']==row['home_team_id']) & (self.df['away_team_id']==row['away_team_id'])]
			home_team_name = match['home_team_id']
			away_team_name = match['away_team_id']
			# print(f"{index}: {home_team_name}v{away_team_name}: score: {match['home_goals']}-{match['away_goals']}")

			m = max(self.y_pred[index_count])
			pred_FTR = list(self.y_pred[index_count]).index(m)
			if pred_FTR==match['FTR_Encode'].values[0]:
				accuracy_count+=1
			# print()
		precision_ratio = accuracy_count / self.x_test.shape[0]
		print(f"Logistic Regression predicts correct Money Line {str(precision_ratio*100)[:5]}%")

		results = {
			"LR_Money_Line_Accuracy": precision_ratio

		}
		return results
	

if __name__ == "__main__":
	pass