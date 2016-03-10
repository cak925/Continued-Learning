import pandas as pd

bat = pd.read_csv('baseball-csvs/batting.csv')
bat.head()
bat['BA'] = bat['H']/bat['AB']
bat['OBP'] = (bat['H']+bat['BB']+bat['HBP'])/(bat['AB']+bat['BB']+bat['HBP']+bat['SF'])

# before we do this wee need to calculate singles, which hopefull is jsut hits - sum(2b, 3b, hr)
bat['1B'] = bat['H'] - (bat['2B'] + bat['3B'] + bat['HR'] )

bat['SLG'] = (bat['1B'] + 2*bat['2B'] + 3*bat['3B'] + 4*bat['HR']) / bat['AB']
bat.drop('SP', axis=1)
bat.tail()

#merge batting and salary to see how much they make
sal = pd.read_csv('baseball-csvs/Salaries.csv')
ls
sal.head()
sal.yearID.min()
sal.yearID.max()
bat.columns
bat.yearID.min()
# remove all batting data before 1985
bat = bat[bat['yearID'] >= 1985]
bat.yearID.min()
bat_sal = pd.merge(bat, sal, on = ['playerID', 'yearID'])
bat_sal.tail()
bat_sal = bat_sal.drop(['G_batting','G_old','teamID_y','lgID_y'], axis=1) 

#create new df containing only oakland a's 2001 team.
two_one = bat_sal[bat_sal['yearID'] == 2001]
oak = bat_sal[bat_sal['teamID_x'] == 'OAK'] 
oak01 = oak[oak['yearID'] == 2001] 
oak01.head()
oak01 = oak01[['playerID','H','2B','3B','HR','OBP','SLG','BA', 'AB']]

lost = ['giambja01','damonjo01','isrinja01','saenzol01']
my_mask = oak01['playerID'].isin(lost)
lostboysdf = oak01[my_mask]
lostboysdf

two_one = bat_sal[bat_sal['yearID'] == 2001]    # all 2001 players
two_one = two_one[two_one['teamID_x'] != 'OAK']
two_one = two_one[two_one['AB'] >= 367]
two_one = two_one[two_one['salary'] < 15000000]

two_one['OBP'].describe()

m = two_one.sort('OBP', ascending=False)
m.head(4)
