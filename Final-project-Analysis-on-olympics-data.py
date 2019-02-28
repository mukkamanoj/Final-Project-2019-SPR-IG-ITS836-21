# Importing Required Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # Importing data

path1 = 'athlete_events.csv'
path2 = 'noc_regions.csv'
athlete_data = pd.read_csv(path1)
regions = pd.read_csv(path2)

# Overview of athlete events data
# Summary of athlete events data
print('\x1b[1;30mData Columns:\n\x1b[1;34m {}\n'.format(athlete_data.columns.tolist()))
print(athlete_data.head(5))

# We can see there are NaN values in the data lets find which all data is missing.<br>
# we should be considering not null values of Age, Weight, Height and Medals while analysis to avoid issues.

athlete_data.isnull().sum()

# Overview on Region Codes Data


# Summary of NOC regions data
print('\x1b[1;30mData Columns:\n\x1b[1;34m {}\n'.format(regions.columns.tolist()))
regions.head(5)

# Data Preparation

# Data Preparation by merging the Athletes data and NOC data using pandas merge. 
data = pd.merge(athlete_data, regions, on='NOC', how='left')
data.head(5)

# # Data Analysis based on Gender and Age


# Summary of Atheltes praticipated catagorized by Gender
gender_data = data['Sex'].value_counts()
labels = ['Male', 'Female']
plt.figure(figsize=(7, 7))
plt.pie(gender_data, labels=labels, autopct='%1.0f%%')
plt.title('Sex Ratio of Participants')
plt.legend(labels)
plt.show()

# Summary of Athletes winning medals catagorized by Age

medal_data = data[data.Medal.notnull()]
medals = medal_data[np.isfinite(medal_data['Age'])]
plt.figure(figsize=(70, 50))
sns.catplot(x='Age', y='Medal', data=medals, hue='Medal')
plt.show()

# **It is pretty interesting that there are Athletes who are older than 60 years winning Medals.**
# 
# **Let's get deeper into Athletes with age>60**


# It is pretty interesting that there are Athletes who are older than 60 years winning Medals
# Let's get deeper into Athletes with age>60

sports_over_60 = medals['Sport'][medals['Age'] > 60]
print('\x1b[1;30mSports Played By Athletes with Age>60.\n\x1b[1;34m {}\n'.format(sports_over_60.unique()))

# Let's get deeper into Athletes with age>60 by Gender
female_sports_over_60 = medals['Sport'][medals['Age'] > 60][medals['Sex'] == 'F']
print('\x1b[1;30mSports Played By Female Athletes with Age>60.\n\x1b[1;34m {}\n'.format(female_sports_over_60.unique()))

male_sports_over_60 = medals['Sport'][medals['Age'] > 60][medals['Sex'] == 'M']
print('\x1b[1;30mSports Played By Male Athletes with Age>60.\n\x1b[1;34m {}\n'.format(male_sports_over_60.unique()))

# It is clear that all the sports with winning atheltes older than 60 are about having keen eye sight.

# ## Let's get the youngest and oldest winning Athletes


print('## Youngest Athlete in olympics who won medal')
print(medals.loc[medals['Age'].idxmin()])

print('## Oldest Athlete in olympics who won medal')
print(medals.loc[medals['Age'].idxmax()])

top_sports = data.Sport.value_counts().nlargest(15)
top_sports.plot(kind='Bar', width=0.8, figsize=(40, 20), fontsize=24, color='teal')
plt.title('Top sports based on number of Athelets participated', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Sport', fontsize=36, fontweight='bold')
plt.ylabel('Number of Athletes', fontsize=36, fontweight='bold')
plt.show()

# Data Analysis based on the Weight and Height
# Athletes Weight Data Analysis

# Since we know there is data with no value for Weight lets take only finite valued data and see the trend over years
weight_data = data[np.isfinite(data['Weight'])]
mean_weight = (weight_data.groupby('Year')['Weight'].mean())
mean_weight.plot(figsize=(15, 5), fontsize=16)
plt.title('Average Weight in Olympics', fontsize=16)
plt.ylabel('Weight in KGs', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()

# ## Athletes Height Data Analysis

# Since we know there is data with no value for Weight lets take only finite valued data and see the trend over years
height_data = data[np.isfinite(data['Height'])]
mean_height = (height_data.groupby('Year')['Height'].mean())
mean_height.plot(figsize=(15, 5), fontsize=16)
plt.title('Average Height in Olympics', fontsize=16)
plt.ylabel('Height in cms', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()

gymnastics = weight_data[weight_data['Sport'] == 'Gymnastics']
gymnastics_medals = gymnastics[gymnastics.Medal.notnull()]
mean_gymnastics_weights = gymnastics_medals.groupby('Year')['Weight'].mean()
mean_gymnastics_weights.plot(figsize=(15, 5), fontsize=16)
plt.title('Average Weight in Gymnastics who won medals', fontsize=16)
plt.ylabel('Weight in KGs', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()
gymnastics = height_data[height_data['Sport'] == 'Gymnastics']
gymnastics_medals = gymnastics[gymnastics.Medal.notnull()]
mean_gymnastics_heights = gymnastics_medals.groupby('Year')['Height'].mean()
mean_gymnastics_heights.plot(figsize=(15, 5), fontsize=16)
plt.title('Average height in Gymnastics who won medals', fontsize=16)
plt.ylabel('Height in cms', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()

skiing = weight_data[weight_data['Sport'] == 'Alpine Skiing']
skiing_medals = skiing[skiing.Medal.notnull()]
mean_skiing_weights = skiing_medals.groupby('Year')['Weight'].mean()
mean_skiing_weights.plot(figsize=(15, 5), fontsize=16)
plt.title('Average Weight of Skiing won medals', fontsize=16)
plt.ylabel('Weight in KGs', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()
skiing = height_data[height_data['Sport'] == 'Alpine Skiing']
skiing_medals = skiing[skiing.Medal.notnull()]
mean_skiing_heights = skiing_medals.groupby('Year')['Height'].mean()
mean_skiing_heights.plot(figsize=(15, 5), fontsize=16)
plt.title('Average Height of Skiing won medals', fontsize=16)
plt.ylabel('Height in cms', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()

# We can see that the Weight and Height of Skiing Atheletes who won the medals increased while Weight and Height of
# Gymnastics Athletes who won the medals this is because gymnastics need less weight and height to achive good balance
# while skiing needs more height and weight to reach more distance and increase speed**

top_teams = data.Team.value_counts().nlargest(30)
print(top_teams[:10])

top_teams_medal_count = medal_data.Team.value_counts().nlargest(30)
print(top_teams_medal_count[:10])

dataf = pd.DataFrame({'Participants': top_teams, 'Medals': top_teams_medal_count})
dataf = dataf.dropna().sort_values(by='Participants', ascending=False)
dataf.plot(kind='bar', width=0.8, figsize=(40, 20), fontsize=24)
plt.title('Top Teams based on number of Medals and Participants', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('Number of Medals', fontsize=36, fontweight='bold')
plt.legend(fontsize=36)
plt.show()

medal_ratio = (top_teams_medal_count.div(top_teams)).mul(100).sort_values(ascending=False)
medal_ratio = medal_ratio.dropna()
medal_ratio.plot(kind='Bar', width=0.8, figsize=(40, 20), fontsize=24, color='green')
plt.title('Percentage of winning medals over participation by Team', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('% Percentage of Medals Won', fontsize=36, fontweight='bold')
plt.show()

gold_medals = data[data.Medal == 'Gold']
top_teams_gold_medal = gold_medals.Team.value_counts().nlargest(30)
dataf = pd.DataFrame({'Participants': top_teams, 'Gold Medals': top_teams_gold_medal})
dataf = dataf.dropna().sort_values(by='Participants', ascending=False)
dataf.plot(kind='bar', width=0.8, figsize=(40, 20), fontsize=24, color=['blue', 'gold'])
plt.title('Top Teams based on number of Medals and Participants', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('Number of Gold Medals', fontsize=36, fontweight='bold')
plt.legend(fontsize=36)
plt.show()

gold_medal_ratio = (top_teams_gold_medal.div(top_teams)).mul(100).sort_values(ascending=False)
gold_medal_ratio = gold_medal_ratio.dropna()
gold_medal_ratio.plot(kind='Bar', width=0.8, figsize=(40, 20), fontsize=24, color='gold')
plt.title('Percentage of winning gold medals over participation by Team', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('% Percentage of Gold Medals Won', fontsize=36, fontweight='bold')
plt.show()

silver_medals = data[data.Medal == 'Silver']
top_teams_silver_medal = silver_medals.Team.value_counts().nlargest(30)
dataf = pd.DataFrame({'Participants': top_teams, 'Silver Medals': top_teams_silver_medal})
dataf = dataf.dropna().sort_values(by='Participants', ascending=False)
dataf.plot(kind='bar', width=0.8, figsize=(40, 20), fontsize=24, color=['blue', 'grey'])
plt.title('Top Teams based on number of Medals and Participants', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('Number of Silver Medals', fontsize=36, fontweight='bold')
plt.legend(fontsize=36)
plt.show()

silver_medal_ratio = (top_teams_silver_medal.div(top_teams)).mul(100).sort_values(ascending=False)
silver_medal_ratio = silver_medal_ratio.dropna()
silver_medal_ratio.plot(kind='Bar', width=0.8, figsize=(40, 20), fontsize=24, color='grey')
plt.title('Percentage of winning silver medals over participation by Team', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('% Percentage of silver Medals Won', fontsize=36, fontweight='bold')
plt.show()

bronze_medals = data[data.Medal == 'Bronze']
top_teams_bronze_medal = bronze_medals.Team.value_counts().nlargest(30)
dataf = pd.DataFrame({'Participants': top_teams, 'Bronze Medals': top_teams_bronze_medal})
dataf = dataf.dropna().sort_values(by='Participants', ascending=False)
dataf.plot(kind='bar', width=0.8, figsize=(40, 20), fontsize=24, color=['blue', 'chocolate'])
plt.title('Top Teams based on number of Medals and Participants', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('Number of Silver Medals', fontsize=36, fontweight='bold')
plt.legend(fontsize=36)
plt.show()

bronze_medal_ratio = (top_teams_bronze_medal.div(top_teams)).mul(100).sort_values(ascending=False)
bronze_medal_ratio = bronze_medal_ratio.dropna()
bronze_medal_ratio.plot(kind='Bar', width=0.8, figsize=(40, 20), fontsize=24, color='chocolate')
plt.title('Percentage of winning bronze medals over participation by Team', fontsize=36, fontweight='bold')
plt.xlabel('Name of the Team', fontsize=36, fontweight='bold')
plt.ylabel('% Percentage of Bronze Medals Won', fontsize=36, fontweight='bold')
plt.show()
