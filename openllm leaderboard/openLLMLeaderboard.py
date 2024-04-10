from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import subprocess

shutil.rmtree("./data")
os.mkdir("./data")

subprocess.call("python ./scrape-open-llm-leaderboard/main.py -csv")

data = pd.read_csv('open-llm-leaderboard.csv')
data = data[(data['T'] == '💬') & (data['Flagged'] == False)]
data['Average'] = data[['HellaSwag', 'Winogrande']].mean(axis=1)
data = data[['Architecture', 'Average','HellaSwag', 'Winogrande', '#Params (B)', 'Model']]

data = data.sort_values('Average', ascending= False)



sns.histplot(data=data, x='#Params (B)',  bins=1, kde=True)
plt.grid(True)
plt.savefig("./data/histogram.png")
dataLow = data[(data['#Params (B)'] >= 0) & (data['#Params (B)'] < 25)].reset_index(drop=True)
dataLow.index = dataLow.index + 1
dataMediumLow = data[(data['#Params (B)'] >= 25) & (data['#Params (B)'] < 42)].reset_index(drop=True)
dataMediumLow.index = dataMediumLow.index + 1
dataMediumHigh = data[(data['#Params (B)'] >= 42) & (data['#Params (B)'] < 58)].reset_index(drop=True)
dataMediumHigh.index = dataMediumHigh.index + 1
dataHigh = data[(data['#Params (B)'] >= 58) & (data['#Params (B)'] <= 80)].reset_index(drop=True)
dataHigh.index = dataHigh.index + 1


print(f"""
data low 			[0 , 25[		: 
{dataLow.head().to_string(justify='left', index= True)}
data medium low		[25, 42[		: 
{dataMediumLow.head().to_string(justify='left', index= True)}
data medium high 	[42, 58[		: 
{dataMediumHigh.head().to_string(justify='left', index= True)}
data high			[58, 80]		: 
{dataHigh.head().to_string(justify='left', index= True)}
      """)

dataLow.to_csv("./data/ModelPerformance_0_25.csv")
dataMediumLow.to_csv("./data/ModelPerformance_25_42.csv")
dataMediumHigh.to_csv("./data/ModelPerformance_42_58.csv")
dataHigh.to_csv("./data/ModelPerformance_58_80.csv")


average_25_percent_largest = lambda x: x.nlargest(ceil(len(x)*0.25)).mean()
filtered_data = data[~data['Architecture'].isin(['?', 'Unknown'])]
averages = filtered_data.groupby('Architecture')['Average'].apply(average_25_percent_largest)
counts = filtered_data.groupby('Architecture').size()
averageSize = filtered_data.groupby('Architecture')['#Params (B)'].median()
grouped = pd.DataFrame({'Average': averages,'#Params (B)': averageSize,'# Entries': counts}).sort_values('Average', ascending= False).reset_index()
grouped.index = grouped.index + 1
print(grouped)	

grouped.to_csv("./data/ArchitecturePerformance.csv")