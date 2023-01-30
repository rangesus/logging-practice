import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
from string import punctuation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from pprint import pprint
from collections import Counter
from collections import defaultdict
from itertools import chain
import operator
import csv

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

from sentence_transformers import SentenceTransformer, util


#df = pd.read_csv('python_event_log.csv', sep='|', on_bad_lines='skip')
python_df = pd.read_csv('python_event_log.csv', sep='|')
python_loc = pd.read_csv('python_programs.csv')

java_df = pd.read_csv('java_event_log.csv', sep='|')
java_loc = pd.read_csv('java_programs.csv')
# filter out empty log statements for sentiment analysis
python_df['Activity'] = python_df['Activity'].str.strip()
python_df = python_df[python_df['Activity'].str.len() > 0]

java_df['Activity'] = java_df['Activity'].str.strip()
java_df = java_df[java_df['Activity'].str.len() > 0]


# merge dataframes to add full LOC to calculate percentage
merged_python_df = pd.merge(python_df, python_loc, on=["Case ID", "Project"])
merged_java_df = pd.merge(java_df, java_loc, on=["Case ID", "Project"])

# counts number of LS per file
merged_python_df['# of LS'] = merged_python_df.groupby(["Case ID", "Project"])['LOC'].transform('count')
merged_java_df['# of LS'] = merged_java_df.groupby(["Case ID", "Project"])['LOC'].transform('count')

# count at which percentage of file LS in located
merged_python_df['percentage'] = (merged_python_df['Start Timestamp'] * 100) / merged_python_df['LOC']
merged_python_df.to_csv("merged_python_df.csv", sep='|')

merged_java_df['percentage'] = (merged_java_df['Start Timestamp'] * 100) / merged_java_df['LOC']
#merged_java_df.to_csv("merged_java_df.csv", sep='|')

# calculate mean position percentage per file
merged_python_df = merged_python_df.groupby(["Case ID", "Project"])['percentage'].mean().reset_index(name='mean_percentage')
merged_java_df = merged_java_df.groupby(["Case ID", "Project"])['percentage'].mean().reset_index(name='mean_percentage')

#in case of multiple files with same name within a project the percentage could exceed 100% - should be fixed now
merged_python_df = merged_python_df[merged_python_df['mean_percentage'] <= 100]
#merged_python_df.to_csv("mean_merged_python_df.csv", sep='|')

merged_java_df = merged_java_df[merged_java_df['mean_percentage'] <= 100]
merged_java_df.to_csv("mean_merged_java_df.csv", sep='|')
# histogram of where logging points are distributed in files
bins = np.linspace(0, 100, 5)
plt.xticks(bins)
plt.hist(merged_python_df.iloc[:,2], bins)
plt.ylabel("Number of Files")
plt.savefig("figures/Python_file_location_distribution.svg")
plt.show()

bins = np.linspace(0, 100, 5)
plt.xticks(bins)
plt.hist(merged_java_df.iloc[:,2], bins)
plt.ylabel("Number of Files")
plt.savefig("figures/Java_file_location_distribution.svg")
plt.show()

# sentiment analysis as well as activity extraction

nlp =spacy.load("en_core_web_trf")

sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

python_sentiment_scores = []
python_activities_strict = []
python_activities = []
python_loose_activities = []
for index, row in python_df.iterrows():
    file = row[0]
    log_level = row[3]
    stmnt = row[4]
    stmnt = stmnt.lower()
    #stmnt = "".join([part for part in stmnt if part not in punctuation])
    if(stmnt != ""):
        blob = TextBlob(stmnt)
        sentiment = blob.sentiment.polarity
        scores = sid.polarity_scores(stmnt)
        python_sentiment_scores.append([blob, log_level, sentiment])
        for key in sorted(scores):
            result = scores[key]
            #print('{0}: {1}, '.format(key, scores[key]), end='') 
        #print("\n")

        stmnt = nlp(stmnt)
        prev = stmnt[0]
        activity = ""
        verb_found = False
        for word in stmnt:
            #or (prev != word and prev.pos_ == "NOUN" and word.pos_ == "NOUN")
            #if(str(word) == "API"):
                #print(word.pos_)
            if(word.pos_ == "VERB"):
                verb_found = True
            if(verb_found and word.pos_ == "NOUN"):
                python_loose_activities.append([stmnt, log_level, file])
            if((prev.pos_ == "VERB" and word.pos_ == "NOUN") ):
                #print("---------------------- TAGS: " + prev.tag_ + " --------------------")
                #print(str(prev) + " " + str(word) + " ACTIVITY FOUND")
                python_activities.append([str(prev) + " " + str(word), log_level, file])
                if(prev.tag_ == "VB"):
                    python_activities_strict.append([str(prev) + " " + str(word), log_level, file])
                prev = lemmatizer.lemmatize(str(prev), 'v')
                activity = prev + " " + str(word)

                python_df.loc[index, 'Parsed Activity'] = activity
            prev = word
            
# mostly for testing, since activity extraction takes a long time saving them in a csv and skipping the extraction in
# subsequent executions can be helpful            
with open('python_regular_activities.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(python_activities)   
with open('python_strict_activities.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(python_activities_strict) 
with open('python_loose_activities.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(python_loose_activities) 
    
java_sentiment_scores = []
java_activities_strict = []
java_activities = []
java_loose_activities = []
for index, row in java_df.iterrows():
    file = row[0]
    log_level = row[3]
    stmnt = row[4]
    stmnt = stmnt.lower()
    #stmnt = "".join([part for part in stmnt if part not in punctuation])
    if(stmnt != ""):
        blob = TextBlob(stmnt)
        sentiment = blob.sentiment.polarity
        scores = sid.polarity_scores(stmnt)
        java_sentiment_scores.append([blob, log_level, sentiment])

        stmnt = nlp(stmnt)
        prev = stmnt[0]
        activity = ""
        verb_found = False
        for word in stmnt:
            #or (prev != word and prev.pos_ == "NOUN" and word.pos_ == "NOUN")
            #if(str(word) == "API"):
                #print(word.pos_)
            if(word.pos_ == "VERB"):
                verb_found = True
            if(verb_found and word.pos_ == "NOUN"):
                java_loose_activities.append([stmnt, log_level, file])
            if((prev.pos_ == "VERB" and word.pos_ == "NOUN") ):
                java_activities.append([str(prev) + " " + str(word), log_level, file])
                if(prev.tag_ == "VB"):
                    java_activities_strict.append([str(prev) + " " + str(word), log_level, file])
                prev = lemmatizer.lemmatize(str(prev), 'v')
                activity = prev + " " + str(word)

                java_df.loc[index, 'Parsed Activity'] = activity
            prev = word
            
# mostly for testing, since activity extraction takes a long time saving them in a csv and skipping the extraction in
# subsequent executions can be helpful
with open('java_regular_activities.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(java_activities)   
with open('java_strict_activities.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(java_activities_strict) 
with open('java_loose_activities.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(java_loose_activities)

python_sentiment_df=pd.DataFrame(python_sentiment_scores, columns=["log statement", "log level", "sentiment"])
java_sentiment_df=pd.DataFrame(java_sentiment_scores, columns=["log statement", "log level", "sentiment"])


# Verteilung sentiment scores
plt.hist(python_sentiment_df.iloc[:,2])
plt.ylabel("Number of Log Statements")
plt.xlabel("Sentiment Score")
plt.savefig("figures/Python_sentiment_scores_verteilung.svg")
plt.show()

plt.hist(java_sentiment_df.iloc[:,2])
plt.ylabel("Number of Log Statements")
plt.xlabel("Sentiment Score")
plt.yticks(np.arange(0, 9001, step=500))
plt.savefig("figures/Java_sentiment_scores_verteilung.svg")
plt.show()


# PLOT durchschnittliche score pro log level
python_log_level_scores = python_sentiment_df.groupby('log level')["sentiment"].mean().reset_index()
ll_scores_plot = plt.bar(python_log_level_scores['log level'], python_log_level_scores['sentiment'])
plt.ylim(-1, 1)
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel("Mean sentiment")
plt.savefig("figures/Python_log_level_avg_sentiment.svg")
plt.show()

java_log_level_scores = java_sentiment_df.groupby('log level')["sentiment"].mean().reset_index()
ll_scores_plot = plt.bar(java_log_level_scores['log level'], java_log_level_scores['sentiment'])
plt.ylim(-1, 1)
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel("Mean sentiment")
plt.savefig("figures/Java_log_level_avg_sentiment.svg")
plt.show()

# in case activity extraction was skipped
with open('python_strict_activities.csv', newline='') as csvfile:
    python_activities_strict = list(csv.reader(csvfile))
with open('python_regular_activities.csv', newline='') as csvfile:
    python_activities = list(csv.reader(csvfile))
with open('python_loose_activities.csv', newline='') as csvfile:
    python_loose_activities  = list(csv.reader(csvfile))
print(python_activities_strict)

with open('java_strict_activities.csv', newline='') as csvfile:
    java_activities_strict = list(csv.reader(csvfile))
with open('java_regular_activities.csv', newline='') as csvfile:
    java_activities = list(csv.reader(csvfile))
with open('java_loose_activities.csv', newline='') as csvfile:
    java_loose_activities  = list(csv.reader(csvfile))


# count duplicate activities
python_strict_strings = [python_activities_strict[i][0] for i in range (len(python_activities_strict))]
python_activities_strings = [python_activities[i][0] for i in range (len(python_activities))]
python_loose_strings = [python_loose_activities[i][0] for i in range (len(python_loose_activities))]

java_strict_strings = [java_activities_strict[i][0] for i in range (len(java_activities_strict))]
java_activities_strings = [java_activities[i][0] for i in range (len(java_activities))]
java_loose_strings = [java_loose_activities[i][0] for i in range (len(java_loose_activities))]

python_strict_duplicates = []
python_activities_duplicates = []
python_loose_duplicates = []

java_strict_duplicates = []
java_activities_duplicates = []
java_loose_duplicates = []

for i in range (len(python_strict_strings)):
    if (python_strict_strings[i] in python_strict_strings[:i]) or (python_strict_strings[i] in python_strict_strings[i+1:]): python_strict_duplicates.append(python_strict_strings[i])

for i in range (len(python_activities_strings)):
    if (python_activities_strings[i] in python_activities_strings[:i]) or (python_activities_strings[i] in python_activities_strings[i+1:]): python_activities_duplicates.append(python_activities_strings[i])

for i in range (len(python_loose_strings)):
    if (python_loose_strings[i] in python_loose_strings[:i]) or (python_loose_strings[i] in python_loose_strings[i+1:]): python_loose_duplicates.append(python_loose_strings[i])

for i in range (len(java_strict_strings)):
    if (java_strict_strings[i] in java_strict_strings[:i]) or (java_strict_strings[i] in java_strict_strings[i+1:]): java_strict_duplicates.append(java_strict_strings[i])

for i in range (len(java_activities_strings)):
    if (java_activities_strings[i] in java_activities_strings[:i]) or (java_activities_strings[i] in java_activities_strings[i+1:]): java_activities_duplicates.append(java_activities_strings[i])

for i in range (len(java_loose_strings)):
    if (java_loose_strings[i] in java_loose_strings[:i]) or (java_loose_strings[i] in java_loose_strings[i+1:]): java_loose_duplicates.append(java_loose_strings[i])




python_strict_duplicates = {i:python_strict_duplicates.count(i) for i in python_strict_duplicates}
python_activities_duplicates = {i:python_activities_duplicates.count(i) for i in python_activities_duplicates}
python_loose_duplicates = {i:python_loose_duplicates.count(i) for i in python_loose_duplicates}

java_strict_duplicates = {i:java_strict_duplicates.count(i) for i in java_strict_duplicates}
java_activities_duplicates = {i:java_activities_duplicates.count(i) for i in java_activities_duplicates}
java_loose_duplicates = {i:java_loose_duplicates.count(i) for i in java_loose_duplicates}


python_strict_duplicates_df = pd.DataFrame.from_dict(python_strict_duplicates, orient='index', columns=["duplicates"])
python_strict_duplicates_df["kind of activity"] = "strict"
python_strict_duplicates_df.to_csv("python_duplicate_activities.csv")                                                         # alte Daten überschreiben, Header einfügen

python_activities_duplicates_df = pd.DataFrame.from_dict(python_activities_duplicates, orient='index', columns=["duplicates"])
python_activities_duplicates_df["kind of activity"] = "regular"
python_activities_duplicates_df.to_csv("duplicate_activities.csv", mode="a", header=None)                              # Header nicht duplizieren

python_loose_duplicates_df = pd.DataFrame.from_dict(python_loose_duplicates, orient='index', columns=["duplicates"])
python_loose_duplicates_df["kind of activity"] = "loose"
python_loose_duplicates_df.to_csv("duplicate_activities.csv", mode="a", header=None)


java_strict_duplicates_df = pd.DataFrame.from_dict(java_strict_duplicates, orient='index', columns=["duplicates"])
java_strict_duplicates_df["kind of activity"] = "strict"
java_strict_duplicates_df.to_csv("java_duplicate_activities.csv")                                                         # alte Daten überschreiben, Header einfügen

java_activities_duplicates_df = pd.DataFrame.from_dict(java_activities_duplicates, orient='index', columns=["duplicates"])
java_activities_duplicates_df["kind of activity"] = "regular"
java_activities_duplicates_df.to_csv("java_duplicate_activities.csv", mode="a", header=None)                              # Header nicht duplizieren

java_loose_duplicates_df = pd.DataFrame.from_dict(java_loose_duplicates, orient='index', columns=["duplicates"])
java_loose_duplicates_df["kind of activity"] = "loose"
java_loose_duplicates_df.to_csv("java_duplicate_activities.csv", mode="a", header=None)


# plot how many times activities have appearances to pie chart
python_strict_duplicates_df = python_strict_duplicates_df.groupby("duplicates").agg('count').reset_index()
python_activities_duplicates_df = python_activities_duplicates_df.groupby("duplicates").agg('count').reset_index()
python_loose_duplicates_df = python_loose_duplicates_df.groupby("duplicates").agg('count').reset_index()

java_strict_duplicates_df = java_strict_duplicates_df.groupby("duplicates").agg('count').reset_index()
java_activities_duplicates_df = java_activities_duplicates_df.groupby("duplicates").agg('count').reset_index()
java_loose_duplicates_df = java_loose_duplicates_df.groupby("duplicates").agg('count').reset_index()


#Anzahl Aktivitäten aufsummieren
python_total_numbers_strict_activities = python_strict_duplicates_df["kind of activity"].sum()
python_total_numbers_reg_activities = python_activities_duplicates_df["kind of activity"].sum()
python_total_numbers_loose_activities = python_loose_duplicates_df["kind of activity"].sum()

java_total_numbers_strict_activities = java_strict_duplicates_df["kind of activity"].sum()
java_total_numbers_reg_activities = java_activities_duplicates_df["kind of activity"].sum()
java_total_numbers_loose_activities = java_loose_duplicates_df["kind of activity"].sum()


# calculate percentages of total
python_strict_activities_percentages = (python_strict_duplicates_df["kind of activity"]/python_total_numbers_strict_activities) * 100
python_strict_activities_percentages = python_strict_activities_percentages.values
python_reg_activities_percentages = (python_activities_duplicates_df["kind of activity"]/python_total_numbers_reg_activities) * 100
python_reg_activities_percentages = python_reg_activities_percentages.values
python_loose_activities_percentages = (python_loose_duplicates_df["kind of activity"]/python_total_numbers_loose_activities) * 100
python_loose_activities_percentages = python_loose_activities_percentages.values

java_strict_activities_percentages = (java_strict_duplicates_df["kind of activity"]/java_total_numbers_strict_activities) * 100
java_strict_activities_percentages = java_strict_activities_percentages.values
java_reg_activities_percentages = (java_activities_duplicates_df["kind of activity"]/java_total_numbers_reg_activities) * 100
java_reg_activities_percentages = java_reg_activities_percentages.values
java_loose_activities_percentages = (java_loose_duplicates_df["kind of activity"]/java_total_numbers_loose_activities) * 100
java_loose_activities_percentages = java_loose_activities_percentages.values


# extract labels
python_strict_duplicates_labels = python_strict_duplicates_df["duplicates"].unique()
python_reg_duplicates_labels = python_activities_duplicates_df["duplicates"].unique()
python_loose_duplicates_labels = python_loose_duplicates_df["duplicates"].unique()

java_strict_duplicates_labels = java_strict_duplicates_df["duplicates"].unique()
java_reg_duplicates_labels = java_activities_duplicates_df["duplicates"].unique()
java_loose_duplicates_labels = java_loose_duplicates_df["duplicates"].unique()


pie1, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(11,7))
ax1.pie(python_strict_activities_percentages, autopct=lambda x: format(x, '.2f') if x > 5 else None, shadow=True)
ax1.axis('equal')  
ax2.pie(python_reg_activities_percentages, autopct=lambda x: format(x, '.2f') if x > 5 else None, shadow=True, labeldistance= 1.3)
ax2.axis('equal') 
ax3.pie(python_loose_activities_percentages, autopct=lambda x: format(x, '.2f') if x > 5 else None, shadow=True, labeldistance= 1.3)
ax3.axis('equal')


ax1.set_title('Strict Definition', y=-0)
ax2.set_title('Regular Definition', y=0)
ax3.set_title('Loose Definition', y=0)

strict_legend = ax1.legend(python_strict_duplicates_labels,loc = 'best', title="Number of appearances")
reg_legend = ax2.legend(python_reg_duplicates_labels,loc = 'upper center',ncol=1, title="Number of appearances")
loose_legend = ax3.legend(python_loose_duplicates_labels,loc = 'upper right',ncol=4, title="Number of appearances")
plt.tight_layout()
plt.savefig("figures/Python_amount_of_duplicate_activities.pdf")
plt.show()

pie2, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(12,8))
ax1.pie(java_strict_activities_percentages, autopct=lambda x: format(x, '.2f') if x > 5 else None, shadow=True)   #, startangle=90
ax1.axis('equal')  
ax2.pie(java_reg_activities_percentages, autopct=lambda x: format(x, '.2f') if x > 5 else None, shadow=True)
ax2.axis('equal') 
ax3.pie(java_loose_activities_percentages, autopct=lambda x: format(x, '.2f') if x > 5 else None, shadow=True)
ax3.axis('equal')
plt.tight_layout()


ax1.set_title('Strict Definition', y=-0)
ax2.set_title('Regular Definition', y=-0)
ax3.set_title('Loose Definition', y=-0)

strict_legend = ax1.legend(java_strict_duplicates_labels,loc = 'best',ncol=2, title="Number of appearances")
reg_legend = ax2.legend(java_reg_duplicates_labels,loc = 'upper center',ncol=3, title="Number of appearances")
loose_legend = ax3.legend(java_loose_duplicates_labels,loc = 'upper right',ncol=4, title="Number of appearances")
plt.savefig("figures/Java_amount_of_duplicate_activities.pdf")
plt.show()



# count duplicates per file
python_strict_act_per_file = [[python_activities_strict[i][0], python_activities_strict[i][2]] for i in range (len(python_activities_strict))]
python_act_per_file = [[python_activities[i][0], python_activities[i][2]] for i in range (len(python_activities))]
python_loose_act_per_file = [[python_loose_activities[i][0], python_loose_activities[i][2]] for i in range (len(python_loose_activities))]


java_strict_act_per_file = [[java_activities_strict[i][0], java_activities_strict[i][2]] for i in range (len(java_activities_strict))]
java_act_per_file = [[java_activities[i][0], java_activities[i][2]] for i in range (len(java_activities))]
java_loose_act_per_file = [[java_loose_activities[i][0], java_loose_activities[i][2]] for i in range (len(java_loose_activities))]

python_ls_per_file_strict = defaultdict(list)
python_ls_per_file = defaultdict(list)
python_ls_per_file_loose = defaultdict(list)


java_ls_per_file_strict = defaultdict(list)
java_ls_per_file = defaultdict(list)
java_ls_per_file_loose = defaultdict(list)

for activity, filename in python_strict_act_per_file:
    python_ls_per_file_strict[filename].append(activity)

for activity, filename in python_act_per_file:
    python_ls_per_file[filename].append(activity)

for activity, filename in python_loose_act_per_file:
    python_ls_per_file_loose[filename].append(activity)
    
    
for activity, filename in java_strict_act_per_file:
    java_ls_per_file_strict[filename].append(activity)

for activity, filename in java_act_per_file:
    java_ls_per_file[filename].append(activity)

for activity, filename in java_loose_act_per_file:
    java_ls_per_file_loose[filename].append(activity)

python_duplicates_per_file_strict = []
for key, activities in python_ls_per_file_strict.items():
    if(pd.Series(activities)[pd.Series(activities).duplicated()].values.size > 0):
        python_duplicates_per_file_strict.append([key, pd.Series(activities)[pd.Series(activities).duplicated()].values])

python_duplicates_per_file = []
for key, activities in python_ls_per_file.items():
    if(pd.Series(activities)[pd.Series(activities).duplicated()].values.size > 0):
        python_duplicates_per_file.append([key, pd.Series(activities)[pd.Series(activities).duplicated()].values])

python_duplicates_per_file_loose = []
for key, activities in python_ls_per_file_loose.items():
    if(pd.Series(activities)[pd.Series(activities).duplicated()].values.size > 0):
        python_duplicates_per_file_loose.append([key, pd.Series(activities)[pd.Series(activities).duplicated()].values])


java_duplicates_per_file_strict = []
for key, activities in java_ls_per_file_strict.items():
    if(pd.Series(activities)[pd.Series(activities).duplicated()].values.size > 0):
        java_duplicates_per_file_strict.append([key, pd.Series(activities)[pd.Series(activities).duplicated()].values])

java_duplicates_per_file = []
for key, activities in java_ls_per_file.items():
    if(pd.Series(activities)[pd.Series(activities).duplicated()].values.size > 0):
        java_duplicates_per_file.append([key, pd.Series(activities)[pd.Series(activities).duplicated()].values])

java_duplicates_per_file_loose = []
for key, activities in java_ls_per_file_loose.items():
    if(pd.Series(activities)[pd.Series(activities).duplicated()].values.size > 0):
        java_duplicates_per_file_loose.append([key, pd.Series(activities)[pd.Series(activities).duplicated()].values])



# Duplikate pro file zählen
python_duplicates_per_file_strict_df = pd.DataFrame(python_duplicates_per_file_strict, columns=["File", "Duplicate Activities"])
python_duplicates_per_file_df = pd.DataFrame(python_duplicates_per_file, columns=["File", "Duplicate Activities"])
python_duplicates_per_file_loose_df = pd.DataFrame(python_duplicates_per_file_loose, columns=["File", "Duplicate Activities"])


java_duplicates_per_file_strict_df = pd.DataFrame(java_duplicates_per_file_strict, columns=["File", "Duplicate Activities"])
java_duplicates_per_file_df = pd.DataFrame(java_duplicates_per_file, columns=["File", "Duplicate Activities"])
java_duplicates_per_file_loose_df = pd.DataFrame(java_duplicates_per_file_loose, columns=["File", "Duplicate Activities"])


python_duplicates_per_file_strict_df["Duplicate Activities"]=[len(dupl_activity_array) for dupl_activity_array in python_duplicates_per_file_strict_df["Duplicate Activities"]]
python_duplicates_per_file_df["Duplicate Activities"]=[len(dupl_activity_array) for dupl_activity_array in python_duplicates_per_file_df["Duplicate Activities"]]
python_duplicates_per_file_loose_df["Duplicate Activities"]=[len(dupl_activity_array) for dupl_activity_array in python_duplicates_per_file_loose_df["Duplicate Activities"]]

java_duplicates_per_file_strict_df["Duplicate Activities"]=[len(dupl_activity_array) for dupl_activity_array in java_duplicates_per_file_strict_df["Duplicate Activities"]]
java_duplicates_per_file_df["Duplicate Activities"]=[len(dupl_activity_array) for dupl_activity_array in java_duplicates_per_file_df["Duplicate Activities"]]
java_duplicates_per_file_loose_df["Duplicate Activities"]=[len(dupl_activity_array) for dupl_activity_array in java_duplicates_per_file_loose_df["Duplicate Activities"]]

# count how many files have which amount of duplicates
python_duplicates_per_file_strict_df = python_duplicates_per_file_strict_df.groupby("Duplicate Activities").size().reset_index()
python_duplicates_per_file_df = python_duplicates_per_file_df.groupby("Duplicate Activities").size().reset_index()
python_duplicates_per_file_loose_df = python_duplicates_per_file_loose_df.groupby("Duplicate Activities").size().reset_index()


java_duplicates_per_file_strict_df = java_duplicates_per_file_strict_df.groupby("Duplicate Activities").size().reset_index()
java_duplicates_per_file_df = java_duplicates_per_file_df.groupby("Duplicate Activities").size().reset_index()
java_duplicates_per_file_loose_df = java_duplicates_per_file_loose_df.groupby("Duplicate Activities").size().reset_index()


# plotten wie viele files wie viele Duplikate haben
max_loose_act_value = python_duplicates_per_file_loose_df["Duplicate Activities"].max()
max_reg_act_value = python_duplicates_per_file_df["Duplicate Activities"].max()
max_strict_act_value = python_duplicates_per_file_strict_df["Duplicate Activities"].max()
max_tick = max(max_loose_act_value, max_reg_act_value, max_strict_act_value)
python_ticks = np.arange(0, max_tick+1, 2)
plt.bar(python_duplicates_per_file_loose_df["Duplicate Activities"], python_duplicates_per_file_loose_df[0], alpha=0.5, color="b")
plt.bar(python_duplicates_per_file_df["Duplicate Activities"], python_duplicates_per_file_df[0], alpha=0.5, color="tab:green")
plt.bar(python_duplicates_per_file_strict_df["Duplicate Activities"], python_duplicates_per_file_strict_df[0], color="c")

plt.xticks(python_ticks, rotation='vertical')
plt.xlabel("Number of duplicate activities")
plt.ylabel("Number of files with x amount of duplicate activities")
plt.legend(['Loose', 'Regular', 'Strict'])
plt.savefig("figures/Python_file_with_x_amount_of_duplicate_activities.svg")
plt.show()

max_loose_act_value = java_duplicates_per_file_loose_df["Duplicate Activities"].max()
max_reg_act_value = java_duplicates_per_file_df["Duplicate Activities"].max()
max_strict_act_value = java_duplicates_per_file_strict_df["Duplicate Activities"].max()
max_tick = max(max_loose_act_value, max_reg_act_value, max_strict_act_value)
java_ticks = np.arange(0, max_tick+1, 2)
f, ax = plt.subplots(figsize=(15,6))
plt.bar(java_duplicates_per_file_loose_df["Duplicate Activities"], java_duplicates_per_file_loose_df[0], alpha=0.5, color="b")
plt.bar(java_duplicates_per_file_df["Duplicate Activities"], java_duplicates_per_file_df[0], alpha=0.5, color="tab:green")
plt.bar(java_duplicates_per_file_strict_df["Duplicate Activities"], java_duplicates_per_file_strict_df[0], color="c")

plt.xticks(java_ticks, rotation='vertical')
plt.xlabel("Number of duplicate activities")
plt.ylabel("Number of files with x amount of duplicate activities")
plt.legend(['Loose', 'Regular', 'Strict'])
plt.savefig("figures/java_file_with_x_amount_of_duplicate_activities.svg")
plt.show()



# grouping in which files are majority of LS in if-statements, method etc.
# for every file: where do LS appear the most
# diagram: bar plot with every type of node counting how many files had them at number 1
most_common_node_per_python_file = python_df.pivot_table(index='Case ID', values='Resource', aggfunc=pd.Series.mode)
most_common_node_per_python_file = most_common_node_per_python_file.applymap(lambda x: x if isinstance(x, str) else x[0])


# count amount of times method, if...
majority_of_ls_in_python = most_common_node_per_python_file['Resource'].value_counts(sort=False)

majority_of_ls_in_python.plot.bar(xlabel="Nodes", ylabel="# of files", rot=0)
plt.savefig("figures/Python_Verteilung_LS_over_nodes.svg")
plt.show()


most_common_node_per_java_file = java_df.pivot_table(index='Case ID', values='Resource', aggfunc=pd.Series.mode)
most_common_node_per_java_file = most_common_node_per_java_file.applymap(lambda x: x if isinstance(x, str) else x[0])

# count amount of times method, if...
majority_of_ls_in_java = most_common_node_per_java_file['Resource'].value_counts(sort=False)

majority_of_ls_in_java.plot.bar(xlabel="Nodes", ylabel="# of files", rot=0)
plt.savefig("figures/Java_Verteilung_LS_over_nodes.svg")
plt.show()

