


import tkinter as tk
import webbrowser
from PIL import Image, ImageDraw, ImageFont
import os


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from time import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

df_scraped = pd.read_csv('./data/label_tweets.csv')
df_public = pd.read_csv('./data/plabeled.csv')
df_scraped.drop_duplicates(inplace = True)
df_scraped.drop('id', axis = 'columns', inplace = True)

df_public.drop_duplicates(inplace = True)
df_scraped.head(2)

df_public.head(2)

df = pd.concat([df_scraped, df_public])
df.shape

plt.figure(figsize = (7,7))
sorted_counts = df['label'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},
       autopct='%1.1f%%', pctdistance = 0.7, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,
        colors = sns.color_palette("Paired")[7:])
plt.text(x = -0.35, y = 0, s = 'Total Tweets: {}'.format(df.shape[0]))
plt.title('Distribution of Tweets in the Dataset', fontsize = 16);


df['label'] = df.label.map({'Offensive': 1, 'Non-offensive': 0})
X_train, X_test, y_train, y_test = train_test_split(df['full_text'], 
                                                    df['label'], 

                                                    random_state=42)






# Instantiate the CountVectorizer method
count_vector = CountVectorizer(stop_words = 'english', lowercase = True)

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


def pipeline(learner_list, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    # Get length of Training Data:
    size = len(y_train)
    
    results = {}
    final_results = []
    
    for learner in learner_list:
        
        # Store the learner name:
        results['Algorithm'] = learner.__class__.__name__
        
        # Fit the learner:
        start = time() # Get start time
        print("Training {}".format(learner.__class__.__name__))
        learner = learner.fit(X_train, y_train)
        
        end = time() # Get end time

        # Store the training time
        results['Training Time'] = end - start
        
        start = time() # Get start time
        predictions_test = learner.predict(X_test)
        predictions_train = learner.predict(X_train)
        end = time() # Get end time


        
        final_results.append(results.copy())
    # Return a dataframe of the results
    return learner
# make a list of models
models = [DecisionTreeClassifier()]

re = pipeline(models, training_data, y_train, testing_data, y_test)




def show_entry_fields():
    print("First Name: %s" % (e1.get()))
    # name of the file to save
    
   

def show_entry_fields1():
     print("First Name: %s" % (e1.get()))
     text=e1.get()
     testing_data = count_vector.transform([text])
     predictions_test = re.predict(testing_data)
     print(predictions_test[0])
     tk.Label(master,  text="").grid(row=500,column=400)
     if predictions_test[0]==1:
         tk.Label(master,  text="").grid(row=500,column=400)
         tk.Label(master,  text="   cyberbullying   ").grid(row=500,column=400)
     else:
         tk.Label(master,  text="").grid(row=500,column=400)
         tk.Label(master,  text="        Non-cyberbully        ").grid(row=500,column=400)
    # name of the file to save
    
master = tk.Tk()
master.geometry("700x400")
tk.Label(master,  text="").grid(row=50)


e1 = tk.Entry(master)


e1.grid(column=400, row=300, ipady=50, ipadx=200)
 #must be -column, -columnspan, -in, -ipadx, -ipady, -padx, -pady, -row, -rowspan, or -sticky


tk.Button(master,text='Submit', command=show_entry_fields1).grid(row=301, 
                                                       column=300, 
                                                       sticky=tk.W, 
                                                       ipadx=30)

tk.mainloop()
