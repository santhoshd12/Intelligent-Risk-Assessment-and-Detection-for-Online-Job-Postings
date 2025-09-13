

# %%
import re
import string
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, confusion_matrix
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# %%
df = pd.read_excel('FakeJobPostings.xlsx')
df.head()

# %%
df.shape

# %% [markdown]
# ## Managing the features for Visualization

# %%
# checking for null values

df.isnull().sum()

# %%
# deleting the columns which have no importance or have many null values, more will deleted later after visualization

columns = ['job_id','salary_range','telecommuting','has_company_logo','has_questions','employment_type']

for i in columns:
    del df[i]

df.head()

# %%
df.shape

# %%
df.fillna('',inplace = True)
# It is going to replace null values with spaces becuase some columns have few null values or it is an important column for the model

# %%
df.isnull().sum()

# %% [markdown]
# ## Visualization of the Data

# %%
# Visualizing the total number of fraudulent cases

plt.figure(figsize = (10,5))
sns.countplot(y = 'fraudulent', data = df)
plt.show()

# %%
# Total number of fraudulent and non-fraudulent cases

df.groupby('fraudulent')['fraudulent'].count()

# %%
# It counts the total number of unique required experience except the empty required experience

exp = dict(df.required_experience.value_counts())
del exp['']
exp

# %%
plt.figure(figsize = (10,5))
plt.bar(exp.keys(),exp.values())
plt.title("Required Experience wise Job Posting", size = 20)
plt.xlabel('Required Experience')
plt.ylabel('No. of Jobs')
plt.xticks(rotation = 30)
plt.show()

# %%
#It is creating a new column in the dataset which will extract the country from the location and put it in the new column named country

def split(location):
    l = location.split(',')
    return l[0]
df['country'] = df.location.apply(split)
df.head()

# %%
# It counts the total number of countries except the empty countries, there are many countries in the column which has low job posting so it will be showing only top 15 countries (if empty country is also included in the top 15 then it will be removed)

countr = dict(df.country.value_counts()[:15])
del countr['']
countr

# %%
plt.figure(figsize = (9,6))
plt.bar(countr.keys(),countr.values())
plt.title("Country wise Job Posting", size = 20)
plt.xlabel('Countries')
plt.ylabel('No. of Jobs')
plt.show()

# %%
# It counts the total number of required education except the empty required education, there are many entries for required education in the column which has low job posting so it will be showing only top 9 countries (if empty required education is also included in the top 9 then it will be removed)

edu = dict(df.required_education.value_counts()[:9])
del edu['']
edu

# %%
plt.figure(figsize = (15,6))
plt.bar(edu.keys(),edu.values())
plt.title("Education wise Job Posting", size = 20)
plt.xlabel('Required Education')
plt.ylabel('No. of Jobs')
plt.xticks(rotation = 30)
plt.show()

# %%
# It will count the top 10 titles of Job which are not fraudulent

print(df[df.fraudulent==0].title.value_counts()[:10])

# %%
# It will count the top 10 titles of Job which are fraudulent

print(df[df.fraudulent==1].title.value_counts()[:10])

# %% [markdown]
# ## Combining the features

# %%
# combing the features
df['text'] = df['title'] +' '+df['company_profile']+' '+df['description']+' '+df['requirements']+' '+df['benefits']

#deleting the remaining features
columns_to_delete = ['title', 'location', 'department', 'company_profile', 'description',
                     'requirements', 'benefits', 'required_experience', 'required_education',
                     'industry', 'function','country']

for j in columns_to_delete:
    del df[j]

df.head()

# %%
FraudJobs_text = df[df.fraudulent==1].text
RealJobss_text = df[df.fraudulent==0].text

# %%
#Creating WordCloud for Fraudulent Jobs

STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3, max_words = 3000, width = 1600, height = 800, stopwords = STOPWORDS).generate(str(" ".join(FraudJobs_text)))
plt.imshow(wc,interpolation = 'bilinear')

# %%
#Creating WordCloud for Non-Fraudulent Jobs

STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3, max_words = 3000, width = 1600, height = 800, stopwords = STOPWORDS).generate(str(" ".join(RealJobss_text)))
plt.imshow(wc,interpolation = 'bilinear')

# %% [markdown]
# ## NLP Pipelining



# %%
# Cleaning and preprocessing

# List of punctuation marks
punctuations = string.punctuation

# List of stop words
nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = English()

# Tokenizing function
def spacy_tokenizer(sentence):
    # Process the sentence using spaCy
    doc = nlp(sentence)
    
    # Lemmatize each token and convert to lowercase
    lemmatized_tokens = [token.lemma_.lower() for token in doc]
    
    return lemmatized_tokens

# %%
df['text'] = df['text'].apply(spacy_tokenizer)

# %%
df.to_csv("cleaned_jobs.csv", index=False)

print("✅ Cleaned dataset saved as cleaned_jobs.csv")

# %%
df1 = pd.read_csv("cleaned_jobs.csv")

# %%
import ast
df1['text'] = df1['text'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# %%
df1['text'].values[2]

# %%
df1.head()

# %%


# %%
df1['text'] = df1['text'].apply(lambda x: " ".join(x)) # Converting tokens(list) into string b/c fit_transform() only takes string

# %%
df1.head()

# %%
# It will count the frequency of top 100 words and put it into the new data frame (main_df), It is basically converting string text to vector text

cv = TfidfVectorizer(max_features = 100, stop_words='english')
x = cv.fit_transform(df1['text'])
df2 = pd.DataFrame(x.toarray(), columns = cv.get_feature_names_out())
df1.drop(["text"], axis = 1, inplace = True)
main_df = pd.concat([df2,df1], axis = 1)

main_df.head()

# %% [markdown]
# ## Splitting the dataset for training and prediction

# %%
X = main_df.drop(columns=['fraudulent'])
y = main_df['fraudulent']

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# The train_test_split function shuffles the data by default. Setting random_state ensures reproducibility.

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# %% [markdown]
# ## Using Random Forest Classifier to train and predict

# %% [markdown]
# - `n_jobs`: Number of cores
# - `oob_score = True`: It will include the oob (out of bag, these datasets are not included in the training because theseare left out to give to the base learners which are desicion trees.) - dataset for the training too.
# - `n_estimators`: Total number of desicion trees
# - `criterion`:  It refers to the function used to measure the quality of a split during the tree-building process. The criterion is a crucial aspect of constructing decision trees as it guides the algorithm in making decisions about how to split the data at each node to create branches.

# %%
# Training the model

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs = 4, oob_score = True, n_estimators = 100, criterion = "entropy")
model = rfc.fit(X_train, y_train)

# %%
# Calculate the training accuracy using the 'score' method
training_accuracy = rfc.score(X_train, y_train)

# Access the OOB score
oob_score = rfc.oob_score_

# Print the training accuracy
print("Training Accuracy:", training_accuracy*100)

# Print the OOB score/accuarcy
print("OOB Score:", oob_score*100)

# %%
pred = rfc.predict(X_test)
score = accuracy_score(y_test, pred)
print("Accuracy after prediction on Test datasets: ", score*100)

# %%
# Generate the classification report
class_report = classification_report(y_test, pred)

# Print the classification report
print("Classification Report:")
print(class_report)

# %%
# Create confusion matrix
test_confusion_matrix = confusion_matrix(y_test, pred)

# Plot confusion matrix
sns.heatmap(test_confusion_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [markdown]
# ## Saving the models

# %%
import joblib

# Save the trained model
joblib.dump(rfc,'RFCmodel')

# Save the TF-IDF vectorizer
joblib.dump(cv, 'TFIDF_vectorizer.joblib')

# %%
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1. Make predictions on the test set
y_pred = model.predict(X_test)

# 2. Prepare results dictionary
results = {
    # Full dataset info
    "total_samples": int(len(df)),           # total records in the dataset
    "real_jobs": int(np.sum(df['fraudulent'] == 0)),  # total real jobs
    "fake_jobs": int(np.sum(df['fraudulent'] == 1)),  # total fake jobs

    # Model performance (based on test set)
    "model_accuracy": float(accuracy_score(y_test, y_pred)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

# 3. Save to JSON
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Results saved to results.json")
print(json.dumps(results, indent=4))



