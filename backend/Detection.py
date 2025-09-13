import pandas as pd
import spacy
import joblib

def preprocess_data(file_path):
    # Read data from Excel file
    df = pd.read_excel(file_path)

    # Delete unnecessary columns
    columns = ['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions', 'employment_type']
    for col in columns:
        del df[col]

    df.fillna('', inplace=True)

    # Extract country from location
    def split(location):
        l = location.split(',')
        return l[0]
    df['country'] = df.location.apply(split)

    # Combine text features into a single column
    df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']

    # Delete remaining features
    columns_to_delete = ['title', 'location', 'department', 'company_profile', 'description',
                         'requirements', 'benefits', 'required_experience', 'required_education',
                         'industry', 'function', 'country']
    for col in columns_to_delete:
        del df[col]

    # Tokenizing function
    nlp = spacy.load("en_core_web_sm")
    def spacy_tokenizer(sentence):
        doc = nlp(sentence)
        lemmatized_tokens = [token.lemma_.lower() for token in doc]
        return lemmatized_tokens

    df['text'] = df['text'].apply(spacy_tokenizer)
    df['text'] = df['text'].apply(lambda x: " ".join(x))

    # Use the previously fitted TfidfVectorizer to transform the text data
    cv = joblib.load(r'D:\fakejob\backend\TFIDF_vectorizer.joblib')
    x = cv.transform(df['text'])
    df1 = pd.DataFrame(x.toarray(), columns=cv.get_feature_names_out())
    df.drop(["text"], axis=1, inplace=True)
    main_df = pd.concat([df1, df], axis=1)

    return main_df

def predict_fraudulent_jobs(file_path):
    # Load the trained Random Forest Classifier
    rfc = joblib.load(r'D:\fakejob\backend\RFCmodel')

    # Preprocess the data
    preprocessed_data = preprocess_data(file_path)

    # Extract features
    X = preprocessed_data

    # Make predictions
    predictions = rfc.predict(X)

    return predictions

# Example usage
file_path = input("Enter file path: ")
predictions = predict_fraudulent_jobs(file_path)
print(predictions)
