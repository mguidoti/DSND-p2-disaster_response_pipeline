import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine

# Import natural language library and specific packages
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Import stuff from sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Import Grid Search
from sklearn.model_selection import GridSearchCV

# Import estimators
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

nltk.download(["punkt", "wordnet"])


def load_data(database_filepath):
    """[summary]

    Args:
        database_filepath ([type]): [description]

    Returns:
        [type]: [description]
    """
    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql('SELECT * FROM Message', engine)

    df2 = df.head(100)

    X = df2.message.values
    y = df2.drop(columns=["id", "message", "original", "genre"])
    category_names = y.columns[4:]

    return X, y, category_names


def tokenize(text):
    """[summary]

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
    'vect__max_df': [0.5, 0.7],
    'tfidf__use_idf': [True, False],
    'clf__estimator__weights': ["uniform", "distance"],
    }

    grid_search = GridSearchCV(pipeline, param_grid=parameters)

    return grid_search


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        if i > 0:
            print('column {}, index {}: '.format(col, i))
            print(classification_report(y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()