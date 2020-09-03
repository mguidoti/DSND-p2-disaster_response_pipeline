import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data from CSV files

    Args:
        messages_filepath (string): Filepath to the messages file
        categories_filepath (string): Filepath to the catagories file

    Returns:
        pandas.DataFrame: DataFrame with both files merged
    """
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)

    # Merge the two previously loaded datasets
    df = messages.merge(categories, how='outer', on=['id'])

    return df


def clean_data(df):
    """Clean the provided data by converting 'catagories' column into individual
    columns, correctly named, with binary values correctly converted to number

    Args:
        df (pandas.DataFrame): DataFrame with the data

    Returns:
        pandas.DataFrame: DataFrame with the converted data
    """
    # Create a temp df to manipulate categories data
    categories = df["categories"].str.split(";", expand=True)

    # Create column names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Isolate numbers in each cell and convert all columns to numeric
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the old 'categories' column on the original df
    df.drop(columns="categories", inplace=True)

    # Concatenate the original and the temp dfs
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Save the data in a SQLite database

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned data
        database_filename (string): Filepath to save the transformed data in a
            sqlite database
    """
    # Create db engine
    engine = create_engine('sqlite:///' + database_filename)

    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()