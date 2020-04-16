import sys
import time

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import util
from fs_ds_util import sysout as out


def load_data(messages_filepath, categories_filepath):
    """
    Load data from messages and categories file

    :param messages_filepath: csv-file containing messages
    :param categories_filepath: csv-file containing categories
    :return: pandas.DataFrame containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    out.info("{} messages loaded".format(messages.shape[0]))

    categories = pd.read_csv(categories_filepath)
    out.info("{} categories loaded".format(categories.shape[0]))

    df = pd.merge(messages, categories, how='inner', on=[util.COL_ID])
    return df


def print_duplicates(df, column):
    """
    Debugging method used to print duplicate entries in the given column

    :param df: pandas.DataFrame
    :param column: Name of column to be analysed
    :return: None
    """
    # info('Showing duplicate {}\'s'.format(column))
    unique_dates, count = np.unique(df[column].tolist(), return_counts=True)

    found_none_unique = False
    for i in range(len(unique_dates)):
        if count[i] >= 2:
            print("{:2}x {:6}".format(count[i], unique_dates[i]))
            found_none_unique = True

    if not found_none_unique :
        print("No duplicates in column {}".format(column))


def transform_categories_to_columns(df):
    """
    Transform values in column 'categories' into multiple columns.

    Each item in the column 'categories' contains a separated list of
    message classifications.

    :param df: pandas.DataFrame containing messages and categories
    :return: pandas.DataFrame containing messages with column of each category
    """
    # Split values in COL_CATEGORIES into separate columns
    df_categories = df[util.COL_CATEGORIES].str.split(util.DELIM_CATEGORIES, expand=True)

    # Verify column-entry and convert "-[0|1]" part into an integer.
    labels = []
    messy_columns = []

    out.info("Separating labels and values...")
    for i in df_categories.columns:
        # check if each column contains only one label (<label>-<set|unset>)
        col_i_series = df_categories[i].str.replace('-\\d*', '')
        df_categories[i] = df_categories[i].str.replace('.*-', '').astype(int)

        eq = col_i_series.value_counts() == df_categories.shape[0]
        if eq.all():
            label = col_i_series[0]
            labels.append(label)
            out.info("{:2} {}".format(len(labels), label))
        else:
            messy_columns.append(i)
            out.warn("Column {} has mixed labels".format(i))

    # Check if number of labels matches number of columns
    out.info("Found {} labels".format(len(labels)))
    if len(labels) == len(df_categories.columns):
        out.info("All columns have a unique label")
    else:
        message = "Some columns are ambiguous! Take a closer look at columns {}".format(messy_columns)
        out.error(message)
        raise ValueError(message)

    # Check if some labels are duplicated
    len_of_uniques = len(set(labels))
    if len_of_uniques != len(labels):
        message = "Found {} duplicate labels".format(len(labels)-len_of_uniques)
        out.error(message)
        raise ValueError(message)

    df_categories.columns = labels

    df = pd.concat([df, df_categories], axis=1)
    return df


def remove_duplicates(df):
    """
    Remove duplicate entries from given DataFrame

    :param df: pandas.DataFrame containing messages and categories
    :return: pandas.DataFrame containing messages and categories without duplicates
    """
    # print_duplicates(df, 'id')
    # print_duplicates(df, 'message')
    df.drop_duplicates(subset=[util.COL_ID, util.COL_MESSAGE], inplace=True)

    return df


def clean_data(df):
    """
    Clean DataFrame of duplicates, morph categories into separate columns, ...

    :param df: pandas.DataFrame to be cleaned
    :return: cleaned pandas.DataFrame containing messages and labels
    """
    initial_record_count = df.shape[0]

    df = remove_duplicates(df)

    df = transform_categories_to_columns(df)

    columns_to_drop = [util.COL_CATEGORIES]
    out.info("Dropping columns: {}".format(columns_to_drop))
    df = df.drop(columns_to_drop, axis=1)

    # cleaning messed values
    df = df[df[util.COL_MESSAGE] != '#NAME?']
    df.loc[(df[util.COL_CATEGORY_RELATED] != 0) & (df[util.COL_CATEGORY_RELATED] != 1), util.COL_CATEGORY_RELATED] = 1

    final_record_count = df.shape[0]
    out.info("Removed {} ({} -> {}) records...".format((initial_record_count-final_record_count),
                                                       initial_record_count, final_record_count))
    return df


def save_data(df, database_filename):
    """
    Store pandas.DataFrame into sqlite database

    :param df: pandas.DataFrame to store
    :param database_filename: file name of sqlite database
    :return: None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(util.DB_TABLE_MESSAGES, engine, index=False, if_exists='replace')
    out.info("Table name is: {}".format(util.DB_TABLE_MESSAGES))


def main():
    if len(sys.argv) == 4:
        start_time_program = time.time()

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print()

        start_time = time.time()
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        util.print_elapsed_time(start_time, time.time())

        start_time = time.time()
        print('Cleaning data...')
        df = clean_data(df)
        util.print_elapsed_time(start_time, time.time())

        start_time = time.time()
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        util.print_elapsed_time(start_time, time.time())

        print()
        print('Cleaned data saved to database!')
        util.print_elapsed_time(start_time_program, time.time(), prompt="Total execution time")
    
    else:
        print("""
Invalid number of arguments. Please respect the usage message.
        
Usage: python {} MESSAGES CATEGORIES DATABASE

    MESSAGES:   Message dataset (.csv-file, input)
    CATEGORIES: Categories dataset (.csv-file, input)
    DATABASE:   Database to store the cleaned messages (.sqlite-file, output)
    
Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.sqlite
""".format(sys.argv[0]))


if __name__ == '__main__':
    main()
