
# Name of database table containing cleaned and labeled messages

DB_TABLE_MESSAGES = 'Messages'

# Deliminator used in 'categories' column of categories file
DELIM_CATEGORIES = ';'

# Name of categories column
COL_ID = 'id'
COL_MESSAGE = 'message'
COL_ORIGINAL_MESSAGE = 'original'
COL_GENRE = 'genre'
COL_CATEGORIES = 'categories'

COL_CATEGORY_RELATED = 'related'

COL_SET_NO_CATEGORY = {COL_ID, COL_MESSAGE, COL_ORIGINAL_MESSAGE, COL_GENRE}


def print_elapsed_time(start_time, end_time, prompt="Elapsed time"):
    """
    Print the elapsed time

    :param start_time: Time when the frame started (in seconds, time.time())
    :param end_time: Time when the frame ended (in seconds, time.time())
    :param prompt: Prompt to be printed before timing information
    :return: None
    """
    elapsed_seconds = end_time - start_time

    elapsed_hours = int(elapsed_seconds // (60**2))
    elapsed_seconds -= elapsed_hours * (60**2)

    elapsed_minutes = int(elapsed_seconds // 60)
    elapsed_seconds -= elapsed_minutes * 60

    elapsed_seconds = int(elapsed_seconds)

    print('{}: {}:{:02}:{:02}\n'.format(prompt, elapsed_hours, elapsed_minutes, elapsed_seconds))
    return end_time - start_time

