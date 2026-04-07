import pandas as pd

def load_nav_data(file_path: str) -> pd.DataFrame:
    """
    Load NAV data from CSV file

    Parameters:
    - file_path: path to csv file

    Returns:
    - DataFrame with datetime index and nav column
    """

    df = pd.read_csv(file_path)

    # convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # sort by date
    df = df.sort_values('date')

    # set index
    df.set_index('date', inplace=True)

    return df