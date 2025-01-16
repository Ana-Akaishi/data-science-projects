# Import packages
import pandas as pd
from sqlalchemy import create_engine

# set postgres path
db_url = 'postgresql+psycopg2://admin:admin@localhost:5432/postgres'

# create the engine to connect with postgres based on the url I defined
engine = create_engine(db_url)

# function to make que conversion
def load_csv2db(csv_file, table_name):
    df = pd.read_csv(csv_file)
    df.to_sql(table_name, engine,if__exists='replace', index=False)
    print(f'Table {table_name} succesfully loaded')

# csv files paths
csv_files = {
    'train': 'Telecom_Churn/0_sql_container/datasets/train.csv',
    'validation': 'Telecom_Churn/0_sql_container/datasets/validation.csv',
    'test': 'Telecom_Churn/0_sql_container/datasets/test.csv'
}

# loop
for table_name in csv_files.items():
    load_csv2db(file_path,table_name)