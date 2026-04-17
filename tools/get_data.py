from sqlalchemy import create_engine
from dotenv import load_dotenv
from os import getenv
from pathlib import Path
import pandas as pd


def main():
    load_dotenv()
    engine = create_engine(getenv("DATABASE_URL"))

    with open("tools/query.sql", "r") as f:
        query = f.read()

    path = Path("data/")
    if path.exists() is False:
        path.mkdir()

    df = pd.read_sql_query(query, engine)
    df.to_csv("data/training_data.csv", index=False)


if __name__ == "__main__":
    main()
