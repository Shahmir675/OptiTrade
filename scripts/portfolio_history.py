import logging
import os
from datetime import date
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv
from psycopg2 import OperationalError

logging.basicConfig(level=logging.INFO)

load_dotenv(dotenv_path="/home/shahmir/Backend/OptiTrade/.env.test")

DATABASE_URL = os.environ.get("DATABASE_URL")
parsed_url = urlparse(DATABASE_URL)

db_config = {
    "dbname": parsed_url.path[1:],
    "user": parsed_url.username,
    "password": parsed_url.password,
    "host": parsed_url.hostname,
    "port": parsed_url.port,
}


def connect_database(db_config):
    try:
        conn = psycopg2.connect(**db_config)
        logging.info("Database Connected Successfully.")
        return conn

    except OperationalError as e:
        logging.error("Error Connecting to Database: %s", e)
        return None


def check_last_insert(conn):
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT DATE(MAX(snapshot_date)) FROM portfolio_history
            """
            )
            last_insert = cur.fetchone()[0]
            today_date = date.today()

            if today_date == last_insert:
                return 1

            else:
                return 0

        except OperationalError as e:
            logging.error("Error occurred: %s", e)
            return None


def get_ids(conn):
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT * FROM portfolio
            """
            )
            output = cur.fetchall()
            user_ids = [row[0] for row in output]
            unique_ids = list(set(user_ids))
            logging.info("Fetched IDs successfully.")
            return unique_ids

        except OperationalError as e:
            logging.error("Error occurred: %s", e)
            return None


def update_portfolio_history(conn, unique_ids):
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT * FROM portfolio
                WHERE user_id = ANY(%s)
            """,
                (unique_ids,),
            )
            records = cur.fetchall()
            snapshot_date = "NOW()"
            values = [(*record, snapshot_date) for record in records]
            cur.executemany(
                """
                INSERT INTO portfolio_history
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                values,
            )
            conn.commit()
            logging.info("Portfolio history updated successfully.")

        except OperationalError as e:
            conn.rollback()
            logging.error("Error occurred: %s", e)


if __name__ == "__main__":
    conn = connect_database(db_config)

    if check_last_insert(conn):
        logging.info("Portfolio history has already been updated for today.")
        logging.info("Script completed.")
        exit()

    logging.info("Updating portfolio history")
    unique_ids = get_ids(conn)
    update_portfolio_history(conn, unique_ids)
    conn.close()
    logging.info("Script completed.")
