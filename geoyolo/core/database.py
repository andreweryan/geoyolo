import json
from typing import Any, Dict, Union
from urllib.parse import quote

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine


def read_creds(credentials: Union[str, Dict[Any, Any]]) -> Dict[Any, Any]:
    """Read database credentials json

    Args:
        credentials (str): Path to json continiang database connection information

    Return:
        credentials (dcit) : database credentials dictionary
    """
    if isinstance(credentials, str):
        with open(credentials, "r") as f:
            creds = json.load(f)
        return creds
    elif isinstance(credentials, dict):
        return credentials
    else:
        raise ValueError("Database credentials not supplied!")


def connect(creds: Union[str, dict], driver: str = "sqlalchemy"):
    """Create db connection using specified driver

    Args:
        creds (str or dict): Dict containing database credentials or path to json file containing credentials
        driver (str): Driver to use (e.g. 'psycopg2' or  'sqlalchemy'); default='sqlalchemy' to work with GeoPandas

    Reutrn:
        con : Database connection
    """
    database = read_creds(creds)

    host = database["host"]
    dbname = database["dbname"]
    user = database["user"]
    password = database["password"]
    port = database["port"]

    if driver == "psycopg2":
        con = psycopg2.connect(
            host=host, database=dbname, user=user, password=password, port=port
        )
    elif driver == "sqlalchemy":
        con_url = f"postgresql://{user}:{quote(password)}@{host}:{port}/{dbname}"
        con = create_engine(con_url)
    else:
        print(f"{driver} is not implemented")

    return con


def setup_db(connection, detects_table="detects"):
    if isinstance(connection, Engine):
        with connection.connect() as conn:
            db_setup = text(
                f"""
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

                CREATE TABLE IF NOT EXISTS {detects_table} (
                    confidence REAL,
                    label TEXT,
                    image_id TEXT,
                    model_name TEXT,
                    model_type TEXT,
                    bands INTEGER[],
                    image_datetime_utc TIMESTAMPTZ,
                    processed_date_utc TIMESTAMPTZ,
                    geometry GEOMETRY(Geometry, 4326),
                    global_id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                );
            """
            )
            conn.execute(db_setup)
            conn.commit()
