# Class to interact with the data stored in the MySQL server
# Allows reads and writes. Also checks for duplicate data

import json
import logging
import sqlalchemy
import pandas as pd
import numpy as np
import pypika
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy_utils import get_mapper
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import Integer, String, Text
from sqlalchemy.dialects.mysql import LONGTEXT

from src.utils.uQuery import select


class SqlAlchConnection(object):
    def __init__(self, host, port, database, user, password):
        """
        By default the connection is made to the sql_server client
        :param sql_connection: str. Type of connection. Default is sql_server
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = database
        self.connection_mysql = sqlalchemy.create_engine('mysql+pymysql://{}:{}@{}:{}/{}'
                                                         .format(self.user, self.password, self.host, self.port,
                                                                 self.db))
        self.metadata = sqlalchemy.MetaData(self.connection_mysql)
        self.inspector = sqlalchemy.inspect(self.connection_mysql)
        self.session = sessionmaker(bind=self.connection_mysql)

    def reflect(self, schema=None):
        self.metadata.reflect(self.connection_mysql, schema=schema)

    def automap_tables(self, schema):
        Base = automap_base()
        Base.prepare(self.connection_mysql, reflect=True, schema=schema)
        return Base

    def create_schema_if_not_exists(self, schema_name):
        if not schema_name in self.connection_mysql.dialect.get_schema_names(self.connection_mysql):
            self.connection_mysql.execute(sqlalchemy.schema.CreateSchema(schema_name))

    def define_and_create_table(self, tbl_name, schema, columns_names, columns_types, primary_key_flags,
                                nullable_flags):
        logging.info(f"Creating table {tbl_name} into Schema {schema}")
        if all(isinstance(item, str) for item in columns_types):
            columns_types = [eval(item) for item in columns_types]
        table = sqlalchemy.Table(tbl_name, self.metadata,
                                 *(sqlalchemy.Column(column_name, column_type,
                                                     primary_key=primary_key_flag,
                                                     nullable=nullable_flag)
                                   for column_name,
                                       column_type,
                                       primary_key_flag,
                                       nullable_flag in zip(columns_names,
                                                            columns_types,
                                                            primary_key_flags,
                                                            nullable_flags)),
                                 schema=schema)

        table.create()

    def define_index(self, tbl_name, schema, index, name, unique=False):
        if any(name in item['name'] for item in self.inspector.get_indexes(tbl_name, schema)):
            logging.info("Index_name in table. Passing")
        else:
            logging.info(f"Creating index name {name} for table {schema}.{tbl_name}")
            table = sqlalchemy.Table(tbl_name, self.metadata, schema=schema, autoload=True)
            index_field = sqlalchemy.Index(name, *[table.c[item] for item in index], unique=unique)
            index_field.create()

    def table_in_schema(self, table_name, schema):
        self.reflect(schema=schema)
        if self.metadata.tables.get(f"{schema}.{table_name}") is not None:
            return True
        else:
            return False

    def read_table(self, table, schema):
        try:
            with self.connection_mysql.connect() as conn:
                table = pd.read_sql_table(table, con=conn, schema=schema)
                return table
        except Exception as e:
            logging.error("Something happen when reading table. Reason %s" % str(e))

    def read_query(self, query):
        """
        :param query: str. Query to be performed in sql format
        :return: Dataframe. Table with the resultant query
        """
        if isinstance(query, pypika.queries.QueryBuilder):  # This works for mysql the get sql for others check.
            query = query.get_sql(quote_char=None)
        try:
            with self.connection_mysql.connect() as conn:
                table = pd.read_sql_query(query, con=conn)
                return table
        except Exception as e:
            logging.error("Something happen when reading table. Reason %s" % str(e))

    def define_and_read_query(self, from_query_dict):
        schema = from_query_dict['schema']
        table = from_query_dict['table']
        columns = from_query_dict.get('columns', '*')
        where = from_query_dict.get('where', None)
        order = from_query_dict.get('order', None)
        distinct = from_query_dict.get('distinct', False)

        query = select(schema=schema, table=table, columns=columns,
                       where=where, order=order)
        if distinct:
            query = query.distinct()
        retDF = self.read_query(query=query)
        return retDF

    def write_table(self, df, table, schema, sql_types_dict=None, action='append'):
        """
        :param df:
        :param table:
        :param action: str. What to do with the table if exists. Options accepted 'append' & 'replace' (default is fail)
        :return:
        """
        logging.info("Inserting table %s" % str(table))
        try:
            with self.connection_mysql.connect() as conn:
                df.to_sql(table, if_exists=action, con=conn, schema=schema, index=False, dtype=sql_types_dict)
        except Exception as e:
            logging.error("Something happen when reading table. Reason %s" % str(e))

    # Todo Deprecated
    def iterator_data(self, data, schema, table):
        if isinstance(data, pd.DataFrame):
            self.reflect(schema=schema)
            # Option 1
            data.apply(lambda x: self.insert_to_db(x.to_dict(), schema, table), axis=1)
            # Option 2
            # for index, row in data.iterrows():
            #     self.insert_to_db(row.to_dict(), schema, table)
        else:
            raise TypeError("Only DataFrames are allowed")

    def insert_to_db(self, data, schema, table):
        """
        Inserts dictionary as row in the corresponding table in the database.
        :param data: Dict with keys being the columns of the table
        :param schema: str. Schema name
        :param table: str Table name
        """
        # Reflect metadata
        if isinstance(data, dict):
            data = [data]
        try:
            self.metadata.tables['{schema}.{table}'.format(schema=schema, table=table)].insert().execute(data)
        except sqlalchemy.exc.IntegrityError:
            pass

    def map_tables(self, schema, tbl_name):
        base = self.automap_tables(schema)
        if base.classes.get(tbl_name) is not None:
            table = base.classes.get(tbl_name)
            return table
        else:
            logging.error("Table name not in schema")
            raise ValueError

    def get_session(self):
        return self.session()

    def bulk_insert(self, chunk, tbl_name, schema, query):
        table_mapper = self.map_tables(schema, tbl_name)
        session = self.get_session()
        mask_exists = self.validate_bulk_insert(chunk, eval(query))
        chunk_without_duplicates = chunk[[not elem for elem in mask_exists]]
        if not chunk_without_duplicates.empty:
            logging.info(f"Inserting non duplicated values into SQL TABLE {tbl_name}")
            session.bulk_insert_mappings(get_mapper(table_mapper), chunk_without_duplicates.to_dict(orient='records'))
            session.commit()

    @staticmethod
    def validate_bulk_insert(chunk, query):
        results = query.all()
        return chunk['date_time'].isin([item[0] for item in results]).tolist()


def read_unique_elem_column_tables(sql_reader, table_name, schema, column):
    query_dict = {'table': table_name, 'schema': schema, 'columns': [column], 'distinct': True}
    table_df = sql_reader.define_and_read_query(query_dict)
    return table_df[column].tolist()


def get_columns_table(sql_reader, table_name, schema):
    table = sqlalchemy.Table(table_name, sql_reader.metadata, schema=schema, autoload=True)
    columns = [column.key for column in table.columns]

    return columns


def validate_frames(data):
    data = data.applymap(transform_to_json)
    data = data.where(pd.notnull(data), None)
    return data


def validate_frames_before_insert(sql_reader, table_name, data, schema, sql_types_dict=None):
    sql_reader.create_schema_if_not_exists(schema_name=schema)
    data = data.applymap(transform_to_json)
    data = data.where(pd.notnull(data), None)
    if sql_reader.table_in_schema(table_name, schema):
        sql_reader.iterator_data(data, schema, table_name)
    else:
        sql_reader.write_table(data, table_name, schema, sql_types_dict=sql_types_dict, action='append')


def transform_to_json(x):
    if isinstance(x, (list, dict)):
        x = json.dumps(x)
    elif isinstance(x, np.ndarray):
        x = x.tolist()
        x = json.dumps(x)
    else:
        x
    return x
