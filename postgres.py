"""
Using pyscopg3 to connect to Postgres
v1.0 forked from util on 11/26/2023
pip install "psycopg[binary]"
poetry add "psycopg[binary]"
DB Helper functions for reporting
"""
import re
import psycopg
from dataclasses import dataclass
import logging
import datetime as dtTime
import time
from src.helper import list_to_df
from typing import List
from src.settings import sett

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class Postgres:
    PostgresServerHost: str = "ugali"
    PostgresServerDatabase: str = "money"
    PostgresServerPort: str = "5432"
    PostgresServerUser: str = ""
    PostgresServerPassword: str = ""
    default_schema: str = "money"

    def __post_init__(self):
        self.__connection = None
        self.PostgresServerHost = sett.PostgresServerHost if not self.PostgresServerHost else self.PostgresServerHost
        self.PostgresServerDatabase = sett.PostgresServerDatabase if not self.PostgresServerDatabase else self.PostgresServerDatabase
        self.PostgresServerPort = sett.PostgresServerPort if not self.PostgresServerPort else self.PostgresServerPort
        self.PostgresServerUser = sett.PostgresServerUser if not self.PostgresServerUser else self.PostgresServerUser
        self.PostgresServerPassword = sett.PostgresServerPassword if not self.PostgresServerPassword else self.PostgresServerPassword

    @property
    def connection(self):
        if not self.__connection:
            self.__connection = psycopg.connect(
                host=self.PostgresServerHost,
                port=self.PostgresServerPort,
                user=self.PostgresServerUser,
                password=self.PostgresServerPassword,
                dbname=self.PostgresServerDatabase,
                sslmode="allow"
            )
        return self.__connection

    @property
    def curs(self):
        _c = self.connection.cursor()
        _c.execute(f"SET SEARCH_PATH TO {self.default_schema},public")
        return _c

    def cursor(self):
        _c = self.connection.cursor()
        _c.execute(f"SET SEARCH_PATH TO {self.default_schema},public")
        return _c

    def truncate_table(self, table_name):
        sql = f"TRUNCATE TABLE {table_name}"
        with self.curs as curs:
            try:
                curs.execute(sql)
                self.connection.commit()
                return 1
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error}")
                self.connection.rollback()
                print(f"[money]Postgres error {sql}")
                return -1, error

    @property
    def jdbc_url(self) -> str:
        return (
            f"jdbc:postgresql://{self.PostgresServerHost}:{self.PostgresServerPort}"
            f"/{self.PostgresServerDatabase}"
            f"?user={self.PostgresServerUser}&password={self.PostgresServerPassword}"
        )

    def get_df(self, sql: str):
        """Pass in a SQL statement and I will return a DataFrame with Column Headers"""
        data = self.get_rows(sql, True)
        return list_to_df(data)

    def get_rows(self, sql: str, return_headers=True) -> List[List[str]]:
        if '&amp;' in sql:
            logger.error(f"Possible problem with SQL text &amp; found {sql}")
            print(f"[money]Possible problem with SQL text &amp; found {sql}")
        with self.curs as curs:
            try:
                curs.execute(sql)
                self.connection.commit()
                if return_headers:
                    rows = [[desc[0] for desc in curs.description]]
                else:
                    rows = []
                for row in curs:
                    rows.append(list(row))
                return rows
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error} {sql}")
                self.connection.rollback()
                print(f"[money]Postgres error {sql}")
                return -1

    def get_col(self, sql: str, return_headers=True) -> List[str]:
        rows = self.get_rows(sql, False)
        return [x[0] for x in rows]

    def strip_alpha(self, theString):
        # Strip characters, return a number
        if type(theString) is str:
            theString2 = '0' + (''.join(x for x in theString if x.isdigit()))
            return float(theString2)
        elif isinstance(theString, dtTime.date) or isinstance(theString, dtTime.date):
            return float(theString.strftime('%Y%m%d%H%M%S'))
        elif theString is None or theString == '':
            return float(0)
        else:
            return float(theString)

    def remove_trailing_semicolon(self, sql):
        if ';' not in sql:
            return sql
        sql = re.sub(r'[\s\n\r\t]+$', '', sql)
        return sql.strip()[:-1]

    def get_num(self, sql: str):
        """Give me SQL and I will return a single cell number"""
        with self.curs as curs:
            if ' LIMIT ' not in sql.upper():
                sql2 = f'{self.remove_trailing_semicolon(sql)} LIMIT 1'
            else:
                sql2 = sql
            try:
                curs.execute(sql2)
                self.connection.commit()
                result = curs.fetchone()
                if result is None:
                    return 0
                elif len(list(result)) >= 1 and result[0] is None:
                    return 0
                elif type(result[0]) is str:
                    return float(self.strip_alpha(result[0]))
                else:
                    return result[0]

            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error} {sql}")
                self.connection.rollback()
                print(f"[money]Postgres error {sql}")
                return -1

    def get_str(self, sql: str) -> str:
        """Give me SQL and I will return a single cell string"""
        with self.curs as curs:
            if ' LIMIT ' not in sql.upper():
                sql2 = f'{self.remove_trailing_semicolon(sql)} LIMIT 1'
            else:
                sql2 = sql
            try:
                curs.execute(sql2)
                self.connection.commit()
                result = curs.fetchone()
                if result is None:
                    return ''
                elif type(result[0]) is not str:
                    return str(result[0])
                else:
                    return result[0]

            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error} {sql}")
                self.connection.rollback()
                print(f"[money]Postgres error {sql}")
                return ''

    def exec_sql(self, sql: str, commit=True):
        with self.curs as curs:
            try:
                curs.execute(sql)
                if commit:
                    self.connection.commit()
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error} {sql}")
                self.connection.rollback()
                print(f"[money]Postgres error {sql}")
                return -1, error
            res = ''
            try:
                res = curs.fetchone()
            except (Exception, psycopg.DatabaseError) as error:
                pass
            return 1, res

    def update_rows(self, sql: str, data: List[any], commit=True):
        """Updating a table setting some columns = '' based on the Primary Key `id`
            Use `execute`
            Hardcode the static `SET` values
            Pass in a list of keys to `ANY(%s)` converted to a set without a second element
            `.curs.execute(sql, (list_of_ids, ))`"""
        with self.curs as curs:
            try:
                res = self.curs.execute(sql, (data,))
                if commit:
                    self.connection.commit()
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error}")
                self.connection.rollback()
                print(f"[money]Postgres error {sql}")
                return -1, error
            res = ''
            try:
                res = curs.fetchone()
            except (Exception, psycopg.DatabaseError) as error:
                pass
            return 1, res

    def insert_rows(self, sql: str, data, commit=True):
        if sql.count('%s') != len(data[0]):
            logger.warning(f"SQL Insert {sql.count('%s')=} does not have proper number of columns as data being passed {len(data)=}")
        with self.curs as curs:
            try:
                curs.executemany(sql, data)
                if commit:
                    self.connection.commit()
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error}")
                self.connection.rollback()
                print(f"[money]Postgres error {sql}")
                return -1, error
            res = ''
            try:
                res = curs.fetchone()
            except (Exception, psycopg.DatabaseError) as error:
                pass
            return 1, res

    def insert_df(self, df, table):
        """
        Using psycopg2.extras.execute_values() to insert the dataframe
        """
        # Create a list of tupples from the dataframe values
        tuples = [tuple(x) for x in df.to_numpy()]
        # Comma-separated dataframe columns
        cols = ','.join(list(df.columns))
        # SQL query to execute
        sql = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
        cur = self.connection.cursor()
        try:
            res = psycopg.execute_values(cur, sql, tuples)
            self.connection.commit()
        except (Exception, psycopg.DatabaseError) as error:
            print("[money]Postgres error: %s" % error)
            print(table, cols)
            print(tuples[0:10])
            self.connection.rollback()
            cur.close()
            return 1
        cur.close()
        return res

    def get_fks(self, table: str):
        """Give me a table and I will return all the Foreign Key constraints for that table"""
        #Need to replace with this SQL
        """select 
  (select r.relname from pg_class r where r.oid = c.conrelid) as table, conname,pg_get_constraintdef(oid),
  (select array_agg(attname) from pg_attribute 
   where attrelid = c.conrelid and ARRAY[attnum] <@ c.conkey) as col, 
  (select r.relname from pg_class r where r.oid = c.confrelid) as ftable 
from pg_constraint c 
where c.confrelid = (select oid from pg_class where relname = 'wms_distributioncenter');"""
        sql = f"""SELECT c.conname as fk_name
              FROM pg_constraint c
                INNER JOIN pg_namespace AS sh ON sh.oid = c.connamespace
                INNER JOIN (SELECT oid, unnest(conkey) as conkey FROM pg_constraint) con ON c.oid = con.oid
                INNER JOIN pg_class tbl ON tbl.oid = c.conrelid
                INNER JOIN pg_attribute col ON (col.attrelid = tbl.oid AND col.attnum = con.conkey)
                INNER JOIN pg_class referenced_tbl ON c.confrelid = referenced_tbl.oid
                INNER JOIN pg_namespace AS referenced_sh ON referenced_sh.oid = referenced_tbl.relnamespace
                INNER JOIN (SELECT oid, unnest(confkey) as confkey FROM pg_constraint) conf ON c.oid = conf.oid
                INNER JOIN pg_attribute referenced_field ON (referenced_field.attrelid = c.confrelid AND referenced_field.attnum = conf.confkey)
            WHERE c.contype = 'f' and lower(tbl.relname)='{table.lower()}' ORDER BY 1"""
        return self.get_col(sql)

    def kill_pids(self, db_name):
        """Check for running processes other than admin/dba and kill them.
        Used just before ETL processes"""
        sql_find = f"""SELECT DISTINCT pid FROM pg_stat_activity 
          WHERE datname = '{db_name}' AND backend_type = 'client backend' AND usename NOT IN ('ods_service','rdsadmin') AND usename IS NOT NULL"""
        pids_to_kill = self.get_col(sql_find)
        if len(pids_to_kill) > 0:
            sql_user = """SELECT datname||'-'||usename||'-'||application_name user_info FROM pg_stat_activity"""
            logger.warning(f'Active Postgres users; killing PIDs')
            for pid in pids_to_kill:
                user_info = self.get_str(f'{sql_user} WHERE pid = {pid}')
                logger.warning(f'Killing PID {pid} user info: {user_info}')
                self.exec_sql(f"""SELECT pg_cancel_backend({pid});""")
            time.sleep(30)
            pids_to_kill = self.get_col(sql_find)
            if len(pids_to_kill) > 0:
                logger.warning(f'Active Postgres users; FORCE killing PIDs')
                for pid in pids_to_kill:
                    user_info = self.get_str(f'{sql_user} WHERE pid = {pid}')
                    logger.warning(f'FORCE Killing PID {pid} user info: {user_info}')
                    self.exec_sql(f"""SELECT pg_terminate_backend({pid});""")

    @property
    def get_pg_class_info(self):
        return {'PostgresServerHost': self.PostgresServerHost,
                'PostgresServerPort': self.PostgresServerPort,
                'PostgresServerUser': self.PostgresServerUser,
                'PostgresServerPassword': self.PostgresServerPassword,
                'PostgresServerDatabase': self.PostgresServerDatabase}

    @property
    def get_db_info(self):
        """Get connection info for the current connection"""
        with self.curs as curs:
            try:
                curs.execute("SELECT version();")
                version = curs.fetchone()[0]
                return {
                    "PostgresServerHost": self.PostgresServerHost,
                    "PostgresServerDatabase": self.PostgresServerDatabase,
                    "PostgresServerUser": self.PostgresServerUser,
                    "version": version
                }
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Error: {error}")
                return None