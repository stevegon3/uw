import psycopg
import json, inspect, os, datetime, time
from dataclasses import dataclass
import pandas as pd
from contextlib import contextmanager
from uw.db_helper import list_to_df, normalize_value
from uw.settings import sett

@dataclass
class Postgres:
    PostgresHost: str = "ugali"
    PostgresDB: str = "uw"
    PostgresPort: str = "5432"
    PostgresUser: str = "postgres"
    PostgresPassword: str = "pg_secret"

    def __post_init__(self):
        self.__connection = None
        self.connection_params = {
            "host": self.PostgresHost,
            "port": self.PostgresPort,
            "user": self.PostgresUser,
            "password": self.PostgresPassword,
            "dbname": self.PostgresDB,
            "options": "-c search_path=invest,public",  # This sets the default search path
            "sslmode": "allow"
        }

    @property
    def connection(self):
        """Provides the current connection or establishes a new one"""
        if not self.__connection:
            self.__connection = psycopg.connect(**self.connection_params)
        return self.__connection

    @contextmanager
    def cursor(self):
        """Provides a NEW cursor from the current connection"""
        with self.connection.cursor() as cur:
            # Set search path for this connection
            cur.execute(f"SET search_path TO {sett.PostgresSchema},public")
            yield cur

    @property
    def jdbc_url(self) -> str:
        return (
            f"jdbc:postgresql://{self.PostgresHost}:{self.PostgresPort}"
            f"/{self.PostgresDB}"
            f"?user={self.PostgresUser}&password={self.PostgresPassword}"
        )

    def strip_sql(self, sql: str) -> str:
        return sql.strip().strip(';')

    def get_df(self, sql: str, coerce=True):
        return list_to_df(self.get_rows(sql, True))

    def get_rows(self, sql: str, return_headers=True):
        with self.cursor() as curs:
            try:
                curs.execute(self.strip_sql(sql))
                self.connection.commit()
                if return_headers and curs.description:
                    rows = [[desc[0] for desc in curs.description]]
                else:
                    rows = []
                for row in curs:
                    rows.append(list(row))
                return rows
            except (Exception, psycopg.DatabaseError) as error:
                sett.log.error("Error: %s" % error)
                self.connection.rollback()
                print(f"Postgres error {sql}")
                return -1

    def get_raw(self, sql):
        """Execute SQL and return back native results"""
        with self.cursor() as curs:
            try:
                curs.execute(self.strip_sql(sql))
                self.connection.commit()
                return curs.fetchall()
            except (Exception, psycopg.DatabaseError) as error:
                sett.log.error("Error: %s" % error)
                self.connection.rollback()
                return -1

    def get_col(self, sql: str, return_headers=True):
        rows = self.get_rows(self.strip_sql(sql), False)
        return [x[0] for x in rows]

    def strip_alpha(self, theString):
        # Strip characters, return a number
        if type(theString) is str:
            theString2 = '0' + (''.join(x for x in theString if x.isdigit()))
            return float(theString2)
        elif isinstance(theString, datetime.date) or isinstance(theString, datetime.date):
            return float(theString.strftime('%Y%m%d%H%M%S'))
        elif theString is None or theString == '':
            return float(0)
        else:
            return float(theString)

    def get_num(self, sql: str):
        """Give me SQL and I will return a single cell number"""
        with self.cursor() as curs:
            sql = self.strip_sql(sql)
            if ' LIMIT ' not in sql.upper():
                sql2 = f'{sql} LIMIT 1'
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
                sett.log.error(f"Error: {error}")
                self.connection.rollback()
                print(f"Postgres error {sql}")
                return -1

    def get_str(self, sql: str):
        """Give me SQL and I will return a single cell string"""
        sql = self.strip_sql(sql)
        with self.cursor() as curs:
            if ' LIMIT ' not in sql.upper() and not sql.startswith('SHOW '):
                sql2 = f'{sql} LIMIT 1'
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
                sett.log.error(f"Error: {error}")
                self.connection.rollback()
                print(f"Postgres error {sql}")
                return ''

    def exec_sql(self, sql: str, commit=True):
        with self.cursor() as curs:
            try:
                curs.execute(sql)
                if commit:
                    self.connection.commit()
            except (Exception, psycopg.DatabaseError) as error:
                sett.log.error(f"Error: {error}")
                self.connection.rollback()
                print(f"Postgres error {sql}")
                return -1, error
            res = ''
            try:
                res = curs.fetchone()
            except (Exception, psycopg.DatabaseError) as error:
                pass
            return 1, res

    def insert_serial(self, sql):
        """
        Insert a row and return the serial ID (works with SERIAL or IDENTITY columns).
        The SQL must be an INSERT statement.
        """
        if not sql.strip().lower().startswith('insert'):
            sett.log.warning("Not an INSERT statement; aborting")
            return None
        try:
            # Clean up the SQL and add RETURNING clause
            sql = sql.rstrip('; ')
            if ' returning ' not in sql.lower():
                sql = f"{sql} RETURNING id;"
            # Execute the query and return the ID
            res = self.exec_sql(sql)
            if res and len(res) > 1 and res[1]:
                return int(res[1][0])
            return None
        except Exception as e:
            sett.log.error(f"Error in insert_serial: {e}\nSQL: {sql}")
            return None

    def insert_rows(self, sql: str, data, commit=True):
        if sql.count('%s') != len(data[0]):
            sett.log.warning(f"SQL Insert {sql.count('%s')=} does not have proper number of columns as data being passed {len(data)=}")
        sql = self.strip_sql(sql)
        with self.cursor() as curs:
            try:
                curs.executemany(sql, data)
                if commit:
                    self.connection.commit()
            except (Exception, psycopg.DatabaseError) as error:
                sett.log.error(f"Error: {error}")
                self.connection.rollback()
                print(f"Postgres error {sql}")
                return -1, error
            res = ''
            try:
                res = curs.fetchone()
            except (Exception, psycopg.DatabaseError) as error:
                pass
            return 1, res

    def get_table_columns(self, table_name: str, schema=sett.PostgresSchema) -> dict:
        """Fetch column names and data types for a table"""
        query = """SELECT column_name, data_type, udt_name
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = %s
                    ORDER BY ordinal_position"""
        if '.' in table_name:
            schema = table_name.split('.')[0]
            table_name = table_name.split('.')[1]
        with self.cursor() as cur:
            cur.execute(query, (table_name, schema))
            return {row[0]: row[1] for row in cur.fetchall()}

    def insert_df(self, df: pd.DataFrame, table_name: str) -> int:
        """CANNOT HAVE NaN/NaT will fail; use df_sync_pg instead
        Uses COPY with psycopg3"""
        cols = ','.join(list(df.columns))
        for i, col in enumerate(df.columns):
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(bool)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        cur = self.connection.cursor()
        try:
            with cur.copy(f"COPY {table_name} ({cols}) FROM STDIN") as copy:
                for idx, row in enumerate(df.itertuples(index=False, name=None)):
                    clean_row = [None if pd.isna(x) else x for x in row]
                    copy.write_row(clean_row)
            self.connection.commit()
            return idx
        except Exception as error:
            sett.log.error(f"Error in COPY FROM: {error} on row {idx} (note: due to buffering that might not be the correct row)")
            sett.log.error(table_name, cols)
            sett.log.error(df.iloc[idx])
            self.connection.rollback()
            return -1
        finally:
            cur.close()

    def get_fks(self, table: str):
        """Give me a table and I will return all the Foreign Key constraints for that table"""
        #Need to replace with this SQL
        """SELECT 
              (SELECT r.relname from pg_class r where r.oid = c.conrelid) as table, conname,pg_get_constraintdef(oid),
              (SELECT array_agg(attname) from pg_attribute 
               WHERE attrelid = c.conrelid and ARRAY[attnum] <@ c.conkey) as col, 
              (SELECT r.relname from pg_class r where r.oid = c.confrelid) as ftable 
            FROM pg_constraint c 
            WHERE c.confrelid = (select oid from pg_class where relname = 'wms_distributioncenter');"""
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
            sett.log.warning(f'Active Postgres users; killing PIDs')
            for pid in pids_to_kill:
                user_info = self.get_str(f'{sql_user} WHERE pid = {pid}')
                sett.log.warning(f'Killing PID {pid} user info: {user_info}')
                self.exec_sql(f"""SELECT pg_cancel_backend({pid});""")
            time.sleep(30)
            pids_to_kill = self.get_col(sql_find)
            if len(pids_to_kill) > 0:
                sett.log.warning(f'Active Postgres users; FORCE killing PIDs')
                for pid in pids_to_kill:
                    user_info = self.get_str(f'{sql_user} WHERE pid = {pid}')
                    sett.log.warning(f'FORCE Killing PID {pid} user info: {user_info}')
                    self.exec_sql(f"""SELECT pg_terminate_backend({pid});""")

    def save_data_log(self, script, function, symbols, data, etl_time_pac=None):
        """Log raw data pulls; Do not fail and stop"""
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime to string
            raise TypeError(f"save_data_log: Type {type(obj)} not serializable")

        if data and len(data) != 10 and script != '':
            caller_name = inspect.currentframe().f_back.f_back.f_code.co_name
            caller_file = os.path.basename(inspect.currentframe().f_back.f_back.f_globals['__file__'])
            if isinstance(symbols, list):
                symbols = ','.join(map(str, symbols))
            if not etl_time_pac:
                etl_time_pac = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sql = """INSERT INTO data_log (script,function,caller_file,symbols,data,etl_time_pac) VALUES (%s,%s,%s,%s,%s,%s)"""
            if isinstance(data, dict):
                data_str = json.dumps(data, default=convert_datetime)  # Use custom serializer
            else:
                data_str = str(data)
            data_ins = (script, function, caller_file, symbols, data_str, etl_time_pac)
            try:
                self.connection.cursor().execute(sql, data_ins)
                self.connection.commit()
                return 1
            except Exception as e:
                sett.log.error(f"WARN: {e} in saving data log; resuming")
                self.connection.rollback()
                return -1

    def get_batches(self, sql, batch_size=1000, return_headers=True):
        """Yield batches of rows from a query result"""
        with self.cursor() as cur:
            cur.execute(sql)
            if return_headers:
                col_names = tuple(desc[0] for desc in cur.description)
                first_batch = [col_names]  # Column names as first row
                first_batch.extend(cur.fetchmany(batch_size - 1) if batch_size > 1 else [])
                if first_batch:  # Only yield if we have data
                    yield first_batch
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield rows

def sync_tables(source_pg, target_pg, source_table, target_table=None, source_sql=None):
    """Give me two connections and I will TRUNCATE the target and sync it to the source
    If no target_table then [schema].table names must match.
    source_sql if you have custom sql, otherwise, it will try to sync all columns and they must be exact
    If using source_sql, be sure to alias the source columns exactly to match the target columns.
    `id` col is understood to be primary key serial"""
    source_sql = f"SELECT * FROM {source_table}" if not source_sql else source_sql
    target_table = target_table or source_table
    target_col_type = target_pg.get_table_columns(target_table)
    target_cols = [k for k in target_col_type.keys()]
    with target_pg.connection.cursor() as cur:
        try:
            sett.log.info(f"TRUNCATING {target_table}")
            cur.execute(f"TRUNCATE TABLE {target_table} CASCADE")
            batch_num = 1
            for batch in source_pg.get_batches(source_sql, batch_size=1000):
                if batch_num == 1:
                    source_cols = batch[0]
                    # get intersection of source and target columns
                    final_cols = [s for s in source_cols if s in target_cols]
                    source_cols_str = ','.join(final_cols)
                    placeholders = ','.join(['%s'] * len(final_cols))
                    if 'id' in final_cols:
                        sql_insert = f"INSERT INTO {target_table} ({source_cols_str}) OVERRIDING SYSTEM VALUE VALUES ({placeholders})"
                    else:
                        sql_insert = f"INSERT INTO {target_table} ({source_cols_str}) VALUES ({placeholders})"
                current_batch = batch[1:] if batch_num == 1 else batch
                rows = tuple(tuple(normalize_value(x) for x in row) for row in current_batch)
                if len(rows[0]) != len(final_cols):
                    sett.log.error(f"WARNING: source columns {source_cols=} do not match target columns {target_cols=} did you forget to pass source_sql?")
                if rows:
                    cur.executemany(sql_insert, rows)
                batch_num += 1
                if batch_num == 11:
                    print('More than 10 batches:', end=' ')
                if batch_num > 10 and batch_num % 1000 == 0:
                    print(batch_num, end=' ')
            target_pg.connection.commit()
            if source_sql and 'WHERE ' in source_sql.upper():
                source_pred = source_sql.upper().split('WHERE ')[1].split('ORDER BY ')[0].split('LIMIT ')[0]
                source_row_ct = source_pg.get_num(f"SELECT COUNT(*) FROM {source_table} WHERE {source_pred}")
            else:
                source_row_ct = source_pg.get_num(f"SELECT COUNT(*) FROM {source_table}")
            target_row_ct = target_pg.get_num(f"SELECT COUNT(*) FROM {target_table}")
            if source_row_ct == target_row_ct:
                sett.log.info(f"Successfully synchronized {target_table} {target_row_ct} rows; attempting to reset sequence")
                sql = f"SELECT MAX(id) FROM {target_table};"
                max_id = target_pg.get_num(sql)
                sql = f"SELECT setval('{target_table}_id_seq', {max_id + 1});"
                target_pg.exec_sql(sql)
            else:
                sett.log.warning(f"WARNING: {target_table} row count mismatch {source_row_ct=} != {target_row_ct=}")
            return source_row_ct, target_row_ct
        except Exception as e:
            target_pg.connection.rollback()
            sett.log.error(f"Error synchronizing {target_table}: {e} on batch {batch_num}")
            raise

pg = Postgres(PostgresHost=sett.PostgresHost,
              PostgresPort=sett.PostgresPort,
              PostgresDB=sett.PostgresDB,
              PostgresUser=sett.PostgresUser,
              PostgresPassword=sett.PostgresPassword)