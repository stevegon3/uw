#!/usr/bin/env python3
""" Script to execute SQL from a file and export results to CSV.
Usage: python export_script.py input.sql [output.csv]
If output file is not provided, it will use the input filename with .csv extension
"""
import os, sys
import re
import pandas as pd
from uw.postgres import Postgres
from uw.settings import sett

def strip_comments(sql: str) -> str:
    """Remove SQL comments from the SQL string.
    Handles both single-line (--) and multi-line (/* */) comments."""
    # Remove single-line comments (-- until end of line)
    sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
    # Remove multi-line comments (/* ... */)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Clean up any resulting double newlines
    sql = re.sub(r'\n\s*\n', '\n\n', sql)
    # Remove any leading/trailing whitespace
    return sql.strip()

def export_sql_to_csv(sql_file: str, output_file: str = None) -> None:
    """Execute SQL from file and export results to CSV
        sql_file: Path to the SQL file to execute
        output_file: Path to the output CSV file (optional)
    """
    if not os.path.exists(sql_file):
        print(f"Error: File not found: {sql_file}")
        sys.exit(1)
    with open(sql_file, 'r') as f:
        sql_text = f.read().strip()
    if not sql_text:
        print("Error: SQL file is empty")
        sys.exit(1)
    if not output_file:
        base_name = os.path.splitext(sql_file)[0]
        output_file = f"{base_name}.txt"
    if os.path.exists(output_file):
        os.remove(output_file)
    pg = Postgres(PostgresPassword=sett.PostgresPassword)
    print(f"Executing SQL from: {sql_file}")
    print(f"Exporting to: {output_file}")
    sql_statements = sql_text.split(';\n')
    for sql in sql_statements:
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        print(f'Working on SQL {sql[:40]}...')
        sql_strip = strip_comments(sql)
        if sql_strip.upper()[:6] in ['DROP T', 'CREATE', 'ALTER ', 'GRANT ', 'INSERT', 'DELETE', 'UPDATE']:
            sql_res = pg.exec_sql(sql)
            res = [[f'result of DDL query: {sql_res[0]}']]
        else:
            res = [['--SQL DML query results:']]
            res += pg.get_rows(sql, return_headers=True)
            res.insert(2, ['-------------------'])
        with open(output_file, "a") as f:
            f.write(sql + "\n")
            for line in res:
                line = [str(c) for c in line]
                line_str = ', '.join(line) + "\n"
                f.write(line_str)
            f.write("\n")
        print(f"Successfully exported {len(res)} rows to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_script.py input.sql [output.txt]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    export_sql_to_csv(input_file, output_file)