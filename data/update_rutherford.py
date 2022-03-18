#!/usr/bin/python3

import datetime
import sys

_CSV_PATH = 'csv/rutherford_data.csv'
_TIME_FORMAT = '%-m/%-d/%Y'

with open(_CSV_PATH) as f:
    lines = f.readlines()
    current = int(lines[-1].split(',')[1])
    last_updated_date = datetime.datetime.strptime(lines[-1].split(',')[0], '%m/%d/%Y')
    new = int(sys.argv[1])
    today = datetime.datetime.now().strftime(_TIME_FORMAT)
    new_line = (f'{today},{new+current},{new}\n')
    with open(_CSV_PATH, 'a') as w:
        last_updated_date += datetime.timedelta(days=1)
        while last_updated_date.strftime(_TIME_FORMAT) != today:
            w.write(f'{last_updated_date.strftime(_TIME_FORMAT)},{current},0\n')
            last_updated_date += datetime.timedelta(days=1)
        w.write(new_line)
