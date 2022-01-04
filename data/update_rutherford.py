#!/usr/bin/python3

import datetime
import sys

with open('csv/rutherford_data.csv') as f:
    lines = f.readlines()
    current = int(lines[-1].split(',')[1])
    new = int(sys.argv[1])
    today = datetime.datetime.now().strftime('%-m/%-d/%Y')
    new_line = (f'{today},{new+current},{new}\n')
    with open('csv/rutherford_data.csv', 'a') as w:
        w.write(new_line)
