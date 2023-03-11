"""
Logging setup for workflow associated with exploring DFE variation.
"""

import datetime
import sys
import os

def print_now():
    return datetime.datetime.now().strftime('[%Y-%m-%d %T]')

def get_today():
    return datetime.datetime.today().strftime('%Y%m%d')

def print_IO(args):
    sys.stdout.write('INFO:{0} - Beginning execution of {1} in directory {2}\n'.format(
        print_now(), sys.argv[0], os.getcwd()))
    sys.stdout.write('INFO:Parsed the following arguments:\n{0}\n'.format(
        '\n'.join(['\t{0} = {1}'.format(*tup) for tup in args.items()])))
    return None

def logINFO(msg):
    sys.stdout.write('INFO:{0} - {1}\n'.format(print_now(), msg))

def logEND(msg):
    sys.stdout.write('INFO:{0} - SUCCESS! {1}\n'.format(print_now(), msg))

def logWARN(msg):
    sys.stderr.write('WARNING:{0} - {1}\n'.format(print_now(), msg))

def join_dict(mydict, sep):
    return sep.join(['{0} = {1}'.format(*tup) for tup in mydict.items()])

# keeping the order of matched files
def join_zip(myzip, sep):
    return sep.join(['{0} = {1}'.format(*tup) for tup in myzip])

