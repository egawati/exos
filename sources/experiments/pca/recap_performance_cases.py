import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)

### https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python

from metrics import recap_performance_by_cases
import argparse

def define_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs="*", type=int, default=['Case1', 'Case2', 'Case3', 'Case4'])
    parser.add_argument('--bname', type=str, default='100K')
    parser.add_argument('--relpath', type=str, default='pickles/performance/cases')
    parser.add_argument('--nstreams', type=int, default='15')
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
	args = define_arguments()
	recap_performance_by_cases(rel_path = args.relpath, 
                               cases=args.cases,
                               bname = args.bname,
                               nstreams=args.nstreams)