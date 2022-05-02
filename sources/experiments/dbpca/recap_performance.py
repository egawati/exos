import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)

### https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python

from metrics import recap_performance_info
import argparse

def define_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstreams', nargs="*", type=int, default=[5,10,15,20,25,30,35,40,45,50])
    parser.add_argument('--bname', type=str, default='100K_Case1')
    parser.add_argument('--relpath', type=str, default='pickles/performance/nstreams')
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
	args = define_arguments()
	recap_performance_info(rel_path  =  args.relpath, 
						   n_streams = args.nstreams,
						   bname = args.bfname)