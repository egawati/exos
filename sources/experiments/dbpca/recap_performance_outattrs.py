import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)


from metrics import recap_performance_by_noutattrs
import argparse

def define_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs="*", type=int, default=['Case1', 'Case4'])
    parser.add_argument('--bname', type=str, default='w1K')
    parser.add_argument('--relpath', type=str, default='pickles/performance/small_cases')
    parser.add_argument('--nstreams', type=int, default=2)
    parser.add_argument('--nsets', type=int, default=1)
    parser.add_argument('--version', type=str, default='')
    parser.add_argument('--min_noutattrs', type=int, default=1)
    parser.add_argument('--max_noutattrs', type=int, default=10)
    args = parser.parse_args()
    return args 


if __name__ == '__main__':

    args = define_arguments()

    version = args.version
    if version == 'v':
        version = ''
	
    recap_performance_by_noutattrs(rel_path = args.relpath, 
                                   cases = args.cases,
                                   min_noutattrs=args.min_noutattrs,
                                   max_noutattrs=args.max_noutattrs,
                                   bname = args.bname,
                                   nstreams=args.nstreams,
                                   nsets = args.nsets,
                                   version = version)