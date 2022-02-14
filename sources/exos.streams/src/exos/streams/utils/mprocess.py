import multiprocessing as mp

def foo(q):
    q.put('hello world')

def queuefun():
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    p.join()
    print(q.get())
