from multiprocessing import Lock, Value

class Counter():
	def __init__(self):
		self.value = None
		self.lock = Lock()

	def decrement(self):
		self.lock.acquire()
		self.value.value -= 1
		self.lock.release()

	def set_value(self, value):
		self.lock.acquire()
		self.value.value = value
		self.lock.release()

	def set_initial_value(self, value):
		self.value = Value('i', value)

value = Counter()
