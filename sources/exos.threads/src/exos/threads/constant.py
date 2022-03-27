from threading import Lock

class Counter():
	def __init__(self):
		self.value = None
		self.lock = Lock()

	def decrement(self):
		self.lock.acquire()
		self.value -= 1
		self.lock.release()

	def set_value(self, value):
		self.lock.acquire()
		self.value = value
		self.lock.release()

	def set_initial_value(self, value):
		self.value = value

value = Counter()

