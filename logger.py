import time
import logging

class PerfomanceLogger(object):
	def __init__(self):
		self.task_time_map = {}
		self.logger = logging.getLogger()
		logging.basicConfig(level=logging.INFO)

	def start_monitoring(self, task_name):
		self.task_time_map[task_name] = time.time()

	def stop_monitoring(self, task_name):
		if task_name in self.task_time_map:
			start_time = self.task_time_map[task_name]
			self.logger.info("PerfLog |"+task_name+"|TT:"+str(time.time()-start_time)+"s")
			self.task_time_map.pop(task_name)
		else:
			raise Exception ("Task" + task_name+" not found")