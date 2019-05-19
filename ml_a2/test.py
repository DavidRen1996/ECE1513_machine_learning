import threading
import time

class thr(object):

  def __init__(self, name):
     self.name = name
     self.x = 0

  def run(self):
     for i in list(range(10)):
         self.x +=1
         print("something {0} {1}".format(self.name, self.x))
         time.sleep(1)

F = thr("First")
S = thr("Second")

t1 = threading.Thread(target=F.run)
t2 = threading.Thread(target=S.run)
t1.start()
t2.start()