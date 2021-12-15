#!/usr/bin/python3

import _thread
import time


def keep_sql_interactive(threadName, delay):
    while True:
        time.sleep(delay)
        # todo 调用一次查询
        print('selct from tables')

# 为线程定义一个函数
def print_time( threadName, delay):
   count = 0
   while count < 5:
      time.sleep(delay)
      count += 1
      print ("%s: %s" % ( threadName, time.ctime(time.time()) ))

# 创建两个线程
try:
   _thread.start_new_thread( print_time, ("Thread-1", 2) )
   _thread.start_new_thread( print_time, ("Thread-2", 4) )
   _thread.start_new_thread(keep_sql_interactive, ("Thread-333", 5))
except:
   print ("Error: 无法启动线程")
cnt = 1
while 1:
    cnt = cnt + 1
    if cnt % 10000000 == 0:
        print(time.time())
