from __future__ import print_function
import time
from six import reraise
import sys


try:
    while True:
        print('asdf')
        time.sleep(1)
except:
    reraise(*sys.exc_info())
finally:
    print('finally')


