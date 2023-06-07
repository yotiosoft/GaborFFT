from concurrent.futures import ThreadPoolExecutor
import time
def func():
    print("start")
    time.sleep(1)
start = time.time()
with ThreadPoolExecutor(max_workers=4) as e:
    for i in range(8):
        e.submit(func)    
print (time.time()-start)
