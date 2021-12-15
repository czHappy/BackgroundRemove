import jpeg4py as jpeg
import cv2 as cv
import time
from tqdm import tqdm
EXAMPLE_JPG = r'E:\Data_Backup\mobile-torch1.x\pexels-photo-307791.jpg'
EXAMPLE_PNG = r'E:\Data_Backup\mobile-torch1.x\pexels-photo-307791.png'
EPOCH = 1000
#E:\Anaconda3\envs\tf2py36\Lib\site-packages\jpeg4py-0.1.4-py3.6.egg!\jpeg4py\_cffi.py
def read_jpeg_tool():
    t0 = time.time()
    for i in tqdm(range(EPOCH)):
        img = jpeg.JPEG(EXAMPLE_JPG).decode()
    t1 = time.time()
    print('jpeg4py: ', t1 - t0)



def read_jpeg_cv():
    t0 = time.time()
    for i in range(EPOCH):
        img = cv.imread(EXAMPLE_JPG)
    t1 = time.time()
    print('cv: ', t1 - t0)

def read_png_cv():
    t0 = time.time()
    for i in range(EPOCH):
        img = cv.imread(EXAMPLE_PNG)
    t1 = time.time()
    print('cv: ', t1 - t0)

#read_jpeg_cv()
#read_png_cv()
read_jpeg_tool()