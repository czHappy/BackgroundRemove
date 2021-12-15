import numpy as np
from queue import Queue
import sys
sys.setrecursionlimit(1000000)
EPS = 0.15 # 设置阈值 如果小于此阈值就不算联通
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
cnt = 0 #记录当前连通分量的面积

# 坐标不能越界
def check(x, y, r, c):
    if x >= 0 and y >= 0 and x < r and y < c:
        return True
    return False

# dfs容易爆栈 使用BFS效率高一些
def dfs(alpha, area, x, y, idx):
    # print(x, y)
    for i in range(4):
        fx = x + dx[i]
        fy = y + dy[i]
        # 下一个坐标不越界 并且该处有像素点
        if check(fx, fy, area.shape[0], area.shape[1]) and alpha[fx][fy] >= EPS and area[fx][fy] == 0:
            area[fx][fy] = idx
            global cnt
            cnt = cnt + 1
            dfs(alpha, area, fx, fy, idx)
    return


def bfs(alpha, area, x, y, idx):
    que = Queue()
    cur_cnt = 0
    que.put((x, y))
    area[x][y] = idx
    while(not que.empty()):
        tp = que.get()
        for i in range(4):
            fx = tp[0] + dx[i]
            fy = tp[1] + dy[i]
            if check(fx, fy, area.shape[0], area.shape[1]) and alpha[fx][fy] >= EPS and area[fx][fy] == 0:
                area[fx][fy] = idx
                cur_cnt = cur_cnt + 1
                que.put((fx, fy))
    return cur_cnt

# 把每个连通分量都标上号
def label_area(alpha, area):
    row = alpha.shape[0]
    col = alpha.shape[1]
    idx = 1 #染色标识
    max_idx = 0
    max_cnt = 0
    for i in range(row):
        for j in range(col):
            if area[i][j] == 0 and alpha[i][j] >= EPS: #还没染色过 并且此处是有像素点
                cur_cnt = bfs(alpha, area, i, j, idx)
                if cur_cnt > max_cnt:
                    max_cnt = cur_cnt
                    max_idx = idx
                idx = idx + 1
    return max_idx

# alpha W*H的单通道灰度图 数值范围0-1 float
# 输入alpha 输出是只包含了一个连通分量的alpha
def get_max_connect_area(alpha):
    area = np.zeros_like(alpha)
    idx = label_area(alpha, area)
    one = np.ones_like(area)
    zero = np.zeros_like(area)
    mask = np.where(area == idx, one, zero)
    return alpha * mask

