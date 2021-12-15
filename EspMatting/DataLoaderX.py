from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



# 原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，
# 但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
# 使用 prefetch_generator，可以保证线程不会等待，
# 每个线程都总有至少一个数据在加载