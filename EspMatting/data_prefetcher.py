import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
            self.img = self.next_input['image']
            self.mask_gt = self.next_input['mask']
            self.alpha_gt = self.next_input['alpha']
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.img = self.img.cuda(non_blocking=True)
            self.mask_gt = self.mask_gt.cuda(non_blocking=True)
            self.alpha_gt = self.alpha_gt.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.img
        mask_gt = self.mask_gt
        alpha_gt = self.alpha_gt
        self.preload()
        return img, mask_gt, alpha_gt


#默认情况下，PyTorch 将所有涉及到 GPU 的操作（比如内核操作，cpu->gpu，gpu->cpu）
# 都排入同一个 stream（default stream）中，并对同一个流的操作序列化，它们永远不会并行。
# 要想并行，两个操作必须位于不同的 stream 中。
#前向传播位于 default stream 中，
# 因此，要想将下一个 batch 数据的预读取（涉及 cpu->gpu）与当前 batch 的前向传播并行处理，就必须：
#（1） cpu 上的数据 batch 必须 pinned;
#（2）预读取操作必须在另一个 stream 上进行
#上面的 data_prefetcher 类满足这两个要求。注意 dataloader 必须设置 pin_memory=True 来满足第一个条件。