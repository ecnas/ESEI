import time, os, random
from torchvision import transforms
import torchvision.transforms.functional as F
           
class MyRandomResizeTransform(transforms.RandomResizedCrop):
    ACTIVE_SIZE = 224
    IMAGE_SIZE_LIST = [224]

    CONTINUOUS = False
    SYNC_DISTRIBUTED = False

    EPOCH = 0
    # __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
    def __call__(self, img):
        # i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resize(img, [MyRandomResizeTransform.ACTIVE_SIZE, MyRandomResizeTransform.ACTIVE_SIZE])

    @staticmethod
    def get_candidate_image_size():
        if MyRandomResizeTransform.CONTINUOUS:
            min_size = min(MyRandomResizeTransform.IMAGE_SIZE_LIST)
            max_size = max(MyRandomResizeTransform.IMAGE_SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if i % 4 == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = MyRandomResizeTransform.IMAGE_SIZE_LIST

        relative_probs = None
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size(batch_id):
        if MyRandomResizeTransform.SYNC_DISTRIBUTED:
            _seed = int('%d%.3d' % (batch_id, MyRandomResizeTransform.EPOCH))
        else:
            _seed = os.getpid() + time.time()
        random.seed(_seed)
        candidate_sizes, relative_probs = MyRandomResizeTransform.get_candidate_image_size()
        MyRandomResizeTransform.ACTIVE_SIZE = random.choices(candidate_sizes, weights=relative_probs)[0]
             
class MyDeterministicResizeTransform(transforms.RandomResizedCrop):
    # __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
    def __call__(self, img):
        # i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resize(img, self.size)