
class Compose(object):
    # Class used to compose different kinds of object transforms
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, object_boxes, transform_randoms):
        #should reuse the random varaible in video transforms
        for t in self.transforms:
            object_boxes = t(object_boxes, transform_randoms)
        return object_boxes


class PickTop(object):
    # pick top scored object boxes.
    def __init__(self, top_k):
        self.top_k = top_k

    def __call__(self, objects, _):
        objects = objects.top_k(self.top_k)
        return objects


class Resize(object):
    def __call__(self, object_boxes, transform_randoms):
        # resize according to video transforms
        size = transform_randoms["Resize"]
        if object_boxes is not None:
            object_boxes = object_boxes.resize(size)
        return object_boxes


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, object_boxes, transform_randoms):
        # flip according to video transforms
        flip_random = transform_randoms["Flip"]
        if flip_random < self.prob:
            object_boxes.transpose(0)
        return object_boxes
