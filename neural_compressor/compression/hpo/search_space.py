import random


class BaseSearchSpace(object):
    def __init__(
            self,
            bound=None,
            interval=None,
            value=None,
            type=None
            ):
        if bound:
            if not isinstance(bound, (list, tuple)):  # pragma: no cover
                raise TypeError("bound sould be list or tuple, not {}".format(type(bound)))
            if len(bound) != 2:  # pragma: no cover
                raise ValueError("bound sould only contain two elements, [start, end)")
            if bound[1] <= bound[0]:  # pragma: no cover
                raise ValueError("empty range for [{}, {})".format(bound[0], bound[1]))
        assert value or bound, "must set value or bound to initialize the search space"

        self.bound = bound
        self.interval = interval
        self.value = value
        self.type = type
        if type == 'discrete':
            if value:
                self.total_num = len(value)
            else:
                self.total_num = int((bound[1] - bound[0]) / interval)
        else:
            self.total_num = float("inf")

    def get_value(self):
        pass


class DiscreteSearchSpace(BaseSearchSpace):
    def __init__(self, bound=None, interval=None, value=None):
        if bound and interval is None:
            if isinstance(bound[0], int) and isinstance(bound[1], int):
                interval = 1
            else:
                interval = 0.01
        super().__init__(bound=bound,
                         interval=interval,
                         value=value,
                         type='discrete')

    def get_random_value(self):
        idx = random.randint(0, self.total_num - 1)
        return self.get_nth_value(idx)

    def get_nth_value(self, idx):
        if self.bound:
            return round(self.bound[0] + idx * self.interval, 10)
        else:
            return self.value[idx]

    def get_all(self):
        return [self.get_nth_value(i) for i in range(self.total_num)]

    def get_value(self, idx=None):
        if idx is not None:
            if not isinstance(idx, int):
                raise TypeError("The type of idx should be int, not {}".format(type(idx)))
            if idx < 0:
                return self.get_all()
            value = self.get_nth_value(idx)
        else:
            value = self.get_random_value()
        return value
    
    def index(self, value):
        if self.value:
            return self.value.index(value)
        else:
            return int((value - self.bound[0]) / self.interval)
    

class ContinuousSearchSpace(BaseSearchSpace):
    def __init__(self, bound):
        super().__init__(bound=bound, type='continuous')

    def get_value(self):
        if self.bound[1] > 1:
            int_num = random.randrange(int(self.bound[0]), int(self.bound[1]) + 1)
        else:
            int_num = 0
        while True:
            value = random.random() * self.bound[1]
            value = int_num + value
            if value > self.bound[0] and value < self.bound[1]:
                break
        return value
