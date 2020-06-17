from abc import abstractmethod
import time
import tracemalloc

'''The objectives supported by iLit, which is driven by accuracy.
   To support new objective, developer just need implement a new subclass in this file.
'''
OBJECTIVES = {}

def objective_registry(cls):
    '''The class decorator used to register all Objective subclasses.

       Args:
           cls (class): The class of register.
    '''
    if cls.__name__.lower() in OBJECTIVES:
        raise ValueError('Cannot have two objectives with the same name')
    OBJECTIVES[cls.__name__.lower()] = cls
    return cls

class Objective(object):
    '''The base class of objectives supported by iLit.

       Args:
           accuracy_criterion (dict): The dict of supported accuracy criterion.
                                      {'relative': 0.01} or {'absolute': 0.01}
    '''
    def __init__(self, accuracy_criterion):
        assert isinstance(accuracy_criterion, dict) and len(accuracy_criterion)
        k, v = list(accuracy_criterion.items())[0]
        assert k in ['relative', 'absolute']
        assert float(v) < 1 and float(v) > -1

        self.acc_goal = float(v)
        self.relative = True if k == 'relative' else False
        self.baseline = None
        self.val = None

    @abstractmethod
    def compare(self, last):
        '''The interface of comparing if metric reaches the goal with acceptable accuracy loss.

           Args:
               last (tuple): The tuple of last metric.
               accuracy_criterion (float): The allowed accuracy absolute loss.
        '''
        raise notimplementederror

    @abstractmethod
    def evaluate(self, eval_func, model, baseline=False):
        '''The interface of calculating the objective.

        '''
        raise notimplementederror

@objective_registry
class Performance(Objective):
    '''The class of calculating topk metric, which usually is used in classification.

       Args:
           predict (tensor): The tensor of predict result.
           label (tensor): The tensor of ground truth.
           topk (integer): The top k number interested.
    '''
    def __init__(self, accuracy_criterion):
        super(Performance, self).__init__(accuracy_criterion)

    def compare(self, last):
        acc, perf = self.val

        if last != None:
            _, last_perf = last
        else:
            last_perf = 0

        assert self.baseline, "baseline of Objective class should be set before reference."
        base_acc, base_perf = self.baseline

        acc_target = base_acc - float(self.acc_goal) if not self.relative else base_acc * (1 - float(self.acc_goal))
        if acc >= acc_target and (last_perf == 0 or perf < last_perf):
            return True
        else:
            return False

    def evaluate(self, eval_func, model, baseline):
        start = time.time()
        accuracy = eval_func(model)
        end = time.time()
        total_time = end - start

        if baseline:
            self.baseline = accuracy, total_time

        self.val = accuracy, total_time
        return self.val

@objective_registry
class Footprint(Objective):
    '''The objective class of calculating peak memory footprint when running quantize model.

    '''
    def __init__(self, accuracy_criterion):
        super(Footprint, self).__init__(accuracy_criterion)

    def compare(self, last):
        acc, peak = self.val

        if last != None:
            _, last_peak = last
        else:
            last_peak = 0

        assert self.baseline, "baseline variable of Objective class should be set before reference."
        base_acc, base_peak = baseline

        acc_target = base_acc - float(self.acc_goal) if not self.relative else base_acc * (1 - float(self.acc_goal))
        if acc >= acc_target and (last_peak == 0 or peak < last_peak):
            return True
        else:
            return False

    def evaluate(self, eval_func, model, baseline):
        tracemalloc.start()
        accuracy = eval_func(model)
        _, peak =  tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if baseline:
            self.baseline = accuracy, peak

        self.val = accuracy, peak
        return self.val

@objective_registry
class ModelSize(Objective):
    '''The class of calculating quanitzed model size.

    '''
    def __init__(self, accuracy_criterion):
        super(ModelSize, self).__init__(accuracy_criterion)

    def compare(self, baseline, accuracy_criterion):
        pass

    def evaluate(self, eval_func, model):
        pass
