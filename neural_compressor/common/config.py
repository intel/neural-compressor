class BaseConfig():
  def __init__(self):
      self.scope = {}

  def __add__(self, obj):
      self.scope = {**self.scope, **obj.scope}

class FP32Config(BaseConfig):
  def __init__(self, scope):
      super().__init__()
      self.scope = {'fp32': scope}

class FP8QuantConfig(BaseConfig):
  def __init__(self, scheme, scope):
      super().__init__()
      assert(scheme in ['fp8_e4m3', 'fp8_e3m4', 'fp8_e5m2']), "The FP8 configuration is wrong! Please double check."
      self.scope = {scheme: scope}

