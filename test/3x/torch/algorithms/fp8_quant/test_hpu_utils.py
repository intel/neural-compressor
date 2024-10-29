import habana_frameworks.torch.hpu as hthpu
import habana_frameworks.torch.utils.experimental as htexp

GAUDI = "GAUDI"
GAUDI2 = "GAUDI2"
GAUDI3 = "GAUDI3"


def get_device_type():
    return htexp._get_device_type()


def get_gaudi2_type():
    return htexp.synDeviceType.synDeviceGaudi2


def get_gaudi3_type():
    return htexp.synDeviceType.synDeviceGaudi3


def get_device_name():
    return hthpu.get_device_name()


def is_device(device_name):
    return hthpu.get_device_name() == device_name


def is_gaudi1():
    return is_device(GAUDI)


def is_gaudi2():
    return is_device(GAUDI2)


def is_gaudi3():
    return is_device(GAUDI3)


device_type = [GAUDI2, GAUDI3]
device_type_id = {GAUDI2: get_gaudi2_type(), GAUDI3: get_gaudi3_type()}
