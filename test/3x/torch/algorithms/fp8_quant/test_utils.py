import habana_frameworks.torch.hpu as hthpu


def is_device(device_name):
    return hthpu.get_device_name() == device_name


def is_gaudi1():
    return is_device("GAUDI")


def is_gaudi2():
    return is_device("GAUDI2")


def is_gaudi3():
    return is_device("GAUDI3")
