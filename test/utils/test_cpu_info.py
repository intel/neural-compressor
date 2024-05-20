from neural_compressor.utils.utility import CpuInfo


class TestCPUInfo:
    def test_get_cpu_info(self):
        cpu_info = CpuInfo()
        assert cpu_info.cores >= 1
        assert cpu_info.sockets >= 1
        assert cpu_info.cores_per_socket >= 1
