import os
import re
import subprocess

from neural_compressor.common.utils import DEFAULT_WORKSPACE

tmp = """
# build file:tmp.py during test process to test benchmark
print("test benchmark")
"""
tmp_path = "./tmp.py"


def build_tmp_file():
    f = open(tmp_path, "w")
    f.write(tmp)
    f.close()


def trigger_process(cmd):
    # trigger subprocess
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )  # nosec
    return p


def check_main_process(message):
    # ASCII: \x1b\[1m\x1b\[91m is bold, red.
    num_i_pattern = r"(.*) \x1b\[1m\x1b\[91m(\d+) instance(.*) triggered"
    num_c_pattern = r"(.*) \x1b\[1m\x1b\[91m(\d+) core(.*) in use"
    log_file_pattern = r"(.*) The log of instance 1 is saved to (.*)"
    num_i = re.search(num_i_pattern, message, flags=re.DOTALL).group(2)
    all_c = re.search(num_c_pattern, message).group(2)
    log_file_path = re.search(log_file_pattern, message).group(2)
    return int(num_i), int(all_c), log_file_path


def check_log_file(log_file_path):
    output_pattern = r"(.*)test benchmark"
    with open(log_file_path, "r") as f:
        output = f.read()
    f.close()
    return re.match(output_pattern, output, flags=re.DOTALL)


class TestBenchmark:
    def setup_class(self):
        build_tmp_file()

    def teardown_class(self):
        os.remove(tmp_path)

    def test_default(self):
        cmd = "incbench tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == 1, "the number of instance should be 1."
        assert check_log_file(log_file_path), "instance output is not correct."

    def test_only_num_i(self):
        cmd = "incbench --num_i 2 tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == 2, "the number of instance should be 2."
        assert check_log_file(log_file_path), "instance output is not correct."

    def test_only_num_c(self):
        cmd = "incbench --num_c 1 tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == all_c, "the number of instance should equal the number of available cores."
        assert check_log_file(log_file_path), "instance output is not correct."

    def test_only_cores(self):
        cmd = "incbench -C 0-1 tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == 1, "the number of instance should be 1."
        assert all_c == 2, "the number of available cores should be 2."
        assert check_log_file(log_file_path), "instance output is not correct."

    def test_num_i_num_c(self):
        cmd = "incbench --num_i 2 --num_c 2 tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == 2, "the number of instance should be 2."
        assert all_c == 4, "the number of available cores should be 4."
        assert check_log_file(log_file_path), "instance output is not correct."

    def test_num_i_cores(self):
        cmd = "incbench --num_i 2 -C 1-5 tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == 2, "the number of instance should be 2."
        assert all_c == 5, "the number of available cores should be 5."
        assert check_log_file(log_file_path), "instance output is not correct."

    def test_num_c_cores(self):
        cmd = "incbench --num_c 2 -C 0-6 tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == 3, "the number of instance should be all_c//num_c=3."
        assert all_c == 6, "the number of available cores should be (all_c//num_c)*num_c=6."
        assert check_log_file(log_file_path), "instance output is not correct."

    def test_num_i_num_c_cores(self):
        cmd = "incbench --num_i 2 --num_c 2 -C 0-7 tmp.py"
        p = trigger_process(cmd)
        stdout, _ = p.communicate()
        num_i, all_c, log_file_path = check_main_process(stdout.decode())
        assert num_i == 2, "the number of instance should be 2."
        assert all_c == 4, "the number of available cores should be num_i*num_c=4."
        assert check_log_file(log_file_path), "instance output is not correct."
