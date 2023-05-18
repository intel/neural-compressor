import unittest
from unittest.mock import patch
import threading
import os
import argparse
import shutil
from neural_solution.backend.Runner import parse_args, main

class TestMain(unittest.TestCase):
    
    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('test.txt')
        shutil.rmtree("ns_workspace")

    def test_parse_args(self):
        args = ['-H', 'path/to/hostfile', '-TMP', '2222', '-RMP', '3333', '-CEN', 'inc']
        with patch('argparse.ArgumentParser.parse_args', \
            return_value=argparse.Namespace(hostfile='path/to/hostfile', \
                task_monitor_port=2222, result_monitor_port=3333, conda_env_name='inc')):
            self.assertEqual(parse_args(args), \
                argparse.Namespace(hostfile='path/to/hostfile', \
                    task_monitor_port=2222, result_monitor_port=3333, conda_env_name='inc'))

    def test_main(self):
        """Test blocking flag in abort_job method."""
        path = "test.txt"
        with open(path, "w") as f:
            f.write("hostname1\nhostname2")
        adding_abort = threading.Thread(
            target=main,
            kwargs={'args': ['-H', 'test.txt', '-TMP', '2222', '-RMP', '3333', '-CEN', 'inc_conda_env']},
            daemon=True,
        )
        adding_abort.start()
        adding_abort.join(timeout=2)
        
if __name__ == '__main__':
    unittest.main()