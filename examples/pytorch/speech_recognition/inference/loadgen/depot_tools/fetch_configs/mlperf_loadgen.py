import sys

import config_util  # pylint: disable=import-error


# This class doesn't need an __init__ method, so we disable the warning
# pylint: disable=no-init
class MLPerfInference(config_util.Config):
  """Basic Config class for the MLPerf Inference load generator."""

  @staticmethod
  def fetch_spec(props):
    solution = {
      'name'     : 'src',
      'url'      : 'https://github.com/mlperf/inference.git',
      'managed'  : False,
    }
    spec = {
      'solutions': [solution]
    }
    if props.get('target_os'):
      spec['target_os'] = props['target_os'].split(',')
    return {
        'type': 'gclient_git',
        'gclient_git_spec': spec,
    }

  @staticmethod
  def expected_root(_props):
    return 'src'


def main(argv=None):
  return MLPerfInference().handle_args(argv)


if __name__ == '__main__':
  sys.exit(main(sys.argv))
