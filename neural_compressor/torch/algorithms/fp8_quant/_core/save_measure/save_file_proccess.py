import sys

from ..measure import save_measurements

if __name__ == "__main__":
    model = sys.argv[0]
    save_measurements(model)