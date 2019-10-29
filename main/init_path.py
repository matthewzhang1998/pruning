import os.path as osp
import sys
import datetime

def add_path(path):
    path = path.replace('/scratch/ssd001', '')
    if path not in sys.path:
        print(path)
        sys.path.insert(0, path)


real_file_path = osp.realpath(__file__.replace('pyc', 'py'))
_this_dir = osp.dirname(real_file_path)

running_start_time = datetime.datetime.now()
time = str(running_start_time.strftime("%Y_%m_%d-%X"))

_base_dir = osp.join(_this_dir, '..')
print(_base_dir)
add_path(_base_dir)
add_path("~/anaconda3/lib")

print(sys.path)


def bypass_frost_warning():
    return 0


def get_base_dir():
    return _base_dir


def get_time():
    return time


def get_abs_base_dir():
    return osp.abspath(_base_dir)
