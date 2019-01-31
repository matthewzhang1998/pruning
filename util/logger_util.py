import os
import os.path as osp

def get_dir(base_path):
    path = base_path
    ix = -1
    while osp.exists(path):
        ix += 1
        path = base_path + str(ix)

    os.makedirs(path)
    return path

class FileWriter(object):
    def __init__(self, Dir, graph):
        self.f = {}
        self.dir = Dir

        if not osp.exists(Dir):
            os.makedirs(Dir)

    def add_summary(self, dict, i):
        for key in dict:
            if key not in self.f:
                self.f[key] = open(osp.join(self.dir, '{}.csv'.format(key)), 'w+t')

            self.f[key].write(str(i))
            self.f[key].write(',')
            self.f[key].write(str(dict[key]))
            self.f[key].write('\n')
            self.f[key].flush()


