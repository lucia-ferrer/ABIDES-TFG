from collections import Iterable
from datetime import datetime


def grid_generator(grid, id=-1, params=None):
    if params is None:
        params = {}

    if isinstance(id, int):
        id = [id]  # workaround to use ints as reference since ints are non-mutable

    grid = grid.copy()
    params = params.copy()
    if len(grid) == 0:
        if id[0] == 0 or id[0] == -1:
            yield params
            if id[0] == 0:
                id[0] = -2
        else:
            id[0] -= 1

    else:
        k = list(grid)[0]
        vals = grid[k]
        del grid[k]
        if isinstance(vals, Iterable):
            for v in vals:
                params[k] = v
                yield from grid_generator(grid, id, params)
        else:
            params[k] = vals
            yield from grid_generator(grid, id, params)


class Logger:
    def __init__(self, name, total, length=50):
        self.path = f"results/log/{name}.log"
        self.total = total
        self.i = 0
        self.length = length
        self.write()
        self.start = datetime.now()

    def __call__(self):
        self.i += 1
        self.write()

    def write(self):
        filled = int(self.length*self.i/self.total) if self.total != 0 else 1
        with open(self.path, 'w') as f:
            f.write('█'*filled+'░'*(self.length-filled))
            f.write(f"  {self.i} / {self.total}")
            if self.i > 0:
                delta = datetime.now() - self.start
                f.write(f" | elapsed: {str(delta)}")
                if self.i == self.total:
                    f.write(" Done!")
                else:
                    f.write(f" estimated left: {str((self.total-self.i) * delta / self.i)}")
