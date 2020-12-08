import pathlib

import h5py
import numpy as np
import pyvista as pv


def save_vtk_file(file_name, data: np.ndarray):
    grid = pv.UniformGrid()
    grid.dimensions = data.shape
    grid.origin = - np.asarray(data.shape) // 2
    grid.spacing = (1, 1, 1)
    grid.point_arrays["values"] = data.flatten(order='F')
    grid.save(file_name)


if __name__ == "__main__":
    data_root = "."
    p = pathlib.Path(data_root)
    patterns = [{'mask': 'tomo*.h5', 'dataset': 'Reconstruction'}]
    for pattern in patterns:
        for f in p.glob(pattern['mask']):
            print(f.name)
            with h5py.File(f, 'r') as h5f:
                data = h5f[pattern['dataset']][()]
                save_vtk_file(p / f'{f.name[:-3]}.vtk', data)
