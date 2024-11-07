# -----------------------------------------------------------------------------
# Copyright (c) 2024, Argonne National Laboratory.
#
# Distributed under the terms of an Open Source License.
#
# The full license is in the file LICENSE.pdf, distributed with this software.
# -----------------------------------------------------------------------------

# Original version at:
# https://anl.app.box.com/s/waqffd2sg5aospk1o6tik7rfibew0zmm

"""
NX SYMMETRY
"""

import os
import sys
import tempfile
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, resource_tracker

import numpy as np
from nexusformat.nexus import nxopen, nxgetconfig, nxsetconfig, NXentry, NXdata


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("transform_file", help="Input")
    parser.add_argument("symm_file", help="Output")
    parser.add_argument("-n", type=int, default=2,
                        help="Parallelism")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with nxopen(args.transform_file, 'r') as transform_root:
        data = transform_root['entry/data/v']

    with nxopen(args.symm_file, 'w') as symm_root:
        symm_root['entry'] = NXentry()
        symm_root['entry/data'] = NXdata()
        symmetry = NXSymmetry(data, laue_group='m-3m')
        symm_root['entry/data/data'] = \
            symmetry.symmetrize(parallelism=args.n)


def report(label, start, stop):
    label += ":"
    print("%-14s %8.3f" % (label, stop - start))


def triclinic(data):
    """Laue group: -1"""
    outarr = np.nan_to_num(data)
    outarr += np.flip(outarr)
    return outarr


def monoclinic(data):
    """Laue group: 2/m"""
    outarr = np.nan_to_num(data)
    outarr += np.rot90(outarr, 2, (0, 2))
    outarr += np.flip(outarr, 1)
    return outarr


def orthorhombic(data):
    """Laue group: mmm"""
    outarr = np.nan_to_num(data)
    outarr += np.flip(outarr, 0)
    outarr += np.flip(outarr, 1)
    outarr += np.flip(outarr, 2)
    return outarr


def tetragonal1(data):
    """Laue group: 4/m"""
    outarr = np.nan_to_num(data)
    outarr += np.rot90(outarr, 1, (1, 2))
    outarr += np.rot90(outarr, 2, (1, 2))
    outarr += np.flip(outarr, 0)
    return outarr


def tetragonal2(data):
    """Laue group: 4/mmm"""
    outarr = np.nan_to_num(data)
    outarr += np.rot90(outarr, 1, (1, 2))
    outarr += np.rot90(outarr, 2, (1, 2))
    outarr += np.rot90(outarr, 2, (0, 1))
    outarr += np.flip(outarr, 0)
    return outarr


def hexagonal(data):
    """Laue group: 6/m, 6/mmm (modeled as 2/m along the c-axis)"""
    outarr = np.nan_to_num(data)
    outarr += np.rot90(outarr, 2, (1, 2))
    outarr += np.flip(outarr, 0)
    return outarr


def cubic(data):
    """Laue group: m-3 or m-3m"""
    print("cubic start")
    sys.stdout.flush()

    cubic_start = time.time()

    start = time.time()
    outarr = np.nan_to_num(data)
    stop = time.time()
    report("nan_to_num", start, stop)

    start = time.time()
    outarr += np.transpose(outarr, axes=(1, 2, 0))
    stop = time.time()
    report("transpose1", start, stop)

    start = time.time()
    outarr += np.transpose(outarr, axes=(2, 0, 1))
    stop = time.time()
    report("transpose2", start, stop)

    start = time.time()
    outarr += np.transpose(outarr, axes=(0, 2, 1))
    stop = time.time()
    report("transpose3", start, stop)

    start = time.time()
    outarr += np.flip(outarr, 0)
    stop = time.time()
    report("flip1", start, stop)

    start = time.time()
    outarr += np.flip(outarr, 1)
    stop = time.time()
    report("flip2", start, stop)

    start = time.time()
    outarr += np.flip(outarr, 2)
    stop = time.time()
    report("flip3", start, stop)

    stop = time.time()
    report("cubic time", cubic_start, stop)
    return outarr


def symmetrize_entries(symm_function, data_type, data_file, data_path):
    """
    Symmetrize data from multiple entries in a file.

    Parameters
    ----------
    symm_function : function
        The function to use for symmetrizing the data.
    data_type : str
        The type of data to symmetrize; either 'signal' or 'weights'.
    data_file : str
        The name of the file containing the data to symmetrize.
    data_path : str
        The path to the data in the file.

    Returns
    -------
    data_type : str
        The type of data that was symmetrized.
    filename : str
        The name of the file containing the symmetrized data.
    """
    nxsetconfig(lock=3600, lockexpiry=28800)
    with nxopen(data_file, 'r') as data_root:
        data_path = os.path.basename(data_path)
        for i, entry in enumerate([e for e in data_root if e[-1].isdigit()]):
            data_size = int(
                data_root[entry][data_path].nxsignal.nbytes / 1e6) + 1000
            nxsetconfig(memory=data_size)
            if i == 0:
                if data_type == 'signal':
                    data = data_root[entry][data_path].nxsignal.nxvalue
                elif data_root[entry][data_path].nxweights:
                    data = data_root[entry][data_path].nxweights.nxvalue
                else:
                    signal = data_root[entry][data_path].nxsignal.nxvalue
                    data = np.zeros(signal.shape, dtype=signal.dtype)
                    data[np.where(signal > 0)] = 1
            else:
                if data_type == 'signal':
                    data += data_root[entry][data_path].nxsignal.nxvalue
                elif data_root[entry][data_path].nxweights:
                    data += data_root[entry][data_path].nxweights.nxvalue
    result = symm_function(data)
    with nxopen(tempfile.mkstemp(suffix='.nxs')[1], mode='w') as root:
        root['data'] = result
    return data_type, root.nxfilename


def symmetrize_data(symm_function, data_type, data_file, data_path):
    """
    Symmetrize data from a single entry in a file.

    Parameters
    ----------
    symm_function : function
        The function to use for symmetrizing the data.
    data_type : str
        The type of data to symmetrize; either 'signal' or 'weights'.
    data_file : str
        The name of the file containing the data to symmetrize.
    data_path : str
        The path to the data in the file.

    Returns
    -------
    data_type : str
        The type of data that was symmetrized.
    filename : str
        The name of the file containing the symmetrized data.
    """
    start = time.time()
    nxsetconfig(lock=3600, lockexpiry=28800)
    with nxopen(data_file, 'r') as data_root:
        data_size = int(data_root[data_path].nbytes / 1e6) + 1000
        nxsetconfig(memory=data_size)
        if data_type == 'signal':
            data = data_root[data_path].nxvalue
        else:
            signal = data_root[data_path].nxvalue
            data = np.zeros(signal.shape, signal.dtype)
            data[np.where(signal > 0)] = 1
    stop = time.time()
    report("startup", start, stop)

    start = time.time()
    result = symm_function(data)
    stop = time.time()
    report("compute", start, stop)

    start = time.time()
    nx_tmp = tempfile.mkstemp(suffix='.nxs')[1]
    print("nx_tmp: '%s'" % nx_tmp)
    with nxopen(nx_tmp, mode='w') as root:
        root['data'] = result
    stop = time.time()
    report("write", start, stop)

    return data_type, root.nxfilename


laue_functions = {'-1': triclinic,
                  '2/m': monoclinic,
                  'mmm': orthorhombic,
                  '4/m': tetragonal1,
                  '4/mmm': tetragonal2,
                  '-3': triclinic,
                  '-3m': triclinic,
                  '6/m': hexagonal,
                  '6/mmm': hexagonal,
                  'm-3': cubic,
                  'm-3m': cubic}


class NXSymmetry:

    def __init__(self, data, laue_group=None):
        """
        Parameters
        ----------
        data : NXdata
            The data to be symmetrized.
        laue_group : str, optional
            The Laue group of the crystal structure of the sample. If not
            specified, a triclinic crystal structure is assumed.

        Attributes
        ----------
        symm_function : function
            The function to use for symmetrizing the data.
        data_file : str
            The name of the file containing the data to symmetrize.
        data_path : str
            The path to the data in the file.
        """
        if laue_group and laue_group in laue_functions:
            print("using laue function: " + laue_group)
            self.symm_function = laue_functions[laue_group]
        else:
            print("using default laue function triclinic")
            self.symm_function = triclinic
        self.data_file = data.nxfilename
        self.data_path = data.nxpath

    def symmetrize(self, entries=False, parallelism=2):
        """
        Symmetrize the data.

        Parameters
        ----------
        entries : bool, optional
            Flag to indicate whether to symmetrize multiple entries in a file,
            by default False.

        Returns
        -------
        array-like
            The symmetrized data.
        """
        print("symmetrize: entries=" + str(entries))
        if entries:
            symmetrize = symmetrize_entries
        else:
            symmetrize = symmetrize_data
        with NXExecutor(max_workers=parallelism) as executor:
            futures = []
            for data_type in ['signal', 'weights']:
                futures.append(executor.submit(
                    symmetrize, self.symm_function, data_type,
                    self.data_file, self.data_path))
        weights = None
        signal = None
        for future in as_completed(futures):
            data_type, result_file = future.result()
            with nxopen(result_file, 'r') as result_root:
                data_size = result_root['data'].nbytes / 1e6
                if data_size > nxgetconfig('memory'):
                    nxsetconfig(memory=data_size+1000)
                if data_type == 'signal':
                    signal = result_root['data'].nxvalue
                else:
                    weights = result_root['data'].nxvalue
            os.remove(result_file)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(weights > 0, signal / weights, 0.0)
        return result


class NXExecutor(ProcessPoolExecutor):
    """ProcessPoolExecutor class using 'spawn' for new processes."""

    def __init__(self, max_workers=None, mp_context='spawn'):
        # print("NXExecutor: %i workers" % max_workers)
        if mp_context:
            mp_context = get_context(mp_context)
        else:
            mp_context = None
        super().__init__(max_workers=max_workers, mp_context=mp_context)

    def __repr__(self):
        return f"NXExecutor(max_workers={self._max_workers})"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        if self._mp_context.get_start_method(allow_none=False) != 'fork':
            resource_tracker._resource_tracker._stop()
        return False


if __name__ == '__main__':
    main()
