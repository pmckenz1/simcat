#!/usr/bin/env python

"""
Pulls a slice from the database and runs simulation to get SNP counts.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

import h5py
import ipcoal
import os
import toytree
import time
import numpy as np
from .utils import SimcatError, Progress
from .parallel import Parallel
import fasteners

import sqlite3
import io


# sqlite register functions to handle np arrays
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

class Simulator:
    """
    This is the object that points to an existing database, extracts some rows, 
    and runs them!
    """
    def __init__(
        self,
        name,
        workdir,
        quiet=False,
        ):

        # database locations
        self.name = name

        # labels data file
        self.labels = os.path.realpath(
            os.path.join(workdir, "{}.labels.h5".format(self.name)))
        # counts data file
        self.counts = os.path.realpath(
            os.path.join(workdir, "{}.counts.h5".format(self.name)))
        # sql counts data file
        self.sqldb = os.path.realpath(
            os.path.join(workdir, "{}.counts.db".format(self.name)))
        self._quiet = quiet

        # store ipcluster information
        self.ipcluster = {
            "cluster_id": "",
            "profile": "default",
            "engines": "Local",
            "quiet": 0,
            "timeout": 60,
            "cores": 0,
            "threads": 1,
            "pids": {},
        }

        self.checkpoint = 0


    def _run(self, nsims, ipyclient, children=[]):
        """
        Sends jobs to parallel engines to run Simulator.run().
        """
        # if outfile exists and not force then find checkpoint
        # ...

        # load-balancer for distributed parallel jobs
        lbview = ipyclient.load_balanced_view()

        # set chunksize based on ncores and stored_labels
        ncores = len(ipyclient)
        self.chunksize = int(np.ceil(nsims / (ncores * 8)))
        #self.chunksize = min(12, self.chunksize)
        self.chunksize = max(4, self.chunksize)
        #self.chunksize = 4

        # designate lock files
        labslock = fasteners.InterProcessLock(self.labels+'.lock')
        #countslock = fasteners.InterProcessLock(self.counts+'.lock')

        labslock.acquire(blocking=True,
            delay=np.random.uniform(0.008, 0.015),
            max_delay=np.random.uniform(0.1, 0.5),
            timeout=60)
        with h5py.File(self.labels,'r+') as i5:
            finished_sims = i5['finished_sims']
            avail = np.where(~np.array(finished_sims).astype(bool))[0]
            sim_idxs = avail[:nsims]
            finished_sims[sim_idxs] = 2  # code of 2 indicates that these have started
        labslock.release()
        # an iterator to return chunked slices of jobs
        jobs = range(0, nsims, self.chunksize)
        njobs = int(np.ceil(nsims / self.chunksize))

        # submit jobs to engines
        rasyncs = {}
        for slice0 in jobs:
            slice1 = min(nsims, slice0 + self.chunksize)
            if slice1 > slice0:
                args = (self.labels, sim_idxs[slice0:slice1], True)
                rasyncs[slice0] = lbview.apply(IPCoalWrapper, *args)

        # catch results as they return and enter into H5 to keep mem low.
        progress = Progress(njobs, "Simulating count matrices", children)
        progress.increment_all(self.checkpoint)
        if not self._quiet:
            progress.display()
        done = self.checkpoint
        try:
            #io5 = h5py.File(self.counts, mode='r+')
            while 1:
                # gather finished jobs
                finished = [i for i, j in rasyncs.items() if j.ready()]

                # iterate over finished list and insert results
                for job in finished:
                    rasync = rasyncs[job]
                    if rasync.successful():

                        # store result
                        done += 1
                        progress.increment_all()

                        # object returns, pull out results
                        res = rasync.get()

                        con = sqlite3.connect(self.sqldb,
                                              timeout=15,
                                              detect_types=sqlite3.PARSE_DECLTYPES)
                        cur = con.cursor()
                        for id_ in range(res.counts.shape[0]):
                            new_arr = res.counts[id_]
                            cur.execute("update counts set arr=? where id={}".format(sim_idxs[job+id_]), (new_arr, ))

                        #countslock.acquire(blocking=True,
                        #                   delay=np.random.uniform(0.008, 0.015),
                        #                   max_delay=np.random.uniform(0.1, 0.5),
                        #                   timeout=120)
                        #with h5py.File(self.counts, mode='r+') as io5:
                        #    for rownum in range(res.counts.shape[0]):
                        #        io5["counts"][sim_idxs[(job+rownum)], :] = res.counts[rownum]
                        #        #io5["counts"][job:job + self.chunksize, :] = res.counts
                        #countslock.release()

                        con.commit()
                        con.close()

                        # free up memory from job
                        del rasyncs[job]

                    else:
                        raise SimcatError(rasync.get())

                # print progress
                progress.increment_time()

                # finished: break loop
                if len(rasyncs) == 0:
                    break
                else:
                    time.sleep(0.5)
            labslock.acquire(blocking=True,
                             delay=np.random.uniform(0.008, 0.015),
                             max_delay=np.random.uniform(0.1, 0.5),
                             timeout=60)
            with h5py.File(self.labels, 'r+') as i5:
                finished_sims = i5['finished_sims']
                finished_sims[sim_idxs] = 1
            labslock.release()

            # on success: close the progress counter
            progress.widget.close()
            print(
                "completed {} simulations in {}."
                .format(nsims, progress.elapsed)
            )

        finally:
            # close the hdf5 handle
            #io5.close()
            pass


    def run(self, nsims=None, force=True, ipyclient=None, show_cluster=False, auto=False):
        pool=Parallel(
            tool=self,
            rkwargs={'nsims': nsims},
            ipyclient=ipyclient,
            show_cluster=show_cluster,
            auto=auto,
            quiet=self._quiet
            )
        pool.wrap_run()


#    def join_queue(self, filename, writedir='.'):
#        with open(os.path.join(writedir, filename+'.queue'), 'a+') as f:
#            f.write(str(os.getpid()))
#            f.write('\n')


#    def first_in_queue(self, filename, writedir='.'):
#        with open(os.path.join(writedir, filename+'.queue'), 'r') as f:
#            first = f.read().split()[0]
#        if str(os.getpid) == first:
#            return(True)
#        else:
#            return(False)

#    def lock_exists(self, filename, writedir='.'):
#        if os.path.exists(os.path.join(writedir, filename+'.lock')):
#            return(True)
#        else:
#            return(False)

#    def lock_file(self, filename, writedir='.'):
#        while 1:
#            time.sleep(np.random.uniform(0,1))
#            is_first = self.first_in_queue(filename, writedir)
#            is_lock = self.lock_exists(filename, writedir)
#            if is_first:
#                if not is_lock:
#                    break
#        with os.path.join(writedir, filename+'.lock') as f:
#            f.write()




class IPCoalWrapper:
    """
    This is the object that runs on the engines by loading data from the HDF5,
    building the msprime simulations calls, and then calling .run() to fill
    count matrices and return them.
    """
    def __init__(self, database_file, idxs, run=True):

        # location of data
        self.database = database_file
        self.idxs = idxs

        # load the slice of data from .labels
        self.load_slice()

        # fill the vector of simulated data for .counts
        if run:
            self.run()


    def load_slice(self):
        """
        Pull data from .labels for use in ipcoal sims
        """
        # open view to the data
        with h5py.File(self.database, 'r') as io5:

            # sliced data arrays
            self.node_Nes = io5["node_Nes"][self.idxs, ...]
            self.admixture = io5["admixture"][self.idxs, ...]
            self.treeheight = io5["treeheight"][self.idxs, ...]
            self.slide_seeds = io5["slide_seeds"][self.idxs]

            # attribute metadata
            self.tree = toytree.tree(io5.attrs["tree"])
            self.tree = self.tree.mod.make_ultrametric()  # imprecision
            self.nsnps = io5.attrs["nsnps"]
            self.ntips = len(self.tree)
            self.node_slide_prop = io5.attrs["node_slide_prop"]

            # store aligned SNPs
            self.nvalues = len(self.idxs)
            self.counts = np.zeros(
                (self.nvalues, self.tree.ntips, self.nsnps), dtype=np.int64)


    def run(self):
        """
        iterate through ipcoal simulations across label values.
        """
        # run simulations
        for idx in range(self.nvalues):
            # shift root height
            tree = self.tree.mod.node_scale_root_height(treeheight=self.treeheight[idx])

            # node slide
            tree = self.tree.mod.node_slider(
                prop=self.node_slide_prop, seed=self.slide_seeds[idx])

            # set Nes default and override on internal nodes with stored vals
            tree = tree.set_node_values("Ne", default=1e5)
            nes = iter(self.node_Nes[idx])
            for node in tree.treenode.traverse():
                #if not node.is_leaf():
                node.Ne = next(nes)

            # get admixture tuples (only supports 1 edge like this right now)
            admix = list()

            for ad in self.admixture[idx]:
                admix.append((
                    int(ad[0]),
                    int(ad[1]),
                    ad[2],
                    ad[3],
                ))

            # build ipcoal Model object
            model = ipcoal.Model(
                tree=tree,
                admixture_edges=admix,
                Ne=None,
                )

            # simulate genealogies and snps
            model.sim_snps(self.nsnps)

            # stack to mat
            #mat = get_snps_count_matrix(tree, model.seqs)

            # store results
            self.counts[idx] = model.seqs


def split_snps_to_chunks(nsnps, nchunks):
    "split nsnps into int chunks for threaded jobs summing to nsnps."
    out = []
    for i in range(nchunks):
        if i == nchunks - 1:
            out.append((nsnps // nchunks) + (nsnps % nchunks))
        else:
            out.append(nsnps // nchunks)
    return out
