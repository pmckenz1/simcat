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
import numpy as np
from .utils import get_snps_count_matrix
from .utils import get_all_admix_edges, SimcatError, Progress
from .parallel import Parallel


class Simulator:
    """
    This is the object that points to an existing database, extracts some rows, 
    and runs them!
    """
    def __init__(
        self,
        name,
        workdir
        ):

        # database locations
        self.name = name

        # labels data file
        self.labels = os.path.realpath(
            os.path.join(workdir, "{}.labels.h5".format(self.name)))
        # counts data file
        self.counts = os.path.realpath(
            os.path.join(workdir, "{}.counts.h5".format(self.name)))
        


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

        with h5py.File(self.labels,'r+') as i5:
            finished_sims = i5['finished_sims']
            avail = np.where(~np.array(finished_sims).astype(bool))[0]
            sim_idxs = avail[:nsims]
            finished_sims[sim_idxs] = 2  # code of 2 indicates that these have started

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
        progress.increment_all(0)#self.checkpoint)
        if not self._quiet:
            progress.display()
        done = self.checkpoint
        try:
            io5 = h5py.File(self.counts, mode='r+')
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
                        for rownum in range(res.counts.shape[0]):
                            io5["counts"][sim_idxs[(job+rownum)], :] = res.counts[rownum]
                        #io5["counts"][job:job + self.chunksize, :] = res.counts

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

            # on success: close the progress counter
            progress.widget.close()
            print(
                "completed {} simulations in {}."
                .format(self.nstored_labels, progress.elapsed)
            )

        finally:
            # close the hdf5 handle
            io5.close()

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
            self.nsnps = io5.attrs["nsnps"]
            self.ntips = len(self.tree)

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
                prop=0.25, seed=self.slide_seeds[idx])

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
