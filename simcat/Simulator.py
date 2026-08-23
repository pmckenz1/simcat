#!/usr/bin/env python

"""
Pulls a slice from the database and runs simulation to get SNP counts.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

import h5py
import ipcoal
import msprime as ms
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
    return np.load(out, allow_pickle=False)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

class Simulator:
    """
    This is the object that points to an existing database, extracts some rows,
    and runs them!
    """
    PENDING = 0
    COMPLETE = 1
    RESERVED = 2

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


    def _labels_lock(self):
        """Acquire and return the inter-process labels-file lock."""
        lock = fasteners.InterProcessLock(self.labels + ".lock")
        acquired = lock.acquire(
            blocking=True,
            delay=np.random.uniform(0.008, 0.015),
            max_delay=np.random.uniform(0.1, 0.5),
            timeout=60,
        )
        if not acquired:
            raise SimcatError(
                "Timed out waiting for the simulation reservation lock: "
                f"{self.labels}.lock"
            )
        return lock


    def status(self):
        """Return counts of pending, complete, and reserved simulations."""
        if not os.path.exists(self.labels):
            raise FileNotFoundError(f"Labels database not found: {self.labels}")
        lock = self._labels_lock()
        try:
            with h5py.File(self.labels, "r") as io5:
                states = np.asarray(io5["finished_sims"])
        finally:
            lock.release()
        return {
            "total": int(states.size),
            "pending": int(np.count_nonzero(states == self.PENDING)),
            "complete": int(np.count_nonzero(states == self.COMPLETE)),
            "reserved": int(np.count_nonzero(states == self.RESERVED)),
        }


    def recover(self):
        """Release reservations left by interrupted workers.

        Call this only after confirming that no other process is currently
        simulating rows from this database. Normal Python task failures release
        their own reservations automatically; this method is for hard process or
        scheduler interruptions.
        """
        lock = self._labels_lock()
        try:
            with h5py.File(self.labels, "r+") as io5:
                states = io5["finished_sims"]
                reserved = np.where(
                    np.asarray(states) == self.RESERVED
                )[0]
                if reserved.size:
                    states[reserved] = self.PENDING
        finally:
            lock.release()
        if not self._quiet:
            print(f"Released {reserved.size} interrupted simulation rows.")
        return int(reserved.size)


    def _reserve_simulations(self, nsims):
        """Atomically reserve at most ``nsims`` currently pending rows."""
        if nsims is not None and (int(nsims) != nsims or nsims <= 0):
            raise ValueError("nsims must be a positive integer or None")

        lock = self._labels_lock()
        try:
            with h5py.File(self.labels, "r+") as io5:
                states = io5["finished_sims"]
                available = np.where(
                    np.asarray(states) == self.PENDING
                )[0]
                requested = available.size if nsims is None else int(nsims)
                selected = available[:requested]
                if selected.size:
                    states[selected] = self.RESERVED
        finally:
            lock.release()

        if (
            nsims is not None
            and selected.size < int(nsims)
            and not self._quiet
        ):
            print(
                f"Requested {int(nsims)} simulations; reserved the "
                f"{selected.size} pending rows that remain."
            )
        return selected


    def _set_simulation_status(self, idxs, status, only_if=None):
        """Atomically update status for row IDs, optionally conditionally."""
        idxs = np.asarray(idxs, dtype=int)
        if not idxs.size:
            return 0
        lock = self._labels_lock()
        try:
            with h5py.File(self.labels, "r+") as io5:
                states = io5["finished_sims"]
                if only_if is None:
                    selected = idxs
                else:
                    current = np.asarray(states[idxs])
                    selected = idxs[current == only_if]
                if selected.size:
                    states[selected] = status
        finally:
            lock.release()
        return int(selected.size)


    def _run(self, nsims, ipyclient, children=None):
        """
        Sends jobs to parallel engines to run Simulator.run().
        """
        # if outfile exists and not force then find checkpoint
        # ...

        for path, label in (
            (self.labels, "labels HDF5"),
            (self.sqldb, "counts SQLite"),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} database not found: {path}")

        sim_idxs = self._reserve_simulations(nsims)
        nsims_reserved = len(sim_idxs)
        if not nsims_reserved:
            if not self._quiet:
                print("No pending simulations remain.")
            return 0

        children = [] if children is None else children
        progress = None
        try:
            # load-balancer for distributed parallel jobs
            lbview = ipyclient.load_balanced_view()

            # set chunksize based on ncores and stored_labels
            ncores = len(ipyclient)
            if ncores < 1:
                raise SimcatError("No ipyparallel engines are connected.")
            self.chunksize = int(np.ceil(nsims_reserved / (ncores * 8)))
            self.chunksize = max(4, self.chunksize)

            # submit jobs to engines
            rasyncs = {}
            for slice0 in range(0, nsims_reserved, self.chunksize):
                slice1 = min(nsims_reserved, slice0 + self.chunksize)
                row_ids = sim_idxs[slice0:slice1]
                args = (self.labels, row_ids, True)
                rasyncs[slice0] = (
                    lbview.apply(IPCoalWrapper, *args),
                    row_ids,
                )

            # catch results as they return and enter into SQLite to keep memory
            # use low in the parent process.
            njobs = len(rasyncs)
            progress = Progress(
                njobs, "Simulating count matrices", children
            )
            progress.increment_all(self.checkpoint)
            if not self._quiet:
                progress.display()

            #io5 = h5py.File(self.counts, mode='r+')
            while 1:
                # gather finished jobs
                finished = [
                    key for key, (job, _) in rasyncs.items() if job.ready()
                ]

                # iterate over finished list and insert results
                for job in finished:
                    rasync, row_ids = rasyncs[job]
                    if rasync.successful():

                        # store result
                        progress.increment_all()

                        # object returns, pull out results
                        res = rasync.get()
                        timeout_time = 900
                        con = sqlite3.connect(
                            self.sqldb,
                            timeout=timeout_time,
                            detect_types=sqlite3.PARSE_DECLTYPES,
                        )
                        try:
                            cur = con.cursor()
                            if res.counts.shape[0] != len(row_ids):
                                raise SimcatError(
                                    "Simulation worker returned an unexpected "
                                    "number of rows for IDs "
                                    f"{row_ids.tolist()}."
                                )
                            for id_, new_arr in zip(row_ids, res.counts):
                                cur.execute(
                                    "update counts set arr=? where id=?",
                                    (new_arr, int(id_)),
                                )

                            con.commit()
                        finally:
                            con.close()

                        # Mark each chunk complete immediately after its counts
                        # transaction commits. This preserves completed work if a
                        # later chunk fails or the parent process is interrupted.
                        self._set_simulation_status(
                            row_ids, self.COMPLETE, only_if=self.RESERVED
                        )

                        # free up memory from job
                        del rasyncs[job]

                    else:
                        try:
                            rasync.get()
                        except Exception as exc:
                            raise SimcatError(
                                "Simulation failed for row IDs "
                                f"{row_ids.tolist()}: {exc}"
                            ) from exc
                        raise SimcatError(
                            f"Simulation failed for row IDs {row_ids.tolist()}."
                        )

                # print progress
                progress.increment_time()

                # finished: break loop
                if len(rasyncs) == 0:
                    break
                else:
                    time.sleep(0.5)
            # on success: close the progress counter
            progress.widget.close()
            if not self._quiet:
                print(
                    "completed {} simulations in {}."
                    .format(nsims_reserved, progress.elapsed)
                )
            return nsims_reserved

        finally:
            # Any row not already committed is safe to retry. Conditional
            # updates avoid reverting chunks that completed successfully.
            self._set_simulation_status(
                sim_idxs, self.PENDING, only_if=self.RESERVED
            )
            if progress is not None:
                progress.widget.close()


    def run(
        self,
        nsims=None,
        force=True,
        ipyclient=None,
        show_cluster=False,
        auto=False,
        recover=False,
    ):
        """Run pending simulations through an ipyparallel client.

        ``force`` is retained for API compatibility and has no effect. Set
        ``recover=True`` only when no other process is using the database to
        release reservations left by a hard interruption.
        """
        del force
        if recover:
            self.recover()
        pool=Parallel(
            tool=self,
            rkwargs={'nsims': nsims},
            ipyclient=ipyclient,
            show_cluster=show_cluster,
            auto=auto,
            quiet=self._quiet
            )
        return pool.wrap_run()


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
            self.slide_seeds = io5["slide_seeds"][self.idxs, ...]

            # attribute metadata
            self.tree = toytree.tree(io5.attrs["tree"])
            self.tree = self.tree.mod.edges_extend_tips_to_align()  # imprecision
            self.nsnps = io5.attrs["nsnps"]
            self.rate_vector = io5.attrs["rate_vector"]
            self.pi_vector = io5.attrs["pi_vector"]
            self.ntips = self.tree.ntips
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
            tree = self.tree.mod.edges_scale_to_root_height(treeheight=self.treeheight[idx])

            # node slide
            tree = tree.mod.edges_slider(
                prop=self.node_slide_prop, seed=self.slide_seeds[idx])

            # set Nes default and override on internal nodes with stored vals
            tree = tree.set_node_data("Ne", default=1e5)
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

            # define the model if there is one...
            if not np.any(np.isnan(self.rate_vector)):
                subst_model = ms.GTR(self.rate_vector, self.pi_vector)
            else:
                subst_model = 'JC69'


            # build ipcoal Model object
            model = ipcoal.Model(
                tree=tree,
                admixture_edges=admix,
                Ne=None,
                subst_model=subst_model,
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
