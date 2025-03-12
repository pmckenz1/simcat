#!/usr/bin/env python

"Parallelization with ipyparallel following ipyrad framework"

from __future__ import print_function
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import os
import sys
import time
import socket
import subprocess
from random import getrandbits

import ipyparallel as ipp
from ipywidgets import HTML, Box
from IPython.display import display

from .utils import SimcatError


IPCLUSTERBIN = os.path.join(sys.prefix, "bin", "ipcluster")



class Parallel(object):
    """
    Connect or launch ipcluster and wrap jobs running on Client engines so 
    that engines can be interrupted or killed with a pleasant cleanup.
    """
    def __init__(self, tool, rkwargs=None, ipyclient=None, show_cluster=True, quiet=False, auto=False):

        # if no kwargs then empty dict
        if rkwargs is None:
            rkwargs = {}

        self._quiet = quiet

        # the tool with a ._run() func and its run kwargs to be parallelized
        self.tool = tool
        self.rkwargs = rkwargs

        # parallel client connect or launch params
        self.ipyclient = ipyclient
        self.show_cluster = show_cluster
        self.auto = auto

        # setup the widget message to be passed on to _run 
        self.message = HTML(
            layout={"height": "25px", "margin": "0px"})
        self.widget = Box(
            children=[self.message], 
            layout={"margin": "5px 0px 5px 0px"})

        self.update_message("Establishing parallel connection: ...")

        if not self._quiet:
            # show the widget message
            display(self.widget)


    def update_message(self, inner):
        s1 = "<span style='font-size:14px; font-family:monospace'>"
        s2 = "</span>"
        self.message.value = s1 + inner + s2


    def start_ipcluster(self):
        """
        The name is a unique id that keeps this __init__ of ipyrad distinct
        from interfering with other ipcontrollers. Run statements are wrapped
        so that ipcluster SHOULD be killed on exit.
        """
        # use random num for to cluster_id
        rand = getrandbits(32)
        self.tool.ipcluster["cluster_id"] = "ipp-{}".format(rand)

        # if engines=="MPI" then add --ip arg to view all sockets  
        iparg = ("--ip=*" if "MPI" in self.tool.ipcluster["engines"] else "")

        # make ipcluster arg call
        standard = [
            IPCLUSTERBIN, "start",
            "--daemonize", 
            "--cluster-id={}".format(self.tool.ipcluster["cluster_id"]),
            "--engines={}".format(self.tool.ipcluster["engines"]),
            "--profile={}".format(self.tool.ipcluster["profile"]),
            "--n={}".format(self.tool.ipcluster["cores"]),
            "{}".format(iparg),
        ]

        # wrap ipcluster start
        try:
            subprocess.check_call(
                standard,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE)

        # if cluster with THIS ID is running then kill it and try again
        except subprocess.CalledProcessError:
            subprocess.check_call([
                IPCLUSTERBIN, "stop", 
                "--cluster-id", self.tool.ipcluster["cluster_id"],
            ], 
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
            )

            # after waiting try again to start it
            time.sleep(3)
            try:
                subprocess.check_call(
                    standard, 
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE)

            # if fails again then report it
            except subprocess.CalledProcessError as inst:
                print(inst)
                raise

        except Exception as inst:
            sys.exit(
                "Error launching ipcluster for parallelization:\n({})\n"
                .format(inst))



    def wait_for_connection(self):
        """ 
        Creates a client to view ipcluster engines for a given profile and 
        returns it with at least one engine spun up and ready to go. If no 
        engines are found after nwait amount of time then an error is raised.
        If engines==MPI it waits a bit longer to find engines. If the number
        of engines is set then it waits even longer to try to find that number
        of engines.
        """
        # save stds for later, hide here to prevent ipp enforced print()
        save_stdout = sys.stdout 
        save_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # wrapped search for ipcluster
        try: 
            args = {
                "profile": self.tool.ipcluster["profile"],
                "timeout": self.tool.ipcluster["timeout"],
                "cluster_id": self.tool.ipcluster["cluster_id"],            
            }
            ipyclient = ipp.Client(**args)

            # restore std printing now that Client print statement has passed
            # sys.stdout = save_stdout
            # sys.stderr = save_stderr

            # allow time to find the connection; count cores to break
            for _ in range(6000):

                # how many cores can we find right now?
                ncores = len(ipyclient)
                # self.update_message(
                # "Establishing parallel connection: {} cores"
                # .format(ncores))
                time.sleep(0.01)

                # are all cores found yet? if so, break.
                if self.tool.ipcluster["cores"]:
                    time.sleep(0.1)
                    if ncores == self.tool.ipcluster["cores"]:
                        break

                # If MPI and not all found, break if no more found in 3 secs
                elif self.tool.ipcluster["engines"] == "MPI":
                    # are any cores found yet? do long wait.
                    if ncores:
                        time.sleep(1)
                        if len(ipyclient) == ncores:
                            break

                # if Local then if at least one core we're happy to move on.
                else:
                    if ncores:
                        time.sleep(0.5)
                        break

        except KeyboardInterrupt as inst:
            raise inst

        except (IOError, OSError, ipp.TimeoutError, ipp.NoEnginesRegistered):
            raise SimcatError(
                "\nipcluster not found, use 'auto=True' or see docs.")

        finally:
            # no matter what we reset the stds
            sys.stdout = save_stdout
            sys.stderr = save_stderr

        self.update_message(
            "Parallel connection: {}".format(len(ipyclient)))

        return ipyclient



    def get_cluster_info(self):
        """ reports host and engine info for an ipyclient """    
        # get engine data, skips busy engines.
        hosts = []
        for eid in self.ipyclient.ids:
            engine = self.ipyclient[eid]
            if not engine.outstanding:
                hosts.append(engine.apply(socket.gethostname))

        # report it
        hosts = [i.get() for i in hosts]
        hostdict = {}
        for hostname in set(hosts):
            hostdict[hostname] = hosts.count(hostname)
        hpairs = [
            "<i>{}</i>: {} cores".format(key, val) for 
            (key, val) in hostdict.items()
        ]
        self.update_message(
            "Parallelization: {}".format(", ".join(hpairs)))



    def store_pids_for_shutdown(self):
        "reset tool ipcluster dict pids dict and set with current engine pids"
        self.tool.ipcluster["pids"] = {}
        for eid in self.ipyclient.ids:
            engine = self.ipyclient[eid]
            if not engine.outstanding:
                pid = engine.apply(os.getpid).get()
                self.tool.ipcluster["pids"][eid] = pid



    def wrap_run(self, dry_run=False):
        """
        Takes an analysis tools object with an associated _ipcluster attribute
        dictionary and either launches an ipcluster instance or connects to a 
        running one. The ipyclient arg overrides the auto arg.
        """

        try:
            # check that ipyclient is connected (3 seconds tries)
            if self.ipyclient:
                for i in range(3):
                    if len(self.ipyclient):
                        break
                    else:
                        time.sleep(1)
                assert len(self.ipyclient), "ipcluster not connected/running."

            # set ncores to max if user did not set
            else:
                if not self.tool.ipcluster["cores"]:
                    self.tool.ipcluster["cores"] = get_allocated_cores()

                # launch ipcluster and get the parallel client with ipp-{} id
                if self.auto:
                    # start ipcluster and attach ipyrad-cli cluster-id
                    self.start_ipcluster()
                    self.ipyclient = self.wait_for_connection()

                # neither auto or ipyclient we'll still look for default
                # profile running ipcluster.
                else:
                    self.ipyclient = self.wait_for_connection()                

            # print cluster stats at this point
            self.widget.close()
            self.get_cluster_info()

            # before running any jobs store engine pids for hard shutdown
            self.store_pids_for_shutdown()

            # run the job
            if not dry_run:
                self.tool._run(
                    **self.rkwargs,
                    ipyclient=self.ipyclient,
                    children=[self.message],
                    )

        # print the error and cleanup
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt by user\n")

        except Exception as inst:
            print("\nEncountered an error:\n{}\n".format(inst))
            raise

        # cancel/kill any unfinished jobs and shutdown hub if 'auto=True'
        finally:           
            self.cleanup()


    def cleanup(self):
        "Cancel or kill unfinished jobs and shutdown hub if auto=True"
        try:
            # can't close client if it was never open
            if self.ipyclient:

                # Interrupt: send SIGINT (2) to all engines if any engines
                try:
                    self.ipyclient.abort()
                    time.sleep(1)
                    for eid, pid in self.tool.ipcluster["pids"].items():
                        if self.ipyclient.queue_status()[eid]["tasks"]:
                            # hard kill the process
                            os.kill(pid, 2)
                    time.sleep(3)
                except ipp.NoEnginesRegistered:
                    pass

                # Cleanup: purge memory so we can reuse the Client
                if not self.ipyclient.outstanding:
                    self.ipyclient.purge_everything()
                else:
                    self.auto = True
                    self.update_message(
                        "Error: ipcluster shutdown and must be restarted")
                    # print("\nerror: ipcluster shutdown and must be restarted")

                # Shutdown the hub if it was auto-launched or broken
                if self.auto:
                    self.ipyclient.shutdown(hub=True, block=False)
                    self.ipyclient.close()
                    if self.show_cluster:
                        self.update_message("Parallel connection closed.")
                        time.sleep(0.5)

            # close the cluster info
            self.widget.close()

        except Exception as inst2:
            print("warning: error during shutdown:\n{}".format(inst2))


def detect_cpus():
    # Detects total CPUs on the system (works for personal computers)
    if hasattr(os, "sysconf"):
        if os.sysconf_names.get("SC_NPROCESSORS_ONLN"):
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.ncpu"])
            return int(output.strip())
        except Exception:
            pass
    if os.environ.get("NUMBER_OF_PROCESSORS"):
        try:
            ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
            if ncpus > 0:
                return ncpus
        except ValueError:
            pass
    return 1


def get_allocated_cores():
    """
    Returns the number of cores allocated by SLURM if available.
    Falls back to the total number of CPUs otherwise.
    """
    slurm_cores = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cores:
        try:
            return int(slurm_cores)
        except ValueError:
            pass
    slurm_job_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if slurm_job_cpus:
        try:
            # In some cases, this may be a comma-separated list.
            return int(slurm_job_cpus.split(',')[0])
        except ValueError:
            pass
    return detect_cpus()
