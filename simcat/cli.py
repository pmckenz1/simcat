#!/usr/bin/env python

import argparse
import platform
import sys
import logging
import re

import simcat


###### workflow:
# specify tree file, specify args
# specify directory in which to store progress file, simulation directories
# point to a single template slurm script
# copy the slurm script to each simulation directory and run it. 

# adapted from stdpopsim cli
#(https://github.com/popsim-consortium/stdpopsim/blob/master/stdpopsim/cli.py)

def exit(message):
    """
    Exit with the specified error message, setting error status.
    """
    sys.exit(f"{sys.argv[0]}: {message}")


class CLIFormatter(logging.Formatter):
    # Log levels
    # ERROR: only output errors. This is the level when --quiet is specified.
    # WARN: Output warnings or any other messages that the user should be aware of
    # INFO: Write out log messages for things the user might be interested in.
    # DEBUG: Write out debug messages for the user/developer.
    def format(self, record):
        if record.name == "py.warnings":
            # trim the ugly warnings.warn message
            match = re.search(
                r"Warning:\s*(.*?)\s*warnings.warn\(", record.args[0], re.DOTALL
            )
            record.args = (match.group(1),)
            self._style = logging.PercentStyle("WARNING: %(message)s")
        else:
            if record.levelno == logging.WARNING:
                self._style = logging.PercentStyle("%(message)s")
            else:
                self._style = logging.PercentStyle("%(levelname)s: %(message)s")
        return super().format(record)


def setup_logging(args):
    log_level = "ERROR"
    if args.verbose == 1:
        log_level = "WARN"
    elif args.verbose == 2:
        log_level = "INFO"
    elif args.verbose > 2:
        log_level = "DEBUG"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(CLIFormatter())
    logging.basicConfig(handlers=[handler], level=log_level)
    logging.captureWarnings(True)


def get_environment():
    """
    Returns a dictionary describing the environment in which stdpopsim
    is currently running.
    """
    env = {
        "os": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "libraries": {
            "simcat": {"version": simcat.__version__},
        },
    }
    return env


def simcat_cli_parser():
    class QuietAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.verbose = 0

    top_parser = argparse.ArgumentParser(
        description="Command line interface for simcat."
    )
    top_parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s {}".format(simcat.__version__),
    )
    logging_group = top_parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase logging verbosity (can use be used multiple times).",
    )
    logging_group.add_argument(
        "-q",
        "--quiet",
        nargs=0,
        help="Do not write any non-essential messages",
        action=QuietAction,
    )

#    top_parser.add_argument(
#        "-c",
#        "--cache-dir",
#        type=str,
#        default=None,
#        help=(
#            "Set the cache directory to the specified value. "
#            "Note that this can also be set using the environment variable "
#            "STDPOPSIM_CACHE. If both the environment variable and this "
#            "option are set, the option takes precedence. "
#            f"Default: {stdpopsim.get_cache_dir()}"
#        ),
#    )

#    top_parser.add_argument(
#        "-e",
#        "--engine",
#        default=simcat.get_default_engine().id,
#        choices=[e.id for e in simcat.all_engines()],
#        help="Specify a simulation engine.",
#    )
#
#    supported_models = stdpopsim.get_engine("msprime").supported_models
    sc_parser = top_parser.add_argument_group("simcat specific parameters")
    sc_parser.add_argument(
        "-m",
        "--mutation-rate",
        default=1e-8,
        help="Set the mutation rate."
    )
#
#    def time_or_model(
#        arg,
#        _arg_is_time=[
#            True,
#        ],
#        parser=top_parser,
#    ):
#        if _arg_is_time[0]:
#            try:
#                arg = float(arg)
#            except ValueError:
#                parser.error(f"`{arg}' is not a number")
#        else:
#            if arg not in supported_models:
#                parser.error(f"`{arg}' is not a supported model")
#        _arg_is_time[0] = not _arg_is_time[0]
#        return arg

#    msprime_parser.add_argument(
#        "--msprime-change-model",
#        metavar=("T", "MODEL"),
#        type=time_or_model,
#        default=[],
#        action="append",
#        nargs=2,
#        help="Change to the specified simulation MODEL at generation T. "
#        "This option may provided multiple times.",
#    )


#    subparsers = top_parser.add_subparsers(dest="subcommand")
#    subparsers.required = True

#    for species in stdpopsim.all_species():
#        add_simulate_species_parser(subparsers, species)


    return top_parser



# This function only exists to make mocking out the actual running of
# the program easier.
def run(args):
    args.runner(args)


def simcat_main(arg_list=None):
    parser = simcat_cli_parser()
    args = parser.parse_args(arg_list)
    setup_logging(args)
    if args.cache_dir is not None:
        simcat.set_cache_dir(args.cache_dir)
    run(args)
