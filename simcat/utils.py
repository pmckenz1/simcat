#!/usr/bin/env python

import time
import datetime
import itertools
import numpy as np

from ipywidgets import IntProgress, HTML, Box
from IPython.display import display
from .jitted import count_matrix_int


__INVARIANTS__ = """
AAAA AAAC AAAG AAAT  AACA AACC AACG AACT  AAGA AAGC AAGG AAGT  AATA AATC AATG AATT
ACAA ACAC ACAG ACAT  ACCA ACCC ACCG ACCT  ACGA ACGC ACGG ACGT  ACTA ACTC ACTG ACTT
AGAA AGAC AGAG AGAT  AGCA AGCC AGCG AGCT  AGGA AGGC AGGG AGGT  AGTA AGTC AGTG AGTT
ATAA ATAC ATAG ATAT  ATCA ATCC ATCG ATCT  ATGA ATGC ATGG ATGT  ATTA ATTC ATTG ATTT

CAAA CAAC CAAG CAAT  CACA CACC CACG CACT  CAGA CAGC CAGG CAGT  CATA CATC CATG CATT
CCAA CCAC CCAG CCAT  CCCA CCCC CCCG CCCT  CCGA CCGC CCGG CCGT  CCTA CCTC CCTG CCTT
CGAA CGAC CGAG CGAT  CGCA CGCC CGCG CGCT  CGGA CGGC CGGG CGGT  CGTA CGTC CGTG CGTT
CTAA CTAC CTAG CTAT  CTCA CTCC CTCG CTCT  CTGA CTGC CTGG CTGT  CTTA CTTC CTTG CTTT

GAAA GAAC GAAG GAAT  GACA GACC GACG GACT  GAGA GAGC GAGG GAGT  GATA GATC GATG GATT
GCAA GCAC GCAG GCAT  GCCA GCCC GCCG GCCT  GCGA GCGC GCGG GCGT  GCTA GCTC GCTG GCTT
GGAA GGAC GGAG GGAT  GGCA GGCC GGCG GGCT  GGGA GGGC GGGG GGGT  GGTA GGTC GGTG GGTT
GTAA GTAC GTAG GTAT  GTCA GTCC GTCG GTCT  GTGA GTGC GTGG GTGT  GTTA GTTC GTTG GTTT

TAAA TAAC TAAG TAAT  TACA TACC TACG TACT  TAGA TAGC TAGG TAGT  TATA TATC TATG TATT
TCAA TCAC TCAG TCAT  TCCA TCCC TCCG TCCT  TCGA TCGC TCGG TCGT  TCTA TCTC TCTG TCTT
TGAA TGAC TGAG TGAT  TGCA TGCC TGCG TGCT  TGGA TGGC TGGG TGGT  TGTA TGTC TGTG TGTT
TTAA TTAC TTAG TTAT  TTCA TTCC TTCG TTCT  TTGA TTGC TTGG TTGT  TTTA TTTC TTTG TTTT
"""
INVARIANTS = np.array(__INVARIANTS__.strip().split()).reshape(16, 16)
ABBA_IDX = [
    (1, 4), (2, 8), (3, 12), (4, 1),
    (6, 9), (7, 13), (8, 2), (9, 6),
    (11, 14), (12, 3), (13, 7), (14, 11),
]
BABA_IDX = [
    (1, 1), (2, 2), (3, 3), (4, 4), 
    (6, 6), (7, 7), (8, 8), (9, 9),
    (11, 11), (12, 12), (13, 13), (14, 14),
]
FIXED_IDX = [
    (0, 0), (5, 5), (10, 10), (15, 15),
]

# HILS f1/f2
AABB_IDX = [
    (0, 5), (0, 10), (0, 15), 
    (5, 0), (5, 10), (5, 15),
    (10, 0), (10, 5), (10, 15),
    (15, 0), (15, 5), (15, 10), 
]

# HILS f3 (ijii - jiii) and f4 (iiji - iiij)
ABAA_IDX = [
    (1, 0), (2, 0), (3, 0),
    (4, 5), (5, 5), (6, 5),
    (8, 10), (9, 10), (11, 10),
    (12, 15), (13, 15), (14, 15),
]
BAAA_IDX = [
    (4, 0), (8, 0), (12, 0),
    (1, 5), (9, 5), (13, 5),
    (2, 10), (6, 10), (14, 10),
    (3, 15), (7, 15), (11, 15),
]
AABA_IDX = [
    (0, 4), (0, 8), (0, 12), 
    (5, 1), (5, 9), (5, 13),
    (10, 2), (10, 6), (10, 14),
    (15, 3), (15, 7), (15, 11),
]
AAAB_IDX = [
    (0, 1), (0, 2), (0, 3),
    (5, 4), (5, 6), (5, 7),
    (10, 8), (10, 9), (10, 11),
    (15, 12), (15, 13), (15, 14),
]




class SimcatError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class Progress(object):
    def __init__(self, njobs, message, children):

        # data
        self.njobs = njobs
        self.message = message
        self.start = time.time()

        # the progress bar 
        self.bar = IntProgress(
            value=0, min=0, max=self.njobs, 
            layout={
                "width": "350px",
                "height": "30px",
                "margin": "5px 0px 0px 0px",
            })

        # the message above progress bar
        self.label = HTML(
            self.printstr, 
            layout={
                "height": "25px",
                "margin": "0px",
            })

        # the box widget container
        heights = [
            int(i.layout.height[:-2]) for i in 
            children + [self.label, self.bar]
        ]
        self.widget = Box(
            children=children + [self.label, self.bar], 
            layout={
                "display": "flex",
                "flex_flow": "column",
                "height": "{}px".format(sum(heights) + 5),
                "margin": "5px 0px 5px 0px",
            })

    @property
    def printstr(self):
        self.elapsed = datetime.timedelta(seconds=int(time.time() - self.start))
        s1 = "<span style='font-size:14px; font-family:monospace'>"
        s2 = "</span>"
        inner = "{} | {:>3}% | {}".format(
            self.message, 
            int(100 * (self.bar.value / self.njobs)),
            self.elapsed,
        )
        return s1 + inner + s2

    def display(self):
        display(self.widget)

    def increment_all(self, value=1):
        self.bar.value += value
        if self.bar.value == self.njobs:
            self.bar.bar_style = "success"
        self.increment_time()

    def increment_time(self):
        self.label.value = self.printstr



def get_all_admix_edges(ttree, lower=0.25, upper=0.75, exclude_sisters=False):
    """
    Find all possible admixture edges on a tree. Edges are unidirectional, 
    so the source and dest need to overlap in time interval. To retrict 
    migration to occur away from nodes (these can be harder to detect when 
    validating methods) you can set upper and lower limits. For example, to 
    make all source migrations to occur at the midpoint of overlapping 
    intervals in which migration can occur you can set upper=.5, lower=.5.   
    """
    # bounds on edge overlaps
    if lower is None:
        lower = 0.0
    if upper is None:
        upper = 1.0

    # for all nodes map the potential admixture interval
    for snode in ttree.treenode.traverse():
        if snode.is_root():
            snode.interval = (None, None)
        else:
            snode.interval = (snode.height, snode.up.height)

    # for all nodes find overlapping intervals
    intervals = {}
    for snode in ttree.treenode.traverse():
        for dnode in ttree.treenode.traverse():
            if not any([snode.is_root(), dnode.is_root(), dnode == snode]):

                # [option] skip sisters
                if (exclude_sisters) & (dnode.up == snode.up):
                    continue

                # check for overlap
                smin, smax = snode.interval
                dmin, dmax = dnode.interval

                # find if nodes have interval where admixture can occur
                low_bin = np.max([smin, dmin])
                top_bin = np.min([smax, dmax])              
                if top_bin > low_bin:

                    # restrict migration within bin to a smaller interval
                    length = top_bin - low_bin
                    low_limit = low_bin + (length * lower)
                    top_limit = low_bin + (length * upper)
                    intervals[(snode.idx, dnode.idx)] = (low_limit, top_limit)
    return intervals



def get_snps_count_matrix(tree, seqs):
    """
    Compiles SNP data into a nquartets x 16 x 16 count matrix with the order
    of quartets determined by the shape of the tree.
    """
    # get all quartets for this size tree
    quarts = list(itertools.combinations(range(tree.ntips), 4))

    # shape of the arr (count matrix)
    arr = np.zeros((len(quarts), 16, 16), dtype=np.int64)

    # iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
    quartidx = 0
    for currquart in quarts:
        # cols indices match tip labels b/c we named tips node.idx
        quartsnps = seqs[currquart, :]
        # save as stacked matrices
        arr[quartidx] = count_matrix_int(quartsnps)
        # save flattened to counts
        quartidx += 1
    return arr



def calculate_dstat(mat):
    """
    Calculate ABBA-BABA (D-statistic) from a count matrix. 
    """
    # calculate
    abba = sum([mat[i] for i in ABBA_IDX])
    baba = sum([mat[i] for i in BABA_IDX])
    if abba + baba == 0:
        dstat = 0.
    else:
        dstat = (abba - baba) / (abba + baba)
    return dstat



def calculate_simple_f12(mat):
    """
    Returns the f1/f2 ratio from Kubatko and Chifman 2019, in this case it 
    is not normalized into a test statistic.
    """
    nsites = mat.sum()
    nabba = sum([mat[i] for i in ABBA_IDX])
    nbaba = sum([mat[i] for i in BABA_IDX])
    naabb = sum([mat[i] for i in AABB_IDX])
    abba = nabba / nsites
    baba = nbaba / nsites
    aabb = naabb / nsites   
    f1 = float(aabb - baba)
    f2 = float(abba - baba)
    if f2 == 0.:
        return 0.
    return f1 / f2



def calculate_hils_f12(mat, gamma=0.):
    """
    Calculate the f12 Hils statistic from Kubatko and Chifman 2019.
    """
    nsites = mat.sum()

    # calculate
    nabba = sum([mat[i] for i in ABBA_IDX])
    nbaba = sum([mat[i] for i in BABA_IDX])
    naabb = sum([mat[i] for i in AABB_IDX])
    abba = nabba / nsites
    baba = nbaba / nsites
    aabb = naabb / nsites

    f1 = aabb - baba
    f2 = abba - baba
    if f2 == 0.:
        return 0.

    sigmaf1 = (
        (1. / nsites) * sum([
            aabb * (1. - aabb),
            baba * (1. - baba), 
            2. * aabb * baba
            ])
        )
    sigmaf2 = (
        (1. / nsites) * sum([
            abba * (1. - abba),
            baba * (1. - baba),
            2. * abba * baba,
            ])
        )

    covf1f2 = (
        (1. / nsites) * sum([
            abba * (1. - aabb),
            aabb * baba,
            abba * baba,
            baba * (1. - baba)
            ])
        )

    ratio = gamma / (1. - gamma)
    num = f2 * ((f1 / f2) - ratio)
    p1 = (sigmaf2 * (ratio**2))
    p2 = ((2. * covf1f2 * ratio) + sigmaf1)
    denom = p1 - p2

    # calculate hils
    H = num / np.sqrt(abs(denom))
    return H



def get_sister_idxs(tre):
    sisters = []
    for node in tre.treenode.traverse():
        if len(node.children) == 2:
            sisters.append(list(np.sort([i.idx for i in node.children])))
    return(sisters)


# def progress_bar(njobs, nfinished, start, message=""):
#     "prints a progress bar"
#     ## measure progress
#     if njobs:
#         progress = 100 * (nfinished / njobs)
#     else:
#         progress = 100

#     ## build the bar
#     hashes = "#" * int(progress / 5.)
#     nohash = " " * int(20 - len(hashes))

#     ## get time stamp
#     elapsed = datetime.timedelta(seconds=int(time.time() - start))

#     ## print to stderr
#     args = [hashes + nohash, int(progress), elapsed, message]
#     print("\r[{}] {:>3}% | {} | {}".format(*args), end="")
#     sys.stderr.flush()



# def tile_reps(array, nreps):
#     "used to fill labels in the simcat.Database for replicates"
#     ts = array.size
#     nr = nreps
#     result = np.array(
#         np.tile(array, nr)
#         .reshape((nr, ts))
#         .T.flatten())
#     return result
