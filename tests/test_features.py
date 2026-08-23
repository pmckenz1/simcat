import numpy as np
import toytree

from simcat.utils import get_all_admix_edges, get_snps_count_matrix


def test_count_matrix_preserves_quartet_order_and_counts():
    tree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    alignment = np.array(
        [
            [0, 3, 1],
            [1, 2, 1],
            [2, 1, 1],
            [3, 0, 1],
        ],
        dtype=np.int8,
    )
    matrices = get_snps_count_matrix(tree, alignment)
    assert matrices.shape == (1, 16, 16)
    assert matrices.sum() == 3
    assert matrices[0, 1, 11] == 1
    assert matrices[0, 14, 4] == 1
    assert matrices[0, 5, 5] == 1


def test_admixture_edges_are_directional_and_sister_filter_is_effective():
    tree = toytree.rtree.imbtree(ntips=5, treeheight=1e6)
    all_edges = get_all_admix_edges(tree)
    nonsister_edges = get_all_admix_edges(tree, exclude_sisters=True)
    assert all(source != destination for source, destination in all_edges)
    assert all(
        (destination, source) in all_edges
        for source, destination in all_edges
    )
    assert len(nonsister_edges) < len(all_edges)
