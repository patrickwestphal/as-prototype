#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
_log = logging.getLogger()

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD

###############################################################################
# basic configuration
#
#DATA_FILE_PATH = 'example/dbpedia/dump.nt'
#DATA_FILE_PATH = 'example/lgd/bremen_sampled.nt'
DATA_FILE_PATH = 'example/musicbrainz/dump.nt'

OUT_FILE_PATH = 'cluster_info.txt'

# the number of resources a cluster must at least have to appear in the result
MIN_RESOURCES_PER_CLUSTER = 5
# the number of properties a cluster must at least have; e.g. a value of 2
# means that clusters that group resources based on just one common property
# are not added to the result set
MIN_PROPERTIES_IN_CLUSTER = 2
###############################################################################

def read_data(filepath):
    sp_counts_dict = {}
    subjects = set()
    predicates = set()

    with open(filepath) as f:
        line_counter = 0

        for line in f:
            if line_counter % 10000 == 0:
                _log.debug('%i lines read' % line_counter)

            s, p = line.split()[:2]

            # decomment this in case you want to skip container membership properties
            # if p.startswith('<http://www.w3.org/1999/02/22-rdf-syntax-ns#_'):
            #     _log.debug('skipped %s' % p)
            #     line_counter += 1
            #     continue

            subjects.add(s)
            predicates.add(p)

            if s not in sp_counts_dict.keys():
                sp_counts_dict[s] = {p: 1}

            else:
                p_counts = sp_counts_dict[s]
                if p not in p_counts.keys():
                    sp_counts_dict[s][p] = 1
                else:
                    sp_counts_dict[s][p] += 1

            line_counter += 1

    return sp_counts_dict, list(subjects), list(predicates)

def make_mtrx(sp_counts_dict, subjects, predicates):
    mtrx = np.zeros((len(subjects), len(predicates)), dtype=np.bool)

    row_idx = 0
    _log.debug('going to update %i matrix rows' % len(subjects))
    for s in subjects:
        if row_idx % 10000 == 0:
            _log.debug('matrix row %i updated' % row_idx)
        # (Pdb) pp(p_count)
        # {'<http://dbpedia.org/ontology/abstract>': 1,
        #  '<http://dbpedia.org/ontology/activeYearsEndYear>': 1,
        #  '<http://dbpedia.org/ontology/activeYearsStartYear>': 1,
        #  '<http://dbpedia.org/ontology/country>': 1,
        #  '<http://dbpedia.org/ontology/identificationSymbol>': 1}
        p_counts = sp_counts_dict[s]
        col_idx = 0

        for p in predicates:
            count = p_counts.get(p)

            # NOTE: here variations are possible:
            # - setting up the matrix with the actual counts (then the dtype of
            #   mtrx has to be changed!!!)
            # - using a binary matrix but also take properties into account
            #   that appear more than once (i.e. if count >= 1)
            if count == 1:
                # mtrx[row_idx, col_idx] = count
                mtrx[row_idx, col_idx] = 1

            col_idx +=1
        row_idx += 1

    return mtrx

def decompose_mtrx(mtrx):
    _log.info('started decomposition')
    tsvd = TruncatedSVD().fit(mtrx)
    svd_mtrx = tsvd.components_
    svs = mtrx.dot(svd_mtrx.T)
    mtrx_approx = svs.dot(svd_mtrx)
    _log.info('finished decomposition')

    return mtrx_approx

def find_clusters(mtrx):
    _log.info('getting clusters')
    clusters = {}

    for row_idx in range(mtrx.shape[0]):
        if row_idx % 10000 == 0:
            _log.debug('reading matrix row %i' % row_idx)

        row_hash = ''.join([str(int(v)) for v in mtrx[row_idx]])

        if clusters.get(row_hash) is None:
            clusters[row_hash] = []

        clusters[row_hash].append(row_idx)

    cleaned_clusters = {}

    _log.info('cleaning clusters')
    for cluster_hash in clusters:
        if len(clusters[cluster_hash]) >= MIN_RESOURCES_PER_CLUSTER \
                and cluster_hash.count('1') >= MIN_PROPERTIES_IN_CLUSTER:

            cleaned_clusters[cluster_hash] = clusters[cluster_hash]

    _log.info('got clusters')
    return cleaned_clusters

def write_cluster_info_to_file(clusters, mtrx, subjects, predicates):
    with open(OUT_FILE_PATH, 'w') as f:
        cluster_counter = 1
        for cluster_hash in clusters:
            f.write('cluster %i (%s)\n' % (cluster_counter, cluster_hash))
            f.write('############\n')

            row_idxs = clusters[cluster_hash]
            for row_idx in row_idxs:
                f.write('%s\n' % subjects[row_idx])
                row = mtrx[row_idx]
                f.write('\t')

                for i in range(mtrx.shape[1]):
                    if row[i] > 0:
                        f.write('%s ' % predicates[i])
                f.write('\n\n')

            cluster_counter += 1
            f.write('###############################################################\n')
        f.write('%i clusters found' % len(clusters))

def run():
    sp_counts_dict, subjects, predicates = read_data(DATA_FILE_PATH)
    mtrx = make_mtrx(sp_counts_dict, subjects, predicates)
    del(sp_counts_dict)

    mtrx_approx = decompose_mtrx(lil_matrix(mtrx))
    clusters = find_clusters(mtrx_approx)
    write_cluster_info_to_file(clusters, mtrx, subjects, predicates)
