# Test suite for testing the utility functions for      #
# convolutional oligo kernel networks.                  #
#                                                       #
# Author: Jonas Ditz                                    #
# Contact: ditz@informatik.uni-tuebingen.de             #
#########################################################

import unittest
from .utils import kmer2dict, find_kmer_positions


class KmerTestCase(unittest.TestCase):
    """
    TestCase for the functions handling k-meres.
    """
    def setUp(self):
        # set up default parameters for the tests in this TestCase
        self.sequence = 'aatctcagcgggcatcgatcggctaaagcga'
        self.alphabet = ['c', 'g', 't', 'a']
        self.kmer_size = 3

        # set up test parameters for the test in this TestCase
        #   -> these variables will be used to assess whether functions work correctly
        self.test_dict = {'ccc': 0, 'ccg': 1, 'cct': 2, 'cca': 3, 'cgc': 4, 'cgg': 5, 'cgt': 6, 'cga': 7, 'ctc': 8,
                          'ctg': 9, 'ctt': 10, 'cta': 11, 'cac': 12, 'cag': 13, 'cat': 14, 'caa': 15, 'gcc': 16,
                          'gcg': 17, 'gct': 18, 'gca': 19, 'ggc': 20, 'ggg': 21, 'ggt': 22, 'gga': 23, 'gtc': 24,
                          'gtg': 25, 'gtt': 26, 'gta': 27, 'gac': 28, 'gag': 29, 'gat': 30, 'gaa': 31, 'tcc': 32,
                          'tcg': 33, 'tct': 34, 'tca': 35, 'tgc': 36, 'tgg': 37, 'tgt': 38, 'tga': 39, 'ttc': 40,
                          'ttg': 41, 'ttt': 42, 'tta': 43, 'tac': 44, 'tag': 45, 'tat': 46, 'taa': 47, 'acc': 48,
                          'acg': 49, 'act': 50, 'aca': 51, 'agc': 52, 'agg': 53, 'agt': 54, 'aga': 55, 'atc': 56,
                          'atg': 57, 'att': 58, 'ata': 59, 'aac': 60, 'aag': 61, 'aat': 62, 'aaa': 63}
        self.test_list = [[], [], [], [], [], [8, 19], [], [15, 28], [3], [], [], [22], [], [5], [12], [], [], [7, 27],
                          [21], [11], [10, 20], [9], [], [], [], [], [], [], [], [], [16], [], [], [14, 18], [2], [4],
                          [], [], [], [], [], [], [], [], [], [], [], [23], [], [], [], [], [6, 26], [], [], [],
                          [1, 13, 17], [], [], [], [], [25], [0], [24]]

    def test_kmer2dict(self):
        kmer_dict = kmer2dict(self.kmer_size, self.alphabet)
        self.assertEqual(kmer_dict, self.test_dict)

    def test_find_kmer_positions(self):
        positions = find_kmer_positions(self.sequence, self.test_dict, self.kmer_size)
        self.assertEqual(positions, self.test_list)


if __name__ == '__main__':
    unittest.main()