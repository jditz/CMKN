#########################################################
# Test suite for testing the utility functions for      #
# convolutional oligo kernel networks.                  #
#                                                       #
# Author: Jonas Ditz                                    #
# Contact: ditz@informatik.uni-tuebingen.de             #
#########################################################

import unittest
import random
import time
import numpy as np
from cmkn import kmer2dict, find_kmer_positions, CMKNDataset, CMKNLayer, seq2ppm
import torch
from torch.utils.data import DataLoader


class KmerTestCase(unittest.TestCase):
    """
    TestCase for the functions handling k-meres.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # set up default parameters for the tests in this TestCase
        cls.sequence = 'aatctcagcgggcatcgatcggctaaagcga'
        cls.alphabet = ['c', 'g', 't', 'a']
        cls.kmer_size = 3
        cls.ppmtestseq = ['AGTC', 'ATGN']

        # set up test parameters for the test in this TestCase
        #   -> these variables will be used to assess whether functions work correctly
        cls.test_dict = {'ccc': 0, 'ccg': 1, 'cct': 2, 'cca': 3, 'cgc': 4, 'cgg': 5, 'cgt': 6, 'cga': 7, 'ctc': 8,
                         'ctg': 9, 'ctt': 10, 'cta': 11, 'cac': 12, 'cag': 13, 'cat': 14, 'caa': 15, 'gcc': 16,
                         'gcg': 17, 'gct': 18, 'gca': 19, 'ggc': 20, 'ggg': 21, 'ggt': 22, 'gga': 23, 'gtc': 24,
                         'gtg': 25, 'gtt': 26, 'gta': 27, 'gac': 28, 'gag': 29, 'gat': 30, 'gaa': 31, 'tcc': 32,
                         'tcg': 33, 'tct': 34, 'tca': 35, 'tgc': 36, 'tgg': 37, 'tgt': 38, 'tga': 39, 'ttc': 40,
                         'ttg': 41, 'ttt': 42, 'tta': 43, 'tac': 44, 'tag': 45, 'tat': 46, 'taa': 47, 'acc': 48,
                         'acg': 49, 'act': 50, 'aca': 51, 'agc': 52, 'agg': 53, 'agt': 54, 'aga': 55, 'atc': 56,
                         'atg': 57, 'att': 58, 'ata': 59, 'aac': 60, 'aag': 61, 'aat': 62, 'aaa': 63}
        cls.test_list = [[], [], [], [], [], [8, 19], [], [15, 28], [3], [], [], [22], [], [5], [12], [], [], [7, 27],
                         [21], [11], [10, 20], [9], [], [], [], [], [], [], [], [], [16], [], [], [14, 18], [2], [4],
                         [], [], [], [], [], [], [], [], [], [], [], [23], [], [], [], [], [6, 26], [], [], [],
                         [1, 13, 17], [], [], [], [], [25], [0], [24]]
        cls.ppmtestvec = np.array([[1., 0., 0., 0.], [0., 0., 0., .5], [0., .5, .5, 0.], [0., .5, .5, 0.],
                                   [0., 0., 0., .5]])

    def test_kmer2dict(self):
        kmer_dict = kmer2dict(self.kmer_size, self.alphabet)
        self.assertEqual(kmer_dict, self.test_dict)

    def test_find_kmer_positions(self):
        positions = find_kmer_positions(self.sequence, self.test_dict, self.kmer_size)
        self.assertEqual(positions, self.test_list)

    def test_seq2ppm(self):
        ppm = seq2ppm(self.ppmtestseq)
        self.assertTrue(np.allclose(ppm, self.ppmtestvec))


class DatasetTestCase(unittest.TestCase):
    """
    TestCase for the custom CMKN Dataset class.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # set the random seeds
        current_time = int(time.time())
        print("RNG seed for DatasetTestCase: {}".format(current_time))
        random.seed(current_time)
        torch.manual_seed(current_time)
        np.random.seed(current_time)

        # set up important parameters
        cls.batch_size = random.randint(1, 4)
        cls.dna_file = "data/ground_truth/dna_test.fasta"
        cls.protein_file = "data/ground_truth/protein_test.fasta"

        # set up DNA and Protein DataLoader
        dataset_dna = CMKNDataset(filepath=cls.dna_file, alphabet='DNA_FULL')
        dataset_protein = CMKNDataset(filepath=cls.protein_file, alphabet='PROTEIN_FULL')
        dataloader_dna = DataLoader(dataset_dna, batch_size=cls.batch_size)
        dataloader_protein = DataLoader(dataset_protein, batch_size=cls.batch_size)

        cls.samples_dna = next(iter(dataloader_dna))
        cls.samples_protein = next(iter(dataloader_protein))

    def test_outputDims(self):
        # make sure the DataLoader produces tensors with correct shape
        dna_shape = self.samples_dna[0].shape
        protein_shape = self.samples_protein[0].shape

        self.assertEqual(dna_shape[0], self.batch_size)
        self.assertEqual(dna_shape[1], 5)
        self.assertEqual(dna_shape[2], 8)

        self.assertEqual(protein_shape[0], self.batch_size)
        self.assertEqual(protein_shape[1], 26)
        self.assertEqual(protein_shape[2], 8)

    def test_onehotTensors(self):
        # make sure that the one-hot encoding works fine
        #     test sequence dna: AGGAGACT
        #     test sequence protein: ETVPVKLK
        dna_goal = torch.tensor([[1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
        protein_goal = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)

        self.assertTrue(torch.equal(self.samples_dna[0][0, :, :], dna_goal))
        self.assertTrue(torch.equal(self.samples_protein[0][0, :, :], protein_goal))

        # test if all columns have unit l1 norm
        self.assertEqual(torch.sum(self.samples_dna[0].norm(p=1, dim=1)),
                         self.samples_dna[0].shape[0] * self.samples_dna[0].shape[2])
        self.assertEqual(torch.sum(self.samples_protein[0].norm(p=1, dim=1)),
                         self.samples_protein[0].shape[0] * self.samples_protein[0].shape[2])


class CMKNLayerTestCase(unittest.TestCase):
    """
    TestCase for the CMKNLayer object.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # set the random seeds
        current_time = int(time.time())
        print("RNG seed for CMKNLayerTestCase: {}".format(current_time))
        random.seed(current_time)
        torch.manual_seed(current_time)
        np.random.seed(current_time)

        # initialize the CMKNLayer
        cls.in_channels = random.randint(1, 20)
        cls.out_channels = random.randint(1, 20)
        cls.kmer_length = random.randint(1, 20)
        cls.layer = CMKNLayer(cls.in_channels, cls.out_channels, cls.kmer_length)

        # initialize weights with random values
        torch.nn.init.xavier_normal_(cls.layer.weight)
        torch.nn.init.xavier_normal_(cls.layer.pos_anchors)

    def test_normalize(self):
        self.layer.normalize_()

        # check if all weights have columns with unit l2 norm
        self.assertEqual(torch.sum(self.layer.weight.norm(p=2, dim=1)).item(),
                         self.out_channels * self.kmer_length)
        self.assertEqual(torch.sum(self.layer.pos_anchors.norm(p=2, dim=1)).item(), self.out_channels)


if __name__ == '__main__':
    unittest.main()
