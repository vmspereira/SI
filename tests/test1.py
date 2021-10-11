import unittest
import sys
import os


try:
    import si
except:
    DIR = os.path.dirname(os.path.realpath(__file__))
    PATH = os.path.join(DIR, '../src/')
    sys.path.insert(0, PATH)


class TestUnlabeledDataset(unittest.TestCase):

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=False)

    def testLen(self):
        self.assertGreater(len(self.dataset), 0)


class TestLabeledDataset(TestUnlabeledDataset):

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
