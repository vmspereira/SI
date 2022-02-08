import unittest


class TestUnlabeledDataset(unittest.TestCase):

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=False)

    def testLen(self):
        self.assertGreater(len(self.dataset), 0)

    def testSummary(self):
        from si.data import summary
        summary(self.dataset)


class TestLabeledDataset(TestUnlabeledDataset):

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)


class TestFeatureSelection(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
