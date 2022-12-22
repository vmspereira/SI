from .transformer import Transformer
import numpy as np

class LabelEncoder(Transformer):
    
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, dataset):
        y = dataset.y
        self.classes = np.unique(y)
        return self
    
    def transform(self, dataset, inline: bool = False):
        _map = {val: i for i, val in enumerate(self.classes)}
        _y = np.array([_map[x] for x in dataset.y])
        if inline:
            dataset.y = _y
            return dataset
        else:
            from .dataset import Dataset
            from copy import copy
            return Dataset(copy(dataset.X),
                           _y,
                           copy(dataset._xnames),
                           copy(dataset._yname)
                           )

        
class OneHotEncoder(Transformer):
    
    def transform(self, dataset, inline: bool = False):
        n_values = np.max(dataset.y) + 1
        _y = np.eye(n_values)[dataset.y]
        if inline:
            dataset.y = _y
            return dataset
        else:
            from .dataset import Dataset
            from copy import copy
            return Dataset(copy(dataset.X),
                           _y,
                           copy(dataset._xnames),
                           copy(dataset._yname)
                           )
