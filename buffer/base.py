# buffer/base.py
from abc import ABC, abstractmethod
class Buffer(ABC):
    def __init__(self, max_size: int, batch_size: int, shuffle: bool = True) -> None:
        super().__init__()
        self.shuffle = shuffle; self.max_size = max_size; self.batch_size = batch_size
    @abstractmethod
    def add(self, *args): ...
    @abstractmethod
    def clear(self): ...
    @abstractmethod
    def get_len(self): ...
    @abstractmethod
    def sample(self): ...
    def __len__(self): return self.get_len()
