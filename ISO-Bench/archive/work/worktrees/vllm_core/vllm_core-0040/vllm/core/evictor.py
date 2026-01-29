import enum
import heapq
from heapq import heapify, heappush

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed PhysicalTokenBlocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    __slots__ = ("content_hash", "num_hashed_tokens", "last_accessed")

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # Limit how large the priority queue can grow compared to live entries
    CLEANUP_THRESHOLD = 1
    # Rebuild heap after too many stale pops to keep eviction fast
    STALE_POP_THRESHOLD = 32
    # Absolute slack for heap growth before rebuild
    CLEANUP_DELTA_ABS = 4096



    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        # heap of tuples: (last_accessed, -num_hashed_tokens, block_id)
        self.priority_queue = []
        # counter for consecutive stale pops
        self._stale_pops = 0


    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _maybe_cleanup(self):
        # Rebuild the heap if it has grown disproportionately due to lazy updates
        live = len(self.free_table)
        if live == 0:
            self.priority_queue.clear()
            self._stale_pops = 0
            return
        pq_len = len(self.priority_queue)
        if pq_len <= live:
            return
        delta = pq_len - live
        # Allow up to 25% slack or an absolute slack before rebuilding
        if delta <= max(live >> 2, self.CLEANUP_DELTA_ABS):
            return
        self._rebuild_heap()
    def _rebuild_heap(self):
        # Build heap from current live entries
        self.priority_queue = [
            (meta.last_accessed, -meta.num_hashed_tokens, bid)
            for bid, meta in self.free_table.items()
        ]
        heapify(self.priority_queue)
        self._stale_pops = 0

    def evict(self) -> Tuple[int, int]:
        free = self.free_table
        if not free:
            raise ValueError("No usable cache memory left")

        heap = self.priority_queue
        heappop = heapq.heappop
        maybe_cleanup = self._maybe_cleanup
        stale_pops = 0

        # Pop until we find a live, up-to-date entry
        while True:
            try:
                last_accessed, neg_tokens, bid = heappop(heap)
            except IndexError:
                self._rebuild_heap()
                heap = self.priority_queue
                stale_pops = 0
                last_accessed, neg_tokens, bid = heappop(heap)
            meta = free.get(bid)
            if meta is None or meta.last_accessed != last_accessed or meta.num_hashed_tokens != -neg_tokens:
                stale_pops += 1
                if stale_pops >= max(64, len(heap) >> 3):
                    self._rebuild_heap()
                    heap = self.priority_queue
                    stale_pops = 0
                continue
            evicted = free.pop(bid)
            maybe_cleanup()
            return bid, evicted.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        self.free_table[block_id] = meta
        heappush(self.priority_queue,
                       (last_accessed, -num_hashed_tokens, block_id))
        self._maybe_cleanup()

    def update(self, block_id: int, last_accessed: float):
        meta = self.free_table[block_id]
        if meta.last_accessed == last_accessed:
            return
        meta.last_accessed = last_accessed
        # push a new entry; stale one will be lazily ignored
        heappush(self.priority_queue,
                       (last_accessed, -meta.num_hashed_tokens, block_id))

    def remove(self, block_id: int):
        # Lazy removal: delete from map; heap entry becomes stale
        if self.free_table.pop(block_id, None) is None:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self._maybe_cleanup()

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
