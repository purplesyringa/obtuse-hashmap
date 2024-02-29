# obtuse hashmap

This is an attempt at designing a novel hashtable implementation. obtuse-hashmap mixes ideas from existing implementations, mostly F14 and sparsehash. Please note that this is more of a "what if" than a production-grade project. Don't use it except as a source of ideas.


## Design

In this hashtable, data is split into buckets. Each bucket stores keys of different hashes, at most 512 distinct hashes per bucket. That is, the first bucket contains hashes ranging from 0 to 511, the second bucket contains hashes from 512 to 1023 and so on.

Each bucket stores all of its key-value pairs in a single dynamic array, ordered by key hashes. This is more memory-efficient than using a linked list, or using a separate vector/list per hash (rather than per bucket).

For each bucket, we also store a bitset of 512 bits indicating keys of which hashes are present. This is merely a bitmask, not a histogram: if there's a collision and two keys map to the same hash, the bitset stores just a single '1' bit in the appropriate location.

The key detail is that we pay for unused hashes with just one bit of memory. This is unlike chaining-based hashtables, which store a machine word per hash, and unlike open addressing-based hashtables, which store a whole data entry per hash. This allows us to decrease the load factor significantly without increasing memory consumption, which in turn decreases the number of *hash* collisions (while increasing *bucket* collision rate).

Nevertheless, it turns out that *hash* collisions is what affects the time complexity most. Given the bucket bitset and a hash of a key, we can easily compute the position of the corresponding entry in the bucket vector under the assumption that no *hash* collisions happen, as the vector is sorted by key hash. If collisions do happen, though, the guesstimate is still approximately correct, and linear probing turns out to be quite efficient in finding the correct position.
