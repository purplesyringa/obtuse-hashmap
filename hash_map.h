#pragma once

// This is not strictly necessary, but optimizes some bitwise operations and
// speeds up memory access
#pragma GCC target("avx,tune=native")

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <forward_list>
#include <functional>
#include <immintrin.h>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace obtuse_hashtable {

// In this hashtable, data is split into buckets. Each bucket stores keys of
// different hashes, at most HASHES_PER_BUCKET distinct hashes per bucket. That
// is, the first bucket contains hashes ranging from 0 to HASHES_PER_BUCKET-1,
// the second bucket contains hashes from HASHES_PER_BUCKET to
// 2*HASHES_PER_BUCKET-1 and so on.
//
// Each bucket stores all of its key-value pairs in a single dynamic array,
// ordered by key hashes. This is more memory-efficient than using a linked
// list, or using a separate vector/list per hash (rather than per bucket).
//
// For each bucket, we also store a bitset of HASHES_PER_BUCKET bits indicating
// keys of which hashes are present. This is merely a bitmask, not a histogram:
// if there's a collision and two keys map to the same hash, the bitset stores
// just a single '1' bit in the appropriate location.
//
// The key detail is that we pay for unused hashes with just one bit of memory.
// This is unlike chaining-based hashtables, which store a machine word per
// hash, and unlike open addressing-based hashtables, which store a whole data
// entry per hash. This allows us to decrease the load factor significantly
// without increasing memory consumption, which in turn decreases the number of
// *hash* collisions (while increasing *bucket* collision rate).
//
// Nevertheless, it turns out that *hash* collisions is what affects the time
// complexity most. Given the bucket bitset and a hash of a key, we can easily
// compute the position of the corresponding entry in the bucket vector under
// the assumption that no *hash* collisions happen, as the vector is sorted by
// key hash. If collisions do happen, though, the guesstimate is still
// approximately correct, and linear probing turns out to be quite efficient in
// finding the correct position.

// Increasing this constant reduces memory footprint, which might in turn enable
// the index to fit into cache better, but slows down insertion and erasure.
static const size_t HASHES_PER_BUCKET = 64 * 8;

// `using std::popcount;` would be available in <bit> in C++20.
inline size_t popcount(uint64_t n) { return __builtin_popcountll(n); }

using HashMaskIndex = uint16_t;

// A fancy bitset, enabling efficient prefix popcount.
struct HashMask {
  std::array<uint64_t, HASHES_PER_BUCKET / 64> value{};

  bool has_bit(HashMaskIndex index) const {
    return (value[index / 64] >> (index % 64)) & 1;
  }
  void set_bit(HashMaskIndex index) {
    value[index / 64] |= uint64_t{1} << (index % 64);
  }
  void reset_bit(HashMaskIndex index) {
    value[index / 64] &= ~(uint64_t{1} << (index % 64));
  }
  size_t popcount_before(HashMaskIndex index) const {
    size_t acc = 0;
    for (int i = 0; i < index / 64; i++) {
      acc += popcount(value[i]);
    }
    acc += popcount(value[index / 64] & ((uint64_t{1} << (index % 64)) - 1));
    return acc;
  }
};

// These prime numbers represent the count of buckets the hashtable might store.
// Yes, 1 is not a prime. Screw me. Nevertheless, a single bucket (i.e. a flat
// hashtable) might be more efficient for small sizes, so we prefer to specify
// it here.
constexpr std::array<size_t, 63> SIZES = {1,
                                          2,
                                          5,
                                          11,
                                          17,
                                          37,
                                          67,
                                          131,
                                          257,
                                          521,
                                          1031,
                                          2053,
                                          4099,
                                          8209,
                                          16411,
                                          32771,
                                          65537,
                                          131101,
                                          262147,
                                          524309,
                                          1048583,
                                          2097169,
                                          4194319,
                                          8388617,
                                          16777259,
                                          33554467,
                                          67108879,
                                          134217757,
                                          268435459,
                                          536870923,
                                          1073741827,
                                          2147483659,
                                          4294967311,
                                          8589934609,
                                          17179869209,
                                          34359738421,
                                          68719476767,
                                          137438953481,
                                          274877906951,
                                          549755813911,
                                          1099511627791,
                                          2199023255579,
                                          4398046511119,
                                          8796093022237,
                                          17592186044423,
                                          35184372088891,
                                          70368744177679,
                                          140737488355333,
                                          281474976710677,
                                          562949953421381,
                                          1125899906842679,
                                          2251799813685269,
                                          4503599627370517,
                                          9007199254740997,
                                          18014398509482143,
                                          36028797018963971,
                                          72057594037928017,
                                          144115188075855881,
                                          288230376151711813,
                                          576460752303423619,
                                          1152921504606847009,
                                          2305843009213693967,
                                          4611686018427388039};

template <typename KeyType, typename ValueType,
          typename Hash = std::hash<KeyType>>
class HashMap {
  // We extend/shorten the bucket list when the load factor is higher/lower than
  // these constants. SHRINK_THRESHOLD must be at least about twice lower than
  // ENLARGE_THRESHOLD to avoid memory thrashing in case the load factor of the
  // hashtable oscillates around a critical value.
  static const size_t ENLARGE_THRESHOLD = 70; // in percents
  static const size_t SHRINK_THRESHOLD = 30;  // in percents

  struct Item {
    std::pair<const KeyType, ValueType> pair;

    Item() = delete;
    Item(KeyType key, ValueType value) { tee() = {key, value}; }
    Item(const Item &rhs) { *this = rhs; }
    Item(Item &&rhs) { *this = std::move(rhs); }
    Item &operator=(const Item &rhs) {
      tee() = rhs.pair;
      return *this;
    }
    Item &operator=(Item &&rhs) {
      tee() = rhs.tee();
      return *this;
    }

    // We need to return references to std::pair<const KeyType, ValueType> to
    // the user, while modifying keys internally. Strictly speaking, we
    // "actually" store a `KeyType` object at a `const KeyType` location, so
    // std::launder should work for interpreting a `const KeyType` location as
    // `KeyType`. This is *probably* not undefined behavior.
    std::pair<KeyType &, ValueType &> tee() {
      return {*std::launder(const_cast<KeyType *>(&pair.first)), pair.second};
    }
  };

  struct Bucket {
    HashMask hash_mask;
    std::vector<Item> items;

    // Insert an item to `items` at a given iterator. The implementation is
    // somewhat complicated due to the fact that `Item` has a very non-trivial
    // move assignment operator, which seems to stop the compiler from
    // optimizing array rotation to memmove in simple cases, so we need to
    // handle that manually. This is critical for performance.
    void insert_item(typename std::vector<Item>::iterator it,
                     std::pair<KeyType, ValueType> &&pair) {
      if constexpr (std::is_trivially_constructible_v<KeyType> &&
                    std::is_trivially_constructible_v<ValueType> &&
                    std::is_trivially_move_assignable_v<KeyType> &&
                    std::is_trivially_move_assignable_v<ValueType>) {
        size_t offset = it - items.begin();
        items.emplace_back(KeyType{}, ValueType{});
        // I know what I am doing
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
        std::memmove(items.data() + offset + 1, items.data() + offset,
                     sizeof(Item) * (items.size() - offset - 1));
#pragma GCC diagnostic pop
        items[offset] = Item{std::move(pair.first), std::move(pair.second)};
      } else {
        items.emplace(it, std::move(pair.first), std::move(pair.second));
      }
    }

    // Erase an item from `items` at a given iterator. The comments at
    // `insert_item` apply here just as well.
    typename std::vector<Item>::iterator
    erase_item(typename std::vector<Item>::iterator it) {
      if constexpr (std::is_trivially_constructible_v<KeyType> &&
                    std::is_trivially_constructible_v<ValueType> &&
                    std::is_trivially_move_assignable_v<KeyType> &&
                    std::is_trivially_move_assignable_v<ValueType>) {
        // I know what I am doing
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
        std::memmove(&*it, &*it + 1, sizeof(Item) * (items.end() - it - 1));
#pragma GCC diagnostic pop
        items.pop_back();
        return it;
      } else {
        return items.erase(it);
      }
    }
  };

  Hash _hasher;
  std::vector<Bucket> _buckets;
  // How many elements are stored, exactly -- not the count of buckets.
  size_t _size;

  // An iterator stores a reference to the current bucket and to the current
  // item within the bucket. end() is a bit of an exception: it stores the
  // reference to the one-after-last bucket and to the *first item of the last
  // bucket*. This is to avoid UB regarding calling begin() on a non-existent
  // vector.
  template <typename IteratorReferenceType, typename BucketIteratorType,
            typename ItemIteratorType>
  class Iterator {
  public:
    using value_type = std::pair<const KeyType, ValueType>;
    using reference_type = IteratorReferenceType;
    using pointer_type = decltype(&std::declval<IteratorReferenceType>());

    Iterator() {}

    Iterator(BucketIteratorType bucket_iterator,
             typename std::vector<Bucket>::const_iterator end_bucket,
             ItemIteratorType item_iterator)
        : _bucket_iterator(bucket_iterator), _end_bucket(end_bucket),
          _item_iterator(item_iterator) {}

    // Enable creation of const_iterator from iterator
    Iterator(const Iterator<std::pair<const KeyType, ValueType> &,
                            typename std::vector<Bucket>::iterator,
                            typename std::vector<Item>::iterator> &it)
        : _bucket_iterator(it._bucket_iterator), _end_bucket(it._end_bucket),
          _item_iterator(it._item_iterator) {}

    Iterator &operator++() {
      ++_item_iterator;
      if (_item_iterator != _bucket_iterator->items.end()) {
        return *this;
      }
      do {
        ++_bucket_iterator;
      } while (_bucket_iterator != _end_bucket &&
               _bucket_iterator->items.empty());
      if (_bucket_iterator == _end_bucket) {
        _item_iterator = (_bucket_iterator - 1)->items.begin();
      } else {
        _item_iterator = _bucket_iterator->items.begin();
      }
      return *this;
    }

    Iterator operator++(int) {
      auto copy = *this;
      ++*this;
      return copy;
    }

    reference_type operator*() const { return _item_iterator->pair; }
    pointer_type operator->() const { return &_item_iterator->pair; }

    bool operator==(const Iterator &rhs) {
      return _bucket_iterator == rhs._bucket_iterator &&
             _item_iterator == rhs._item_iterator;
    }

    bool operator!=(const Iterator &rhs) { return !(*this == rhs); };

  private:
    BucketIteratorType _bucket_iterator;
    typename std::vector<Bucket>::const_iterator _end_bucket;
    ItemIteratorType _item_iterator;

    friend class HashMap;
  };

public:
  using iterator = Iterator<std::pair<const KeyType, ValueType> &,
                            typename std::vector<Bucket>::iterator,
                            typename std::vector<Item>::iterator>;
  using const_iterator = Iterator<const std::pair<const KeyType, ValueType> &,
                                  typename std::vector<Bucket>::const_iterator,
                                  typename std::vector<Item>::const_iterator>;

  HashMap(Hash hasher = Hash{})
      : _hasher(std::move(hasher)), _buckets(1), _size(0) {}

  template <typename Iterator>
  HashMap(Iterator begin, Iterator end, Hash hasher = Hash{})
      : HashMap(std::move(hasher)) {
    for (auto it = begin; it != end; ++it) {
      insert(*it);
    }
  }

  HashMap(std::initializer_list<std::pair<KeyType, ValueType>> list,
          Hash hasher = Hash{})
      : HashMap(list.begin(), list.end(), std::move(hasher)) {}

  HashMap(const HashMap &rhs) = default;

  HashMap(HashMap &&rhs)
      : _hasher(std::move(rhs._hasher)), _buckets(std::move(rhs._buckets)),
        _size(rhs._size) {}

  HashMap &operator=(const HashMap &rhs) = default;

  HashMap &operator=(HashMap &&rhs) {
    // Lambda functions are not move-assignable, but are move-constructible in
    // C++17.
    if constexpr (std::is_move_assignable_v<Hash>) {
      _hasher = std::move(rhs._hasher);
    } else {
      std::destroy_at(&_hasher);
      new (&_hasher) Hash{std::move(rhs._hasher)};
    }
    _buckets = std::move(rhs._buckets);
    _size = rhs._size;
    return *this;
  }

  size_t size() const { return _size; }
  bool empty() const { return size() == 0; }

  Hash hash_function() const { return _hasher; }

  void insert(std::pair<KeyType, ValueType> pair) {
    if (_do_insert(std::move(pair))) {
      _maybe_enlarge();
    }
  }

  void erase(const KeyType &key) {
    // This seems to be more efficient than calling find() and then deleting
    // by iterator, because integrating the search logic into this method
    // allows us to determine whether hash_mask has to be reset more
    // efficiently.
    size_t hash = _hasher(key);
    HashMaskIndex relative_hash = _get_relative_hash(hash);
    size_t bucket_id = _get_bucket_id(hash);
    auto &bucket = _buckets[bucket_id];
    if (__builtin_expect(!bucket.hash_mask.has_bit(relative_hash), 0)) {
      return;
    }
    size_t hash_index = bucket.hash_mask.popcount_before(relative_hash);
    auto it = bucket.items.begin() + hash_index;
    while (it != bucket.items.end() &&
           _get_relative_hash(*it) < relative_hash) {
      ++it;
    }
    if (it == bucket.items.end() || _get_relative_hash(*it) != relative_hash) {
      return;
    }
    if (it->pair.first == key) {
      it = bucket.erase_item(it);
      --_size;
      if (it == bucket.items.end() ||
          _get_relative_hash(*it) != relative_hash) {
        bucket.hash_mask.reset_bit(relative_hash);
      }
      _maybe_shrink();
      return;
    }
    ++it;
    while (it != bucket.items.end() &&
           _get_relative_hash(*it) == relative_hash) {
      if (it->pair.first == key) {
        bucket.erase_item(it);
        --_size;
        _maybe_shrink();
        return;
      }
      ++it;
    }
  }

  iterator begin() { return _remove_iterator_constantness(cbegin()); }
  iterator end() {
    return iterator{_buckets.end(), _buckets.end(),
                    _buckets.back().items.begin()};
  }
  const_iterator begin() const {
    if (__builtin_expect(empty(), 0)) {
      return end();
    }
    // Well, this is awkward. One would expect that keys are distributed among
    // buckets uniformly, and therefore finding the first non-empty bucket
    // should work in expected constant time, as we maintain that capacity =
    // Theta(size). It is with deep regret that I have to inform you this
    // assumption is false.
    //
    // The problem is demonstrated by the following snippet:
    //     HashMap map;
    //     for(int i = 0; i < 1000000; i++)
    //         map.insert({i, i});
    //     for(int i = 0; i < 1000000; i++)
    //         map.erase(map.begin()->first);
    // Contrary to expectations, this works in quadratic time.
    //
    // Fixing this is not feasible without hurting performance a lot, and no one
    // does this anyway: builtin hashtables of Java and C# suffer from the same
    // problem, and production-grade C++ hashtables such as Facebook's, Google's
    // and others don't handle this either.
    auto bucket_iterator = _buckets.begin();
    while (bucket_iterator->items.empty()) {
      ++bucket_iterator;
    }
    return const_iterator{bucket_iterator, _buckets.end(),
                          bucket_iterator->items.begin()};
  }
  const_iterator end() const {
    // For end(), _item_iterator has to be set to some fixed value that can
    // easily be computed in the iterator's code without accessing hashtable
    // internals. An iterator to the beginning/end of the last bucket seems
    // like the easiest solution.
    return const_iterator{_buckets.end(), _buckets.end(),
                          _buckets.back().items.begin()};
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  iterator find(const KeyType &key) {
    return _remove_iterator_constantness(
        static_cast<const HashMap &>(*this).find(key));
  }
  const_iterator find(const KeyType &key) const {
    size_t hash = _hasher(key);
    HashMaskIndex relative_hash = _get_relative_hash(hash);
    size_t bucket_id = _get_bucket_id(hash);
    auto &bucket = _buckets[bucket_id];
    if (!bucket.hash_mask.has_bit(relative_hash)) {
      return end();
    }
    size_t hash_index = bucket.hash_mask.popcount_before(relative_hash);
    auto it = bucket.items.begin() + hash_index;
    while (it != bucket.items.end() &&
           _get_relative_hash(*it) < relative_hash) {
      ++it;
    }
    while (it != bucket.items.end() &&
           _get_relative_hash(*it) == relative_hash) {
      if (it->pair.first == key) {
        return const_iterator{_buckets.begin() + bucket_id, _buckets.end(), it};
      }
      ++it;
    }
    return end();
  }

  ValueType &operator[](const KeyType &key) {
    auto it = find(key);
    if (it != end()) {
      return it->second;
    }
    insert({key, {}});
    return find(key)->second;
  }
  ValueType &at(const KeyType &key) {
    return const_cast<ValueType &>(
        static_cast<const HashMap &>(*this)->at(key));
  }
  const ValueType &at(const KeyType &key) const {
    auto it = find(key);
    if (it == end()) {
      throw std::out_of_range("Missing key");
    }
    return it->second;
  }

  void clear() {
    _buckets.clear();
    _buckets.resize(1);
    _size = 0;
  }

private:
  void _maybe_enlarge() {
    if (_size * 100 >
        (_buckets.size() * HASHES_PER_BUCKET) * ENLARGE_THRESHOLD) {
      size_t next_size =
          *(std::lower_bound(SIZES.begin(), SIZES.end(), _buckets.size()) + 1);
      _set_count_of_buckets(next_size);
    }
  }

  void _maybe_shrink() {
    if (_size * 100 <
            (_buckets.size() * HASHES_PER_BUCKET) * SHRINK_THRESHOLD &&
        _buckets.size() != SIZES[0]) {
      size_t prev_size =
          *(std::lower_bound(SIZES.begin(), SIZES.end(), _buckets.size()) - 1);
      _set_count_of_buckets(prev_size);
    }
  }

  void _set_count_of_buckets(size_t new_size) {
    HashMap new_hashmap(std::move(_hasher));
    new_hashmap._buckets.resize(new_size);
    for (auto &bucket : _buckets) {
      for (auto &item : bucket.items) {
        new_hashmap._do_insert(std::move(item.pair));
      }
      // We'll need to cleanup memory anyway, so better do it now, without
      // unnecessarily bumping memory consumption during resize.
      bucket.items.clear();
      bucket.items.shrink_to_fit();
    }
    std::swap(*this, new_hashmap);
  }

  iterator _remove_iterator_constantness(const_iterator it) {
    if (it == end()) {
      return end();
    }
    // Bruh.
    auto bucket_iterator =
        _buckets.begin() + (it._bucket_iterator - _buckets.begin());
    auto item_iterator = bucket_iterator->items.begin() +
                         (it._item_iterator - bucket_iterator->items.begin());
    return iterator{bucket_iterator, it._end_bucket, item_iterator};
  }

  // Returns true if the key is new, false if it is present already.
  bool _do_insert(std::pair<KeyType, ValueType> pair) {
    size_t hash = _hasher(pair.first);
    size_t bucket_id = _get_bucket_id(hash);
    auto &bucket = _buckets[bucket_id];
    HashMaskIndex relative_hash = _get_relative_hash(hash);
    bucket.hash_mask.set_bit(relative_hash);
    size_t hash_index = bucket.hash_mask.popcount_before(relative_hash);
    auto it = bucket.items.begin() + hash_index;
    while (it != bucket.items.end() &&
           _get_relative_hash(*it) < relative_hash) {
      ++it;
    }
    while (it != bucket.items.end() &&
           _get_relative_hash(*it) == relative_hash) {
      if (it->pair.first == pair.first) {
        return false;
      }
      ++it;
    }
    bucket.insert_item(it, std::move(pair));
    ++_size;
    return true;
  }

  HashMaskIndex _get_relative_hash(size_t hash) const {
    return hash % HASHES_PER_BUCKET;
  }
  HashMaskIndex _get_relative_hash(const Item &item) const {
    return _get_relative_hash(_hasher(item.pair.first));
  }

  size_t _get_bucket_id(size_t hash) const {
    return (hash / HASHES_PER_BUCKET) % _buckets.size();
  }
};

} // namespace obtuse_hashtable

using obtuse_hashtable::HashMap;
