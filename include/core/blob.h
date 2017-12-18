// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_CORE_BLOB_H_
#define DRAGON_CORE_BLOB_H_

#include <vector>
#include <string>
#include <memory>

typedef int64_t TIndex;

template <typename T>
class Blob {
 public:
    Blob();

    void Reshape(const std::vector<TIndex>& dims);
    void Assign(const std::vector<TIndex>& dims, T* data);

    TIndex axis(const TIndex i) const;
    TIndex num_axes() const;

    TIndex shape(const TIndex i) const;
    const std::vector<TIndex>& shape() const;
    std::string shape_string() const;

    TIndex nbytes() const;

    TIndex count() const;
    TIndex count(const TIndex start, const TIndex end) const;
    TIndex count(const TIndex start) const;

    const T* data();
    T* mutable_data();

 private:
    std::vector<TIndex> shape_;
    TIndex size_, capacity_;
    std::shared_ptr<T> memory_;
};

#endif    // DRAGON_CORE_BLOB_H_