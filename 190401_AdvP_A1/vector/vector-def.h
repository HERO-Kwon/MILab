#ifndef _VECTOR_DEF_H_
#define _VECTOR_DEF_H_

template <typename T> class iterator_t;

// Vector
template <typename T>
class vector_t {
public:
    vector_t();                                     // Default constructor
    vector_t(const size_t s);                       // Constructor
    vector_t(const size_t s, const T &t);           // Constructor
    vector_t(const vector_t &v);                    // Copy constructor
    ~vector_t();                                    // Destructor

    typedef iterator_t<T> iterator;                 // Typedef iterator
    
    void push_back(const T &t);                     // Push back
    void pop_back();                                // Pop back
    iterator insert(const iterator &it, const T &t);// Insert
    iterator erase(const iterator &it);             // Erase
    void reserve(const size_t s);                   // Reserve
    void clear();                                   // Clear

    size_t capacity() const;                        // Capacity
    size_t size() const;                            // Size
    bool empty() const;                             // Is empty?
    iterator begin() const;                         // Begin iterator
    iterator end() const;                           // End iterator
    T& operator[](const size_t i) const;            // [] operator

private:
    T *array;                                       // Array
    size_t array_size;                              // Capacity
    size_t num_elements;                            // Size
};

// Vector iterator
template <typename T>
class iterator_t {
public:
    iterator_t();                                   // Default constructor
    iterator_t(T *t);                               // Constructor
    iterator_t(const iterator_t<T> &it);            // Copy constructor
    ~iterator_t();                                  // Destructor 

    T& operator*() const;                           // Dereference operator
    iterator_t<T> operator++();                     // ++it operator
    iterator_t<T> operator++(int);                  // it++ operator
    iterator_t<T> operator--();                     // --it operator
    iterator_t<T> operator--(int);                  // it-- operator
    bool operator!=(const iterator_t<T> &it) const; // != operator
    bool operator==(const iterator_t<T> &it) const; // == operator

private:
    T *ptr;                                         // Pointer to vector element
    friend class vector_t<T>;                       // vector_t<T> access private
};

#endif

