#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <cstdlib>
#include "vector-def.h"

/* Vector */

// Default constructor
template <typename T>
vector_t<T>::vector_t() :
    array(0),
    array_size(0),
    num_elements(0) {
}

// Constructor
template <typename T>
vector_t<T>::vector_t(const size_t s) :
    array(0),
    array_size(s),
    num_elements(0) {
    // Reserve the array of size s.
    array = (T*)malloc(array_size * sizeof(T));
}

// Constructor
template <typename T>
vector_t<T>::vector_t(const size_t s, const T &t) :
    array(0),
    array_size(s),
    num_elements(s) {
    // Reserve the array of size s, and fill all entries with t.
    array = (T*)malloc(array_size * sizeof(T));
    for(size_t i = 0; i < num_elements; i++) { new (&array[i]) T(t); }
}

// Copy constructor
template <typename T>
vector_t<T>::vector_t(const vector_t &v) :
    array(0),
    array_size(v.array_size),
    num_elements(v.num_elements) { 
    /* Assignment */
    // Reserve the array of same size, and fill all entries with t.
    array = (T*)malloc(v.array_size * sizeof(T));
    for(size_t i = 0; i < v.num_elements; i++) { new (&array[i]) T(v[i]); }
}

// Destructor
template <typename T>
vector_t<T>::~vector_t() {
    // Blow up the array.
    for(size_t i = 0; i < num_elements; i++) { array[i].~T(); }
    free(array);
}

// Push back
template <typename T>
void vector_t<T>::push_back(const T &t) {
    if(num_elements == array_size) {
        // Double up the array when full.
        array_size *= 2;
        T *old_array = array;
        array = (T*)malloc(array_size * sizeof(T));
        // Copy elements from the old array.
        for(size_t i = 0; i < num_elements; i++) {
            new (&array[i]) T(old_array[i]);
            old_array[i].~T();
        }
        // Deallocate the old array.
        free(old_array);
    }
    // Add a new element at the end.
    new (&array[num_elements++]) T(t);
}

// Pop back
template <typename T>
void vector_t<T>::pop_back() {
    /* Assignment */
    // Delete an element at the end
    array[num_elements-1].~T();
    //new(&array[num_elements-1]) T();
    num_elements--;
}

// Insert
template <typename T>
typename vector_t<T>::iterator vector_t<T>::insert(const iterator &it, const T &t) {
    /* Assignment */
    
    //get array index number of the iterator
    size_t index = it.ptr - array;
    
    // Double up the array when full
    if(num_elements == array_size) {
        // Double up the array when full.
        array_size *= 2;
        T *old_array = array;
        array = (T*)malloc(array_size * sizeof(T));
        // Copy elements from the old array.
        for(size_t i = 0; i < num_elements; i++) {
            new (&array[i]) T(old_array[i]);
            old_array[i].~T();
        }
        // Deallocate the old array.
        free(old_array);
    }
    
    // shift array elements
    for(size_t j=num_elements; j != index ; --j)
    {
        // erase array elements in shifting position
        array[j].~T();
        // shift array elements to right
        new(&array[j]) T(array[j-1]);
    }
    
    // add new element at the specified position
    new(&array[index]) T(t);
    // increment number of elements
    num_elements ++; 
    // return iterator pointing to the inserted element
    return iterator(&array[index]);
}

// Erase
template <typename T>
typename vector_t<T>::iterator vector_t<T>::erase(const iterator &it) {

    /* Assignment */
    
    //get array index number of the iterator
    int index = it.ptr - array;    

    // shift array elements
    for(size_t j=index; j != num_elements-1 ; ++j)
    {
        // erase array elements in shifting position
        array[j].~T();
        // shift array elements to left
        new(&array[j]) T(array[j+1]);
    }
    //delete last element
    pop_back();
    
    //set pointer of the iterator to next of removed one
    return iterator(&array[index+1]);
}

// Reserve
template <typename T>
void vector_t<T>::reserve(const size_t s) {
    if(s > array_size) {
        // Increase the array size.
        array_size = s;
        T *old_array = array;
        array = (T*)malloc(array_size * sizeof(T));
        // Copy all elements from the old array.
        for(size_t i = 0; i < num_elements; i++) {
            new (&array[i]) T(old_array[i]);
            old_array[i].~T();
        }
        // Blow up the old array.
        free(old_array);
    }
}

// Clear
template <typename T>
void vector_t<T>::clear() {
    /* Assignment */
    // Delete all elements from the array.
    for(size_t i = 0; i < num_elements; i++) {
        array[i].~T();
    }
    // set number of elements to zero
    num_elements=0;
}

// Capacity
template <typename T>
size_t vector_t<T>::capacity() const {
    // Return the array capacity.
    return array_size;
}

// Size
template <typename T>
size_t vector_t<T>::size() const {
    // Return the number of elements in the array.
    return num_elements;
}

// Is empty?
template <typename T>
bool vector_t<T>::empty() const {
    // Is the array empty?
    return num_elements == 0;
}

// Begin iterator
template <typename T>
typename vector_t<T>::iterator vector_t<T>::begin() const {
    // Return the iterator pointing to the first element.
    return iterator(array);
}

// End iterator
template <typename T>
typename vector_t<T>::iterator vector_t<T>::end() const {
    /* Assignment */
    // Return the iterator pointing to the last element.
    return iterator(&array[num_elements]);
}

// [] operator
template <typename T>
T& vector_t<T>::operator[](const size_t i) const {
    /* Assignment */
    return(array[i]); // return array elements
}



/* Vector iterator */

// Default constructor
template <typename T>
iterator_t<T>::iterator_t() :
    ptr(0) {
    // Nothing to do
}

// Constructor
template <typename T>
iterator_t<T>::iterator_t(T *t) :
    ptr(t) {
    // Nothing to do
}

// Copy constructor
template <typename T>
iterator_t<T>::iterator_t(const iterator_t<T> &it) :
    ptr(it.ptr) {
    // Nothing to do
}

// Destructor
template <typename T>
iterator_t<T>::~iterator_t() {
    // Nothing to do
}

// Dereference operator
template <typename T>
T& iterator_t<T>::operator*() const {
    /* Assignment */
    // values of pointer address
    return *ptr;
}

// ++it operator
template <typename T>
iterator_t<T> iterator_t<T>::operator++() {
    // Pre-increment the pointer.
    return iterator_t<T>(++ptr);
}

// it++ operator
template <typename T>
iterator_t<T> iterator_t<T>::operator++(int) {
    /* Assignment */
    // Post-increment the pointer.
    return iterator_t<T>(ptr++);
}

// --it operator
template <typename T>
iterator_t<T> iterator_t<T>::operator--() {
    /* Assignment */
    // Pre-decrement the pointer.
    return iterator_t<T>(--ptr);
}

// it-- operator
template <typename T>
iterator_t<T> iterator_t<T>::operator--(int) {
    /* Assignment */
    // Post-decrement the pointer.
    return iterator_t<T>(ptr--);
}

// != operator
template <typename T>
bool iterator_t<T>::operator!=(const iterator_t<T> &it) const {
    /* Assignment */
    // Pointers of two iterators are not equal.
    return ptr != it.ptr;
}

// == operator
template <typename T>
bool iterator_t<T>::operator==(const iterator_t<T> &it) const {
    // Pointers of two iterators are equal.
    return ptr == it.ptr;
}

#endif

