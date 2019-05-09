#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#include <iostream>
#include <cstring>

class complex_t {
public:
    complex_t();                            // Constructor
    complex_t(float m_re, float m_im);      // Real and imaginary
    complex_t(const complex_t &c);          // Copy constructor
    ~complex_t();                           // Destructor

    // Operators
    complex_t operator+(const complex_t &c) const;
    complex_t operator-(const complex_t &c) const;
    complex_t operator*(const complex_t &c) const;
    complex_t operator/(const complex_t &c) const;
    complex_t magnitude() const;            // Magnitude of complex number
    complex_t conjugate() const;            // Conjugate of complex number
    void print() const;                     // Print the complex number.

    float re;                               // Real part of complex number
    float im;                               // Imaginary part of complex number
};

#endif

