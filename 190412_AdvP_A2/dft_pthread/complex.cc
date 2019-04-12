#include <cmath>
#include "complex.h"

using namespace std;

// Constructors
complex_t::complex_t() :
    re(0.0),
    im(0.0) {
}

complex_t::complex_t(float m_re, float m_im) :
    re(m_re),
    im(m_im) {
}

// Copy constructor
complex_t::complex_t(const complex_t &c) :
    re(c.re),
    im(c.im) {
}

// Destructor
complex_t::~complex_t() {
}

// Operators
complex_t complex_t::operator+(const complex_t &c) const {
    return complex_t(re + c.re, im + c.im);
}

complex_t complex_t::operator-(const complex_t &c) const {
    return complex_t(re - c.re, im - c.im);
}

complex_t complex_t::operator*(const complex_t &c) const {
    return complex_t(re*c.re - im*c.im, re*c.im + im*c.re);
}

complex_t complex_t::operator/(const complex_t &c) const {
    complex_t dividend = (*this) * c.conjugate();
    complex_t divisor  = c.magnitude() * c.magnitude();
    return complex_t(dividend.re / divisor.re, dividend.im / divisor.re);
}

complex_t complex_t::magnitude() const {
    return complex_t(sqrt(re * re + im * im), 0.0);
}

complex_t complex_t::conjugate() const {
    return complex_t(re, -im);
}

void complex_t::print() const {
    cout << "(" << re << ", " << im << ")";
}

