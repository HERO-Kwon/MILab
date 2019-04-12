#ifndef __FILE_HANDLER_H__
#define __FILE_HANDLER_H__

#include <cstring>
#include "complex.h"

class file_handler_t {
public:
    file_handler_t();                               // Constructor
    ~file_handler_t();                              // Destructor

    void read_data(const std::string m_file_name,   // Read data from a file
                   complex_t *&m_data, unsigned &m_width, unsigned &m_height);
    void save_data(const std::string m_file_name,   // Save data to a file
                   complex_t *m_data, unsigned m_width, unsigned m_height);
};

#endif

