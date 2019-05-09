#include <cstdlib>
#include <fstream>
#include <stdint.h>
#include "file_handler.h"

using namespace std;

// Constructor
file_handler_t::file_handler_t() {
}

// Destructor
file_handler_t::~file_handler_t() {
}

// Read data from a file.
void file_handler_t::read_data(const string m_file_name, complex_t *&m_data,
                               unsigned &m_width, unsigned &m_height) {
    // Open the file.
    fstream file_stream;
    file_stream.open(m_file_name.c_str(), fstream::in|fstream::binary);
    if(!file_stream.is_open()) {
        cerr << "Error: failed to open " << m_file_name << endl;
        exit(1);
    }

    // Read the dimension information.
    file_stream >> m_width >> m_height;

    // Allocate data.
    m_data = new complex_t[m_width * m_height];

    // Load the data.
    for(unsigned r = 0; r < m_height; r++) {      // Rows
        for(unsigned c = 0; c < m_width; c++) {   // Columns
            file_stream >> m_data[r * m_width + c].re;
        }
    }

    // Close the file. 
    file_stream.close();
}

// Save data to a file.
void file_handler_t::save_data(const string m_file_name, complex_t *m_data,
                               unsigned m_width, unsigned m_height) {
    // Open the file.
    fstream file_stream;
    file_stream.open(m_file_name.c_str(), fstream::out|fstream::binary);
    if(!file_stream.is_open()) {
        cerr << "Error: failed to open " << m_file_name << endl;
        exit(1);
    }

    // Store the dimension information.
    file_stream << m_width << " " << m_height << endl;

    // Store the magnitude data.
    for(unsigned r = 0; r < m_height; r++) {      // Rows
        for(unsigned c = 0; c < m_width; c++) {   // Columns
            file_stream << m_data[r * m_width + c].magnitude().re << " ";
        }
        file_stream << endl;
    }

    // Close the file.
    file_stream.close();
}

