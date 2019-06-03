#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include "mergesort.h"

using namespace std;

unsigned *data = 0;

int main(void) {
    // Allocate host memory space for data to sort.
    data = new unsigned[num_data];

    // Read a binary file named "data".
    fstream file_stream;
    file_stream.open("data", fstream::in|fstream::binary);
    if(!file_stream.is_open()) {
        cerr << "Error: failed to open data" << endl;
        exit(1);
    }
    if(!file_stream.read((char*)data, num_data * sizeof(unsigned))) {
        cerr << "Error: failed to read data" << endl;
        delete [] data;
        file_stream.close();
        exit(1);
    }
    file_stream.close();

    // Invoke mergesort().
    mergesort(data);

    // Print several selected data points.
    cout << "data[0] = " << data[0] << endl;
    cout << "data[1] = " << data[1] << endl;
    cout << "data[" << num_data/2 << "] = " << data[num_data/2] << endl;
    cout << "data[" << num_data-1 << "] = " << data[num_data-1] << endl;

    // Deallocate host memory.
    delete [] data;

    return 0;
}

