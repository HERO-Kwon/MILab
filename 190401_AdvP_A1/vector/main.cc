#include <cstring>
#include <iostream>
#include "vector.h"

using namespace std;

int main(void) {
    vector_t<string> v1;

    // v1 is empty.
    cout << "v1.empty() = " << (v1.empty() ? "yes" : "no") << endl;
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl << endl;

    // Reserve a memory block in v1.
    v1.reserve(2);
    cout << "v1.empty() = " << (v1.empty() ? "yes" : "no") << endl;
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl << endl;

    // Push back and check capacity.
    v1.push_back("43");
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl;

    v1.push_back("7");
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl;
    /*
    // Print v1 elements using [] operator. *ADDED*
    for(size_t i = 0; i < v1.size(); i++) {
        cout << "v1[" << i << "] = " << v1[i] << endl;
    }*/
    cout << endl;

    // Insert into v1 using begin().
    v1.insert(v1.begin(), "41");
    v1.insert(++v1.begin(), "37");
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl;
    
    // Insert into v1 using end().
    v1.insert(v1.end(), "89");
    v1.insert(--v1.end(), "11");
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl;

    // Print v1 elements using iterator.
    for(vector_t<string>::iterator it = v1.begin();
        it != v1.end();it++) {
        cout << "*it = " << *it << endl;
    }
    cout << endl;

    // Pop back.
    v1.pop_back();
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl;
    for(size_t i = 0; i < v1.size(); i++) {
        cout << "v1[" << i << "] = " << v1[i] << endl;
    }
    cout << endl;

    // Erase the last and first elements.
    v1.erase(--v1.end());
    v1.erase(v1.begin());

    // Copy-create v2.
    vector_t<string> v2(v1);

    // Clear v1.
    v1.clear();
    cout << "v1.capacity() = " << v1.capacity() << endl;
    cout << "v1.size() = " << v1.size() << endl << endl;

    // Check if copy-created vector is sane.
    cout << "v2.capacity() = " << v2.capacity() << endl;
    cout << "v2.size() = " << v2.size() << endl;
    for(vector_t<string>::iterator it = v2.begin();
        it != v2.end(); it++) {
        cout << "*it = " << *it << endl;
    }

    return 0;
}
