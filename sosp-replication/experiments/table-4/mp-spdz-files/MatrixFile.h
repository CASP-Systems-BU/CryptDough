/*
 * MatrixFile.h
 *
 * To replace file in: Protocols/MatrixFile.h
 */

#ifndef PROTOCOLS_MATRIXFILE_H_
#define PROTOCOLS_MATRIXFILE_H_

#include "Processor/Data_Files.h"
#include "ShareMatrix.h"

template<class T>
class MatrixFile : public Preprocessing<ShareMatrix<T>>
{
    typedef Preprocessing<ShareMatrix<T>> super;

    array<int, 3> dims;

    ifstream file;

public:
    MatrixFile(array<int, 3> dims, DataPositions& usage, Player& P) :
            super(usage), dims(dims)
    {
        string filename = PrepBase::get_matrix_prefix(
                        get_prep_sub_dir<T>(P.num_players()), dims) + "-P"
                        + to_string(P.my_num());
        file.open(filename);
        check_file_signature<T>(file, filename);
    }

    void get_three_no_count(Dtype type, ShareMatrix<T>& A, ShareMatrix<T>& B,
            ShareMatrix<T>& C)
    {
        assert(type == DATA_TRIPLE);
        A = {dims[0], dims[1]};
        B = {dims[1], dims[2]};
        C = {dims[0], dims[2]};
        // A.input(file);
        // B.input(file);
        // C.input(file);
    }
};

#endif /* PROTOCOLS_MATRIXFILE_H_ */
