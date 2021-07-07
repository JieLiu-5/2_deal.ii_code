#ifndef COMMON_FUNCTION_BASIC_H
#define COMMON_FUNCTION_BASIC_H


template <typename T>
void saving_vector_of_Point_to_a_file(string& obj_string,
                                 vector<T> &obj_vector)
{
    unsigned int column_no= obj_vector[0].dimension; 
    unsigned int row_no = obj_vector.size();
    
    ofstream fid;
    fid.open(obj_string);
    
    for (unsigned int k=0; k<row_no; ++k)
    {
        for (unsigned int j=0; j<column_no; ++j)
        {
            fid << obj_vector[k][j] << " ";
        }
        fid << "\n";
    }
    fid.close();
    fid.clear();
}



#endif
