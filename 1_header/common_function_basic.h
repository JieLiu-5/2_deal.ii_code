#ifndef COMMON_FUNCTION_BASIC_H
#define COMMON_FUNCTION_BASIC_H

template <typename T>
void print_map(map<T,double>& obj_map)
{
  for (auto& x: obj_map) 
  {
    cout << x.first << ": " << x.second << '\n';
  }    
}

bool getFileContent(std::string fileName,
                    std::vector<double> & vecOfStrs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            vecOfStrs.push_back(stod(str));
    }
    //Close The File
    in.close();
    return true;
}



void sequence_numbers_in_a_txt_file(string& obj_string)
{

  vector<double> coords_of_dofs_sequenced;
  
  getFileContent(obj_string+".txt", coords_of_dofs_sequenced);

  
//   cout << "coords_of_dofs_sequenced before sorting:\n";
//   print_vector(coords_of_dofs_sequenced);

  sort( coords_of_dofs_sequenced.begin(), coords_of_dofs_sequenced.end() );

//   cout << "coords_of_dofs_sequenced after sorting:\n";
//   print_vector(coords_of_dofs_sequenced);
  
  string obj_new_string = obj_string + "_sequenced.txt";
  save_vector_of_numbers_to_a_file(obj_new_string,
                                   obj_new_string,
                                   coords_of_dofs_sequenced);
}  


template <typename T>
void save_vector_of_Point_to_txt(string& obj_string, vector<T> &obj_vector)
{
    unsigned int column_no= obj_vector[0].dimension; 
    unsigned int row_no = obj_vector.size();
    
    ofstream fid;
    fid.open(obj_string+".txt");
    
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


void printPattern(int radius)
{
  // dist represents distance to the center 
  float dist; 
  
  // for horizontal movement 
  for (int i = 0; i <= 2 * radius; i++) { 
  
    // for vertical movement 
    for (int j = 0; j <= 2 * radius; j++) { 
      dist = sqrt((i - radius) * (i - radius) +  
                  (j - radius) * (j - radius)); 
  
      // dist should be in the range (radius - 0.5) 
      // and (radius + 0.5) to print stars(*) 
      if (dist > radius - 0.5 && dist < radius + 0.5)  
        cout << "*"; 
      else 
        cout << " ";       
    } 
  
    cout << "\n"; 
  } 
} 


#endif
