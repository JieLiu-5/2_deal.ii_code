
#ifndef COMMON_FUNCTION_ADVANCED_H
#define COMMON_FUNCTION_ADVANCED_H


template <>
void print_vector_vertically<Vector<double>>(vector<Vector<double>> &vector)
{
  for (unsigned int i=0; i<vector.size();++i)
  {
    cout << "  [" << i << "]: ";
    vector[i].print(cout, 2);
  }
}


template <>
void print_vector_horizontally(const vector<Point<1>> &vector)                      // please specify another function for printing 2D Points
{
  for (unsigned int i=0; i<vector.size();++i)
  {
    cout << vector[i] << ", ";
  }
  cout << "\n";
}


template <>
void print_vector_horizontally(const vector<Vector<double>> &vector)
{
  for (unsigned int i=0; i<vector.size();++i)
  {
    cout << vector[i][0] << ", ";
  }
  cout << "\n";
}


template <typename T>
void save_Vector_to_txt(string& obj_string,
                        Vector<T> &obj_vector)
{
  cout << "Saving " << obj_string << " to a file\n";
  ofstream fid(obj_string+".txt"); 
  for (unsigned int i=0;i<obj_vector.size();i++)
  {
    fid << obj_vector[i] << " ";
    fid << "\n";
  } 
  fid.close();
  fid.clear();
}   


template <int dim>
void print_geometry_info()
{
  cout << "  Info of geometry: \n"
            << "    faces_per_cell: " << GeometryInfo<dim>::faces_per_cell << endl
            << "    vertices_per_cell: " << GeometryInfo<dim>::vertices_per_cell << endl
            << "    vertices_per_face: " << GeometryInfo<dim>::vertices_per_face << endl;
}
    


void print_geometry_info()
{
  cout << "  Info of geometry: \n"
            << "    faces_per_cell: " << GeometryInfo<2>::faces_per_cell << endl
            << "    vertices_per_cell: " << GeometryInfo<2>::vertices_per_cell << endl
            << "    vertices_per_face: " << GeometryInfo<2>::vertices_per_face << endl;
}


template <int dim>
void print_tria_info(const Triangulation<dim> &triangulation)
{
  cout << "  Info of triangulation:\n"
       << "    number of levels: " << triangulation.n_levels() << "\n"
       << "    number of vertices: " << triangulation.n_vertices() << "\n"
       << "    number of active cells: " << triangulation.n_active_cells() << "\n";
//         << "    Total number of cells: " << triangulation.n_cells() << "\n";
//         << "    vertices_per_cell: " << GeometryInfo< dim >::vertices_per_cell << "\n";
  
# if 1
  cout << "    coordinates of vertices of each active cell\n";
  {
    map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    
    for (; cell!=endc; ++cell)                                                                  
      {
        cout << "    [" << cell->active_cell_index() << "] ";
          
        for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
        {
            cout << cell->vertex_index(vertex) << " (" << cell->vertex(vertex) << "), ";     
        }
        
        cout << "\n";
      }
  }
#endif
}

template <int dim>
void print_boundary_info(const Triangulation<dim> &triangulation)
{
  std::cout << "  Info of boundary: \n" ;
  
  std::map<unsigned int, unsigned int> boundary_count;
  
  Triangulation<2>::active_face_iterator face = triangulation.begin_active_face();
  Triangulation<2>::active_face_iterator endface = triangulation.end_face();    
    
  for (; face!=endface; ++face)
  {
    if(face->at_boundary())
    {
      boundary_count[face->boundary_id()]++;
      if (face->boundary_id() == 0)
      {
        std::cout << "    Dirichlet BCs imposed on ";
        std::cout << "face " << face->index() << ": ";
        for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
        {
          std::cout << "(" << face->vertex(vertex) << ") ";
        }
        std::cout << "\n";
      }
      else if(face->boundary_id() == 1)
      {
        std::cout << "    Neumann BCs imposed on ";
        std::cout << "face " << face->index() << ": ";
        for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
        {
          std::cout << "(" << face->vertex(vertex) << ") ";
        }
        std::cout << "\n";
      }          
    }
  }

  std::cout << "    boundary indicators: ";
  for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
      it!=boundary_count.end();
      ++it)
  {
    std::cout << it->first << "(" << it->second << " times) ";
  }
  std::cout << std::endl;     
}


template <int dim>
void adjust_boundary_id_2d(Triangulation<dim>& triangulation)
{
  unsigned int id_method_adjusting_boundary = 1;
  
  if(id_method_adjusting_boundary==0)
  {
//     triangulation.begin_active()->face(0)->set_boundary_id(1);       // left
//     triangulation.begin_active()->face(1)->set_boundary_id(1);       // right
    triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom
    triangulation.begin_active()->face(3)->set_boundary_id(1);       // top
  }else if(id_method_adjusting_boundary==1)
  {
    const Point<dim> p1 (0.0, 0.0);                         
    const Point<dim> p2 (1.0, 1.0);         // 1.0   0.2
    
    Triangulation<2>::active_face_iterator face = triangulation.begin_active_face();      // method 2 for setting boundary ids
    Triangulation<2>::active_face_iterator endface = triangulation.end_face();    
    
    for (; face!=endface; ++face)                                 // this iterates from the first active face to the last active face
    {
      if(face->at_boundary())
      {
        if(face->vertex(0)(0)==0 && face->vertex(1)(0)==0)
        {
//                    cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the left boundary";
//                     face->set_boundary_id(1);
//                     cout << endl;
        }else if(face->vertex(0)(1)==p2[1] && face->vertex(1)(1)==p2[1])
        {
//                     cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the upper boundary";
                    face->set_boundary_id(1);
//                     cout << endl;                    
        }else if(face->vertex(0)(1)==p1[1] && face->vertex(1)(1)==p1[1])
        {
//                     cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the bottom boundary";
                    face->set_boundary_id(1);
//                     cout << endl;                    
        }
      }
    }
  }
}


template <typename T>
void print_fe_info (string& obj_string, T& fe)
{
  cout << "  Info of " << obj_string << ": \n";
  cout << "    name: " << fe.get_name() << "\n";
  cout << "    # of base elements: " << fe.n_base_elements() << "\n";
  cout << "    degree: " << fe.degree << "\n";                                                      // degree of fe is equal to the degree of fe_velocity in the mixed FEM (P, RT or BDM)
                                                                                                    // degree of fe_velocity is equal to the degree of input for P
                                                                                                    //                          one order higher than the degree of input for RT and BDM
                                                                                                    
                                                                                                    // fe.degree is used in project_boundary_values_div_conforming()
  
  cout << "    # of components: " << fe.n_components() << "\n";
  cout << "    # of blocks: " << fe.n_blocks() << "\n";
//   cout << "  fe.n_dofs_per_vertex(): " << fe.n_dofs_per_vertex() << "\n";
  cout << "    # of dofs per cell: " << fe.n_dofs_per_cell() << "\n";
  
  
#if 0
  for (unsigned int i = 0; i < fe.n_base_elements(); ++i)
  {
    cout << "    " << i << "th base\n";
    cout << "      name: " << fe.base_element(i).get_name() << endl;
    
    if (fe.base_element(i).has_support_points())
    {
      cout << "      # of support points in a reference cell: " << fe.base_element(i).get_unit_support_points().size() << "\n";

    //     for (unsigned int i = 0; i < fe.base_element(i).get_unit_support_points().size(); ++i)
    //     {
    //       cout << "  [" << i << "]: ";
    //       for (unsigned int j=0; j<fe.base_element(i).dimension; ++j)                           // source fe.base_element(i).dimension?
    //       {
    //         cout << fe.base_element(i).get_unit_support_points()[i][j] << " ";  
    //       }
    //       cout << "\n";
    //     }
    //     cout << "\n";
    }else
    {
      
      cout << "      having no defined support points\n";
        
//       vector<Point<2>> generalized_support_points = fe.base_element(i).get_generalized_support_points(); 
//       
//       cout << "      generalized_support_points (" << generalized_support_points.size() << " in total):\n";      
//       print_vector(generalized_support_points);
    }    
    
    cout << "      degree: " << fe.base_element(i).degree << endl;
    cout << "      dofs_per_cell: " << fe.base_element(i).dofs_per_cell << endl;
    cout << "      dofs_per_quad: " << fe.base_element(i).dofs_per_quad << endl;  
    cout << "      n_dofs_per_line: " << fe.base_element(i).n_dofs_per_line() << endl; 
    cout << "      n_dofs_per_face: " << fe.base_element(i).n_dofs_per_face() << endl; 
    cout << "      dofs_per_vertex: " << fe.base_element(i).dofs_per_vertex << endl;
    
  }
#endif
  
}


template <int dim>
void print_local_dof_indices(string& obj_string, DoFHandler<dim>& dof_handler)
{
  cout << "  local_dof_indices of " << obj_string << "\n";
  
  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;  
  vector<types::global_dof_index> local_dof_indices;
  local_dof_indices.resize(dofs_per_cell);
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices); 
    cout << "  cell [" << cell->active_cell_index() << "] ";
    
    print_vector_horizontally(local_dof_indices);
    
  }
}

template <int dim>
void save_local_dof_indices(string& obj_string, DoFHandler<dim>& dof_handler)
{
  cout << "  saving local_dof_indices of " << obj_string << "(" << dof_handler.n_dofs() << " in total)\n";
  
  ofstream fid;
  fid.open("local_dof_indices_temp.txt", ofstream::trunc);
  
  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;  
  vector<types::global_dof_index> local_dof_indices;
  local_dof_indices.resize(dofs_per_cell);
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  
  
  cell->get_dof_indices (local_dof_indices);

  for (unsigned int k=0; k<local_dof_indices.size(); ++k)
  {
    fid << local_dof_indices[k] << "\n";
  }
  
  ++cell;
  

  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices); 

    for (unsigned int k=1; k<local_dof_indices.size(); ++k)
    {
      fid << local_dof_indices[k] << "\n";
    }    
  }
  fid.close();    
  fid.clear();
}

template <int dim>
void print_local_face_dof_indices(string& obj_string, DoFHandler<dim>& dof_handler)
{
  cout << "  local_face_dof_indices of " << obj_string << ": \n";
  
  const unsigned int dofs_per_face=dof_handler.get_fe().dofs_per_face;     
  vector<types::global_dof_index> local_face_dof_indices;
  local_face_dof_indices.resize(dofs_per_face);
       
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  
  for (; cell!=endc; ++cell)
  {
    cout << "  cell [" << cell->active_cell_index() << "]\n";
      
    for (unsigned int face_n = 0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
    {
        
      cout << "    face [" << face_n << "] ";
        
      cell->face(face_n)->get_dof_indices (local_face_dof_indices);
        
      print_vector_horizontally(local_face_dof_indices);
      
    }
  }
}


template <int dim>
void print_coords_of_dofs(string obj_string, DoFHandler<dim>& dof_handler)
{
  cout << "  coordinates of " << obj_string << "\n";

  vector<Point<dim> > coords_dofs(dof_handler.n_dofs());                       // coordinates are for global dof indices
  
  MappingQ<dim, dim> mapping(dof_handler.get_fe().degree);                     // 
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, coords_dofs);
  
  cout << "  ";
  print_vector_horizontally(coords_dofs); 
}


template <int dim>
void save_coords_of_dofs(string obj_string_argument,
                         DoFHandler<dim>& dof_handler,
                         unsigned int current_refinement_level)
{
  cout << "  Saving coordinates of dofs of " << current_refinement_level << "th refinement to a file\n";

  vector<Point<dim> > coords_dofs(dof_handler.n_dofs());
  MappingQ<dim, dim> mapping(dof_handler.get_fe().degree);            
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, coords_dofs);
  
  save_vector_of_Point_to_txt(obj_string_argument, coords_dofs);  
}


template <typename T>
void save_system_matrix_to_txt(string& obj_string, SparseMatrix<T> &obj_matrix)
{
  cout << "Saving " << obj_string << " to a file\n";
  ofstream fid(obj_string+".txt");
  for (unsigned int i=0;i<obj_matrix.m();i++) 
  {
    for (unsigned int j=0;j<obj_matrix.n();j++)
    {
      fid << obj_matrix.el(i, j) << " ";
    }
    fid << "\n";
  } 
  fid.close();
  fid.clear();
}


template <typename T>
void print_quadrature_info_on_reference_cell (T &quadrature_formula)
{
  cout << "\n";
  cout << "  Info of quadrature: \n";
  
  cout << "    nr. of quadrature points: " << quadrature_formula.size() << "\n";
  
  cout << "    coords: \n";
  for (unsigned int i=0; i<quadrature_formula.size();++i)
  {
    cout << "  [" << i << "] " << quadrature_formula.get_points()[i] << "\n";
  }
  cout << endl;
  
  
  cout << "    weights: \n";
  for (unsigned int i=0; i<quadrature_formula.size();++i)
  {
    cout << "  [" << i << "] " << quadrature_formula.weight(i) << "\n";
  }  
  cout << "\n\n";  
}


template <int dim>
void print_constraints_info(string& obj_string, AffineConstraints<real_t> &constraints, DoFHandler<dim>& dof_handler)
{
  cout << "    @Info of " << obj_string << "\n";

//   cout << "    the constrained dofs are: \n";
//   for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
//   {
//     if(constraints.is_constrained(i))
//     {
//       cout << "      " << i << " ";
//       if(constraints.is_inhomogeneously_constrained(i))
//       {
//         cout << "(inhomogeneous)\n";
//       }else
//       {
//         cout << "(homogeneous)\n";
//       }
//     }
//   }
    
  cout << "    content " << "(" << constraints.n_constraints() << "/" << dof_handler.n_dofs() << "): \n";
  constraints.print(cout);
}



template <typename T>
void scale_quad_coords_small(vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    for (unsigned int j=0; j<quad_coords_vector[0].dimension; ++j)
    {
      quad_coords_vector[i](j)=quad_coords_vector[i](j)*2.0;
    }
  }
}


template <typename T>
void scale_quad_coords_big(vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    for (unsigned int j=0; j<quad_coords_vector[0].dimension; ++j)
    {
      quad_coords_vector[i](j)=(quad_coords_vector[i](j)-0.5)*2.0;   
    }
  }
}


template <typename T>
void scale_quad_coords_1_0(vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    quad_coords_vector[i](0)=(quad_coords_vector[i](0)-0.5)*2.0;
    quad_coords_vector[i](1)=quad_coords_vector[i](1)*2.0;
  }
}


template <typename T>
void scale_quad_coords_0_1(vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    quad_coords_vector[i](0)=quad_coords_vector[i](0)*2.0;
    quad_coords_vector[i](1)=(quad_coords_vector[i](1)-0.5)*2.0;
  }
}


template <typename T>
void extract_basis_function (T& fe)                                // only available for one-dimensional case, Aug. 28, 2020
{
  cout << "  Extracting basis function\n";
  cout << "    name of fe: " << fe.get_name() << "\n";
  
  unsigned int n_cell_interpolate = 100;
  unsigned int n_node_interpolate = n_cell_interpolate+1;
  double size_interpolate = 1.0/n_cell_interpolate;
    
  vector<Point<fe.dimension>> coords_interpolate(n_node_interpolate);
  vector<vector<double>> values_interpolate(n_node_interpolate,vector<double>(fe.degree+1,0));
    
  if(fe.dimension==1)
  {
    coords_interpolate[0][0]=0;
        
    for (unsigned int i = 1; i<n_node_interpolate; ++i)
    {
        coords_interpolate[i][0]=coords_interpolate[i-1][0]+size_interpolate;
    }
        
    for (unsigned int i = 0; i < n_node_interpolate; i++)
    {
        for (unsigned int j = 0; j < fe.degree+1; j++)
        {
        values_interpolate[i][j] = fe.shape_value(j,coords_interpolate[i]);
        }
    }
  }else if(fe.dimension==2)
  {
    cout << "    dimension 2 is not available\n";
  }
  
  string obj_string_file_name = "";
  obj_string_file_name = "data_unit_coords";
  save_vector_of_Point_to_txt(obj_string_file_name,
                              coords_interpolate);
  
  obj_string_file_name = "data_basis_values_deg_" + to_string(fe.degree) + ".txt";
  save_vector_of_vector_to_a_file(obj_string_file_name,
                                  obj_string_file_name,
                                  values_interpolate);
  
}


// template <typename T1, int dim>
// void implementing_the_dirichlet_boundary_condition (T1 &obj_class, FEValues<dim>& fe_values)
// {
//     cout << "implementing_the_dirichlet_boundary_condition\n";
//     cout << "dimension: " << dim << "\n";
// }


#endif
