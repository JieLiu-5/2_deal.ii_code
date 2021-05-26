#ifndef COMPUTE_L2_NORMS_COEFFICIENT_H
#define COMPUTE_L2_NORMS_COEFFICIENT_H


// template <typename T, int dim>
// class Computing_L2_Norms_of_Coefficients
// {
// public:
//     
//   Computing_L2_Norms_of_Coefficients(Triangulation<dim>& triangulation,
//                                      DoFHandler<dim>& dof_handler,
//                                      FEValues<dim>& fe_values,
//                                      T& obj_coeff);
//     
//   Triangulation<dim> triangulation;
//   DoFHandler<dim> dof_handler;
//   FEValues<dim> fe_values;
//   T& obj_coeff;
// };


template <typename T, int dim>
void compute_l2_norms_of_coefficient_numerically_core(Triangulation<dim>& triangulation,
                                                      DoFHandler<dim>& dof_handler,
                                                      FEValues<dim>& fe_values,
                                                      T& obj_coeff,
                                                      unsigned int& n_components)
{                                                                                               // capable of both the real and complex solution  <-- needs to be adjusted after considering d in 2D
  
  unsigned int n_q_points = fe_values.get_quadrature_points().size();               // we need fe_values to be initialized before using this function
  
  vector<Vector<double>> coeff_values(n_q_points, Vector<double>(n_components));
  
  Vector<double> cellwise_inte_coeff_square(n_components);
  vector<Vector<double>> cellwise_sqrt_inte_coeff_square (triangulation.n_active_cells(), Vector<double>(n_components));
  
  Vector<double> inte_L2_coeff(n_components);
  
  double inte_L2_coeff_both = 0.0;
  

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    vector<Point<dim>> coords_of_quadrature = fe_values.get_quadrature_points();  
    obj_coeff.value(coords_of_quadrature, coeff_values);
    
//     cout << "  coords of quadrature points: \n";
//     print_vector(coords_of_quadrature);
//     
//     cout << "  coeff_values: \n";
//     print_vector(coeff_values);
    
    for (unsigned int k = 0; k < n_components; ++k)
    {
      for (unsigned int q=0; q</*1*/n_q_points; ++q)
      {
        cellwise_inte_coeff_square[k]+= pow(coeff_values[q][k], 2.0) * fe_values.JxW(q);  
      }
      cellwise_sqrt_inte_coeff_square[cell->active_cell_index()][k] += sqrt(cellwise_inte_coeff_square[k]);
    }
    
    cellwise_inte_coeff_square.reinit(n_components);

  }
  
//   cout << "  cellwise_sqrt_inte_coeff_real_square: ";
//   cout << cellwise_sqrt_inte_coeff_real_square;
  
  for (unsigned int k = 0; k < n_components; ++k)
  {
    for (unsigned int i = 0; i < triangulation.n_active_cells(); ++i)
    {
      inte_L2_coeff[k] += pow(cellwise_sqrt_inte_coeff_square[i][k],2.0);
    }
    
    inte_L2_coeff_both += inte_L2_coeff[k];
    inte_L2_coeff[k] = sqrt(inte_L2_coeff[k]);
  }
  
  inte_L2_coeff_both = sqrt(inte_L2_coeff_both);
  
  cout << "containing " << n_components << " component(s), which reads " << inte_L2_coeff;
  if(n_components>1)
  {
    cout << ", of which the square root: " << inte_L2_coeff_both << ", ";
  }
  cout << "\n";
}


template <typename T1, int dim>
double compute_l2_norms_of_coefficient_numerically_core_real(Triangulation<dim>& triangulation,
                                                                  DoFHandler<dim>& dof_handler,
                                                                  FEValues<dim>& fe_values,
                                                                  T1& obj_coeff)
{
  unsigned int n_q_points = fe_values.get_quadrature_points().size();
  
  vector<double> coeff_values(n_q_points);
  
  double inte_coeff_square;
  Vector<double> cellwise_sqrt_inte_coeff_square (triangulation.n_active_cells());

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    vector<Point<dim>> coords_of_quadrature = fe_values.get_quadrature_points();  
    obj_coeff.value(coords_of_quadrature, coeff_values);
    
    for (unsigned int q=0; q</*1*/n_q_points; ++q)
    {
      inte_coeff_square+= pow(coeff_values[q], 2.0) * fe_values.JxW(q);
    }
    cellwise_sqrt_inte_coeff_square[cell->active_cell_index()] += sqrt(inte_coeff_square);
    
    inte_coeff_square=0;
  }
  
  double inte_l2_coeff = cellwise_sqrt_inte_coeff_square.l2_norm();
  
  return inte_l2_coeff;
}

template <typename T1, int dim>
double compute_l2_norms_of_coefficient_numerically_core_tensor_2_dim_real(Triangulation<dim>& triangulation,
                                                                  DoFHandler<dim>& dof_handler,
                                                                  FEValues<dim>& fe_values,
                                                                  T1& obj_coeff)
{
  unsigned int n_q_points = fe_values.get_quadrature_points().size();
  
  vector<Tensor<2,dim>> coeff_values(n_q_points, Tensor<2,dim>());                              // main change is here
  
  double inte_coeff_square;
  Vector<double> cellwise_sqrt_inte_coeff_square (triangulation.n_active_cells());

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    vector<Point<dim>> coords_of_quadrature = fe_values.get_quadrature_points();  
    obj_coeff.value(coords_of_quadrature, coeff_values);
    
    for (unsigned int q=0; q</*1*/n_q_points; ++q)
    {
      inte_coeff_square+= pow(coeff_values[q][0][0], 2.0) * fe_values.JxW(q);                   // as can be seen, only restricted to the upper left component of the diffusion coefficient
                                                                                                // not finished
    }
    cellwise_sqrt_inte_coeff_square[cell->active_cell_index()] += sqrt(inte_coeff_square);
    
    inte_coeff_square=0;
  }
  
  double inte_l2_coeff = cellwise_sqrt_inte_coeff_square.l2_norm();
  return inte_l2_coeff;
}

template <typename T1, int dim>
double compute_l2_norms_of_coefficient_numerically_core_tensor_1_dim_real(Triangulation<dim>& triangulation,
                                                                  DoFHandler<dim>& dof_handler,
                                                                  FEValues<dim>& fe_values,
                                                                  T1& obj_coeff)
{
  unsigned int n_q_points = fe_values.get_quadrature_points().size();
  
  vector<Tensor<1,dim>> coeff_values(n_q_points, Tensor<1,dim>());                              // main change is here
  
  double inte_coeff_square;
  Vector<double> cellwise_sqrt_inte_coeff_square (triangulation.n_active_cells());

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    vector<Point<dim>> coords_of_quadrature = fe_values.get_quadrature_points();  
    obj_coeff.value(coords_of_quadrature, coeff_values);
    
    for (unsigned int q=0; q</*1*/n_q_points; ++q)
    {
      inte_coeff_square += pow(coeff_values[q][0], 2.0) * fe_values.JxW(q);                   // as can be seen, only restricted to the first component of the tensor<1,dim> object
                                                                                              // not finished because the unclearance of the definition of the l2 norm
    }
    cellwise_sqrt_inte_coeff_square[cell->active_cell_index()] += sqrt(inte_coeff_square);
    
    inte_coeff_square=0;
  }
  
  double inte_l2_coeff = cellwise_sqrt_inte_coeff_square.l2_norm();
  return inte_l2_coeff;
}


template <typename T, int dim>
double compute_l2_norms_of_coefficient_numerically_core_complex(Triangulation<dim>& triangulation,                 
                                                                     DoFHandler<dim>& dof_handler,
                                                                     FEValues<dim>& fe_values,
                                                                     T& obj_coeff)
{
  
  unsigned int n_q_points = fe_values.get_quadrature_points().size();
  
  vector<complex<double>> coeff_values(n_q_points, complex<double>());                // main change is here
  
  Vector<double> cellwise_inte_coeff_square(2);
  vector<Vector<double>> cellwise_sqrt_inte_coeff_square (triangulation.n_active_cells(), Vector<double>(2));
  
  Vector<double> inte_L2_coeff(2);
  
  double inte_L2_coeff_both = 0.0;
  

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    vector<Point<dim>> coords_of_quadrature = fe_values.get_quadrature_points();  
    obj_coeff.value(coords_of_quadrature, coeff_values);
    
    for (unsigned int k = 0; k < 2; ++k)
    {
      for (unsigned int q=0; q</*1*/n_q_points; ++q)
      {
        cellwise_inte_coeff_square[k]+= pow(abs(coeff_values[q].real()), 2.0) * fe_values.JxW(q);  
      }
      cellwise_sqrt_inte_coeff_square[cell->active_cell_index()][k] += sqrt(cellwise_inte_coeff_square[k]);
    }
    
    cellwise_inte_coeff_square.reinit(2);

  }

  for (unsigned int k = 0; k < 2; ++k)
  {
    for (unsigned int i = 0; i < triangulation.n_active_cells(); ++i)
    {
      inte_L2_coeff[k] += pow(cellwise_sqrt_inte_coeff_square[i][k],2.0);
    }
    
    inte_L2_coeff_both += inte_L2_coeff[k];
    inte_L2_coeff[k] = sqrt(inte_L2_coeff[k]);
  }
  
  inte_L2_coeff_both = sqrt(inte_L2_coeff_both);
//   cout << "inte_L2_coeff_both: " << inte_L2_coeff_both << "\n";
  
  return inte_L2_coeff[0];
  
}


template <typename T, int dim>
double compute_l2_norms_of_coefficient_numerically_core_tensor_2_dim_complex(Triangulation<dim>& triangulation,                 
                                                                     DoFHandler<dim>& dof_handler,
                                                                     FEValues<dim>& fe_values,
                                                                     T& obj_coeff)
{
  
  unsigned int n_q_points = fe_values.get_quadrature_points().size();
  
  vector<Tensor<2,dim, complex<double>>> coeff_values(n_q_points, Tensor<2,dim, complex<double>>());                // main change is here
  
  Vector<double> cellwise_inte_coeff_square(2);
  vector<Vector<double>> cellwise_sqrt_inte_coeff_square (triangulation.n_active_cells(), Vector<double>(2));
  
  Vector<double> inte_L2_coeff(2);
  
  double inte_L2_coeff_both = 0.0;

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    vector<Point<dim>> coords_of_quadrature = fe_values.get_quadrature_points();  
    obj_coeff.value(coords_of_quadrature, coeff_values);
    
    for (unsigned int k = 0; k < 2; ++k)
    {
      for (unsigned int q=0; q</*1*/n_q_points; ++q)
      {
        cellwise_inte_coeff_square[k]+= pow(abs(coeff_values[q][0][0].real()), 2.0) * fe_values.JxW(q);  
      }
      cellwise_sqrt_inte_coeff_square[cell->active_cell_index()][k] += sqrt(cellwise_inte_coeff_square[k]);
    }
    
    cellwise_inte_coeff_square.reinit(2);

  }

  for (unsigned int k = 0; k < 2; ++k)
  {
    for (unsigned int i = 0; i < triangulation.n_active_cells(); ++i)
    {
      inte_L2_coeff[k] += pow(cellwise_sqrt_inte_coeff_square[i][k],2.0);
    }
    
    inte_L2_coeff_both += inte_L2_coeff[k];
    inte_L2_coeff[k] = sqrt(inte_L2_coeff[k]);
  }
  
  inte_L2_coeff_both = sqrt(inte_L2_coeff_both);
  
  
  return inte_L2_coeff[0];
  
}

#endif
