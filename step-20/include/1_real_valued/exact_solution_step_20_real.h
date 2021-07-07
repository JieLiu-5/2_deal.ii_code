
#ifndef EXACT_SOLUTION_STEP_20_REAL_H
#define EXACT_SOLUTION_STEP_20_REAL_H

template <int dim>
class ExactSolution_Step_20_Real : 
public Function<dim>,
public ExactSolution_Step_4_Real_Base<dim>,
public Coeff_Diff_Real<dim>
{
  public:
    ExactSolution_Step_20_Real (const unsigned int id_case,
                                const double coeff_var_inner_x,
                                const unsigned int id_coeff_diff_value,
                                const unsigned int id_loc_coeff_diff);

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    virtual void vector_gradient (const Point<dim> &p,
                                  std::vector<Tensor<1,dim>>   &gradients) const;   
                               
  protected:
    const unsigned int id_loc_coeff_diff;
};

template <int dim>
class GradientBoundary_Step_20_Real :                                   // !!! do not delete me because I cannot be replaced by ExactSolution_Step_20_Real<dim> by now
public ExactSolution_Step_20_Real<dim>
{
  public:
    GradientBoundary_Step_20_Real (const unsigned int id_case, const double coeff_var_inner_x, const unsigned int id_coeff_diff_value, const unsigned int id_loc_coeff_diff);
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  protected:
    const unsigned int id_loc_coeff_diff;                               // !!! do not delete me because we use different forms of the auxiliary variable v
};


template <int dim>
ExactSolution_Step_20_Real<dim>::ExactSolution_Step_20_Real(const unsigned int id_case, 
                                                            const double coeff_var_inner_x,
                                                            const unsigned int id_coeff_diff_value,
                                                            const unsigned int id_loc_coeff_diff):
Function<dim>(dim+1),
ExactSolution_Step_4_Real_Base<dim>(id_case,coeff_var_inner_x),
Coeff_Diff_Real<dim>(id_case,id_coeff_diff_value,1.0),
id_loc_coeff_diff(id_loc_coeff_diff)
{}

template<int dim>
GradientBoundary_Step_20_Real<dim>::GradientBoundary_Step_20_Real(const unsigned int id_case, 
                                                                  const double coeff_var_inner_x, 
                                                                  const unsigned int id_coeff_diff_value, 
                                                                  const unsigned int id_loc_coeff_diff):
ExactSolution_Step_20_Real<dim>(id_case,coeff_var_inner_x,id_coeff_diff_value,id_loc_coeff_diff),
id_loc_coeff_diff(id_loc_coeff_diff)
{}
  
  
template <int dim>
void
ExactSolution_Step_20_Real<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const                // size of values is dependent on dim, source?
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));
  
  vector<Point<dim>> vector_coords(1,p);                             // we define a vector because Coeff_Diff_Real<dim>::value() only allows for the input of a vector
  vector<Tensor<2,dim>> vector_value_d (1, Tensor<2,dim>());
  Coeff_Diff_Real<dim>::value(vector_coords,vector_value_d);
  
//   cout << "vector_value_d\n";
//   print_vector(vector_value_d);
  
  double value_p = ExactSolution_Step_4_Real_Base<dim>::value_base(p);
  Tensor<1,dim> gradient_p = ExactSolution_Step_4_Real_Base<dim>::gradient_base(p);
  
  Tensor<1,dim> value_u;
  if(id_loc_coeff_diff==0 or id_loc_coeff_diff==1)
  {
    value_u = - vector_value_d[0] * gradient_p;
  }else if(id_loc_coeff_diff==2)
  {
    value_u = - gradient_p;
  }
  
  for(unsigned int k=0; k<dim; ++k)
  {
    values(k) = value_u[k];    // u, affecting error of the first derivative   
  }
  values(dim) = value_p;                               // p, affecting error of the solution
}
  
template <int dim>
void
ExactSolution_Step_20_Real<dim>::vector_gradient (const Point<dim> &p,
                                                  std::vector<Tensor<1,dim>>   &gradients) const                // size of gradients is dependent on dim, source?
{      

  vector<Point<dim>> vector_coords(1,p);                                 // for in compliance with the class Coeff_Diff_Real, which might be modified later
  vector<Tensor<2,dim>> vector_value_d (1, Tensor<2,dim>());
  
//   vector<Tensor<2,dim,Vector<double>>> vector_gradient_d (1, Tensor<2,dim,Vector<double>>());
  
  vector<Tensor<2,dim>> vector_gradient_d_x_direction (1, Tensor<2,dim>());
  vector<Tensor<2,dim>> vector_gradient_d_y_direction (1, Tensor<2,dim>());
  
  Coeff_Diff_Real<dim>::value(vector_coords,vector_value_d);
//   Coeff_Diff_Real<dim>::gradient(vector_coords,vector_gradient_d);
  
  Coeff_Diff_Real<dim>::gradient_x_direction(vector_coords,vector_gradient_d_x_direction);
  Coeff_Diff_Real<dim>::gradient_y_direction(vector_coords,vector_gradient_d_y_direction);
    
  Tensor<1,dim> gradient_p = ExactSolution_Step_4_Real_Base<dim>::gradient_base(p);
  SymmetricTensor<2,dim> hessian_p = ExactSolution_Step_4_Real_Base<dim>::hessian_base(p);
  
  Tensor<2,dim> gradient_u;
  if(id_loc_coeff_diff==0 or id_loc_coeff_diff==1)
  {
    if(dim==1)
    {
      gradient_u[0][0] = -(vector_gradient_d_x_direction[0][0] * gradient_p + (vector_value_d[0] * hessian_p)[0][0]);    // du/dx, equal to rhs, affecting error of du/dx                                                                
    }else if(dim==2)
    {
//         gradients[0][0] = -((vector_gradient_d[0][0] * gradient_p)[0] + (vector_value_d[0] * hessian_p)[0][0]);        // not successful, jan 21, 2021        
      gradient_u[0][0] = -(vector_gradient_d_x_direction[0][0] * gradient_p + (vector_value_d[0] * hessian_p)[0][0]);
      gradient_u[0][1] = -(vector_gradient_d_y_direction[0][0] * gradient_p + (vector_value_d[0] * hessian_p)[0][1]);
      gradient_u[1][0] = -(vector_gradient_d_x_direction[0][1] * gradient_p + (vector_value_d[0] * hessian_p)[1][0]);
      gradient_u[1][1] = -(vector_gradient_d_y_direction[0][1] * gradient_p + (vector_value_d[0] * hessian_p)[1][1]);
    }    
  }else if(id_loc_coeff_diff==2)
  {
    gradient_u = - hessian_p;
  }
  
  for(unsigned int i=0; i<dim; ++i)
  {
    gradients[i] = gradient_u[i];
  }
  gradients[dim] = gradient_p;              // \nabla p, opposite of the gradient boundary for the Poisson equation 
}

template <int dim>
void
GradientBoundary_Step_20_Real<dim>::vector_value (const Point<dim> &p,
                                                  Vector<double>   &values) const
{
  Assert (values.size() == dim,
          ExcDimensionMismatch (values.size(), dim));
  
  Vector<double> boundary_gradient (dim+1);
  ExactSolution_Step_20_Real<dim>::vector_value(p, boundary_gradient);
  
  for(int i=0; i<dim; ++i )                         // only extracting the velocity part
  {
    values(i) = boundary_gradient[i];
  }
}


#endif


