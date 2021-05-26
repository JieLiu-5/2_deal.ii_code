#ifndef DERIVED_CLASS_COMPLEX_H
#define DERIVED_CLASS_COMPLEX_H

template <int dim>
class PressureBoundaryValues_Complex :
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>
{
public:
  PressureBoundaryValues_Complex(const unsigned int id_case,
                                 const double coeff_var_inner_x);
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const;
protected:
  unsigned int id_case;
};


template <int dim>
class RightHandSide_Complex:
public ExactSolution_Step_4_Complex_Both<dim>,
public Coeff_Diff_Complex<dim>,
public Coeff_Helm_Complex<dim>
{
  public:
    RightHandSide_Complex (const unsigned int id_case,
                           const double coeff_var_inner_x,
                           const unsigned int id_coeff_diff_value,
                           const double coeff_diff_inner,
                           const unsigned int id_coeff_helm_value,
                           const double coeff_helm_inner);
    void vector_values_rhs (const vector<Point<dim>> &p,
                            vector<Vector<double>>   &value) const;
  protected:
    const unsigned int id_case;
    const double coeff_var_inner_x;
    const unsigned int id_coeff_diff_value;                   
    const double coeff_diff_inner;
    const unsigned int id_coeff_helm_value;                   
    const double coeff_helm_inner;
};

template <int dim>
PressureBoundaryValues_Complex<dim>::PressureBoundaryValues_Complex(const unsigned int id_case,
                                                                    const double coeff_var_inner_x):
Function<dim>(2),                                   // '2' is mandatory since this class is used together with 'dof_handler', which has two components
ExactSolution_Step_4_Complex_Both<dim>(id_case,
                                       coeff_var_inner_x),
id_case(id_case)
{}

template <int dim>
RightHandSide_Complex<dim>::RightHandSide_Complex(const unsigned int id_case,
                                                  const double coeff_var_inner_x,
                                                  const unsigned int id_coeff_diff_value,
                                                  const double coeff_diff_inner,
                                                  const unsigned int id_coeff_helm_value,
                                                  const double coeff_helm_inner):
ExactSolution_Step_4_Complex_Both<dim>(id_case,
                                       coeff_var_inner_x),
Coeff_Diff_Complex<dim>(id_case,
                        id_coeff_diff_value,
                        coeff_diff_inner),
Coeff_Helm_Complex<dim>(id_case,
                        id_coeff_helm_value,
                        coeff_helm_inner),
id_case(id_case),
coeff_var_inner_x(coeff_var_inner_x),
id_coeff_diff_value(id_coeff_diff_value),
coeff_diff_inner(coeff_diff_inner),
id_coeff_helm_value(id_coeff_helm_value),
coeff_helm_inner(coeff_helm_inner)
{}

template <int dim>
inline
void PressureBoundaryValues_Complex<dim>::vector_value (const Point<dim> &p,
                                                        Vector<double>   &values) const
{
  Assert (values.size() == 2, ExcDimensionMismatch (values.size(), 2));
  
  values(0) = ExactSolution_Step_4_Complex_Both<dim>::value_both(p).real();                    
  values(1) = ExactSolution_Step_4_Complex_Both<dim>::value_both(p).imag();
}

template <int dim>
void
RightHandSide_Complex<dim>::vector_values_rhs (const vector<Point<dim>> &p,
                                               vector<Vector<double>>   &values) const
{
    
//   cout << "RightHandSide_Complex<dim>::vector_values_rhs()\n";
    
  complex<double> value_u = 0.0 + 0.0i;
  Tensor<1,dim,complex<double>> gradient_u;
  SymmetricTensor<2,dim,complex<double>> hessian_u;
  
  vector<Tensor<2,dim, complex<double>>> vector_value_d(p.size(),
                                                        Tensor<2,dim, complex<double>>());
  Coeff_Diff_Complex<dim>::value(p, vector_value_d);
  
  vector<Tensor<2,dim, complex<double>>> vector_gradient_d_x_direction(p.size(),
                                                                       Tensor<2,dim, complex<double>>());
  vector<Tensor<2,dim, complex<double>>> vector_gradient_d_y_direction(p.size(),
                                                                       Tensor<2,dim, complex<double>>());
  Coeff_Diff_Complex<dim>::gradient_x_direction(p,
                                                vector_gradient_d_x_direction);
    
  vector<complex<double>> vector_value_r(p.size(),
                                         complex<double>());
  Coeff_Helm_Complex<dim>::value(p,
                                 vector_value_r);
    
//   cout << "p\n";
//   print_vector_horizontally(p);  
  
//   cout << "vector_value_d\n";
//   print_vector(vector_value_d);
//   
//   cout << "vector_gradient_d_x_direction\n";
//   print_vector(vector_gradient_d_x_direction);
//   
//   cout << "vector_value_r in RightHandSide_Complex<dim>::vector_values_rhs(): ";
//   print_vector_horizontally(vector_value_r);
  
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch(id_case)
    {
      case 1:
        values[i][0] = 1.0;
        values[i][1] = 0.0;        
        break;
      default:
        value_u = ExactSolution_Step_4_Complex_Both<dim>::value_both(p[i]);
        gradient_u = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p[i]);
        hessian_u = ExactSolution_Step_4_Complex_Both<dim>::hessian_both(p[i]);
        
        if(dim==1)
        {
          values[i][0] = (-(vector_gradient_d_x_direction[i][0][0]*gradient_u[0] + vector_value_d[i][0][0]*hessian_u[0][0]) + vector_value_r[i]*value_u).real();
          values[i][1] = (-(vector_gradient_d_x_direction[i][0][0]*gradient_u[0] + vector_value_d[i][0][0]*hessian_u[0][0]) + vector_value_r[i]*value_u).imag();
        }else if(dim == 2)
        {
          values[i][0] = (-((vector_gradient_d_x_direction[i][0][0]*gradient_u[0] + vector_gradient_d_y_direction[i][1][1]*gradient_u[1])+vector_value_d[i][0][0]*(hessian_u[0][0]+hessian_u[1][1]))+vector_value_r[i]*value_u).real();
          values[i][1] = (-((vector_gradient_d_x_direction[i][0][0]*gradient_u[0] + vector_gradient_d_y_direction[i][1][1]*gradient_u[1])+vector_value_d[i][0][0]*(hessian_u[0][0]+hessian_u[1][1]))+vector_value_r[i]*value_u).imag();
        }
        break;
    }
  }
}


#endif
