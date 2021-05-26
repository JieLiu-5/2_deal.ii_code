#ifndef DERIVED_CLASS_REAL_H
#define DERIVED_CLASS_REAL_H

template <int dim>
class RightHandSide_Real :
public ExactSolution_Step_4_Real<dim>,
public Coeff_Diff_Real<dim>,
public Coeff_First_Derivative_A_Real<dim>,
public Coeff_Helm_Real<dim>
{
  public:
    using Coeff_Diff_Real<dim>::Coeff_Diff_Real;
    
    RightHandSide_Real (const unsigned int id_case,
                        const double coeff_var_inner_x,
                        const unsigned int id_coeff_diff_value,
                        const double coeff_diff_inner,
                        const unsigned int id_coeff_helm_value,
                        const double coeff_helm_inner);

    void value_rhs (const vector<Point<dim>>   &p,
                    vector<Vector<double>>     &values) const;
};


template <int dim>
RightHandSide_Real<dim>::RightHandSide_Real(const unsigned int id_case,
                                            const double coeff_var_inner_x,
                                            const unsigned int id_coeff_diff_value,
                                            const double coeff_diff_inner,
                                            const unsigned int id_coeff_helm_value,
                                            const double coeff_helm_inner): 
ExactSolution_Step_4_Real<dim>(id_case, coeff_var_inner_x),
Coeff_Diff_Real<dim>(id_case, id_coeff_diff_value, coeff_diff_inner),
Coeff_First_Derivative_A_Real<dim>(id_case, 1),
Coeff_Helm_Real<dim>(id_case, id_coeff_helm_value, coeff_helm_inner)
{}


template <int dim>
void RightHandSide_Real<dim>::value_rhs (const vector<Point<dim>>  &p,
                                         vector<Vector<double>>     &values) const
{
  vector<double> vector_value_p(p.size());
  vector<Tensor<1,dim>> vector_gradient_p(p.size(),Tensor<1,dim>());
  vector<SymmetricTensor<2,dim>> vector_hessian_p(p.size(),SymmetricTensor<2,dim>());
  
  vector<Tensor<2,dim>> vector_value_d(p.size(),Tensor<2,dim>());
  vector<Tensor<2,dim>> vector_gradient_d_x_direction (p.size(), Tensor<2,dim>());
  vector<Tensor<2,dim>> vector_gradient_d_y_direction (p.size(), Tensor<2,dim>());
  
  vector<Tensor<1,dim>> vector_contribution_gradient_d (p.size(), Tensor<1,dim>());
  
  vector<Tensor<1,dim>> vector_value_a(p.size(),Tensor<1,dim>());
  vector<double> vector_value_r(p.size());
  
  ExactSolution_Step_4_Real<dim>::value_list(p,vector_value_p);
  ExactSolution_Step_4_Real<dim>::gradient_list(p,vector_gradient_p);
  ExactSolution_Step_4_Real<dim>::hessian_list(p,vector_hessian_p);
  
  Coeff_Diff_Real<dim>::value(p,vector_value_d);
  Coeff_Diff_Real<dim>::gradient_x_direction(p,vector_gradient_d_x_direction);
  Coeff_Diff_Real<dim>::gradient_y_direction(p,vector_gradient_d_y_direction);
  
  Coeff_First_Derivative_A_Real<dim>::value(p,vector_value_a);
  
  Coeff_Helm_Real<dim>::value(p,vector_value_r);
  
  double coeff_a_times_nabla_u = 0.0;
  double coeff_r_times_u = 0.0;
  
  for(unsigned int i=0; i<values.size(); ++i)
  {
    vector_contribution_gradient_d[i] = vector_gradient_d_x_direction[i][0];
    if(dim==2)
    {
      vector_contribution_gradient_d[i] += vector_gradient_d_y_direction[i][1];
    }
    
    coeff_a_times_nabla_u = vector_value_a[i]*vector_gradient_p[i];
    coeff_r_times_u = vector_value_r[i] * vector_value_p[i];
    
    values[i][0] = -(vector_contribution_gradient_d[i] * vector_gradient_p[i] 
                     + trace(vector_value_d[i]*vector_hessian_p[i]))
                   + coeff_a_times_nabla_u + coeff_r_times_u;                   
  }
}


#endif
