
#ifndef FUNCTION_FOR_CUSTOM_L2_NORM_NUMERICAL_SOLUTION_MIX_CORE_H
#define FUNCTION_FOR_CUSTOM_L2_NORM_NUMERICAL_SOLUTION_MIX_CORE_H



template <int dim>
void l2_norm_computation_numerical_solution_mix_core(const DoFHandler<dim>& dof_handler,
                                                     const BlockVector<double>& solution,
                                                     const Zero_Function_Custom<dim>& obj_zero_function,
                                                     Vector<double>& cellwise_L2_norms,
                                                     const QIterated<dim>& quadrature_formula,
                                                     const Function< dim, double >* component_mask
                                                    )
{
  cout << "  l2_norm_computation_numerical_solution_mix_core()\n";

  VectorTools::integrate_difference (dof_handler, solution, obj_zero_function,
                                    cellwise_L2_norms, quadrature_formula,
                                    VectorTools::L2_norm,
                                    &component_mask);
  
}

#endif
