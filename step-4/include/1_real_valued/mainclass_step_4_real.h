
#ifndef MAINCLASS_STEP_4_REAL_H
#define MAINCLASS_STEP_4_REAL_H

#include <deal.II/fe/fe_q_hierarchical.h>

#include"../../1_header/base_class_real.h"
#include <1_real_valued/exact_solution_step_4_real.h>

#include"../../1_header/derived_class_real.h"


template <int dim>
class Step_4_Real:
public Step_4_Number_Type_Independent<dim>
{
public:
  Step_4_Real (const int id_quad_assem_incre,
              const double tol_prm,
              const unsigned int is_custom_method_used_for_error,
              const unsigned int id_case,
              const double coeff_var_inner_x,
              const int id_coeff_diff_value, 
              const double coeff_diff_inner, 
              const int id_coeff_helm_value,
              const double coeff_helm_inner,
              const unsigned int degree,
              const unsigned int id_mesh_being_created,
              const unsigned int grid_param_initial,                        // for both the global mesh and the custom mesh
              const unsigned int n_total_refinements);
  
  void run ();
  
private:
  
  void from_setup_system_to_solve_for_the_real_valued_problem ();
  void stage_2a_setup_system_distributing_dofs_for_the_real_valued_problem();
  void stage_3_assemble_system_for_the_real_valued_problem ();
  void computing_norms_of_solution_numerically_for_real_valued_problems();
  void stage_6a_computing_errors_built_in_for_real_valued_problems ();
  void conducting_a_run_for_computing_the_error_customly_for_the_real_valued_problem ();
  
  FE_Q<dim>            fe;
  
  unsigned int is_receiving_iga=0;
  
  unsigned int is_strong = 1;
  double prm_penalty_weak = 1e6;                    // 1e6 50
  
  unsigned int is_basis_extracted = 0;
  
  unsigned int is_fe_info_printed = 0;
  
  unsigned int is_diri_boundary_value_printed = 0;
  
};


template <int dim>
Step_4_Real<dim>::Step_4_Real (const int id_quad_assem_incre,
                             const double tol_prm,
                             const unsigned int is_custom_method_used_for_error,
                             const unsigned int id_case,
                             const double coeff_var_inner_x,
                             const int id_coeff_diff_value, 
                             const double coeff_diff_inner, 
                             const int id_coeff_helm_value,
                             const double coeff_helm_inner,
                             const unsigned int degree,
                             const unsigned int id_mesh_being_created,
                             const unsigned int grid_param_initial,
                             const unsigned int n_total_refinements):
Step_4_Number_Type_Independent<dim>(id_quad_assem_incre,
                                   tol_prm,
                                   is_custom_method_used_for_error,
                                   id_case,
                                   coeff_var_inner_x,
                                   id_coeff_diff_value,
                                   coeff_diff_inner,
                                   id_coeff_helm_value,
                                   coeff_helm_inner,
                                   degree,
                                   id_mesh_being_created,
                                   grid_param_initial,
                                   n_total_refinements),
fe (degree)
{
//     cout << "================== \n"
//          << "  The standard FEM for real-valued problems\n"
//          << "  dimension: " << dim << "\n";
//     cout << "==================\n";
}


template <int dim>
void Step_4_Real<dim>::stage_2a_setup_system_distributing_dofs_for_the_real_valued_problem()
{
  TimerOutput::Scope t(this->computing_timer, "stage_2a_setup_system_distributing_dofs");
    
  if(this->n_times_called_setup==0)
  {
    if(is_basis_extracted==1)
    {
      extract_basis_function(fe);  
    }  
      
    if(is_fe_info_printed==1)
    {
        this->obj_string = "fe";
        print_fe_info(this->obj_string, fe);  
    }
  }
  
  this->dof_handler.distribute_dofs (fe);
}


template <int dim>
void Step_4_Real<dim>::stage_3_assemble_system_for_the_real_valued_problem ()
{
//   cout << "Assembling\n";
  TimerOutput::Scope t(this->computing_timer, "stage_3_assemble_system");
  QGauss<dim-1> face_quadrature_formula(this->degree+2);   
  
  const Coeff_Diff_Real<dim> obj_coeff_diff(this->id_case,
                                            this->id_coeff_diff_value,
                                            this->coeff_diff_inner);
  const Coeff_First_Derivative_A_Real<dim> obj_coeff_first_derivative_a(this->id_case,
                                                                        1);                        // second argument not used
  
  const Coeff_Helm_Real<dim> obj_coeff_helm(this->id_case,
                                            this->id_coeff_helm_value,
                                            this->coeff_helm_inner); 
  const RightHandSide_Real<dim> obj_right_hand_side(this->id_case,
                                                    this->coeff_var_inner_x,
                                                    this->id_coeff_diff_value,
                                                    this->coeff_diff_inner,
                                                    this->id_coeff_helm_value,
                                                    this->coeff_helm_inner);

  FEValues<dim> fe_values (fe, this->quadrature_formula,
                           update_values   | update_gradients | update_hessians |
                           update_quadrature_points | update_JxW_values | update_jacobians);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  
  vector<Tensor<2,dim>> value_coeff_diff (this->n_q_points, Tensor<2,dim>());
  vector<Tensor<2,dim>> value_coeff_diff_face (n_face_q_points, Tensor<2,dim>());
  
  vector<Tensor<1,dim>> value_coeff_first_derivative_a (this->n_q_points, Tensor<1,dim>());
  
  vector<double> value_coeff_helm (this->n_q_points);
  
  vector<Vector<double>> value_rhs (this->n_q_points, Vector<double>(this->n_number_components));
  
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

//   Computing_L2_Norms_of_Coefficients<T, dim> obj_computing_l2_norms_of_coefficients(this->triangulation,                 // not possible because it contains two templates
//                                                                                     this->dof_handler,                   // which is more than the number of templates in step-4
//                                                                                     fe_values,
//                                                                                     obj_coeff_diff);
  
  
  const ExactSolution_Step_4_Real<dim> exact_solution(this->id_case,                                    // used for every cell
                                                      this->coeff_var_inner_x);  
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = this->dof_handler.begin_active(),
  endc = this->dof_handler.end();
  
  fe_values.reinit (cell);
  
  if(this->is_l2_norm_coefficient_calculated_numerically == 1 && (this->n_times_called_assemble%2 == 0))
  {
    this->l2_norm_coeff_diff = compute_l2_norms_of_coefficient_numerically_core_tensor_2_dim_real(this->triangulation,
                                                                                                  this->dof_handler,
                                                                                                  fe_values,
                                                                                                  obj_coeff_diff);
    this->l2_norm_coeff_helm = compute_l2_norms_of_coefficient_numerically_core_real(this->triangulation,
                                                                                     this->dof_handler,
                                                                                     fe_values,
                                                                                     obj_coeff_helm);
    this->l2_norm_coeff_first_derivative_a = compute_l2_norms_of_coefficient_numerically_core_tensor_1_dim_real(this->triangulation,
                                                                                                                this->dof_handler,
                                                                                                                fe_values,
                                                                                                                obj_coeff_first_derivative_a);
    
    this->saving_norms_of_various_coefficients_to_vectors();
  }     
  
  
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    cell_matrix = 0;
    cell_rhs = 0;
    
    obj_coeff_diff.value (fe_values.get_quadrature_points(), value_coeff_diff);
    
//     cout << "value_coeff_diff: \n";
//     print_vector_vertically(value_coeff_diff);
//     
//     for(unsigned int i = 0; i<value_coeff_diff.size(); ++i)
//     {
//       cout << std::scientific << value_coeff_diff[i] << "\n";
//     }
    
    
    obj_coeff_first_derivative_a.value (fe_values.get_quadrature_points(), value_coeff_first_derivative_a);
    
    obj_coeff_helm.value (fe_values.get_quadrature_points(), value_coeff_helm);
    
    obj_right_hand_side.value_rhs (fe_values.get_quadrature_points(), value_rhs);
    
    
//     cout << "value_coeff_first_derivative_a\n";
//     print_vector_vertically(value_coeff_first_derivative_a);
    
    for (unsigned int q_index=0; q_index<this->n_q_points; ++q_index)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          if (this->id_case!=9)
          {
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) * (value_coeff_diff[q_index] * fe_values.shape_grad (j, q_index))
                                   + fe_values.shape_value (i, q_index) * (fe_values.shape_value (j, q_index) * value_coeff_helm[q_index]            // shape_value() defined in fe_values.h
                                                                          + value_coeff_first_derivative_a[q_index]*fe_values.shape_grad (j, q_index)))
                                   * fe_values.JxW (q_index);
          }
          
          if(this->id_case==9)                           // dealing with first-order differential equations
          {
            cell_matrix(i,j) -= fe_values.shape_value (i, q_index) *
                             fe_values.shape_grad (j, q_index)[0] *
                             fe_values.JxW (q_index);
          } 
        }
        
        cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                        value_rhs[q_index][0] *
                        fe_values.JxW (q_index));
      }
    }
    
    for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
    {
      if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 1))                                  // treat the Neumann boundary conditions
      {
             
        fe_face_values.reinit (cell, face_n);
        
        obj_coeff_diff.value (fe_face_values.get_quadrature_points(), value_coeff_diff_face);

        for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
        {
          const double neumann_value
          = (value_coeff_diff_face[q_point]*exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
            fe_face_values.normal_vector(q_point));
          
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            cell_rhs(i) += (neumann_value *
                           fe_face_values.shape_value(i,q_point) *
                           fe_face_values.JxW(q_point));
          }
        }
      }
    }
      
    cell->get_dof_indices (local_dof_indices);
    
    
    if (this->id_basis==0)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          this->system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));
        }

        this->system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }  
    }else if(this->id_basis==1)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          this->system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));
        }

        this->system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    }
  }
  
  if (this->is_lhs_and_rhs_before_diri_BC_printed==1)
  {
    this->printing_lhs_and_rhs_before_applying_Diri_BC();
  }    

  if (is_strong == 1)
  {
    map<types::global_dof_index,double> boundary_values;
    
     ExactSolution_Step_4_Real<dim> obj_exact_solution(this->id_case,
                                                        this->coeff_var_inner_x);
    
    VectorTools::interpolate_boundary_values (this->dof_handler,
                                              0,
                                              ExactSolution_Step_4_Real<dim>(this->id_case,
                                                                             this->coeff_var_inner_x),                  // note we only use the value item
                                              boundary_values);
    
    
    if (is_diri_boundary_value_printed == 1)
    {
        cout << "    Dirichlet boundary values: \n";
        print_map(boundary_values);
    }

    MatrixTools::apply_boundary_values (boundary_values,
                                        this->system_matrix,
                                        this->solution,
                                        this->system_rhs);
    
//     implementing_the_dirichlet_boundary_condition(obj_exact_solution, fe_values);                       // not successful
    
  }else if (is_strong == 0)
  {
    cout << "    weak way\n";
    ExactSolution_Step_4_Real<dim> boundary_values_for_weak_imposition(this->id_case,
                                                                       this->coeff_var_inner_x);  
    
    vector<Point<dim>> coords_quad_boundary(2);
    coords_quad_boundary[0](0)=0.0;
    coords_quad_boundary[1](0)=1.0;
    
    FEValues<dim> fe_values_boundary (fe, coords_quad_boundary,
                             update_values   | update_gradients | update_quadrature_points );
  
    this->dof_handler.distribute_dofs (fe);  
    
    typename DoFHandler<dim>::active_cell_iterator 
    first_cell = this->dof_handler.begin_active(), 
    intermediate_cell = this->dof_handler.begin_active(),
    last_cell = this->dof_handler.begin_active();
  
    fe_values_boundary.reinit(first_cell);
    first_cell->get_dof_indices (local_dof_indices);

    if (first_cell->face(0)->boundary_id() == 0) 
    { 
      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        this->system_matrix.add (local_dof_indices[0],
                            local_dof_indices[j],
                            fe_values_boundary.shape_grad(j,0)[0]);
        this->system_matrix.add (local_dof_indices[j],
                           local_dof_indices[0],
                            -fe_values_boundary.shape_grad(j,0)[0]);  
        this->system_matrix.add (local_dof_indices[0], local_dof_indices[0], prm_penalty_weak);  
        
        this->system_rhs(local_dof_indices[j]) -= fe_values_boundary.shape_grad(j,0)[0]*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
        this->system_rhs(local_dof_indices[0]) += prm_penalty_weak*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
      }
    }

    for(unsigned int i = 0; i < this->triangulation.n_active_cells()-1; ++i)
    {
      ++intermediate_cell;
    }
  
    last_cell=intermediate_cell;
  
    fe_values.reinit (last_cell);
    last_cell->get_dof_indices (local_dof_indices);

    if (last_cell->face(1)->boundary_id() == 0)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        this->system_matrix.add (local_dof_indices[1],
                            local_dof_indices[j],
                            -fe_values_boundary.shape_grad(j,1)[0]);
        this->system_matrix.add (local_dof_indices[j],
                           local_dof_indices[1],
                            fe_values_boundary.shape_grad(j,1)[0]);  
        this->system_matrix.add (local_dof_indices[1], local_dof_indices[1], -prm_penalty_weak);  
        
        this->system_rhs(local_dof_indices[j]) += fe_values_boundary.shape_grad(j,1)[0]*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
        this->system_rhs(local_dof_indices[1]) -= prm_penalty_weak*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
      }
    }
  } else
  {
    cout << "  Dirichlet boundary conditions not treated\n";
  }

  if (this->is_lhs_and_rhs_after_diri_BC_printed==1)
  {
    this->printing_lhs_and_rhs_after_applying_Diri_BC();
  }

  if (this->is_lhs_and_rhs_after_diri_BC_saved == 1)
  {
    this->saving_lhs_and_rhs_after_applying_Drii_BC();  
  }
  
  this->n_times_called_assemble++;
  
}


template <int dim>
void Step_4_Real<dim>::computing_norms_of_solution_numerically_for_real_valued_problems ()
{
//   cout << "Computing integrals\n";
  const Functions::ZeroFunction<dim> zero_function;                          // defined in /base/function.h
  
  const ExactSolution_Step_4_Real<dim> exact_solution(this->id_case, this->coeff_var_inner_x);
    
  Vector<double> difference_per_cell (this->triangulation.n_active_cells());
  
  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution,
                                     zero_function,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::L2_norm);
  this->solution_L2_norm_numerical = VectorTools::compute_global_error(this->triangulation,
                                                          difference_per_cell,
                                                          VectorTools::L2_norm);
  
  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution,
                                     zero_function,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  this->solution_H1_semi_norm_numerical = VectorTools::compute_global_error(this->triangulation,
                                                          difference_per_cell,
                                                          VectorTools::H1_seminorm);  
  
  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution,
                                     zero_function,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  this->solution_H2_semi_norm_numerical = VectorTools::compute_global_error(this->triangulation,
                                                          difference_per_cell,
                                                          VectorTools::H2_seminorm);  
  
  /*
  
  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution_zero,
                                     exact_solution,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::L2_norm);
  this->solution_L2_norm_analytical = VectorTools::compute_global_error(this->triangulation,
                                                            difference_per_cell,
                                                            VectorTools::L2_norm);

  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution_zero,
                                     exact_solution,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  this->solution_H1_semi_norm_analytical = VectorTools::compute_global_error(this->triangulation,
                                                            difference_per_cell,
                                                            VectorTools::H1_seminorm);

  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution_zero,
                                     exact_solution,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);                                  // selft-defined
  
  this->solution_H2_semi_norm_analytical = VectorTools::compute_global_error(this->triangulation,
                                                                     difference_per_cell,
                                                                     VectorTools::H2_seminorm); 
  
  if(dim==2)
  {
    VectorTools::integrate_difference (this->dof_handler,
                                        this->solution_zero,
                                        exact_solution,
                                        difference_per_cell,
                                        this->qgauss_for_integrating_difference,
                                        VectorTools::H2_divnorm);                                  // selft-defined
    this->solution_Hdiv_semi_norm_analytical = VectorTools::compute_global_error(this->triangulation,
                                                                difference_per_cell,
                                                                VectorTools::H2_divnorm);
  }
  
  */
}

template <int dim>
void Step_4_Real<dim>::stage_6a_computing_errors_built_in_for_real_valued_problems ()
{
//   cout << "Computing errors using built-in functions\n";
  TimerOutput::Scope t(this->computing_timer, "stage_6a_computing_errors_built_in");
  
  ExactSolution_Step_4_Real<dim> exact_solution(this->id_case, this->coeff_var_inner_x);
  
  Vector<double> difference_per_cell (this->triangulation.n_active_cells());
  
#if 1
  
  VectorTools::integrate_difference (this->dof_handler,
                                    this->solution,
                                    exact_solution,
                                    difference_per_cell,
                                    this->qgauss_for_integrating_difference,
                                    VectorTools::L2_norm);
  this->solution_L2_error_abs_built_in = VectorTools::compute_global_error(this->triangulation,
                                                   difference_per_cell,
                                                   VectorTools::L2_norm);

  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution,
                                     exact_solution,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  this->solution_H1_semi_error_abs_built_in = VectorTools::compute_global_error(this->triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H1_seminorm);

#endif

  VectorTools::integrate_difference (this->dof_handler,
                                     this->solution,
                                     exact_solution,
                                     difference_per_cell,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  this->solution_H2_semi_error_abs_built_in = VectorTools::compute_global_error(this->triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H2_seminorm);
  
  /*
  if(dim==2)
  {
    VectorTools::integrate_difference (this->dof_handler,
                                        this->solution,
                                        exact_solution,
                                        difference_per_cell,
                                        this->qgauss_for_integrating_difference,
                                        VectorTools::H2_divnorm);
    this->solution_Hdiv_semi_error_abs = VectorTools::compute_global_error(this->triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H2_divnorm);

    VectorTools::integrate_difference (this->dof_handler,
                                       this->solution,
                                       exact_solution,
                                       difference_per_cell,
                                       this->qgauss_for_integrating_difference,
                                       VectorTools::H2_seminorm_00);
    this->solution_H2_semi_error_abs_00 = VectorTools::compute_global_error(this->triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_00);

    VectorTools::integrate_difference (this->dof_handler,
                                       this->solution,
                                       exact_solution,
                                       difference_per_cell,
                                       this->qgauss_for_integrating_difference,
                                       VectorTools::H2_seminorm_01);
    this->solution_H2_semi_error_abs_01 = VectorTools::compute_global_error(this->triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_01);  
  
    VectorTools::integrate_difference (this->dof_handler,
                                       this->solution,
                                       exact_solution,
                                       difference_per_cell,
                                       this->qgauss_for_integrating_difference,
                                       VectorTools::H2_seminorm_10);
    this->solution_H2_semi_error_abs_10 = VectorTools::compute_global_error(this->triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_10);  

    VectorTools::integrate_difference (this->dof_handler,
                                       this->solution,
                                       exact_solution,
                                       difference_per_cell,
                                       this->qgauss_for_integrating_difference,
                                       VectorTools::H2_seminorm_11);
    this->solution_H2_semi_error_abs_11 = VectorTools::compute_global_error(this->triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_11);  
  }*/
  
  this->n_times_called_computing_error_built_in++;
}


template <int dim>
void Step_4_Real<dim>::from_setup_system_to_solve_for_the_real_valued_problem ()
{
  stage_2a_setup_system_distributing_dofs_for_the_real_valued_problem();
  this->stage_2b_setup_system_the_rest ();
  
  stage_3_assemble_system_for_the_real_valued_problem ();
  this->stage_4_solve ();
}


template <int dim>
void Step_4_Real<dim>::conducting_a_run_for_computing_the_error_customly_for_the_real_valued_problem ()
{
//   cout << "Computing errors customly\n";
//   printPattern(5);
  
  this->is_refining_mesh_for_an_independent_run = 0;
  this->refining_mesh_globally(1);
  this->printing_current_grid_status();
  this->current_refinement_level_for_showing--;                                 // back to the refinement level of the active mesh
  
  from_setup_system_to_solve_for_the_real_valued_problem ();

  this->stage_6b_computing_the_error_customly();
  
}
  

template <int dim>
void Step_4_Real<dim>::run ()
{
    
  this->id_type_of_number = 0;
  this->n_number_components = 1;    
  if(dim == 1)
  {
    this->is_containing_neumann = 0;
  }else if(dim == 2)
  {
    this->is_containing_neumann = 1;
  }
  
  this->preparing_for_the_initial_grid_and_other_settings();
  
  for (unsigned int cycle = 0; cycle < this->n_total_refinements; cycle++)                        // cycle used for adaptive mesh refinement
  {
    this->cycle_global = cycle;

#if 1
    
    if (this->is_mesh_uniform_computed == 1)
    {
      this->is_before_mesh_being_distorted = 1;
      
      from_setup_system_to_solve_for_the_real_valued_problem ();

      this->n_vertices_first_refine = this->triangulation.n_vertices();
      this->n_dofs_first_refine = this->dof_handler.n_dofs();
    
      this->stage_5_dealing_with_the_result_of_the_first_run_that_is_independent_of_the_type_of_number();

      if (this->are_norms_of_solution_computed_numerically==1)
      {
        computing_norms_of_solution_numerically_for_real_valued_problems ();
      }
      
      if(this->is_built_in_method_used_for_error == 1)
      {
        stage_6a_computing_errors_built_in_for_real_valued_problems ();
      }
      
      this->copying_triangulation_and_solution_of_the_last_refinement_for_further_refinements();
      this->dof_handler_first_refine.distribute_dofs (fe);
      
      if (this->is_custom_method_used_for_error == 1)
      {   
        conducting_a_run_for_computing_the_error_customly_for_the_real_valued_problem ();
      }      
      
      this->dealing_with_the_error_and_cpu_time_etc_after_one_complete_run();
    }
    
    if(this->is_mesh_distorted_computed == 1)                       // only the input mesh is different from that of the uniform case
    {
      this->distorting_the_mesh();
      
#if 0
      
      this->is_before_mesh_being_distorted = 0;
      
      from_setup_system_to_solve_for_the_real_valued_problem ();
      
      this->stage_5_dealing_with_the_result_of_the_first_run_that_is_independent_of_the_type_of_number();
      
      if(this->is_built_in_method_used_for_error == 1)
      {
        stage_6a_computing_errors_built_in_for_real_valued_problems ();
      }
      
      this->copying_triangulation_and_solution_of_the_last_refinement_for_further_refinements();
      this->dof_handler_first_refine.distribute_dofs (fe);      
      
      if (this->is_custom_method_used_for_error == 1)
      {   
        conducting_a_run_for_computing_the_error_customly_for_the_real_valued_problem ();
      }      
      
      this->dealing_with_the_error_and_cpu_time_etc_after_one_complete_run();
      
#endif
    }
    
    if(cycle < this->n_total_refinements - 1)
    {
      this->preparing_for_the_next_run();
    }
    
#endif

  }
  
  if(this->is_results_of_all_refinements_printed == 1)
  {
      this->printing_results_of_all_refinements();
  }
  
}

#endif
