
#ifndef MAINCLASS_STEP_4_COMPLEX_H
#define MAINCLASS_STEP_4_COMPLEX_H

#include"../../1_header/base_class_complex.h"
#include <2_complex_valued/exact_solution_step_4_complex.h>

#include"../../1_header/derived_class_complex.h"

template <int dim>
class Step_4_Complex:
public Step_4_Number_Type_Independent<dim>
{
public:
  Step_4_Complex (const int id_quad_assem_incre, 
                 const double tol_prm,
                 const unsigned int is_custom_method_used_for_error,
                 const unsigned int id_case,
                 const double coeff_var_inner_x,
                 const int id_coeff_diff_value, 
                 const double coeff_diff_inner, 
                 const int id_coeff_helm_value,
                 const double coeff_helm_inner,
                 const double length, 
                 const unsigned int degree,
                 const unsigned int id_mesh_being_created,
                 const int grid_param_initial,
                 const unsigned int n_total_refinements);
  void run ();
  
  double solution_real_L2_norm_numerical, solution_imag_L2_norm_numerical;
  
private:
    
  void stage_2a_setup_system_distributing_dofs_for_the_complex_valued_problem();
  void from_setup_system_to_solve_for_the_complex_valued_problem();
  void setup_system_for_dof_handler_single();
  void stage_3_assemble_system_for_the_complex_valued_problem ();
  void splitting_real_and_imag_part_of_the_solution();
  void computing_norms_of_solution_numerically_for_complex_valued_problems ();
  void stage_6a_computing_errors_built_in_for_complex_valued_problems ();

  void conducting_a_run_for_computing_the_error_customly_for_the_complex_valued_problem ();
  

  const double length;
  
  QGauss<dim-1> face_quadrature_formula = QGauss<dim-1>(this->degree+2);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  
  FESystem<dim>        fe;
  FE_Q<dim>            fe_single;
  
  DoFHandler<dim>      dof_handler_single;
  
  Vector<double>       solution_amp;
  
  Vector<double>       solution_real;
  Vector<double>       solution_real_first_refine;
  
  Vector<double>       solution_imag;
  Vector<double>       solution_imag_second_refine;
  
  unsigned int is_l2_norm_of_the_numerical_real_and_imaginary_parts_printed = 0;
  
};


template <int dim>
Step_4_Complex<dim>::Step_4_Complex (const int id_quad_assem_incre,
                                   const double tol_prm,
                                   const unsigned int is_custom_method_used_for_error, 
                                   const unsigned int id_case, 
                                   const double coeff_var_inner_x,
                                   const int id_coeff_diff_value, 
                                   const double coeff_diff_inner, 
                                   const int id_coeff_helm_value,
                                   const double coeff_helm_inner,
                                   const double length,
                                   const unsigned int degree,
                                   const unsigned int id_mesh_being_created,
                                   const int grid_param_initial,
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
length(length),
fe (FE_Q<dim>(degree),2),
fe_single (degree),
dof_handler_single (this->triangulation)
{
/*  cout << "================== \n"
       << "  The standard FEM for complex-valued problems\n"
       << "  case: " << this->id_case << "\n"
       << "  dimension: " << dim << "\n"
       << "  n_q_points: " << this->n_q_points << "\n"
       << "  n_face_q_points: " << n_face_q_points << "\n";    
  cout << "================== \n";  */
}


template <int dim>
void Step_4_Complex<dim>::setup_system_for_dof_handler_single ()
{
  dof_handler_single.distribute_dofs (fe_single);    
  
  solution_amp.reinit (dof_handler_single.n_dofs());
  
  solution_real.reinit (dof_handler_single.n_dofs());
  solution_imag.reinit (dof_handler_single.n_dofs());
}


template <int dim>
void Step_4_Complex<dim>::stage_2a_setup_system_distributing_dofs_for_the_complex_valued_problem()
{
  TimerOutput::Scope t(this->computing_timer, "stage_2a_setup_system_distributing_dofs");
    
  this->dof_handler.distribute_dofs (fe);
  setup_system_for_dof_handler_single();    
}


template <int dim>
void Step_4_Complex<dim>::stage_3_assemble_system_for_the_complex_valued_problem ()
{
  TimerOutput::Scope t(this->computing_timer, "stage_3_assemble_system");
//   cout << "Assembling\n";
  
  FEValues<dim> fe_values (fe, this->quadrature_formula,
                           update_values   | update_gradients | update_hessians |
                           update_quadrature_points | update_JxW_values);
  
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);  

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   dofs_per_face = fe.dofs_per_face;
  
  const RightHandSide_Complex<dim> right_hand_side(this->id_case,
                                                   this->coeff_var_inner_x,
                                                   this->id_coeff_diff_value,
                                                   this->coeff_diff_inner,
                                                   this->id_coeff_helm_value,
                                                   this->coeff_helm_inner);
  const Coeff_Diff_Complex<dim> obj_coeff_diff(this->id_case,
                                               this->id_coeff_diff_value,
                                               this->coeff_diff_inner);
  const Coeff_Helm_Complex<dim> obj_coeff_helm(this->id_case,
                                               this->id_coeff_helm_value,
                                               this->coeff_helm_inner);
  
  vector<Point<dim>> coords_of_quadrature_on_cell (this->n_q_points, Point<dim>());
  vector<Vector<double>> rhs_values (this->n_q_points,Vector<double>(2));
  vector<Tensor<2,dim, complex<double>>> coeff_diff_values(this->n_q_points,Tensor<2,dim, complex<double>>());
  vector<complex<double>> coeff_helm_values(this->n_q_points,complex<double>());
  
  vector<Tensor<2,dim, complex<double>>> coeff_diff_values_face(n_face_q_points,Tensor<2,dim, complex<double>>());
  vector<Vector<double>> coeff_helm_face_values(n_face_q_points,Vector<double>(2));
  
  
  Tensor<1,dim, complex<double>> coeff_diff_times_gradient;
  complex<double> gradient_times_coeff_diff_times_gradient;
  
  complex<double> coeff_helm_times_value;
  complex<double> value_times_coeff_helm_times_value;
  
  complex<double> diff_and_helm_together;
  
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  
  vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  vector<types::global_dof_index> local_face_dof_indices (dofs_per_face);
  
  
  const ExactSolution_Step_4_Complex_Both<dim> exact_solution_both(this->id_case,
                                                                   this->coeff_var_inner_x);                // all cases are analyzed upon creating an object
  
  Tensor<1,dim, complex<double>> neumann_boundary;
  complex<double> neumann_boundary_times_normal_vector = 0.0 + 0.0i;

  typename DoFHandler<dim>::active_cell_iterator
  cell = this->dof_handler.begin_active(),
  endc = this->dof_handler.end();
  
  fe_values.reinit (cell);
  
  
  if(this->is_l2_norm_coefficient_calculated_numerically == 1  && (this->n_times_called_assemble%2 == 0))
  {  
    this->l2_norm_coeff_diff = compute_l2_norms_of_coefficient_numerically_core_tensor_2_dim_complex(this->triangulation,                   // only computing the upper left of the real part
                                                                                                                      this->dof_handler,
                                                                                                                      fe_values,
                                                                                                                      obj_coeff_diff);
    this->l2_norm_coeff_helm = compute_l2_norms_of_coefficient_numerically_core_complex(this->triangulation,
                                                                                        this->dof_handler,
                                                                                        fe_values,
                                                                                        obj_coeff_helm);
    this->saving_norms_of_various_coefficients_to_vectors();
  }
  

  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    cell->get_dof_indices (local_dof_indices);
    
    cell_matrix = 0;
    cell_rhs = 0;
    
    coords_of_quadrature_on_cell = fe_values.get_quadrature_points();
    
    right_hand_side.vector_values_rhs(coords_of_quadrature_on_cell, rhs_values);
    
    obj_coeff_diff.value(coords_of_quadrature_on_cell, coeff_diff_values);
    obj_coeff_helm.value(coords_of_quadrature_on_cell, coeff_helm_values);
    
    for (unsigned int q_index=0; q_index</*1*/ this->n_q_points; ++q_index)             //diagonal part
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          coeff_diff_times_gradient = coeff_diff_values[q_index] * fe_values.shape_grad (j, q_index);
          gradient_times_coeff_diff_times_gradient = fe_values.shape_grad (i, q_index) * coeff_diff_times_gradient;
          
          coeff_helm_times_value = coeff_helm_values[q_index] * fe_values.shape_value (j, q_index);
          value_times_coeff_helm_times_value = fe_values.shape_value (i, q_index)*coeff_helm_times_value;
          
          diff_and_helm_together = gradient_times_coeff_diff_times_gradient + value_times_coeff_helm_times_value;
          
          if (fe.system_to_component_index(i).first == 0)
          {
            if (fe.system_to_component_index(j).first == 0)
            {
              cell_matrix(i,j) += diff_and_helm_together.real() *fe_values.JxW(q_index);
            }else if (fe.system_to_component_index(j).first == 1)
            {
              cell_matrix(i,j) -= diff_and_helm_together.imag() *fe_values.JxW(q_index);
            }
          }else if (fe.system_to_component_index(i).first == 1)
          {
            if (fe.system_to_component_index(j).first == 0)
            {
              cell_matrix(i,j) += diff_and_helm_together.imag() *fe_values.JxW(q_index);
            }else if (fe.system_to_component_index(j).first == 1)
            {
              cell_matrix(i,j) += diff_and_helm_together.real() *fe_values.JxW(q_index);
            }              
          }
        }
        cell_rhs(i) += fe_values.shape_value (i, q_index) 
                       * rhs_values[q_index][fe.system_to_component_index(i).first]
                       * fe_values.JxW (q_index);
      } 
    }
    
#if 1
    
    for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      if (cell->face(face_number)->at_boundary())
      {
        fe_face_values.reinit (cell, face_number);
        cell->face(face_number)->get_dof_indices (local_face_dof_indices);
        
        if(cell->face(face_number)->boundary_id() == 0)
        {}else if (cell->face(face_number)->boundary_id() == 1)
        {
          obj_coeff_diff.value(fe_face_values.get_quadrature_points(), coeff_diff_values_face);

          for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
          {
              
            if(this->id_case==1)
            {
              neumann_boundary[0] = 0.0 + 0.0i;
            }else
            {
              neumann_boundary = coeff_diff_values_face[q_point]*exact_solution_both.gradient_both (fe_face_values.quadrature_point(q_point));
            }
            
            
            neumann_boundary_times_normal_vector = neumann_boundary * fe_face_values.normal_vector(q_point);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              if (fe.system_to_component_index(i).first == 0)
              {
                cell_rhs(i) += fe_face_values.shape_value(i,q_point)*neumann_boundary_times_normal_vector.real()*fe_face_values.JxW(q_point);
              }else if (fe.system_to_component_index(i).first == 1)
              {
                cell_rhs(i) += fe_face_values.shape_value(i,q_point)*neumann_boundary_times_normal_vector.imag()*fe_face_values.JxW(q_point);
              }
            }
          }
        }
      }
    }
    
#endif

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

  if (this->is_lhs_and_rhs_before_diri_BC_printed==1)
  {
    this->printing_lhs_and_rhs_before_applying_Diri_BC();
  } 
  
  map<types::global_dof_index,double> boundary_values_for_pressure;                      // note the location of this function cluster
  VectorTools::interpolate_boundary_values (this->dof_handler,
                                            0,
                                            PressureBoundaryValues_Complex<dim>(this->id_case,
                                                                                this->coeff_var_inner_x),
                                            boundary_values_for_pressure);

  MatrixTools::apply_boundary_values (boundary_values_for_pressure,
                                      this->system_matrix,
                                      this->solution,
                                      this->system_rhs);  
  
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
void Step_4_Complex<dim>::splitting_real_and_imag_part_of_the_solution ()
{

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   dofs_per_cell_single = fe_single.dofs_per_cell;
  vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  vector<types::global_dof_index> local_dof_indices_single (dofs_per_cell_single);

  typename DoFHandler<dim>::active_cell_iterator
  cell = this->dof_handler.begin_active(),
  endc = this->dof_handler.end();
  
  typename DoFHandler<dim>::active_cell_iterator
  cell_single = dof_handler_single.begin_active();

  unsigned int id_real = 0;
  unsigned int id_imag = 0;
  
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    
    cell_single->get_dof_indices (local_dof_indices_single);
    cell_single++;
    
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      
      if(fe.system_to_component_index(i).first==0)
      {
        solution_real[local_dof_indices_single[id_real]]=this->solution[local_dof_indices[i]];
        id_real++;
      }else if (fe.system_to_component_index(i).first==1)
      {
        solution_imag[local_dof_indices_single[id_imag]]=this->solution[local_dof_indices[i]];
        id_imag++;
      }
    }
    id_real=0;
    id_imag=0;
  }
  
  for (unsigned int i = 0; i < this->dof_handler.n_dofs()/2.0; i++)
  {
    solution_amp[i] = sqrt(this->solution_real[i]*solution_real[i] + solution_imag[i]*solution_imag[i]);
  }    
}

  
template <int dim>
void Step_4_Complex<dim>::computing_norms_of_solution_numerically_for_complex_valued_problems ()
{
//   cout << "Computing l2 norms using the numerical solution\n";

  Vector<double> difference_per_cell (this->triangulation.n_active_cells());
  Vector<double> difference_per_cell_real (this->triangulation.n_active_cells());                 // L2 norm of the real part of the solution in each cell
  Vector<double> difference_per_cell_imag (this->triangulation.n_active_cells());                 // L2 norm of the imag part of the solution in each cell

  vector<Vector<double>> difference_per_cell_both (this->triangulation.n_active_cells(),Vector<double>(2));
  
  VectorTools::integrate_difference (dof_handler_single,                                       // real part of the solution
                                     solution_real,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_real,                              
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::L2_norm);
  solution_real_L2_norm_numerical = difference_per_cell_real.l2_norm();
  
  
  VectorTools::integrate_difference (dof_handler_single,                                       // imaginary part of the solution
                                     solution_imag,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_imag,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::L2_norm);
  solution_imag_L2_norm_numerical = difference_per_cell_imag.l2_norm(); 
  
   
  this->solution_L2_norm_numerical = pow(pow(solution_real_L2_norm_numerical,2)+pow(solution_imag_L2_norm_numerical,2),0.5);        // which is the most convenient way
  
  VectorTools::integrate_difference (dof_handler_single,                                       // gradient     
                                     solution_real,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_real,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_imag,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  
  for(unsigned int i = 0; i<this->triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  this->solution_H1_semi_norm_numerical = difference_per_cell.l2_norm();


  VectorTools::integrate_difference (dof_handler_single,                                       // 2nd derivative                                   
                                     solution_real,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_real,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_imag,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  
  for(unsigned int i = 0; i<this->triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  this->solution_H2_semi_norm_numerical = difference_per_cell.l2_norm();
  
  if(is_l2_norm_of_the_numerical_real_and_imaginary_parts_printed==1)
  {
    cout << "  @norms of the real and imaginary parts\n";
    cout << "    solution_real_L2_norm_numerical: " << solution_real_L2_norm_numerical << ", ";
    cout << "    solution_imag_L2_norm_numerical: " << solution_imag_L2_norm_numerical << "\n";
  }
  
}
    
    
template <int dim>
void Step_4_Complex<dim>::stage_6a_computing_errors_built_in_for_complex_valued_problems ()
{
  TimerOutput::Scope t(this->computing_timer, "stage_6a_computing_errors_built_in");
//   cout << "  Invoking core\n";
    
  ExactSolution_Step_4_Complex_Real<dim> exact_solution_real(this->id_case,
                                                             this->coeff_var_inner_x);
  ExactSolution_Step_4_Complex_Imag<dim> exact_solution_imag(this->id_case,
                                                             this->coeff_var_inner_x); 

  Vector<double> difference_per_cell (this->triangulation.n_active_cells());
  Vector<double> difference_per_cell_real (this->triangulation.n_active_cells());                 // L2 norm of the real part of the solution in each cell
  Vector<double> difference_per_cell_imag (this->triangulation.n_active_cells());                 // L2 norm of the imag part of the solution in each cell
  
  VectorTools::integrate_difference (dof_handler_single,                                      
                                     solution_real,
                                     exact_solution_real,
                                     difference_per_cell_real,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::L2_norm);
  
  VectorTools::integrate_difference (dof_handler_single,                                      
                                     solution_imag,
                                     exact_solution_imag,
                                     difference_per_cell_imag,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::L2_norm);
  
  for(unsigned int i = 0; i<this->triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
   
  this->solution_L2_error_abs_built_in = difference_per_cell.l2_norm();
  this->solution_L2_error_rela_to_itself = this->solution_L2_error_abs_built_in/this->solution_L2_norm_numerical;
  
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_real,
                                     exact_solution_real,
                                     difference_per_cell_real,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     exact_solution_imag,
                                     difference_per_cell_imag,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H1_seminorm);
  
  for(unsigned int i = 0; i<this->triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  this->solution_H1_semi_error_abs_built_in = difference_per_cell.l2_norm();
  this->solution_H1_semi_error_rela_to_itself = this->solution_H1_semi_error_abs_built_in/this->solution_H1_semi_norm_numerical;
  
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_real,
                                     exact_solution_real,
                                     difference_per_cell_real,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     exact_solution_imag,
                                     difference_per_cell_imag,
                                     this->qgauss_for_integrating_difference,
                                     VectorTools::H2_seminorm);
  
  for(unsigned int i = 0; i<this->triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  this->solution_H2_semi_error_abs_built_in = difference_per_cell.l2_norm();
  this->solution_H2_semi_error_rela_to_itself = this->solution_H2_semi_error_abs_built_in/this->solution_H2_semi_norm_numerical;
  
  this->n_times_called_computing_error_built_in++;
  
}


template <int dim>
void Step_4_Complex<dim>::from_setup_system_to_solve_for_the_complex_valued_problem()
{
  stage_2a_setup_system_distributing_dofs_for_the_complex_valued_problem();
  this->stage_2b_setup_system_the_rest ();
  
#if 0
  stage_3_assemble_system_for_the_complex_valued_problem ();
  this->stage_4_solve ();    
#endif
  
}


template <int dim>
void Step_4_Complex<dim>::conducting_a_run_for_computing_the_error_customly_for_the_complex_valued_problem ()
{
//   cout << "Computing errors customly\n";
  
  this->is_refining_mesh_for_an_independent_run = 0;
  this->refining_mesh_globally(1);
  this->printing_current_grid_status();
  this->current_refinement_level_for_showing--;                                 // back to the refinement level of the active mesh
  
  from_setup_system_to_solve_for_the_complex_valued_problem();
    
  this->stage_6b_computing_the_error_customly();
  
}


template <int dim>
void Step_4_Complex<dim>::run ()
{
  this->id_type_of_number = 1;
  this->n_number_components = 2;
  
  if(this->id_case == 1)
  {
      this->is_built_in_method_used_for_error = 0;
  }
  
  this->preparing_for_the_initial_grid_and_other_settings();
  
  for (unsigned int cycle = 0; cycle < this->n_total_refinements; cycle++)
  {
    this->cycle_global = cycle;
    
    from_setup_system_to_solve_for_the_complex_valued_problem();

    this->n_vertices_first_refine = this->triangulation.n_vertices();
    this->n_dofs_first_refine = this->dof_handler.n_dofs();     

    this->stage_5_dealing_with_the_result_of_the_first_run_that_is_independent_of_the_type_of_number();

    splitting_real_and_imag_part_of_the_solution();
    
    if(this->are_norms_of_solution_computed_numerically == 1)
    {
      computing_norms_of_solution_numerically_for_complex_valued_problems();
    }
    
    if (this->is_built_in_method_used_for_error == 1)
    {
        stage_6a_computing_errors_built_in_for_complex_valued_problems ();
    }
    
    this->copying_triangulation_and_solution_of_the_last_refinement_for_further_refinements();
    this->dof_handler_first_refine.distribute_dofs (fe);
      
    if(this->is_custom_method_used_for_error == 1)
    {
      conducting_a_run_for_computing_the_error_customly_for_the_complex_valued_problem ();  
    }
        
    this->dealing_with_the_error_and_cpu_time_etc_after_one_complete_run();    
    
    if(cycle < this->n_total_refinements - 1)
    {
      this->preparing_for_the_next_run();
    }
  }
  
  if(this->is_results_of_all_refinements_printed == 1)
  {
      this->printing_results_of_all_refinements();
  }  
     
}

#endif
