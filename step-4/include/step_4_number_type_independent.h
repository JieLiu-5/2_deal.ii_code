#ifndef STEP_4_NUMBER_TYPE_INDEPENDENT_COMPLEX_H
#define STEP_4_NUMBER_TYPE_INDEPENDENT_COMPLEX_H

#include <deal.II/grid/grid_refinement.h>                               // these two for the local mesh refinement
#include <deal.II/numerics/error_estimator.h>


#include <step_4_external.h>



template <int dim>
class Step_4_Number_Type_Independent
{
public:
  Step_4_Number_Type_Independent(const int id_quad_assem_incre,
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
                                const unsigned int n_total_refinements);
    
  const int   id_quad_assem_incre;
  const double tol_prm = 1e-16;
  
  const unsigned int   is_custom_method_used_for_error;             // the error for the post-processing is the customly computed error if we use the custom method for computing the error
                                                                    // "is_custom_method_used_for_error" is used for this option
  const unsigned int   id_case;
  const double coeff_var_inner_x;
  const int   id_coeff_diff_value;
  const double coeff_diff_inner;
  const int id_coeff_helm_value;
  const double coeff_helm_inner;    
  
  const unsigned int   degree;
  
  const unsigned int id_mesh_being_created;                                   // '0' for the built-in mesh
                                                                            // '1' for the custom mesh  
  const unsigned int grid_param_initial;
  
  unsigned int   initial_refinement_level;
    
  unsigned int n_dofs_custom;
  unsigned int n_dofs_custom_of_one_component;
  unsigned int n_dofs_custom_in_one_direction;
  unsigned int n_vertices_custom;                          //    for custom meshes, not used if using the global mesh
  unsigned int n_vertices_custom_in_one_direction;
  unsigned int n_cells_custom_in_one_direction;
    
  const unsigned int n_total_refinements;
   
  
  vector<string> vec_type_of_number = {"0_real", "1_complex"};
  unsigned int id_type_of_number = 0;                                   // '0' for real-valued problems
                                                                        // '1' for complex-valued problems
  unsigned int n_number_components;                                 // 
  
  unsigned int cycle_global;
  
  
  const unsigned int n_q_points_in_one_direction = degree+id_quad_assem_incre;
  QGauss<dim> quadrature_formula = QGauss<dim>(n_q_points_in_one_direction);
  const unsigned int n_q_points = quadrature_formula.size();  
  
  
  Triangulation<dim>   triangulation;
  Triangulation<dim>   triangulation_first_refine;
  
//   Triangulation<dim>   triangulation_uniform_refine;                        // used only when the distorted mesh is involved; the commands on it is deactivated by now
  
  unsigned int is_containing_neumann = 1;
  
  unsigned int current_refinement_level_for_showing = 0;
    
  unsigned int id_method_mesh_distortion = 1;               // '0' for randomly distorted mesh
                                                            // '1' for regularly distorted mesh
  
  double factor_distorting = 0.4;                               // used for randomly distorted mesh
  unsigned int id_type_of_regular_grid = 1;                             // cases for regularly distorted mesh
                                                                                                   // '1' for the method on line 410, which is a linear case
                                                                                                   // '2' for the method on line 420, which is a sine case
  
  
  string obj_string="";
  unsigned int n_dofs_bspline = 1;
  
  DoFHandler<dim>      dof_handler;
  DoFHandler<dim>      dof_handler_first_refine;                    // we define it here because it is used both in refining_mesh_globally() and compute_errors_custom()
  
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       system_rhs;
  
  double l2_norm_coeff_diff;
  double l2_norm_coeff_helm;
  double l2_norm_coeff_first_derivative_a;
  
  Vector<double>       solution;
  Vector<double>       solution_first_refine;
  Vector<double>       solution_zero;
  

  QGauss<dim> qgauss_for_integrating_difference = QGauss<dim>(degree+1);
  
  double solution_L2_error_abs_built_in = 1.0;
  double solution_H1_semi_error_abs_built_in = 1.0;
  double solution_H2_semi_error_abs_built_in = 1.0;
  
  double solution_L2_norm_numerical;
  double solution_H1_semi_norm_numerical;
  double solution_H2_semi_norm_numerical;
  
  double solution_L2_norm_analytical;
  double solution_H1_semi_norm_analytical;
  double solution_Hdiv_semi_norm_analytical;
  double solution_H2_semi_norm_analytical;
  
  double solution_L2_error_rela_to_itself;
  double solution_H1_semi_error_rela_to_itself;
  double solution_H2_semi_error_rela_to_itself;    
  
  double solution_H2_semi_error_abs_00, solution_H2_semi_error_abs_01;
  double solution_H2_semi_error_abs_10, solution_H2_semi_error_abs_11;
    
  double solution_Hdiv_semi_error_abs;

  
  unsigned int n_vertices_first_refine = 1;
  unsigned int n_dofs_first_refine = 1;
  
  double solution_L2_error_abs_custom = 1.0;
  double solution_H1_semi_error_abs_custom = 1.0;
  double solution_H2_semi_error_abs_custom = 1.0;
  
  double total_CPU_time_per_run = 0.0;
  
  
  vector<double> vec_l2_norm_coeff_diff;
  vector<double> vec_l2_norm_coeff_helm;
  vector<double> vec_l2_norm_coeff_first_derivative_a;
  
  vector<double> vec_l2_norm_coeff_diff_distorted;
  vector<double> vec_l2_norm_coeff_helm_distorted;
  vector<double> vec_l2_norm_coeff_first_derivative_a_distorted;  

  vector<double> vec_solution_L2_norm;

  vector<vector<double>> vec_error;
  vector<vector<double>> vec_error_distorted;
  
  vector<vector<float>> vec_convergence_rate;
  vector<vector<float>> vec_convergence_rate_distorted;
  
  vector<float> convergence_rate_sum;
  vector<float> convergence_rate_average;

  vector<double> vec_cpu_time;
  vector<double> vec_cpu_time_distorted;
  
  unsigned int n_times_called_setup = 0;
  unsigned int n_times_called_assemble = 0;
  unsigned int n_times_called_solve = 0;
  unsigned int n_times_called_computing_error_built_in = 0;
  unsigned int n_times_called_computing_error_customly = 0;
  
  TimerOutput          computing_timer;
    
  


  void preparing_for_the_initial_grid_and_other_settings();
  void stage_1_making_grid_and_adjusting_boundary_conditions ();                                                                        // various functions
  void refining_mesh_globally (unsigned int refinements_to_be_done);
  void printing_current_grid_status();
  
  void refining_mesh_locally ();
  void distorting_the_mesh();
    
  void stage_2b_setup_system_the_rest ();
  void post_processing_info_of_dof_handler ();
  
  void printing_lhs_and_rhs_before_applying_Diri_BC();
  
  void printing_lhs_and_rhs_after_applying_Diri_BC();
  
  void saving_lhs_and_rhs_after_applying_Drii_BC();
  
  void saving_norms_of_various_coefficients_to_vectors();
  
  void stage_4_solve ();
  
  
  void stage_5_dealing_with_the_result_of_the_first_run_that_is_independent_of_the_type_of_number();
  void output_results();
  
    
  void copying_triangulation_and_solution_of_the_last_refinement_for_further_refinements();
  void stage_6b_computing_the_error_customly();
  
  void dealing_with_the_error_and_cpu_time_etc_after_one_complete_run();
  void printing_errors_order_of_convergence_etc_and_storing_them_to_vectors();
  void storing_error_cpu_time_to_file();
  
  void preparing_for_the_next_run();
  
  void printing_results_of_all_refinements ();
  
                                                                    // parameters controlling the execution of variouis functions
  unsigned int is_tria_info_printed = 0;  
  
  unsigned int is_boundary_info_printed = 0;
  
  unsigned int is_dofhandler_info_printed = 0;
  
  unsigned int is_dof_handler_info_postprocessed = 0;  
  unsigned int id_basis = 0;
  
  unsigned int is_local_dof_indices_printed=0;
  unsigned int is_local_dof_indices_saved=0;  
  
  unsigned int is_coords_of_dofs_of_uniform_printed = 0;
  unsigned int is_coords_of_dofs_of_distorted_printed = 0;
  
  unsigned int is_coords_of_dofs_of_uniform_saved = 0;
  unsigned int is_coords_of_dofs_of_distorted_saved = 0;
  
  unsigned int is_lhs_and_rhs_before_diri_BC_printed = 0;
  unsigned int is_lhs_and_rhs_after_diri_BC_printed = 0;
  unsigned int is_lhs_and_rhs_after_diri_BC_saved = 0;
  
  unsigned int is_lhs_after_BC_stored = 0;
  unsigned int is_rhs_after_BC_stored = 0;  
    
  
  unsigned int is_mesh_uniform_computed = 1;
  
  unsigned int is_before_mesh_being_distorted = 1;                            // becoming 0 when is_mesh_distorted_computed == 1
                                                                              // used to distinguish the uniform mesh and the distorted mesh in the ouput
  
  unsigned int is_l2_norm_coefficient_calculated_numerically = 1;
    
  unsigned int is_UMFPACK = 1;
  unsigned int CG_iteration_times = 1;
  unsigned int is_solver_info_stored = 0;
  
  unsigned int is_results_outputted = 0;
  unsigned int is_solution_printed = 0;
  
  
  unsigned int is_built_in_method_used_for_error = 1;

                                                                              
  unsigned int are_norms_of_solution_computed_numerically = 1;
  
    
  unsigned int is_the_error_printed_built_in = 0;
  unsigned int is_the_l2_norm_printed_built_in = 0;
  
  unsigned int is_the_error_printed_custom = 0;
  
  unsigned int is_cpu_time_printed = 0;
  
  unsigned int is_refining_mesh_for_an_independent_run = 0;
  
  unsigned int is_mesh_distorted_computed = 0;
    
  unsigned int is_results_of_all_refinements_printed = 1;
    
private:
        

};


template <int dim>
Step_4_Number_Type_Independent<dim>::Step_4_Number_Type_Independent(const int id_quad_assem_incre,
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
id_quad_assem_incre(id_quad_assem_incre),
tol_prm(tol_prm),
is_custom_method_used_for_error(is_custom_method_used_for_error),
id_case(id_case),
coeff_var_inner_x(coeff_var_inner_x),
id_coeff_diff_value(id_coeff_diff_value),
coeff_diff_inner(coeff_diff_inner),
id_coeff_helm_value(id_coeff_helm_value),
coeff_helm_inner(coeff_helm_inner),
degree(degree),
id_mesh_being_created(id_mesh_being_created),
grid_param_initial(grid_param_initial),
n_total_refinements(n_total_refinements),
dof_handler (triangulation),
dof_handler_first_refine (triangulation_first_refine),
computing_timer  (cout, TimerOutput::never, TimerOutput::cpu_times)                 // summary, never
{}


template <int dim>
void Step_4_Number_Type_Independent<dim>::stage_1_making_grid_and_adjusting_boundary_conditions ()
{
//   cout << "Making grid\n";
  TimerOutput::Scope t(computing_timer, "stage_1_making_grid_and_adjusting_boundary_conditions");
    
  if(id_mesh_being_created == 0)
  {
//     cout << "  mesh built_in\n";
      
    initial_refinement_level = grid_param_initial;
    
    GridGenerator::hyper_cube (triangulation, 0, 1);
//     GridGenerator::hyper_cube (triangulation_uniform_refine, 0, 1);
    
    refining_mesh_globally(initial_refinement_level);
    
  }else if(id_mesh_being_created == 1)
  {
    if (dim == 1)
    {
        
      n_dofs_custom = grid_param_initial;
      
      cout << "    n_dofs_custom: " << n_dofs_custom << endl;
      
      n_dofs_custom_of_one_component = n_dofs_custom/(id_type_of_number+1);
      
      cout << "    n_dofs_custom_of_one_component: " << n_dofs_custom_of_one_component << endl;
      
      n_vertices_custom = (n_dofs_custom_of_one_component-1)/degree + 1;                  // we need to transform n_dofs_custom to n_vertices_custom when only the former is received
                                                                                          // c++ rounds down numbers, c.f. http://www.cplusplus.com/forum/beginner/60827/
      initial_refinement_level = 0;
        
      cout << "    n_vertices_custom: " << n_vertices_custom << endl;      
        
      vector<Point<dim>> vertices(n_vertices_custom);

      const double delta_vertex = 1.0/(n_vertices_custom-1);
      for (unsigned int i = 0; i < n_vertices_custom; ++i)
      {
        vertices[i](0) = i * delta_vertex;
      } 

      vector<array<unsigned int,GeometryInfo<dim>::vertices_per_cell>> cell_vertices;
      array<unsigned int,GeometryInfo<dim>::vertices_per_cell> array_update;
      for (unsigned int i = 0; i < n_vertices_custom-1; ++i)
      {
        array_update = {i,i+1};
        cell_vertices.push_back(array_update);
      }

      vector<CellData<dim>> cells(cell_vertices.size(), CellData<dim>());
      for (unsigned int i=0; i<cell_vertices.size(); ++i)
      {
        for (unsigned int j=0; j<GeometryInfo<dim>::vertices_per_cell; ++j)
        {
          cells[i].vertices[j] = cell_vertices[i][j];
        }
      }
      triangulation.create_triangulation (vertices, cells, SubCellData());  
        
    }else if(dim == 2)
    {
#if 0
      
      initial_refinement_level = 0;
      
      n_dofs_custom = grid_param_initial;
      
      n_dofs_custom_in_one_direction = sqrt(n_dofs_custom/(id_type_of_number+1));
      
      n_vertices_custom_in_one_direction = (n_dofs_custom_in_one_direction - 1)/degree + 1;
      n_cells_custom_in_one_direction = n_vertices_custom_in_one_direction - 1;
      
//       cout << "n_dofs_custom_in_one_direction: " << n_dofs_custom_in_one_direction << "\n";
//       cout << "n_vertices_custom_in_one_direction: " << n_vertices_custom_in_one_direction << "\n";
//       cout << "n_cells_custom_in_one_direction: " << n_cells_custom_in_one_direction << "\n";
      
      std::vector<unsigned int> repetitions(2);
      repetitions[0] = n_cells_custom_in_one_direction;
      repetitions[1] = n_cells_custom_in_one_direction;
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                Point<2>(0.0, 0.0),
                                                Point<2>(1.0, 1.0));  
#endif
      
    }
  }else
  {
    cout << "  unavailable mesh scheme\n";
    exit(1);
  }
  
#if 1
  if(is_tria_info_printed == 1)
  {
    print_tria_info(triangulation);
  }
  
  if (is_containing_neumann==0)
  {
    if (dim==1)
    {
//       triangulation.begin_active()->face(1)->set_boundary_id(0);
      triangulation.last_active()->face(1)->set_boundary_id(0);                                 // suitable for the custom mesh
//       triangulation_uniform_refine.begin_active()->face(1)->set_boundary_id(0);
    }
  }else if (is_containing_neumann==1)
  {
    if(dim==2)
    {
      adjust_boundary_id_2d(triangulation);
        
//       triangulation.begin_active()->face(0)->set_boundary_id(1);       // left
//       triangulation.begin_active()->face(1)->set_boundary_id(1);       // right
      
//       triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom                // not successful in changing the boundary condition after the refinement
//       triangulation.begin_active()->face(3)->set_boundary_id(1);       // top                   // so we use 'adjust_boundary_id_2d()' uniformly
      
//      triangulation_uniform_refine.begin_active()->face(2)->set_boundary_id(1);       // bottom
//      triangulation_uniform_refine.begin_active()->face(3)->set_boundary_id(1);       // top
      
    }
  }
  
#endif

  if(is_boundary_info_printed == 1)
  {
    print_boundary_info(triangulation);   
  }

  printing_current_grid_status();                                        // for both the built-in and custom meshes
  
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::printing_current_grid_status()                 
{
  if (id_mesh_being_created == 0)
  {
    cout << "    (deal.ii) degree: " << degree << ", refinement level " << current_refinement_level_for_showing;
  }else
  {
    cout << "    (deal.ii) degree: " << degree << ", n_vertices_custom: " << triangulation.n_vertices();
  }
  
  if(current_refinement_level_for_showing == initial_refinement_level)                      // we increase 'current_refinement_level_for_showing' when refining the mesh globally
  {
    cout << " (initial)";
  }else
  {
    if(is_refining_mesh_for_an_independent_run == 1)
    {     
      cout << " (finer grid for an independent computation)";
    }else
    {
      cout << " (finer grid for computing the error customly)";
    }
  }
  cout << "\n";
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::refining_mesh_globally(unsigned int refinements_to_be_done)                 
{
//   cout << "Step_4_Number_Type_Independent<dim>::refining_mesh_globally()\n";
  
  triangulation.refine_global (refinements_to_be_done);    
//   triangulation_uniform_refine.refine_global(refinements_to_be_done);
  
  current_refinement_level_for_showing += refinements_to_be_done;
  
  if(is_tria_info_printed == 1)
  {
    print_tria_info(triangulation);
  }  
  
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::refining_mesh_locally ()                 
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
  
  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      {},
                                      solution,
                                      estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.4, 0.00);
  triangulation.execute_coarsening_and_refinement ();
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::distorting_the_mesh()                 
{
  cout << "\n";
  
  if(id_method_mesh_distortion == 0)
  {
      
    cout << "Distorting the mesh randomly with the factor " << factor_distorting << "\n";
    
    GridTools::distort_random(factor_distorting, triangulation);
    
  }else if(id_method_mesh_distortion == 1)
  {
    cout << "Distorting the mesh using a function\n";
    
#if 0
      GridTools::transform(GridFunc_1D(id_type_of_regular_grid), triangulation);
#else
//     distorting_the_2d_mesh_moving_top_vertices_upwards(triangulation);
    
//     distorting_the_2d_mesh_like_a_sine_function(triangulation);
    
#endif
    
  }
  
  if(is_tria_info_printed==1)
  {
    print_tria_info(triangulation);
  }
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::preparing_for_the_initial_grid_and_other_settings ()
{
  is_before_mesh_being_distorted = 1;
  
  vec_error.resize(n_total_refinements,vector<double>(3,0));
  vec_error_distorted.resize(n_total_refinements,vector<double>(3,0));
  
  vec_convergence_rate.resize(n_total_refinements-1,vector<float>(3,0));
  vec_convergence_rate_distorted.resize(n_total_refinements-1,vector<float>(3,0));
  convergence_rate_sum.resize(3);
  convergence_rate_average.resize(3);
  
  stage_1_making_grid_and_adjusting_boundary_conditions();
  
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::post_processing_info_of_dof_handler ()
{
  
  cout << "Post-processing info of dof_handler\n";
    
  obj_string = "dof_handler";
  if (id_basis==0)
  {
    if(is_local_dof_indices_printed==1)
    {
      print_local_dof_indices(obj_string,dof_handler);
      print_local_face_dof_indices(obj_string,dof_handler);  
    }
    
    if(is_local_dof_indices_saved==1)
    {
      save_local_dof_indices(obj_string,dof_handler);  
    }
    
  }else if(id_basis==1)
  {
    n_dofs_bspline = degree+pow(2,initial_refinement_level);
    cout << n_dofs_bspline;
  }
  
  string obj_string_path = "../../3_writing/2_overleaf/1_article_2d/2_figure/5_matlab_process/0_data/";
  
  if(is_before_mesh_being_distorted == 1)
  {
    if(is_coords_of_dofs_of_uniform_printed==1)
    {
      print_coords_of_dofs(obj_string,
                           dof_handler);
    }
  
    if(is_coords_of_dofs_of_uniform_saved==1)
    {
      obj_string = obj_string_path + "coords_of_uniform_dofs_of_degree_" + to_string(degree) + "_refine_"+to_string(current_refinement_level_for_showing);
      // ../step-super/matlab_process/0_data/
      save_coords_of_dofs(obj_string,
                          dof_handler,
                          current_refinement_level_for_showing);
      
      sequence_numbers_in_a_txt_file(obj_string);
      
    }
  }else if(is_before_mesh_being_distorted == 0)
  {
    if(is_coords_of_dofs_of_distorted_printed==1)
    {
      obj_string = "dof_handler_distorted";
      print_coords_of_dofs(obj_string, dof_handler);
    }
    
    if(is_coords_of_dofs_of_distorted_saved==1)
    {
        
      if(id_method_mesh_distortion==0)
      {
        obj_string = obj_string_path + "coords_of_randomly_distorted_dofs_of_degree_" + to_string(degree) + "_refine_"+to_string(current_refinement_level_for_showing);
      }else if(id_method_mesh_distortion==1)
      {
        obj_string = obj_string_path + "coords_of_regularly_" + to_string(id_type_of_regular_grid) + "_distorted_dofs_of_degree_" + to_string(degree) + "_refine_"+to_string(current_refinement_level_for_showing);
      }
      
      save_coords_of_dofs(obj_string,
                          dof_handler,
                          current_refinement_level_for_showing);  
      
      sequence_numbers_in_a_txt_file(obj_string);
      
    }
  }
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::stage_2b_setup_system_the_rest ()
{
  TimerOutput::Scope t(computing_timer, "stage_2b_setup_system_the_rest");
  
//   cout << "Setting up\n";

  obj_string = "dof_handler";
  if(is_dofhandler_info_printed == 1)             //  && (n_times_called_setup==0)           or id_mesh_being_created == 1
  {
    cout << "  Info of dof_handler:\n";
      
    cout << "    # of dofs: " << dof_handler.n_dofs() << "\n";
//     print_local_dof_indices(obj_string, dof_handler);  
//     print_coords_of_dofs(obj_string, dof_handler);  
  }
//   save_coords_of_dofs(obj_string, dof_handler);  
      
  
  if (is_dof_handler_info_postprocessed==1)
  {
      post_processing_info_of_dof_handler();
  }  
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);               // dispensing non-zero entries

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  solution_zero.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
  
  n_times_called_setup++;
  
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::printing_lhs_and_rhs_before_applying_Diri_BC ()
{
    cout << '\n';
    cout << "system_matrix before applying Dirichlet BC:\n";
    system_matrix.print_formatted(cout);
    cout << "system_rhs after applying Neumann BC but not Dirichlet BC:\n";
    system_rhs.print(cout);    
}
    
    
template <int dim>
void Step_4_Number_Type_Independent<dim>::printing_lhs_and_rhs_after_applying_Diri_BC ()
{
    cout << '\n';
    cout << "system_matrix after applying Dirichlet BC:\n";
    system_matrix.print_formatted(cout);
    cout << "system_rhs after applying both Neumann and Dirichlet BC:\n";
    system_rhs.print(cout);
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::saving_lhs_and_rhs_after_applying_Drii_BC()
{
  ostringstream streamObj;
  streamObj << fixed;
  streamObj << setprecision(0);
  streamObj << coeff_var_inner_x;
  
  if (is_lhs_after_BC_stored==1)
  {
    obj_string="system_matrix_coeff_"+ streamObj.str() +"_deg_"+to_string(degree)+"_ref_"+to_string(initial_refinement_level);
    save_system_matrix_to_txt(obj_string,system_matrix);
  }
  
  if(is_rhs_after_BC_stored==1)
  {
    obj_string="system_rhs_coeff_"+ streamObj.str() +"_deg_"+to_string(degree)+"_ref_"+to_string(initial_refinement_level);
    save_Vector_to_txt(obj_string,system_rhs);
  }    
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::saving_norms_of_various_coefficients_to_vectors ()
{
    if(is_before_mesh_being_distorted == 1)
    {
      vec_l2_norm_coeff_diff.push_back(l2_norm_coeff_diff);
      vec_l2_norm_coeff_helm.push_back(l2_norm_coeff_helm);
      vec_l2_norm_coeff_first_derivative_a.push_back(l2_norm_coeff_first_derivative_a);
    }else if(is_before_mesh_being_distorted == 0)
    {
      vec_l2_norm_coeff_diff_distorted.push_back(l2_norm_coeff_diff);
      vec_l2_norm_coeff_helm_distorted.push_back(l2_norm_coeff_helm);
      vec_l2_norm_coeff_first_derivative_a_distorted.push_back(l2_norm_coeff_first_derivative_a);        
    }    
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::copying_triangulation_and_solution_of_the_last_refinement_for_further_refinements ()
{
//   cout << "  dealing with info of the last refinement\n";
  
  solution_first_refine.reinit (dof_handler.n_dofs());
  solution_first_refine=solution;
  
//     cout << "  solution_first_refine: ";
//     solution_first_refine.print(cout,5); 

  triangulation_first_refine.clear();
  triangulation_first_refine.copy_triangulation(triangulation);
  
}

    
template <int dim>
void Step_4_Number_Type_Independent<dim>::stage_4_solve ()
{
  TimerOutput::Scope t(computing_timer, "stage_4_solve");
//   cout << "Solving\n";

  if (tol_prm != 0)
  {
    is_UMFPACK = 0;
  }
    
  if (is_UMFPACK==1)
  {
//     cout << "  UMFPack solver\n";
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
  }else if(is_UMFPACK==0)   
  {
//     cout << "  CG solver, tol_prm: " << tol_prm << endl;
    SolverControl           solver_control (1e+8, tol_prm);
    SolverCG<>              solver (solver_control);
    solver.solve (system_matrix, solution, system_rhs,
                    PreconditionIdentity());
    CG_iteration_times = solver_control.last_step();
    cout << "  " << solver_control.last_step() << " CG iterations needed to obtain convergence.\n";
    
    cout << "   " << solver_control.last_value()
         << "  is the convergence value of last iteration step.\n";    
  }
  
  n_times_called_solve++;
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::stage_5_dealing_with_the_result_of_the_first_run_that_is_independent_of_the_type_of_number ()
{
  TimerOutput::Scope t(computing_timer, "stage_5_dealing_with_the_result_of_the_first_run");
    
  if (is_results_outputted == 1)
  {
    output_results ();
  }
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::output_results ()
{
//   cout << "Storing results\n";
  
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  
  data_out.build_patches ();
  
  obj_string = "solution_" + to_string(dim) + "d_0_sm_" + vec_type_of_number[id_type_of_number] + "_refine_" + to_string(current_refinement_level_for_showing) + ".vtk";
  
  if (is_before_mesh_being_distorted == 0)
  {
    obj_string = "solution_" + to_string(dim) + "d_0_sm_" + vec_type_of_number[id_type_of_number] + "_distorted.vtk";  
  }
  
  ofstream output (obj_string);
  data_out.write_vtk (output);
}


#if 0
template <int dim>
void Step_4_Number_Type_Independent<dim>::stage_6b_computing_the_error_customly ()
{
//   cout << "error computation custom\n";
  TimerOutput::Scope t(computing_timer, "stage_6b_computing_the_error_customly");
  
  vector<double> vector_error_abs(3, 1.0);
  
  error_computation_custom_core_std(solution_first_refine,
                                    solution,
                                    dof_handler_first_refine,
                                    dof_handler,
                                    quadrature_formula,
                                    n_number_components,
                                    vector_error_abs);
  
  solution_L2_error_abs_custom = vector_error_abs[0];
  solution_H1_semi_error_abs_custom = vector_error_abs[1];
  solution_H2_semi_error_abs_custom = vector_error_abs[2];

  n_times_called_computing_error_customly++;
  
}
#endif


template <int dim>
void Step_4_Number_Type_Independent<dim>::dealing_with_the_error_and_cpu_time_etc_after_one_complete_run ()
{
    
  total_CPU_time_per_run = computing_timer.return_total_cpu_time ();
  if(total_CPU_time_per_run < 0.001)
  {
      total_CPU_time_per_run = 0.001;
  }  
  
  printing_errors_order_of_convergence_etc_and_storing_them_to_vectors();
  storing_error_cpu_time_to_file();
}



template <int dim>
void Step_4_Number_Type_Independent<dim>::printing_errors_order_of_convergence_etc_and_storing_them_to_vectors ()
{
    
  if (is_the_error_printed_built_in == 1)
  {
    cout << "  @errors built-in\n";
    cout << "    solution_L2_error_abs_built_in: " << solution_L2_error_abs_built_in << "\n";
    cout << "    solution_H1_semi_error_abs_built_in: " << solution_H1_semi_error_abs_built_in << "\n";
    cout << "    solution_H2_semi_error_abs_built_in: " << solution_H2_semi_error_abs_built_in << "\n";
    
    if(dim==2 && id_type_of_number==0)
    {
      cout << "      00: " << solution_H2_semi_error_abs_00 << "\n";
      cout << "      01: " << solution_H2_semi_error_abs_01 << "\n";
      cout << "      10: " << solution_H2_semi_error_abs_10 << "\n";
      cout << "      11: " << solution_H2_semi_error_abs_11 << "\n";
      cout << "    solution_Hdiv_semi_error_abs: " << solution_Hdiv_semi_error_abs << "\n";
    }
  }
  
  if(is_the_l2_norm_printed_built_in == 1)
  {
    cout << "  @various norms\n";
    
    /*
    if(id_type_of_number == 0)
    {
      cout << "    1. solution analytically\n";
      cout << "    solution_L2_norm_analytical: " << solution_L2_norm_analytical << "\n";
      cout << "    solution_H1_semi_norm_analytical: " << solution_H1_semi_norm_analytical << "\n";    
      cout << "    solution_H2_semi_norm_analytical: " << solution_H2_semi_norm_analytical << "\n";
      
      if(dim==2)
      {
        cout << "    solution_Hdiv_semi_norm_analytical: " << solution_Hdiv_semi_norm_analytical << "\n";
      }
    }*/
    
    cout << "    2. solution numerically\n";
    cout << "    solution_L2_norm_numerical = " << solution_L2_norm_numerical << "\n";
    cout << "    solution_H1_semi_norm_numerical: " << solution_H1_semi_norm_numerical << "\n";
    cout << "    solution_H2_semi_norm_numerical: " << solution_H2_semi_norm_numerical << "\n";
    
    cout << "    3. coefficients numerically\n";
    cout << "    l2_norm_coeff_diff: " << l2_norm_coeff_diff << "\n";
    cout << "    l2_norm_coeff_helm: " << l2_norm_coeff_helm << "\n";
    cout << "    l2_norm_coeff_first_derivative_a: " << l2_norm_coeff_first_derivative_a << "\n";
  }
  
  if(is_the_error_printed_custom == 1)
  {
    cout << "  @errors custom\n";
    cout << "    solution_L2_error_abs_custom = " << solution_L2_error_abs_custom << "\n";
    cout << "    solution_H1_semi_error_abs_custom = " << solution_H1_semi_error_abs_custom << "\n";
    cout << "    solution_H2_semi_error_abs_custom = " << solution_H2_semi_error_abs_custom << "\n";
  }
  
  if(is_cpu_time_printed == 1)
  {
    cout << "  @cpu time\n";
    cout << "    total_CPU_time_per_run: " << total_CPU_time_per_run << "\n";
  }
  
  vec_solution_L2_norm.push_back(solution_L2_norm_numerical);
  
  if (is_before_mesh_being_distorted == 1)
  {
    if(is_custom_method_used_for_error == 0)
    {
      vec_error[cycle_global][0] = solution_L2_error_abs_built_in;
      vec_error[cycle_global][1] = solution_H1_semi_error_abs_built_in;
      vec_error[cycle_global][2] = solution_H2_semi_error_abs_built_in;
    }else
    {
      vec_error[cycle_global][0] = solution_L2_error_abs_custom;
      vec_error[cycle_global][1] = solution_H1_semi_error_abs_custom;
      vec_error[cycle_global][2] = solution_H2_semi_error_abs_custom;        
    }
      
    vec_cpu_time.push_back(total_CPU_time_per_run);
    
  }else if (is_before_mesh_being_distorted == 0)
  {
    if(is_custom_method_used_for_error == 0)
    {    
      vec_error_distorted[cycle_global][0] = solution_L2_error_abs_built_in;
      vec_error_distorted[cycle_global][1] = solution_H1_semi_error_abs_built_in;
      vec_error_distorted[cycle_global][2] = solution_H2_semi_error_abs_built_in;
    }else
    {
      vec_error_distorted[cycle_global][0] = solution_L2_error_abs_custom;
      vec_error_distorted[cycle_global][1] = solution_H1_semi_error_abs_custom;
      vec_error_distorted[cycle_global][2] = solution_H2_semi_error_abs_custom;
    }
    
    vec_cpu_time_distorted.push_back(total_CPU_time_per_run);
  }
  
  if(cycle_global>0)
  {
    float convergence_rate = 1;
      
    for (unsigned int i=0; i<3; ++i)
    {
      convergence_rate = -log10(vec_error[cycle_global][i]/vec_error[cycle_global-1][i])/log10(2);
//       cout << "convergence_rate " << convergence_rate << "\n";
      vec_convergence_rate[cycle_global-1][i] = convergence_rate;
      
      convergence_rate_sum[i] += convergence_rate;
      
      convergence_rate = -log10(vec_error_distorted[cycle_global][i]/vec_error_distorted[cycle_global-1][i])/log10(2);
      vec_convergence_rate_distorted[cycle_global-1][i] = convergence_rate;
    }
    
    for (unsigned int i = 0; i<3; ++i)
    {
      convergence_rate_average[i] = convergence_rate_sum[i]/(cycle_global);
    }
    
  }
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::storing_error_cpu_time_to_file()
{
//   cout << "Storing errors and cpu time to a file\n";
  ofstream myfile;
  
  if (is_before_mesh_being_distorted == 1)
  {
    obj_string = "data_output_error_0_sm_" + vec_type_of_number[id_type_of_number] + ".txt";                  // _alter
  }else if(is_before_mesh_being_distorted == 0)
  {
    obj_string = "data_output_error_0_sm_" + vec_type_of_number[id_type_of_number] + "_distorted.txt";
  }
  myfile.open (obj_string, ofstream::app);                                                           // trunc  app
  
  myfile << current_refinement_level_for_showing << " ";        
#if 1 
  myfile << n_vertices_first_refine <<" ";
  myfile << n_dofs_first_refine << " ";
  
  if(is_custom_method_used_for_error == 0)
  {
    myfile << solution_L2_error_abs_built_in << " ";
    myfile << solution_H1_semi_error_abs_built_in << " ";
    myfile << solution_H2_semi_error_abs_built_in << " ";      
  }else
  {
    myfile << solution_L2_error_abs_custom << " ";
    myfile << solution_H1_semi_error_abs_custom << " ";
    myfile << solution_H2_semi_error_abs_custom << " ";
  }
  
//   if(dim==2 && id_type_of_number==0)
//   {
//     myfile << solution_H2_semi_error_abs_00 << " ";
//     myfile << solution_H2_semi_error_abs_01 << " ";
//     myfile << solution_H2_semi_error_abs_10 << " ";
//     myfile << solution_H2_semi_error_abs_11 << " ";
//     myfile << solution_Hdiv_semi_error_abs << " ";
//   }
  
  myfile << solution_L2_norm_numerical << " ";
  myfile << l2_norm_coeff_diff << " ";
  myfile << l2_norm_coeff_helm << " ";
  myfile << total_CPU_time_per_run;

  if (is_solver_info_stored == 1)
  {
    if (is_UMFPACK == 1)
    {
      myfile << "  UMF" << "\n";
    }else if (is_UMFPACK == 0)
    {
      myfile << CG_iteration_times << "\n";
    }
  }else
  {
    myfile << "\n";
  }
#endif
  myfile.close();
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::preparing_for_the_next_run ()
{
    is_refining_mesh_for_an_independent_run = 1;
    
    is_before_mesh_being_distorted = 1;
    
    if (is_tria_info_printed == 1)
    {
      print_tria_info(triangulation_first_refine);
    }
    
    triangulation.clear();
    triangulation.copy_triangulation(triangulation_first_refine);                 // we base on the uniform mesh for the next refinement
    
    if(cycle_global < n_total_refinements-1)
    {
      refining_mesh_globally(1);
      printing_current_grid_status();
    }
}


template <int dim>
void Step_4_Number_Type_Independent<dim>::printing_results_of_all_refinements ()
{
  cout << "\n";
  cout << "===========================\n";
  cout << "Results of all refinements\n";
  
  cout << "  initial_refinement_level: " << initial_refinement_level << "\n";
  cout << "  n_total_refinements: " << n_total_refinements << "\n";
  
  if(is_mesh_uniform_computed == 1)
  {
    cout << "  vec_error:\n";
    print_vector_in_vector(vec_error);
    cout << "  vec_convergence_rate:\n";
    print_vector_in_vector(vec_convergence_rate);
    cout << "  convergence_rate_average:\n  ";
    print_vector_horizontally(convergence_rate_average);
    
    
    cout << "  vec_solution_L2_norm:\n";
    print_vector_vertically(vec_solution_L2_norm);    

    
    cout << "  vec_l2_norm_coeff_diff:\n";
    print_vector_vertically(vec_l2_norm_coeff_diff);  
    cout << "  vec_l2_norm_coeff_helm:\n";
    print_vector_vertically(vec_l2_norm_coeff_helm);  
    cout << "  vec_l2_norm_coeff_first_derivative_a:\n";
    print_vector_vertically(vec_l2_norm_coeff_first_derivative_a);

    cout << "  CPU time:\n";
    print_vector_vertically(vec_cpu_time);
  }

  if(is_mesh_distorted_computed==1)
  {
    cout << "\n";
    cout << "  vec_error_distorted:\n";
    print_vector_in_vector(vec_error_distorted);
    cout << "  vec_convergence_rate_distorted:\n";
    print_vector_in_vector(vec_convergence_rate_distorted);

    cout << "  vec_l2_norm_coeff_diff_distorted:\n";
    print_vector_vertically(vec_l2_norm_coeff_diff_distorted);  
    cout << "  vec_l2_norm_coeff_helm_distorted:\n";
    print_vector_vertically(vec_l2_norm_coeff_helm_distorted);
    cout << "  vec_l2_norm_coeff_first_derivative_a_distorted:\n";
    print_vector_vertically(vec_l2_norm_coeff_first_derivative_a_distorted);  

    cout << "  CPU time distorted:\n";
    print_vector_vertically(vec_cpu_time_distorted);
  }

  cout << "  nr. of functions called:\n";
  cout << "    setup: " << n_times_called_setup << "\n";
  cout << "    assemble: " << n_times_called_assemble << "\n";
  cout << "    solve: " << n_times_called_solve << "\n";
  cout << "    computing error built-in: " << n_times_called_computing_error_built_in << "\n";
  cout << "    computing error customly: " << n_times_called_computing_error_customly << "\n";
  
}



#endif
