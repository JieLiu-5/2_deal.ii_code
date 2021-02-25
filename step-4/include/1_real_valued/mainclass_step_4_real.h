
#ifndef MAINCLASS_STEP_4_REAL_H
#define MAINCLASS_STEP_4_REAL_H

#include <deal.II/fe/fe_q_hierarchical.h>

#include <deal.II/grid/grid_refinement.h>       
#include <deal.II/numerics/error_estimator.h>


#include"../../6_headers/1_cpp/1_dealii/base_class_real.h"
#include <1_real_valued/exact_solution_step_4_real.h>

#include"../../6_headers/1_cpp/1_dealii/derived_class_real.h"


template <int dim>
class Step4_Real                     
{
public:
  Step4_Real (const unsigned int id_case,
              const double coeff_var_inner_x,
              const int id_quad_assem_incre, 
              const int id_coeff_diff_value, 
              const double coeff_diff_inner, 
              const int id_coeff_helm_value,
              const double coeff_helm_inner,
              const double tol_prm, 
              const unsigned int degree, 
              const unsigned int initial_refinement_level,
              const unsigned int n_total_refinements);
  
  void run ();
  
  const unsigned int   id_case;
  const double coeff_var_inner_x;
  const int   id_quad_assem_incre;
  const int   id_coeff_diff_value;
  const double coeff_diff_inner;
  const int id_coeff_helm_value;
  const double coeff_helm_inner;
  const double tol_prm = 1e-16;
  const unsigned int   degree;
  const unsigned int   initial_refinement_level; 
  const unsigned int   n_total_refinements; 
  
  string obj_string="";
  
private:
  void make_grid ();
  void setup_system();
  void post_processing_info_of_dof_handler();
  
  void assemble_system ();
  void solve ();
  void compute_integrals();
  void compute_errors_built_in ();
  
  void saving_info_of_the_last_refinement();
  void compute_errors_custom ();
  
  void print_integrals_errors_and_CPU();
  void refining_mesh_globally (unsigned int refinement_level);
  void distorting_the_mesh();
  void refining_mesh_locally ();
  void output_results ();                         
  void storing_error_cpu_time_to_file();
  
  void printPattern(int radius);
  
  void conducting_one_complete_run();
  
  void printing_results_of_all_refinements();

  const int n_vertices_custom;                          
  int cycle_global;
  
  unsigned int current_refinement_level=1;

  unsigned int is_mesh_built_in=1;
  
  Triangulation<dim>   triangulation;
  Triangulation<dim>   triangulation_uniform_refine;
  Triangulation<dim>   triangulation_first_refine;
  
  unsigned int is_containing_neumann = 0;
  
  unsigned int id_basis = 0;
  unsigned int n_dofs_bspline = 1;
  
  FE_Q<dim>            fe;                  // _Q   _Bernstein   _Hierarchical         
                                            // basis functions make a difference to system_matrix
                                            
  DoFHandler<dim>      dof_handler;
  DoFHandler<dim>      dof_handler_first_refine;                    // we define it here because it is used both in refining_mesh_globally() and compute_errors_custom()

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       system_rhs;
  
  Vector<double>       solution;
  Vector<double>       solution_first_refine;
  
  Vector<double>       solution_0;  
  
  AffineConstraints<real_t>      constraints;
  
  unsigned int is_receiving_iga=0;
  
  const unsigned int n_q_points_in_one_direction = degree+id_quad_assem_incre;
  QGauss<dim> quadrature_formula = QGauss<dim>(n_q_points_in_one_direction);
  const unsigned int n_q_points = quadrature_formula.size();
  
  
  unsigned int is_l2_norm_coefficient_numerical = 1;
  
  unsigned int is_strong = 1;
  double prm_penalty_weak = 1e6;                    // 1e6 50
  
  unsigned int is_matrix_after_BC_stored = 0;
  unsigned int is_rhs_after_BC_stored = 0;
  
  unsigned int is_UMFPACK = 1;
  unsigned int CG_iteration_times = 1;
  
  QGauss<dim> qgauss_integrate_difference = QGauss<dim>(degree+1);
  
  
  
  unsigned int is_mesh_uniform_computed = 1;
  unsigned int is_mesh_distorted_computed = 1;
  
  unsigned int is_distorted = 0;
  double factor_distorting = 0.4;
  
  unsigned int is_coords_of_dofs_of_uniform_printed = 1;
  unsigned int is_coords_of_dofs_of_distorted_printed = 0;
  
  unsigned int is_coords_of_dofs_of_uniform_saved = 1;
  unsigned int is_coords_of_dofs_of_distorted_saved = 1;

  
  unsigned int is_results_outputted = 1;
  unsigned int are_integrals_computed = 1;
  unsigned int is_custom_method_used_for_error = 0;
  
  double l2_error_abs, h1_semi_error_abs, h2_semi_error_abs = 1.0;
  double l2_error_delta_u;
  double h2_semi_error_abs_00, h2_semi_error_abs_01;
  double h2_semi_error_abs_10, h2_semi_error_abs_11;
  
  vector<double> vec_l2_norm_coeff_diff;
  vector<double> vec_l2_norm_coeff_helm;
  vector<double> vec_l2_norm_coeff_first_derivative_a;
  
  vector<double> vec_l2_norm_coeff_diff_distorted;
  vector<double> vec_l2_norm_coeff_helm_distorted;
  vector<double> vec_l2_norm_coeff_first_derivative_a_distorted;


  vector<vector<double>> vec_error;
  vector<vector<double>> vec_error_distorted;
  
  vector<double> vec_cpu_time;
  vector<double> vec_cpu_time_distorted;
  
  float convergence_rate=1;
  vector<vector<float>> vec_convergence_rate;
  vector<vector<float>> vec_convergence_rate_distorted;
  
  
  unsigned int is_basis_extracted=0;
  
  double l2_norm_coeff_diff;
  double l2_norm_coeff_helm;
  double l2_norm_coeff_first_derivative_a;
  
  double l2_inte_u_analytical, l2_inte_nabla_u_analytical, l2_inte_delta_u_analytical, l2_inte_grad_grad_u_analytical = 1.0;
//   double l2_error_rela, h1_semi_error_rela, h2_semi_error_rela;
  
  double l2_error_custom, h1_semi_error_custom, h2_semi_error_custom;
  
  double l2_inte_u_numerical;
  double u_mean_inte_numerical, u_mean_inte_analytical;
  
  double f_l2_inte_numerical, f_l2_inte_analytical=0;
  
  unsigned int n_number_components = 1;
  
  TimerOutput          computing_timer;
  double total_CPU_time_per_run;
  
  unsigned int is_L2_norm_stored = 0;
  unsigned int is_solver_info_stored = 1;
  
  unsigned int n_times_called_setup = 0;
  unsigned int n_times_called_assemble = 0;
  unsigned int n_times_called_solve = 0;
  unsigned int n_times_called_computing_error_built_in = 0;
  
  
  unsigned int is_tria_info_printed=1;
  unsigned int is_boundary_info_printed=1;
  unsigned int is_fe_info_printed=0;
  unsigned int is_local_dof_indices_printed=1;

  unsigned int is_lhs_and_rhs_before_diri_BC_printed = 0;
  unsigned int is_lhs_and_rhs_after_diri_BC_printed = 0;
  
  unsigned int is_integrals_and_errors_printed=1;
  
  unsigned int is_solution_printed = 0;
  
};


template <int dim>
Step4_Real<dim>::Step4_Real (const unsigned int id_case,
                             const double coeff_var_inner_x,
                             const int id_quad_assem_incre,
                             const int id_coeff_diff_value,
                             const double coeff_diff_inner,
                             const int id_coeff_helm_value,
                             const double coeff_helm_inner,                             
                             const double tol_prm,
                             const unsigned int degree,
                             const unsigned int initial_refinement_level,
                             const unsigned int n_total_refinements):
id_case(id_case),
coeff_var_inner_x(coeff_var_inner_x),
id_quad_assem_incre(id_quad_assem_incre),
id_coeff_diff_value(id_coeff_diff_value),
coeff_diff_inner(coeff_diff_inner),
id_coeff_helm_value(id_coeff_helm_value),
coeff_helm_inner(coeff_helm_inner),
tol_prm(tol_prm),
degree(degree),
initial_refinement_level(initial_refinement_level),
n_total_refinements(n_total_refinements),
n_vertices_custom(initial_refinement_level),
fe (degree),
dof_handler (triangulation),
dof_handler_first_refine (triangulation_first_refine),
computing_timer  (cout, TimerOutput::never, TimerOutput::cpu_times)
{
    cout << "================== \n"
         << "  The standard FEM for real-valued problems\n"
         << "  dimension: " << dim << "\n"
         << "  # of quadrature points in one direction: " << n_q_points_in_one_direction << "\n";
    cout << "==================\n";
}


template <int dim>
void Step4_Real<dim>::make_grid ()
{
  cout << "Making grid\n";
  TimerOutput::Scope t(computing_timer, "make_grid");
    
  if(is_mesh_built_in==1)
  {
//     cout << "  mesh built_in\n";
    GridGenerator::hyper_cube (triangulation, 0, 1);
    GridGenerator::hyper_cube (triangulation_uniform_refine, 0, 1);
  }else if(is_mesh_built_in==0 && dim==1)
  {
//     cout << "  mesh custom\n";                                                    // only for one-dimension
//     cout << "  number of vertices: " << n_vertices_custom << "\n";
    vector<Point<dim>> vertices(n_vertices_custom);

    const double delta_vertex = 1.0/(n_vertices_custom-1);
    for (int i = 0; i < n_vertices_custom; ++i)
    {
      vertices[i](0) = i * delta_vertex;
    } 

    vector<array<int,GeometryInfo<dim>::vertices_per_cell>> cell_vertices;
    array<int,GeometryInfo<dim>::vertices_per_cell> array_update;
    for (int i = 0; i<n_vertices_custom-1; ++i)
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
  }else
  {
    cout << "  unavailable mesh scheme\n";
    exit(1);
  }
  
  if(is_tria_info_printed==1)
  {
    print_tria_info(triangulation);
  }
    
//   print_boundary_info(triangulation);
  
  if (is_containing_neumann==0)
  {
    if (dim==1)
    {
      cout << "setting the rightest boundary as Dirichlet type\n";
      triangulation.begin_active()->face(1)->set_boundary_id(0);
      triangulation_uniform_refine.begin_active()->face(1)->set_boundary_id(0);
    }
  }else if (is_containing_neumann==1)
  {
    if(dim==2)
    {
//       adjust_boundary_id_2d(triangulation);
        
//       triangulation.begin_active()->face(0)->set_boundary_id(1);       // left
//       triangulation.begin_active()->face(1)->set_boundary_id(1);       // right
      triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom
      triangulation.begin_active()->face(3)->set_boundary_id(1);       // top      
      
      triangulation_uniform_refine.begin_active()->face(2)->set_boundary_id(1);       // bottom
      triangulation_uniform_refine.begin_active()->face(3)->set_boundary_id(1);       // top      
    }
  }

  if(is_boundary_info_printed==1)
  {
    print_boundary_info(triangulation);   
  }
  
}


template <int dim>
void Step4_Real<dim>::saving_info_of_the_last_refinement ()                             // including the solution and dof_handler
{
  cout << "  saving info of the last refinement\n";
  
  solution_first_refine.reinit (dof_handler.n_dofs());
  solution_first_refine=solution;
  
//     cout << "  solution_first_refine: ";
//     solution_first_refine.print(cout,5); 

  
  triangulation_first_refine.clear();
  triangulation_first_refine.copy_triangulation(triangulation);
  dof_handler_first_refine.distribute_dofs (fe);                          // only used for computing the error customly
  
}


template <int dim>
void Step4_Real<dim>::refining_mesh_globally (unsigned int refinement_level)                 
{
  cout << "\n";
  cout << "Refining the mesh globally";
  
  if(current_refinement_level == initial_refinement_level)
  {
    cout << " for the initial refinement level " << initial_refinement_level << "\n";
  }else
  {
    cout << " for the refinement level " << current_refinement_level << "\n";
  }
  
  
  if (is_mesh_built_in==1)
  {
//     cout << "  Refining mesh\n";
    triangulation.refine_global (refinement_level);    
    triangulation_uniform_refine.refine_global(refinement_level);
  }
  
  if(is_tria_info_printed==1)
  {
    print_tria_info(triangulation);
  }  
  
}

template <int dim>
void Step4_Real<dim>::distorting_the_mesh()                 
{
  cout << "\n";
  cout << "Distorting the mesh with the factor " << factor_distorting << "\n";
  GridTools::distort_random(factor_distorting, triangulation);  
    
  if(is_tria_info_printed==1)
  {
    print_tria_info(triangulation);
  }
}


template <int dim>
void Step4_Real<dim>::refining_mesh_locally ()                 
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
void Step4_Real<dim>::post_processing_info_of_dof_handler ()
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
    
    save_local_dof_indices(obj_string,dof_handler);
    
  }else if(id_basis==1)
  {
    n_dofs_bspline = degree+pow(2,initial_refinement_level);
    cout << n_dofs_bspline;
  }
  
  if(is_distorted==0)
  {
    if(is_coords_of_dofs_of_uniform_printed==1)
    {
      print_coords_of_dofs(obj_string, dof_handler);
    }
  
    if(is_coords_of_dofs_of_uniform_saved==1)
    {
      obj_string = "coords_of_uniform_dofs_of_refine_"+to_string(current_refinement_level);
      
      save_coords_of_dofs(obj_string, dof_handler, current_refinement_level);
      
      sequence_numbers_in_a_txt_file(obj_string);
      
    }
  }else if(is_distorted==1)
  {
    if(is_coords_of_dofs_of_distorted_printed==1)
    {
      obj_string = "dof_handler_distorted";
      print_coords_of_dofs(obj_string, dof_handler);
    }
    
    if(is_coords_of_dofs_of_distorted_saved==1)
    {
      obj_string = "coords_of_distorted_dofs_of_refine_"+to_string(current_refinement_level);
      save_coords_of_dofs(obj_string, dof_handler, current_refinement_level);  
      
      sequence_numbers_in_a_txt_file(obj_string);
      
    }
  }
  
  
}

template <int dim>
void Step4_Real<dim>::setup_system ()
{
  TimerOutput::Scope t(computing_timer, "setup_system");
  cout << "Setting up\n";
  
  cout << "  current_element_degree: " << degree << "\n";
  cout << "  current_refinement_level: " << current_refinement_level << "\n";
  
  if(n_times_called_setup==0)
  {
    if(is_basis_extracted==1)
    {
      extract_basis_function(fe);  
    }      
      
    if(is_fe_info_printed==1)
    {
        obj_string = "fe";
        print_fe_info(obj_string, fe);  
    }
    //   print_quadrature_info_on_reference_cell(quadrature_formula);
  }
  
  dof_handler.distribute_dofs (fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  
  DoFTools::make_sparsity_pattern (dof_handler, dsp);               // dispensing non-zero entries

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  solution_0.reinit (dof_handler.n_dofs());
  
  system_rhs.reinit (dof_handler.n_dofs());
  
  n_times_called_setup++;
  
}

template <int dim>
void Step4_Real<dim>::assemble_system ()
{
  cout << "Assembling\n";
  TimerOutput::Scope t(computing_timer, "assemble_system");
  QGauss<dim-1> face_quadrature_formula(degree+2);   
  
  const Coeff_Diff_Real<dim> obj_coeff_diff(id_case, id_coeff_diff_value, coeff_diff_inner);
  const Coeff_First_Derivative_A_Real<dim> obj_coeff_first_derivative_a(id_case, 1);                        // second argument not used
  
  const Coeff_Helm_Real<dim> obj_coeff_helm(id_case, id_coeff_helm_value, coeff_helm_inner); 
  const RightHandSide_Real<dim> obj_right_hand_side(id_case, coeff_var_inner_x, id_coeff_diff_value, coeff_diff_inner, id_coeff_helm_value, coeff_helm_inner);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients | update_hessians |
                           update_quadrature_points | update_JxW_values | update_jacobians);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  
  vector<Tensor<2,dim>> value_coeff_diff (n_q_points, Tensor<2,dim>());
  vector<Tensor<2,dim>> value_coeff_diff_face (n_face_q_points, Tensor<2,dim>());
  
  vector<Tensor<1,dim>> value_coeff_first_derivative_a (n_q_points, Tensor<1,dim>());
  
  vector<double> value_coeff_helm (n_q_points);
  
  vector<Vector<double>> value_rhs (n_q_points, Vector<double>(n_number_components));
  
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  
  
  
//   vector<types::global_dof_index> all_dof_indices (dof_handler.n_dofs());
//   
//   cout << "all_dof_indices: \n";
//   print_vector(all_dof_indices);
//   
//   dof_handler.get_dof_indices(all_dof_indices);
  

  const ExactSolution_Step_4_Real<dim> exact_solution(id_case,coeff_diff_inner);  
  
  
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  
  fe_values.reinit (cell);
  
  if(is_l2_norm_coefficient_numerical == 1)
  {
    l2_norm_coeff_diff = compute_l2_norms_of_coefficient_numerically_core_tensor_2_dim_real(triangulation, dof_handler, fe_values, obj_coeff_diff);
    l2_norm_coeff_helm = compute_l2_norms_of_coefficient_numerically_core_real(triangulation, dof_handler, fe_values, obj_coeff_helm);
    l2_norm_coeff_first_derivative_a = compute_l2_norms_of_coefficient_numerically_core_tensor_1_dim_real(triangulation, dof_handler, fe_values, obj_coeff_first_derivative_a);
    
    if(is_distorted==0)
    {
      vec_l2_norm_coeff_diff.push_back(l2_norm_coeff_diff);
      vec_l2_norm_coeff_helm.push_back(l2_norm_coeff_helm);
      vec_l2_norm_coeff_first_derivative_a.push_back(l2_norm_coeff_first_derivative_a);
    }else if(is_distorted==1)
    {
      vec_l2_norm_coeff_diff_distorted.push_back(l2_norm_coeff_diff);
      vec_l2_norm_coeff_helm_distorted.push_back(l2_norm_coeff_helm);
      vec_l2_norm_coeff_first_derivative_a_distorted.push_back(l2_norm_coeff_first_derivative_a);        
    }
  }     
  
  
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    cell_matrix = 0;
    cell_rhs = 0;
    
//     cout << "  cell " << cell->active_cell_index() << endl;
      
/*    for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
    {
      cout << "      at vertex " << cell->vertex_index(vertex) << ", coords: " << cell->vertex(vertex) << "\n";
    }
    cout << endl; */     
    
    obj_coeff_diff.value (fe_values.get_quadrature_points(), value_coeff_diff);
    
//     cout << "value_coeff_diff: \n";
//     print_vector(value_coeff_diff);
//     
//     for(unsigned int i = 0; i<value_coeff_diff.size(); ++i)
//     {
//       cout << std::scientific << value_coeff_diff[i] << "\n";
//     }
    
    
    obj_coeff_first_derivative_a.value (fe_values.get_quadrature_points(), value_coeff_first_derivative_a);
    
    obj_coeff_helm.value (fe_values.get_quadrature_points(), value_coeff_helm);
    
    obj_right_hand_side.value_rhs (fe_values.get_quadrature_points(), value_rhs);
    
    
//     cout << "value_coeff_first_derivative_a\n";
//     print_vector(value_coeff_first_derivative_a);
    

//     cout  << "  jacobian: " << fe_values.jacobian(0)[0] << "\n";                                      // jacobians of each quadrature point are the same
//     cout  << "  coordinates: ";
//     for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//     {
//         cout << fe_values.quadrature_point(q_index)(0) << " ";
//     }    
//     cout << "\n";    
//     cout  << "  JxW: ";
//     for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//     {
//         cout << fe_values.JxW (q_index) << " ";
//     }    
//     cout << "\n";
    
//     cout << "values\n";
//     for (unsigned int i=0; i<dofs_per_cell; ++i)
//     {
//         cout << "basis " << i << ": ";
//         for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//         {
//           cout << fe_values.shape_value (i, q_index) << " ";                  // 
//         }
//         cout << "\n";
//     }
//     
//     cout << "gradients\n";
//     for (unsigned int i=0; i<dofs_per_cell; ++i)
//     {
//         cout << "basis " << i << ": ";
//         for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//         {
//           cout << fe_values.shape_grad (i, q_index) << " ";  // gradients update according to jacobian, 
                                                                //  while values do not, cf. transformation from original coordinates to reference coordinates
//         }
//         cout << "\n";
//     }
    
#if 0    
    unsigned int n_rows = 3;
    unsigned int n_cols = 3;
  
    vector<vector<double>> data_value(n_rows);
    vector<vector<double>> data_gradient(n_rows);
    for (unsigned int i = 0 ; i < n_rows; i++)
    {
      data_value[i].resize(n_cols);  
      data_gradient[i].resize(n_cols);  
    }

    cout << "\n";
    cout << "reading data of isogeometric analysis\n";
    ifstream fid_value;
    fid_value.open("basis_bspline/"+to_string(initial_refinement_level+1)+"_refine_"+to_string(initial_refinement_level)+"/bspline_fe_values_"+to_string(cell->active_cell_index())+".txt");
    ifstream fid_gradient;
    fid_gradient.open("basis_bspline/"+to_string(initial_refinement_level+1)+"_refine_"+to_string(initial_refinement_level)+"/bspline_fe_gradients_"+to_string(cell->active_cell_index())+".txt");
  
    for (unsigned int i = 0 ; i < n_rows; ++i)
    {
      for (unsigned int j = 0; j < n_cols; ++j)
      {  
        fid_value >> data_value[i][j];  
        fid_gradient >> data_gradient[i][j];
        
//         if (i==0)
//         {
//           fid_value >> data_value[i][j];  
//           fid_gradient >> data_gradient[i][j];    
//         }else if (i==1)
//         {
//           fid_value >> data_value[2][j];  
//           fid_gradient >> data_gradient[2][j];  
//         }else if (i==2)
//         {
//           fid_value >> data_value[1][j];  
//           fid_gradient >> data_gradient[1][j];
//         }
      }
    }  
    fid_value.close();
    fid_gradient.close();

    cout << "data_value: \n";
    for (unsigned int i=0; i<data_value.size(); ++i)
    {
      cout << "basis " << i << ": ";
      for (unsigned int j = 0; j < n_cols; ++j)
      {    
        cout << data_value[i][j] << " ";
      }
      cout << "\n";
    }
  
    cout << "data_gradient: \n";
    for (unsigned int i=0; i<data_gradient.size(); ++i)
    {
      cout << "basis " << i << ": ";
      for (unsigned int j = 0; j < n_cols; ++j)
      {    
        cout << data_gradient[i][j] << " ";
      }
      cout << "\n";
    }
    
#endif
    
    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          if (id_case!=9)
          {
//             if (id_basis==0)
//             {
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) * (value_coeff_diff[q_index] * fe_values.shape_grad (j, q_index))
                                   + fe_values.shape_value (i, q_index) * (fe_values.shape_value (j, q_index) * value_coeff_helm[q_index]            // shape_value() defined in fe_values.h
                                                                          + value_coeff_first_derivative_a[q_index]*fe_values.shape_grad (j, q_index)))
                                   * fe_values.JxW (q_index);  
//             }else if(id_basis==1)
//             {
// //               cell_matrix(i,j) += (data_gradient [i][q_index] * data_gradient [j][q_index]) * fe_values.JxW (q_index) * value_coeff_diff[q_index][0][0]);
//             }
          }
          
          if(id_case==9)                           // dealing with first-order differential equations
          {
            cell_matrix(i,j) -= fe_values.shape_value (i, q_index) *
                             fe_values.shape_grad (j, q_index)[0] *
                             fe_values.JxW (q_index);
          } 
        }
        
//         if (id_basis==0)
//         {
          cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                          value_rhs[q_index][0] *
                          fe_values.JxW (q_index));
//         }else if(id_basis==1)
//         {
//           cell_rhs(i) += (data_value[i][q_index] *
//                           value_rhs[q_index][0] *
//                           fe_values.JxW (q_index));  
//         }
      }
      
//       cout << "at quadrature point: " << fe_values.quadrature_point(q_index)[0] << "\n";
//       cell_matrix.print(cout, 5);
      
    }
    
//     cout << "the resulting cell_matrix: \n";
//     cell_matrix.print(cout, 5);
//     cout << "the resulting cell_rhs: \n";
//     cell_rhs.print(cout, 5);
    
      
    for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
    {
      if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 1))                                  //treat the Neumann boundary conditions
      {
//         cout  << "  Imposing Neumann boundaries on ";
//         cout << "cell " << cell->active_cell_index() << " (" << cell->vertex(0) << ", " << cell->vertex(dim==1?1:3) << "), ";                
//         cout << "face " << face_n << " ";   
//         for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
//         {
//           cout << "(" << cell->face(face_n)->vertex(vertex) << ")";
//         }
//         cout << endl;
             
        fe_face_values.reinit (cell, face_n);
        
        obj_coeff_diff.value (fe_face_values.get_quadrature_points(), value_coeff_diff_face);
        
/*        cout << "value_coeff_diff_face: \n";
        print_vector(value_coeff_diff_face);  */      
        
/*        cout << "fe_face_values\n";
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            cout << "basis " << i << ": ";
            for (unsigned int q_index=0; q_index<n_face_q_points; ++q_index)
            {
            cout << fe_face_values.shape_value (i, q_index) << " ";
            }
            cout << "\n";
        }  */       

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
    
//     cout << "  local_dof_indices: ";
//     for (unsigned int i=0; i<dofs_per_cell; ++i)
//     {
//       cout << local_dof_indices[i] << " ";
//     }
//     cout << "\n";
    
    
    if (id_basis==0)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
//         cout << "[" << i << "] " << local_dof_indices[i] << "\n";
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
//           cout << "[" << j << "] " << local_dof_indices[j] << "\n";
        
//           cout << "cell_matrix(" << i << "," << j << "): " << cell_matrix(i,j) << "\n";
        
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));
        }
//         cout << "\n";

        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }  
    }else if(id_basis==1)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        
//         cout << "[" << i << "] " << local_dof_indices[i] << "\n";
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
//           cout << "[" << j << "] " << local_dof_indices[j] << "\n";
//           cout << "cell_matrix(" << i << "," << j << "): " << cell_matrix(i,j) << "\n";
        
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));
        }

        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    
//       cout  << "\n";  
//     
//       cout << "system_matrix after cell " << cell->active_cell_index() << endl;
//       system_matrix.print_formatted(cout);
//       cout << '\n';
//       cout << "system_rhs after cell " << cell->active_cell_index() << endl;
//       system_rhs.print(cout,5);
    }
  }
  
  if (is_lhs_and_rhs_before_diri_BC_printed==1)
  {
    cout << '\n';
    cout << "system_matrix before applying Dirichlet BC:\n";
    system_matrix.print_formatted(cout,20);
    cout << "system_rhs after applying Neumann BC but not Dirichlet BC:\n";
    system_rhs.print(cout,5);
  }    

//   cout << "  Imposing Dirichlet boundary conditions\n";
  if (is_strong == 1)
  {
//     cout << "    strong way\n";
    map<types::global_dof_index,double> boundary_values;
    
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ExactSolution_Step_4_Real<dim>(id_case,coeff_var_inner_x),                  // note we only use the value item
                                              boundary_values);
    
    cout << "    Dirichlet boundary values: \n";
    print_map(boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
    
  }else if (is_strong == 0)
  {
    cout << "    weak way\n";
    ExactSolution_Step_4_Real<dim> boundary_values_for_weak_imposition(id_case,coeff_var_inner_x);  
    
    vector<Point<dim>> coords_quad_boundary(2);
    coords_quad_boundary[0](0)=0.0;
    coords_quad_boundary[1](0)=1.0;
    
    FEValues<dim> fe_values_boundary (fe, coords_quad_boundary,
                             update_values   | update_gradients | update_quadrature_points );
  
    dof_handler.distribute_dofs (fe);  
    
    typename DoFHandler<dim>::active_cell_iterator 
    first_cell = dof_handler.begin_active(), 
    intermediate_cell = dof_handler.begin_active(),
    last_cell = dof_handler.begin_active();
  
    fe_values_boundary.reinit(first_cell);
    first_cell->get_dof_indices (local_dof_indices);

    if (first_cell->face(0)->boundary_id() == 0) 
    { 
      for (unsigned int j=0; j<dofs_per_cell; ++j)                            // 
      {
        system_matrix.add (local_dof_indices[0],
                            local_dof_indices[j],
                            fe_values_boundary.shape_grad(j,0)[0]);
        system_matrix.add (local_dof_indices[j],
                           local_dof_indices[0],
                            -fe_values_boundary.shape_grad(j,0)[0]);  
        system_matrix.add (local_dof_indices[0], local_dof_indices[0], prm_penalty_weak);  
        
        system_rhs(local_dof_indices[j]) -= fe_values_boundary.shape_grad(j,0)[0]*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
        system_rhs(local_dof_indices[0]) += prm_penalty_weak*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
      }
    }

    for(unsigned int i = 0; i < triangulation.n_active_cells()-1; ++i)
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
        system_matrix.add (local_dof_indices[1],
                            local_dof_indices[j],
                            -fe_values_boundary.shape_grad(j,1)[0]);
        system_matrix.add (local_dof_indices[j],
                           local_dof_indices[1],
                            fe_values_boundary.shape_grad(j,1)[0]);  
        system_matrix.add (local_dof_indices[1], local_dof_indices[1], -prm_penalty_weak);  
        
        system_rhs(local_dof_indices[j]) += fe_values_boundary.shape_grad(j,1)[0]*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
        system_rhs(local_dof_indices[1]) -= prm_penalty_weak*boundary_values_for_weak_imposition.value(fe_values_boundary.quadrature_point(0)); 
      }
    }
  } else
  {
    cout << "  Dirichlet boundary conditions not treated\n";
  }

  if (is_lhs_and_rhs_after_diri_BC_printed==1)
  {
    cout << '\n';
    cout << "system_matrix after applying Dirichlet BC:\n";
    system_matrix.print_formatted(cout);
    cout << "system_rhs after applying both Neumann and Dirichlet BC:\n";
    system_rhs.print(cout,5);
    cout << '\n';
  }

  ostringstream streamObj;
  streamObj << fixed;
  streamObj << setprecision(0);
  streamObj << this->coeff_var_inner_x;
  
  if (is_matrix_after_BC_stored==1)
  {
    obj_string="system_matrix_coeff_"+ streamObj.str() +"_deg_"+to_string(degree)+"_ref_"+to_string(initial_refinement_level);
    save_system_matrix_to_txt(obj_string,system_matrix);
  }
  
  if(is_rhs_after_BC_stored==1)
  {
    obj_string="system_rhs_coeff_"+ streamObj.str() +"_deg_"+to_string(degree)+"_ref_"+to_string(initial_refinement_level);
    save_Vector_to_txt(obj_string,system_rhs);
  }
  
  n_times_called_assemble++;
  
}


template <int dim>
void Step4_Real<dim>::solve ()
{
  TimerOutput::Scope t(computing_timer, "solve");
  cout << "Solving\n";

  if (tol_prm != 0)
  {
    is_UMFPACK = 0;
  }
    
  if (is_UMFPACK==1)          
  {
    cout << "  UMFPack solver\n";
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
  }else if(is_UMFPACK==0)   
  {
    cout << "  CG solver, tol_prm: " << tol_prm << endl;
    SolverControl           solver_control (1e+8, tol_prm);
    SolverCG<>              solver (solver_control);
    solver.solve (system_matrix, solution, system_rhs,
                    PreconditionIdentity());
    CG_iteration_times = solver_control.last_step();
    cout << "  " << solver_control.last_step() << " CG iterations needed to obtain convergence." << endl;
  }
  
  n_times_called_solve++;
}


template <int dim>
void Step4_Real<dim>::compute_integrals ()
{
  cout << "Computing integrals\n";
  const Functions::ZeroFunction<dim> zero_function;                          // defined in /base/function.h
  
  const ExactSolution_Step_4_Real<dim> exact_solution(id_case, coeff_var_inner_x);
    
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  
  VectorTools::integrate_difference (dof_handler,                 // alternative: 1. the opposite of u_mean_inte_numerical: VectorTools::mean, VectorTools::mean   
                                     solution,                    //              2. f_l2_inte_numerical: solution -> system_rhs
                                     zero_function,
                                     difference_per_cell,
                                     qgauss_integrate_difference,
                                     VectorTools::L2_norm);
  l2_inte_u_numerical = VectorTools::compute_global_error(triangulation,
                                                          difference_per_cell,
                                                          VectorTools::L2_norm);
  
  VectorTools::integrate_difference (dof_handler,
                                     solution_0,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_integrate_difference,
                                     VectorTools::L2_norm);
  l2_inte_u_analytical = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::L2_norm);

  VectorTools::integrate_difference (dof_handler,
                                     solution_0,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_integrate_difference,
                                     VectorTools::H1_seminorm);
  l2_inte_nabla_u_analytical = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::H1_seminorm);

  VectorTools::integrate_difference (dof_handler,
                                     solution_0,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_integrate_difference,
                                     VectorTools::H2_seminorm);                                  // selft-defined
  
  l2_inte_grad_grad_u_analytical = VectorTools::compute_global_error(triangulation,
                                                                     difference_per_cell,
                                                                     VectorTools::H2_seminorm); 
  
  if(dim==2)
  {
    VectorTools::integrate_difference (dof_handler,
                                        solution_0,
                                        exact_solution,
                                        difference_per_cell,
                                        qgauss_integrate_difference,
                                        VectorTools::H2_divnorm);                                  // selft-defined
    l2_inte_delta_u_analytical = VectorTools::compute_global_error(triangulation,
                                                                difference_per_cell,
                                                                VectorTools::H2_divnorm);
  }
  
    
//   VectorTools::integrate_difference (dof_handler,
//                                      solution_0,
//                                      exact_solution,
//                                      difference_per_cell,
//                                      qgauss_integrate_difference,
//                                      VectorTools::mean);
//   u_mean_inte_analytical = VectorTools::compute_global_error(triangulation,
//                                                              difference_per_cell,
//                                                              VectorTools::mean);
  
 
}

template <int dim>
void Step4_Real<dim>::compute_errors_built_in ()
{
  cout << "Computing errors using built-in functions\n";
//   TimerOutput::Scope t(computing_timer, "compute_errors_built_in");
  ExactSolution_Step_4_Real<dim> exact_solution(id_case, coeff_var_inner_x);
  
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  
#if 1
  
  VectorTools::integrate_difference (dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    qgauss_integrate_difference,
                                    VectorTools::L2_norm);
  l2_error_abs = VectorTools::compute_global_error(triangulation,
                                                   difference_per_cell,
                                                   VectorTools::L2_norm);

//   cout << "  difference_per_cell\n";
//   cout << difference_per_cell;

  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_integrate_difference,
                                     VectorTools::H1_seminorm);
  h1_semi_error_abs = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H1_seminorm);

#endif

  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     exact_solution,
                                     difference_per_cell,
                                     qgauss_integrate_difference,
                                     VectorTools::H2_seminorm);
  h2_semi_error_abs = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H2_seminorm);
  
  if(dim==2)
  {
    VectorTools::integrate_difference (dof_handler,
                                        solution,
                                        exact_solution,
                                        difference_per_cell,
                                        qgauss_integrate_difference,
                                        VectorTools::H2_divnorm);
    l2_error_delta_u = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::H2_divnorm);
  }
  
  if (is_distorted==0)
  {
    vec_error[cycle_global][0] = l2_error_abs;
    vec_error[cycle_global][1] = h1_semi_error_abs;
    vec_error[cycle_global][2] = h2_semi_error_abs;
  }else if (is_distorted==1)
  {
    vec_error_distorted[cycle_global][0] = l2_error_abs;
    vec_error_distorted[cycle_global][1] = h1_semi_error_abs;
    vec_error_distorted[cycle_global][2] = h2_semi_error_abs;      
  }
  
  
  if(cycle_global>0)
  {
    for (unsigned int i=0; i<3; ++i)
    {
      convergence_rate = -log10(vec_error[cycle_global][i]/vec_error[cycle_global-1][i])/log10(2);
//       cout << "convergence_rate " << convergence_rate << "\n";
      vec_convergence_rate[cycle_global-1][i] = convergence_rate;
      
      convergence_rate = -log10(vec_error_distorted[cycle_global][i]/vec_error_distorted[cycle_global-1][i])/log10(2);
      vec_convergence_rate_distorted[cycle_global-1][i] = convergence_rate;
      
      
    }
  }
  
  
  if(dim==2)
  {
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       exact_solution,
                                       difference_per_cell,
                                       qgauss_integrate_difference,
                                       VectorTools::H2_seminorm_00);
    h2_semi_error_abs_00 = VectorTools::compute_global_error(triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_00);  

    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       exact_solution,
                                       difference_per_cell,
                                       qgauss_integrate_difference,
                                       VectorTools::H2_seminorm_01);
    h2_semi_error_abs_01 = VectorTools::compute_global_error(triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_01);  
  
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       exact_solution,
                                       difference_per_cell,
                                       qgauss_integrate_difference,
                                       VectorTools::H2_seminorm_10);
    h2_semi_error_abs_10 = VectorTools::compute_global_error(triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_10);  

    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       exact_solution,
                                       difference_per_cell,
                                       qgauss_integrate_difference,
                                       VectorTools::H2_seminorm_11);
    h2_semi_error_abs_11 = VectorTools::compute_global_error(triangulation,
                                                             difference_per_cell,
                                                             VectorTools::H2_seminorm_11);  
  }
  
  n_times_called_computing_error_built_in++;
  
}


template <int dim>
void Step4_Real<dim>::compute_errors_custom ()
{
  cout << "\n";
  cout << "//////////////////\n";
  cout << "Computing errors customly\n";

  saving_info_of_the_last_refinement();
  
  refining_mesh_globally(1);
  
  setup_system ();
  assemble_system ();
  solve ();
  
//   cout << "solution_second_refine reads ";
//   solution.print(cout,5);   
  
  vector<double> vector_error(3);
  error_computation_custom_core_std(solution_first_refine, solution, dof_handler_first_refine, dof_handler, quadrature_formula, n_number_components,vector_error);
    
  l2_error_custom = vector_error[0];
  h1_semi_error_custom = vector_error[1];
  h2_semi_error_custom = vector_error[2]; 
  
  cout << "  l2_error_custom: " << l2_error_custom << "\n";                  // we allow slight difference from that using the built-in function since we use finer solution as exact
  cout << "  h1_semi_error_custom: " << h1_semi_error_custom << "\n";
  cout << "  h2_semi_error_custom: " << h2_semi_error_custom << "\n";     
  
  cout << "//////////////////\n\n";
}


template <int dim>
void Step4_Real<dim>::print_integrals_errors_and_CPU ()
{
  cout << "Printing integrals, errors and CPU time\n";
  
  cout << "  @errors\n";
  cout << "    l2_error_abs: " << l2_error_abs << "\n";
  cout << "    h1_semi_error_abs: " << h1_semi_error_abs << "\n";
  cout << "    h2_semi_error_abs: " << h2_semi_error_abs << "\n";
  if(dim==2)
  {
    cout << "      00: " << h2_semi_error_abs_00 << "\n";
    cout << "      01: " << h2_semi_error_abs_01 << "\n";
    cout << "      10: " << h2_semi_error_abs_10 << "\n";
    cout << "      11: " << h2_semi_error_abs_11 << "\n";
    cout << "    l2_error_delta_u: " << l2_error_delta_u << "\n";
  }
  
  
  cout << "  @l2 norms\n";
  cout << "    1. analytically\n";
  cout << "    l2_inte_u_analytical: " << l2_inte_u_analytical << "\n";
  cout << "    l2_inte_nabla_u_analytical: " << l2_inte_nabla_u_analytical << "\n";    
  cout << "    l2_inte_grad_grad_u_analytical: " << l2_inte_grad_grad_u_analytical << "\n";
  cout << "    l2_inte_delta_u_analytical: " << l2_inte_delta_u_analytical << "\n";
  
  cout << "    2. numerically\n";
  cout << "    l2_inte_u_numerical = " << l2_inte_u_numerical << "\n";
  cout << "    l2_norm_coeff_diff: " << l2_norm_coeff_diff << "\n";
  cout << "    l2_norm_coeff_helm: " << l2_norm_coeff_helm << "\n";
  //     cout << "  u_mean_inte_numerical = " << u_mean_inte_numerical << ", u_mean_inte_analytical = " << u_mean_inte_analytical << "\n";
  //     cout << "  f_l2_inte_numerical = " << f_l2_inte_numerical << ", f_l2_inte_analytical = " << f_l2_inte_analytical << "\n";
  
  cout << "  @cpu time\n";
  cout << "    total_CPU_time_per_run: " << total_CPU_time_per_run << "\n";
  
  
//   l2_error_rela = l2_error_abs/l2_inte_u_analytical;
//   h1_semi_error_rela = h1_semi_error_abs/l2_inte_nabla_u_analytical;  
//   h2_semi_error_rela = h2_semi_error_abs/l2_inte_grad_grad_u_analytical;
  
//   cout << "  l2_error_rela = " << l2_error_rela << "\n";
//   cout << "  h1_semi_error_rela = " << h1_semi_error_rela << "\n";
//   cout << "  h2_semi_error_rela: " << h2_semi_error_rela << "\n";
//   
//   cout << "\n";
}


template <int dim>
void Step4_Real<dim>::output_results ()
{
  TimerOutput::Scope t(computing_timer, "output results");
//   cout << "Storing results\n";
  
//   ComputeSurface_Real<dim> surface;
//   ComputeVelocity1_Real<dim> velocity;
//   Compute2ndderivative1_Real<dim> secondderivative;

  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  
  data_out.build_patches ();
  
  ofstream output ("solution-real-sm.vtk");
  data_out.write_vtk (output);  
}


template <int dim>
void Step4_Real<dim>::storing_error_cpu_time_to_file()
{
//   cout << "Storing errors and cpu time to a file\n";
  ofstream myfile;
  
  if (is_distorted==0)
  {
    myfile.open ("data_error_sm_real.txt", ofstream::app);             // trunc  
  }else if(is_distorted==1)
  {
    myfile.open ("data_error_sm_real_distorted.txt", ofstream::app);
  }
  
  myfile << current_refinement_level << " ";        
#if 1 
  myfile << triangulation.n_vertices() <<" ";
  myfile << dof_handler.n_dofs() << " ";
  myfile << l2_error_abs << " ";
  myfile << h1_semi_error_abs << " ";
  myfile << h2_semi_error_abs << " ";
  
  if(dim==2)
  {
    myfile << h2_semi_error_abs_00 << " ";
    myfile << h2_semi_error_abs_01 << " ";
    myfile << h2_semi_error_abs_10 << " ";
    myfile << h2_semi_error_abs_11 << " ";
    myfile << l2_error_delta_u << " ";
  }
  
  myfile << l2_inte_u_numerical << " ";
  myfile << l2_norm_coeff_diff << " ";
  myfile << l2_norm_coeff_helm << " ";
  myfile << total_CPU_time_per_run << " ";
    
  if (is_L2_norm_stored == 1)
  {
    myfile << l2_inte_u_analytical << "\n";
  }
  if (is_solver_info_stored == 1)
  {
    if (is_UMFPACK == 1)
    {
      myfile << "UMF" << "\n";
    }else if (is_UMFPACK == 0)
    {
      myfile << CG_iteration_times << "\n";
    }
  }
#endif
  myfile.close();
}


template <int dim>
void Step4_Real<dim>::printPattern(int radius)
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


template <int dim>
void Step4_Real<dim>::conducting_one_complete_run ()
{
  printPattern(5);
  cout << "Conducting one complete run";
  
  if (is_distorted==0)
  {
    cout << " for the uniform mesh\n";
  }else if(is_distorted==1)
  {
    cout << " for the distorted mesh\n";
  }
  
  setup_system ();
  post_processing_info_of_dof_handler();
  
  assemble_system ();
  
  solve ();
  
  if (are_integrals_computed==1)
  {
    compute_integrals ();
  }
  
  compute_errors_built_in ();
    
  if(is_results_outputted==1)
  {
    output_results ();  
  }
    
  if (is_solution_printed == 1)
  {
    cout << "  solution:\n  ";
    solution.print(cout,5);
  }     
    
  if (is_custom_method_used_for_error==1)
  {
    compute_errors_custom ();
  }

  total_CPU_time_per_run = computing_timer.return_total_cpu_time();
  
  if(is_distorted==0)
  {
    vec_cpu_time.push_back(total_CPU_time_per_run);
  }else if(is_distorted==1)
  {
    vec_cpu_time_distorted.push_back(total_CPU_time_per_run);
  }
    
  if(is_integrals_and_errors_printed==1)
  {
    print_integrals_errors_and_CPU();
  }
    
  storing_error_cpu_time_to_file();
  
}


template <int dim>
void Step4_Real<dim>::printing_results_of_all_refinements ()
{
  cout << "\n";
  cout << "===========================\n";
  cout << "Results of all refinements\n";
  
  cout << "  initial_refinement_level: " << initial_refinement_level << "\n";
  cout << "  n_total_refinements: " << n_total_refinements << "\n";
  
  if(is_mesh_uniform_computed==1)
  {
    cout << "  vec_error:\n";
    print_vector_in_vector(vec_error);
    cout << "  vec_convergence_rate:\n";
    print_vector_in_vector(vec_convergence_rate);

    cout << "  vec_l2_norm_coeff_diff:\n";
    print_vector(vec_l2_norm_coeff_diff);  
    cout << "  vec_l2_norm_coeff_helm:\n";
    print_vector(vec_l2_norm_coeff_helm);  
    cout << "  vec_l2_norm_coeff_first_derivative_a:\n";
    print_vector(vec_l2_norm_coeff_first_derivative_a);

    cout << "  CPU time:\n";
    print_vector(vec_cpu_time);
  }

  if(is_mesh_distorted_computed==1)
  {
    cout << "  vec_error_distorted:\n";
    print_vector_in_vector(vec_error_distorted);
    cout << "  vec_convergence_rate_distorted:\n";
    print_vector_in_vector(vec_convergence_rate_distorted);

    cout << "  vec_l2_norm_coeff_diff_distorted:\n";
    print_vector(vec_l2_norm_coeff_diff_distorted);  
    cout << "  vec_l2_norm_coeff_helm_distorted:\n";
    print_vector(vec_l2_norm_coeff_helm_distorted);  
    cout << "  vec_l2_norm_coeff_first_derivative_a_distorted:\n";
    print_vector(vec_l2_norm_coeff_first_derivative_a_distorted);  

    cout << "  CPU time distorted:\n";
    print_vector(vec_cpu_time_distorted);
  }
  

  cout << "  nr. of functions called:\n";
  cout << "    setup: " << n_times_called_setup << "\n";
  cout << "    assemble: " << n_times_called_assemble << "\n";
  cout << "    solve: " << n_times_called_solve << "\n";
  cout << "    computing error built-in: " << n_times_called_computing_error_built_in << "\n";
  
}
  

template <int dim>
void Step4_Real<dim>::run ()
{
  
  vec_error.resize(n_total_refinements,vector<double>(3,0));
  vec_error_distorted.resize(n_total_refinements,vector<double>(3,0));
  
  vec_convergence_rate.resize(n_total_refinements-1,vector<float>(3,0));
  vec_convergence_rate_distorted.resize(n_total_refinements-1,vector<float>(3,0));
    
  make_grid();
  current_refinement_level = initial_refinement_level;
  refining_mesh_globally(initial_refinement_level);
  
  
  for (unsigned int cycle = 0; cycle < n_total_refinements; cycle++)                        // cycle used for adaptive mesh refinement
  {
    cycle_global = cycle;

#if 1
    
    if (is_mesh_uniform_computed == 1)
    {
      is_distorted = 0;
      conducting_one_complete_run();  
    }
    
    if(is_mesh_distorted_computed==1)                       // only the input mesh is different from that of the uniform case
    {
      is_distorted = 1;
      distorting_the_mesh();
      conducting_one_complete_run();
      is_distorted = 0;
    }
    
    triangulation.clear();
    triangulation.copy_triangulation(triangulation_uniform_refine);                 // we base on the uniform mesh for the next refinement
    
    if(cycle < n_total_refinements-1)
    {  
      current_refinement_level++;
      refining_mesh_globally(1);
    }
    
//     if (cycle>0)
//     {
//       refining_mesh_locally();
//     }
    
#endif

  }
  
  printing_results_of_all_refinements();
  
}



#endif
