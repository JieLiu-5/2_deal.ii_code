
#ifndef MAINCLASS_STEP_20_REAL_H
#define MAINCLASS_STEP_20_REAL_H

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bdm.h>
// #include <deal.II/fe/fe_bdm_custom.h>                   // including me for adjusting the function in class FE_BDM

#include <deal.II/fe/fe_dgp.h>
// #include <deal.II/fe/fe_dgp_custom.h>                   

#include"../../6_headers/1_cpp/1_dealii/base_class_real.h"

#include<schur_complement_step_20.h>
#include<../../step-4/include/1_real_valued/exact_solution_step_4_real.h>
#include<1_real_valued/exact_solution_step_20_real.h>

#include"../../6_headers/1_cpp/1_dealii/derived_class_real.h"

template <int dim>
class MixedLaplaceProblem_Real
{
  public:
    MixedLaplaceProblem_Real (const unsigned int id_fe_system,
                              const unsigned int id_case, 
                              const double coeff_var_inner_x,
                              const int id_quad_assem_incre, 
                              const unsigned int id_coeff_diff_value, 
                              const double coeff_diff_inner,
                              const unsigned int id_loc_coeff_diff, 
                              const unsigned int id_coeff_helm_value,                       // not used yet, jan. 15, 2021
                              const double coeff_helm_inner,
                              const double tol_prm_schur, 
                              const unsigned int degree, 
                              const unsigned int refine);
    
    void run ();
    
  private:
    const unsigned int id_fe_system = 1;
    
    const unsigned int id_case = 1;
    const double coeff_var_inner_x;
    
    const int id_quad_assem_incre = 1;
    
    const unsigned int id_coeff_diff_value = 1;
    const double coeff_diff_inner;
    const unsigned int id_loc_coeff_diff = 0;                 // '0' for normal; 
                                                              // '1' for coeff_diff staying with dp/dx; 
                                                              // '2' for u=-dp/dx
    
    const unsigned int id_coeff_helm_value;
    const double coeff_helm_inner;
    
    const double tol_prm_schur = 1e-16;
    const unsigned int   degree;
    const unsigned int   refine;    

    
    void make_grid_and_dofs ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void compute_errors_and_l2_norms_built_in ();
    void compute_errors_custom ();
    void output_results () const;
    void printToFile();    
    

    Triangulation<dim>   triangulation;
    Triangulation<dim>   triangulation_first_refine;
                                                           
#if 1
    
    FE_RaviartThomas<dim>        fe_velocity;                    // changes on this line also involves line ~217
    FE_DGQ<dim>                  fe_pressure;
    
//     FE_BDM<dim>                  fe_velocity;
//     FE_DGP<dim>                  fe_pressure;

#else
    
    FE_Q<dim>        fe_velocity;
    FE_DGQ<dim>                  fe_pressure;

#endif
    
    
    FESystem<dim>        fe;
    
    AffineConstraints<real_t> constraints;
    
    const unsigned int n_q_points_in_one_direction = degree+id_quad_assem_incre;
    QGauss<dim> quadrature_formula = QGauss<dim>(n_q_points_in_one_direction);
    const unsigned int n_q_points = quadrature_formula.size();      // affecting round-off error;   
                                                                    // degree+0 is not possible for P4/P3 elements with schur complement
    
    DoFHandler<dim>      dof_handler;
    
    DoFHandler<dim>      dof_handler_first_refine;
    DoFHandler<dim>      dof_handler_pressure_first_refine;
    DoFHandler<dim>      dof_handler_velocity_first_refine;
    
    unsigned int n_u, n_p;
    unsigned int n_u_per_cell_boundary=0;
    
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<real_t> system_matrix;
    BlockSparseMatrix<real_t> system_matrix_no_essential_bc;

    BlockVector<double>       solution;
    BlockVector<double>       solution_first_refine;
    BlockVector<double>       solution_0;
    
    Vector<double>            solution_pressure;
    Vector<double>            solution_velocity;
    
    BlockVector<double>       system_rhs;
    BlockVector<double>       system_rhs_no_essential_bc;
    
    TimerOutput                               computing_timer;
    
    unsigned int n_number_components = 1;
    string obj_string="";    
    
    unsigned int is_neumann = 1;
    unsigned int is_constraints=0;
    
    unsigned int is_dof_handler_first_refine_dealt = 0;
    
    unsigned int is_matrices_assembled_no_essential_bc = 1;
    unsigned int is_l2_norm_coefficient_numerical = 1;
    
    unsigned int is_quadrature_info_printed_on_global_cell = 0;
    
    unsigned int is_basis_related_stored=0;
    
    unsigned int is_value_coeff_printed = 0;
    
    unsigned int is_matrices_only_pressure_BC_printed = 0;
    unsigned int is_matrices_after_all_BC_printed = 0;
    
    unsigned int is_matrices_after_BC_stored = 0;
    unsigned int is_matrix_after_BC_stored = 1;
    unsigned int is_rhs_after_BC_stored = 1;       
    
    unsigned int is_UMFPACK_mono = 1;
    const unsigned int is_CG_M_inverse=0;
    unsigned int CG_iteration_schur = 0;
    
    unsigned int is_error_built_in = 1;
    unsigned int is_error_built_in_printed = 1;
    unsigned int is_error_custom=0;
    
    unsigned int is_solution_before_distributing_printed = 0;
    unsigned int is_solution_after_distributing_printed = 0;
    
    unsigned int is_results_outputted = 1;    
    
    
    real_t pressure_inte_L2_analytical, velocity_inte_L2_analytical, velocity_inte_H1_semi_analytical=1.0;
    real_t l2_error_abs_pressure, l2_error_abs_velocity, h1_semi_error_abs_velocity=0.0;
    
    double h1_semi_error_abs_velocity_00,h1_semi_error_abs_velocity_01;
    double h1_semi_error_abs_velocity_10,h1_semi_error_abs_velocity_11;
    
    
    real_t pressure_rela_error_l2, velocity_rela_error_l2, velocity_rela_error_h1_semi=0.0;
    
    real_t velocity_abs_error_hdiv_semi=0.0;
    real_t velocity_rela_error_hdiv_semi=0.0;
    
    double l2_error_abs_pressure_custom=0.0;    
    double l2_error_abs_velocity_custom=0.0;
    double h1_semi_error_abs_velocity_custom=0.0;
        
    
    unsigned int n_times_called_setup = 0;
    unsigned int n_times_called_assemble = 0;
    unsigned int n_times_called_solve = 0;
        
    real_t total_CPU_time;
    
};


template <int dim>
MixedLaplaceProblem_Real<dim>::MixedLaplaceProblem_Real (const unsigned int id_fe_system,
                                                         const unsigned int id_case, 
                                                         const double coeff_var_inner_x,
                                                         const int id_quad_assem_incre, 
                                                         const unsigned int id_coeff_diff_value, 
                                                         const double coeff_diff_inner,
                                                         const unsigned int id_loc_coeff_diff, 
                                                         const unsigned int id_coeff_helm_value,
                                                         const double coeff_helm_inner,
                                                         const double tol_prm_schur, 
                                                         const unsigned int degree, 
                                                         const unsigned int refine)
:
id_fe_system{id_fe_system},
id_case{id_case},
coeff_var_inner_x(coeff_var_inner_x),
id_quad_assem_incre(id_quad_assem_incre),
id_coeff_diff_value{id_coeff_diff_value},
coeff_diff_inner (coeff_diff_inner),
id_loc_coeff_diff(id_loc_coeff_diff),
id_coeff_helm_value(id_coeff_helm_value),
coeff_helm_inner(coeff_helm_inner),
tol_prm_schur (tol_prm_schur),
degree (degree),
refine (refine),

// swaping the elements based on the space dimension, synchronized with line ~37

#if 1

fe_velocity(degree),                  // first choice for RT, second choice for BDM
fe_pressure(degree),


/*fe_velocity(degree),
fe_pressure(degree-1), */   

fe (fe_velocity, 1,                   // 3 components but 2 blocks
    fe_pressure, 1),
    
#else                                // for FE_Q/FE_DGQ
    
fe_velocity(degree),
fe_pressure(degree-1),

fe (fe_velocity, dim,                             // 3 components and blocks
    fe_pressure, 1), 
    
#endif

dof_handler (triangulation),
dof_handler_first_refine (triangulation_first_refine),
dof_handler_pressure_first_refine (triangulation_first_refine),
dof_handler_velocity_first_refine (triangulation_first_refine),
computing_timer  (MPI_COMM_WORLD,cout, TimerOutput::never, TimerOutput::cpu_times)
{
  cout << "================== \n";
  
  cout << "  Space dimension: " << dim << "\n"
       << "  MixedLaplaceProblem_Real\n"
       << "  Elements: " << (id_fe_system==0?"FE_Q/FE_DGQ":(id_fe_system==1?"RT":"BDM")) << "\n";
            
  obj_string = "fe";
  print_fe_info (obj_string, fe);
  
//   obj_string = "fe_velocity";
//   print_fe_info (obj_string, fe_velocity);
//   obj_string = "fe_pressure";
//   print_fe_info (obj_string, fe_pressure);
  
  cout << "  # of quadrature points in one direction: " << n_q_points_in_one_direction << "\n";
                
  cout << "==================\n";
            
  if (id_case == 21 && id_loc_coeff_diff == 2)
  {
    cout << "  u=-dp/dx in diffusion equations" << endl;
  }
}

    
template <int dim>
void MixedLaplaceProblem_Real<dim>::make_grid_and_dofs ()
{    
//   print_geometry_info();
    
  cout << "Making grid " << endl;
  TimerOutput::Scope t(computing_timer, "make_grid");
    
  GridGenerator::hyper_cube (triangulation, 0, 1);
  
  if (dim==1)
  {
    if(is_neumann==0)
    {
      cout << "setting the rightest boundary as Dirichlet type" << endl;
      triangulation.begin_active()->face(1)->set_boundary_id(0);        
    }
  }else if(dim==2)
  {
    if (is_neumann==1)
    {
    //       adjust_boundary_id_2d(triangulation);
        
//       triangulation.begin_active()->face(1)->set_boundary_id(1);       // right       
      triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom
      triangulation.begin_active()->face(3)->set_boundary_id(1);       // top
    }
    
  }
  
  print_boundary_info(triangulation);
  
  triangulation.refine_global (refine);
  triangulation_first_refine.copy_triangulation(triangulation);

  print_tria_info (triangulation);
  
  if (is_dof_handler_first_refine_dealt == 1)
  {
    std::vector<unsigned int> block_component (dim+1,0);
    block_component[dim] = 1;

    dof_handler_first_refine.distribute_dofs (fe);
    dof_handler_first_refine.initialize_local_block_info();                       // local dofs denote dofs in a reference cell
    DoFRenumbering::component_wise (dof_handler_first_refine, block_component);


    dof_handler_pressure_first_refine.distribute_dofs (fe_pressure);
    DoFRenumbering::component_wise (dof_handler_pressure_first_refine);

    dof_handler_velocity_first_refine.distribute_dofs (fe_velocity);
    DoFRenumbering::component_wise (dof_handler_velocity_first_refine);


    obj_string = "dof_handler_first_refine";
    print_dofhandler_info(obj_string, dof_handler_first_refine);
    //   print_coords_of_dofs(obj_string, dof_handler_first_refine);

    obj_string = "dof_handler_pressure_first_refine";
    print_dofhandler_info(obj_string, dof_handler_pressure_first_refine);
    //   print_coords_of_dofs(obj_string, dof_handler_pressure_first_refine);
    //   save_coords_of_dofs(obj_string, dof_handler_pressure_first_refine);

    obj_string = "dof_handler_velocity_first_refine";
    print_dofhandler_info(obj_string, dof_handler_velocity_first_refine);
    //   print_coords_of_dofs(obj_string, dof_handler_velocity_first_refine);
    //   save_coords_of_dofs(obj_string, dof_handler_velocity_first_refine);

  }
}
  

  
template <int dim>
void MixedLaplaceProblem_Real<dim>::setup_system ()
{
  cout << "Setting up the system\n";
  
  if(is_neumann==1)
  {
    is_constraints=1;
  }
  
  dof_handler.distribute_dofs (fe);
  
  
  if(id_fe_system==0)
  {
    std::vector<unsigned int> block_component (dim+1,0);
    block_component[dim] = 1;
    
    DoFRenumbering::component_wise (dof_handler,block_component);
    
    cout << "  block_component: ";
    print_vector_horizontally(block_component);
    
    const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    n_u = dofs_per_block[0];
    n_p = dofs_per_block[1];    
    
  }else
  {
    DoFRenumbering::component_wise (dof_handler);
    
    const std::vector<types::global_dof_index> dofs_per_component = DoFTools::count_dofs_per_fe_component(dof_handler);
    
//     cout << "  dofs_per_component: \n";
//     print_vector(dofs_per_component);
    
    n_u = dofs_per_component[0];
    n_p = dofs_per_component[dim];
    
    n_u_per_cell_boundary = 4*(degree+1);
    
  }
  
//   DoFRenumbering::Cuthill_McKee (dof_handler);                  // the order of the last basis function of a cell is 1 in this way, see step-22 for purpose of this renumbering, July 27, 2020
  
  cout << "  n_u: " << n_u << "\n"
       << "  n_p: " << n_p << "\n";
  
  BlockDynamicSparsityPattern dsp(2, 2);
  dsp.block(0, 0).reinit (n_u, n_u);
  dsp.block(1, 0).reinit (n_p, n_u);
  dsp.block(0, 1).reinit (n_u, n_p);
  dsp.block(1, 1).reinit (n_p, n_p);
  dsp.collect_sizes ();

  DoFTools::make_sparsity_pattern (dof_handler, dsp);


  system_matrix.clear();
  system_matrix_no_essential_bc.clear();

  sparsity_pattern.copy_from(dsp);


  system_matrix.reinit (sparsity_pattern);
  system_matrix_no_essential_bc.reinit (sparsity_pattern);

  solution.reinit (2);
  solution.block(0).reinit (n_u);
  solution.block(1).reinit (n_p);
  solution.collect_sizes ();

  system_rhs.reinit (2);
  system_rhs.block(0).reinit (n_u);
  system_rhs.block(1).reinit (n_p);
  system_rhs.collect_sizes ();

  system_rhs_no_essential_bc.reinit (2);
  system_rhs_no_essential_bc.block(0).reinit (n_u);
  system_rhs_no_essential_bc.block(1).reinit (n_p);
  system_rhs_no_essential_bc.collect_sizes ();

  solution_0.reinit (2);
  solution_0.block(0).reinit (n_u);
  solution_0.block(1).reinit (n_p);
  solution_0.collect_sizes ();
  
  
  if (is_neumann==1)
  {
    cout << "  dealing with Neumann boundary conditions using constraints\n";
    {
      constraints.clear ();
      
      FEValuesExtractors::Vector velocity(0);                                       // also appears in the assembling process
      DoFTools::make_hanging_node_constraints (dof_handler, constraints);
             
      if(id_fe_system==0)
      {
        cout << "    using VectorTools::interpolate_boundary_values ()\n";
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  ExactSolution_Step_20_Real<dim>(id_case, coeff_var_inner_x, id_coeff_diff_value, id_loc_coeff_diff),
                                                  constraints,
                                                  fe.component_mask(velocity));       
      }else if(id_fe_system==1)
      {

        VectorTools::project_boundary_values_div_conforming_rt (dof_handler,
                                                                 0,                             // setting constraints from the first component of the velocity
                                                                 GradientBoundary_Step_20_Real<dim>(id_case, coeff_var_inner_x, id_coeff_diff_value, id_loc_coeff_diff), 
                                                                 1,                             // a class defined in exact_solution_step_20_real.h
                                                                 constraints
                                                                );
      }else if(id_fe_system==2)
      {
        VectorTools::project_boundary_values_div_conforming_bdm (dof_handler,
                                                                 0,
                                                                 GradientBoundary_Step_20_Real<dim>(id_case, coeff_var_inner_x, id_coeff_diff_value, id_loc_coeff_diff),
                                                                 1,
                                                                 constraints
                                                                );        
      }
    }
    constraints.close ();
    
//     obj_string = "constraints";
//     print_constraints_info(obj_string, constraints, dof_handler);

  }
  
  n_times_called_setup++;
}


template <int dim>
void MixedLaplaceProblem_Real<dim>::assemble_system ()
{
  cout << "Assembling\n";
  TimerOutput::Scope t(computing_timer, "assemble_system");
  
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    | update_gradients |
                           update_quadrature_points  | update_JxW_values | update_jacobians); 
   
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values    | update_normal_vectors |
                                    update_quadrature_points  | update_JxW_values | update_jacobians);

  const unsigned int   n_face_q_points = face_quadrature_formula.size();
  const unsigned int   dofs_per_cell=fe.dofs_per_cell;
  
//   cout << "  # of quadrature points on a face: " << n_face_q_points << "\n";
  

  vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  vector<types::global_dof_index> local_dof_indices_on_a_face(fe.n_dofs_per_face());
  
  FullMatrix<real_t>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<real_t>       local_rhs (dofs_per_cell);

  const RightHandSide_Real<dim>      obj_right_hand_side(id_case, coeff_var_inner_x, id_coeff_diff_value, coeff_diff_inner, id_coeff_helm_value, coeff_helm_inner);
  const ExactSolution_Step_4_Real<dim> pressure_boundary_values(id_case, coeff_var_inner_x);
  const GradientBoundary_Step_20_Real<dim> gradient_boundary_values(id_case, coeff_var_inner_x, id_coeff_diff_value, id_loc_coeff_diff);
  
  const Coeff_Diff_Real<dim>               obj_coeff_diff(id_case, id_coeff_diff_value, coeff_diff_inner);
  
  const Coeff_First_Derivative_A_Real<dim> obj_coeff_first_derivative_a(id_case, 1);                        // second argument unused
  
  const Coeff_Helm_Real<dim>               obj_coeff_helm(id_case, id_coeff_helm_value, coeff_helm_inner);
  
  
  vector<Point<dim>> coords_of_quadrature_on_a_cell (n_q_points, Point<dim>());
  vector<Point<dim>> coords_of_quadrature_on_a_face (n_face_q_points, Point<dim>());
  
  vector<double> JxW_of_quadrature_on_cell (n_q_points);
  vector<double> JxW_of_quadrature_on_face (n_face_q_points);
  
  const unsigned int spacedim = dim;
  DerivativeForm< 1, dim, spacedim > jacobian_of_one_quadrature_point;
  
  vector<Vector<double>> value_rhs (n_q_points, Vector<double>(n_number_components));
  vector<real_t> boundary_values_for_pressure (n_face_q_points);
  vector<Vector<real_t>> boundary_values_for_gradient (n_face_q_points, Vector<double>(2));
  
  vector<Tensor<2,dim>> coeff_diff_values (n_q_points, Tensor<2,dim>());
  vector<Tensor<2,dim>> coeff_diff_gradients_x_direction (n_q_points, Tensor<2,dim>());
  vector<Tensor<2,dim>> coeff_diff_gradients_y_direction (n_q_points, Tensor<2,dim>());
  vector<Tensor<2,dim>> coeff_diff_inverse (n_q_points, Tensor<2,dim>());
  
  vector<Tensor<1,dim>> value_coeff_first_derivative_a (n_q_points, Tensor<1,dim>());
  
  vector<double> coeff_helm_values (n_q_points);
  vector<Tensor<2,dim>> coeff_diff_face_values (n_face_q_points, Tensor<2,dim>());

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar velocity_from_second (1);                        // only extracting the second component of the velocity
  const FEValuesExtractors::Scalar pressure (dim);
  
  double contribution_diri_2_rhs_per_quad = 0;
  


  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  
  
  fe_values.reinit (cell);                                                          // for applying the following three functions
  jacobian_of_one_quadrature_point = fe_values.jacobian(0);                         // getting jacobian on the first quadrature point
  coords_of_quadrature_on_a_cell = fe_values.get_quadrature_points();
  JxW_of_quadrature_on_cell = fe_values.get_JxW_values();

  if(dim == 1 & is_l2_norm_coefficient_numerical == 1 && n_times_called_assemble == 0)
  {
    cout << "  @l2 norms of coefficient\n";
    cout << "    For the diffusion coefficient, ";
    compute_l2_norms_of_coefficient_numerically_core_tensor_2_dim_real(triangulation, dof_handler, fe_values, obj_coeff_diff);
  
//     cout << "    For the Helmholtz coefficient, ";
//     compute_l2_norms_of_coefficient_numerically_core(triangulation, dof_handler, fe_values, obj_coeff_helm, n_number_components);
  }        
    
  if(is_quadrature_info_printed_on_global_cell == 1)
  {
    cout << "  Info of jacobian of the first quadrature point of the first cell: \n";
    for (unsigned int j = 0; j < spacedim; ++j)
    {
    cout << "    " << jacobian_of_one_quadrature_point[j] << "\n";
    }
    cout << "    with determinant: " << jacobian_of_one_quadrature_point.determinant() << "\n";

    cout << "  coordinates of quadrature points of the first cell: \n";
    print_vector(coords_of_quadrature_on_a_cell);  

    cout << "  JxW of each quadrature of the first cell: \n";
    print_vector(JxW_of_quadrature_on_cell); 
        
  }
  
//   unsigned int dofs_per_face_velocity = fe_velocity.dofs_per_face;                              // one higher than the element degree
//   cout << "  dofs_per_face of velocity: " << dofs_per_face_velocity << "\n";
//     
//     ofstream fid_nodal_values_on_faces("data_nodal_values_on_faces.txt");
//     vector<double> nodal_values_per_face(dofs_per_face_velocity);
//     const unsigned int face_coordinate_direction[GeometryInfo<2>::faces_per_cell] = {1, 1, 0, 0};
//     double param_jacobian;
  
  for (; cell!=endc; ++cell)
  {
//     cout << "  ##############\n";
//     cout  << "  cell: " << cell->active_cell_index();
//     cout << ", with vertex global indices and coordinates \n";
//     for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
//     {
//       cout << "  " << cell->vertex_index(vertex) << " (" << cell->vertex(vertex) << ") \n";     
//     }
    
    
    cell->get_dof_indices (local_dof_indices);
/*    cout << "  with global dof indices: ";
    print_vector_horizontally(local_dof_indices);
      */  

    fe_values.reinit (cell);
    
    local_matrix = 0;
    local_rhs = 0;
    
    coords_of_quadrature_on_a_cell = fe_values.get_quadrature_points();
    JxW_of_quadrature_on_cell = fe_values.get_JxW_values();
 
//     cout << "  coords_of_quadrature_on_a_cell: \n";
//     print_vector(coords_of_quadrature_on_a_cell);
//     cout << "  JxW_of_quadrature_on_cell: ";
//     print_vector_horizontally(JxW_of_quadrature_on_cell); 
    
    obj_right_hand_side.value_rhs (coords_of_quadrature_on_a_cell, value_rhs);
    
    obj_coeff_diff.value (coords_of_quadrature_on_a_cell, coeff_diff_values);
    obj_coeff_diff.gradient_x_direction (coords_of_quadrature_on_a_cell, coeff_diff_gradients_x_direction);
    obj_coeff_diff.gradient_y_direction (coords_of_quadrature_on_a_cell, coeff_diff_gradients_y_direction);
    obj_coeff_diff.inverse (coords_of_quadrature_on_a_cell, coeff_diff_inverse);
    
    obj_coeff_first_derivative_a.value (fe_values.get_quadrature_points(), value_coeff_first_derivative_a);
    
    obj_coeff_helm.value (coords_of_quadrature_on_a_cell, coeff_helm_values);
    
    if(is_value_coeff_printed==1)
    {
      cout << "  coeff_diff_values:\n";
      print_vector(coeff_diff_values);
      cout << "  coeff_diff_gradients_x_direction:\n";
      print_vector(coeff_diff_gradients_x_direction);
      cout << "  coeff_diff_gradients_y_direction:\n";
      print_vector(coeff_diff_gradients_y_direction);
      cout << "  coeff_diff_inverse:\n";
      print_vector(coeff_diff_inverse);
//     
//     cout << "  coeff_diff_values[0][0]: \n";
//     cout << coeff_diff_values[0][0] << "\n";
//     cout << "  coeff_diff_values[0][1]: \n";
//     cout << coeff_diff_values[0][1] << "\n";
//     
//     cout << "  value_coeff_first_derivative_a: ";
//     print_vector_horizontally(value_coeff_first_derivative_a);
//     
//     
//     cout << "  coeff_helm_values:\n";
//     print_vector(coeff_helm_values);
      
    }
    
#if 0
    if(is_basis_related_stored==1)
    {
      ofstream fid_pressure;
      fid_pressure.open("dgq_fe_values_deg_"+to_string(degree)+"_refine_"+to_string(refine)+"_cell_"+to_string(cell->active_cell_index())+".txt",ios::trunc);
      cout << "values of the pressure\n";
      for (unsigned int i=degree+1; i<dofs_per_cell; ++i)
      {
        cout << "basis " << i << ": ";
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          fid_pressure << setprecision(16) << fe_values[pressure].value (i, q_index) << " ";
          cout << fe_values[pressure].value (i, q_index) << " ";                  // 
        }
        fid_pressure << "\n";
        cout << "\n";
      }
      fid_pressure.close();
        
      ofstream fid_velocity;
      fid_velocity.open("q_fe_values_deg_"+to_string(degree)+"_refine_"+to_string(refine)+"_cell_"+to_string(cell->active_cell_index())+".txt",ios::trunc);
      cout << "values of the velocity\n";
      for (unsigned int i=0; i<degree+1; ++i)
      {
        cout << "basis " << i << ": ";
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          fid_velocity << setprecision(16) << fe_values[velocities].value (i, q_index)[0] << " ";
          cout << fe_values[velocities].value (i, q_index)[0] << " ";
        }
        fid_velocity << "\n";
        cout << "\n";
      }
      fid_velocity.close();
        
        
      ofstream fid_divergence;
      fid_divergence.open("q_fe_gradients_deg_"+to_string(degree)+"_refine_"+to_string(refine)+"_cell_"+to_string(cell->active_cell_index())+".txt",ios::trunc);
      cout << "divergence of the velocity\n";
      for (unsigned int i=0; i<degree+1; ++i)
      {
        cout << "basis " << i << ": ";
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          fid_divergence << setprecision(16) << fe_values[velocities].divergence (i, q_index) << " ";
          cout << fe_values[velocities].divergence (i, q_index) << " ";
        }
        fid_divergence << "\n";
        cout << "\n";
      }    
      fid_divergence.close();
        
        
      ofstream fid_rhs;
      fid_rhs.open("rhs_values_deg_"+to_string(degree)+"_refine_"+to_string(refine)+"_cell_"+to_string(cell->active_cell_index())+".txt",ios::trunc);
      cout << "values of the rhs\n";
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
        fid_rhs << setprecision(16) << value_rhs[q_index] << " ";
        cout << value_rhs[q_index] << " ";
      }
      fid_rhs << "\n";
      cout << "\n";
      fid_rhs.close();
    }
#endif
    
    for (unsigned int q=0; q < /*1*/ n_q_points; ++q)
    {
//       cout << "at quadrature point " << coords_of_quadrature_on_a_cell[q] << "\n";
        
//       cout << "coeff_diff_inverse manually:\n";
//       cout << 1.0/(coeff_diff_values[q][0][0]) << "\n";
        
      for (unsigned int i=0; i< /*1*/ dofs_per_cell; ++i)
      {
          
        const Tensor<1,dim> phi_i_u     = fe_values[velocities].value (i, q);                                   // invoking functions in fe_values.h
        const Tensor<2,dim> grad_phi_i_u     = fe_values[velocities].gradient (i, q);
        
//         const double phi_i_u_from_second     = fe_values[velocity_from_second].value (i, q);
        const real_t        div_phi_i_u = fe_values[velocities].divergence (i, q);
        const real_t        phi_i_p     = fe_values[pressure].value (i, q);
        
//         cout << "grad_phi_i_u: " << grad_phi_i_u << "\n";
        
//         if(i<n_u)                                                  // working on the velocity component
//         {
//           cout << "RT, ";
//           cout << "i: " << i << ", global dof index: " << local_dof_indices[i] << ", ";
//           cout << "phi_i_u: " << phi_i_u << ", ";
//           cout << "div_phi_i_u: " << div_phi_i_u << "\n";
//         }
//         
//         if(i>=n_u)
//         {
//           cout << "DGQ, ";
//           cout << "i: " << i << ", global dof index: " << local_dof_indices[i] << ", ";
//           cout << "phi_i_p: " << phi_i_p << "\n";
//         }
        
        
        for (unsigned int j=0; j< /*1*/ dofs_per_cell; ++j)                                                   // every j denotes one form of basis function of u
        {
          const Tensor<1,dim> phi_j_u     = fe_values[velocities].value (j, q);
          const Tensor<2,dim> grad_phi_j_u     = fe_values[velocities].gradient (j, q);
          const real_t        div_phi_j_u = fe_values[velocities].divergence (j, q);
          const real_t        phi_j_p     = fe_values[pressure].value (j, q);
            
          if (id_loc_coeff_diff==0)                                                                    // 1/d(x)v = - \nabla u
          {                         
            local_matrix(i,j) += (phi_i_u * (coeff_diff_inverse[q] * phi_j_u)                  
                                  - div_phi_i_u * phi_j_p
                                  - phi_i_p * (div_phi_j_u
                                               + value_coeff_first_derivative_a[q] * (-coeff_diff_inverse[q] * phi_j_u)
                                               + coeff_helm_values[q]*phi_j_p))
                                  * JxW_of_quadrature_on_cell[q];
                                  
          } else if (id_loc_coeff_diff==1)                                                             // v = - d \nabla u
          {
            local_matrix(i,j) += (phi_i_u * phi_j_u                        
                                  - ((coeff_diff_gradients_x_direction[q][0]+coeff_diff_gradients_y_direction[q][1]) * phi_i_u + trace(coeff_diff_values[q] * grad_phi_i_u)) * phi_j_p                                     
                                  - phi_i_p * (div_phi_j_u
                                               + value_coeff_first_derivative_a[q] * (-coeff_diff_inverse[q] * phi_j_u)
                                               + coeff_helm_values[q]*phi_j_p))
                                * JxW_of_quadrature_on_cell[q];
                
          } else if (id_loc_coeff_diff==2)                                                              // v = - \nabla u
          {
            local_matrix(i,j) += (phi_i_u * phi_j_u                       
                                  - div_phi_i_u * phi_j_p      
                                  - phi_i_p * ((coeff_diff_gradients_x_direction[q][0]+coeff_diff_gradients_y_direction[q][1]) * phi_j_u + trace(coeff_diff_values[q] * grad_phi_j_u)
                                               + value_coeff_first_derivative_a[q] * (-phi_j_u)
                                               + coeff_helm_values[q]*phi_j_p))
                                * JxW_of_quadrature_on_cell[q];                          
          }
        }
        local_rhs(i) += -phi_i_p *
                        value_rhs[q][0] *
                        JxW_of_quadrature_on_cell[q];
      }
    }
    


    for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)                     // implementing Dirichlet boundary conditions
    {
      if (cell->at_boundary(face_n))
      {
          
        fe_face_values.reinit (cell, face_n);
          
        cell->face(face_n)->get_dof_indices(local_dof_indices_on_a_face);
        
        coords_of_quadrature_on_a_face = fe_face_values.get_quadrature_points();
        JxW_of_quadrature_on_face = fe_face_values.get_JxW_values();
        
//         cout << "coords_of_quadrature_on_a_face: \n";
//         print_vector(coords_of_quadrature_on_a_face);
//         
//         cout << "local_dof_indices_on_a_face: \n";
//         print_vector(local_dof_indices_on_a_face);
        
        
//         unsigned int index_dof_on_face_of_reference_cell;
        Tensor<1,dim> phi_i_u_on_a_boundary;
        Tensor<1,dim> term_velocity_boundary;

        
        if (cell->face(face_n)->boundary_id() == 0)
        {
//           cout << "  Dirichlet BCs are imposed\n";
            
          pressure_boundary_values.value_list (coords_of_quadrature_on_a_face, boundary_values_for_pressure);
          obj_coeff_diff.value (coords_of_quadrature_on_a_face, coeff_diff_face_values);
                
//           cout << "  coeff_diff_face_values:\n";
//           print_vector(coeff_diff_face_values);
          
          for (unsigned int q=0; q< /*1*/ n_face_q_points; ++q)
          {
            
//             cout << "at quadrature point " << coords_of_quadrature_on_a_face[q] << "\n";
          
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              phi_i_u_on_a_boundary = fe_face_values[velocities].value (i, q);
              
              if(id_loc_coeff_diff==0 or id_loc_coeff_diff == 2)
              {
                term_velocity_boundary = phi_i_u_on_a_boundary;
              }else if(id_loc_coeff_diff==1)
              {
                term_velocity_boundary = coeff_diff_face_values[q] * phi_i_u_on_a_boundary;
              }
              
              contribution_diri_2_rhs_per_quad = -(term_velocity_boundary *
                                                   fe_face_values.normal_vector(q) *
                                                   boundary_values_for_pressure[q])
                                                  *JxW_of_quadrature_on_face[q];
              local_rhs(i) += contribution_diri_2_rhs_per_quad;              
              
              
//               if(i<n_u_per_cell_boundary)
//               {
//                 cout << "i: " << i << ", ";                             //", global dof index: " << local_dof_indices_on_a_face[i] << ", ";
//                 cout << "phi_i_u_on_a_boundary: " << phi_i_u_on_a_boundary << "\n";  
//               }
              
//               if (id_loc_coeff_diff==0 or id_loc_coeff_diff == 2)
//               {
//                 contribution_diri_2_rhs_per_quad = -(phi_i_u_on_a_boundary *
//                                                      fe_face_values.normal_vector(q) *
//                                                      boundary_values_for_pressure[q])
//                                                     *JxW_of_quadrature_on_face[q];
//                 local_rhs(i) += contribution_diri_2_rhs_per_quad;
//             
//               } else if (id_loc_coeff_diff==1)
//               {
//                 contribution_diri_2_rhs_per_quad = -(coeff_diff_face_values[q] * phi_i_u_on_a_boundary *
//                                                      fe_face_values.normal_vector(q) *
//                                                      boundary_values_for_pressure[q] *
//                                                      JxW_of_quadrature_on_face[q]); 
//                 local_rhs(i) += contribution_diri_2_rhs_per_quad;
//               }
            }
          }
        }else 
        {
          
#if 0
            cout << "  ~~~~~~~~~~\n";
            cout << "  on face " << face_n << " of cell " << cell->active_cell_index() << "\n";
            cout << "  with global index " << cell->face_index(face_n) << "\n";
            cout << "  with vertices ";
            for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
            {
              cout << "(" << cell->face(face_n)->vertex(vertex) << ")";
            }
            cout << "\n";   
            cout << "  with global indices of dofs: ";
            print_vector_horizontally(local_dof_indices_on_a_face);        

            cout << "  with quadrature points: \n";
            print_vector(coords_of_quadrature_on_a_face);
            
            
            cout << "  with JxW: \n";
            print_vector(JxW_of_quadrature_on_face);
            
          cout << "  Neumann BCs are imposed\n";
            
          gradient_boundary_values.vector_value_list(coords_of_quadrature_on_a_face, boundary_values_for_gradient);
          cout << "  boundary values for gradient: \n";
          print_vector(boundary_values_for_gradient);
          
          cout << "  ++++++++++\n";                                         // an alternative for obtaining the values of constaints
          
          fe_velocity.convert_generalized_support_point_values_to_nodal_values_for_face(face_n, boundary_values_for_gradient, nodal_values_per_face);
          
          param_jacobian = sqrt(jacobian_of_one_quadrature_point[0][face_coordinate_direction[face_n]]
                                *jacobian_of_one_quadrature_point[0][face_coordinate_direction[face_n]]
                                +jacobian_of_one_quadrature_point[1][face_coordinate_direction[face_n]]
                                *jacobian_of_one_quadrature_point[1][face_coordinate_direction[face_n]]);
          
          cout << "  param_jacobian: " << param_jacobian << "\n";
          
          std::transform(nodal_values_per_face.begin(), nodal_values_per_face.end(), nodal_values_per_face.begin(),
               std::bind1st(std::multiplies<real_t>(), param_jacobian));
          
          cout << "  nodal_values_per_face (" << dofs_per_face_velocity << " in total): \n";
          print_vector(nodal_values_per_face);
          
          for (unsigned int i=0;i<nodal_values_per_face.size();i++)
          {
            fid_nodal_values_on_faces << "[" << i << "] " << local_dof_indices_on_a_face[i] << " " << nodal_values_per_face[i] << " ";
            fid_nodal_values_on_faces << "\n";
          }
          
#endif
          
        }
      }
    }
    
    if (is_constraints==0)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          system_matrix.add (local_dof_indices[i],                                                // local_dof_indices initiated at the beginning
                             local_dof_indices[j],
                             local_matrix(i,j));
        }
      
        system_rhs(local_dof_indices[i]) += local_rhs(i);
      }
      
    }else if (is_constraints==1)
    {
      constraints.distribute_local_to_global (local_matrix, local_rhs,
                                              local_dof_indices,
                                              system_matrix, system_rhs);
      
      if (is_matrices_assembled_no_essential_bc == 1)
      {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            system_matrix_no_essential_bc.add (local_dof_indices[i],
                                               local_dof_indices[j],
                                               local_matrix(i,j));
          }
          system_rhs_no_essential_bc(local_dof_indices[i]) += local_rhs(i);
        }  
      }
      
    }
  }
  
  
/*  fid_nodal_values_on_faces.close();
  fid_nodal_values_on_faces.clear();   */
  
  
  if (is_constraints ==1 && is_matrices_only_pressure_BC_printed == 1)
  {
    cout << "\n";
    cout << "system_matrix only applying pressure BC is\n";
    system_matrix_no_essential_bc.print_formatted(cout);
    cout << "system_rhs only applying pressure BC is\n";
    system_rhs_no_essential_bc.print(cout);
  }
  
  if (is_matrices_after_all_BC_printed==1)
  {
    cout << "\n";
    cout << "system_matrix after applying all BCs is\n";
    system_matrix.print_formatted(cout);
    cout << "system_rhs after applying all BCs is\n";
    system_rhs.print(cout);
  }

  if (is_matrices_after_BC_stored==1)
  {
    ostringstream streamObj;
    streamObj << fixed;
    streamObj << setprecision(0);
    streamObj << coeff_diff_inner;   
    
    cout << "\n";
    if (is_matrix_after_BC_stored==1)
    {
      cout << "  storing system_matrix" << endl;
        
      ofstream output("system_matrix_coeff_"+ streamObj.str() +"_deg_"+to_string(degree)+"_ref_"+to_string(refine)+".txt");               // +"_cycle_"+to_string(cycle_global)
      for (unsigned int i=0;i<dof_handler.n_dofs();i++) 
      {
        for (unsigned int j=0;j<dof_handler.n_dofs();j++)
        {
          output << system_matrix.el(i, j) << " ";                 // behaves like cout - cout is also a stream
        }
        output << "\n";
      } 
      output.close();
      output.clear();
    }

    if(is_rhs_after_BC_stored==1)
    {
      cout << "  storing system_rhs" << endl;
    
      ofstream output_rhs("system_rhs_coeff_"+ streamObj.str() +"_deg_"+to_string(degree)+"_ref_"+to_string(refine)+".txt"); 
      for (unsigned int i=0;i<dof_handler.n_dofs();i++)
      {
        output_rhs << system_rhs[i] << " ";
        output_rhs << "\n";
      }
      output_rhs.close();
      output_rhs.clear();
    } 
    cout << "\n";
  }
  
  n_times_called_assemble++;
}


template <int dim>
void MixedLaplaceProblem_Real<dim>::solve ()
{
  cout << "Solving\n";
  TimerOutput::Scope t(computing_timer, "solve");
  
//     for (unsigned int i=0;i<dof_handler.n_dofs();i++)                                // converting LHS and RHS to float precision
//     {
//         system_rhs[i] = float(system_rhs[i]);                       
//         
//         for (unsigned int j=0;j<dof_handler.n_dofs();j++)
//         {
//             system_matrix.set(i,j, float(system_matrix.el(i,j)));   
//         }
//     }
    
  if (tol_prm_schur != 0)
  {
    is_UMFPACK_mono = 0;
  }    
    
  if (is_UMFPACK_mono==1)
  {
    cout << "  UMFPack solver for monolithic\n";
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);

  }else if(is_UMFPACK_mono==0)
  {
    cout << "  schur complement approach\n"
              << "  tol_prm_schur: " << tol_prm_schur << endl;
    
    InverseMatrix_UMF<SparseMatrix<real_t> > inverse_mass (system_matrix.block(0,0));                       // _CG, an alternative
    Vector<real_t> tmp (solution.block(0).size());
    {
      Vector<real_t> schur_rhs (solution.block(1).size());
      inverse_mass.vmult (tmp, system_rhs.block(0));
      system_matrix.block(1,0).vmult (schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);
        
      cout << "  computing solution\n";            

      SchurComplement<InverseMatrix_UMF<SparseMatrix<real_t> >> schur_complement (system_matrix, inverse_mass);   // this passes system_matrix and inverse_mass to the private variables of SchurComplement
                    
      double l2_norm_schur_rhs = schur_rhs.l2_norm();
      cout << "  l2_norm_schur_rhs: " << l2_norm_schur_rhs << endl;
      
      SolverControl solver_control (solution.block(1).size()*10,                      
                                    tol_prm_schur*l2_norm_schur_rhs);    
      
            // 1. by varying tol_prm_schur while using InverseMatrix_UMF for the inverse_mass, we investigate the influence of the CG solver when the lhs is the schur complement
            // 2. by setting tol_prm_schur=1e-16, using InverseMatrix_CG with different tolerance, we investigate the influence of the CG solver when the lhs is M
        
      SolverCG<> cg (solver_control);                // For the mixed FEM, we only use SolverCG1 to deal with the Schur complement

        
//             cout << "  <initialize approximate_inverse>";
//             ApproximateSchurComplement approximate_schur (system_matrix);                               // we create an object of ApproximateSchurComplement that takes system_matrix
//             InverseMatrix_CG<ApproximateSchurComplement> approximate_inverse (approximate_schur);          // ApproximateSchurComplement is also a type name of InverseMatrix
            
            
/*            cout << "  before solving the gradient, the solution vector reads " << endl;
            solution.print(cout);    */        
            
//             cout << endl;
//             cout << "   <CG solver for the schur_complement> ";
            
            
      cg.solve (schur_complement, solution.block(1), schur_rhs,                       // third use of the CG solver, first for the Schur complement
                PreconditionIdentity());                                                                  // approximate_inverse

//             cout << "--> done" << endl;
        
      CG_iteration_schur = solver_control.last_step();
        
      cout << "  "
                << solver_control.last_step()
                << " CG iterations needed."
                << endl;
                    
    }

    cout << "  computing gradient";        
    {
      system_matrix.block(0,1).vmult (tmp, solution.block(1));                // tmp = B*U
      tmp *= -1;                                                              // tmp = -B*U
      tmp += system_rhs.block(0);                                             // tmp = G-B*U
        
//             cout << "  <G-BU>, ";
      inverse_mass.vmult (solution.block(0), tmp);                            // obtain V         
    }
        
//         cout << "--> done" << endl;
        
    cout << endl;     
  }
    
  if (is_solution_before_distributing_printed == 1)
  {    
    cout << "solution before distributing: \n";
    solution.print(cout);
    cout << endl;
  }

  if (is_constraints==1)
  {
    constraints.distribute (solution);
  }
  
  solution_velocity = solution.block(0);
  solution_pressure = solution.block(1);
        
  if (is_solution_after_distributing_printed == 1)
  {
    cout << "solution after distributing: \n";
    solution.print(cout);
    cout << endl;
  }

//         cout << "  converting solution to float precision" << endl;
//         for (unsigned int i=0;i<dof_handler.n_dofs();i++)                 
//         {
//             solution[i]=(float)solution[i];
//         }

  n_times_called_solve++;
}


template <int dim>
void MixedLaplaceProblem_Real<dim>::compute_errors_and_l2_norms_built_in ()
{
  cout << "Computing errors and l2 norms using the built-in function\n";
  TimerOutput::Scope t(computing_timer, "compute_errors_and_l2_norms_built_in");
   
  const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);                                      // select the dim th component for computing the error
  const ComponentSelectFunction<dim> velocity_mask (std::make_pair(0, dim), dim+1);                   // select the first and second components by std::make_pair(0, dim)

  ExactSolution_Step_20_Real<dim> exact_solution(id_case, coeff_var_inner_x, id_coeff_diff_value, id_loc_coeff_diff);
  Vector<double> cellwise_errors (triangulation.n_active_cells());
  
  QTrapez<1>     q_trapez;
  QIterated<dim> quadrature (q_trapez, degree+2);
  
#if 1
    
  VectorTools::integrate_difference (dof_handler, solution_0, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
  pressure_inte_L2_analytical = VectorTools::compute_global_error(triangulation,
                                           cellwise_errors,
                                           VectorTools::L2_norm);

  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);    
  l2_error_abs_pressure = VectorTools::compute_global_error(triangulation,
                                               cellwise_errors,
                                               VectorTools::L2_norm);
  
  pressure_rela_error_l2 = l2_error_abs_pressure/pressure_inte_L2_analytical;
  
  
    
  VectorTools::integrate_difference (dof_handler, solution_0, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
  velocity_inte_L2_analytical = VectorTools::compute_global_error(triangulation,
                                           cellwise_errors,
                                           VectorTools::L2_norm);

  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
  l2_error_abs_velocity = VectorTools::compute_global_error(triangulation,
                                               cellwise_errors,
                                               VectorTools::L2_norm);    
  
  velocity_rela_error_l2 = l2_error_abs_velocity/velocity_inte_L2_analytical;
  
    

  VectorTools::integrate_difference (dof_handler, solution_0, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
  velocity_inte_H1_semi_analytical = VectorTools::compute_global_error(triangulation,
                                           cellwise_errors,
                                           VectorTools::H1_seminorm);

  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
  h1_semi_error_abs_velocity = VectorTools::compute_global_error(triangulation,
                                               cellwise_errors,
                                               VectorTools::H1_seminorm);
  
  velocity_rela_error_h1_semi = h1_semi_error_abs_velocity/velocity_inte_H1_semi_analytical;
  
  
  if(dim == 2)
  {
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::H1_seminorm_velocity_00,
                                       &velocity_mask);
    h1_semi_error_abs_velocity_00 = VectorTools::compute_global_error(triangulation,
                                                                      cellwise_errors,
                                                                      VectorTools::H1_seminorm_velocity_00);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::H1_seminorm_velocity_01,
                                        &velocity_mask);
    h1_semi_error_abs_velocity_01 = VectorTools::compute_global_error(triangulation,
                                                cellwise_errors,
                                                VectorTools::H1_seminorm_velocity_01);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::H1_seminorm_velocity_10,
                                        &velocity_mask);
    h1_semi_error_abs_velocity_10 = VectorTools::compute_global_error(triangulation,
                                                cellwise_errors,
                                                VectorTools::H1_seminorm_velocity_10);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::H1_seminorm_velocity_11,
                                        &velocity_mask);
    h1_semi_error_abs_velocity_11 = VectorTools::compute_global_error(triangulation,
                                                cellwise_errors,
                                                VectorTools::H1_seminorm_velocity_11);
  }
  
  
  
  
  
#endif
  
  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::Hdiv_seminorm,
                                     &velocity_mask);
  velocity_abs_error_hdiv_semi = VectorTools::compute_global_error(triangulation,
                                               cellwise_errors,
                                               VectorTools::Hdiv_seminorm);
  
  if (is_error_built_in_printed==1)
  {

    cout << "  @errors\n";
    cout << "  l2_error_abs_pressure: " << l2_error_abs_pressure << "\n";
    cout << "  l2_error_abs_velocity: " << l2_error_abs_velocity << "\n";
    cout << "  velocity_abs_error_hdiv_semi: " << velocity_abs_error_hdiv_semi << "\n";
    
    cout << "  h1_semi_error_abs_velocity: " << h1_semi_error_abs_velocity << "\n";
    if(dim == 2)
    {
      cout << "    00: " << h1_semi_error_abs_velocity_00 << "\n";
      cout << "    01: " << h1_semi_error_abs_velocity_01 << "\n";
      cout << "    10: " << h1_semi_error_abs_velocity_10 << "\n";
      cout << "    11: " << h1_semi_error_abs_velocity_11 << "\n";
    }
        
    
    cout << "  @l2 norms\n";
    cout << "  pressure_inte_L2_analytical: " << pressure_inte_L2_analytical << "\n";
    cout << "  velocity_inte_L2_analytical: " << velocity_inte_L2_analytical << "\n";
    cout << "  velocity_inte_H1_semi_analytical: " << velocity_inte_H1_semi_analytical << "\n";
    
//     cout << "  pressure_rela_error_l2: " << pressure_rela_error_l2 << "\n";
//     cout << "  velocity_rela_error_l2: " << velocity_rela_error_l2 << "\n";
//     cout << "  velocity_rela_error_h1_semi: " << velocity_rela_error_h1_semi << "\n";
    
  }  
}


template <int dim>
void MixedLaplaceProblem_Real<dim>::compute_errors_custom ()                      
{
  cout << "Computing errors customly\n";

  solution_first_refine.reinit (dof_handler.n_dofs());
  solution_first_refine=solution;
  
  cout << "\n//////////////////\n";
  triangulation.refine_global ();                 
  setup_system ();
  assemble_system ();
  solve ();
  
  vector<double> vector_error(3);
  vector<Vector<double>> cellwise_error(3, Vector<double>(dof_handler_first_refine.get_triangulation().n_active_cells()));
  
  error_computation_custom_real_mix_core(solution_first_refine, solution, dof_handler_first_refine, dof_handler, cellwise_error);
    
  l2_error_abs_pressure_custom = cellwise_error[0].l2_norm();
  l2_error_abs_velocity_custom = cellwise_error[1].l2_norm();
  h1_semi_error_abs_velocity_custom = cellwise_error[2].l2_norm(); 
  
  cout << "  @results for custom functions\n";
  cout << "  l2_error_abs_pressure_custom: " << l2_error_abs_pressure_custom << endl;  
  cout << "  l2_error_abs_velocity_custom: " << l2_error_abs_velocity_custom << endl;  
  cout << "  h1_semi_error_abs_velocity_custom: " << h1_semi_error_abs_velocity_custom << endl;    
  
  cout << "//////////////////\n\n";

}


template <int dim>
void MixedLaplaceProblem_Real<dim>::output_results () const
{
  cout << "Outputting results\n";
  
#if 1                                                                                   // output the gradients of u separately
  
  std::vector<std::string> solution_names(1,"u_direction_1");
  std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation (1,DataComponentInterpretation::component_is_scalar);
  
  if (dim==2)
  {
    solution_names.push_back ("u_direction_2");
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);  
  }
    
#else                                                                                   // output the magnitude of u
  
  vector<string> solution_names(dim, "u");
  vector<DataComponentInterpretation::DataComponentInterpretation> interpretation (dim, DataComponentInterpretation::component_is_part_of_vector);

#endif
  
  solution_names.push_back ("p");
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);

  
  DataOut<dim> data_out;
  data_out.add_data_vector (dof_handler, solution, solution_names, interpretation);

  data_out.build_patches (degree+1);

  ofstream output ("solution-real-mm.vtk");
  data_out.write_vtk (output);
}


template <int dim>
void MixedLaplaceProblem_Real<dim>::printToFile()
{
  ofstream myfile;
    
  myfile.open ("data_error_mm_real.txt", ofstream::app);
  
//   myfile << "degree: " << degree << "\n";
  myfile << refine << " ";
  myfile << triangulation.n_vertices() << " ";
  myfile << dof_handler.n_dofs() << " ";
  myfile << n_u << " ";      // 
  myfile << n_p << " ";
  myfile << l2_error_abs_pressure << " ";
  myfile << l2_error_abs_velocity << " ";
  myfile << velocity_abs_error_hdiv_semi << " ";
  
  if (dim==2)
  {
    myfile << h1_semi_error_abs_velocity_00 << " ";
    myfile << h1_semi_error_abs_velocity_01 << " ";
    myfile << h1_semi_error_abs_velocity_10 << " ";
    myfile << h1_semi_error_abs_velocity_11 << " ";
  }
  myfile << h1_semi_error_abs_velocity << " ";
  
  myfile << pressure_inte_L2_analytical << " ";
  myfile << velocity_inte_L2_analytical << " ";
  myfile << total_CPU_time << " ";
  if (is_UMFPACK_mono == 1)
  {
    myfile << "UMF" << "\n";
  }else if (is_UMFPACK_mono == 0)
  {
    myfile << CG_iteration_schur << "\n";
  }
  myfile.close();
}


template <int dim>
void MixedLaplaceProblem_Real<dim>::run ()
{
  make_grid_and_dofs();
  
  
  setup_system();
  
  assemble_system ();
  
#if 1
  solve ();
  
//   obj_string="solution_pressure";
//   save_Vector_to_txt(obj_string,solution_pressure);
//   obj_string="solution_velocity";
//   save_Vector_to_txt(obj_string,solution_velocity);
  
  if (is_error_built_in==1)
  {
    compute_errors_and_l2_norms_built_in ();  
  }
  
  if (is_error_custom==1)
  {
    compute_errors_custom (); 
  }
  
  if (is_results_outputted==1)
  {
    output_results ();  
  }
  
//   computing_timer.print_summary();
  
  total_CPU_time = computing_timer.return_total_cpu_time ();
  
  printToFile();
  
#endif
  
  cout << "Summary \n";
  cout << "  # of setup_system() called: " << n_times_called_setup << "\n";
  cout << "  # of assemble_system() called: " << n_times_called_assemble << "\n";
  cout << "  # of solve() called: " << n_times_called_solve << "\n";
  cout << "  total_CPU_time: " << total_CPU_time << "\n";  
}

#endif
