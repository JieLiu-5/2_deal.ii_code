
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/lac/constraint_matrix_custom_function.h>


#include"../../6_headers/1_cpp/1_dealii/common_class_complex.h"
#include<auxiliaryclass_step_20.h>
#include<2_complex_valued/auxiliaryclass_step_20_complex.h>

// #include"../../6_headers/1_cpp/1_dealii/function_for_custom_l2_norm_mix.h"               // not successful, Oct. 26, 2020

template <int dim>
class MixedLaplaceProblem_Complex
{
  public:
    MixedLaplaceProblem_Complex (const unsigned int id_case, const unsigned int n_quad_incre_assem, const unsigned int id_coeff_diff_value, const unsigned int id_coeff_helm_value, const double length, const unsigned int degree, const unsigned int refine);
    void run ();

  private:
    void make_grid_and_dofs ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void store_solution ();
    
    void compute_l2_norms_of_variable_using_numerical_solution();
    void compute_errors_and_l2_norms_built_in ();
    void compute_errors_custom ();
    void output_results () const;

    const unsigned int   id_case;
    const unsigned int   n_quad_incre_assem;
    const unsigned int   id_coeff_diff_value;
    const unsigned int   id_coeff_helm_value;
    const double length;
    const unsigned int   degree;
    const unsigned int   refine;
    const double tol_prm_schur = 1e-16;
    unsigned int CG_iteration_schur = 0;
    
    unsigned int n_number_components = 2;
    
    unsigned int is_constraints_used = 1;
    unsigned int is_constraints_printed = 1;
    
    unsigned int n_times_called_setup = 0;
    unsigned int n_times_called_assemble = 0;
    unsigned int n_times_called_solve = 0;
    
    
    Triangulation<dim>   triangulation;
    Triangulation<dim>   triangulation_first_refine;
    
    
    unsigned int id_fe_system = 1;                   // '0' for Q/DGQ, '1' for either RT/DGQ or BDM/DGP,       
                                                     // also making changes in 
                                                     //                        1. the declaration of fe_velocity below
                                                     //                        2. constructor for fe
                                                     //
                                                     // affecting operation on the Neumann boundary condition
   
#if 1
    
    FE_RaviartThomas<dim>        fe_velocity;
    FE_DGQ<dim>                  fe_pressure;
    
//     FE_BDM<dim>                  fe_velocity;
//     FE_DGP<dim>                  fe_pressure;
    
#else
    
    FE_Q<dim>        fe_velocity;
    FE_DGQ<dim>                  fe_pressure;

#endif
    
    
    FESystem<dim>        fe;
    FESystem<dim>        fe_single;   
    
    DoFHandler<dim>      dof_handler;
    
    DoFHandler<dim>      dof_handler_single;
    DoFHandler<dim>      dof_handler_single_first_refine; 
    
    DoFHandler<dim>      dof_handler_velocity_single_first_refine;
    DoFHandler<dim>      dof_handler_pressure_single_first_refine;
    
    unsigned int n_q_points_one_direction; 
    unsigned int n_q_points; 
    QGauss<dim>  quadrature_formula;
    
    string obj_string="";
    
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double>       system_rhs;
    
    unsigned int is_matrices_before_neum_BC_printed = 0;                // only for not using constraints
    unsigned int is_matrices_after_neum_BC_printed = 0;
    
    unsigned int is_matrices_after_BC_stored = 0;
    unsigned int is_matrix_after_BC_stored = 1;
    unsigned int is_rhs_after_BC_stored = 1;      
    
    unsigned int is_UMFPACK_mono = 1;

    BlockVector<double>       solution;  
    
    Vector<double>            solution_velocity_real;
    Vector<double>            solution_velocity_imag;
    Vector<double>            solution_pressure_real;
    Vector<double>            solution_pressure_imag;  
    Vector<double>            solution_amp;
    
    BlockVector<double>            solution_real;
    BlockVector<double>            solution_imag; 
    
    BlockVector<double>            solution_real_first_refine;
    BlockVector<double>            solution_imag_first_refine;  
    
    BlockVector<double>       solution_0;
    
    unsigned int is_solution_after_distributing_printed = 0;
    unsigned int is_solution_stored = 0;
    
    unsigned int is_results_outputted=1;
    
    unsigned int is_l2_norm_coefficient_numerical = 1;
    unsigned int is_l2_norm_variable_numerical = 1;
    unsigned int is_error_and_l2_norm_built_in = 1;
    unsigned int is_error_custom = 0;
    
    double pressure_abs_error_L2, velocity_abs_error_L2, velocity_abs_error_H1_semi;
    double pressure_abs_error_L2_real, velocity_abs_error_L2_real, velocity_abs_error_H1_semi_real;
    double pressure_abs_error_L2_imag, velocity_abs_error_L2_imag, velocity_abs_error_H1_semi_imag;
    
    double pressure_inte_L2_analytical, velocity_inte_L2_analytical, velocity_inte_H1_semi_analytical;
    double pressure_inte_L2_analytical_real, velocity_inte_L2_analytical_real, velocity_inte_H1_semi_analytical_real;
    double pressure_inte_L2_analytical_imag, velocity_inte_L2_analytical_imag, velocity_inte_H1_semi_analytical_imag;
    
    double pressure_inte_L2_numerical, velocity_inte_L2_numerical, velocity_inte_H1_semi_numerical;
    double pressure_inte_L2_numerical_real, velocity_inte_L2_numerical_real, velocity_inte_H1_semi_numerical_real;
    double pressure_inte_L2_numerical_imag, velocity_inte_L2_numerical_imag, velocity_inte_H1_semi_numerical_imag;


    double pressure_rel_error_L2, velocity_rel_error_L2, velocity_rel_error_H1_semi;
    
    double pressure_abs_error_L2_custom, velocity_abs_error_L2_custom, velocity_abs_error_H1_semi_custom;
    double pressure_abs_error_L2_custom_real, velocity_abs_error_L2_custom_real, velocity_abs_error_H1_semi_custom_real;
    double pressure_abs_error_L2_custom_imag, velocity_abs_error_L2_custom_imag, velocity_abs_error_H1_semi_custom_imag;
    
    
    TimerOutput                               computing_timer;
    void print_errors_to_file();
    void print_l2_norms_to_file();
    double total_CPU_time;

    AffineConstraints<real_t> constraints;
    AffineConstraints<real_t> constraints_auxiliary;
};

  
template <int dim>
MixedLaplaceProblem_Complex<dim>::MixedLaplaceProblem_Complex (const unsigned int id_case, const unsigned int n_quad_incre_assem, const unsigned int id_coeff_diff_value, const unsigned int id_coeff_helm_value, const double length, const unsigned int degree, const unsigned int refine)
:
id_case (id_case),
n_quad_incre_assem (n_quad_incre_assem),
id_coeff_diff_value (id_coeff_diff_value),
id_coeff_helm_value (id_coeff_helm_value),
length(length),
degree (degree),
refine(refine),

#if 1

fe_velocity(degree),
fe_pressure(degree),            

/*fe_velocity(degree),
fe_pressure(degree-1), */           

fe (fe_velocity, 1,
    fe_pressure, 1,
    fe_velocity, 1,
    fe_pressure, 1),    
    
fe_single (fe_velocity, 1,
           fe_pressure, 1),
            
#else 

fe_velocity(degree),
fe_pressure(degree-1),            

fe (fe_velocity, dim,
    fe_pressure, 1,
    fe_velocity, dim,
    fe_pressure, 1),
    
fe_single (fe_velocity, dim,
            fe_pressure, 1),
            

#endif
            
dof_handler (triangulation),
dof_handler_single (triangulation), 

dof_handler_single_first_refine (triangulation_first_refine), 

dof_handler_velocity_single_first_refine(triangulation_first_refine),
dof_handler_pressure_single_first_refine(triangulation_first_refine),

n_q_points_one_direction(degree+2),
n_q_points(int(pow(n_q_points_one_direction,dim))),
quadrature_formula(n_q_points_one_direction),                                // important: quadrature_formula is constructed by # of quadrature points in one side


computing_timer  (cout, TimerOutput::summary, TimerOutput::cpu_times)
{
  cout << "================== \n" 
          << "  Space dimension: " << dim << "\n"
          << "  MixedLaplaceProblem_Complex\n"
          << "  Elements: " << (id_fe_system==0?"FE_Q/FE_DGQ":"RT or BDM") << "\n"
          << "  # of quadrature points in one direction: " << n_q_points_one_direction << "\n"
          << "==================\n";
}
  

template <int dim>
void MixedLaplaceProblem_Complex<dim>::make_grid_and_dofs ()
{
  GridGenerator::hyper_cube (triangulation, 0, this->length);
  
  if (dim==1)
  {  
//   triangulation.begin_active()->face(1)->set_boundary_id(0);            // set both boundary conditions as Dirichlet type    
  }else if(dim==2)
  {
    triangulation.begin_active()->face(0)->set_boundary_id(1);       // left
//     triangulation.begin_active()->face(1)->set_boundary_id(1);       // right
//     triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom
//     triangulation.begin_active()->face(3)->set_boundary_id(1);       // top
  }
  
  print_tria_info (triangulation);
  print_boundary_info(triangulation);
  
  obj_string = "fe";
  print_fe_info (obj_string, fe);
  
  triangulation.refine_global (refine);
  triangulation_first_refine.copy_triangulation(triangulation);
  
    
  dof_handler_single_first_refine.distribute_dofs(fe_single);
  DoFRenumbering::component_wise (dof_handler_single_first_refine);
  
  dof_handler_pressure_single_first_refine.distribute_dofs(fe_pressure);
  DoFRenumbering::component_wise (dof_handler_pressure_single_first_refine);
    
  dof_handler_velocity_single_first_refine.distribute_dofs(fe_velocity);
  DoFRenumbering::component_wise (dof_handler_velocity_single_first_refine);
  
  obj_string = "dof_handler_single_first_refine";
  print_dofhandler_info(obj_string,dof_handler_single_first_refine);
  
  obj_string = "dof_handler_pressure_single_first_refine";
  print_dofhandler_info(obj_string,dof_handler_pressure_single_first_refine);
//   save_coords_of_dofs(obj_string, dof_handler_pressure_single_first_refine);
  
  obj_string = "dof_handler_velocity_single_first_refine";
  print_dofhandler_info(obj_string,dof_handler_velocity_single_first_refine);
//   save_coords_of_dofs(obj_string, dof_handler_velocity_single_first_refine);
  
}
  

template <int dim>
void MixedLaplaceProblem_Complex<dim>::setup_system ()
{
  cout << "Setting up the system\n";
  
  dof_handler.distribute_dofs (fe);
  dof_handler_single.distribute_dofs (fe_single);
    
  vector<unsigned int> block_component (2*(dim+1),0);
  
  if(dim==1)
  {
    block_component[0] = 0;
    block_component[1] = 1;
    block_component[2] = 2;
    block_component[3] = 3; 
  }else if(dim==2)
  {
    block_component[0] = 0;
    block_component[1] = 1;
    block_component[dim] = 2;
    block_component[dim+1] = 3;
    block_component[dim+2] = 4;
    block_component[2*dim+1] = 5;         
  }
  
  DoFRenumbering::component_wise (dof_handler, block_component);
  
//   obj_string = "dof_handler";
//   print_coords_of_dofs(obj_string, dof_handler);
    
  if(is_constraints_used==1)
  {
    cout << "  dealing with Neumann boundary conditions using constraints\n";
    
    constraints.clear ();
    constraints_auxiliary.clear ();
    
    FEValuesExtractors::Vector velocity_real(0); 
    FEValuesExtractors::Vector velocity_imag(dim+1);

    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints_auxiliary);      
    
    if(id_fe_system==0)
    {
      cout << "    using VectorTools::interpolate_boundary_values ()\n";
        
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                GradientBoundary_Step_20_Complex<dim>(id_case, id_coeff_diff_value),
	                                            constraints,
	                                            fe.component_mask(velocity_real)
                                               );
    
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                GradientBoundary_Step_20_Complex<dim>(id_case, id_coeff_diff_value),
                                                constraints,
	                                            fe.component_mask(velocity_imag)
                                               );
    }else if(id_fe_system==1)
    {
      cout << "    using VectorTools::project_boundary_values_div_conforming_rt ()\n";
                                                                                         // this function is declared to be developed for RT elements, but it works for BDM elements, too.
                                                                                         // transferred to problems with the real-valued solution
      VectorTools::project_boundary_values_div_conforming_rt(dof_handler,
                                                          0,                             // first_vector_component of velocity_real
                                                          GradientBoundary_Step_20_Complex_Real<dim>(id_case, id_coeff_diff_value),
                                                          1,                             // boundary_component
                                                          constraints 
                                                          );
        
      VectorTools::project_boundary_values_div_conforming_rt(dof_handler,
                                                          3,
                                                          GradientBoundary_Step_20_Complex_Imag<dim>(id_case, id_coeff_diff_value),
                                                          1,
                                                          constraints_auxiliary           // only used for adjusting the constraints for the imag of the Gradient boundary
                                                          );
      
      cout << "  Before adjusting constraints\n";
      obj_string = "constraints";
      print_constraints_info(obj_string, constraints, dof_handler);
      obj_string = "constraints_auxiliary";
      print_constraints_info(obj_string, constraints_auxiliary, dof_handler);    
        
      constraints.adjust_constraints(constraints_auxiliary);
    }
    
    if (is_constraints_printed == 1)
    {
      cout << "  After adjusting constraints\n";
      obj_string = "constraints";
      print_constraints_info(obj_string, constraints, dof_handler);  
//     obj_string = "constraints_auxiliary";
//     print_constraints_info(obj_string, constraints_auxiliary, dof_handler);
    }
    
    
    constraints.close ();
    constraints_auxiliary.close ();
  }else 
  {
    cout << "  not using constraints\n";
  }
   
  DoFRenumbering::component_wise (dof_handler);
  DoFRenumbering::component_wise (dof_handler_single); 
  
//   vector<types::global_dof_index> dofs_per_component_real (dim+1);
//   DoFTools::count_dofs_per_component (dof_handler_single, dofs_per_component_real);
//     
//   vector<types::global_dof_index> dofs_per_component (2*(dim+1));
//   DoFTools::count_dofs_per_component (dof_handler, dofs_per_component, true);
  
  
    const std::vector<types::global_dof_index> dofs_per_component_real =
  DoFTools::count_dofs_per_fe_component(dof_handler);  
    const std::vector<types::global_dof_index> dofs_per_component =
  DoFTools::count_dofs_per_fe_component(dof_handler);  
    
//     cout << "dofs_per_component[0] is: " << dofs_per_component[0] << endl;
//     cout << "dofs_per_component[1] is: " << dofs_per_component[1] << endl;
//     cout << "dofs_per_component[2] is: " << dofs_per_component[2] << endl;
//     cout << "dofs_per_component[3] is: " << dofs_per_component[3] << endl;
                       
//     cout << "dofs_per_component_real[0] is: " << dofs_per_component_real[0] << endl;
//     cout << "dofs_per_component_real[1] is: " << dofs_per_component_real[1] << endl;
    
  unsigned int n_component_velocity = dim;
  
  if(id_fe_system == 1 || id_fe_system==2)
  {
    n_component_velocity = 1;
  }
  
  unsigned int n_u = 0;
  unsigned int n_p = 0;
    
  for(unsigned int i=0; i<n_component_velocity; ++i)
  {
    n_u += dofs_per_component[i];  
  }
  n_p = dofs_per_component[dim];
                       
  cout << "  n_u: " << n_u << "\n"
       << "  n_p: " << n_p << "\n";             //this could be taken as the number of degrees of freedom of u and p for real/imag part

  BlockDynamicSparsityPattern dsp(4, 4);
  dsp.block(0, 0).reinit (n_u, n_u);
  dsp.block(0, 1).reinit (n_u, n_p);
  dsp.block(0, 2).reinit (n_u, n_u);
  dsp.block(0, 3).reinit (n_u, n_p);
    
  dsp.block(1, 0).reinit (n_p, n_u);
  dsp.block(1, 1).reinit (n_p, n_p);
  dsp.block(1, 2).reinit (n_p, n_u);
  dsp.block(1, 3).reinit (n_p, n_p);

  dsp.block(2, 0).reinit (n_u, n_u);
  dsp.block(2, 1).reinit (n_u, n_p);
  dsp.block(2, 2).reinit (n_u, n_u);
  dsp.block(2, 3).reinit (n_u, n_p);
    
  dsp.block(3, 0).reinit (n_p, n_u);
  dsp.block(3, 1).reinit (n_p, n_p);
  dsp.block(3, 2).reinit (n_p, n_u);
  dsp.block(3, 3).reinit (n_p, n_p);
    
  dsp.collect_sizes ();
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  
  BlockDynamicSparsityPattern dsp_single(2, 2);
  dsp_single.block(0, 0).reinit (n_u, n_u);
  dsp_single.block(1, 0).reinit (n_p, n_u);
  dsp_single.block(0, 1).reinit (n_u, n_p);
  dsp_single.block(1, 1).reinit (n_p, n_p);
  dsp_single.collect_sizes ();
  DoFTools::make_sparsity_pattern (dof_handler_single, dsp_single);
    
  system_matrix.clear();

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);

  solution.reinit (4);
  solution.block(0).reinit (n_u);
  solution.block(1).reinit (n_p);
  solution.block(2).reinit (n_u);
  solution.block(3).reinit (n_p);    
  solution.collect_sizes ();
    
  solution_velocity_real.reinit (n_u);
  solution_velocity_imag.reinit (n_u);    
  solution_pressure_real.reinit (n_p);
  solution_pressure_imag.reinit (n_p);   
  solution_amp.reinit (n_p); 
    
    
  solution_real.reinit (2);
  solution_real.block(0).reinit (n_u);
  solution_real.block(1).reinit (n_p);    
  solution_real.collect_sizes ();    
   
  solution_imag.reinit (2);
  solution_imag.block(0).reinit (n_u);
  solution_imag.block(1).reinit (n_p); 
  solution_imag.collect_sizes ();
    
  solution_0.reinit (2);
  solution_0.block(0).reinit (n_u);
  solution_0.block(1).reinit (n_p); 
  solution_0.collect_sizes ();
    
  system_rhs.reinit (4);
  system_rhs.block(0).reinit (n_u);
  system_rhs.block(1).reinit (n_p);
  system_rhs.block(2).reinit (n_u);
  system_rhs.block(3).reinit (n_p);    
  system_rhs.collect_sizes ();
  
  n_times_called_setup++;
}

template <int dim>
void MixedLaplaceProblem_Complex<dim>::assemble_system ()
{
  cout << "Assembling\n";
  TimerOutput::Scope t(computing_timer, "assemble_system");
  
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    | update_gradients |
                           update_quadrature_points  | update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values    | update_normal_vectors |
                                    update_quadrature_points  | update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs (dofs_per_cell);
  vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  
  const RightHandSide_Complex<dim> right_hand_side(id_case, id_coeff_diff_value, id_coeff_helm_value);
  const Coeff_Diff_Complex_Inverse<dim> obj_coeff_diff_inverse(id_case, id_coeff_diff_value, 1.0);
  const Coeff_Helm_Complex<dim> obj_coeff_helm(id_case, id_coeff_helm_value, 1.0);
  const PressureBoundaryValues_Complex<dim> pressure_boundary_values(id_case);
  const GradientBoundary_Step_20_Complex<dim> gradient_boundary_values(id_case, id_coeff_diff_value);
  
  
  if(is_l2_norm_coefficient_numerical == 1 && n_times_called_assemble==0)
  {
    const Coeff_Diff_Complex<dim> obj_coeff_diff(id_case, id_coeff_diff_value, 1.0);

    cout << "  @l2 norms of coefficients\n";
    cout << "  For the diffusion coefficient, ";
    compute_l2_norms_of_coefficient_numerically_core(triangulation, dof_handler, fe_values, obj_coeff_diff, n_number_components);

    cout << "  For the Helmholtz coefficient, ";
    compute_l2_norms_of_coefficient_numerically_core(triangulation, dof_handler, fe_values, obj_coeff_helm, n_number_components);

  }
  
  vector<Point<dim>> coords_of_quadrature (n_q_points, Point<dim>());
  vector<Vector<double>> rhs_values (n_q_points,Vector<double>(2));
  vector<Vector<double>> coeff_diff_inverse_values(n_q_points,Vector<double>(2));
  vector<Vector<double>> coeff_helm_values(n_q_points,Vector<double>(2)); 
  vector<Vector<double>> boundary_values_for_pressure (n_face_q_points,Vector<double>(2));
  vector<Vector<double>> boundary_values_for_gradient (n_face_q_points,Vector<double>(2*(dim+1)));
  
  const FEValuesExtractors::Vector velocity_real (0);
  const FEValuesExtractors::Scalar pressure_real (dim);
  const FEValuesExtractors::Vector velocity_imag (dim+1);
  const FEValuesExtractors::Scalar pressure_imag (dim+dim+1);
  
//   obj_string = "dof_handler";
//   print_coords_of_dofs(obj_string,dof_handler);
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
//     cout << "  Cell #: " << cell->active_cell_index() << endl;
        
    fe_values.reinit (cell);
    local_matrix = 0;
    local_rhs = 0;
    
    coords_of_quadrature = fe_values.get_quadrature_points();
    
    right_hand_side.vector_values(coords_of_quadrature, rhs_values);
    obj_coeff_diff_inverse.value(coords_of_quadrature, coeff_diff_inverse_values);
    obj_coeff_helm.value(coords_of_quadrature, coeff_helm_values); 
    
    cell->get_dof_indices (local_dof_indices);
    
//     cout << "  coords_of_quadrature: \n";
//     print_vector(coords_of_quadrature);
    
//     cout << "  rhs_values: \n";
//     print_vector(rhs_values);

//     cout << "  coeff_diff_inverse_values: \n";
//     print_vector(coeff_diff_inverse_values);
//     
//     cout << "  coeff_helm_values: \n";
//     print_vector(coeff_helm_values);
    
//     cout << "  local_dof_indices: \n";
//     print_vector(local_dof_indices);    
    
//     for (unsigned int i=0; i<dofs_per_cell; ++i)
//     {
//       cout << "  at " << i << " th dof" << ", of dof " << local_dof_indices[i] << " globally ";
//       cout << "fe.system_to_component_index(" << i << ").first: " << fe.system_to_base_index(i).first.first << ", ";
//       cout << ".second: " << fe.system_to_component_index(i).second << ", ";
//       cout << "system_to_base_index(" << i << ").first.first: " << fe.system_to_base_index(i).first.first << ", .first.second: " << fe.system_to_base_index(i).first.second << ", ";
//       cout << ".second: " << fe.system_to_base_index(i).second << "\n";  // note that "fe.system_to_base_index(i).first.first" equals "fe.system_to_component_index(i).first" in the 1D case
//     }

    for (unsigned int q=0; q</*1*/ n_q_points; ++q)
    { 
      for (unsigned int i=0; i</*1*/dofs_per_cell; ++i)
      {
        const Tensor<1,dim> phi_i_velocity_real     = fe_values[velocity_real].value (i, q);
        const double        div_phi_i_velocity_real = fe_values[velocity_real].divergence (i, q);
        const double        phi_i_pressure_real     = fe_values[pressure_real].value (i, q);
        
        const Tensor<1,dim> phi_i_velocity_imag     = fe_values[velocity_imag].value (i, q);
        const double        div_phi_i_velocity_imag = fe_values[velocity_imag].divergence (i, q);             
        const double        phi_i_pressure_imag     = fe_values[pressure_imag].value (i, q);
        
//         cout << "  phi_i_velocity_real: " << phi_i_velocity_real << "\n";
//         cout << "  div_phi_i_velocity_real: " << div_phi_i_velocity_real << "\n";
//         cout << "  phi_i_pressure_real: " << phi_i_pressure_real << "\n";
// 
//         cout << "  phi_i_velocity_imag: " << phi_i_velocity_imag << "\n";
//         cout << "  div_phi_i_velocity_imag: " << div_phi_i_velocity_imag << "\n";
//         cout << "  phi_i_pressure_imag: " << phi_i_pressure_imag << "\n";

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          const Tensor<1,dim> phi_j_velocity_real     = fe_values[velocity_real].value (j, q);
          const double        div_phi_j_velocity_real = fe_values[velocity_real].divergence (j, q);
          const double        phi_j_pressure_real     = fe_values[pressure_real].value (j, q);
        
          const Tensor<1,dim> phi_j_velocity_imag     = fe_values[velocity_imag].value (j, q);
          const double        div_phi_j_velocity_imag = fe_values[velocity_imag].divergence (j, q);             
          const double        phi_j_pressure_imag     = fe_values[pressure_imag].value (j, q);

          if (fe.system_to_base_index(i).first.first == 0)
          {
            if(fe.system_to_base_index(j).first.first == 0)
            {
              local_matrix(i,j) += coeff_diff_inverse_values[q][0] *(phi_i_velocity_real * phi_j_velocity_real) * fe_values.JxW(q);
            }else if(fe.system_to_base_index(j).first.first == 1)
            {
              local_matrix(i,j) += - div_phi_i_velocity_real * phi_j_pressure_real * fe_values.JxW(q);
            }else if(fe.system_to_base_index(j).first.first == 2)
            {
              local_matrix(i,j) += - coeff_diff_inverse_values[q][1] * (phi_i_velocity_real * phi_j_velocity_imag) * fe_values.JxW(q);
            }
          }else if(fe.system_to_base_index(i).first.first == 1)
          {
            if(fe.system_to_base_index(j).first.first == 0)
            {
              local_matrix(i,j) += - phi_i_pressure_real * div_phi_j_velocity_real * fe_values.JxW(q);
            }else
            if(fe.system_to_base_index(j).first.first == 1)
            {
              local_matrix(i,j) += - phi_i_pressure_real * (coeff_helm_values[q][0] * phi_j_pressure_real) * fe_values.JxW(q);
            }else if(fe.system_to_base_index(j).first.first == 3)
            {
              local_matrix(i,j) += phi_i_pressure_real * (coeff_helm_values[q][1] * phi_j_pressure_imag) * fe_values.JxW(q);
            }
            
          }else if(fe.system_to_base_index(i).first.first == 2)
          {
            if(fe.system_to_base_index(j).first.first == 0)
            {
              local_matrix(i,j) += coeff_diff_inverse_values[q][1] * (phi_i_velocity_imag * phi_j_velocity_real) * fe_values.JxW(q);
            }else if(fe.system_to_base_index(j).first.first == 2)
            {
              local_matrix(i,j) += coeff_diff_inverse_values[q][0] * (phi_i_velocity_imag * phi_j_velocity_imag) * fe_values.JxW(q);
            }else if(fe.system_to_base_index(j).first.first == 3)
            {
              local_matrix(i,j) += -div_phi_i_velocity_imag * phi_j_pressure_imag * fe_values.JxW(q);
            }
          }else if(fe.system_to_base_index(i).first.first == 3)
          {
            if(fe.system_to_base_index(j).first.first == 1)
            {
              local_matrix(i,j) += - phi_i_pressure_imag * (coeff_helm_values[q][1] * phi_j_pressure_real) * fe_values.JxW(q);
            }else if(fe.system_to_base_index(j).first.first == 2)
            {
              local_matrix(i,j) += - phi_i_pressure_imag * div_phi_j_velocity_imag * fe_values.JxW(q);
            }else if(fe.system_to_base_index(j).first.first == 3)
            {
              local_matrix(i,j) += - phi_i_pressure_imag * (coeff_helm_values[q][0] *phi_j_pressure_imag) * fe_values.JxW(q);
            }
          }
          
        }
        
        if(fe.system_to_base_index(i).first.first == 1)                          // note that the contribution of the rhs is independent of j
        {
          local_rhs(i) += -phi_i_pressure_real * rhs_values[q][0] * fe_values.JxW(q);
        }else if(fe.system_to_base_index(i).first.first == 3)
        {
          local_rhs(i) += -phi_i_pressure_imag * rhs_values[q][1] * fe_values.JxW(q);
        }
      }
    }
    
    for (unsigned int face_n=0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n)
    {
      if (cell->at_boundary(face_n))
      {
          
        fe_face_values.reinit (cell, face_n);
        vector<Point<dim>> coords_of_boundary = fe_face_values.get_quadrature_points();
        
//         cout << "  ##############\n";
//         cout << "  on face " << cell->face_index(face_n) << " of cell " << cell->active_cell_index() << "\n";
//         cout << "  with vertices ";
//         for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
//         {
//           cout << "(" << cell->face(face_n)->vertex(vertex) << ")";
//         }
//         cout << "\n";
//         cout << "  with quadrature points \n";
//         print_vector(coords_of_boundary);
        
        if(cell->face(face_n)->boundary_id() == 0)
        {
          pressure_boundary_values.vector_value_list(coords_of_boundary, boundary_values_for_pressure);
            
//           cout << "  Dirichlet BCs are imposed\n";
//           cout << "  boundary values for pressure: \n";
//           print_vector(boundary_values_for_pressure);

          for (unsigned int q=0; q<n_face_q_points; ++q)
          {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              local_rhs(i) += -(fe_face_values[velocity_real].value (i, q) *
                              fe_face_values.normal_vector(q) *
                              boundary_values_for_pressure[q][0] *
                              fe_face_values.JxW(q));
              local_rhs(i) += -(fe_face_values[velocity_imag].value (i, q) *
                              fe_face_values.normal_vector(q) *
                              boundary_values_for_pressure[q][1] *
                              fe_face_values.JxW(q));
            }
          }
        }else if(cell->face(face_n)->boundary_id() == 1)
        {
//           cout << "  Neumman BCs are imposed\n";
//           gradient_boundary_values.vector_value_list(coords_of_boundary, boundary_values_for_gradient); 
//           cout << "  boundary values for gradient: \n";
//           print_vector(boundary_values_for_gradient);
        }
      }
    }
    
//     cout << "  local_matrix: \n";
//     local_matrix.print(std::cout);
//     cout << "  local_rhs: \n";
//     cout << local_rhs << "\n";
    
    if(is_constraints_used==1)      // dealing with the global matrix and right hand side and implement (inhomogeneous) essential boundary conditions
    {
      constraints.condense (system_matrix, system_rhs);
      constraints.distribute_local_to_global (local_matrix, local_rhs,
                                              local_dof_indices,
                                              system_matrix, system_rhs);
      
    }else if(is_constraints_used==0)                                     // assembling using the normal way, which requires eliminating the essential BCs later
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             local_matrix(i,j));
        }
      }
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        system_rhs(local_dof_indices[i]) += local_rhs(i);    
      }  
    }
  }
    
  if(is_constraints_used==0)                           // dealing with the velocity boundary condition mannualy when not using constraints, only applicable for 1d
  {
    if (is_matrices_before_neum_BC_printed==1)
    {
      cout << "system_matrix before applying the velocity boundary conditions: \n";
      system_matrix.print_formatted(cout);
      cout << '\n';    
      cout << "system_rhs before applying the velocity boundary conditions: \n";
      system_rhs.print(cout);     
      cout << '\n';
    }
    
    unsigned int ur_row_last_dof = 1+triangulation.n_active_cells()*(degree)-(degree);             
    unsigned int ui_row_last_dof = dof_handler.n_dofs()/2+1+triangulation.n_active_cells()*(degree)-(degree);
//     cout << "    ur_row_last_dof is " << ur_row_last_dof << ", ui_row_last_dof is " << ui_row_last_dof << endl;
//     ur_row_last_dof and ui_row_last_dof denote the row number of the last degree of freedom of ur and ui, respectively   
  
    for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)                   // influence of the latter elimination of columns on the rhs
    {
      if(system_matrix.el(i,ur_row_last_dof)!=0)
      {
//         cout << "    at row " << i << ", system_matrix.el(i,ur_row_last_dof) reads " << system_matrix.el(i,ur_row_last_dof);
//         if(i!=ur_row_last_dof )                                           
//         {   
//             cout << "; its product with the velocity boundary " << system_matrix.el(i,ur_row_last_dof)*pressure_boundary_values.coeff_b << " goes to the rhs" << endl;        
//         }else 
//         {
//             cout << endl;
//         }
            
        if(i!=ur_row_last_dof)
        {   
            system_rhs(i)-=system_matrix.el(i,ur_row_last_dof)*boundary_values_for_gradient[0][0];
        }              
      }    
    }

    for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
    {
      if(system_matrix.el(ur_row_last_dof,i)!=0 || system_matrix.el(ui_row_last_dof,i)!=0 )       // setting rows involving the last test function of ur and ui zero
      {
        system_matrix.set(ur_row_last_dof,i,0.0);            
        system_matrix.set(ui_row_last_dof,i,0.0); 
      }
    }
    
    for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
    {
      if(system_matrix.el(i,ur_row_last_dof)!=0 || system_matrix.el(i,ui_row_last_dof)!=0 )     // setting columns involving the last test function of ur and ui zero
      {   
        system_matrix.set(i,ur_row_last_dof,0.0);            
        system_matrix.set(i,ui_row_last_dof,0.0);
      }        
    }

    system_matrix.set(ur_row_last_dof,ur_row_last_dof,1.0);
    system_matrix.set(ui_row_last_dof,ui_row_last_dof,1.0);
    
    system_rhs(ur_row_last_dof) = boundary_values_for_gradient[0][0];       //this->coeff_b       // satisfying the gradient boundary conditions manually
    system_rhs(ui_row_last_dof) = boundary_values_for_gradient[0][2];         
  }
    
  if(is_matrices_after_neum_BC_printed==1)
  {
    cout << "After applying velocity boundary conditions\n";
    cout << "system_matrix is \n";
    system_matrix.print_formatted(cout);
    cout << '\n';    
    cout << "system_rhs is \n";
    system_rhs.print(cout);     
    cout << '\n'; 
  }
  
  if (is_matrices_after_BC_stored==1)
  {
    ostringstream streamObj;
    streamObj << fixed;
    streamObj << setprecision(0);
    streamObj << 1;   
    
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
void MixedLaplaceProblem_Complex<dim>::solve ()
{
  TimerOutput::Scope t(computing_timer, "solve");
  cout << "Solving \n";
  
  if(is_UMFPACK_mono==1)                    
  {
    cout << "  UMFPack solver for monolithic\n";
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);  
  }else if (is_UMFPACK_mono==0)                                 // copied from real-valued problems
  {
    cout << "  Schur complement\n";
    InverseMatrix_UMF<SparseMatrix<real_t> > inverse_mass (system_matrix.block(0,0));
    Vector<real_t> tmp (solution.block(0).size());
    {
      Vector<real_t> schur_rhs (solution.block(1).size());
      inverse_mass.vmult (tmp, system_rhs.block(0));
      system_matrix.block(1,0).vmult (schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);
        
      cout << "  computing solution\n";            

      SchurComplement<InverseMatrix_UMF<SparseMatrix<real_t> >> schur_complement (system_matrix, inverse_mass);
                    
      double l2_norm_schur_rhs = schur_rhs.l2_norm();
      cout << "  l2_norm_schur_rhs: " << l2_norm_schur_rhs << endl;
      
      SolverControl solver_control (solution.block(1).size()*10,                      
                                    tol_prm_schur*l2_norm_schur_rhs);    
      SolverCG<> cg (solver_control);
            
      cg.solve (schur_complement, solution.block(1), schur_rhs,
                PreconditionIdentity());
      CG_iteration_schur = solver_control.last_step();
        
      cout << "  "
                << solver_control.last_step()
                << " CG iterations needed."
                << endl; 
    }

    cout << "  computing gradient";        
    {
      system_matrix.block(0,1).vmult (tmp, solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);
      inverse_mass.vmult (solution.block(0), tmp);
    }
  }
  
  if (is_constraints_used==1)
  {
    constraints.distribute (solution);    
  }
        
  solution_velocity_real = solution.block(0);
  solution_pressure_real = solution.block(1);
  solution_velocity_imag = solution.block(2);
  solution_pressure_imag = solution.block(3);
  
  solution_real.block(0) = solution.block(0);
  solution_real.block(1) = solution.block(1);
  solution_imag.block(0) = solution.block(2);
  solution_imag.block(1) = solution.block(3);   
  
  for (unsigned int r = 0; r < solution_pressure_real.size(); r++)
  {
    solution_amp[r] = sqrt(solution_pressure_real[r]*solution_pressure_real[r] + solution_pressure_imag[r]*solution_pressure_imag[r]);
  }

  if(is_solution_after_distributing_printed == 1)
  {
    cout << "  solution: \n";
    solution.print(cout);  
    cout << "  solution_velocity_real: ";
    cout << solution_velocity_real;  
    cout << "  solution_pressure_real: ";
    cout << solution_pressure_real;
    cout << "  solution_velocity_imag: ";
    cout << solution_velocity_imag;
    cout << "  solution_pressure_imag: ";
    cout << solution_pressure_imag;
  }

  n_times_called_solve++;
  
}


template <int dim>
void MixedLaplaceProblem_Complex<dim>::store_solution ()
{
  obj_string="solution_pressure_real";
  save_Vector_to_txt(obj_string,solution_pressure_real);
//   obj_string="solution_velocity_real";
//   save_Vector_to_txt(obj_string,solution_velocity_real);
//   obj_string="solution_velocity_imag";
//   save_Vector_to_txt(obj_string,solution_velocity_imag);  
    
}


template <int dim>
void MixedLaplaceProblem_Complex<dim>::compute_l2_norms_of_variable_using_numerical_solution ()
{
  cout << "Computing L2 norms of variable using numerical solution\n";
  
#if 1
  
  const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);                    // notice the 'const' identifier
  const ComponentSelectFunction<dim> velocity_mask(0, dim+1);

  Zero_Function_Custom<dim> obj_zero_function;
  
  Vector<double> cellwise_L2_norms (triangulation.n_active_cells());
  Vector<double> cellwise_L2_norms_real (triangulation.n_active_cells());
  Vector<double> cellwise_L2_norms_imag (triangulation.n_active_cells());
    
  QTrapez<1>     q_trapez;
  QIterated<dim> quadrature (q_trapez, degree+2);


  VectorTools::integrate_difference (dof_handler_single, solution_real, obj_zero_function,
                                    cellwise_L2_norms_real, quadrature,
                                    VectorTools::L2_norm,
                                    &pressure_mask);

  VectorTools::integrate_difference (dof_handler_single, solution_imag, obj_zero_function,
                                    cellwise_L2_norms_imag, quadrature,
                                    VectorTools::L2_norm,
                                    &pressure_mask);

  pressure_inte_L2_numerical_real = cellwise_L2_norms_real.l2_norm();
  pressure_inte_L2_numerical_imag = cellwise_L2_norms_imag.l2_norm();
  pressure_inte_L2_numerical = sqrt(pow(pressure_inte_L2_numerical_real,2.0)+pow(pressure_inte_L2_numerical_imag,2.0));
  
//   cout << "  @l2 norms\n";
//   cout << "  pressure_inte_L2: " << pressure_inte_L2_numerical << ", real: " << pressure_inte_L2_numerical_real << ", imag: " << pressure_inte_L2_numerical_imag << endl;
//   
//   l2_norm_computation_numerical_solution_mix_core(dof_handler_single, solution_real, obj_zero_function, cellwise_L2_norms, quadrature, &pressure_mask);
//   
//   pressure_inte_L2_numerical_real = cellwise_L2_norms.l2_norm();
//   
//   cout << "  @l2 norms furhter\n";
//   cout << "  pressure_inte_L2: " << pressure_inte_L2_numerical << ", real: " << pressure_inte_L2_numerical_real << ", imag: " << pressure_inte_L2_numerical_imag << endl;
//   
  
  VectorTools::integrate_difference (dof_handler_single, solution_real, obj_zero_function,
                                    cellwise_L2_norms_real, quadrature,
                                    VectorTools::L2_norm,
                                    &velocity_mask);

  VectorTools::integrate_difference (dof_handler_single, solution_imag, obj_zero_function,
                                    cellwise_L2_norms_imag, quadrature,
                                    VectorTools::L2_norm,
                                    &velocity_mask);
  
  velocity_inte_L2_numerical_real = cellwise_L2_norms_real.l2_norm();
  velocity_inte_L2_numerical_imag = cellwise_L2_norms_imag.l2_norm();
  velocity_inte_L2_numerical = sqrt(pow(velocity_inte_L2_numerical_real,2.0)+pow(velocity_inte_L2_numerical_imag,2.0));
  
  
  VectorTools::integrate_difference (dof_handler_single, solution_real, obj_zero_function,
                                     cellwise_L2_norms_real, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_imag, obj_zero_function,               
                                     cellwise_L2_norms_imag, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  velocity_inte_H1_semi_numerical_real = cellwise_L2_norms_real.l2_norm();
  velocity_inte_H1_semi_numerical_imag = cellwise_L2_norms_imag.l2_norm();
  velocity_inte_H1_semi_numerical = sqrt(pow(velocity_inte_H1_semi_numerical_real,2.0)+pow(velocity_inte_H1_semi_numerical_imag,2.0));
  
  
  cout << "  @l2 norms\n";
  cout << "  pressure_inte_L2_numerical: " << pressure_inte_L2_numerical << ", real: " << pressure_inte_L2_numerical_real << ", imag: " << pressure_inte_L2_numerical_imag << endl;        
  cout << "  velocity_inte_L2_numerical: " << velocity_inte_L2_numerical << ", real: " << velocity_inte_L2_numerical_real << ", imag: " << velocity_inte_L2_numerical_imag << endl;
  cout << "  velocity_inte_H1_semi_numerical: " << velocity_inte_H1_semi_numerical << ", real: " << velocity_inte_H1_semi_numerical_real << ", imag: " << velocity_inte_H1_semi_numerical_imag << endl;
  
#endif
      
}


template <int dim>
void MixedLaplaceProblem_Complex<dim>::compute_errors_and_l2_norms_built_in ()
{
  cout << "Computing errors and l2 norms built-in\n";
  
  const ComponentSelectFunction<dim>
  pressure_mask (dim, dim+1);
  const ComponentSelectFunction<dim>
  velocity_mask(0, dim+1);

  ExactSolution_Step_20_Complex_Real<dim> exact_solution_real(id_case, id_coeff_diff_value);
  ExactSolution_Step_20_Complex_Imag<dim> exact_solution_imag(id_case, id_coeff_diff_value);
    
  Vector<double> cellwise_errors (triangulation.n_active_cells());
  Vector<double> cellwise_errors_real (triangulation.n_active_cells());
  Vector<double> cellwise_errors_imag (triangulation.n_active_cells());    

  QTrapez<1>     q_trapez;
  QIterated<dim> quadrature (q_trapez, degree+2);
    
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_real,            // the L2 norm concerning the pressure
                                     cellwise_errors_real, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_imag,
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
  
  pressure_inte_L2_analytical_real = cellwise_errors_real.l2_norm();
  pressure_inte_L2_analytical_imag = cellwise_errors_imag.l2_norm();
  pressure_inte_L2_analytical = pow(pow(pressure_inte_L2_analytical_real,2.0)+pow(pressure_inte_L2_analytical_imag,2.0),0.5);
   
  VectorTools::integrate_difference (dof_handler_single, solution_real, exact_solution_real,               
                                     cellwise_errors_real, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_imag, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
  
//   cout << "  cellwise error of the pressure\n";
//   cout << "  cellwise_errors_real: ";
//   cout << cellwise_errors_real;
//   cout << "  cellwise_errors_imag: ";
//   cout << cellwise_errors_imag;
  
  pressure_abs_error_L2_real = cellwise_errors_real.l2_norm();
  pressure_abs_error_L2_imag = cellwise_errors_imag.l2_norm();
  pressure_abs_error_L2 = pow(pow(pressure_abs_error_L2_real,2.0)+pow(pressure_abs_error_L2_imag,2.0),0.5);  
  
  
  pressure_rel_error_L2 = pressure_abs_error_L2/pressure_inte_L2_analytical;
  
  
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_real,                         // the L2_norm concerning the gradient
                                     cellwise_errors_real, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
    
  velocity_inte_L2_analytical_real = cellwise_errors_real.l2_norm();
  velocity_inte_L2_analytical_imag = cellwise_errors_imag.l2_norm();
  velocity_inte_L2_analytical = pow(pow(velocity_inte_L2_analytical_real,2.0)+pow(velocity_inte_L2_analytical_imag,2.0),0.5);
  
  VectorTools::integrate_difference (dof_handler_single, solution_real, exact_solution_real,               
                                     cellwise_errors_real, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_imag, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
  
//   cout << "  cellwise error of the velocity\n";
//   cout << "  cellwise_errors_real: ";
//   cout << cellwise_errors_real;
//   cout << "  cellwise_errors_imag: ";
//   cout << cellwise_errors_imag;  
    
  velocity_abs_error_L2_real = cellwise_errors_real.l2_norm();
  velocity_abs_error_L2_imag = cellwise_errors_imag.l2_norm();
  velocity_abs_error_L2 = pow(pow(velocity_abs_error_L2_real,2.0)+pow(velocity_abs_error_L2_imag,2.0),0.5);  
  
  velocity_rel_error_L2 = velocity_abs_error_L2/velocity_inte_L2_analytical;
  
   
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_real,                         // the H1_seminorm concerning the gradient          
                                     cellwise_errors_real, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  velocity_inte_H1_semi_analytical_real = cellwise_errors_real.l2_norm();
  velocity_inte_H1_semi_analytical_imag = cellwise_errors_imag.l2_norm();
  velocity_inte_H1_semi_analytical = pow(pow(velocity_inte_H1_semi_analytical_real,2.0)+pow(velocity_inte_H1_semi_analytical_imag,2.0),0.5);
  
   
  VectorTools::integrate_difference (dof_handler_single, solution_real, exact_solution_real,               
                                     cellwise_errors_real, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_imag, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);

//   cout << "  cellwise error of the derivative of the velocity\n";
//   cout << "  cellwise_errors_real: ";
//   cout << cellwise_errors_real;
//   cout << "  cellwise_errors_imag: ";
//   cout << cellwise_errors_imag;  
  
  velocity_abs_error_H1_semi_real = cellwise_errors_real.l2_norm();
  velocity_abs_error_H1_semi_imag = cellwise_errors_imag.l2_norm();
  velocity_abs_error_H1_semi = pow(pow(velocity_abs_error_H1_semi_real,2.0)+pow(velocity_abs_error_H1_semi_imag,2.0),0.5);  
  
  velocity_rel_error_H1_semi = velocity_abs_error_H1_semi/velocity_inte_H1_semi_analytical;
  
  cout << "  @l2 norms\n"
       << "  pressure_inte_L2_analytical: " << pressure_inte_L2_analytical << ", "
       << "real: " << pressure_inte_L2_analytical_real << ", "
       << "imag: " << pressure_inte_L2_analytical_imag << "\n";
  
  cout << "  velocity_inte_L2_analytical: " << velocity_inte_L2_analytical << ", "
       << "real: " << velocity_inte_L2_analytical_real << ", "
       << "imag: " << velocity_inte_L2_analytical_imag << "\n";
       
  cout << "  velocity_inte_H1_semi_analytical: " << velocity_inte_H1_semi_analytical << ", "
       << "real: " << velocity_inte_H1_semi_analytical_real << ", "
       << "imag: " << velocity_inte_H1_semi_analytical_imag << "\n";
       
       
  cout << "  @errors\n";
  cout << "  pressure_abs_error_L2 = " << pressure_abs_error_L2 << ", "
       << "real: " << pressure_abs_error_L2_real << ", "
       << "imag: " << pressure_abs_error_L2_imag << "\n";
       
  cout << "  velocity_abs_error_L2 = " << velocity_abs_error_L2 << ", " 
       << "real: " << velocity_abs_error_L2_real << ", "
       << "imag: " << velocity_abs_error_L2_imag << "\n";
  
  cout << "  velocity_abs_error_H1_semi = " << velocity_abs_error_H1_semi << ", "
       << "real: " << velocity_abs_error_H1_semi_real << ", "
       << "imag: " << velocity_abs_error_H1_semi_imag << "\n";
  

       
//   cout << "\n";
//   cout << "  pressure_rel_error_L2 = " << pressure_rel_error_L2 << "\n"
//        << "  velocity_rel_error_L2 = " << velocity_rel_error_L2 << "\n"
//        << "  velocity_rel_error_H1_semi = " << velocity_rel_error_H1_semi << "\n";
}

  
template <int dim>
void MixedLaplaceProblem_Complex<dim>::compute_errors_custom ()                      
{
  cout << "Computing errors customly\n";
  
//   solution_real_first_refine.reinit (2);
//   solution_real_first_refine.block(0).reinit (n_u);
//   solution_real_first_refine.block(1).reinit (n_p);    
//   solution_real_first_refine.collect_sizes ();
//   
//   solution_imag_first_refine.reinit (2);
//   solution_imag_first_refine.block(0).reinit (n_u);
//   solution_imag_first_refine.block(1).reinit (n_p);    
//   solution_imag_first_refine.collect_sizes ();

  solution_real_first_refine=solution_real;
  solution_imag_first_refine=solution_imag;
  

  cout << "\n//////////////////\n";
  triangulation.refine_global ();                 
  setup_system ();
  assemble_system ();
  solve ();
  
  vector<Vector<double>> cellwise_error(3, Vector<double>(dof_handler_single_first_refine.get_triangulation().n_active_cells()));               // '0' for pressure, '1' for velocity, '2' for 2nnd
  vector<Vector<double>> cellwise_error_real(3, Vector<double>(dof_handler_single_first_refine.get_triangulation().n_active_cells()));
  vector<Vector<double>> cellwise_error_imag(3, Vector<double>(dof_handler_single_first_refine.get_triangulation().n_active_cells()));
  
  error_computation_custom_real_mix_core(solution_real_first_refine, solution_real, dof_handler_single_first_refine, dof_handler_single, cellwise_error_real);      
  error_computation_custom_real_mix_core(solution_imag_first_refine, solution_imag, dof_handler_single_first_refine, dof_handler_single, cellwise_error_imag);      
                                                                // checked using the Sep 28, 2020 version for simp_coe, also checked using code for case for validation
  for (unsigned int k = 0; k<3; ++k)
  {
    for(unsigned int i = 0; i<triangulation_first_refine.n_active_cells(); ++i)
    {
      cellwise_error[k][i] = pow(cellwise_error_real[k][i], 2)+pow(cellwise_error_imag[k][i], 2);
      cellwise_error[k][i] = pow(cellwise_error[k][i], 0.5);                                               
    }
  }
  
  pressure_abs_error_L2_custom = cellwise_error[0].l2_norm();
  velocity_abs_error_L2_custom = cellwise_error[1].l2_norm();
  velocity_abs_error_H1_semi_custom = cellwise_error[2].l2_norm();
  
  pressure_abs_error_L2_custom_real = cellwise_error_real[0].l2_norm();
  velocity_abs_error_L2_custom_real = cellwise_error_real[1].l2_norm();
  velocity_abs_error_H1_semi_custom_real = cellwise_error_real[2].l2_norm();

  pressure_abs_error_L2_custom_imag = cellwise_error_imag[0].l2_norm();
  velocity_abs_error_L2_custom_imag = cellwise_error_imag[1].l2_norm();
  velocity_abs_error_H1_semi_custom_imag = cellwise_error_imag[2].l2_norm();

  cout << "  @results for custom functions\n";
  cout << "  pressure_abs_error_L2_custom: " << pressure_abs_error_L2_custom << ", "
       << "real: " << pressure_abs_error_L2_custom_real << ", imag: " << pressure_abs_error_L2_custom_imag << "\n";  
  cout << "  velocity_abs_error_L2_custom: " << velocity_abs_error_L2_custom << ", "
       << "real: " << velocity_abs_error_L2_custom_real << ", imag: " << velocity_abs_error_L2_custom_imag << "\n";  
  cout << "  velocity_abs_error_H1_semi_custom: " << velocity_abs_error_H1_semi_custom << ", "
       << "real: " << velocity_abs_error_H1_semi_custom_real << ", imag: " << velocity_abs_error_H1_semi_custom_imag << "\n";    
  
  cout << "//////////////////\n\n";  
}

template <int dim>
void MixedLaplaceProblem_Complex<dim>::output_results () const
{
  std::vector<std::string> solution_names(1,"u_x_real");
  solution_names.push_back ("u_y_real");
  solution_names.push_back ("p_real");
    
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation (1,DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);    
    
  DataOut<dim> data_out;
  data_out.add_data_vector (dof_handler_single, solution_real, solution_names, interpretation);

  data_out.build_patches (degree+1);

  std::ofstream output ("solution_mixed_complex_real.vtk");
  data_out.write_vtk (output);
    
    
    
  std::vector<std::string> solution_names_imag(1,"u_x_imag");
  solution_names_imag.push_back ("u_y_imag");
  solution_names_imag.push_back ("p_imag");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation_imag (1,DataComponentInterpretation::component_is_scalar);
  interpretation_imag.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation_imag.push_back (DataComponentInterpretation::component_is_scalar);
    
  DataOut<dim> data_out_imag;
  data_out_imag.add_data_vector (dof_handler_single, solution_imag, solution_names_imag, interpretation_imag);

  data_out_imag.build_patches (degree+1);

  std::ofstream output_imag ("solution_mixed_complex_imag.vtk");
  data_out_imag.write_vtk (output_imag);  
}


template <int dim>
void MixedLaplaceProblem_Complex<dim>::print_errors_to_file()
{
  ofstream myfile;
  if(dim==1)
  {
    obj_string="oned";  
  }else if(dim==2)
  {
    obj_string="twod";
  }  
  myfile.open ("data_error_"+obj_string+"_complex_mm.txt", ofstream::app);
  
  myfile << refine << " ";
  myfile << triangulation_first_refine.n_vertices() <<" ";    
  myfile << dof_handler_pressure_single_first_refine.n_dofs()*2 << " ";  
  myfile << dof_handler_velocity_single_first_refine.n_dofs()*2 << " ";
  myfile << pressure_abs_error_L2 << " ";                       // _custom
  myfile << velocity_abs_error_L2 << " ";
//   myfile << velocity_abs_error_L2_real << " ";
//   myfile << velocity_abs_error_L2_imag << " ";
  myfile << velocity_abs_error_H1_semi << " ";
  myfile << total_CPU_time << "\n";
  
  myfile.close();
}


template <int dim>
void MixedLaplaceProblem_Complex<dim>::print_l2_norms_to_file()
{
  ofstream myfile;
  if(dim==1)
  {
    obj_string="oned";  
  }else if(dim==2)
  {
    obj_string="twod";
  }  
  myfile.open ("data_l2_norm_"+obj_string+"_complex_mm.txt", ofstream::app);
  
  myfile << refine << " ";
  myfile << triangulation_first_refine.n_vertices() <<" ";    
  myfile << dof_handler_pressure_single_first_refine.n_dofs()*2 << " ";  
  myfile << dof_handler_velocity_single_first_refine.n_dofs()*2 << " ";

  myfile << pressure_inte_L2_numerical_real << " ";
  myfile << pressure_inte_L2_analytical_real << " ";
  
  myfile << velocity_inte_L2_numerical_real << " ";
  myfile << velocity_inte_L2_analytical_real << "\n";
  
  myfile.close();
  
}


template <int dim>
void MixedLaplaceProblem_Complex<dim>::run ()
{
  make_grid_and_dofs();
  setup_system ();
  assemble_system ();
     
  solve ();
    
  if (is_solution_stored == 1)
  {
    store_solution();
  }
  
  if(is_l2_norm_variable_numerical==1)
  {
    compute_l2_norms_of_variable_using_numerical_solution();
  }
  
  if(is_error_and_l2_norm_built_in==1)
  {
    compute_errors_and_l2_norms_built_in ();  
  }
  
  if(is_error_custom==1)
  {
    compute_errors_custom ();      
  }
  
  if(dim==2 && is_results_outputted==1)
  {
    output_results ();
  }

  total_CPU_time += computing_timer.return_total_cpu_time ();               // total_CPU_time is above print_errors_to_file(), because it is used there
  
  print_errors_to_file();   
  print_l2_norms_to_file();   
  
  
  cout << "Summary: \n";
  cout << "  # of setup_system() called: " << n_times_called_setup << "\n";
  cout << "  # of assemble_system() called: " << n_times_called_assemble << "\n";
  cout << "  # of solve() called: " << n_times_called_solve << "\n";
  cout << "  total_CPU_time: " << total_CPU_time << "\n";
    
}

