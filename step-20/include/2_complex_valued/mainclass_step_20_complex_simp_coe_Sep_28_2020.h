
#include <deal.II/fe/fe_raviart_thomas.h>
#include <auxiliaryclass_step_20.h>

#include<2_complex_valued/auxiliaryclass_step_20_complex.h>

  template <int dim>
  class MixedLaplaceProblem_Complex_Simp_Coe
  {
  public:
    MixedLaplaceProblem_Complex_Simp_Coe (const unsigned int id_case, const double length, const unsigned int degree, const unsigned int refine);
    void run ();

  private:
    double L2_exact_pr, pressure_abs_error_L2_custom, velocity_abs_error_L2_custom, L2_error_2ndd_custom;
    
    double pressure_abs_error_L2, pressure_inte_L2, pressure_rel_error_L2;
    double velocity_abs_error_L2, velocity_inte_L2, velocity_rel_error_L2;
    double velocity_abs_error_H1_semi, velocity_inte_H1_semi, velocity_rel_error_H1_semi;
    
    
    void make_grid_and_dofs ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void compute_errors ();
    void compute_errors_quad ();
    void output_results () const;

    const unsigned int   id_case;
    const double length;
    const unsigned int   degree;
    const unsigned int   refine;
    const double tol_prm_schur = 1e-16;
    unsigned int CG_iteration_schur = 0;
    
    unsigned int   n_u;
    unsigned int   n_p;
    
    unsigned int n_q_points; 
    QGauss<dim>  quadrature_formula;     
    
    Triangulation<dim>   triangulation;
    Triangulation<dim>   triangulation_first_refine;
    
    FE_Q<dim>          fe_q;
    FE_DGQ<dim>          fe_dgq;
    
    FESystem<dim>        fe;
    FESystem<dim>        fe_single;   
    
    DoFHandler<dim>      dof_handler;
    
    DoFHandler<dim>      dof_handler_single;
    DoFHandler<dim>      dof_handler_single_first_refine; 
    
    DoFHandler<dim>      dof_handler_velocity_single;
    DoFHandler<dim>      dof_handler_velocity_single_first_refine;
    DoFHandler<dim>      dof_handler_pressure_single;
    DoFHandler<dim>      dof_handler_pressure_single_first_refine;
    
    unsigned int is_constraints=1;
    
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double>       system_rhs;
    
    unsigned int is_matrices_before_diri_BC_printed = 0;
    unsigned int is_matrices_after_diri_BC_printed = 0;
    
    unsigned int is_UMFPACK_mono = 1;

    BlockVector<double>       solution;  
    
    Vector<double>            solution_velocity_real;
    Vector<double>            solution_velocity_imag;
    Vector<double>            solution_pressure_real;
    Vector<double>            solution_pressure_imag;  
    Vector<double>            solution_amp;
    
    BlockVector<double>            solution_real;
    BlockVector<double>            solution_imag;    
    
    BlockVector<double>       solution_0;
    TimerOutput                               computing_timer;
    void printToFile() const;
    double total_CPU_time;

    ConstraintMatrix constraints;

  };

  template <int dim>
void MixedLaplaceProblem_Complex_Simp_Coe<dim>::printToFile() const
  {
    std::ofstream myfile;
    myfile.open ("OneD_Helm_simp_MM.txt", std::ofstream::app);
      myfile << refine <<" ";
      myfile << triangulation.n_vertices() <<" ";
      myfile << dof_handler.n_dofs() << " ";
      myfile << n_u << " ";
      myfile << n_p << " ";      
      
//       myfile << pressure_abs_error_L2_custom << " ";
//       myfile << velocity_abs_error_L2_custom << " ";
//       myfile << L2_error_2ndd_custom << " ";
      
    myfile << pressure_abs_error_L2 << " ";
      myfile << pressure_rel_error_L2 << " ";
      myfile << velocity_abs_error_L2 << " ";
      myfile << velocity_rel_error_L2 << " ";
      myfile << velocity_abs_error_H1_semi << " ";
      myfile << velocity_rel_error_H1_semi << " ";
      
      myfile << total_CPU_time << "\n";
  }
  
  template <int dim>
  MixedLaplaceProblem_Complex_Simp_Coe<dim>::MixedLaplaceProblem_Complex_Simp_Coe (const unsigned int id_case, const double length, const unsigned int degree, const unsigned int refine)
    :
    id_case (id_case),
    length(length),
    degree (degree),
    refine(refine),
    
    n_q_points(degree+1),
    quadrature_formula(n_q_points),
    
    fe_q(degree),
    fe_dgq(degree-1),
    
	fe (FE_Q<dim>(degree), 1,                  // elements for the real part of the gradient
        FE_DGQ<dim>(degree-1), 1,              // elements for the real part of the solution
        FE_Q<dim>(degree), 1,                  // elements for the imag part of the gradient
	    FE_DGQ<dim>(degree-1), 1),             // elements for the imag part of the solution
	    
	fe_single (FE_Q<dim>(degree), 1,
             FE_DGQ<dim>(degree-1), 1),	    
             
    dof_handler (triangulation),
    dof_handler_single (triangulation), 
    
    dof_handler_single_first_refine (triangulation_first_refine), 
    
    dof_handler_velocity_single (triangulation),
    dof_handler_velocity_single_first_refine(triangulation_first_refine),
    dof_handler_pressure_single (triangulation),
    dof_handler_pressure_single_first_refine(triangulation_first_refine),
    
	computing_timer  (std::cout, TimerOutput::summary, TimerOutput::cpu_times)
  {
    cout << "================== \n" 
              << "  MixedLaplaceProblem_Complex\n"
              << "  dim: " << dim << "\n"
              << "  n_q_points is " << n_q_points << "\n"
              << "==================\n" ;      
  }

template <int dim>
void MixedLaplaceProblem_Complex_Simp_Coe<dim>::make_grid_and_dofs ()
{
  GridGenerator::hyper_cube (triangulation, 0, this->length);
  
//     triangulation.begin_active()->face(1)->set_boundary_id(0);		//set the right-hand boundary indicator to 0
  
  print_tria_info (triangulation);
  
  
  triangulation.refine_global (refine);
    
  GridGenerator::hyper_cube (triangulation_first_refine, 0, this->length);
  triangulation_first_refine.refine_global (refine);       
    
  dof_handler_single_first_refine.distribute_dofs(fe_single);
  DoFRenumbering::component_wise (dof_handler_single_first_refine);
  
  dof_handler_pressure_single_first_refine.distribute_dofs(fe_dgq);
  DoFRenumbering::component_wise (dof_handler_pressure_single_first_refine);
    
  dof_handler_velocity_single_first_refine.distribute_dofs(fe_q);
  DoFRenumbering::component_wise (dof_handler_velocity_single_first_refine);
  
  cout << "dof_handler_single_first_refine: \n";
  print_dofs_info(dof_handler_single_first_refine);
  cout << "dof_handler_pressure_single_first_refine: \n";
  print_dofs_info(dof_handler_pressure_single_first_refine);
  cout << "dof_handler_velocity_single_first_refine: \n";
  print_dofs_info(dof_handler_velocity_single_first_refine);
}
  
template <int dim>
void MixedLaplaceProblem_Complex_Simp_Coe<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_handler_single.distribute_dofs (fe_single);
    
  dof_handler_velocity_single.distribute_dofs (fe_q);
  dof_handler_pressure_single.distribute_dofs (fe_dgq);
    
//========================================Below are extracted from step-22==========================================
    
  std::vector<unsigned int> block_component (4,0);
  block_component[0] = 0;
  block_component[1] = 1;
  block_component[2] = 2;
  block_component[3] = 3;
    
  DoFRenumbering::component_wise (dof_handler, block_component);  
  DoFRenumbering::component_wise (dof_handler_pressure_single);
    
  if(is_constraints==1)
  {
    cout << "  using constraints for Neumann boundary conditions\n";
    {
      constraints.clear ();
      FEValuesExtractors::Vector velocity_real(0); 
      FEValuesExtractors::Vector pressure_real(1);   
      FEValuesExtractors::Vector velocity_imag(2); 
      FEValuesExtractors::Vector pressure_imag(3); 
    
      DoFTools::make_hanging_node_constraints (dof_handler,
                                               constraints);
    
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                GradientBoundary_Step_20_Complex<dim>(id_case),          // here the values in GradientBoundary_Step_20_Complex are taken as the negative gradient
	                                            constraints,
	                                            fe.component_mask(velocity_real)
                                               );
    
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                GradientBoundary_Step_20_Complex<dim>(id_case),
                                                constraints,
	                                            fe.component_mask(velocity_imag)
                                               );
    }
    constraints.close ();   
//     print_constraints_info(constraints, dof_handler);    
  }else 
  {
    cout << "  not using constraints\n";
  }
 

//======================================== Above are extracted from step-22 ==========================================
   
    DoFRenumbering::component_wise (dof_handler);
    DoFRenumbering::component_wise (dof_handler_single); 

    std::vector<types::global_dof_index> dofs_per_component_real (dim+1);
    DoFTools::count_dofs_per_component (dof_handler_single, dofs_per_component_real);
    
    std::vector<types::global_dof_index> dofs_per_component (2*(dim+1));
        
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    
//     std::cout << "dofs_per_component[0] is: " << dofs_per_component[0] << std::endl;
//     std::cout << "dofs_per_component[1] is: " << dofs_per_component[1] << std::endl;
//     std::cout << "dofs_per_component[2] is: " << dofs_per_component[2] << std::endl;
//     std::cout << "dofs_per_component[3] is: " << dofs_per_component[3] << std::endl;
                       
//     std::cout << "dofs_per_component_real[0] is: " << dofs_per_component_real[0] << std::endl;
//     std::cout << "dofs_per_component_real[1] is: " << dofs_per_component_real[1] << std::endl;
    
    n_u = dofs_per_component[0],
    n_p = dofs_per_component[1];
                       
//     std::cout << "n_u is: " << n_u << ", n_p is: " << n_p << std::endl;             //this could be taken as the number of degrees of freedom of u and p for the real/imag part

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
    
    
    BlockDynamicSparsityPattern dsp_real(2, 2);
    dsp_real.block(0, 0).reinit (n_u, n_u);
    dsp_real.block(1, 0).reinit (n_p, n_u);
    dsp_real.block(0, 1).reinit (n_u, n_p);
    dsp_real.block(1, 1).reinit (n_p, n_p);
    dsp_real.collect_sizes ();
    DoFTools::make_sparsity_pattern (dof_handler_single, dsp_real);
    
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
  
  }

template <int dim>
void MixedLaplaceProblem_Complex_Simp_Coe<dim>::assemble_system ()
{      
  cout << "Assembling\n";
  TimerOutput::Scope t(computing_timer, "assemble_system");
  
  QGauss<dim-1> face_quadrature_formula(degree+2);

  double T1r = 1.0, T1i = 1.0;
  double T2r = 1.0, T2i = 1.0;
  double T3 = -2.0;
  
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    | update_gradients |
                           update_quadrature_points  | update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values    | update_normal_vectors |
                                    update_quadrature_points  | update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const PressureBoundaryValues_Step_20_Complex<dim> pressure_boundary_values(id_case);

  std::vector<double> rhs_values (n_q_points);
  std::vector<double> boundary_values (n_face_q_points);

  const FEValuesExtractors::Vector ur (0);
  const FEValuesExtractors::Scalar pr (dim);
  const FEValuesExtractors::Vector ui (dim+1);
  const FEValuesExtractors::Scalar my_pi (dim+dim+1);
    
  cout << "  coeff_a: " << pressure_boundary_values.coeff_a << "\n";
  cout << "  coeff_b: " << pressure_boundary_values.coeff_b << "\n";
    
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
//         std::cout << "    Cell no.: " << cell->active_cell_index() << std::endl;
    fe_values.reinit (cell);
    local_matrix = 0;
    local_rhs = 0;
    
    for (unsigned int q=0; q</*1*/n_q_points; ++q)
    {
      for (unsigned int i=0; i</*1*/dofs_per_cell; ++i)
      {
        const Tensor<1,dim> phi_i_ur     = fe_values[ur].value (i, q);
        const double        div_phi_i_ur = fe_values[ur].divergence (i, q);
        const double        phi_i_pr     = fe_values[pr].value (i, q);
        
        const Tensor<1,dim> phi_i_ui     = fe_values[ui].value (i, q);
        const double        div_phi_i_ui = fe_values[ui].divergence (i, q);             
        const double        phi_i_pi     = fe_values[my_pi].value (i, q);
        

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          const Tensor<1,dim> phi_j_ur     = fe_values[ur].value (j, q);
          const double        div_phi_j_ur = fe_values[ur].divergence (j, q);
          const double        phi_j_pr     = fe_values[pr].value (j, q);
        
          const Tensor<1,dim> phi_j_ui     = fe_values[ui].value (j, q);
          const double        div_phi_j_ui = fe_values[ui].divergence (j, q);             
          const double        phi_j_pi     = fe_values[my_pi].value (j, q);
            
          if (fe.system_to_component_index(i).first == 0)
          {
            if(fe.system_to_component_index(j).first == 0)
            {
                local_matrix(i,j) += (phi_i_ur[0] * phi_j_ur[0]) * fe_values.JxW(q);                                                           //1
            }else if(fe.system_to_component_index(j).first == 1)
            {
              local_matrix(i,j) += (- div_phi_i_ur * phi_j_pr) * fe_values.JxW(q);                                                         //2
            }
          }else if(fe.system_to_component_index(i).first == 1)
          {
            if(fe.system_to_component_index(j).first == 0)
            {
              local_matrix(i,j) += (- T1r * phi_i_pr * div_phi_j_ur + T2r * phi_i_pr * phi_j_ur[0] ) * fe_values.JxW(q);                   //3
            }else
            if(fe.system_to_component_index(j).first == 1)
            {
              local_matrix(i,j) += (- T3 * phi_i_pr * phi_j_pr) * fe_values.JxW(q);                                                    //4
            }else if(fe.system_to_component_index(j).first == 2)
            {
              local_matrix(i,j) += (T1i * phi_i_pr * div_phi_j_ui - T2i * phi_i_pr * phi_j_ui[0] ) * fe_values.JxW(q);           //5
            }
          }else if(fe.system_to_component_index(i).first == 2)
          {
            if(fe.system_to_component_index(j).first == 2)
            {
              local_matrix(i,j) += (phi_i_ui[0] * phi_j_ui[0]) * fe_values.JxW(q);                                                     //6
            }else if(fe.system_to_component_index(j).first == 3)
            {
              local_matrix(i,j) += (- div_phi_i_ui * phi_j_pi) * fe_values.JxW(q);                                                 //7
            }
          }else if(fe.system_to_component_index(i).first == 3)
          {
            if(fe.system_to_component_index(j).first == 0)
            {
              local_matrix(i,j) += ( -T1i * phi_i_pi * div_phi_j_ur + T2i * phi_i_pi * phi_j_ur[0] ) * fe_values.JxW(q);           //8
            }else if(fe.system_to_component_index(j).first == 2)
            {
              local_matrix(i,j) += (- T1r * phi_i_pi * div_phi_j_ui + T2r * phi_i_pi * phi_j_ui[0] ) * fe_values.JxW(q);       //9
            }else if(fe.system_to_component_index(j).first == 3)
            {
              local_matrix(i,j) += (- T3 * phi_i_pi * phi_j_pi) * fe_values.JxW(q);                                        //10
            }
          }                      
        }
        local_rhs(i) += -phi_i_ur[0] *
                        rhs_values[q] *
                        fe_values.JxW(q);                
      }
    }
    
    for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
    {
      if (cell->at_boundary(face_n) && (cell->face(face_n)->boundary_id() == 0))       //deal with the contribution of the pressure BC to the rhs in the formulation,                                                 
                                                                                                    //here it goes to the first test function of ur, i.e. the first row of the rhs
      {
        fe_face_values.reinit (cell, face_n);

        pressure_boundary_values                                                    // assign boundary values at quadrature points
        .value_list (fe_face_values.get_quadrature_points(),
                    boundary_values);

        for (unsigned int q=0; q<n_face_q_points; ++q)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            local_rhs(i) += -(fe_face_values[ur].value (i, q) *                     // to approximate a line integral
                                fe_face_values.normal_vector(q) *
                                boundary_values[q] *
                                fe_face_values.JxW(q));
        }
      }
    }
        
    cell->get_dof_indices (local_dof_indices);
    
    if(is_constraints==1) // dealing with the global matrix and right hand side and implement (inhomogeneous) essential boundary conditions, which are the velocity boundary condition here
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)                              // extracted from step-22
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          constraints.condense (system_matrix, system_rhs);
          constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                    local_dof_indices,
                                                    system_matrix, system_rhs);
        }      
      }
    }else if(is_constraints==0)                                     // using normal assembling and then eliminating the essential BCs
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
    
  if(is_constraints==0)                           // dealing with the velocity boundary condition mannualy when not using constraints, only applicable for 1d
  {
    if (is_matrices_before_diri_BC_printed==1)
    {
      std::cout << "system_matrix before applying the velocity boundary conditions is " << std::endl;
      system_matrix.print_formatted(std::cout);
      std::cout << '\n';    
      std::cout << "system_rhs before applying the velocity boundary conditions is " << std::endl;
      system_rhs.print(std::cout);     
      std::cout << '\n';
    }
    
    unsigned int ur_row_last_dof = 1+triangulation.n_active_cells()*(degree)-(degree);             
    unsigned int ui_row_last_dof = dof_handler.n_dofs()/2+1+triangulation.n_active_cells()*(degree)-(degree);
//     std::cout << "    ur_row_last_dof is " << ur_row_last_dof << ", ui_row_last_dof is " << ui_row_last_dof << std::endl;
//     ur_row_last_dof and ui_row_last_dof denote the row number of the last degree of freedom of ur and ui, respectively   
  
    for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)                   // influence of the latter elimination of columns on the rhs
    {
        if(system_matrix.el(i,ur_row_last_dof)!=0)
        {
//             std::cout << "    at row " << i << ", system_matrix.el(i,ur_row_last_dof) reads " << system_matrix.el(i,ur_row_last_dof);
//             if(i!=ur_row_last_dof )                                           
//             {   
//                 std::cout << "; its product with the velocity boundary " << system_matrix.el(i,ur_row_last_dof)*pressure_boundary_values.coeff_b << " goes to the rhs" << std::endl;        
//             }else 
//             {
//                 std::cout << std::endl;
//             }
            
            if(i!=ur_row_last_dof)
            {   
                system_rhs(i)-=system_matrix.el(i,ur_row_last_dof)*pressure_boundary_values.coeff_b;                    // a better choice is       
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
    
    system_rhs(ur_row_last_dof) = pressure_boundary_values.coeff_b;       //this->coeff_b       // satisfying the gradient boundary conditions manually
    system_rhs(ui_row_last_dof) = 0.0;         
  }
    
  if(is_matrices_before_diri_BC_printed==1)
  {
    std::cout << "system_matrix after applying velocity boundary conditions is " << std::endl;
    system_matrix.print_formatted(std::cout);
    std::cout << '\n';    
    std::cout << "system_rhs after applying velocity boundary conditions is " << std::endl;
    system_rhs.print(std::cout);     
    std::cout << '\n'; 
  }
}

template <int dim>
void MixedLaplaceProblem_Complex_Simp_Coe<dim>::solve ()
{
  
  if(is_UMFPACK_mono==1)                    
  {
    cout << "UMFPack solver for monolithic\n";
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);  
  }else if (is_UMFPACK_mono==0)                                 // copied from real-valued problems
  {
    cout << "Schur complement\n";
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
  
  if (is_constraints==1)
  {
    constraints.distribute (solution);    
  }
        
  solution_velocity_real = solution.block(0);
  solution_pressure_real = solution.block(1);
  solution_velocity_imag = solution.block(2);
  solution_pressure_imag = solution.block(3);
        
  for (unsigned int r = 0; r < n_p; r++)                  // computing the amplitude
  {
    solution_amp[r] = std::sqrt(solution_pressure_real[r]*solution_pressure_real[r] + solution_pressure_imag[r]*solution_pressure_imag[r]);
  }
        
  for (unsigned int i = 0; i < n_u; i++)                                                      // select the gradient and solution distinguished by the real and imaginary part
  {
    solution_real[i]=solution_velocity_real[i];
    solution_imag[i]=solution_velocity_imag[i];
  }
    
  for (unsigned int i = 0; i < n_p; i++)
  {
    solution_real[n_u+i]=solution_pressure_real[i];
    solution_imag[n_u+i]=solution_pressure_imag[i];
  }
}

template <int dim>
void MixedLaplaceProblem_Complex_Simp_Coe<dim>::compute_errors ()
{
  const ComponentSelectFunction<dim>
  pressure_mask (dim, dim+1);
  const ComponentSelectFunction<dim>
  velocity_mask(0, dim+1);

  ExactSolution_Step_20_Complex_Real<dim> exact_solution_real(id_case);
  ExactSolution_Step_20_Complex_Imag<dim> exact_solution_imag(id_case);
    
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
    
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    cellwise_errors[i] = pow(cellwise_errors_real[i], 2)+pow(cellwise_errors_imag[i], 2);
    cellwise_errors[i] = pow(cellwise_errors[i], 0.5);                                               // we get the square root here to use the function l2_norm() later
  }
   
  pressure_inte_L2 = cellwise_errors.l2_norm();
   
  VectorTools::integrate_difference (dof_handler_single, solution_real, exact_solution_real,               
                                     cellwise_errors_real, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_imag, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
    
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    cellwise_errors[i] = pow(cellwise_errors_real[i], 2)+pow(cellwise_errors_imag[i], 2);
    cellwise_errors[i] = pow(cellwise_errors[i], 0.5);                                               
  }
   
  pressure_abs_error_L2 = cellwise_errors.l2_norm();
  pressure_rel_error_L2 = pressure_abs_error_L2/pressure_inte_L2;
  
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_real,                         // the L2_norm concerning the gradient
                                     cellwise_errors_real, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
    
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    cellwise_errors[i] = pow(cellwise_errors_real[i], 2)+pow(cellwise_errors_imag[i], 2);
    cellwise_errors[i] = pow(cellwise_errors[i], 0.5);                                               
  }
   
  velocity_inte_L2 = cellwise_errors.l2_norm();
   
  VectorTools::integrate_difference (dof_handler_single, solution_real, exact_solution_real,               
                                     cellwise_errors_real, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_imag, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
    
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    cellwise_errors[i] = pow(cellwise_errors_real[i], 2)+pow(cellwise_errors_imag[i], 2);
    cellwise_errors[i] = pow(cellwise_errors[i], 0.5);                                               
  }
   
  velocity_abs_error_L2 = cellwise_errors.l2_norm();
  velocity_rel_error_L2 = velocity_abs_error_L2/velocity_inte_L2;
   
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_real,                         // the H2_seminorm concerning the gradient          
                                     cellwise_errors_real, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_0, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    cellwise_errors[i] = pow(cellwise_errors_real[i], 2)+pow(cellwise_errors_imag[i], 2);
    cellwise_errors[i] = pow(cellwise_errors[i], 0.5);                                               
  }
   
  velocity_inte_H1_semi = cellwise_errors.l2_norm();
   
  VectorTools::integrate_difference (dof_handler_single, solution_real, exact_solution_real,               
                                     cellwise_errors_real, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  VectorTools::integrate_difference (dof_handler_single, solution_imag, exact_solution_imag,               
                                     cellwise_errors_imag, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
    
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    cellwise_errors[i] = pow(cellwise_errors_real[i], 2)+pow(cellwise_errors_imag[i], 2);
    cellwise_errors[i] = pow(cellwise_errors[i], 0.5);                                               
  }
   
  velocity_abs_error_H1_semi = cellwise_errors.l2_norm();
  velocity_rel_error_H1_semi = velocity_abs_error_L2/velocity_inte_H1_semi;
   
  std::cout << "  pressure_abs_error_L2 = " << pressure_abs_error_L2 << ", pressure_inte_L2 = " << pressure_inte_L2 << ", pressure_rel_error_L2 = " << pressure_rel_error_L2 << std::endl; 
  std::cout << "  velocity_abs_error_L2 = " << velocity_abs_error_L2 << ", velocity_inte_L2 = " << velocity_inte_L2 << ", velocity_rel_error_L2 = " << velocity_rel_error_L2 << std::endl;
  std::cout << "  velocity_abs_error_H1_semi = " << velocity_abs_error_H1_semi << ", velocity_inte_H1_semi = " << velocity_inte_H1_semi  << ", velocity_rel_error_H1_semi = " << velocity_rel_error_H1_semi << std::endl;
}

  
template <int dim>
void MixedLaplaceProblem_Complex_Simp_Coe<dim>::compute_errors_quad ()                      
{
    unsigned int n_q_points = degree+2; 
    QGauss<dim>  quadrature_formula(n_q_points);
        
// ======================= obtain the pressure at quadrature points ====================
    
    std::vector<double> solution_values_cell_quads(n_q_points);
    std::vector<double> coords_all_quads;
    std::vector<double> solution_values_all_quads_first_refine;    

//     std::cout << "   solution_pressure_real is "; 
//     solution_pressure_real.print(std::cout);
    
    FEValues<dim> fe_values_pressure (fe_dgq, quadrature_formula,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);

    typename DoFHandler<dim>::active_cell_iterator  
    cell = dof_handler_pressure_single.begin_active(),
    endc = dof_handler_pressure_single.end();
    
    for (; cell!=endc; ++cell)
    {
//         std::cout << "cell " << cell->active_cell_index() << ", vertex(0): " << cell->vertex(0) ;
        
        fe_values_pressure.reinit (cell);
        
//         std::cout << ", quad coords: ";
        for (unsigned int i=0; i<fe_values_pressure.get_quadrature_points().size();++i)
        {
//             std::cout << fe_values_pressure.get_quadrature_points()[i](0) << " ";
            coords_all_quads.push_back(fe_values_pressure.get_quadrature_points()[i](0));
        }
        
//         std::cout << ", quad coords: ";
//         for (unsigned int i=0; i<n_q_points;++i)
//         {
//             std::cout << fe_values_velocity.get_quadrature_points()[i](0) << " ";
//         }
        
/*        std::cout << ", J*W: ";
        for (unsigned int i=0; i<fe_values_pressure.get_JxW_values ().size();++i)
        {
            std::cout << fe_values_pressure.get_JxW_values()[i] << " ";
        } */ 
//         std::cout << std::endl;        
        
        fe_values_pressure.get_function_values (solution_pressure_real, solution_values_cell_quads);      
//         std::cout << ", solution values at quad coords: ";
        for (unsigned int i=0; i<n_q_points;++i)
        {
            solution_values_all_quads_first_refine.push_back(solution_values_cell_quads[i]);
//             std::cout << solution_values_cell_quads[i] << " ";
        }
//         std::cout << std::endl; 

    }
        
//         std::ofstream coords_quad;
//         coords_quad.open("coords_quad_deg_"+std::to_string(degree)+"_ref_"+std::to_string(refine)+".txt",std::ofstream::app);              // save data into files
//         for (double n : coords_all_quads)
//         {
//             coords_quad << std::setprecision(20) << n << "\n";
//         }
//         coords_quad.close();    
//         
//         std::ofstream solution_quad;
//         solution_quad.open("solution_pressure_real_quad_deg_"+std::to_string(degree-1)+"_ref_"+std::to_string(refine)+".txt",std::ofstream::app);
//         for (double n : solution_values_all_quads)
//         {
//             solution_quad << std::setprecision(20) << n << "\n";
//         }
//         solution_quad.close();
        
// ======================= obtain the velocity at quadrature points ====================
    
    std::vector<double> solution_gradients_cell_quads(n_q_points);
    std::vector<double> solution_gradients_all_quads_first_refine;    
    
    FEValues<dim> fe_values_velocity (fe_q, quadrature_formula,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);

    cell = dof_handler_velocity_single.begin_active();
    endc = dof_handler_velocity_single.end();
    
    for (; cell!=endc; ++cell)
    {
        fe_values_velocity.reinit (cell);
        
        fe_values_velocity.get_function_values (solution_velocity_real, solution_gradients_cell_quads);      
        for (unsigned int i=0; i<n_q_points;++i)
        {
            solution_gradients_all_quads_first_refine.push_back(solution_gradients_cell_quads[i]);
        }

    }
//         std::ofstream solution_gradients_quad;
//         solution_gradients_quad.open("solution_velocity_real_quad_deg_"+std::to_string(degree)+"_ref_"+std::to_string(refine)+".txt",std::ofstream::app);
//         for (double n : solution_gradients_all_quads_first_refine)
//         {
//             solution_gradients_quad << std::setprecision(20) << n << "\n";
//         }
//         solution_gradients_quad.close();

// ======================= obtain the second-order derivative at quadrature points ====================
    
    std::vector<Tensor<1, dim>> solution_hessians_cell_quads(n_q_points);
    std::vector<Tensor<1, dim>> solution_hessians_all_quads_first_refine;    
    
    cell = dof_handler_velocity_single.begin_active();
    
    for (; cell!=endc; ++cell)
    {
        fe_values_velocity.reinit (cell);
        fe_values_velocity.get_function_gradients (solution_velocity_real, solution_hessians_cell_quads);      
        for (unsigned int i=0; i<n_q_points;++i)
        {
            solution_hessians_all_quads_first_refine.push_back(solution_hessians_cell_quads[i]);
        }
    }
    
//         std::ofstream solution_hessians_quad;
//         solution_hessians_quad.open("solution_hessians_real_quad_deg_"+std::to_string(degree)+"_ref_"+std::to_string(refine)+".txt",std::ofstream::app);
//         for (double n : solution_hessians_all_quads_first_refine)
//         {
//             solution_hessians_quad << std::setprecision(20) << n << "\n";
//         }
//         solution_hessians_quad.close();

// ===========================================


    std::cout << "   ~~~~~~~~~~ second refinement " << std::endl;
    
    std::vector<double> coords_all_quads_second_refine;
    std::vector<double> solution_values_all_quads_second_refine;   
    std::vector<double> solution_gradients_all_quads_second_refine;    
    std::vector<Tensor<1,dim>> solution_hessians_all_quads_second_refine;
        
    std::cout << "   computing the finer solution ... " << std::endl;

    triangulation.refine_global ();  
    setup_system ();
    assemble_system ();
    solve ();
    
//     std::cout << "   solution_pressure_real_second_refine is "; 
//     solution_pressure_real.print(std::cout);    
//     
//     std::cout << "   solution_velocity_real_second_refine is "; 
//     solution_velocity_real.print(std::cout);    
//     
//     std::cout << std::endl;
    
    std::cout << "   obtaining the pressure at quadrature points " << std::endl;
    
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ on the left quadrature points

    unsigned int n_q_points_left = (n_q_points+1)/2.0;
    std::vector<double > solution_values_left_cell_quads_second_refine(n_q_points_left);
    std::vector<Point<dim> > left_quad_vector(n_q_points_left);
    for (unsigned int i=0; i<left_quad_vector.size();++i)
    {
        left_quad_vector[i]=quadrature_formula.get_points()[i]*2;
//         std::cout << left_quad_vector[i] << " ";
    }
//     std::cout << std::endl;
    Quadrature<dim>  quadrature_formula_high_2_low_left(left_quad_vector);                   // fe_values_pressure.get_quadrature_points()[2](0)  left_quad_vector       
    FEValues<dim> fe_values_pressure_high_2_low_left (fe_dgq, quadrature_formula_high_2_low_left,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ on the right quadrature points

    unsigned int n_q_points_right = (n_q_points+1)/2.0;
    std::vector<double > solution_values_right_cell_quads_second_refine(n_q_points_right);
    std::vector<Point<dim> > right_quad_vector(n_q_points_right);
    for (unsigned int i=0; i<right_quad_vector.size();++i)
    {
        right_quad_vector[i](0) = (quadrature_formula.get_points()[i+(n_q_points_left*2==n_q_points?n_q_points_left:n_q_points_left-1)](0)-0.5)*2;
//         std::cout << right_quad_vector[i] << " ";
    }
//     std::cout << std::endl;
    Quadrature<dim>  quadrature_formula_high_2_low_right(right_quad_vector);                  
    FEValues<dim> fe_values_pressure_high_2_low_right (fe_dgq, quadrature_formula_high_2_low_right,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);   
    
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ loop over left and right quadrature points
    
//     typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_pressure_single.begin_active();
    endc = dof_handler_pressure_single.end();    
    for (; cell!=endc; ++cell)
    {
        
//         std::cout << "cell " << cell->active_cell_index() << ", vertex(0): " << cell->vertex(0) ;
        if (cell->active_cell_index()%2==0)
        {
            fe_values_pressure_high_2_low_left.reinit (cell);
            
//             std::cout << ", coords of left quandrature points: ";
//             for (unsigned int i=0; i<fe_values_pressure_high_2_low_left.get_quadrature_points().size();++i)
//             {
//                 std::cout << fe_values_pressure_high_2_low_left.get_quadrature_points()[i] << " ";                           //<< std::setprecision(10) 
//             }
//             
            fe_values_pressure_high_2_low_left.get_function_values (solution_pressure_real, solution_values_left_cell_quads_second_refine);
//             
//             std::cout << ", solution at left quad coords: ";
            for (unsigned int i=0; i<n_q_points_left;++i)
            {
                solution_values_all_quads_second_refine.push_back(solution_values_left_cell_quads_second_refine[i]);
//                 std::cout << solution_values_left_cell_quads_second_refine[i] << " ";
            }  
//             std::cout << std::endl;
        }else
        {
            fe_values_pressure_high_2_low_right.reinit (cell);
            
//             std::cout << ", coords of right quandrature points: ";
//             for (unsigned int i=0; i<fe_values_pressure_high_2_low_right.get_quadrature_points().size();++i)
//             {
//                 std::cout << fe_values_pressure_high_2_low_right.get_quadrature_points()[i] << " ";                           //<< std::setprecision(10) 
//             }
//             
            fe_values_pressure_high_2_low_right.get_function_values (solution_pressure_real, solution_values_right_cell_quads_second_refine);
            
//             std::cout << ", solution at right quad coords: ";
//             for (unsigned int i=0; i<n_q_points_right;++i)
//             {
//                 std::cout << solution_values_right_cell_quads_second_refine[i]  << " ";
//             }   
//             std::cout << std::endl;
            
            if (2*n_q_points_right==n_q_points)
            {
                for (unsigned int i=0; i<n_q_points_right;++i)
                {
                    solution_values_all_quads_second_refine.push_back(solution_values_right_cell_quads_second_refine[i]);
                }  
            }else
            {
                for (unsigned int i=1; i<n_q_points_right;++i)
                {
                    solution_values_all_quads_second_refine.push_back(solution_values_right_cell_quads_second_refine[i]);
                }  
            }
        }
    }
    
/*    std::cout << "  solution of the second refinement at all quad points reads: " << std::endl;
    for(double n : solution_values_all_quads_second_refine) 
    {
        std::cout << n << ' ';
    }    
    std::cout << std::endl;  */   

    std::cout << "   obtaining the gradient and second-order derivative at quadrature points " << std::endl;

    FEValues<dim> fe_values_velocity_high_2_low_left (fe_q, quadrature_formula_high_2_low_left,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);    
    
    std::vector<double> solution_gradients_left_cell_quads_second_refine(n_q_points_left);
    std::vector<Tensor<1, dim>> solution_hessians_left_cell_quads_second_refine(n_q_points_left);
        
    FEValues<dim> fe_values_velocity_high_2_low_right (fe_q, quadrature_formula_high_2_low_right,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);   
    std::vector<double> solution_gradients_right_cell_quads_second_refine(n_q_points_right);
    std::vector<Tensor<1, dim>> solution_hessians_right_cell_quads_second_refine(n_q_points_right);
         

    cell = dof_handler_velocity_single.begin_active();
    endc = dof_handler_velocity_single.end();    
    for (; cell!=endc; ++cell)
    {
        
//         std::cout << "    cell " << cell->active_cell_index() << ", vertex(0): " << cell->vertex(0) ;
        
        fe_values_velocity.reinit(cell);
        
//         std::cout << ", quad coords: ";
//         for (unsigned int i=0; i<n_q_points;++i)
//         {
//             std::cout << fe_values_velocity.get_quadrature_points()[i](0) << " ";
//         }
        
        for (unsigned int i=0; i<n_q_points;++i)
        {
            coords_all_quads_second_refine.push_back(fe_values_velocity.get_quadrature_points()[i](0));
        }        
        
        if (cell->active_cell_index()%2==0)
        {
            fe_values_velocity_high_2_low_left.reinit (cell);
            
//             std::cout << ", coords of left quandrature points of the previous refinement: ";
//             for (unsigned int i=0; i<n_q_points_left;++i)
//             {
//                 std::cout << fe_values_velocity_high_2_low_left.get_quadrature_points()[i] << " ";                           //<< std::setprecision(10) 
//             }
//             
            fe_values_velocity_high_2_low_left.get_function_values (solution_velocity_real, solution_gradients_left_cell_quads_second_refine);
//             
//             std::cout << ", gradients at left quad coords: ";
//             for (unsigned int i=0; i<n_q_points_left;++i)
//             {
//                 std::cout << solution_gradients_left_cell_quads_second_refine[i] << " ";
//             }  

            
            for (unsigned int i=0; i<n_q_points_left;++i)
            {
                solution_gradients_all_quads_second_refine.push_back(solution_gradients_left_cell_quads_second_refine[i]);
            }
            
    
            fe_values_velocity_high_2_low_left.get_function_gradients (solution_velocity_real, solution_hessians_left_cell_quads_second_refine);
//             std::cout << ", 2ndds at left quad coords: ";
//             for (unsigned int i=0; i<n_q_points_left;++i)
//             {
//                 std::cout << solution_hessians_left_cell_quads_second_refine[i] << " ";
//             }  
            
            for (unsigned int i=0; i<n_q_points_left;++i)
            {
                solution_hessians_all_quads_second_refine.push_back(solution_hessians_left_cell_quads_second_refine[i]);
            }             
            
//             std::cout << std::endl;            
        }else
        {
            fe_values_velocity_high_2_low_right.reinit (cell);
            
//             std::cout << ", coords of right quandrature points of the previous refinement: ";
//             for (unsigned int i=0; i<n_q_points_right;++i)
//             {
//                 std::cout << fe_values_velocity_high_2_low_right.get_quadrature_points()[i] << " ";                           //<< std::setprecision(10) 
//             }
//             
            fe_values_velocity_high_2_low_right.get_function_values (solution_velocity_real, solution_gradients_right_cell_quads_second_refine);
            fe_values_velocity_high_2_low_right.get_function_gradients (solution_velocity_real, solution_hessians_right_cell_quads_second_refine);
            
//             std::cout << ", gradients at right quad coords: ";
//             for (unsigned int i=0; i<n_q_points_right;++i)
//             {
//                 std::cout << solution_gradients_right_cell_quads_second_refine[i] << " ";
//             } 
            
/*            std::cout << ", 2ndds at right quad coords: ";
            for (unsigned int i=0; i<n_q_points_right;++i)
            {
                std::cout << solution_hessians_right_cell_quads_second_refine[i] << " ";
            } */           
            
            if (2*n_q_points_right==n_q_points)
            {
                for (unsigned int i=0; i<n_q_points_right;++i)
                {
                    solution_gradients_all_quads_second_refine.push_back(solution_gradients_right_cell_quads_second_refine[i]);
                    solution_hessians_all_quads_second_refine.push_back(solution_hessians_right_cell_quads_second_refine[i]);
                }  
            }else
            {
                for (unsigned int i=1; i<n_q_points_right;++i)
                {
                    solution_gradients_all_quads_second_refine.push_back(solution_gradients_right_cell_quads_second_refine[i]);
                    solution_hessians_all_quads_second_refine.push_back(solution_hessians_right_cell_quads_second_refine[i]);
                } 
            }
            
 
//             std::cout << std::endl;

        }
    }
    
/*    std::cout << "  solution gradients of the second refinement at all quad points reads: " << std::endl;
    for(double n : solution_gradients_all_quads_second_refine) 
    {
        std::cout << n << ' ';
    }    
    std::cout << std::endl; */  

//     std::cout << "    solution 2ndds of the second refinement at all quad points reads: ";
//     for(Tensor<1,dim> n : solution_hessians_all_quads_second_refine) 
//     {
//         std::cout << n[0] << ' ';
//     }    
//     std::cout << std::endl; 
    
//         std::ofstream coords_quad_first_refine;
//         coords_quad_first_refine.open("coords_quad_deg_"+std::to_string(degree)+"_ref_"+std::to_string(refine)+"_2nd_ref.txt",std::ofstream::app);
//         for (double n : coords_all_quads_second_refine)
//         {
//             coords_quad_first_refine << std::setprecision(20) << n << "\n";
//         }
//         coords_quad_first_refine.close();   
    
/*        std::ofstream solution_velocity_real_quad_second_refine;
        solution_velocity_real_quad_second_refine.open("solution_velocity_real_quad_deg_"+std::to_string(degree)+"_ref_"+std::to_string(refine)+"_2nd_ref.txt",std::ofstream::app);
        for (double n : solution_gradients_all_quads_second_refine)
        {
            solution_velocity_real_quad_second_refine << std::setprecision(20) << n << "\n";
        }
        solution_velocity_real_quad_second_refine.close();   */ 
        
// ========================================== computing the error 
        
    std::cout << "   ~~~~~~~~~~ start computing the error " << std::endl;

    Vector<double> cellwise_error(triangulation_first_refine.n_active_cells());

    std::cout << "    for the pressure " << std::endl;    
    
    double error_quad_pressure = 0.0;
        
    cell = dof_handler_pressure_single_first_refine.begin_active();
    endc = dof_handler_pressure_single_first_refine.end();
    
    for (; cell!=endc; ++cell)
    {
        int cell_index = cell->active_cell_index();
        fe_values_pressure.reinit (cell);
/*        std::cout << "    J*W: ";
        for (unsigned int i=0; i<fe_values_pressure.get_JxW_values ().size();++i)
        {
            std::cout << fe_values_pressure.get_JxW_values()[i] << " ";
        }  
        std::cout << std::endl; */  
        
        error_quad_pressure=0;        
        
//         std::cout << "    solution_values_all_quads_first_refine:  ";
//         for (int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             std::cout << solution_values_all_quads_first_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         std::cout << std::endl;
        
//         std::cout << "    solution_values_all_quads_second_refine:  ";
//         for (int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             std::cout << solution_values_all_quads_second_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         std::cout << std::endl;

        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            error_quad_pressure += std::pow(solution_values_all_quads_first_refine[n_q_points*cell_index + q_index] - solution_values_all_quads_second_refine[n_q_points*cell_index + q_index],2 ) * fe_values_pressure.JxW (q_index) ;   // * 
        }
//         std::cout << std::endl;        
        
//         std::cout << "    cell_index is :  " << cell_index << std::endl;
        cellwise_error[cell_index] = std::sqrt(error_quad_pressure);
    }
    pressure_abs_error_L2_custom = cellwise_error.l2_norm();
    std::cout << "    pressure_abs_error_L2_custom is :  " << pressure_abs_error_L2_custom << std::endl;
    

    std::cout << "    for the velocity " << std::endl; 

    double error_quad_velocity = 0.0;
    
    cell = dof_handler_velocity_single_first_refine.begin_active();
    endc = dof_handler_velocity_single_first_refine.end();

    for (; cell!=endc; ++cell)
    {
        int cell_index = cell->active_cell_index();
        fe_values_velocity.reinit (cell);
//         std::cout << "    J*W: ";
//         for (unsigned int i=0; i<fe_values_velocity.get_JxW_values ().size();++i)
//         {
//             std::cout << fe_values_velocity.get_JxW_values()[i] << " ";
//         }  
//         std::cout << std::endl;     
        
        error_quad_velocity=0;

//         std::cout << "    solution_gradients_all_quads_first_refine:  ";
//         for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             std::cout << solution_gradients_all_quads_first_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         std::cout << std::endl;
//         
//         std::cout << "    solution_gradients_all_quads_second_refine:  ";
//         for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             std::cout << solution_gradients_all_quads_second_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         std::cout << std::endl;
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            error_quad_velocity += std::pow(solution_gradients_all_quads_first_refine[n_q_points*cell_index + q_index] - solution_gradients_all_quads_second_refine[n_q_points*cell_index + q_index],2 ) * fe_values_velocity.JxW (q_index) ;   // * 
        }
//         std::cout << std::endl;        

        cellwise_error[cell_index] = std::sqrt(error_quad_velocity);
    }
    
    velocity_abs_error_L2_custom = cellwise_error.l2_norm();
    std::cout << "    velocity_abs_error_L2_custom is :  " << velocity_abs_error_L2_custom << std::endl;
    

    std::cout << "    for the second-order derivative " << std::endl; 
    
    double error_quad_2ndd = 0.0;
    
    cell = dof_handler_velocity_single_first_refine.begin_active();
    endc = dof_handler_velocity_single_first_refine.end();

    for (; cell!=endc; ++cell)
    {
        int cell_index = cell->active_cell_index();
        fe_values_velocity.reinit (cell);
        error_quad_2ndd=0;

/*        std::cout << "    solution_hessians_all_quads_first_refine:  ";
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            std::cout << solution_hessians_all_quads_first_refine[n_q_points*cell_index + q_index] << " ";
        }
        std::cout << std::endl;
//         
        std::cout << "    solution_hessians_all_quads_second_refine:  ";
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            std::cout << solution_hessians_all_quads_second_refine[n_q_points*cell_index + q_index] << " ";
        }
        std::cout << std::endl;  */     
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
//             error_quad_2ndd += solution_hessians_all_quads_first_refine[n_q_points*cell_index + q_index][0];
            error_quad_2ndd += std::pow(solution_hessians_all_quads_first_refine[n_q_points*cell_index + q_index][0] - solution_hessians_all_quads_second_refine[n_q_points*cell_index + q_index][0],2 ) * fe_values_velocity.JxW (q_index) ;   // * 
        }
//         std::cout << std::endl;        
        cellwise_error[cell_index] = std::sqrt(error_quad_2ndd);
    }
    L2_error_2ndd_custom = cellwise_error.l2_norm();
    std::cout << "    L2_error_2ndd_custom is :  " << L2_error_2ndd_custom << std::endl;
    
}

  template <int dim>
  void MixedLaplaceProblem_Complex_Simp_Coe<dim>::output_results () const
  {
    std::vector<std::string> solution_names(dim, "u");
    solution_names.push_back ("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector (dof_handler_single, solution_real, solution_names, interpretation);

    data_out.build_patches (degree+1);

    std::ofstream output ("solution.vtk");
    data_out.write_vtk (output);
  }


  template <int dim>
  void MixedLaplaceProblem_Complex_Simp_Coe<dim>::run ()
  {
    make_grid_and_dofs();
    setup_system ();
    assemble_system ();
       
    solve ();
    compute_errors ();
//     compute_errors_quad ();
    output_results ();

//     
    total_CPU_time += computing_timer.return_total_cpu_time ();
    std::cout << "total_CPU_time is " << total_CPU_time << std::endl;
    
    printToFile();    
  }
  


