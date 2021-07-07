
#ifndef MAINCLASS_STEP_20_COMPLEX_VALIDATION_H
#define MAINCLASS_STEP_20_COMPLEX_VALIDATION_H

#include<2_complex_valued/auxiliaryclass_step_20_complex.h>                 // note we put these two header files below using name space dealii

  template <int dim>
  class MixedLaplaceProblem_Complex_Validation
  {
  public:
    MixedLaplaceProblem_Complex_Validation (const unsigned int id_case, const double length, const unsigned int degree, const int grid_parm);
    
    void run ();

    double inte_solu_real, inte_solu_imag;
    double inte_grad_real, inte_grad_imag;
    
    double L2_inte_solu_real, L2_inte_solu_imag, L2_inte_solu;
    double L2_inte_grad_real, L2_inte_grad_imag, L2_inte_grad;

    double L2_error_solu_real, L2_error_grad_real, L2_error_2ndd_real;
    double L2_error_solu_imag, L2_error_grad_imag, L2_error_2ndd_imag;
    double L2_error_solu, L2_error_grad, L2_error_2ndd;

  private:
    void make_grid_and_dofs ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void compute_l2_norms_numerically ();
    void compute_errors_quad ();
    void print_norms_errors_CPU ();
    void output_results () const;

    
    unsigned int is_matrix_after_BC_stored = 0;
    unsigned int is_rhs_after_BC_stored = 0;
    
    const int is_UMFPACK=1;
    
    const unsigned int   id_case;
    const double length;    
    const unsigned int   degree;
    const unsigned int   refine;
    
    string obj_string="";
    
    unsigned int   n_u;
    unsigned int   n_p;
    
    const int n_dofs_custom;
    const int n_vertices_custom;
    
    int n_cells;
    
    unsigned int n_q_points; 
    QGauss<dim>  quadrature_formula;     
    
    int is_global_mesh = 1;           // '1' for the global mesh; '0' for the 'vertex' mesh
    
    Triangulation<dim>   triangulation;
    Triangulation<dim>   triangulation_first_refine;
    
    FE_Q<dim>          fe_q; 
    FE_DGQ<dim>          fe_dgq;     
    FESystem<dim>        fe;
    FESystem<dim>        fe_single;
    
    DoFHandler<dim>      dof_handler;
    DoFHandler<dim>      dof_handler_first_refine;
    
    DoFHandler<dim>      dof_handler_single;
    DoFHandler<dim>      dof_handler_single_first_refine;  
    
    DoFHandler<dim>      dof_handler_pressure_single;
    DoFHandler<dim>      dof_handler_pressure_single_first_refine;
    
    DoFHandler<dim>      dof_handler_velocity_single;
    DoFHandler<dim>      dof_handler_velocity_single_first_refine;
    
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double>       solution;  
    BlockVector<double>       system_rhs;
    
    Vector<double>            solution_grad_real;
    Vector<double>            solution_grad_imag;
    Vector<double>            solution_solu_real;
    Vector<double>            solution_solu_imag;  
    Vector<double>            solution_solu_amp;

    BlockVector<double>            solution_real;
    BlockVector<double>            solution_imag;    
    
    BlockVector<double>       solution_0;
    TimerOutput               computing_timer;
    void printToFile() const;
    double total_CPU_time;

    ConstraintMatrix constraints;

  };

  template <int dim>
  void MixedLaplaceProblem_Complex_Validation<dim>::printToFile() const
  {
      ofstream myfile;
      myfile.open ("data_error_oned_helm_xi_mm.txt", ofstream::out);
      myfile << refine << " ";
      myfile << triangulation_first_refine.n_vertices() <<" ";    
      myfile << dof_handler_pressure_single_first_refine.n_dofs()*2 << " ";  
      myfile << dof_handler_velocity_single_first_refine.n_dofs()*2 << " ";  
      myfile << L2_error_solu << " ";
      myfile << L2_error_grad << " ";
      myfile << L2_error_2ndd << " ";
      
      myfile << total_CPU_time << " ";
      myfile << L2_inte_solu << " ";
      myfile << L2_inte_grad << "\n";                   // the data above are needed for prediction, Sep. 26, 2019

      
//       myfile << n_u << " ";
//       myfile << n_p << " ";      
//       myfile << 0 << " ";
//       myfile << inte_solu_real << " ";
//       myfile << inte_solu_imag << " ";
      
  }


  template <int dim>
  MixedLaplaceProblem_Complex_Validation<dim>::MixedLaplaceProblem_Complex_Validation (const unsigned int id_case, const double length, const unsigned int degree, const int grid_parm)
    :
    id_case(id_case),
    length(length),
    degree (degree),
	refine(grid_parm),
	n_dofs_custom(grid_parm),
    n_vertices_custom((n_dofs_custom/2-1)/degree+1),  // id_scaling_scheme==2 ? (n_dofs_custom/2)/degree+1 : ; we do not differentiate the pressure and the gradient because the difference is only 2
    
    n_q_points(degree+1),
    quadrature_formula(n_q_points),
        
    fe_q (degree),
    fe_dgq (degree-1),
    
	fe (FE_Q<dim>(degree), 1,            //elements for the real part of the gradient
        FE_DGQ<dim>(degree-1), 1,              //elements for the real part of the solution
        FE_Q<dim>(degree), 1,            //elements for the imag part of the gradient
	    FE_DGQ<dim>(degree-1), 1),             //elements for the imag part of the solution
	    
	fe_single (FE_Q<dim>(degree), 1,
             FE_DGQ<dim>(degree-1), 1),	   
   
    dof_handler (triangulation),
    dof_handler_first_refine(triangulation_first_refine),
    
    dof_handler_single (triangulation), 
    dof_handler_single_first_refine (triangulation),

    dof_handler_pressure_single(triangulation),
    dof_handler_pressure_single_first_refine(triangulation_first_refine),
    
    dof_handler_velocity_single(triangulation),
    dof_handler_velocity_single_first_refine(triangulation_first_refine),    
    
	computing_timer  (cout, TimerOutput::summary, TimerOutput::cpu_times)
  {
    
    cout << "================== \n" 
              << "  MixedLaplaceProblem_Complex_Validation\n"
              << "  dim: " << dim << "\n"
              << "  n_q_points is " << n_q_points << "\n"
              << "==================\n" ;
  }


  template <int dim>
  void MixedLaplaceProblem_Complex_Validation<dim>::make_grid_and_dofs ()
  {
    
    if(is_global_mesh==1)
    {
        GridGenerator::hyper_cube (triangulation, 0, length);
    }else if(is_global_mesh == 0)
    {
        
        n_cells = n_vertices_custom-1;
        
        cout << "    n_dofs_custom: " << n_dofs_custom << endl;
        cout << "    n_vertices_custom: " << n_vertices_custom << endl;        
        cout << "    n_cells: " << n_cells << endl; 
                
        
        vector<Point<dim>> vertices(n_vertices_custom);

        const double delta_vertex = 1.0/(n_vertices_custom-1);
        
        for (int i = 0; i < n_vertices_custom; ++i)
        {
            vertices[i](0) = i * delta_vertex;
        }


        vector<array<int,GeometryInfo<1>::vertices_per_cell>> cell_vertices;
        
        array<int,GeometryInfo<1>::vertices_per_cell> array_update;
        
        for (int i = 0; i<n_vertices_custom-1; ++i)
        {
            array_update = {i,i+1};
            cell_vertices.push_back(array_update);
        }
        
        vector<CellData<1>> cells(cell_vertices.size(), CellData<1>());
        for (unsigned int i=0; i<cell_vertices.size(); ++i)
        {
            for (unsigned int j=0; j<GeometryInfo<1>::vertices_per_cell; ++j)
            {
            cells[i].vertices[j] = cell_vertices[i][j];
            }
        }
        
        triangulation.create_triangulation (vertices, cells, SubCellData());  
        
        triangulation_first_refine.create_triangulation (vertices, cells, SubCellData());  
    
    }
    
//   triangulation.begin_active()->face(1)->set_boundary_id(0);    
//   triangulation.last_active()->face(1)->set_boundary_id(0);  

    print_boundary_info(triangulation);

    triangulation.refine_global (refine);
    triangulation_first_refine.copy_triangulation(triangulation);
  
    dof_handler_first_refine.distribute_dofs(fe);
    DoFRenumbering::component_wise (dof_handler_first_refine);
    
    dof_handler_single_first_refine.distribute_dofs(fe_single);
    DoFRenumbering::component_wise (dof_handler_single_first_refine);
    
    dof_handler_pressure_single_first_refine.distribute_dofs(fe_dgq);
    DoFRenumbering::component_wise (dof_handler_pressure_single_first_refine);
    
    dof_handler_velocity_single_first_refine.distribute_dofs(fe_q);
    DoFRenumbering::component_wise (dof_handler_velocity_single_first_refine);
    
    
    obj_string = "dof_handler_first_refine";
    print_dofs_info(obj_string, dof_handler_first_refine);
    obj_string = "dof_handler_single_first_refine";
    print_dofs_info(obj_string, dof_handler_single_first_refine);
    obj_string = "dof_handler_pressure_single_first_refine";
    print_dofs_info(obj_string, dof_handler_pressure_single_first_refine);
    obj_string = "dof_handler_velocity_single_first_refine";
    print_dofs_info(obj_string, dof_handler_velocity_single_first_refine);
        
  }
  
  
  template <int dim>
  void MixedLaplaceProblem_Complex_Validation<dim>::setup_system ()
  {
      
    dof_handler.distribute_dofs (fe);
    dof_handler_single.distribute_dofs (fe_single);
    dof_handler_pressure_single.distribute_dofs (fe_dgq);
    dof_handler_velocity_single.distribute_dofs (fe_q);   
    
//========================================Below are extracted from step-22==========================================
    
    vector<unsigned int> block_component (4,0);
    block_component[0] = 0;
    block_component[1] = 1;
    block_component[2] = 2;
    block_component[3] = 3;
    
    DoFRenumbering::component_wise (dof_handler, block_component);  
    
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
                                              GradientBoundary_Step_20_Complex<dim>(id_case),
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
    
    print_constraints_info(constraints, dof_handler);

//========================================Above are extracted from step-22==========================================
    
    DoFRenumbering::component_wise (dof_handler);
    DoFRenumbering::component_wise (dof_handler_single);
    DoFRenumbering::component_wise (dof_handler_pressure_single);
    DoFRenumbering::component_wise (dof_handler_velocity_single);
    
    vector<types::global_dof_index> dofs_per_component_real (dim+1);
    DoFTools::count_dofs_per_component (dof_handler_single, dofs_per_component_real);
    
    vector<types::global_dof_index> dofs_per_component (2*(dim+1));
        
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    
//     cout << "dofs_per_component[0] is: " << dofs_per_component[0] << endl;
//     cout << "dofs_per_component[1] is: " << dofs_per_component[1] << endl;
//     cout << "dofs_per_component[2] is: " << dofs_per_component[2] << endl;
//     cout << "dofs_per_component[3] is: " << dofs_per_component[3] << endl;
                       
//     cout << "dofs_per_component_real[0] is: " << dofs_per_component_real[0] << endl;
//     cout << "dofs_per_component_real[1] is: " << dofs_per_component_real[1] << endl;
    
    n_u = dofs_per_component[0],
    n_p = dofs_per_component[1];
                       
    cout << "n_u: " << n_u << "\n";
    cout << "n_p: " << n_p << "\n";
              
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
    
    solution_grad_real.reinit (n_u);
    solution_grad_imag.reinit (n_u);    
    solution_solu_real.reinit (n_p);
    solution_solu_imag.reinit (n_p);   
    solution_solu_amp.reinit (n_p); 
    
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
void MixedLaplaceProblem_Complex_Validation<dim>::assemble_system ()
{
  QGauss<dim-1> face_quadrature_formula(degree+2);

  double T_1r = 1.0, T_2r= 1.0;
  double T_3i = 1.0e-2;       // 1.4e-4;
    
  double coord_z_quad;
  double zs = 0.01, zb = 0.01;
  double H=length;    
  
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

  const PressureBoundaryValues_Step_20_Complex<dim> pressure_boundary_values(id_case);
  const GradientBoundary_Step_20_Complex<dim> gradient_boundary_values(id_case);

  Vector<double> boundary_values_for_pressure (2);
  Vector<double> boundary_values_for_gradient (4);

  vector<double> rhs_values (n_q_points);
    
  const FEValuesExtractors::Vector velocity_real (0);
  const FEValuesExtractors::Scalar pressure_real (dim);
  const FEValuesExtractors::Vector velocity_imag (dim+1);
  const FEValuesExtractors::Scalar pressure_imag (dim+dim+1);
    
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    local_matrix = 0;
    local_rhs = 0;

//     cout << "    Cell no.: " << cell->active_cell_index() << endl;  
        
    cell->get_dof_indices (local_dof_indices);
    
    for (unsigned int q=0; q</*1*/n_q_points; ++q)
    {
    
      coord_z_quad = fe_values.quadrature_point (q)[0]-H;
      T_1r = (zs-coord_z_quad)*(coord_z_quad+(H+zb));
      T_2r = -((zs-coord_z_quad)-(coord_z_quad+H+zb));

//       cout << "  at " << fe_values.quadrature_point (q) << ", T_1r: " << T_1r << "\n";
      
      for (unsigned int i=0; i</*1*/dofs_per_cell; ++i)
      {
        const Tensor<1,dim> phi_i_velocity_real     = fe_values[velocity_real].value (i, q);
        const double        div_phi_i_velocity_real = fe_values[velocity_real].divergence (i, q);
        const double        phi_i_pressure_real     = fe_values[pressure_real].value (i, q);
        
        const Tensor<1,dim> phi_i_velocity_imag     = fe_values[velocity_imag].value (i, q);
        const double        div_phi_i_velocity_imag = fe_values[velocity_imag].divergence (i, q);             
        const double        phi_i_pressure_imag     = fe_values[pressure_imag].value (i, q);

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          const Tensor<1,dim> phi_j_velocity_real     = fe_values[velocity_real].value (j, q);
          const double        div_phi_j_velocity_real = fe_values[velocity_real].divergence (j, q);
          const double        phi_j_pressure_real     = fe_values[pressure_real].value (j, q);
    
          const Tensor<1,dim> phi_j_velocity_imag     = fe_values[velocity_imag].value (j, q);
          const double        div_phi_j_velocity_imag = fe_values[velocity_imag].divergence (j, q);             
          const double        phi_j_pressure_imag     = fe_values[pressure_imag].value (j, q);
    
          if (fe.system_to_component_index(i).first == 0)
          {
            if(fe.system_to_component_index(j).first == 0)
            {
              local_matrix(i,j) += (phi_i_velocity_real[0] * phi_j_velocity_real[0]) * fe_values.JxW(q);
            }else if(fe.system_to_component_index(j).first == 1)
            {
              local_matrix(i,j) += (- div_phi_i_velocity_real * phi_j_pressure_real) * fe_values.JxW(q);
            }
          }else if(fe.system_to_component_index(i).first == 1)
          {
            if(fe.system_to_component_index(j).first == 0)
            {
              local_matrix(i,j) += (- T_1r * phi_i_pressure_real * div_phi_j_velocity_real + T_2r * phi_i_pressure_real * phi_j_velocity_real[0] ) * fe_values.JxW(q);
            }else if(fe.system_to_component_index(j).first == 3)
            {
              local_matrix(i,j) += ( -T_3i * phi_i_pressure_real * phi_j_pressure_imag) * fe_values.JxW(q);
            }        // Note that this setting and the following manufacture of the pressure boundary condition are for repeating the results of Dec. 17, 2019. 
                     // That is, for the correct lhs, please change '-' to '+'; and the pressure boundary condition do not need to be manufactured to be 0
            
          }else if(fe.system_to_component_index(i).first == 2)
          {
            if(fe.system_to_component_index(j).first == 2)
            {
              local_matrix(i,j) += (phi_i_velocity_imag[0] * phi_j_velocity_imag[0]) * fe_values.JxW(q);
            }else if(fe.system_to_component_index(j).first == 3)
            {
              local_matrix(i,j) += (- div_phi_i_velocity_imag * phi_j_pressure_imag) * fe_values.JxW(q);
            }
          }else if(fe.system_to_component_index(i).first == 3)
          {
            if(fe.system_to_component_index(j).first == 0)
            {
//            local_matrix(i,j) += ( -T1i * phi_i_pressure_imag * div_phi_j_velocity_real + T2i * phi_i_pressure_imag * phi_j_velocity_real[0] ) * fe_values.JxW(q);
            }else if(fe.system_to_component_index(j).first == 1)
            {
              local_matrix(i,j) += (- T_3i * phi_i_pressure_imag * phi_j_pressure_real) * fe_values.JxW(q);
            }else if(fe.system_to_component_index(j).first == 2)
            {
              local_matrix(i,j) += (- T_1r * phi_i_pressure_imag * div_phi_j_velocity_imag + T_2r * phi_i_pressure_imag * phi_j_velocity_imag[0] ) * fe_values.JxW(q);
            }
          }                             
        }
                
        if(fe.system_to_component_index(i).first == 1)                          // note that, the contribution of the rhs is independent of j
        {
          local_rhs(i) += phi_i_pressure_real * 1.0 * fe_values.JxW(q);
        }             
      }
    }
        
    for (unsigned int face_n=0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n)
    {
      if (cell->at_boundary(face_n))
      {
        fe_face_values.reinit (cell, face_n);
        
        vector<Point<dim>> coords_of_boundary = fe_face_values.get_quadrature_points();
//           print_vector(coords_of_boundary);        
        
        if(cell->face(face_n)->boundary_id() == 0)
        {
          pressure_boundary_values.vector_value(coords_of_boundary[0], boundary_values_for_pressure);
          
          cout << "  boundary values for the pressure: ";
          cout << boundary_values_for_pressure;

          for (unsigned int q=0; q<n_face_q_points; ++q)
          {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              local_rhs(i) += -(fe_face_values[velocity_real].value (i, q) *
                              fe_face_values.normal_vector(q) *
                              boundary_values_for_pressure[0] *
                              fe_face_values.JxW(q));
            }
          }
        }else if(cell->face(face_n)->boundary_id() == 1)
        {
          gradient_boundary_values.vector_value(coords_of_boundary[0], boundary_values_for_gradient); 
          cout << "  boundary values for the gradient: ";
          cout << boundary_values_for_gradient;          
        }
      }
    }
    
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                                local_dof_indices[j],
                                local_matrix(i,j));


        for (unsigned int i=0; i<dofs_per_cell; ++i)
	{
            system_rhs(local_dof_indices[i]) += local_rhs(i);
	}

   
//         cout << "system_matrix before applying velocity boundary conditions is " << endl;
//         system_matrix.print_formatted(cout);
//         cout << '\n';    
//         
//         cout << "system_rhs before applying velocity boundary conditions is " << endl;
//         system_rhs.print(cout);     
//         cout << '\n';    
//     //     
    //     
        
//========================================Below are extracted from step-22==========================================  

//         cell->get_dof_indices (local_dof_indices);                  //deal with the global matrix and right hand side and implement (inhomogeneous) Dirichlet boundary conditions
//         
//         constraints.condense (system_matrix, system_rhs);
//         
//         constraints.distribute_local_to_global (local_matrix, local_rhs,
//                                                 local_dof_indices,
//                                                 system_matrix, system_rhs);

//==================================================================================
      
          
    }
      
//     cout << "system_matrix after applying BCs is " << endl;
//     system_matrix.print_formatted(cout);
//     cout << '\n';
//     cout << "system_rhs after applying BCs is " << endl;
//     system_rhs.print(cout);
//     cout << '\n';  
    
    int ur_row_last_dof = 1+triangulation.n_active_cells()*(degree)-(degree);     
    int ui_row_last_dof = dof_handler.n_dofs()/2+1+triangulation.n_active_cells()*(degree)-(degree);
//     
//     int pr_row_first_dof = 1+triangulation.n_active_cells()*(degree);     
//     int pi_row_last_dof = dof_handler.n_dofs()/2+1+triangulation.n_active_cells()*(degree);
//     
//     cout << "    ur_row_last_dof is " << ur_row_last_dof << ", ui_row_last_dof is " << ui_row_last_dof << endl;           //ur_row_last_dof denotes the row number of the last degree of freedom of ur, ui_row_last_dof denotes the row number of the last degree of freedom of ui
//     cout << "    pr_row_first_dof is " << pr_row_first_dof << ", pi_row_last_dof is " << pi_row_last_dof << endl;           //it is the same meaning for pr_row_first_dof and pi_row_last_dof. We use them to get rid of the impact of boundary conditions below.

//     ============================================================================================================    
//     for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
//     {
//         if(system_matrix.el(i,pr_row_first_dof)!=0 )                                                  //deal with the contribution of the natural BC to the rhs
//         {
//             system_rhs(i)=system_rhs(i)-system_matrix(i,pr_row_first_dof)*this->coeff_a;            
//         }        
//     }    
    
//     ============================================================================================================

    for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
    {
        if(system_matrix.el(ur_row_last_dof,i)!=0 || system_matrix.el(ui_row_last_dof,i)!=0 )             //set the rows involving the last test function of ur and ui zero
        {
            system_matrix.set(ur_row_last_dof,i,0.0);            
            system_matrix.set(ui_row_last_dof,i,0.0); 
        }        
    }
    
    for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
    {
        if(system_matrix.el(i,ur_row_last_dof)!=0 || system_matrix.el(i,ui_row_last_dof)!=0 )             //set the columns involving the last test function of ur and ui zero
        {   
            system_matrix.set(i,ur_row_last_dof,0.0);            
            system_matrix.set(i,ui_row_last_dof,0.0);
        }        
    }

    system_matrix.set(ur_row_last_dof,ur_row_last_dof,1.0);
    system_matrix.set(ui_row_last_dof,ui_row_last_dof,1.0);
    system_rhs(ur_row_last_dof)=0.0;
    system_rhs(ui_row_last_dof)=0.0;   
    
//     ============================================================================================================
//     for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
//     {
//         if(system_matrix.el(my_row_pr,i)!=0 || system_matrix.el(my_row_pi,i)!=0)                       //set the columns involving the first test function of pr and pi zero
//         {
//             system_matrix.set(my_row_pr,i,0.0);
//             system_matrix.set(my_row_pi,i,0.0);            
//         }        
//     }
//     
//     for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
//     {
//         if(system_matrix.el(i,my_row_pr)!=0 || system_matrix.el(i,my_row_pi)!=0 )                                                //set the rows involving the first test function of pr and pi zero
//         {
//             system_matrix.set(i,my_row_pr,0.0);   
//             system_matrix.set(i,my_row_pi,0.0);              
//         }        
//     }
//     
//     system_matrix.set(my_row_pr,my_row_pr,1.0);
//     system_matrix.set(my_row_pi,my_row_pi,1.0);
//     
// 
//     system_rhs(my_row_pr)=1.35;
//     system_rhs(my_row_pi)=0.0;
    
/*    cout << "system_matrix after applying velocity boundary conditions is " << endl;
    system_matrix.print_formatted(cout);
    cout << '\n';    
    
    cout << "system_rhs after applying velocity boundary conditions is " << endl;
    system_rhs.print(cout);     
    cout << '\n';   */ 
    
    if (is_matrix_after_BC_stored==1)
    {
        ofstream output_matrix("OneD_Helm_simp_MM_system_matrix_P"+to_string(degree)+"P"+to_string(degree-1)+"_ref_"+to_string(refine)+".txt");                                           
        
        for (unsigned int i=0;i<dof_handler.n_dofs();i++)
        {
            for (unsigned int j=0;j<dof_handler.n_dofs();j++)
            {
                output_matrix << system_matrix.el(i, j) << " "; // behaves like cout - cout is also a stream
            }
            output_matrix << "\n";
        } 
        output_matrix.close();
        output_matrix.clear();
    }
    
    if (is_rhs_after_BC_stored)
    {
        ofstream output_rhs("OneD_Helm_simp_MM_system_rhs_P"+to_string(degree)+"P"+to_string(degree-1)+"_ref_"+to_string(refine)+".txt");    
        
        for (unsigned int i=0;i<dof_handler.n_dofs();i++)
        {
            output_rhs << system_rhs[i] << " ";               // behaves like cout - cout is also a stream
            output_rhs << "\n";
        } 

        output_rhs.close();
        output_rhs.clear();  
    }
}
  

template <int dim>
void MixedLaplaceProblem_Complex_Validation<dim>::solve ()
{
  cout << "Solving\n";
  TimerOutput::Scope t(computing_timer, "solve");
    
  if(is_UMFPACK==0)
  {
    cout << "    CG solver" << endl;
    
    InverseMatrix_CG<SparseMatrix<double> > inverse_mass (system_matrix.block(0,0));
    Vector<double> tmp (solution.block(0).size());
    
    {
      SchurComplement<InverseMatrix_CG<SparseMatrix<real_t> >> schur_complement (system_matrix, inverse_mass);
      Vector<double> schur_rhs (solution.block(1).size());
      inverse_mass.vmult (tmp, system_rhs.block(0));
        
    system_matrix.block(1,0).vmult (schur_rhs, tmp);
    schur_rhs -= system_rhs.block(1);

    SolverControl solver_control (solution.block(1).size(),
                                    1e-12*schur_rhs.l2_norm());
    SolverCG<> cg (solver_control);

    ApproximateSchurComplement approximate_schur (system_matrix);
    InverseMatrix_CG<ApproximateSchurComplement> approximate_inverse
    (approximate_schur);
    cg.solve (schur_complement, solution.block(1), schur_rhs,
                approximate_inverse);

    constraints.distribute (solution);

    cout << solver_control.last_step()
        << " CG Schur complement iterations to obtain convergence."
        << endl;
    }

    {
    system_matrix.block(0,1).vmult (tmp, solution.block(1));
    tmp *= -1;
    tmp += system_rhs.block(0);

    inverse_mass.vmult (solution.block(0), tmp);
    constraints.distribute (solution);
    }
    }else if(is_UMFPACK==1)
    {
    cout << "    UMFPACK monolithic" << endl;
    SparseDirectUMFPACK  A_direct;		                      //
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
    }
    
//     cout << "\n";
//     cout << "solution before distribution \n";
//     solution.print(cout);
    
    constraints.distribute (solution);

//     cout << "\n";
//     cout << "solution after distribution \n";
//     solution.print(cout);
    
    for (unsigned int r = 0; r < n_u; r++)                  //extract the velocity part
    {
        solution_grad_real[r] = solution[r];
        solution_grad_imag[r] = solution[dof_handler.n_dofs()/2+r];
    }
    
//     solution_solu_real[0] = solution[n_u];
    for (unsigned int r = 1; r < n_p; r++)                  //extract the surface part
    {
        solution_solu_real[r] = solution[n_u+r];
        solution_solu_imag[r] = solution[dof_handler.n_dofs()/2+n_u+r];
    }
    
//         double inter_ur, inter_ui, inter_pr, inter_pi;          //adjust the order of velocity and surface
//         
//         inter_ur = solution_grad_real[n_u-(degree+1)];
//         inter_ui = solution_grad_imag[n_u-(degree+1)];
//         inter_pr = solution_solu_real[n_p-(degree)];
//         inter_pi = solution_solu_imag[n_p-(degree)];        
//         
//         for (unsigned int i = 0; i < degree; i++)
//         {
//             solution_grad_real[n_u-(degree-i+1)]=solution_grad_real[n_u-(degree-i)];
//             solution_grad_imag[n_u-(degree-i+1)]=solution_grad_imag[n_u-(degree-i)];            
//         }
//         
//         for (unsigned int i = 0; i < degree-1; i++)
//         {
//             solution_solu_real[n_p-(degree-i)]=solution_solu_real[n_p-(degree-i-1)];
//             solution_solu_imag[n_p-(degree-i)]=solution_solu_imag[n_p-(degree-i-1)];            
//         }        
//         
//         solution_grad_real[n_u-1]=inter_ur;
//         solution_grad_imag[n_u-1]=inter_ui;
//         solution_solu_real[n_p-1]=inter_pr;
//         solution_solu_imag[n_p-1]=inter_pi;
    
    
    solution_solu_real[0]=0.0;                                 // this->coeff_a
    for (unsigned int r = 0; r < n_p; r++)                  //compute the amplitude
    {
        solution_solu_amp[r] = sqrt(solution_solu_real[r]*solution_solu_real[r] + solution_solu_imag[r]*solution_solu_imag[r]);
    }
    
//         cout << "solution_solu_real is: " << endl << solution_solu_real << endl;
//         cout << "solution_solu_amp is: " << endl << solution_solu_amp << endl;
    
//         solution_solu_amp[0] = 1.35; 


    for (unsigned int i = 0; i < n_u; i++)                                                      //assemble the gradient and solution differentiated by the real and imaginary part
    {
        solution_real[i]=solution_grad_real[i];
        solution_imag[i]=solution_grad_imag[i];
    }
    
    for (unsigned int i = 0; i < n_p; i++)
    {
        solution_real[n_u+i]=solution_solu_real[i];
        solution_imag[n_u+i]=solution_solu_imag[i];
    }
    
//         for (unsigned int i = 0; i < dof_handler.n_dofs()/2.0; i++)
//         {
//                 solution_solu_amp[i] = sqrt(solution_grad_real[i]*solution_grad_real[i] + solution_grad_imag[i]*solution_grad_imag[i]);
//         }  
    
//     cout << "\n";
//     cout << "solution_grad_real: " << solution_grad_real;
//     cout << "solution_solu_real: " << solution_solu_real;
//     cout << "solution_grad_imag: " << solution_grad_imag;        
//     cout << "solution_solu_imag: " << solution_solu_imag;  
//         
//        cout << "solution_solu_amp: " << solution_solu_amp;

//     cout << "\n";
//     cout << "solution_real: \n";
//     solution_real.print(cout);
//     cout << "solution_imag: \n";
//     solution_imag.print(cout);
    
}

  template <int dim>
  void MixedLaplaceProblem_Complex_Validation<dim>::compute_l2_norms_numerically ()
  {
    const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
    const ComponentSelectFunction<dim>
    velocity_mask(0, dim+1);

    Zero_Function_Custom<dim> obj_zero_function;
    
    Vector<double> cellwise_L2_norms (triangulation.n_active_cells());
    Vector<double> cellwise_L2_norms_real (triangulation.n_active_cells());
    Vector<double> cellwise_L2_norms_imag (triangulation.n_active_cells());
    
    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature (q_trapez, degree+2);

// ====================================================================================================================================  L2 norm on the pressure

    VectorTools::integrate_difference (dof_handler_single, solution_real, obj_zero_function,         // ZeroFunction<dim>(),      //
                                       cellwise_L2_norms_real, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);

    VectorTools::integrate_difference (dof_handler_single, solution_imag, obj_zero_function,
                                       cellwise_L2_norms_imag, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    
   for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
   {    
       cellwise_L2_norms[i] = pow(pow(cellwise_L2_norms_real[i],2)+pow(cellwise_L2_norms_imag[i],2), 0.5);
   }

    L2_inte_solu = cellwise_L2_norms.l2_norm();
    L2_inte_solu_real = cellwise_L2_norms_real.l2_norm();
    L2_inte_solu_imag = cellwise_L2_norms_imag.l2_norm();
    
// ====================================================================================================================================  L2 norm on the velocity

    VectorTools::integrate_difference (dof_handler_single, solution_real, obj_zero_function,
                                       cellwise_L2_norms_real, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);

    VectorTools::integrate_difference (dof_handler_single, solution_imag, obj_zero_function,
                                       cellwise_L2_norms_imag, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    
   for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
   {    
       cellwise_L2_norms[i] = pow(pow(cellwise_L2_norms_real[i],2)+pow(cellwise_L2_norms_imag[i],2), 0.5);
   }
    
    L2_inte_grad = cellwise_L2_norms.l2_norm();
    L2_inte_grad_real = cellwise_L2_norms_real.l2_norm();
    L2_inte_grad_imag = cellwise_L2_norms_imag.l2_norm();
    
    
    
    
//     ===================================================================================================================================
    
//     VectorTools::integrate_difference (dof_handler_single, solution_real, obj_zero_function,
//                                        cellwise_L2_norms_real, quadrature,
//                                        VectorTools::mean,
//                                        &pressure_mask);
//     inte_solu_real = -cellwise_L2_norms_real.mean_value()* triangulation.n_active_cells();
// 
//     VectorTools::integrate_difference (dof_handler_single, solution_imag, obj_zero_function,
//                                        cellwise_L2_norms_imag, quadrature,
//                                        VectorTools::mean,
//                                        &pressure_mask);    
//     inte_solu_imag = -cellwise_L2_norms_imag.mean_value()* triangulation.n_active_cells();
//     
//     cout << "  inte_solu_real = " << inte_solu_real << ", inte_solu_imag = " << inte_solu_imag << endl;
    
// ====================================================================================================================================  u
    
//     VectorTools::integrate_difference (dof_handler_single, solution_real, obj_zero_function,
//                                        cellwise_L2_norms_real, quadrature,
//                                        VectorTools::mean,
//                                        &velocity_mask);
//     inte_grad_real = -cellwise_L2_norms_real.mean_value()* triangulation.n_active_cells();
// 
//     VectorTools::integrate_difference (dof_handler_single, solution_imag, obj_zero_function,
//                                        cellwise_L2_norms_imag, quadrature,
//                                        VectorTools::mean,
//                                        &velocity_mask);    
//     inte_grad_imag = -cellwise_L2_norms_imag.mean_value()* triangulation.n_active_cells();
//     
//     cout << "  inte_grad_real = " << inte_grad_real << ", inte_grad_imag = " << inte_grad_imag << endl;    
//     
    
//     cout << "\n";
    
  }



template <int dim>
void MixedLaplaceProblem_Complex_Validation<dim>::compute_errors_quad ()                      
{
  cout << "Computing errors customly\n";
    
#if 1
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  
  unsigned int n_q_points = degree+2;                   // using deg+2, the accuracy is slightly higher than that using deg+1, Aug. 16, 2019
  QGauss<dim>  quadrature_formula(n_q_points);
  
//   cout << "solution_solu_real is: \n" << solution_solu_real << "\n";
//   cout << "solution_solu_imag is: \n" << solution_solu_imag << "\n";
    
    vector<double> solution_values_real_at_cell_quads_first_refine(n_q_points);
    vector<double > solution_gradients_real_at_cell_quads_first_refine(n_q_points);
    vector<Tensor<1, dim> > solution_2ndd_real_at_cell_quads_first_refine(n_q_points);
    
    vector<double> solution_values_imag_at_cell_quads_first_refine(n_q_points);
    vector<double > solution_gradients_imag_at_cell_quads_first_refine(n_q_points);
    vector<Tensor<1, dim> > solution_2ndd_imag_at_cell_quads_first_refine(n_q_points);
        
    vector<double> coords_all_quads_first_refine;

    vector<double> solution_values_real_all_quads_first_refine;    
    vector<double> solution_gradients_real_all_quads_first_refine;
    vector<Tensor<1, dim> > solution_2ndds_real_all_quads_first_refine;

    vector<double> solution_values_imag_all_quads_first_refine;    
    vector<double> solution_gradients_imag_all_quads_first_refine;
    vector<Tensor<1, dim> > solution_2ndds_imag_all_quads_first_refine;
    
    
    FEValues<dim> fe_values_pressure (fe_dgq, quadrature_formula,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    
    FEValues<dim> fe_values_velocity (fe_q, quadrature_formula,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_single (fe_single, quadrature_formula,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);    
    
    
    typename DoFHandler<dim>::active_cell_iterator  
    cell = dof_handler_pressure_single.begin_active(),
    endc = dof_handler_pressure_single.end();
    
    for (; cell!=endc; ++cell)
    {
//         cout << "cell " << cell->active_cell_index() << ", vertex(0): " << cell->vertex(0) ;
        
        fe_values_pressure.reinit (cell);
        
//         cout << ", quad coords: ";
        for (unsigned int i=0; i<fe_values_pressure.get_quadrature_points().size();++i)
        {
//             cout << fe_values_pressure.get_quadrature_points()[i](0) << " ";
            coords_all_quads_first_refine.push_back(fe_values_pressure.get_quadrature_points()[i](0));
        }
        
//         cout << ", quad coords: ";
//         for (unsigned int i=0; i<n_q_points;++i)
//         {
//             cout << fe_values_velocity.get_quadrature_points()[i](0) << " ";
//         }
        
/*        cout << ", J*W: ";
        for (unsigned int i=0; i<fe_values_pressure.get_JxW_values ().size();++i)
        {
            cout << fe_values_pressure.get_JxW_values()[i] << " ";
        } */ 
//         cout << endl;        
        
        fe_values_pressure.get_function_values (solution_solu_real, solution_values_real_at_cell_quads_first_refine);   
        fe_values_pressure.get_function_values (solution_solu_imag, solution_values_imag_at_cell_quads_first_refine);
        
//         cout << ", solution values at quad coords: ";
        for (unsigned int i=0; i<n_q_points;++i)
        {
            solution_values_real_all_quads_first_refine.push_back(solution_values_real_at_cell_quads_first_refine[i]);
            solution_values_imag_all_quads_first_refine.push_back(solution_values_imag_at_cell_quads_first_refine[i]);
            
//             cout << solution_values_real_at_cell_quads_first_refine[i] << " ";
        }
//         cout << endl;

    }
     
    cell = dof_handler_velocity_single.begin_active(),
    endc = dof_handler_velocity_single.end();
    
    for (; cell!=endc; ++cell)
    {
//         cout << "    cell " << cell->active_cell_index() << ", vertex(0): " << cell->vertex(0) ;
        
        fe_values_velocity.reinit (cell);

//         cout << ", quad coords: ";
//         for (unsigned int i=0; i<n_q_points;++i)
//         {
//             cout << fe_values_velocity.get_quadrature_points()[i](0) << " ";
//         }
        
        fe_values_velocity.get_function_values (solution_grad_real, solution_gradients_real_at_cell_quads_first_refine);
        fe_values_velocity.get_function_values (solution_grad_imag, solution_gradients_imag_at_cell_quads_first_refine);
        
//         cout << ", solution gradients at quad coords: ";
//         for (unsigned int i=0; i<n_q_points;++i)
//         {
//             cout << solution_gradients_real_at_cell_quads_first_refine[i] << " ";
//         } 
        
        for (unsigned int i=0; i<n_q_points;++i)
        {
            solution_gradients_real_all_quads_first_refine.push_back(solution_gradients_real_at_cell_quads_first_refine[i]);
            solution_gradients_imag_all_quads_first_refine.push_back(solution_gradients_imag_at_cell_quads_first_refine[i]);
        }         
        
        fe_values_velocity.get_function_gradients (solution_grad_real, solution_2ndd_real_at_cell_quads_first_refine);             // work on the first-order derivative of the gradient
        fe_values_velocity.get_function_gradients (solution_grad_imag, solution_2ndd_imag_at_cell_quads_first_refine);
        
//         cout << ", solution 2ndds at quad coords: ";
//         for (unsigned int i=0; i<n_q_points;++i)
//         {
//             cout << solution_2ndd_real_at_cell_quads_first_refine[i] << " ";
//         } 
        for (unsigned int i=0; i<n_q_points;++i)
        {
            solution_2ndds_real_all_quads_first_refine.push_back(solution_2ndd_real_at_cell_quads_first_refine[i]);
            solution_2ndds_imag_all_quads_first_refine.push_back(solution_2ndd_imag_at_cell_quads_first_refine[i]);
        }
        
//         cout << endl;

    }       
    
/*        ofstream coords_quad_first_refine;
        coords_quad_first_refine.open("coords_quad_deg_"+to_string(degree)+"_ref_"+to_string(refine)+".txt",ofstream::app);              // save data into files
        for (double n : coords_all_quads_first_refine)
        {
            coords_quad_first_refine << setprecision(20) << n << "\n";
        }
        coords_quad_first_refine.close(); */   
        
//         ofstream solution_quad_first_refine;
//         solution_quad_first_refine.open("solution_solu_real_quad_deg_"+to_string(degree-1)+"_ref_"+to_string(refine)+".txt",ofstream::app);
//         for (double n : solution_values_real_all_quads_first_refine)
//         {
//             solution_quad_first_refine << setprecision(20) << n << "\n";
//         }
//         solution_quad_first_refine.close();

//         ofstream solution_grad_real_quad_first_refine;
//         solution_grad_real_quad_first_refine.open("solution_grad_real_quad_deg_"+to_string(degree)+"_ref_"+to_string(refine)+".txt",ofstream::app);
//         for (double n : solution_gradients_real_all_quads_first_refine)
//         {
//             solution_grad_real_quad_first_refine << setprecision(20) << n << "\n";
//         }
//         solution_grad_real_quad_first_refine.close();
        
    
//     cout << "  solution of the first refinement at all quad points reads: " << endl;
//     for(double n : solution_values_real_all_quads_first_refine) 
//     {
//         cout << n << ' ';
//     }    
//     cout << endl; 

//     cout << "    solution 2ndds of the first refinement at all quad points reads: ";
//     for(Tensor<1,dim> n : solution_2ndds_real_all_quads_first_refine) 
//     {
//         cout << n[0] << ' ';
//     }    
//     cout << endl; 

/*        ofstream solution_2ndds_real_quad_first_refine;
        solution_2ndds_real_quad_first_refine.open("solution_2ndds_real_quad_deg_"+to_string(degree)+"_ref_"+to_string(refine)+".txt",ofstream::app);
        for (Tensor<1,dim> n : solution_2ndds_real_all_quads_first_refine)
        {
            solution_2ndds_real_quad_first_refine << setprecision(20) << n[0] << "\n";
        }
        solution_2ndds_real_quad_first_refine.close();   */ 
    

//     cout << "    #### second refinement" << endl;
    
    vector<double> coords_all_quads_second_refine;
    
    vector<double> solution_values_real_all_quads_second_refine;   
    vector<double> solution_gradients_real_all_quads_second_refine;    
    vector<Tensor<1,dim>> solution_2ndds_real_all_quads_second_refine;

    vector<double> solution_values_imag_all_quads_second_refine;   
    vector<double> solution_gradients_imag_all_quads_second_refine;    
    vector<Tensor<1,dim>> solution_2ndds_imag_all_quads_second_refine;
    
//     cout << "    computing the finer solution ... " << endl;

    triangulation.refine_global ();  
    setup_system ();
    assemble_system ();
    solve ();
    
//     cout << "   solution_solu_real_second_refine is "; 
//     solution_solu_real.print(cout);    
//     
//     cout << "   solution_grad_real_second_refine is "; 
//     solution_grad_real.print(cout);    
//     
//     cout << endl;
    
//     cout << "   obtaining the pressure at quadrature points " << endl;
    
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ on the left quadrature points

    unsigned int n_q_points_left = (n_q_points+1)/2.0;
    
    vector<double > solution_values_real_left_cell_quads_second_refine(n_q_points_left);
    vector<double > solution_values_imag_left_cell_quads_second_refine(n_q_points_left);
    
    vector<Point<dim> > left_quad_vector(n_q_points_left);
    for (unsigned int i=0; i<left_quad_vector.size();++i)
    {
        left_quad_vector[i]=quadrature_formula.get_points()[i]*2;
//         cout << left_quad_vector[i] << " ";
    }
//     cout << endl;
    Quadrature<dim>  quadrature_formula_high_2_low_left(left_quad_vector);                   // fe_values_pressure.get_quadrature_points()[2](0)  left_quad_vector       
    FEValues<dim> fe_values_pressure_high_2_low_left (fe_dgq, quadrature_formula_high_2_low_left,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ on the right quadrature points

    unsigned int n_q_points_right = (n_q_points+1)/2.0;
    
    vector<double > solution_values_real_right_cell_quads_second_refine(n_q_points_right);
    vector<double > solution_values_imag_right_cell_quads_second_refine(n_q_points_right);
    
    vector<Point<dim> > right_quad_vector(n_q_points_right);
    for (unsigned int i=0; i<right_quad_vector.size();++i)
    {
        right_quad_vector[i](0) = (quadrature_formula.get_points()[i+(n_q_points_left*2==n_q_points?n_q_points_left:n_q_points_left-1)](0)-0.5)*2;
//         cout << right_quad_vector[i] << " ";
    }
//     cout << endl;
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
        
//         cout << "cell " << cell->active_cell_index() << ", vertex(0): " << cell->vertex(0) ;
        if (cell->active_cell_index()%2==0)
        {
            fe_values_pressure_high_2_low_left.reinit (cell);
            
//             cout << ", coords of left quandrature points: ";
//             for (unsigned int i=0; i<fe_values_pressure_high_2_low_left.get_quadrature_points().size();++i)
//             {
//                 cout << fe_values_pressure_high_2_low_left.get_quadrature_points()[i] << " ";                           //<< setprecision(10) 
//             }
//             
            fe_values_pressure_high_2_low_left.get_function_values (solution_solu_real, solution_values_real_left_cell_quads_second_refine);
            fe_values_pressure_high_2_low_left.get_function_values (solution_solu_imag, solution_values_imag_left_cell_quads_second_refine);
            
//             
//             cout << ", solution at left quad coords: ";
            for (unsigned int i=0; i<n_q_points_left;++i)
            {
                solution_values_real_all_quads_second_refine.push_back(solution_values_real_left_cell_quads_second_refine[i]);
                solution_values_imag_all_quads_second_refine.push_back(solution_values_imag_left_cell_quads_second_refine[i]);
//                 cout << solution_values_real_left_cell_quads_second_refine[i] << " ";
            }  
//             cout << endl;
        }else
        {
            fe_values_pressure_high_2_low_right.reinit (cell);
            
//             cout << ", coords of right quandrature points: ";
//             for (unsigned int i=0; i<fe_values_pressure_high_2_low_right.get_quadrature_points().size();++i)
//             {
//                 cout << fe_values_pressure_high_2_low_right.get_quadrature_points()[i] << " ";                           //<< setprecision(10) 
//             }
//             
            fe_values_pressure_high_2_low_right.get_function_values (solution_solu_real, solution_values_real_right_cell_quads_second_refine);
            fe_values_pressure_high_2_low_right.get_function_values (solution_solu_imag, solution_values_imag_right_cell_quads_second_refine);
            
//             cout << ", solution at right quad coords: ";
//             for (unsigned int i=0; i<n_q_points_right;++i)
//             {
//                 cout << solution_values_real_right_cell_quads_second_refine[i]  << " ";
//             }   
//             cout << endl;
            
            for (unsigned int i=(2*n_q_points_right==n_q_points ? 0 : 1); i<n_q_points_right;++i)
            {
                solution_values_real_all_quads_second_refine.push_back(solution_values_real_right_cell_quads_second_refine[i]);
                solution_values_imag_all_quads_second_refine.push_back(solution_values_imag_right_cell_quads_second_refine[i]);
            }
                
        }
    }
    
/*    cout << "  solution of the second refinement at all quad points reads: " << endl;
    for(double n : solution_values_real_all_quads_second_refine) 
    {
        cout << n << ' ';
    }    
    cout << endl;  */   

//     cout << "   obtaining the gradient and second-order derivative at quadrature points " << endl;

    FEValues<dim> fe_values_velocity_high_2_low_left (fe_q, quadrature_formula_high_2_low_left,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);    
    
    vector<double> solution_gradients_real_left_cell_quads_second_refine(n_q_points_left);
    vector<double> solution_gradients_imag_left_cell_quads_second_refine(n_q_points_left);
    
    vector<Tensor<1, dim>> solution_2ndds_real_left_cell_quads_second_refine(n_q_points_left);
    vector<Tensor<1, dim>> solution_2ndds_imag_left_cell_quads_second_refine(n_q_points_left);
    
        
    FEValues<dim> fe_values_velocity_high_2_low_right (fe_q, quadrature_formula_high_2_low_right,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);   
    
    vector<double> solution_gradients_real_right_cell_quads_second_refine(n_q_points_right);
    vector<double> solution_gradients_imag_right_cell_quads_second_refine(n_q_points_right);
    
    vector<Tensor<1, dim>> solution_2ndds_real_right_cell_quads_second_refine(n_q_points_right);
    vector<Tensor<1, dim>> solution_2ndds_imag_right_cell_quads_second_refine(n_q_points_right);

    cell = dof_handler_velocity_single.begin_active();
    endc = dof_handler_velocity_single.end();    
    for (; cell!=endc; ++cell)
    {
        
//         cout << "    cell " << cell->active_cell_index() << ", vertex(0): " << cell->vertex(0) ;
        
        fe_values_velocity.reinit(cell);
        
//         cout << ", quad coords: ";
//         for (unsigned int i=0; i<n_q_points;++i)
//         {
//             cout << fe_values_velocity.get_quadrature_points()[i](0) << " ";
//         }
        
        for (unsigned int i=0; i<n_q_points;++i)
        {
            coords_all_quads_second_refine.push_back(fe_values_velocity.get_quadrature_points()[i](0));
        }        
        
        if (cell->active_cell_index()%2==0)
        {
            fe_values_velocity_high_2_low_left.reinit (cell);
            
//             cout << ", coords of left quandrature points of the previous refinement: ";
//             for (unsigned int i=0; i<n_q_points_left;++i)
//             {
//                 cout << fe_values_velocity_high_2_low_left.get_quadrature_points()[i] << " ";                           //<< setprecision(10) 
//             }
//             
            fe_values_velocity_high_2_low_left.get_function_values (solution_grad_real, solution_gradients_real_left_cell_quads_second_refine);
            fe_values_velocity_high_2_low_left.get_function_values (solution_grad_imag, solution_gradients_imag_left_cell_quads_second_refine);
            
//             cout << ", gradients at left quad coords: ";
//             for (unsigned int i=0; i<n_q_points_left;++i)
//             {
//                 cout << solution_gradients_real_left_cell_quads_second_refine[i] << " ";
//             }  

            
            for (unsigned int i=0; i<n_q_points_left;++i)
            {
                solution_gradients_real_all_quads_second_refine.push_back(solution_gradients_real_left_cell_quads_second_refine[i]);
                solution_gradients_imag_all_quads_second_refine.push_back(solution_gradients_imag_left_cell_quads_second_refine[i]);
            }
            
    
            fe_values_velocity_high_2_low_left.get_function_gradients (solution_grad_real, solution_2ndds_real_left_cell_quads_second_refine);
            fe_values_velocity_high_2_low_left.get_function_gradients (solution_grad_imag, solution_2ndds_imag_left_cell_quads_second_refine);
            
//             cout << ", 2ndds at left quad coords: ";
//             for (unsigned int i=0; i<n_q_points_left;++i)
//             {
//                 cout << solution_2ndds_real_left_cell_quads_second_refine[i] << " ";
//             }  
            
            for (unsigned int i=0; i<n_q_points_left;++i)
            {
                solution_2ndds_real_all_quads_second_refine.push_back(solution_2ndds_real_left_cell_quads_second_refine[i]);
                solution_2ndds_imag_all_quads_second_refine.push_back(solution_2ndds_imag_left_cell_quads_second_refine[i]);
            }             
            
//             cout << endl;            
        }else
        {
            fe_values_velocity_high_2_low_right.reinit (cell);
            
//             cout << ", coords of right quandrature points of the previous refinement: ";
//             for (unsigned int i=0; i<n_q_points_right;++i)
//             {
//                 cout << fe_values_velocity_high_2_low_right.get_quadrature_points()[i] << " ";                           //<< setprecision(10) 
//             }
//             
            fe_values_velocity_high_2_low_right.get_function_values (solution_grad_real, solution_gradients_real_right_cell_quads_second_refine);
            fe_values_velocity_high_2_low_right.get_function_values (solution_grad_imag, solution_gradients_imag_right_cell_quads_second_refine);
            
            fe_values_velocity_high_2_low_right.get_function_gradients (solution_grad_real, solution_2ndds_real_right_cell_quads_second_refine);
            fe_values_velocity_high_2_low_right.get_function_gradients (solution_grad_imag, solution_2ndds_imag_right_cell_quads_second_refine);
            
//             cout << ", gradients at right quad coords: ";
//             for (unsigned int i=0; i<n_q_points_right;++i)
//             {
//                 cout << solution_gradients_real_right_cell_quads_second_refine[i] << " ";
//             } 
            
/*            cout << ", 2ndds at right quad coords: ";
            for (unsigned int i=0; i<n_q_points_right;++i)
            {
                cout << solution_2ndds_real_right_cell_quads_second_refine[i] << " ";
            } */           
            

            for (unsigned int i=(2*n_q_points_right==n_q_points ? 0 : 1); i<n_q_points_right;++i)
            {
                solution_gradients_real_all_quads_second_refine.push_back(solution_gradients_real_right_cell_quads_second_refine[i]);
                solution_gradients_imag_all_quads_second_refine.push_back(solution_gradients_imag_right_cell_quads_second_refine[i]);
                
                solution_2ndds_real_all_quads_second_refine.push_back(solution_2ndds_real_right_cell_quads_second_refine[i]);
                solution_2ndds_imag_all_quads_second_refine.push_back(solution_2ndds_imag_right_cell_quads_second_refine[i]);
            }              
            
 
//             cout << endl;

        }
    }
    
/*    cout << "  solution gradients of the second refinement at all quad points reads: " << endl;
    for(double n : solution_gradients_real_all_quads_second_refine) 
    {
        cout << n << ' ';
    }    
    cout << endl; */  

//     cout << "    solution 2ndds of the second refinement at all quad points reads: ";
//     for(Tensor<1,dim> n : solution_2ndds_real_all_quads_second_refine) 
//     {
//         cout << n[0] << ' ';
//     }    
//     cout << endl; 
    
//         ofstream coords_quad_first_refine;
//         coords_quad_first_refine.open("coords_quad_deg_"+to_string(degree)+"_ref_"+to_string(refine)+"_2nd_ref.txt",ofstream::app);
//         for (double n : coords_all_quads_second_refine)
//         {
//             coords_quad_first_refine << setprecision(20) << n << "\n";
//         }
//         coords_quad_first_refine.close();   
    
/*        ofstream solution_grad_real_quad_second_refine;
        solution_grad_real_quad_second_refine.open("solution_grad_real_quad_deg_"+to_string(degree)+"_ref_"+to_string(refine)+"_2nd_ref.txt",ofstream::app);
        for (double n : solution_gradients_real_all_quads_second_refine)
        {
            solution_grad_real_quad_second_refine << setprecision(20) << n << "\n";
        }
        solution_grad_real_quad_second_refine.close();   */ 
        
// ========================================== computing the error 
        
//     cout << "    ~~~~ computing the error " << endl;

    Vector<double> cellwise_error_real(triangulation_first_refine.n_active_cells());
    Vector<double> cellwise_error_imag(triangulation_first_refine.n_active_cells());
    Vector<double> cellwise_error(triangulation_first_refine.n_active_cells());
    
//     cout << "    for the pressure " << endl;    
    
    cell = dof_handler_pressure_single_first_refine.begin_active();
    endc = dof_handler_pressure_single_first_refine.end();
    
    for (; cell!=endc; ++cell)
    {
        int cell_index = cell->active_cell_index();
        fe_values_pressure.reinit (cell);
/*        cout << "    J*W: ";
        for (unsigned int i=0; i<fe_values_pressure.get_JxW_values ().size();++i)
        {
            cout << fe_values_pressure.get_JxW_values()[i] << " ";
        }  
        cout << endl; */  
        
//         cout << "    solution_values_real_all_quads_first_refine:  ";
//         for (int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             cout << solution_values_real_all_quads_first_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         cout << endl;
        
//         cout << "    solution_values_real_all_quads_second_refine:  ";
//         for (int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             cout << solution_values_real_all_quads_second_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         cout << endl;

        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            cellwise_error_real[cell_index] += pow(solution_values_real_all_quads_first_refine[n_q_points*cell_index + q_index] - solution_values_real_all_quads_second_refine[n_q_points*cell_index + q_index],2 ) * fe_values_pressure.JxW (q_index) ; 
            cellwise_error_imag[cell_index] += pow(solution_values_imag_all_quads_first_refine[n_q_points*cell_index + q_index] - solution_values_imag_all_quads_second_refine[n_q_points*cell_index + q_index],2 ) * fe_values_pressure.JxW (q_index) ;
        }
        
        cellwise_error_real[cell_index] = pow(cellwise_error_real[cell_index], 0.5);
        cellwise_error_imag[cell_index] = pow(cellwise_error_imag[cell_index], 0.5);
        cellwise_error[cell_index] = pow(pow(cellwise_error_real[cell_index],2) + pow(cellwise_error_imag[cell_index],2), 0.5);
        
    }
    
    L2_error_solu_real = cellwise_error_real.l2_norm();
    L2_error_solu_imag = cellwise_error_imag.l2_norm();
    L2_error_solu = cellwise_error.l2_norm();
    
    
/*    cout << "    cellwise_error for the pressure: " << endl;
    cellwise_error.print(cout);*/    

//     cout << "    for the velocity " << endl; 
    
//     cout << "    cellwise_error_real: " << endl;
//     cellwise_error_real.print(cout);
    
    cellwise_error_real.reinit(triangulation_first_refine.n_active_cells());
    cellwise_error_imag.reinit(triangulation_first_refine.n_active_cells());
    cellwise_error.reinit(triangulation_first_refine.n_active_cells());


/*    cout << "    cellwise_error_real: " << endl;
    cellwise_error_real.print(cout);
    cout << "    cellwise_error_imag: " << endl;
    cellwise_error_imag.print(cout);    
    cout << "    cellwise_error: " << endl;
    cellwise_error.print(cout); */   

    
    cell = dof_handler_velocity_single_first_refine.begin_active();
    endc = dof_handler_velocity_single_first_refine.end();

    for (; cell!=endc; ++cell)
    {
        int cell_index = cell->active_cell_index();
        fe_values_velocity.reinit (cell);
//         cout << "    J*W: ";
//         for (unsigned int i=0; i<fe_values_velocity.get_JxW_values ().size();++i)
//         {
//             cout << fe_values_velocity.get_JxW_values()[i] << " ";
//         }  
//         cout << endl;     

//         cout << "    solution_gradients_all_quads_first_refine:  ";
//         for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             cout << solution_gradients_real_all_quads_first_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         cout << endl;
//         
//         cout << "    solution_gradients_real_all_quads_second_refine:  ";
//         for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
//         {
//             cout << solution_gradients_real_all_quads_second_refine[n_q_points*cell_index + q_index] << " ";
//         }
//         cout << endl;
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            cellwise_error_real[cell_index] += pow(solution_gradients_real_all_quads_first_refine[n_q_points*cell_index + q_index] - solution_gradients_real_all_quads_second_refine[n_q_points*cell_index + q_index],2 ) * fe_values_velocity.JxW (q_index) ; 
            cellwise_error_imag[cell_index] += pow(solution_gradients_imag_all_quads_first_refine[n_q_points*cell_index + q_index] - solution_gradients_imag_all_quads_second_refine[n_q_points*cell_index + q_index],2 ) * fe_values_velocity.JxW (q_index) ; 
        }
//         cout << endl;        

        cellwise_error_real[cell_index] = pow(cellwise_error_real[cell_index], 0.5);
        cellwise_error_imag[cell_index] = pow(cellwise_error_imag[cell_index], 0.5);
        cellwise_error[cell_index] = pow(pow(cellwise_error_real[cell_index],2) + pow(cellwise_error_imag[cell_index],2), 0.5);
    }
    
    L2_error_grad_real = cellwise_error_real.l2_norm();
    L2_error_grad_imag = cellwise_error_imag.l2_norm();
    L2_error_grad = cellwise_error.l2_norm();

//     cout << "    for the second-order derivative " << endl; 
    
    cellwise_error_real.reinit(triangulation_first_refine.n_active_cells());
    cellwise_error_imag.reinit(triangulation_first_refine.n_active_cells());
    cellwise_error.reinit(triangulation_first_refine.n_active_cells());
    
    cell = dof_handler_velocity_single_first_refine.begin_active();
    endc = dof_handler_velocity_single_first_refine.end();

    for (; cell!=endc; ++cell)
    {
        int cell_index = cell->active_cell_index();
        fe_values_velocity.reinit (cell);
        
        
/*        cout << "    solution_2ndds_real_all_quads_first_refine:  ";
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            cout << solution_2ndds_real_all_quads_first_refine[n_q_points*cell_index + q_index] << " ";
        }
        cout << endl;
//         
        cout << "    solution_2ndds_real_all_quads_second_refine:  ";
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            cout << solution_2ndds_real_all_quads_second_refine[n_q_points*cell_index + q_index] << " ";
        }
        cout << endl;  */     
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
//             error_quad_real += solution_2ndds_real_all_quads_first_refine[n_q_points*cell_index + q_index][0];
            cellwise_error_real[cell_index] += pow(solution_2ndds_real_all_quads_first_refine[n_q_points*cell_index + q_index][0] - solution_2ndds_real_all_quads_second_refine[n_q_points*cell_index + q_index][0],2 ) * fe_values_velocity.JxW (q_index) ;
            cellwise_error_imag[cell_index] += pow(solution_2ndds_imag_all_quads_first_refine[n_q_points*cell_index + q_index][0] - solution_2ndds_imag_all_quads_second_refine[n_q_points*cell_index + q_index][0],2 ) * fe_values_velocity.JxW (q_index) ;
        }
//         cout << endl;        
        cellwise_error_real[cell_index] = pow(cellwise_error_real[cell_index], 0.5);
        cellwise_error_imag[cell_index] = pow(cellwise_error_imag[cell_index], 0.5);
        cellwise_error[cell_index] = pow(pow(cellwise_error_real[cell_index],2) + pow(cellwise_error_imag[cell_index],2), 0.5);
    }
    
    L2_error_2ndd_real = cellwise_error_real.l2_norm();
    L2_error_2ndd_imag = cellwise_error_imag.l2_norm();
    L2_error_2ndd = cellwise_error.l2_norm();
    
    
//     cout << "\n";

#endif
    
}
  



template <int dim>
void MixedLaplaceProblem_Complex_Validation<dim>::print_norms_errors_CPU ()
{
      
    cout << "\n";
    cout << "    L2_inte_solu: " << L2_inte_solu << ", real: " << L2_inte_solu_real << ", imag: " << L2_inte_solu_imag << endl;           
    cout << "    L2_inte_grad: " << L2_inte_grad << ", real: " << L2_inte_grad_real << ", imag: " << L2_inte_grad_imag << endl;
    
    cout << "\n";
    cout << "    L2_error_solu: " << L2_error_solu << ", real: " << L2_error_solu_real << ", imag: " << L2_error_solu_imag << "\n";
    cout << "    L2_error_grad: " << L2_error_grad << ", real: " << L2_error_grad_real << ", imag: " << L2_error_grad_imag << "\n";
    cout << "    L2_error_2ndd: " << L2_error_2ndd << ", real: " << L2_error_2ndd_real << ", imag: " << L2_error_2ndd_imag << "\n";
    
    cout << endl;
    cout << "    total_CPU_time:   " << total_CPU_time << "\n";
    
      
}


  template <int dim>
  void MixedLaplaceProblem_Complex_Validation<dim>::output_results () const
  {
      
//     ofstream output_solution_r("solution_solu_real_deg_"+to_string(degree-1)+"_ref_"+to_string(refine)+".txt");                                           //output solution_r to a file  
//     for (unsigned int i=0;i<n_p;i++)
//     {
//         output_solution_r << setprecision(20) << solution_solu_real[i] << "\n";
//     }   
//     output_solution_r.close();
    
/*    ofstream output_solution_i("solution_solu_real_imag_deg_"+to_string(degree-1)+"_ref_"+to_string(refine)+".txt");                                           //output solution_i to a file 
    for (unsigned int i=0;i<n_p;i++)
    {
        output_solution_i << setprecision(20) << solution_solu_imag[i] << "\n";      
    }   
    output_solution_i.close(); */ 
  
    vector<string> solution_names(dim, "u");
    solution_names.push_back ("p");
    vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector (dof_handler_single, solution_real, solution_names, interpretation);

    data_out.build_patches (degree+1);

    ofstream output ("solution.vtk");
    data_out.write_vtk (output);
  }


  template <int dim>
  void MixedLaplaceProblem_Complex_Validation<dim>::run ()
  {
    make_grid_and_dofs();
    
    print_fe_info(fe);
    
    setup_system ();
    
    assemble_system ();
       
    solve ();
    
    compute_l2_norms_numerically ();
    
    compute_errors_quad ();
    
//     print_from_a_header_file<dim>(L2_error_solu_real, solution); 
    
//     output_results ();

//     
    total_CPU_time += computing_timer.return_total_cpu_time ();
    if(total_CPU_time<0.001)
    {
        total_CPU_time = 0.001;
    }
    
    print_norms_errors_CPU ();
    
    printToFile();    
  }

#endif
