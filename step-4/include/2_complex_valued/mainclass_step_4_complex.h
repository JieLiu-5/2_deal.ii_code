
#ifndef MAINCLASS_STEP_4_COMPLEX_H
#define MAINCLASS_STEP_4_COMPLEX_H

#include"../../6_headers/1_cpp/1_dealii/base_class_complex.h"
#include <2_complex_valued/exact_solution_step_4_complex.h>

#include"../../6_headers/1_cpp/1_dealii/derived_class_complex.h"

template <int dim>
class Step4_Complex
{
public:
  Step4_Complex (const unsigned int id_case, 
                 const int id_quad_assem_incre, 
                 const unsigned int id_coeff_diff_value, 
                 const unsigned int id_coeff_helm_value, 
                 const double length, 
                 const unsigned int degree, 
                 const int grid_param);
  void run ();
  
  double solution_L2_error;
  double solution_H1_semi_error;
  double solution_H2_semi_error;
  
  double solution_L2_norm, solution_real_L2_norm, solution_imag_L2_norm;
  double solution_H1_semi_norm;
  double solution_H2_semi_norm;

  double r_N;
  double r_Np;
  double r_Npp;
private:
    
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void deal_with_solution ();
  void compute_l2_norm_numerically ();
  void computing_errors_built_in_core ();
  void compute_errors_custom ();
  void output_results ();

  const unsigned int   id_case;
  const unsigned int   id_quad_assem_incre;
  const unsigned int id_coeff_diff_value;
  const unsigned int id_coeff_helm_value;
  const double length;
  const unsigned int   degree; 
  const unsigned int   refine;
  
  const int n_dofs_custom;
  const int n_vertices_custom;
  
  QGauss<dim> quadrature_formula = QGauss<dim>(degree+id_quad_assem_incre);
  const unsigned int n_q_points = quadrature_formula.size();
  
  QGauss<dim-1> face_quadrature_formula= QGauss<dim-1>(degree+2);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  
  unsigned int is_global_mesh=1;
  Triangulation<dim>   triangulation;
  Triangulation<dim>   triangulation_first_refine;
  
  unsigned int id_cell = 0;
  
  FESystem<dim>        fe;
  FE_Q<dim>            fe_single;
  
  DoFHandler<dim>      dof_handler;
  DoFHandler<dim>      dof_handler_first_refine;
  DoFHandler<dim>      dof_handler_single;
  DoFHandler<dim>      dof_handler_single_first_refine;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       system_rhs;
  
  unsigned int is_boundary_info_printed=0;
  unsigned int is_dofhandler_info_printed=0;  
  
  unsigned int is_matrices_before_Diri_BC_printed = 0;
  unsigned int is_matrices_after_Diri_BC_printed = 0;
  
  string obj_string;
  unsigned int is_matrix_after_BC_stored = 0;
  unsigned int is_rhs_after_BC_stored = 0;
  
  unsigned int is_UMFPACK = 1;  
  
  Vector<double>       solution;
  Vector<double>       solution_zero;
  Vector<double>       solution_first_refine;

  Vector<double>       solution_amp;
  
  Vector<double>       solution_real;
  Vector<double>       solution_real_first_refine;
  Vector<double>       solution_real_sequence; 
  
  Vector<double>       solution_imag;
  Vector<double>       solution_imag_second_refine;
  Vector<double>       solution_imag_sequence;
  
  unsigned int is_solution_dealt=0;
  unsigned int is_solution_printed = 1;
  unsigned int is_solution_sequence_handled = 0;
  unsigned int is_solution_stored=0;  
  
  unsigned int is_results_outputted=1;
  
  unsigned int is_l2_norm_computed_numerically=1;
  unsigned int is_l2_norm_numerical_printed=0;
  
  unsigned int is_error_built_in = 0;  
  unsigned int is_error_l2_norm_printed_built_in = 0;  
  unsigned int is_error_custom = 1;
  unsigned int is_error_l2_norm_printed_custom = 0;  
    
  unsigned int n_components_number = 2;
  
  TimerOutput                               computing_timer;
  void printToFile();
  double total_CPU_time;
};


template <int dim>
Step4_Complex<dim>::Step4_Complex (const unsigned int id_case, const int id_quad_assem_incre, const unsigned int id_coeff_diff_value, const unsigned int id_coeff_helm_value, const double length, const unsigned int degree, const int grid_param)
  :
  id_case(id_case),
  id_quad_assem_incre(id_quad_assem_incre),
  id_coeff_diff_value(id_coeff_diff_value),
  id_coeff_helm_value(id_coeff_helm_value),
  length(length),
  degree(degree),
  refine(grid_param),
  n_dofs_custom(grid_param),
//   n_vertices_custom(grid_param),
  n_vertices_custom ((n_dofs_custom/2-1)/degree+1), 
  
  fe (FE_Q<dim>(degree),2),
  fe_single (degree),
  
  dof_handler (triangulation),
  dof_handler_first_refine (triangulation_first_refine),                            // important, do not mistake
  dof_handler_single (triangulation),
  dof_handler_single_first_refine (triangulation_first_refine),  
  computing_timer  (cout, TimerOutput::never, TimerOutput::cpu_times)
{
/*  cout << "================== \n"
       << "  The standard FEM for complex-valued problems\n"
       << "  case: " << id_case << "\n"
       << "  dimension: " << dim << "\n"
       << "  n_q_points: " << n_q_points << "\n"
       << "  n_face_q_points: " << n_face_q_points << "\n";    
  cout << "================== \n";  */  

//   if(id_case==1)
//   {
//     cout << "  exact solution does not exist for this case\n";
//   }
}


template <int dim>
void Step4_Complex<dim>::make_grid ()
{
  TimerOutput::Scope t(computing_timer, "make grid");
    
  if(is_global_mesh==1)
  {
    GridGenerator::hyper_cube (triangulation, 0, length);
  }else if(is_global_mesh == 0)
  {
    cout << "    n_dofs_custom: " << n_dofs_custom << endl;
    cout << "    n_vertices_custom: " << n_vertices_custom << endl;
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
    
    vector<CellData<dim>> cells(cell_vertices.size(), CellData<dim>());
    for (unsigned int i=0; i<cell_vertices.size(); ++i)
    {
        for (unsigned int j=0; j<GeometryInfo<1>::vertices_per_cell; ++j)
        {
            cells[i].vertices[j] = cell_vertices[i][j];
        }
    }
    
    triangulation.create_triangulation (vertices, cells, SubCellData());  
  }
  
  if (dim==1)
  {  
//   triangulation.begin_active()->face(1)->set_boundary_id(0);            // set both boundary conditions as Dirichlet type
//   triangulation.last_active()->face(1)->set_boundary_id(0);      
  }else if(dim==2)
  {
    triangulation.begin_active()->face(0)->set_boundary_id(1);       // left
//     triangulation.begin_active()->face(1)->set_boundary_id(1);       // right
//     triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom
//     triangulation.begin_active()->face(3)->set_boundary_id(1);       // top
  }

  if(is_boundary_info_printed==1)
  {
    print_boundary_info(triangulation);  
  }

  triangulation.refine_global (refine);  
  triangulation_first_refine.copy_triangulation(triangulation);
  
  dof_handler_first_refine.distribute_dofs (fe);
  dof_handler_single_first_refine.distribute_dofs (fe_single);
  
  obj_string = "dof_handler_single_first_refine";
  
  if(is_dofhandler_info_printed==1)
  {
    print_dofhandler_info(obj_string, dof_handler_single_first_refine);  
  }
//   save_coords_of_dofs(obj_string, dof_handler_single_first_refine);
  
}


template <int dim>
void Step4_Complex<dim>::setup_system ()
{
//   TimerOutput::Scope t(computing_timer, "setup_system");
  dof_handler.distribute_dofs (fe);
  dof_handler_single.distribute_dofs (fe_single);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  
  system_rhs.reinit (dof_handler.n_dofs());

  solution_amp.reinit (dof_handler_single.n_dofs());
  
  solution_real.reinit (dof_handler_single.n_dofs());
  solution_imag.reinit (dof_handler_single.n_dofs());
  solution_real_sequence.reinit (dof_handler_single.n_dofs());
  solution_imag_sequence.reinit (dof_handler_single.n_dofs());
}


template <int dim>
void Step4_Complex<dim>::assemble_system ()
{
  TimerOutput::Scope t(computing_timer, "assemble system");
//   cout << "Assembling\n";
  
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients | update_hessians |
                           update_quadrature_points | update_JxW_values);
  
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);  

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   dofs_per_face = fe.dofs_per_face;
  
  const RightHandSide_Complex<dim> right_hand_side(id_case, id_coeff_diff_value, id_coeff_helm_value);
  const Coeff_Diff_Complex<dim> obj_coeff_diff(id_case, id_coeff_diff_value, 1.0);
  const Coeff_Helm_Complex<dim> obj_coeff_helm(id_case, id_coeff_helm_value, 1.0);
  
  vector<Point<dim>> coords_of_quadrature_on_cell (n_q_points, Point<dim>());
  vector<Vector<double>> rhs_values (n_q_points,Vector<double>(2));
  vector<Tensor<2,dim, complex<double>>> coeff_diff_values(n_q_points,Tensor<2,dim, complex<double>>());
  vector<complex<double>> coeff_helm_values(n_q_points,complex<double>());
  
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
  
  const ExactSolution_Step_4_Complex_Both<dim> exact_solution_both(id_case);                // all cases are analyzed upon creating an object
  
  Tensor<1,dim, complex<double>> neumann_boundary;
  complex<double> neumann_boundary_times_normal_vector = 0.0 + 0.0i;

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    cell->get_dof_indices (local_dof_indices);
    
//     cout << "  local_dof_indices: \n";
//     print_vector(local_dof_indices);
    
    cell_matrix = 0;
    cell_rhs = 0;
    
    coords_of_quadrature_on_cell = fe_values.get_quadrature_points();
    
/*    cout << "  coords_of_quadrature_on_cell: \n";
    print_vector(coords_of_quadrature_on_cell); */   
    
    
    right_hand_side.vector_values_rhs(coords_of_quadrature_on_cell, rhs_values);
    obj_coeff_diff.value(coords_of_quadrature_on_cell, coeff_diff_values);
    obj_coeff_helm.value(coords_of_quadrature_on_cell, coeff_helm_values);

/*    cout << "  coeff_diff_values: \n" ;
    print_vector(coeff_diff_values);

    cout << "  coeff_helm_values: \n" ;
    print_vector(coeff_helm_values);
    
    cout << "  rhs_values: \n" ;
    print_vector(rhs_values);  */  
    
//     for (unsigned int i=0; i<dofs_per_cell; ++i)
//     {
//       cout << "  [" << i << "] dof #" << local_dof_indices[i] << ", ";
//       cout << "belonging to component " << fe.system_to_component_index(i).first << ", ";        
//         
//         cout << "  fe_values.shape_grad (i, 0): " << fe_values.shape_grad (i, 0) << endl;
//     }
    
    for (unsigned int q_index=0; q_index</*1*/ n_q_points; ++q_index)             //diagonal part
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
            
//           cell_matrix(i,j) += (gradient_times_coeff_diff_times_gradient.real()+gradient_times_coeff_diff_times_gradient.imag()                       // not applicable
//                                 + value_times_coeff_helm_times_value.real()+value_times_coeff_helm_times_value.imag()) *fe_values.JxW(q_index);
                                
          if (fe.system_to_component_index(i).first == 0)
          {
            if (fe.system_to_component_index(j).first == 0)               // we use this to justify the contribution of the real and imaginary part
            {                                                             // real or imag is only for the coefficient
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
        cell_rhs(i) += fe_values.shape_value (i, q_index) * rhs_values[q_index][fe.system_to_component_index(i).first] * fe_values.JxW (q_index);           // note we use "system_to_component_index" to identify the contribution of real and imaginary parts
      } 
    }
    
#if 1
    
    for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      if (cell->face(face_number)->at_boundary())
      {
//         cout << "  on face " << cell->face_index(face_number) << " of cell " << cell->active_cell_index() << "\n";
//         cout << "  with vertices ";
//         for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
//         {
//           cout << "(" << cell->face(face_number)->vertex(vertex) << ")";
//         }
        
        fe_face_values.reinit (cell, face_number);
        cell->face(face_number)->get_dof_indices (local_face_dof_indices);
        
/*        cout << "\n";
        cout << "  with normal vector " << fe_face_values.normal_vector(0) << "\n";
        cout << "  with dof #:\n"; 
        print_vector(local_face_dof_indices);*/        
        
        if(cell->face(face_number)->boundary_id() == 0)
        {
//           cout << "  Dirichlet BCs are imposed\n";
        }else if (cell->face(face_number)->boundary_id() == 1)
        {
//           cout << "  Neumann BCs are imposed\n";
        
          obj_coeff_diff.value(fe_face_values.get_quadrature_points(), coeff_diff_values_face);
          
/*          cout << "  coeff_diff_values_face: \n" ;
          print_vector(coeff_diff_values_face); */

          for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
          {
              
            if(id_case==1)
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
        system_matrix.add (local_dof_indices[i],
                           local_dof_indices[j],
                           cell_matrix(i,j));
      }
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }

  if(is_matrices_before_Diri_BC_printed==1)
  {
    cout << "  system_matrix before applying Dirichlet BCs is " << endl;
    system_matrix.print_formatted(cout);
    cout << '\n';

    cout << "  system_rhs before applying Dirichlet BCs is " << endl;
    cout << system_rhs << endl;
  }
  
  map<types::global_dof_index,double> boundary_values_for_pressure;                      // note the location of this function cluster
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            PressureBoundaryValues_Complex<dim>(id_case),
                                            boundary_values_for_pressure);

//   cout << "  boundary_values_for_pressure: \n"; 
//   print_map(boundary_values_for_pressure);

  MatrixTools::apply_boundary_values (boundary_values_for_pressure,
                                      system_matrix,
                                      solution,
                                      system_rhs);  
  
  if(is_matrices_after_Diri_BC_printed==1)
  {
    cout << "  system_matrix after applying Dirichlet BCs is " << endl;
    system_matrix.print_formatted(cout);
    cout << '\n';
    cout << "  system_rhs after applying Dirichlet BCs is " << endl;
    cout << system_rhs << endl;      
  }
  
  if(is_matrix_after_BC_stored==1)
  {
    obj_string="system_matrix_helm_xi_"+to_string(dim)+"d_deg_"+to_string(degree)+"_ref_"+to_string(refine);
    save_system_matrix_to_txt(obj_string,system_matrix);
  }
  if(is_rhs_after_BC_stored==1)
  {
    obj_string="system_rhs_oned_helm_xi_"+to_string(dim)+"d_deg_"+to_string(degree)+"_ref_"+to_string(refine);
    save_system_matrix_to_txt(obj_string,system_matrix);
  }
}


template <int dim>
void Step4_Complex<dim>::solve ()
{
  TimerOutput::Scope t(computing_timer, "solve");
//   cout << "Solving\n";

  if (is_UMFPACK == 1)
  {
//     cout << "  UMFPACK solver\n";
    SparseDirectUMFPACK  A_direct;                      
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
        
  }else if (is_UMFPACK == 0)
  {
//     cout << "  CG solver\n";
    SolverControl           solver_control (1e+8, 1e-16);
    SolverCG<>              solver (solver_control);
        
    solver.solve (system_matrix, solution, system_rhs,
                  PreconditionIdentity());
      
    cout << "   " << solver_control.last_value()
         << "  is the convergence value of last iteration step."
         << endl;
  }
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   dofs_per_cell_single = fe_single.dofs_per_cell;
  vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  vector<types::global_dof_index> local_dof_indices_single (dofs_per_cell_single);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  
  typename DoFHandler<dim>::active_cell_iterator
  cell_single = dof_handler_single.begin_active();

  unsigned int id_real=0;
  unsigned int id_imag=0;
  
  for (; cell!=endc; ++cell)
  {
//     cout << "active_cell_index(): " << cell->active_cell_index() << ", vertex(0): (" << cell->vertex(0) << "), at boundary: " << cell->at_boundary() << endl;
    
    cell->get_dof_indices (local_dof_indices);
    
    cell_single->get_dof_indices (local_dof_indices_single);
    cell_single++;
//     cout << "  local_dof_indices_single: \n" ;
//     print_vector(local_dof_indices_single);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
//       cout << "  [" << i << "] dof #" << local_dof_indices[i] << ", ";
//       cout << "belonging to component " << fe.system_to_component_index(i).first << ", ";
//       cout << "solution: " << solution[local_dof_indices[i]] << ", ";
//       if(fe.system_to_component_index(i).first==0)
//       {
//         cout << "goes to real " << local_dof_indices_single[id_real] << "\n";
//       }else if (fe.system_to_component_index(i).first==1)
//       {
//         cout << "goes to imag " << local_dof_indices_single[id_imag] << "\n";
//       }
      
      if(fe.system_to_component_index(i).first==0)
      {
        solution_real[local_dof_indices_single[id_real]]=solution[local_dof_indices[i]];
        id_real++;
      }else if (fe.system_to_component_index(i).first==1)
      {
        solution_imag[local_dof_indices_single[id_imag]]=solution[local_dof_indices[i]];
        id_imag++;
      }
    }
    id_real=0;
    id_imag=0;
  }
  
  for (unsigned int i = 0; i < dof_handler.n_dofs()/2.0; i++)
  {
    solution_amp[i] = sqrt(solution_real[i]*solution_real[i] + solution_imag[i]*solution_imag[i]);
  }
  
  if (is_solution_dealt == 1)
  {
    deal_with_solution();
  }   
  
}


template <int dim>
void Step4_Complex<dim>::deal_with_solution ()
{
  if(is_solution_printed==1)
  {
    cout << "Printing solution of the first refinement\n";
    cout << "  solution is\n  " << solution << endl; 
//     cout << "  solution_real is\n  " << solution_real << endl;
//     cout << "  solution_imag is\n  " << solution_imag << endl;
//     cout << "  solution_amp is\n  " << solution_amp << endl;
  }
  
  if(is_solution_stored==1)
  {
    obj_string = "solution_real";
    save_Vector_to_txt(obj_string, solution_real);
    
//   obj_string="solution_imag";
//   save_Vector_to_txt(obj_string,solution_imag);
//   obj_string="solution_amp";
//   save_Vector_to_txt(obj_string,solution_amp);
    
  }
  
  if(is_solution_sequence_handled==1)
  {
    solution_real_sequence[0]=solution_real[0];
    solution_imag_sequence[0]=solution_imag[0];
    
    for (unsigned int i = 0; i < triangulation.n_active_cells(); i++)
    {
        solution_real_sequence[i*degree+degree] = solution_real[i*degree+1];
        solution_imag_sequence[i*degree+degree] = solution_imag[i*degree+1];
            
        for (unsigned int j = 0; j < degree-1; j++)
        {
        solution_real_sequence[i*degree+j+1] = solution_real[degree*i+j+2];
        solution_imag_sequence[i*degree+j+1] = solution_imag[degree*i+j+2];
        }
    }
      
    cout << "solution_real_sequence is: " << endl << solution_real_sequence << endl;
    cout << "solution_imag_sequence is: " << endl << solution_imag_sequence << endl;  
  }
}


template <int dim>
void Step4_Complex<dim>::output_results ()
{
  TimerOutput::Scope t(computing_timer, "output results");
  cout << "Outputting results\n";
  
  //    <dim> velocity1;
  //Compute2ndderivative1<dim> secondderivative1;
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  ofstream output (dim == 1 ?
                        "solution-complex-sm-1d.vtk" :
                        "solution-complex-sm-2d.vtk");
  data_out.write_vtk (output);
}

  
template <int dim>
void Step4_Complex<dim>::compute_l2_norm_numerically ()
{
//   cout << "Computing l2 norms using the numerical solution\n";

  Vector<double> difference_per_cell (triangulation.n_active_cells());
  Vector<double> difference_per_cell_real (triangulation.n_active_cells());                 // L2 norm of the real part of the solution in each cell
  Vector<double> difference_per_cell_imag (triangulation.n_active_cells());                 // L2 norm of the imag part of the solution in each cell

  vector<Vector<double>> difference_per_cell_both (triangulation.n_active_cells(),Vector<double>(2));
  
  VectorTools::integrate_difference (dof_handler_single,                                       // real part of the solution
                                     solution_real,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_real,                              
                                     QGauss<dim>(degree+2),
                                     VectorTools::L2_norm);
  solution_real_L2_norm = difference_per_cell_real.l2_norm();
  
  
  VectorTools::integrate_difference (dof_handler_single,                                       // imaginary part of the solution
                                     solution_imag,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_imag,
                                     QGauss<dim>(degree+2),
                                     VectorTools::L2_norm);
  solution_imag_L2_norm = difference_per_cell_imag.l2_norm(); 
  
   
  solution_L2_norm = pow(pow(solution_real_L2_norm,2)+pow(solution_imag_L2_norm,2),0.5);        // which is the most convenient way
  
//   cout << "  solution_L2_norm of method 1: " << solution_L2_norm << endl;  
//   
//   for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
//   {
//     difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2), 0.5);
//   }
//   solution_L2_norm = difference_per_cell.l2_norm();
//   cout << "  solution_L2_norm of method 2: " << solution_L2_norm << endl;
//   
  
//   VectorTools::integrate_difference (dof_handler,                     // having tried all parts of the solution, failed due to the unavailability of integrate_difference_inner()
//                                      solution,
//                                      ZeroFunction<dim>(),
//                                      difference_per_cell_both,                              
//                                      QGauss<dim>(degree+2),
//                                      VectorTools::L2_norm);
//   
//   cout << "  difference_per_cell_both: \n";
//   print_vector(difference_per_cell_both);
//   solution_L2_norm = difference_per_cell_both.l2_norm();
  
//   cout << "  solution_L2_norm of method 3: " << solution_L2_norm << endl;


  VectorTools::integrate_difference (dof_handler_single,                                       // gradient     
                                     solution_real,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_real,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H1_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_imag,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H1_seminorm);
  
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  solution_H1_semi_norm = difference_per_cell.l2_norm();


  VectorTools::integrate_difference (dof_handler_single,                                       // 2nd derivative                                   
                                     solution_real,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_real,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H2_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     ZeroFunction<dim>(),
                                     difference_per_cell_imag,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H2_seminorm);
  
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  solution_H2_semi_norm = difference_per_cell.l2_norm();
  
  if(is_l2_norm_numerical_printed==1)
  {
    cout << "  @norms\n";
    cout << "  solution_L2_norm: " << solution_L2_norm << ", ";
    cout << "real: " << solution_real_L2_norm << ", ";
    cout << "imag: " << solution_imag_L2_norm << "\n";
    cout << "  solution_H1_semi_norm: " << solution_H1_semi_norm << "\n";
    cout << "  solution_H2_semi_norm: " << solution_H2_semi_norm << "\n";
  }
  
}

    
    
template <int dim>
void Step4_Complex<dim>::computing_errors_built_in_core ()
{
  TimerOutput::Scope t(computing_timer, "computing errors built-in");
//   cout << "  Invoking core\n";
    
  ExactSolution_Step_4_Complex_Real<dim> exact_solution_real(id_case);  
  ExactSolution_Step_4_Complex_Imag<dim> exact_solution_imag(id_case); 

  Vector<double> difference_per_cell (triangulation.n_active_cells());
  Vector<double> difference_per_cell_real (triangulation.n_active_cells());                 // L2 norm of the real part of the solution in each cell
  Vector<double> difference_per_cell_imag (triangulation.n_active_cells());                 // L2 norm of the imag part of the solution in each cell
  
  VectorTools::integrate_difference (dof_handler_single,                                      
                                     solution_real,
                                     exact_solution_real,
                                     difference_per_cell_real,
                                     QGauss<dim>(degree+2),
                                     VectorTools::L2_norm);
  
  VectorTools::integrate_difference (dof_handler_single,                                      
                                     solution_imag,
                                     exact_solution_imag,
                                     difference_per_cell_imag,
                                     QGauss<dim>(degree+2),
                                     VectorTools::L2_norm);
  
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
//   cout << "  difference_per_cell_real: " << difference_per_cell_real << "\n";
//   cout << "  difference_per_cell_imag: " << difference_per_cell_imag << "\n";
   
  solution_L2_error = difference_per_cell.l2_norm();
  r_N = solution_L2_error/solution_L2_norm;

  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_real,
                                     exact_solution_real,
                                     difference_per_cell_real,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H1_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     exact_solution_imag,
                                     difference_per_cell_imag,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H1_seminorm);
  
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  solution_H1_semi_error = difference_per_cell.l2_norm();
  r_Np = solution_H1_semi_error/solution_H1_semi_norm;
  
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_real,
                                     exact_solution_real,
                                     difference_per_cell_real,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H2_seminorm);
  
  VectorTools::integrate_difference (dof_handler_single,                                       
                                     solution_imag,
                                     exact_solution_imag,
                                     difference_per_cell_imag,
                                     QGauss<dim>(degree+2),
                                     VectorTools::H2_seminorm);
  
  for(unsigned int i = 0; i<triangulation.n_active_cells(); ++i)
  {
    difference_per_cell[i] = pow(pow(difference_per_cell_real[i], 2)+pow(difference_per_cell_imag[i], 2),0.5);
  }
  
  solution_H2_semi_error = difference_per_cell.l2_norm();
  r_Npp = solution_H2_semi_error/solution_H2_semi_norm;
  
  if (is_error_l2_norm_printed_built_in==1)
  {
    cout << "  @errors\n";
    cout << "  solution_L2_error: " << solution_L2_error << "\n";
    cout << "  solution_H1_semi_error: " << solution_H1_semi_error << "\n";
    cout << "  solution_H2_semi_error: " << solution_H2_semi_error << "\n";
    
/*    cout << "\n";
    cout << "  r_N = " << r_N << "\n";  
    cout << "  r_Np = " << r_Np << "\n";  
    cout << "  r_Npp = " << r_Npp << "\n"; */     
  }
}


template <int dim>
void Step4_Complex<dim>::compute_errors_custom ()
{
//   cout << "Computing errors customly\n";
  
  solution_zero.reinit (dof_handler.n_dofs());
  solution_first_refine.reinit (dof_handler.n_dofs());
  solution_first_refine=solution;
  
//   cout << "//////////////////\n";
  
  triangulation.refine_global ();
  setup_system ();
  assemble_system ();
  solve ();  

  vector<double> vector_error(3);
  vector<double> vector_l2_norm(3);
  error_computation_custom_core_std(solution_first_refine, solution, dof_handler_first_refine, dof_handler, quadrature_formula, n_components_number,vector_error);
  error_computation_custom_core_std(solution_zero, solution, dof_handler_first_refine, dof_handler, quadrature_formula, n_components_number,vector_l2_norm);

//   cout << "//////////////////\n";   
  
  solution_L2_error = vector_error[0];
  solution_H1_semi_error = vector_error[1];
  solution_H2_semi_error = vector_error[2];
  
  if(is_error_l2_norm_printed_custom==1)
  {
    cout << "  @errors custom\n";
    cout << "  solution_L2_error = " << solution_L2_error << "\n";
    cout << "  solution_H1_semi_error = " << solution_H1_semi_error << "\n";
    cout << "  solution_H2_semi_error = " << solution_H2_semi_error << "\n";
    
    cout << "  @l2 norms custom\n";
    print_vector(vector_l2_norm);
  }
}


ofstream myfile;
template <int dim>
void Step4_Complex<dim>::printToFile()
{
  if(dim==1)
  {
    obj_string="1d";  
  }else if(dim==2)
  {
    obj_string="2d";
  }
  myfile.open ("data_error_"+obj_string+"_sm_complex.txt", ofstream::trunc);
  
  myfile << refine <<" ";
  myfile << triangulation_first_refine.n_vertices() <<" ";    
  myfile << dof_handler_single_first_refine.n_dofs() * 2 << " ";    
  myfile << solution_L2_error << " ";
  myfile << solution_H1_semi_error << " ";
  myfile << solution_H2_semi_error << " ";
  myfile << total_CPU_time << " ";
  myfile << solution_L2_norm <<"\n";
}


template <int dim>
void Step4_Complex<dim>::run ()
{
  make_grid();
  
  setup_system ();
  assemble_system ();
  
  solve ();
  
  if(dim == 2 && is_results_outputted==1)
  {
    output_results ();
  }
  
  if(is_l2_norm_computed_numerically==1)
  {
    compute_l2_norm_numerically();
  }

  if (is_error_built_in==1)
  {
//     cout << "Trying to compute the error using the built-in function\n";
    if (id_case==1)
    {
      cout << "    !!! exact solution not existent, failed\n";
    }else
    {
      computing_errors_built_in_core ();  
    }
  }
  
  if(is_error_custom==1)
  {
    compute_errors_custom ();  
  }
  
  total_CPU_time += computing_timer.return_total_cpu_time ();
  if(total_CPU_time<0.001)
  {
      total_CPU_time = 0.001;
  }
  
  printToFile();
  
//   computing_timer.print_summary();
  
//   cout << "  @cpu time\n";
//   cout << "  total_CPU_time: " << total_CPU_time << "\n\n";
//   
}



#endif
