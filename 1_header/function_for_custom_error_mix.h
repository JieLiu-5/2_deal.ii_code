
#ifndef FUNCTION_FOR_CUSTOM_ERROR_MIX_H
#define FUNCTION_FOR_CUSTOM_ERROR_MIX_H


template <int dim>
void error_computation_custom_real_mix_core(BlockVector<double>& solution_first_refine, BlockVector<double>& solution_second_refine, DoFHandler<dim>& dof_handler_first_refine, DoFHandler<dim>& dof_handler_second_refine, vector<Vector<double>>& cellwise_error)
{
  cout << "  error_computation_custom_core_mix()\n";
  
  unsigned int n_q_points_one_direction = dof_handler_first_refine.get_fe().degree+2;
  unsigned int n_q_points = pow(n_q_points_one_direction,dim);
  QGauss<dim>  quadrature_formula(n_q_points_one_direction);
  
  cout << "  n_q_points for computing error customly: " << n_q_points << "\n";

  double l2_error_per_cell_soln = 0.0, l2_error_per_cell_grad = 0.0, l2_error_per_cell_2ndd = 0.0;
  
  int cell_index_first_refine = 0;
  
  vector<Point<dim> > quad_coords_vector = quadrature_formula.get_points();
  
//   cout << "  quad_coords_vector: \n";
//   print_vector(quad_coords_vector);
  
  vector<Vector<double>> solution_all_components_values_first_refine_at_quads_per_cell(n_q_points, Vector<double>(dim+1));   // dim for the nr. of components of the velocity, 1 for the pressure
  vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_first_refine_at_quads_per_cell(n_q_points);
  vector<Vector<double> > solution_all_components_values_second_refine_at_quads_per_cell(n_q_points, Vector<double>(dim+1));
  vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_second_refine_at_quads_per_cell(n_q_points);
//   vector<Vector<double> > solution_all_components_values_difference_at_quads_per_cell(n_q_points, Vector<double>(dim+1));
  for (unsigned int i=0; i<n_q_points; ++i)
  {
    solution_all_components_grads_first_refine_at_quads_per_cell[i].resize(dim+1);
    solution_all_components_grads_second_refine_at_quads_per_cell[i].resize(dim+1);
  }
  

  FEValues<dim> fe_values (dof_handler_first_refine.get_fe(), quadrature_formula,
                          update_values   | update_gradients | update_hessians |
                          update_quadrature_points | update_JxW_values);
  
  typename DoFHandler<dim>::active_cell_iterator  
  cell_first_refine = dof_handler_first_refine.begin_active(),
  endc_first_refine = dof_handler_first_refine.end();  
  
  typename DoFHandler<dim>::active_cell_iterator  
  cell_second_refine = dof_handler_second_refine.begin_active();
  
  if(dim==1)
  {
    unsigned int n_q_points_one_side = (n_q_points+1)/2.0;
    cout << "  # of quadrature points on one side: " << n_q_points_one_side << "\n";
    
    vector<Point<dim> > left_quad_vector;
    vector<Point<dim> > right_quad_vector;
  
    vector<Vector<double> > solution_all_components_values_second_refine_at_quads_per_cell_left(n_q_points_one_side, Vector<double>(dim+1));
    vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_second_refine_at_quads_per_cell_left(n_q_points_one_side);
    vector<Vector<double> > solution_all_components_values_second_refine_at_quads_per_cell_right(n_q_points_one_side, Vector<double>(dim+1));
    vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_second_refine_at_quads_per_cell_right(n_q_points_one_side);
    
    for (unsigned int i=0; i<n_q_points_one_side; ++i)
    {
      solution_all_components_grads_second_refine_at_quads_per_cell_left[i].resize(dim+1);
      solution_all_components_grads_second_refine_at_quads_per_cell_right[i].resize(dim+1);
    }
    
    for (unsigned int i=0; i<n_q_points;++i)
    {
//     cout << "  [" << i << "]: " << quad_coords_vector[i] << " ";
      if(quad_coords_vector[i](0)<=0.5)
      {
        left_quad_vector.push_back(quad_coords_vector[i]);
      }
      
      if (quad_coords_vector[i](0)>=0.5)
      {
        right_quad_vector.push_back(quad_coords_vector[i]);
      }
    }
  
//     cout << "  quad_vector after splitting: \n";
//     cout << "  left_quad_vector: \n";
//     print_vector(left_quad_vector);    
//     cout << "  right_quad_vector: \n";
//     print_vector(right_quad_vector);
    
    scale_quad_coords_small(left_quad_vector);
    scale_quad_coords_big(right_quad_vector);
    
//     cout << "  quad_vector after scaling: \n";
//     cout << "  left_quad_vector: \n";
//     print_vector(left_quad_vector);    
//     cout << "  right_quad_vector: \n";
//     print_vector(right_quad_vector);
    
    Quadrature<dim>  quadrature_formula_high_2_low_left(left_quad_vector);
    FEValues<dim> fe_values_high_2_low_left (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_left,
                                                    update_values   | update_gradients | update_hessians |
                                                    update_quadrature_points | update_JxW_values);

    Quadrature<dim>  quadrature_formula_high_2_low_right(right_quad_vector);  
    FEValues<dim> fe_values_high_2_low_right (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_right,
                                                    update_values   | update_gradients | update_hessians |
                                                    update_quadrature_points | update_JxW_values);

    for (; cell_first_refine!=endc_first_refine; ++cell_first_refine)
    {
//       cout << "  \ncell_first_refine " << cell_first_refine->active_cell_index() << ", vertex(0): " << cell_first_refine->vertex(0) << "\n";
      fe_values.reinit (cell_first_refine);
      fe_values.get_function_values (solution_first_refine, solution_all_components_values_first_refine_at_quads_per_cell);  
      fe_values.get_function_gradients (solution_first_refine, solution_all_components_grads_first_refine_at_quads_per_cell);  
        
//       cout << "  solution_all_components_values_first_refine_at_quads_per_cell: \n";
//       print_vector(solution_all_components_values_first_refine_at_quads_per_cell);
//       
//       cout << "  solution_all_components_grads_first_refine_at_quads_per_cell: \n";
//       print_tensor(solution_all_components_grads_first_refine_at_quads_per_cell);

//       cout << "  cell_second_refine " << cell_second_refine->active_cell_index() << ", vertex(0): " << cell_second_refine->vertex(0) << "\n";
      
      fe_values_high_2_low_left.reinit (cell_second_refine);
      
      fe_values_high_2_low_left.get_function_values(solution_second_refine,solution_all_components_values_second_refine_at_quads_per_cell_left);
      fe_values_high_2_low_left.get_function_gradients(solution_second_refine,solution_all_components_grads_second_refine_at_quads_per_cell_left);
      cell_second_refine++;

//       cout << "  solution_all_components_values_second_refine_at_quads_per_cell_left: \n";
//       print_vector(solution_all_components_values_second_refine_at_quads_per_cell_left);
//       
//       cout << "  solution_all_components_grads_second_refine_at_quads_per_cell_left: \n";
//       print_tensor(solution_all_components_grads_second_refine_at_quads_per_cell_left);
      
      
      for (unsigned int i=0; i<n_q_points_one_side;++i)
      {
        solution_all_components_values_second_refine_at_quads_per_cell[i]=solution_all_components_values_second_refine_at_quads_per_cell_left[i];
        solution_all_components_grads_second_refine_at_quads_per_cell[i]=solution_all_components_grads_second_refine_at_quads_per_cell_left[i];
      }
      
//       cout << "  cell_second_refine " << cell_second_refine->active_cell_index() << ", vertex(0): " << cell_second_refine->vertex(0) << "\n";
      fe_values_high_2_low_right.reinit (cell_second_refine);
      fe_values_high_2_low_right.get_function_values(solution_second_refine,solution_all_components_values_second_refine_at_quads_per_cell_right);
      fe_values_high_2_low_right.get_function_gradients(solution_second_refine,solution_all_components_grads_second_refine_at_quads_per_cell_right);
      cell_second_refine++;
        
//       cout << "  solution_all_components_values_second_refine_at_quads_per_cell_right: \n";
//       print_vector(solution_all_components_values_second_refine_at_quads_per_cell_right);      
// 
//       cout << "  solution_all_components_grads_second_refine_at_quads_per_cell_right: \n";
//       print_tensor(solution_all_components_grads_second_refine_at_quads_per_cell_right);
      
      
      if (2*n_q_points_one_side==n_q_points)
      {
        for (unsigned int i=0; i<n_q_points_one_side;++i)
        {
            solution_all_components_values_second_refine_at_quads_per_cell[n_q_points_one_side+i]=solution_all_components_values_second_refine_at_quads_per_cell_right[i];
            solution_all_components_grads_second_refine_at_quads_per_cell[n_q_points_one_side+i]=solution_all_components_grads_second_refine_at_quads_per_cell_right[i];
        }
      }else
      {
        for (unsigned int i=1; i<n_q_points_one_side;++i)                           // quadrature points of the left vector and the right vector overlap
        {
          solution_all_components_values_second_refine_at_quads_per_cell[n_q_points_one_side+i-1]=solution_all_components_values_second_refine_at_quads_per_cell_right[i];
          solution_all_components_grads_second_refine_at_quads_per_cell[n_q_points_one_side+i-1]=solution_all_components_grads_second_refine_at_quads_per_cell_right[i];
        }
      }    

//       cout << "  solution_all_components_values_second_refine_at_quads_per_cell: \n";
//       print_vector(solution_all_components_values_second_refine_at_quads_per_cell);
//       cout << "  solution_all_components_grads_second_refine_at_quads_per_cell: \n";
//       print_tensor(solution_all_components_grads_second_refine_at_quads_per_cell);

      cell_index_first_refine = cell_first_refine->active_cell_index();

      for (unsigned int q_index = 0; q_index<n_q_points; ++q_index)
      {
        l2_error_per_cell_grad+=pow(solution_all_components_values_second_refine_at_quads_per_cell[q_index][0]-solution_all_components_values_first_refine_at_quads_per_cell[q_index][0],2.0)*fe_values.JxW (q_index);
        
        l2_error_per_cell_2ndd+=pow(solution_all_components_grads_second_refine_at_quads_per_cell[q_index][0][0]-solution_all_components_grads_first_refine_at_quads_per_cell[q_index][0][0],2.0)*fe_values.JxW (q_index);
        
        l2_error_per_cell_soln+=pow(solution_all_components_values_second_refine_at_quads_per_cell[q_index][dim]-solution_all_components_values_first_refine_at_quads_per_cell[q_index][dim],2.0)*fe_values.JxW (q_index);
      }
        
      cellwise_error[1][cell_index_first_refine]=sqrt(l2_error_per_cell_grad);
      cellwise_error[2][cell_index_first_refine]=sqrt(l2_error_per_cell_2ndd);
      cellwise_error[0][cell_index_first_refine]=sqrt(l2_error_per_cell_soln);

      l2_error_per_cell_grad=0.0;
      l2_error_per_cell_2ndd=0.0;
      l2_error_per_cell_soln=0.0;
    }
  }else if(dim == 2)
  {
    unsigned int n_q_points_per_block=pow(int((sqrt(n_q_points)+1)/2.0),2);
    cout << "  # of quadrature points in one block: " << n_q_points_per_block << "\n";  
        
    vector<Point<dim> > quad_coords_vector_0_0;                            // split quad_coords_vector into blocks
    vector<Point<dim> > quad_coords_vector_1_0;
    vector<Point<dim> > quad_coords_vector_0_1;
    vector<Point<dim> > quad_coords_vector_1_1;
  
    vector<int> quad_indices_vector_0_0;
    vector<int> quad_indices_vector_1_0;
    vector<int> quad_indices_vector_0_1;
    vector<int> quad_indices_vector_1_1;
  
    vector<int> quad_indices_vector;
  
    for (unsigned int i=0; i<quad_coords_vector.size();++i)
    {
//     cout << "  [" << i << "]: " << quad_coords_vector[i] << " ";
    
      if(quad_coords_vector[i](0)<=0.5 && quad_coords_vector[i](1)<=0.5)
      {
//       cout << "  to 0_0";
        quad_coords_vector_0_0.push_back(quad_coords_vector[i]);
        quad_indices_vector_0_0.push_back(i);
      }
      if (quad_coords_vector[i](0)>=0.5 && quad_coords_vector[i](1)<=0.5)
      {
//       cout << "  to 1_0";
        quad_coords_vector_1_0.push_back(quad_coords_vector[i]);
        quad_indices_vector_1_0.push_back(i);
      }
      if(quad_coords_vector[i](0)<=0.5 && quad_coords_vector[i](1)>=0.5)
      {
//       cout << "  to 0_1";
        quad_coords_vector_0_1.push_back(quad_coords_vector[i]);
        quad_indices_vector_0_1.push_back(i);
      }
      if(quad_coords_vector[i](0)>=0.5 && quad_coords_vector[i](1)>=0.5)
      {
//       cout << "  to 1_1";
        quad_coords_vector_1_1.push_back(quad_coords_vector[i]);
        quad_indices_vector_1_1.push_back(i);
      }
//     cout << "\n";
    }
  
    quad_indices_vector.insert(quad_indices_vector.end(),quad_indices_vector_0_0.begin(),quad_indices_vector_0_0.end());
    quad_indices_vector.insert(quad_indices_vector.end(),quad_indices_vector_1_0.begin(),quad_indices_vector_1_0.end());
    quad_indices_vector.insert(quad_indices_vector.end(),quad_indices_vector_0_1.begin(),quad_indices_vector_0_1.end());
    quad_indices_vector.insert(quad_indices_vector.end(),quad_indices_vector_1_1.begin(),quad_indices_vector_1_1.end());
  
//   cout << "\n";
//   cout << "  quad_indices_vector_0_0\n";
//   print_vector(quad_indices_vector_0_0);
//   cout << "  quad_indices_vector_1_0\n";
//   print_vector(quad_indices_vector_1_0);
//   cout << "  quad_indices_vector_0_1\n";
//   print_vector(quad_indices_vector_0_1);
//   cout << "  quad_indices_vector_1_1\n";
//   print_vector(quad_indices_vector_1_1);
  
//   cout << "  quad_indices_vector\n";
//   print_vector(quad_indices_vector);
//   
//   cout << "\n";
//   cout << "  quad_coords_vector_0_0\n";
//   print_vector(quad_coords_vector_0_0);
//   cout << "  quad_coords_vector_1_0\n";
//   print_vector(quad_coords_vector_1_0);
//   cout << "  quad_coords_vector_0_1\n";
//   print_vector(quad_coords_vector_0_1);
//   cout << "  quad_coords_vector_1_1\n";
//   print_vector(quad_coords_vector_1_1);
  
  
//   obj_string="data_quad_coords_adjusted";
//   save_vector_to_txt(obj_string,quad_coords_vector);

    scale_quad_coords_small(quad_coords_vector_0_0);                        // scale quad_coords_vector_i_j
    scale_quad_coords_1_0(quad_coords_vector_1_0);
    scale_quad_coords_0_1(quad_coords_vector_0_1);
    scale_quad_coords_big(quad_coords_vector_1_1);
  
//     cout << "  quad_coords_vector_i_j after scaling: \n";
//     cout << "  quad_coords_vector_0_0\n";
//     print_vector(quad_coords_vector_0_0);
//     cout << "  quad_coords_vector_1_0\n";
//     print_vector(quad_coords_vector_1_0);
//     cout << "  quad_coords_vector_0_1\n";
//     print_vector(quad_coords_vector_0_1);
//     cout << "  quad_coords_vector_1_1\n";
//     print_vector(quad_coords_vector_1_1);
  
  
//     obj_string="data_quad_coords_0_0";
//     save_vector_to_txt(obj_string,quad_coords_vector_0_0);
//     obj_string="data_quad_coords_1_0";
//     save_vector_to_txt(obj_string,quad_coords_vector_1_0);
//     obj_string="data_quad_coords_0_1";
//     save_vector_to_txt(obj_string,quad_coords_vector_0_1);
//     obj_string="data_quad_coords_1_1";
//     save_vector_to_txt(obj_string,quad_coords_vector_1_1);
  
//     cout << "\n";
  
    Quadrature<dim>  quadrature_formula_high_2_low_0_0(quad_coords_vector_0_0);
    Quadrature<dim>  quadrature_formula_high_2_low_1_0(quad_coords_vector_1_0);
    Quadrature<dim>  quadrature_formula_high_2_low_0_1(quad_coords_vector_0_1);
    Quadrature<dim>  quadrature_formula_high_2_low_1_1(quad_coords_vector_1_1);
  
    FEValues<dim> fe_values_high_2_low_0_0 (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_0_0,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_high_2_low_1_0 (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_1_0,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_high_2_low_0_1 (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_0_1,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_high_2_low_1_1 (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_1_1,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
  
  
    vector<Vector<double> > solution_all_components_values_second_refine_at_quads_per_cell_0_0(n_q_points_per_block,Vector<double>(dim+1));
    vector<Vector<double> > solution_all_components_values_second_refine_at_quads_per_cell_1_0(n_q_points_per_block,Vector<double>(dim+1));
    vector<Vector<double> > solution_all_components_values_second_refine_at_quads_per_cell_0_1(n_q_points_per_block,Vector<double>(dim+1));
    vector<Vector<double> > solution_all_components_values_second_refine_at_quads_per_cell_1_1(n_q_points_per_block,Vector<double>(dim+1));
    
    vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_second_refine_at_quads_per_cell_0_0(n_q_points_per_block);
    vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_second_refine_at_quads_per_cell_1_0(n_q_points_per_block);
    vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_second_refine_at_quads_per_cell_0_1(n_q_points_per_block);
    vector<vector<Tensor<1,dim,double>> > solution_all_components_grads_second_refine_at_quads_per_cell_1_1(n_q_points_per_block);


    for (unsigned int i=0; i<n_q_points_per_block; ++i)
    {
      solution_all_components_grads_second_refine_at_quads_per_cell_0_0[i].resize(dim+1);
      solution_all_components_grads_second_refine_at_quads_per_cell_1_0[i].resize(dim+1);
      solution_all_components_grads_second_refine_at_quads_per_cell_0_1[i].resize(dim+1);
      solution_all_components_grads_second_refine_at_quads_per_cell_1_1[i].resize(dim+1);
    }
    
    for (; cell_first_refine!=endc_first_refine; ++cell_first_refine)
    {
//       cout << "  \ncell_first_refine " << cell_first_refine->active_cell_index() << ", vertex(0): " << cell_first_refine->vertex(0) << "\n";
      
      fe_values.reinit (cell_first_refine);
      fe_values.get_function_values (solution_first_refine, solution_all_components_values_first_refine_at_quads_per_cell);  
      fe_values.get_function_gradients (solution_first_refine, solution_all_components_grads_first_refine_at_quads_per_cell);  
      
//       cout << "  solution_all_components_values_first_refine_at_quads_per_cell: \n";
//       print_vector(solution_all_components_values_first_refine_at_quads_per_cell);
//       cout << "  solution_all_components_grads_first_refine_at_quads_per_cell: \n";
//       print_tensor(solution_all_components_grads_first_refine_at_quads_per_cell);

//       cout << "  cell_second_refine " << cell_second_refine->active_cell_index() << ", vertex(0): " << cell_second_refine->vertex(0) << "\n";
      fe_values_high_2_low_0_0.reinit (cell_second_refine);
      fe_values_high_2_low_0_0.get_function_values(solution_second_refine,solution_all_components_values_second_refine_at_quads_per_cell_0_0);
      fe_values_high_2_low_0_0.get_function_gradients(solution_second_refine,solution_all_components_grads_second_refine_at_quads_per_cell_0_0);
      cell_second_refine++;
      
//       cout << "  cell_second_refine " << cell_second_refine->active_cell_index() << ", vertex(0): " << cell_second_refine->vertex(0) << "\n";
      fe_values_high_2_low_1_0.reinit (cell_second_refine);
      fe_values_high_2_low_1_0.get_function_values(solution_second_refine,solution_all_components_values_second_refine_at_quads_per_cell_1_0);
      fe_values_high_2_low_1_0.get_function_gradients(solution_second_refine,solution_all_components_grads_second_refine_at_quads_per_cell_1_0);
      cell_second_refine++;
      
//       cout << "  cell_second_refine " << cell_second_refine->active_cell_index() << ", vertex(0): " << cell_second_refine->vertex(0) << "\n";
      fe_values_high_2_low_0_1.reinit (cell_second_refine);
      fe_values_high_2_low_0_1.get_function_values(solution_second_refine,solution_all_components_values_second_refine_at_quads_per_cell_0_1);
      fe_values_high_2_low_0_1.get_function_gradients(solution_second_refine,solution_all_components_grads_second_refine_at_quads_per_cell_0_1);
      cell_second_refine++;
      
//       cout << "  cell_second_refine " << cell_second_refine->active_cell_index() << ", vertex(0): " << cell_second_refine->vertex(0) << "\n";
      fe_values_high_2_low_1_1.reinit (cell_second_refine);
      fe_values_high_2_low_1_1.get_function_values(solution_second_refine,solution_all_components_values_second_refine_at_quads_per_cell_1_1);
      fe_values_high_2_low_1_1.get_function_gradients(solution_second_refine,solution_all_components_grads_second_refine_at_quads_per_cell_1_1);
      cell_second_refine++;
      
      for (unsigned int i=0; i<n_q_points_per_block; ++i)
      {
        solution_all_components_values_second_refine_at_quads_per_cell[quad_indices_vector[i]]=solution_all_components_values_second_refine_at_quads_per_cell_0_0[i];
        solution_all_components_values_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_per_block*1+i]]=solution_all_components_values_second_refine_at_quads_per_cell_1_0[i];
        solution_all_components_values_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_per_block*2+i]]=solution_all_components_values_second_refine_at_quads_per_cell_0_1[i];
        solution_all_components_values_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_per_block*3+i]]=solution_all_components_values_second_refine_at_quads_per_cell_1_1[i];

        solution_all_components_grads_second_refine_at_quads_per_cell[quad_indices_vector[i]]=solution_all_components_grads_second_refine_at_quads_per_cell_0_0[i];
        solution_all_components_grads_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_per_block*1+i]]=solution_all_components_grads_second_refine_at_quads_per_cell_1_0[i];
        solution_all_components_grads_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_per_block*2+i]]=solution_all_components_grads_second_refine_at_quads_per_cell_0_1[i];
        solution_all_components_grads_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_per_block*3+i]]=solution_all_components_grads_second_refine_at_quads_per_cell_1_1[i];
      }      
      
//       cout << "  solution_all_components_values_second_refine_at_quads_per_cell: \n";
//       print_vector(solution_all_components_values_second_refine_at_quads_per_cell);
//       cout << "  solution_all_components_grads_second_refine_at_quads_per_cell: \n";
//       print_tensor(solution_all_components_grads_second_refine_at_quads_per_cell);
      
//       for (unsigned int q_index = 0; q_index<n_q_points; ++q_index)
//       {
//         for (unsigned int j=0; j<dim+1; ++j)
//         {
//           solution_all_components_values_difference_at_quads_per_cell[q_index][j]=solution_all_components_values_second_refine_at_quads_per_cell[q_index][j]-solution_all_components_values_first_refine_at_quads_per_cell[q_index][j];
//         }
//       }
      
//       cout << "  solution_all_components_values_difference_at_quads_per_cell: \n";
//       print_vector(solution_all_components_values_difference_at_quads_per_cell);      
      
      cell_index_first_refine = cell_first_refine->active_cell_index();
      
      for (unsigned int q_index = 0; q_index<n_q_points; ++q_index)
      {
//         cout << "  q_index: " << q_index << "\n";
        for (unsigned int i_component=0; i_component<dim; ++i_component)
        {
          l2_error_per_cell_grad+=pow(solution_all_components_values_second_refine_at_quads_per_cell[q_index][i_component]-solution_all_components_values_first_refine_at_quads_per_cell[q_index][i_component],2.0)*fe_values.JxW (q_index);         //
          
          for (unsigned int i_direction=0; i_direction<dim; ++i_direction)
          {
//             cout << "  i_component: " << i_component << ", ";
//             cout << "i_direction: " << i_direction << ": ";
//             cout << solution_all_components_grads_second_refine_at_quads_per_cell[q_index][i_component][i_direction] << "\n";
//             
            l2_error_per_cell_2ndd+=pow(solution_all_components_grads_second_refine_at_quads_per_cell[q_index][i_component][i_direction]-solution_all_components_grads_first_refine_at_quads_per_cell[q_index][i_component][i_direction],2.0)*fe_values.JxW (q_index);
          }
        }
        l2_error_per_cell_soln+=pow(solution_all_components_values_second_refine_at_quads_per_cell[q_index][dim]-solution_all_components_values_first_refine_at_quads_per_cell[q_index][dim],2.0)*fe_values.JxW (q_index);
      }
        
      cellwise_error[1][cell_index_first_refine]=sqrt(l2_error_per_cell_grad);
      cellwise_error[2][cell_index_first_refine]=sqrt(l2_error_per_cell_2ndd);
      cellwise_error[0][cell_index_first_refine]=sqrt(l2_error_per_cell_soln);

      l2_error_per_cell_grad=0.0;
      l2_error_per_cell_2ndd=0.0;
      l2_error_per_cell_soln=0.0;
      
    }
  }
}

#endif
