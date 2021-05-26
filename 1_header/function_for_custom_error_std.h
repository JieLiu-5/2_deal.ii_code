
#ifndef FUNCTION_FOR_CUSTOM_ERROR_STD_H
#define FUNCTION_FOR_CUSTOM_ERROR_STD_H


template <int dim>
void error_computation_custom_core_std(Vector<double>& solution_first_refine, Vector<double>& solution_second_refine, DoFHandler<dim>& dof_handler_first_refine, DoFHandler<dim>& dof_handler_second_refine, QGauss<dim>& quadrature_formula, unsigned int& n_components_number, vector<double>& vector_error)
{
//   cout << "Invoking core\n";
  unsigned int n_q_points = quadrature_formula.size();
  
  double l2_error_per_cell_soln = 0;
  double l2_error_per_cell_grad = 0;
  double l2_error_per_cell_2ndd = 0;
  
  Vector<double> cellwise_error_soln(dof_handler_first_refine.get_triangulation().n_active_cells());
  Vector<double> cellwise_error_grad(dof_handler_first_refine.get_triangulation().n_active_cells());
  Vector<double> cellwise_error_2ndd(dof_handler_first_refine.get_triangulation().n_active_cells());
  
  int cell_index_first_refine = 0;

  vector<Vector<double>> solution_values_first_refine_at_quads_per_cell(n_q_points, Vector<double>(n_components_number));
  vector<Vector<double>> solution_values_second_refine_at_quads_per_cell(n_q_points, Vector<double>(n_components_number));
  
  vector<vector<Tensor<1,dim>>> solution_grads_first_refine_at_quads_per_cell(n_q_points,vector<Tensor<1,dim>>(n_components_number));
  vector<vector<Tensor<1,dim>>> solution_grads_second_refine_at_quads_per_cell(n_q_points,vector<Tensor<1,dim>>(n_components_number));
  
  vector<vector<Tensor<2,dim>>> solution_2ndds_first_refine_at_quads_per_cell(n_q_points,vector<Tensor<2,dim>>(n_components_number));
  vector<vector<Tensor<2,dim>>> solution_2ndds_second_refine_at_quads_per_cell(n_q_points,vector<Tensor<2,dim>>(n_components_number));
  

  FEValues<dim> fe_values (dof_handler_first_refine.get_fe(), quadrature_formula,
                          update_values   | update_gradients | update_hessians |
                          update_quadrature_points | update_JxW_values);  
  
  vector<Point<dim> > quad_coords_vector=quadrature_formula.get_points();
  
//   cout << "  quad_coords_vector:\n";
//   print_vector(quad_coords_vector);
  
  
  typename DoFHandler<dim>::active_cell_iterator  
  cell_iterator_first_refine = dof_handler_first_refine.begin_active();
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler_second_refine.begin_active(),
  endc = dof_handler_second_refine.end();   
  
  if (dim==1)
  {
    unsigned int n_q_points_one_side = (n_q_points+1)/2.0;
//     cout << "  n_q_points_one_side: " << n_q_points_one_side << "\n";
    
    vector<Point<dim> > left_quad_vector(n_q_points_one_side);
    vector<Point<dim> > right_quad_vector(n_q_points_one_side);

    for (unsigned int i=0; i<left_quad_vector.size();++i)
    {
      left_quad_vector[i]=quad_coords_vector[i]*2;
    }
    
    for (unsigned int i=0; i<right_quad_vector.size();++i)
    {
      right_quad_vector[i](0) = (quad_coords_vector[i+(n_q_points_one_side*2==n_q_points?n_q_points_one_side:n_q_points_one_side-1)](0)-0.5)*2;
    }
    
//     cout << "  left_quad_vector:\n";
//     print_vector(left_quad_vector);    
//     
//     cout << "  right_quad_vector:\n";
//     print_vector(right_quad_vector); 
    
    vector<Vector<double>> solution_values_second_refine_at_quads_per_cell_left(n_q_points_one_side, Vector<double>(n_components_number));
    vector<Vector<double>> solution_values_second_refine_at_quads_per_cell_right(n_q_points_one_side, Vector<double>(n_components_number));
    
    vector<vector<Tensor<1,dim>>> solution_grads_second_refine_at_quads_per_cell_left(n_q_points_one_side, vector<Tensor<1,dim>>(n_components_number));
    vector<vector<Tensor<1,dim>>> solution_grads_second_refine_at_quads_per_cell_right(n_q_points_one_side, vector<Tensor<1,dim>>(n_components_number));
    
    vector<vector<Tensor<2,dim>>> solution_2ndds_second_refine_at_quads_per_cell_left(n_q_points_one_side, vector<Tensor<2,dim>>(n_components_number));
    vector<vector<Tensor<2,dim>>> solution_2ndds_second_refine_at_quads_per_cell_right(n_q_points_one_side, vector<Tensor<2,dim>>(n_components_number));
    
    Quadrature<dim>  quadrature_formula_high_2_low_left(left_quad_vector);
    FEValues<dim> fe_values_high_2_low_left (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_left,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    
    Quadrature<dim>  quadrature_formula_high_2_low_right(right_quad_vector);                  
    FEValues<dim> fe_values_high_2_low_right (dof_handler_first_refine.get_fe(), quadrature_formula_high_2_low_right,
                            update_values   | update_gradients | update_hessians |
                            update_quadrature_points | update_JxW_values);
    
    for (; cell!=endc; ++cell)
    {
//       cout << "  cell nr. of the second refinement " << cell->active_cell_index() << "\n";
        
      if (cell->active_cell_index()%2==0)
      {
        fe_values_high_2_low_left.reinit (cell); 
      
        fe_values_high_2_low_left.get_function_values (solution_second_refine, solution_values_second_refine_at_quads_per_cell_left);
        fe_values_high_2_low_left.get_function_gradients (solution_second_refine, solution_grads_second_refine_at_quads_per_cell_left);
        fe_values_high_2_low_left.get_function_hessians (solution_second_refine, solution_2ndds_second_refine_at_quads_per_cell_left);
         
        for (unsigned int i=0; i<solution_values_second_refine_at_quads_per_cell_left.size();++i)
        {
          solution_values_second_refine_at_quads_per_cell[i]=solution_values_second_refine_at_quads_per_cell_left[i];             
          solution_grads_second_refine_at_quads_per_cell[i]=solution_grads_second_refine_at_quads_per_cell_left[i];             
          solution_2ndds_second_refine_at_quads_per_cell[i]=solution_2ndds_second_refine_at_quads_per_cell_left[i];             
        }
      }else
      {
        fe_values_high_2_low_right.reinit (cell);
      
        fe_values_high_2_low_right.get_function_values (solution_second_refine, solution_values_second_refine_at_quads_per_cell_right);
        fe_values_high_2_low_right.get_function_gradients (solution_second_refine, solution_grads_second_refine_at_quads_per_cell_right);
        fe_values_high_2_low_right.get_function_hessians (solution_second_refine, solution_2ndds_second_refine_at_quads_per_cell_right);
            
        if (2*n_q_points_one_side==n_q_points)
        {
          for (unsigned int i=0; i<solution_values_second_refine_at_quads_per_cell_right.size();++i)
          {
            solution_values_second_refine_at_quads_per_cell[n_q_points_one_side+i]=solution_values_second_refine_at_quads_per_cell_right[i];
            solution_grads_second_refine_at_quads_per_cell[n_q_points_one_side+i]=solution_grads_second_refine_at_quads_per_cell_right[i];
            solution_2ndds_second_refine_at_quads_per_cell[n_q_points_one_side+i]=solution_2ndds_second_refine_at_quads_per_cell_right[i];
          }  
        }else
        {
          for (unsigned int i=1; i<solution_values_second_refine_at_quads_per_cell_right.size();++i)
          {
            solution_values_second_refine_at_quads_per_cell[n_q_points_one_side+i-1]=solution_values_second_refine_at_quads_per_cell_right[i];
            solution_grads_second_refine_at_quads_per_cell[n_q_points_one_side+i-1]=solution_grads_second_refine_at_quads_per_cell_right[i];
            solution_2ndds_second_refine_at_quads_per_cell[n_q_points_one_side+i-1]=solution_2ndds_second_refine_at_quads_per_cell_right[i];
          }
        }
        
//         cout << "  solution_values_second_refine_at_quads_per_cell:\n";
//         print_vector(solution_values_second_refine_at_quads_per_cell);
//         cout << "  solution_grads_second_refine_at_quads_per_cell:\n";
//         print_vector_in_vector(solution_grads_second_refine_at_quads_per_cell);
//         cout << "  solution_2ndds_second_refine_at_quads_per_cell:\n";
//         print_vector_in_vector(solution_2ndds_second_refine_at_quads_per_cell);
        
        
        fe_values.reinit (cell_iterator_first_refine);                                     // dealing with data of the first refinement
        
//         cout << "  cell of the first refinement " << cell_iterator_first_refine->active_cell_index() << "\n";
        
        fe_values.get_function_values (solution_first_refine, solution_values_first_refine_at_quads_per_cell);    
        fe_values.get_function_gradients (solution_first_refine, solution_grads_first_refine_at_quads_per_cell);    
        fe_values.get_function_hessians (solution_first_refine, solution_2ndds_first_refine_at_quads_per_cell);    
        
//         cout << "  solution_grads_first_refine_at_quads_per_cell:\n";
//         print_vector_in_vector(solution_grads_first_refine_at_quads_per_cell);
//         cout << "  solution_2ndds_first_refine_at_quads_per_cell:\n";
//         print_vector_in_vector(solution_2ndds_first_refine_at_quads_per_cell);
        
        cell_index_first_refine = cell_iterator_first_refine->active_cell_index();
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i<n_components_number; ++i)
          {
            l2_error_per_cell_soln += pow(solution_values_first_refine_at_quads_per_cell[q_index][i] - solution_values_second_refine_at_quads_per_cell[q_index][i],2 ) * fe_values.JxW (q_index);
            for (unsigned int j = 0; j<dim; ++j)
            {
              l2_error_per_cell_grad += pow(solution_grads_first_refine_at_quads_per_cell[q_index][i][j] - solution_grads_second_refine_at_quads_per_cell[q_index][i][j],2 ) * fe_values.JxW (q_index);  
              for (unsigned int k = 0; k < dim; ++k)
              {
                l2_error_per_cell_2ndd += pow(solution_2ndds_first_refine_at_quads_per_cell[q_index][i][j][k] - solution_2ndds_second_refine_at_quads_per_cell[q_index][i][j][k],2 ) * fe_values.JxW (q_index);   
              }
            }
          }
        }
        
        cellwise_error_soln[cell_index_first_refine] = sqrt(l2_error_per_cell_soln);
        cellwise_error_grad[cell_index_first_refine] = sqrt(l2_error_per_cell_grad);
        cellwise_error_2ndd[cell_index_first_refine] = sqrt(l2_error_per_cell_2ndd);
        
        l2_error_per_cell_soln=0;
        l2_error_per_cell_grad=0;
        l2_error_per_cell_2ndd=0;
        
        ++cell_iterator_first_refine;
      }
    }
  }else if(dim==2)
  {
    unsigned int n_q_points_one_block_in_one_side = int((sqrt(n_q_points)+1)/2.0);
      
    unsigned int n_q_points_one_block = pow(n_q_points_one_block_in_one_side, 2);                       // one block should be 1/4 of a cell
    
//     cout << "  nr. of quadrature points in one block: " << n_q_points_one_block << "\n";
        
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
  
    scale_quad_coords_small(quad_coords_vector_0_0);
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

    vector<Vector<double>> solution_values_second_refine_at_quads_per_cell_0_0(n_q_points_one_block, Vector<double>(n_components_number));
    vector<Vector<double>> solution_values_second_refine_at_quads_per_cell_1_0(n_q_points_one_block, Vector<double>(n_components_number));
    vector<Vector<double>> solution_values_second_refine_at_quads_per_cell_0_1(n_q_points_one_block, Vector<double>(n_components_number));
    vector<Vector<double>> solution_values_second_refine_at_quads_per_cell_1_1(n_q_points_one_block, Vector<double>(n_components_number));
    
    
    vector<vector<Tensor<1,dim>>> solution_grads_second_refine_at_quads_per_cell_0_0(n_q_points_one_block, vector<Tensor<1,dim>>(n_components_number));
    vector<vector<Tensor<1,dim>>> solution_grads_second_refine_at_quads_per_cell_1_0(n_q_points_one_block, vector<Tensor<1,dim>>(n_components_number));
    vector<vector<Tensor<1,dim>>> solution_grads_second_refine_at_quads_per_cell_0_1(n_q_points_one_block, vector<Tensor<1,dim>>(n_components_number));
    vector<vector<Tensor<1,dim>>> solution_grads_second_refine_at_quads_per_cell_1_1(n_q_points_one_block, vector<Tensor<1,dim>>(n_components_number));
    
    
    vector<vector<Tensor<2,dim>>> solution_2ndds_second_refine_at_quads_per_cell_0_0(n_q_points_one_block, vector<Tensor<2,dim>>(n_components_number));
    vector<vector<Tensor<2,dim>>> solution_2ndds_second_refine_at_quads_per_cell_1_0(n_q_points_one_block, vector<Tensor<2,dim>>(n_components_number));
    vector<vector<Tensor<2,dim>>> solution_2ndds_second_refine_at_quads_per_cell_0_1(n_q_points_one_block, vector<Tensor<2,dim>>(n_components_number));
    vector<vector<Tensor<2,dim>>> solution_2ndds_second_refine_at_quads_per_cell_1_1(n_q_points_one_block, vector<Tensor<2,dim>>(n_components_number));
    
    
    for (; cell!=endc; ++cell)
    {
      
      if (cell->active_cell_index()%4==0)
      {

        fe_values_high_2_low_0_0.reinit (cell);
        fe_values_high_2_low_0_0.get_function_values (solution_second_refine, solution_values_second_refine_at_quads_per_cell_0_0);
        fe_values_high_2_low_0_0.get_function_gradients (solution_second_refine, solution_grads_second_refine_at_quads_per_cell_0_0);
        fe_values_high_2_low_0_0.get_function_hessians (solution_second_refine, solution_2ndds_second_refine_at_quads_per_cell_0_0);
      
        for (unsigned int i=0; i<n_q_points_one_block; ++i)
        {
          solution_values_second_refine_at_quads_per_cell[quad_indices_vector[i]]=solution_values_second_refine_at_quads_per_cell_0_0[i];
          solution_grads_second_refine_at_quads_per_cell[quad_indices_vector[i]]=solution_grads_second_refine_at_quads_per_cell_0_0[i];
          solution_2ndds_second_refine_at_quads_per_cell[quad_indices_vector[i]]=solution_2ndds_second_refine_at_quads_per_cell_0_0[i];
        }
      
      }else if (cell->active_cell_index()%4==1)
      {
        fe_values_high_2_low_1_0.reinit (cell);
        fe_values_high_2_low_1_0.get_function_values (solution_second_refine, solution_values_second_refine_at_quads_per_cell_1_0);
        fe_values_high_2_low_1_0.get_function_gradients (solution_second_refine, solution_grads_second_refine_at_quads_per_cell_1_0);
        fe_values_high_2_low_1_0.get_function_hessians (solution_second_refine, solution_2ndds_second_refine_at_quads_per_cell_1_0);
      
        for (unsigned int i=0; i<n_q_points_one_block; ++i)
        {
          solution_values_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*1+i]]=solution_values_second_refine_at_quads_per_cell_1_0[i];
          solution_grads_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*1+i]]=solution_grads_second_refine_at_quads_per_cell_1_0[i];
          solution_2ndds_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*1+i]]=solution_2ndds_second_refine_at_quads_per_cell_1_0[i];
        }
      
      }else if (cell->active_cell_index()%4==2)
      {
        fe_values_high_2_low_0_1.reinit (cell);
        fe_values_high_2_low_0_1.get_function_values (solution_second_refine, solution_values_second_refine_at_quads_per_cell_0_1);
        fe_values_high_2_low_0_1.get_function_gradients (solution_second_refine, solution_grads_second_refine_at_quads_per_cell_0_1);
        fe_values_high_2_low_0_1.get_function_hessians (solution_second_refine, solution_2ndds_second_refine_at_quads_per_cell_0_1);
      
//       print_vector(solution_values_second_refine_at_quads_per_cell_0_1);   
      
        for (unsigned int i=0; i<n_q_points_one_block; ++i)
        {
          solution_values_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*2+i]]=solution_values_second_refine_at_quads_per_cell_0_1[i];
          solution_grads_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*2+i]]=solution_grads_second_refine_at_quads_per_cell_0_1[i];
          solution_2ndds_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*2+i]]=solution_2ndds_second_refine_at_quads_per_cell_0_1[i];
        }
      
      }else if (cell->active_cell_index()%4==3)
      {
        fe_values_high_2_low_1_1.reinit (cell);
        fe_values_high_2_low_1_1.get_function_values (solution_second_refine, solution_values_second_refine_at_quads_per_cell_1_1);
        fe_values_high_2_low_1_1.get_function_gradients (solution_second_refine, solution_grads_second_refine_at_quads_per_cell_1_1);
        fe_values_high_2_low_1_1.get_function_hessians (solution_second_refine, solution_2ndds_second_refine_at_quads_per_cell_1_1);
      
//       print_vector(solution_values_second_refine_at_quads_per_cell_1_1);
      
        for (unsigned int i=0; i<n_q_points_one_block; ++i)
        {
          solution_values_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*3+i]]=solution_values_second_refine_at_quads_per_cell_1_1[i];
          solution_grads_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*3+i]]=solution_grads_second_refine_at_quads_per_cell_1_1[i];
          solution_2ndds_second_refine_at_quads_per_cell[quad_indices_vector[n_q_points_one_block*3+i]]=solution_2ndds_second_refine_at_quads_per_cell_1_1[i];
        }
        
        cell_index_first_refine = cell_iterator_first_refine->active_cell_index();
//       cout << "  cell of first refinement: " << cell_index_first_refine << endl;                   // evaluating solution values of the first refinement
        
        fe_values.reinit (cell_iterator_first_refine);
        fe_values.get_function_values (solution_first_refine, solution_values_first_refine_at_quads_per_cell);
        fe_values.get_function_gradients (solution_first_refine, solution_grads_first_refine_at_quads_per_cell);
        fe_values.get_function_hessians (solution_first_refine, solution_2ndds_first_refine_at_quads_per_cell);
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i<n_components_number; ++i)
          {
            l2_error_per_cell_soln += pow(solution_values_first_refine_at_quads_per_cell[q_index][i] - solution_values_second_refine_at_quads_per_cell[q_index][i],2 ) * fe_values.JxW (q_index); 
            for (unsigned int j = 0; j<dim; ++j)
            {
              l2_error_per_cell_grad += pow(solution_grads_first_refine_at_quads_per_cell[q_index][i][j] - solution_grads_second_refine_at_quads_per_cell[q_index][i][j],2 ) * fe_values.JxW (q_index); 
              for (unsigned int k = 0; k < dim; ++k)
              {
                l2_error_per_cell_2ndd += pow(solution_2ndds_first_refine_at_quads_per_cell[q_index][i][j][k] - solution_2ndds_second_refine_at_quads_per_cell[q_index][i][j][k],2 ) * fe_values.JxW (q_index);   
              }              
            }
          }
        }   
        cellwise_error_soln[cell_index_first_refine] = sqrt(l2_error_per_cell_soln);
        cellwise_error_grad[cell_index_first_refine] = sqrt(l2_error_per_cell_grad);
        cellwise_error_2ndd[cell_index_first_refine] = sqrt(l2_error_per_cell_2ndd);
        
        l2_error_per_cell_soln=0;
        l2_error_per_cell_grad=0;
        l2_error_per_cell_2ndd=0;
        
        ++cell_iterator_first_refine;          
      }
    }
  }
  
//     cout << "  cellwise_error_soln: " << cellwise_error_soln << "\n";
//     cout << "  cellwise_error_2ndd: " << cellwise_error_2ndd << "\n";  
  
  vector_error[0]=cellwise_error_soln.l2_norm();
  vector_error[1]=cellwise_error_grad.l2_norm();
  vector_error[2]=cellwise_error_2ndd.l2_norm();
  
//   cout << "  vector_error:\n";
//   print_vector(vector_error);  
  
}

#endif
