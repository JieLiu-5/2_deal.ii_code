/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */

// @sect3{Include files}

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h>                  //added by me

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>


#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/timer_return_total_cpu_time.h>


#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <iomanip>
#include <complex>
#include <cmath>

// This is new, however: in the previous example we got some unwanted output
// from the linear solvers. If we want to suppress it, we have to include this
// file and add a single line somewhere to the program (see the main()
// function below for that):
#include <deal.II/base/logstream.h>
#define pi 3.141592653589793238462643

using namespace dealii;
using namespace std;

#include"../6_headers/1_cpp/1_dealii/common_function.h"
#include<2_complex_valued/mainclass_step_4_complex.h>


int main (int argc, char *argv[])
{
  double length = 10.0;
  int degree = 2;
  int grid_parm = 3;

  if ( argc != 4 ) 
  {
    cout<<"usage: "<< argv[0] <<" <length> <degree> <refinement>\n";
    exit(EXIT_FAILURE);
  } else 
  {
    length = atof(argv[1]);
    degree = atoi(argv[2]);
    grid_parm = atoi(argv[3]);
  }
  
  deallog.depth_console (0);
  {
    cout << "  refine: " << grid_parm << endl;      
    
//     clock_t t=clock();
//     double time_cpu_self_defined;
    
    Step4<1> laplace_problem_1d(length, degree, grid_parm);
    laplace_problem_1d.run ();
    
//     t = clock() - t;
//     time_cpu_self_defined = ((float)t)/CLOCKS_PER_SEC;    
//     cout << "time_cpu_self_defined: " << time_cpu_self_defined << "\n";
    
  }
   
  return 0;
}

//myfile.close();
