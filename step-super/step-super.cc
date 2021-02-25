

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
// #include <deal.II/lac/solver_cg1.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_bernstein.h>                        
#include <deal.II/fe/fe_dgq.h>                  
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>                   // important for printing coordinates of DoFs

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>
#include <type_traits>
#include <stdlib.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor_function.templates.h>

#include <complex>
#include <cmath>

#include <string>

#define pi 3.141592653589793238462643

typedef double real_t;

using namespace dealii;
using namespace std;

#include"../6_headers/1_cpp/1_dealii/common_function_basic.h"
#include"../6_headers/1_cpp/1_dealii/common_function_advanced.h"
#include"../6_headers/1_cpp/1_dealii/compute_l2_norms_coefficient.h"

#include"../6_headers/1_cpp/1_dealii/function_for_custom_error_std.h"
#include"../6_headers/1_cpp/1_dealii/function_for_custom_error_mix.h"


#include<1_real_valued/mainclass_step_4_real.h>
// #include<2_complex_valued/mainclass_step_4_complex.h>

// #include<1_real_valued/mainclass_step_20_real.h>
// #include<2_complex_valued/mainclass_step_20_complex.h>
// #include<2_complex_valued/mainclass_step_20_complex_validation.h>                // include me for checking the 1d validation results of Dec. 17, 2019


/*
const char rocket[] =
"           _\n\
          /^\\\n\
          |-|\n\
          | |\n\
          |N|\n\
          |A|\n\
          |S|\n\
          |A|\n\
         /| |\\\n\
        / | | \\\n\
       |  | |  |\n\
        `-\"\"\"-`\n\
";
*/

int main ( int argc, char *argv[] )
{
    
//     for (int i = 0; i < 1; i ++) printf("\n"); // jump to bottom of console
//     printf("%s", rocket);
//     int j = 300000;
//     for (int i = 0; i < 1; i ++) {
//         usleep(j); // move faster and faster,
//         j = (int)(j * 0.9); // so sleep less each time
//         printf("\n"); // move rocket a line upward
//     }
//     
    
  unsigned int id_fem_method;                          // '0': standard FEM for the real-valued problem
                                                    // '1': mixed FEM for the real-valued problem
                                                    // '2': standard FEM for the complex-valued problem
                                                    // '3': mixed FEM for the complex-valued problem
                                                    
  unsigned int param_quad;
  unsigned int id_loc_coeff_diff_weak_form;
  
  
  unsigned int id_fe_system = 2;                        // making the choice depending on the space dimension, changes on line ~64 and line ~158 are compulsory
                                                        // '0' for FE_Q/FE_DGQ
                                                        // '1' for RT
                                                        // '2' or BDM
                                                           
  unsigned int id_case;
  double coeff_var_inner_x;
  
  
  unsigned int id_coeff_diff_value;
  double coeff_diff_inner;
  
  unsigned int id_coeff_helm_value;
  double coeff_helm_inner;
  
  double tol_prm = 1e-16;
  
  double length = 1.0;
  cout << "length: " << length << "\n";
  
  unsigned int initial_degree = 1;
  unsigned int initial_refinement_level = 1;
  unsigned int n_total_refinements = 1;
  
  const unsigned int n_arguments = 20;

  if ( argc != n_arguments )
  {
    cout << "# of arguments should be " << n_arguments 
         << ", only " << argc << " received\n";
         
    cout << "usage: \n"
         << argv[0] << "\n"
         << "<id_fem_method>\n"
         << "<id_fe_system>\n"
         << "~~~~~~~~~~~\n"
         << "<param_quad>\n"
         << "<id_loc_coeff_diff_weak_form>\n"
         << "<tol_prm>\n"
         << "~~~~~~~~~~~\n"
         << "<id_case>\n"
         << "<coeff_var_inner_x>\n"
         << "~~~~~~~~~~~\n"
         << "<id_coeff_diff_value>\n"
         << "<coeff_diff_inner>\n"
         << "~~~~~~~~~~~\n"
         << "<id_coeff_helm_value>\n"
         << "<coeff_helm_inner>\n"
         << "~~~~~~~~~~~\n"
         << "<initial_degree>\n"
         << "<initial refinement level>\n"
         << "<nr. of total refinements>\n";
         
    exit(EXIT_FAILURE);
  } else 
  {
    id_fem_method = atoi(argv[1]);
    id_fe_system = atoi(argv[2]);
    // a segmentation
    param_quad = atoi(argv[4]);
    id_loc_coeff_diff_weak_form = atoi(argv[5]);
    tol_prm = atof(argv[6]);
    // a segmentation
    id_case = atoi(argv[8]);
    coeff_var_inner_x = atof(argv[9]);
    // a segmentation
    id_coeff_diff_value = atoi(argv[11]);
    coeff_diff_inner = atof(argv[12]);
    // a segmentation
    id_coeff_helm_value = atoi(argv[14]);
    coeff_helm_inner = atoi(argv[15]);
    // a segmentation
    initial_degree = atoi(argv[17]);
    initial_refinement_level = atoi(argv[18]);
    n_total_refinements = atoi(argv[19]);
  }
  try
  {
    for (int i = 0; i < 1; i++)
    {
      const int dim = 1;
                                                                     
      std::cout << "Input arguments: \n"
                << "  dimension: " << dim << "\n"
                << "  id_fem_method: " << id_fem_method << "\n"
                << "  id_fe: " << (id_fem_method==0?"default":(id_fe_system==1?"RT":"BDM")) << "\n"
                << "  ~~~~~~~~~~~\n"
                << "  param for the number of quadrature points: " << param_quad << "\n"
                << "  param for the location of the diffusion coefficient: " << id_loc_coeff_diff_weak_form << "\n"
                << "  tol_prm: " << tol_prm << "\n"
                << "  ~~~~~~~~~~~\n"
                << "  id_case: " << id_case << "\n"
                << "  coeff_var_inner_x: " << coeff_var_inner_x << "\n"
                << "  ~~~~~~~~~~~\n"
                << "  param for the value of the diffusion coefficient: " << id_coeff_diff_value << "\n"
                << "  coeff_diff_inner: " << coeff_diff_inner << "\n"
                << "  ~~~~~~~~~~~\n"
                << "  param for the value of the Helmholtz coefficient: " << id_coeff_helm_value << "\n"
                << "  coeff_helm_inner: " << coeff_helm_inner << "\n"
                << "  ~~~~~~~~~~~\n"
                << "  initial_degree: " << initial_degree << "\n"
                << "  initial refinement level: " << initial_refinement_level << "\n"
                << "  nr. of total refinements: " << n_total_refinements << "\n";  

      
      switch (id_fem_method)
      {
        case 0:
        {
          Step4_Real<dim> second_order_differential_problem(id_case,
                                                          coeff_var_inner_x,
                                                          param_quad, 
                                                          id_coeff_diff_value, 
                                                          coeff_diff_inner,
                                                          id_coeff_helm_value,
                                                          coeff_helm_inner,
                                                          tol_prm, 
                                                          initial_degree, 
                                                          initial_refinement_level,
                                                          n_total_refinements);
          second_order_differential_problem.run ();
          break;
        }
//         case 1:
//         {
//           MixedLaplaceProblem_Real<2> second_order_differential_problem(id_fe_system,
//                                                                         id_case,             // this dimension 2 is for a relatively fixed pair of mixed elements
//                                                                         coeff_var_inner_x,
//                                                                         param_quad, 
//                                                                         id_coeff_diff_value, 
//                                                                         coeff_diff_inner, 
//                                                                         id_loc_coeff_diff_weak_form, 
//                                                                         id_coeff_helm_value, 
//                                                                         coeff_helm_inner, 
//                                                                         tol_prm,
//                                                                         initial_degree, 
//                                                                         initial_refinement_level);
//           second_order_differential_problem.run ();                            
//           break;
//         }
//         case 2:
//         {
//           Step4_Complex<dim> second_order_differential_problem(id_case,
//                                                                param_quad,
//                                                                id_coeff_diff_value,
//                                                                id_coeff_helm_value, length,
//                                                                initial_degree,
//                                                                initial_refinement_level);
//           second_order_differential_problem.run ();
//           break;
//         }          
//         case 3:
//         {         
//           MixedLaplaceProblem_Complex<1> second_order_differential_problem(id_case,
//                                                                            param_quad,
//                                                                            id_coeff_diff_value,
//                                                                            id_coeff_helm_value,
//                                                                            length, initial_degree,
//                                                                            initial_refinement_level);                // _Validation
//           second_order_differential_problem.run ();             
//           break;
//         }
        default:
        {
          break;
        }
      }
    }
      
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  
  return 0;
}
