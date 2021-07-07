

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

#include"../../0_headers_for_cpp/common_cpp_functions.h"
#include"../1_header/common_function_basic.h"
#include"../1_header/common_function_advanced.h"
#include"../1_header/compute_l2_norms_coefficient.h"

#include"../1_header/function_for_custom_error_std.h"
// #include"../1_header/function_for_custom_error_mix.h"


#include<step_4_number_type_independent.h>
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
  vector<string> vec_FEM={"0_sm", "1_mm"};
  unsigned int id_FEM = 0;                          // '0': standard FEM
                                                    // '1': mixed FEM
  

  unsigned int id_fe_system = 2;                        // making the choice depending on the space dimension, changes on line ~64 and line ~158 are compulsory
                                                        // '0' for FE_Q/FE_DGQ
                                                        // '1' for RT
                                                        // '2' or BDM

  unsigned int id_type_of_number = 0;               // '0': real-valued
                                                    // '1': complex-valued
                                                        
  unsigned int param_quad;
  unsigned int id_loc_coeff_diff_weak_form;
  double tol_prm = 1e-16;
  unsigned int is_custom_method_used_for_error;
  
                                                           
  unsigned int id_case;
  double coeff_var_inner_x;
  
  
  unsigned int id_coeff_diff_value;
  double coeff_diff_inner;
  
  unsigned int id_coeff_helm_value;
  double coeff_helm_inner;
  

  unsigned int initial_degree = 1;
  unsigned int id_mesh_being_created = 0;
  unsigned int grid_param_initial = 1;
  unsigned int n_total_refinements = 1;
  
  const unsigned int n_arguments = 23;

  if ( argc != n_arguments )
  {
    cout << "# of arguments should be " << n_arguments 
         << ", only " << argc << " received\n";
         
    cout << "usage: \n"
         << argv[0] << "\n"
         << "<id_FEM>\n"
         << "<id_fe_system>\n"
         << "<id_type_of_number>\n"
         << "~~~~~~~~~~~\n"
         << "<param_quad>\n"
         << "<id_loc_coeff_diff_weak_form>\n"
         << "<tol_prm>\n"
         << "<is_custom_method_used_for_error>\n"
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
         << "<id_mesh_being_created>\n"
         << "<initial refinement level>\n"
         << "<nr. of total refinements>\n";
         
    exit(EXIT_FAILURE);
  } else 
  {
    id_FEM = atoi(argv[1]);
    id_fe_system = atoi(argv[2]);
    id_type_of_number = atoi(argv[3]);
    // a segmentation
    param_quad = atoi(argv[5]);
    id_loc_coeff_diff_weak_form = atoi(argv[6]);
    tol_prm = atof(argv[7]);
    is_custom_method_used_for_error = atoi(argv[8]);
    // a segmentation
    id_case = atoi(argv[10]);
    coeff_var_inner_x = atof(argv[11]);
    // a segmentation
    id_coeff_diff_value = atoi(argv[13]);
    coeff_diff_inner = atof(argv[14]);
    // a segmentation
    id_coeff_helm_value = atoi(argv[16]);
    coeff_helm_inner = atof(argv[17]);
    // a segmentation
    initial_degree = atoi(argv[19]);
    id_mesh_being_created = atoi(argv[20]);
    grid_param_initial = atoi(argv[21]);
    n_total_refinements = atoi(argv[22]);
  }
  try
  {
    for (int i = 0; i < 1; i++)
    {
      const int dim = 1;
//       cout << "  dimension: " << dim << "\n";
                                                                    
//       std::cout << "Input arguments: \n"
//                 << "  id_FEM: " << id_FEM << "\n"
//                 << "  id_fe_system: " << id_fe_system << "\n"
//                 << "  id_type_of_number: " << id_type_of_number << "\n"
//                 << "  ~~~~~~~~~~~\n"
//                 << "  param for the number of quadrature points: " << param_quad << "\n"
//                 << "  tol_prm: " << tol_prm << "\n"
//                 << "  ~~~~~~~~~~~\n"
//                 << "  id_case: " << id_case << "\n"
//                 << "  coeff_var_inner_x: " << coeff_var_inner_x << "\n"
//                 << "  ~~~~~~~~~~~\n"
//                 << "  param for the value of the diffusion coefficient: " << id_coeff_diff_value << "\n"
//                 << "  coeff_diff_inner: " << coeff_diff_inner << "\n"
//                 << "  ~~~~~~~~~~~\n"
//                 << "  param for the value of the Helmholtz coefficient: " << id_coeff_helm_value << "\n"
//                 << "  coeff_helm_inner: " << coeff_helm_inner << "\n"
//                 << "  ~~~~~~~~~~~\n"
//                 << "  initial_degree: " << initial_degree << "\n"
//                 << "  id_mesh_being_created: " << id_mesh_being_created << "\n"
//                 << "  initial refinement level: " << grid_param_initial << "\n"
//                 << "  nr. of total refinements: " << n_total_refinements << "\n";  

      
      if (id_FEM == 0)
      {
        if(id_type_of_number == 0)
        {
          Step_4_Real<dim> obj_second_order_differential_problem(param_quad, 
                                                          tol_prm, 
                                                          is_custom_method_used_for_error,
                                                          id_case,
                                                          coeff_var_inner_x,
                                                          id_coeff_diff_value, 
                                                          coeff_diff_inner,
                                                          id_coeff_helm_value,
                                                          coeff_helm_inner,
                                                          initial_degree,
                                                          id_mesh_being_created,
                                                          grid_param_initial,
                                                          n_total_refinements);
          obj_second_order_differential_problem.run ();
        }else if(id_type_of_number == 1)
        {
#if 0
          Step_4_Complex<dim> obj_second_order_differential_problem(param_quad,
                                                               tol_prm,
                                                               is_custom_method_used_for_error,
                                                               id_case,
                                                               coeff_var_inner_x,
                                                               id_coeff_diff_value,
                                                               coeff_diff_inner,
                                                               id_coeff_helm_value,
                                                               coeff_helm_inner,
                                                               1.0,
                                                               initial_degree,
                                                               id_mesh_being_created,
                                                               grid_param_initial,
                                                               n_total_refinements);
          obj_second_order_differential_problem.run ();
#endif
            
        }else
        {
            cout << "Unknown type of number using the standard FEM\n";
        }
      }else if(id_FEM == 1)
      {
        if(id_type_of_number==0)
        {
            cout << "  id_fe: " << (id_FEM==0 ? "default" : (id_fe_system==1?"RT":"BDM")) << "\n"
                 << "  param for the location of the diffusion coefficient: " << id_loc_coeff_diff_weak_form << "\n";           
            
//           MixedLaplaceProblem_Real<2> obj_second_order_differential_problem(id_fe_system,
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
//                                                                         grid_param_initial);
//           obj_second_order_differential_problem.run ();            
            
        }else if(id_type_of_number==1)
        {
//           MixedLaplaceProblem_Complex<1> obj_second_order_differential_problem(id_case,
//                                                                            param_quad,
//                                                                            id_coeff_diff_value,
//                                                                            id_coeff_helm_value,
//                                                                            1.0,
//                                                                            initial_degree,
//                                                                            grid_param_initial);                // _Validation
//           obj_second_order_differential_problem.run ();             
//           break;
        }else
        {
            cout << "Unknown type of number using the mixed FEM\n";
        }
      }else
      {
        cout << "Unknown FEM method\n";
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
