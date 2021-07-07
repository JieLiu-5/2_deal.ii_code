
#include<2_complex_valued/mainclass_step-20_complex.h>
      
int main (int argc, char *argv[])
{
    double length;
    int id_scaling_scheme;
    double L2_inte_solu_for_scaling, L2_inte_grad_for_scaling;
	int degree = 4;
	int grid_parm = 3;

	if ( argc != 7 ) {
		std::cout<<"usage: "<< argv[0] <<" <length> <id_scaling_scheme> <L2_inte_solu_for_scaling> <L2_inte_grad_for_scaling> <degree> <grid_parm>\n";
		exit(EXIT_FAILURE);
	} else {
        length = atof(argv[1]);
        id_scaling_scheme = atoi(argv[2]);
        L2_inte_solu_for_scaling = atof(argv[3]);
        L2_inte_grad_for_scaling = atof(argv[4]);
		degree = atoi(argv[5]);
		grid_parm = atoi(argv[6]);
	}

  try
    {
      using namespace Step20;
//       std::cout << "~~~~~~~~~~~~~~~~~~~~ degree is " << degree << " (P" << degree << "/P" << degree-1 << ") ~~~~~~~~~~~~~~~~~~~~" << std::endl;
      
      cout << "  refine: " << grid_parm << endl;   
      
      MixedLaplaceProblem<1> mixed_laplace_problem(length, id_scaling_scheme, L2_inte_solu_for_scaling, L2_inte_grad_for_scaling, degree, grid_parm);
      
      mixed_laplace_problem.run ();
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
