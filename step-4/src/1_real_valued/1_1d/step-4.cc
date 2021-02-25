

#include <mainclass.h>

int main (int argc, char *argv[])
{
  unsigned int id_case = 1;
  int id_quad_assem_incre = 1;
  double coeff_inner = 1.0;
  double tol_prm = 1e-16; 
  int degree = 1;
  int refine = 1;
  
  if ( argc != 7 ) 
  {
    cout<<"usage: "<< argv[0] <<" <id_case> <id_quad_assem_incre> <coeff_inner> <tol_prm> <degree> <refinement>\n";
    exit(EXIT_FAILURE);
  } else 
  {
    id_case = atoi(argv[1]);
    id_quad_assem_incre = atoi(argv[2]);
    coeff_inner = atof(argv[3]);
    tol_prm = atof(argv[4]);
    degree = atoi(argv[5]);
    refine = atoi(argv[6]);
  }

  for (int i = 0; i < 1; i++)           // repeating runs to obtain the CPU time when a single run takes too little time
  {
    deallog.depth_console (0);
    {
      Step4<1> laplace_problem_1d(id_case, id_quad_assem_incre, coeff_inner, tol_prm, degree, refine);
      laplace_problem_1d.run ();
//       cout << "run finished\n";
      cout << endl;
    }
  }
  return 0;
}

