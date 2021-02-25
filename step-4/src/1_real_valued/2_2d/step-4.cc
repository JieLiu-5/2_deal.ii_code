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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>          // for UMFPack solver
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>

#include <deal.II/base/timer.h>
#include <deal.II/base/timer_return_total_cpu_time.h>            // returns the total CPU time, also allows setting the time precision

#include <iostream>
#include <deal.II/grid/grid_refinement.h>       //refine grids locally
#include <deal.II/numerics/error_estimator.h>   //apply adaptive mesh

// This is new, however: in the previous example we got some unwanted output
// from the linear solvers. If we want to suppress it, we have to include this
// file and add a single line somewhere to the program (see the main()
// function below for that):

#include <deal.II/fe/fe_system.h>
#define pi 3.141592653589793238462643

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;
using namespace std;

template <int dim>
class ComputeVelocity1 : public DataPostprocessorScalar<dim>
{
public:
  ComputeVelocity1 ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};

template <int dim>
ComputeVelocity1<dim>::ComputeVelocity1 ()
:
  DataPostprocessorScalar<dim> ("Velocity1",
		  update_gradients)
{}

template <int dim>
void
ComputeVelocity1<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,                        
 std::vector<Vector<double> >               &computed_quantities) const
{
	for (unsigned int i=0; i<inputs.solution_values.size(); i++)
	{
//			  computed_quantities[i] = inputs.solution_values[i];//1;
		 	  computed_quantities[i](0) = inputs.solution_gradients[i][0];
//		 	  computed_quantities[i](2) = inputs.solution_gradients[i][0];
	}
}

template <int dim>
class ComputeVelocity2 : public DataPostprocessorScalar<dim>
{
public:
  ComputeVelocity2 ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};

template <int dim>
ComputeVelocity2<dim>::ComputeVelocity2 ()
:
  DataPostprocessorScalar<dim> ("Velocity2",
		  update_gradients)
{}


template <int dim>
void
ComputeVelocity2<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
	for (unsigned int i=0; i<inputs.solution_values.size(); i++)
	{
		 	  computed_quantities[i](0) = inputs.solution_gradients[i][1];
	}
}

template <int dim>
class Compute2ndderivative1 : public DataPostprocessorScalar<dim>
{
public:
  Compute2ndderivative1 ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};

template <int dim>
Compute2ndderivative1<dim>::Compute2ndderivative1 ()
:
  DataPostprocessorScalar<dim> ("secondderivative1",
		  update_hessians)
{}

template <int dim>
void
Compute2ndderivative1<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
	for (unsigned int i=0; i<inputs.solution_values.size(); i++)
	{
		 	  computed_quantities[i](0) = inputs.solution_hessians[i][0][0];
	}
}

template <int dim>
class Compute2ndderivative2 : public DataPostprocessorScalar<dim>
{
public:
  Compute2ndderivative2 ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};

template <int dim>
Compute2ndderivative2<dim>::Compute2ndderivative2 ()
:
  DataPostprocessorScalar<dim> ("secondderivative2",
		  update_hessians)
{}

template <int dim>
void
Compute2ndderivative2<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
	for (unsigned int i=0; i<inputs.solution_values.size(); i++)
	{
		 	  computed_quantities[i](0) = inputs.solution_hessians[i][0][1];
	}
}


template <int dim>
class Compute2ndderivative3 : public DataPostprocessorScalar<dim>
{
public:
  Compute2ndderivative3 ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};

template <int dim>
Compute2ndderivative3<dim>::Compute2ndderivative3 ()
:
  DataPostprocessorScalar<dim> ("secondderivative3",
		  update_hessians)
{}

template <int dim>
void
Compute2ndderivative3<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
	for (unsigned int i=0; i<inputs.solution_values.size(); i++)
	{
		 	  computed_quantities[i](0) = inputs.solution_hessians[i][1][0];
	}
}


template <int dim>
class Compute2ndderivative4 : public DataPostprocessorScalar<dim>
{
public:
  Compute2ndderivative4 ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};

template <int dim>
Compute2ndderivative4<dim>::Compute2ndderivative4 ()
:
  DataPostprocessorScalar<dim> ("secondderivative4",
		  update_hessians)
{}

template <int dim>
void
Compute2ndderivative4<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
	for (unsigned int i=0; i<inputs.solution_values.size(); i++)
	{
		 	  computed_quantities[i](0) = inputs.solution_hessians[i][1][1];
	}
}

template <int dim>
class Step4
{
public:
  Step4 (double u_L2_inte, int degree, int refine);
  void run ();
  
  double u_L2_inte;
  int degree;
  int refine;
  int cycle_global;
  
  double solu_L2;
  

  double L2_N;
  double L2_error;
  double r_N;
  double H1_N;
  double H1_error;
  double r_Np;
  double H2_N;
  double H2_error;
  double r_Npp;
  double my_CPU_time; 

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void compute_L2_norm ();
  
    
  void compute_errors ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const; 
  void printToFile() const;
  void print_mesh_info(const Triangulation<dim> &triangulation,
                     const std::string        &filename);  
  
  void print_errors_and_CPU_time ();

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  ConstraintMatrix     constraints;
  
  //FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  int is_constraints=0;
  unsigned int is_UMFPACK = 1;
  unsigned int CG_iteration_times = 1;    
  
  Vector<double>       solution;
  Vector<double>       solution_0;
  
//  Vector<double>       solution;
  Vector<double>       system_rhs;
  
  std::vector<double>       vector_diagonal;
  std::vector<double>       vector_offdiagonal;
  
  TimerOutput                               computing_timer;
};

std::ofstream myfile;
template <int dim>
void Step4<dim>::printToFile() const		//print L2 error and CPU time to a text file
{
    myfile.open ("data_error_twod_Pois.txt", std::ofstream::app);               // app
    myfile << refine <<" ";     
//     myfile << cycle_global <<" ";
//     myfile << 1.0/(std::pow(triangulation.n_vertices(),0.5)-1.0) <<" ";
    myfile << triangulation.n_vertices() <<" ";
    myfile << dof_handler.n_dofs() << " ";
    myfile << L2_error << " ";
//     myfile << r_N << " ";
    myfile << H1_error << " ";
//     myfile << r_Np << " ";
    myfile << H2_error << " ";
//     myfile << r_Npp << " ";
    myfile << my_CPU_time << " ";
    
    if (is_UMFPACK == 1)
    {
        myfile << "UMF" << "\n";
    }else if (is_UMFPACK == 0)
    {
        myfile << CG_iteration_times << "\n";
    }
    
//     myfile << solu_L2 << "\n";
//     myfile << iteration_times << "\n";
    myfile.close();
}

template <int dim>
class common_var
{
protected:
 static const double coeff_x;
 static const double coeff_y;    
 static const double denom;
 static const double center_x;
 static const double center_y;
};

// ========================================================== parameters of a unit square

template <int dim>
const double common_var<dim>::coeff_x = 1.0;		
template <int dim>
const double common_var<dim>::coeff_y = 0.0;	
template <int dim>
const double common_var<dim>::denom = 1e0;		//set the denominator
template <int dim>
const double common_var<dim>::center_x = 0.5e0;
template <int dim>
const double common_var<dim>::center_y = 0.5e0;

// ========================================================== parameters of an exponential geometry

// template <int dim>
// const double common_var<dim>::denom = 1.0e2;		//set the denominator
// template <int dim>
// const double common_var<dim>::center_x = 0.5e2;
// template <int dim>
// const double common_var<dim>::center_y = 0.2e1;


template <int dim>
class RightHandSide : public Function<dim>,
protected common_var<dim>
{
public:
  RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  double return_value = -(exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom)*(((4.0*pow(this->coeff_x*(p[0]-this->center_x),2)-2.0*this->coeff_x*this -> denom))/pow(this -> denom,2)+(4.0*pow(this->coeff_y*(p[1]-this->center_y),2)-2.0*this->coeff_y*this -> denom)/pow(this -> denom,2)));
  return return_value;
}


template <int dim>
class BoundaryValues : public Function<dim>,
protected common_var<dim>
{
public:
  BoundaryValues () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  return exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom);
}


template <int dim>
class ExactSolution : public Function<dim>,
protected common_var<dim>
{
public:
  ExactSolution (const double u_L2_inte);
  
  const double u_L2_inte;
  
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim> hessian (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
};

template <int dim>
ExactSolution<dim>::ExactSolution(const double u_L2_inte):
u_L2_inte(u_L2_inte)
{}

template <int dim>
double ExactSolution<dim> :: value (const Point<dim>   &p,
        					 const unsigned int ) const
{
	double return_value = exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom);         //p
    
    return_value = return_value/u_L2_inte;
	return return_value;
}   

template <int dim>
Tensor<1,dim> ExactSolution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
  Tensor<1,dim> return_value;
  return_value[0] = exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom)*(-2*this->coeff_x*(p[0]-this->center_x)/this -> denom);       //px
  return_value[1] = exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom)*(-2*this->coeff_y*(p[1]-this->center_y)/this -> denom);       //py
  
  return_value = return_value/u_L2_inte;
  return return_value;
}

template <int dim>
SymmetricTensor<2,dim> ExactSolution<dim>::hessian (const Point<dim>   &p,
                                       const unsigned int) const
{
  SymmetricTensor<2,dim> return_value;
  return_value[0][0] = exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom)*(((4.0*pow(this->coeff_x*(p[0]-this->center_x),2)-2.0*this->coeff_x*this -> denom))/pow(this -> denom,2));     //pxx
  return_value[0][1] = exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom)*(((4.0*this->coeff_x*(p[0]-this->center_x)*this->coeff_y*(p[1]-this->center_y)))/pow(this -> denom,2));      //pxy
  return_value[1][0] = exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom)*(((4.0*this->coeff_x*(p[0]-this->center_x)*this->coeff_y*(p[1]-this->center_y)))/pow(this -> denom,2));      //pyx
  return_value[1][1] = exp(-(this->coeff_x*pow(p[0]-this->center_x,2)+this->coeff_y*pow(p[1]-this->center_y,2))/this -> denom)*((4.0*pow(this->coeff_y*(p[1]-this->center_y),2)-2.0*this->coeff_y*this -> denom)/pow(this -> denom,2));       //pyy
  
  return_value = return_value/u_L2_inte;
  return return_value;
}


template <int dim>
Step4<dim>::Step4 (double u_L2_inte, int degree, int refine)
  :
  u_L2_inte(u_L2_inte),
  degree(degree),
  refine(refine),
  
  fe (degree),
  //fe(FE_Q<dim>(1), 2)
  dof_handler (triangulation),
  computing_timer  (std::cout, TimerOutput::summary, TimerOutput::cpu_times)
{}


template <int dim>
void Step4<dim>::make_grid ()
{
  std::cout << "==================== Making grid... " << std::endl;
  TimerOutput::Scope t(computing_timer, "make_grid");
  
//     ~~~~~~~~~~~~~~~~~~~ using mesh from the embedded function in deal.ii
  
    const Point<dim> p1 (0.0, 0.0);                         
    const Point<dim> p2 (1.0, 1.0);         // 1.0   0.2  
    
//     GridGenerator::hyper_cube (triangulation, p1[0], p2[0]);                                        // rectangle with one cell segment
    
    const std::vector<unsigned int> repetitions={(unsigned int)(p2[0]/p2[1]),1};            // rectangle with specified cell segments
    GridGenerator::subdivided_hyper_rectangle (triangulation, repetitions, p1, p2);  
  
  

//     ~~~~~~~~~~~~~~~~~~~ using mesh from gmsh
  
//     GridIn<2> gridin;
//     gridin.attach_triangulation(triangulation);
//     std::ifstream f("unit_square.msh");          // exponential     rectangular
//     gridin.read_msh(f);
//   
//     ~~~~~~~~~~~~~~~~~~~ end of generating mesh 

//     ~~~~~~~~~~~~~~~~~~~ operate on the mesh
    
    
//     triangulation.begin_active()->face(1)->set_boundary_id(2);       // right            // method 1 for setting boundary ids
//     triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom
//     triangulation.begin_active()->face(3)->set_boundary_id(1);       // top
    

/*      Triangulation<2>::active_face_iterator face = triangulation.begin_active_face();      // method 2 for setting boundary ids
      Triangulation<2>::active_face_iterator endface = triangulation.end_face();    
      
      for (; face!=endface; ++face)                                 // this iterates from the first active face to the last active face
      {
          if(face->at_boundary())
          {
                if(face->vertex(0)(0)==0 && face->vertex(1)(0)==0)
                {
//                     std::cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the left boundary";
//                     face->set_boundary_id(1);
//                     std::cout << std::endl;
                }else if(face->vertex(0)(1)==p2[1] && face->vertex(1)(1)==p2[1])
                {
//                     std::cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the upper boundary";
//                     face->set_boundary_id(1);
//                     std::cout << std::endl;                    
                }else if(face->vertex(0)(1)==p1[1] && face->vertex(1)(1)==p1[1])
                {
//                     std::cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the bottom boundary";
//                     face->set_boundary_id(1);
//                     std::cout << std::endl;                    
                }
          }

          
      } */    
  
//     print_mesh_info (triangulation, "grid-1.eps");  
  
//     ~~~~~~~~~~~~~~~~~~~ end of operation on the mesh   

  triangulation.refine_global (refine);
  
//     std::cout << "    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangulation information" << std::endl;
    
//     std::cout << "    n_levels(): " << triangulation.n_levels() << ", n_global_levels(): " << triangulation.n_global_levels() << std::endl;
//     std::cout << "    n_lines(): " << triangulation.n_lines() << ", n_active_lines(): " << triangulation.n_active_lines() << std::endl;
//     std::cout << "    n_quads(): " << triangulation.n_quads() << ", n_active_quads(): " << triangulation.n_active_quads() << std::endl;
//     std::cout << "    n_cells(): " << triangulation.n_cells() << ", n_active_cells(): " << triangulation.n_active_cells() << std::endl;
//     std::cout << "    n_faces(): " << triangulation.n_faces() << ", n_active_faces(): " << triangulation.n_active_faces() << std::endl;
//     std::cout << "    has_hanging_nodes? " << triangulation.has_hanging_nodes() << std::endl;
    

  
  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Number of vertices is: "
            << triangulation.n_vertices()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;

      
}


template <int dim>
void Step4<dim>::print_mesh_info(const Triangulation<dim> &triangulation,
                     const std::string        &filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  // Next loop over all faces of all cells and find how often each
  // boundary indicator is used (recall that if you access an element
  // of a std::map object that doesn't exist, it is implicitly created
  // and default initialized -- to zero, in the current case -- before
  // we then increment it):
  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell)
      {
        for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
          {
            if (cell->face(face_n)->at_boundary())
            {
                boundary_count[cell->face(face_n)->boundary_id()]++;
                if (cell->face(face_n)->boundary_id() == 0)
                {
                    cout << "    Dirichlet BCs imposed on: ";
                        
                    cout << "face " << cell->face_index(face_n) << " ";
                    for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
                    {
                        cout << "(coord: " << cell->face(face_n)->vertex(vertex) << "), ";                   // 
                    }
                    cout << "\n";
                }
                else if(cell->face(face_n)->boundary_id() == 1)
                {
                    cout << "    Neumann BCs imposed on: ";
                    
                    cout << "face " << cell->face_index(face_n) << ", ";
                    for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
                    {
                        cout << "(coord: " << cell->face(face_n)->vertex(vertex) << ")";                   // 
                    }
                    
                    cout << "\n";
                }
            }
          }
      }

    std::cout << " boundary indicators: ";
    for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
         it!=boundary_count.end();
         ++it)
      {
        std::cout << it->first << "(" << it->second << " times) ";
      }
    std::cout << std::endl;
  }

  // Finally, produce a graphical representation of the mesh to an output
  // file:
  std::ofstream out (filename.c_str());
  GridOut grid_out;
  grid_out.write_eps (triangulation, out);
  std::cout << " written to " << filename
            << std::endl
            << std::endl;
}

// @sect4{Step4::setup_system}

// This function looks exactly like in the previous example, although it
// performs actions that in their details are quite different if
// <code>dim</code> happens to be 3. The only significant difference from a
// user's perspective is the number of cells resulting, which is much higher
// in three than in two space dimensions!
template <int dim>
void Step4<dim>::setup_system ()
{
  std::cout << "Setting... " << std::endl;
  TimerOutput::Scope t(computing_timer, "setup_system");
  
//   dof_handler.clear ();
  
  dof_handler.distribute_dofs (fe);
  
  solution.reinit (dof_handler.n_dofs());
  solution_0.reinit (dof_handler.n_dofs());
  
  system_rhs.reinit (dof_handler.n_dofs());
  
  vector_diagonal.resize (dof_handler.n_dofs());
  vector_offdiagonal.resize (dof_handler.n_dofs());
  

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
            
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  
            
    if (is_constraints==1)
    {
//   constraints.clear ();
//   DoFTools::make_hanging_node_constraints (dof_handler,
//                                            constraints);
//   
//   std::cout << "   Number of degrees of freedom after making hanging node constraints: "
//             << dof_handler.n_dofs()
//             << std::endl;  
// 
//   VectorTools::interpolate_boundary_values (dof_handler,
//                                             0,
//                                             BoundaryValues<dim>(),
//                                             constraints);
// 
//   constraints.close ();
//   
//   
//     std::cout << "    the number of constraints is: " << constraints.n_constraints() << std::endl;
    
//     cout << "    the orders of the dofs that are constrained are: ";
//     for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
//     {
//         if(constraints.is_constrained(i))
//         {
//             std::cout << i << ", ";
//         }
//     }
//     cout << "\n";
//     
//     std::cout << "    the content of constraints is: " << std::endl;
//     constraints.print(std::cout);
        
        DoFTools::make_sparsity_pattern (dof_handler, dsp,constraints,false);
        
    }else if (is_constraints==0)
    {
        DoFTools::make_sparsity_pattern (dof_handler, dsp);              
    }
            
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);
}


// @sect4{Step4::assemble_system}

// Unlike in the previous example, we would now like to use a non-constant
// right hand side function and non-zero boundary values. Both are tasks that
// are readily achieved with only a few new lines of code in the assemblage of
// the matrix and right hand side.
//
// More interesting, though, is the way we assemble matrix and right hand side
// vector dimension independently: there is simply no difference to the
// two-dimensional case. Since the important objects used in this function
// (quadrature formula, FEValues) depend on the dimension by way of a template
// parameter as well, they can take care of setting up properly everything for
// the dimension for which this function is compiled. By declaring all classes
// which might depend on the dimension using a template parameter, the library
// can make nearly all work for you and you don't have to care about most
// things.
template <int dim>
void Step4<dim>::assemble_system ()
{
  std::cout << "Assembling... " << std::endl;
  TimerOutput::Scope t(computing_timer, "assemble_system");
  QGauss<dim>  quadrature_formula(degree+2);
  QGauss<dim-1> face_quadrature_formula(degree+2);  

  // We wanted to have a non-constant right hand side, so we use an object of
  // the class declared above to generate the necessary data. Since this right
  // hand side object is only used locally in the present function, we declare
  // it here as a local variable:
  const RightHandSide<dim> right_hand_side;

  // Compared to the previous example, in order to evaluate the non-constant
  // right hand side function we now also need the quadrature points on the
  // cell we are presently on (previously, we only required values and
  // gradients of the shape function from the FEValues object, as well as the
  // quadrature weights, FEValues::JxW() ). We can tell the FEValues object to
  // do for us by also giving it the #update_quadrature_points flag:
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients | update_hessians |
                           update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);  

  // We then again define a few abbreviations. The values of these variables
  // of course depend on the dimension which we are presently using. However,
  // the FE and Quadrature classes do all the necessary work for you and you
  // don't have to care about the dimension dependent parts:
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();  
  //int* n_q_points_pr = &n_q_points;

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  // Next, we again have to loop over all cells and assemble local
  // contributions.  Note, that a cell is a quadrilateral in two space
  // dimensions, but a hexahedron in 3D. In fact, the
  // <code>active_cell_iterator</code> data type is something different,
  // depending on the dimension we are in, but to the outside world they look
  // alike and you will probably never see a difference although the classes
  // that this typedef stands for are in fact completely unrelated:
  

  const ExactSolution<dim> exact_solution(u_L2_inte);
  
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  
//   typename Triangulation<dim>::vertex_iterator
//   cell = dof_handler.begin_active(),
//   endc = dof_handler.end();
  
//     std::cout << "    faces_per_cell: " << GeometryInfo<dim>::faces_per_cell << std::endl;

  for (; cell!=endc; ++cell)
    {
        
//       std::cout << "    at cell " << cell->active_cell_index() << std::endl;
      
/*        for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
            {
                std::cout << "      at vertex " << cell->vertex_index(vertex) << ", coords: " << cell->vertex(vertex) << "\n";
            }
        std::cout << std::endl; */  
      
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;

//       std::vector<Tensor<1, dim> > old_solution_gradients(n_q_points);
//       fe_values.get_function_gradients(solution,
//                                        old_solution_gradients);

/*      QGauss<dim>  quadrature_formula(2);
      const unsigned int   n_q_points_1    = quadrature_formula.size();
      FEValues<dim> fe_values (fe, quadrature_formula,
                               update_values   | update_gradients |
                               update_quadrature_points | update_JxW_values);
      std::vector<Tensor<1, dim> > old_solution_gradients(n_q_points_1);
      fe_values.get_function_gradients(solution,
                                       old_solution_gradients);
*/

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));

            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            right_hand_side.value (fe_values.quadrature_point (q_index)) *
                            fe_values.JxW (q_index));
          }
          
      
      for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
      {
//         std::cout << "      at face " << face_number << " ";
        if (cell->face(face_number)->at_boundary())
        {
            if(cell->face(face_number)->boundary_id() == 1)		                                            //treat the Neumann boundary conditions
            {
                
/*                std::cout << "      at cell " << cell->active_cell_index() << ", face " << face_number << ", the boundary id is 1" << std::endl;   
                std::cout << "      coords of the cell: " << std::endl;   
                for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
                    {
                        std::cout << "      at vertex " << cell->vertex_index(vertex) << ", coords: " << cell->vertex(vertex) << "\n";
                    }
                std::cout << std::endl;  */ 
                
                fe_face_values.reinit (cell, face_number);

                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                    const double neumann_value
		      = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
                        fe_face_values.normal_vector(q_point));

                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    cell_rhs(i) += (neumann_value *
                                    fe_face_values.shape_value(i,q_point) *
                                    fe_face_values.JxW(q_point));
                }
            }else
            {
/*                std::cout << "      at cell " << cell->active_cell_index() << ", face " << face_number << ", the boundary id is 0" << std::endl;    
                std::cout << "      coords of the cell: " << std::endl;  
                for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
                    {
                        std::cout << "      at vertex " << cell->vertex_index(vertex) << ", coords: " << cell->vertex(vertex) << "\n";
                    }
                std::cout << std::endl; */                
            }
        }else
        {
//             std::cout << ", not at boundary " << std::endl;      
        }
        
      }
//       std::cout << std::endl;  
      


      cell->get_dof_indices (local_dof_indices);
      
      
      
//       ================================= dealing with Dirichlet boundary conditions
      
      if(is_constraints==1)
      {

        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);          
      }else if(is_constraints==0)
      {
          
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                system_matrix.add (local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i,j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }          
      }
    
/*      std::cout << "    local_dof_indices is ";
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          std::cout << local_dof_indices[i] << " ";
        }
      std::cout << std::endl; */       
        
    }
    
//   std::cout << "system_matrix before applying the Dirichlet BC is " << std::endl;
//   system_matrix.print_formatted(std::cout);
//   std::cout << '\n';
//   std::cout << "system_rhs before applying the Dirichlet BC is " << std::endl;
//   std::cout << system_rhs << std::endl;    

//     std::cout << "    accessed here" << std::endl;
  // As the final step in this function, we wanted to have non-homogeneous
  // boundary values in this example, unlike the one before. This is a simple
  // task, we only have to replace the ZeroFunction used there by an object of
  // the class which describes the boundary values we would like to use
  // (i.e. the <code>BoundaryValues</code> class declared above):
  
      if(is_constraints==0)
      {
        std::map<types::global_dof_index,double> boundary_values;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                    0,
						  BoundaryValues<dim>(),
                                                    boundary_values);
        
        //     std::cout << "      boundary values: " << std::endl;
        //     for (std::map<types::global_dof_index, double>::iterator it=boundary_values.begin();
        //          it!=boundary_values.end();
        //          ++it)
        //       {
        //         std::cout << "      " << it->first << " " << it->second << std::endl;
        //       }
            
        
        //   std::cout << "    accessed here" << std::endl;
        
        MatrixTools::apply_boundary_values (boundary_values,
                                            system_matrix,
                                            solution,
                                            system_rhs);          
      }
  
  
//     std::cout << "system_matrix after applying BCs is " << std::endl;
//     system_matrix.print_formatted(std::cout);
//     std::cout << '\n';
// 
//     std::cout << "system_rhs after applying BCs is " << std::endl;
//     std::cout << system_rhs << std::endl;  


//     for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)                 // obtaining the diagonal and offdiagonal of system_matrix
//     {
//         vector_diagonal[i]=system_matrix.diag_element(i);
//         vector_offdiagonal[i]=system_matrix.el(i,dof_handler.n_dofs()-1-i);
//         
//     }
    
/*    cout << "The diagonal of system_matrix reads: " << endl;
    for (auto &i : vector_diagonal)
    {
        cout << i << " ";
    }
    cout << "\n";
    
    cout << "The offdiagonal of system_matrix reads: " << endl;
    for (auto &i : vector_offdiagonal)
    {
        cout << i << " ";
    }
    cout << "\n";   */ 
    
//     vector_diagonal.print(cout);
//     vector_offdiagonal.print(cout);
    
//   std::ofstream output_matrix("TwoD_uexp_SM_system_matrix_rect_1_1_"+std::to_string(degree)+"_"+std::to_string(refine)+".txt");     // store system_matrix to a file               rect_1_0p2     "_"+std::to_string(cycle_global)+
//   std::cout << "Writing the system matrix to a file..." << std::endl;
//   
//   for (unsigned int i=0;i<dof_handler.n_dofs();i++)
//   {
//     for (unsigned int j=0;j<dof_handler.n_dofs();j++)
//     {
//         output_matrix << system_matrix.el(i, j) << " "; // behaves like cout - cout is also a stream
//     }
//         output_matrix << "\n";
//   } 
//   output_matrix.close();
//   output_matrix.clear();

//   std::ofstream output_rhs("TwoD_uexp_SM_system_rhs_"+std::to_string(degree)+"_"+std::to_string(refine)+".txt");                                           // store system_rhs to a file    
//     std::cout << "Writing the system rhs to a file..." << std::endl;
//   for (unsigned int i=0;i<dof_handler.n_dofs();i++)
//   {
//       output_rhs << system_rhs[i] << " ";               // behaves like cout - cout is also a stream
//       output_rhs << "\n";
//   } 
// 
//   output_rhs.close();
//   output_rhs.clear();  
  
}


// @sect4{Step4::solve}

// Solving the linear system of equations is something that looks almost
// identical in most programs. In particular, it is dimension independent, so
// this function is copied verbatim from the previous example.
template <int dim>
void Step4<dim>::solve ()
{
  std::cout << "Solving... " << std::endl;
  TimerOutput::Scope t(computing_timer, "solve");
  
// ========================= Scaling =========================
//       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
      system_rhs/=u_L2_inte;                // used when scaling u by uL2, f by uL2
      
      
      
/*    std::cout << "    converting LHS and RHS to float precision" << std::endl;

    for (unsigned int i=0;i<dof_handler.n_dofs();i++)                 
    {
        
        system_rhs[i] = float(system_rhs[i]);                       
        
        for (unsigned int j=0;j<dof_handler.n_dofs();j++)
        {
            system_matrix.set(i,j, float(system_matrix.el(i,j)));   
        }
    } */     

      
//     std::cout << "    converting LHS and RHS to half precision" << std::endl;
// 
//     for (unsigned int i=0;i<dof_handler.n_dofs();i++)                 
//     {
//         
//         float rhs_entry_middle(system_rhs[i]);              
//         
//         system_rhs[i] = rhs_entry_middle;                    
//         
//         for (unsigned int j=0;j<dof_handler.n_dofs();j++)
//         {
//             
//             float matrix_entry_middle(system_matrix.el(i,j));
//             
//             system_matrix.set(i,j, matrix_entry_middle);   
//         }
//     }

    
    if (is_UMFPACK==1)          
    {
        cout << "    UMFPack solver " <<  endl;
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult (solution, system_rhs);                  // 
        
    }else if(is_UMFPACK==0)   
    {
        cout << "    CG solver " <<  endl;
            
        SolverControl           solver_control (1e6, 1e-16);
        SolverCG<>       solver (solver_control);
        solver.solve (system_matrix, solution, system_rhs,
                        PreconditionIdentity());
        
        //   solver.compute_eigs_and_cond(vector_diagonal, vector_offdiagonal, solver.this->coefficients_signal, solver.condition_number_signal);

        //   We have made one addition, though: since we suppress output from the
        //   linear solvers, we have to print the number of iterations by hand.
        std::cout << "   " << solver_control.last_step()
                    << " CG iterations needed to obtain convergence."
                    << std::endl;
                    
        CG_iteration_times = solver_control.last_step();
    }
    
    if(is_constraints==1)
    {
        constraints.distribute (solution);
    }
            
/*  std::cout << "solution is: ";
  solution.print(std::cout);      */      
    
}


// @sect4{Step4::output_results}

// This function also does what the respective one did in step-3. No changes
// here for dimension independence either.
//
// The only difference to the previous example is that we want to write output
// in VTK format, rather than for gnuplot. VTK format is currently the most
// widely used one and is supported by a number of visualization programs such
// as Visit and Paraview (for ways to obtain these programs see the ReadMe
// file of deal.II). To write data in this format, we simply replace the
// <code>data_out.write_gnuplot</code> call by
// <code>data_out.write_vtk</code>.
//
// Since the program will run both 2d and 3d versions of the Laplace solver,
// we use the dimension in the filename to generate distinct filenames for
// each run (in a better program, one would check whether <code>dim</code> can
// have other values than 2 or 3, but we neglect this here for the sake of
// brevity).


template <int dim>
void Step4<dim>::compute_L2_norm ()
{

   Vector<double> difference_per_cell (triangulation.n_active_cells());
  
   VectorTools::integrate_difference (dof_handler,             
                                    solution,
				    ZeroFunction<dim>(),
                                    difference_per_cell,
                                    QGauss<dim>(3),
                                    VectorTools::L2_norm);
   solu_L2 = difference_per_cell.l2_norm();

}

template <int dim>
void Step4<dim>::compute_errors ()
{
  std::cout << "Computing errors... " << std::endl;      
  
  ExactSolution<dim> exact_solution(u_L2_inte);
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  
  
  int gauss_plus_error = 2;  

   VectorTools::integrate_difference (dof_handler,
                                     solution_0,
									exact_solution,
                                     difference_per_cell,
                                     QGauss<dim>(degree+gauss_plus_error),
                                     VectorTools::L2_norm);
  L2_N = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::L2_norm);
  
//  std::cout << "||u||_L2 = " << L2_N << std::endl;    
  
  VectorTools::integrate_difference (dof_handler,
                                     solution,
									 exact_solution,
                                     difference_per_cell,
                                     QGauss<dim>(degree+gauss_plus_error),
                                     VectorTools::L2_norm);
  L2_error = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::L2_norm);

 r_N = L2_error/L2_N;
   
 
  VectorTools::integrate_difference (dof_handler,
                                     solution_0,
									 exact_solution,
                                     difference_per_cell,
                                     QGauss<dim>(degree+gauss_plus_error),
                                     VectorTools::H1_seminorm);
  H1_N = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::H1_seminorm);

  VectorTools::integrate_difference (dof_handler,
                                     solution,
									 exact_solution,
                                     difference_per_cell,
                                     QGauss<dim>(degree+gauss_plus_error),
                                     VectorTools::H1_seminorm);
  H1_error = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::H1_seminorm);
  
  
  r_Np = H1_error/H1_N;
  
  VectorTools::integrate_difference (dof_handler,
                                     solution_0,
									 exact_solution,
                                     difference_per_cell,
                                     QGauss<dim>(degree+gauss_plus_error),
                                     VectorTools::H2_seminorm);  
  H2_N = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::H2_seminorm);

  VectorTools::integrate_difference (dof_handler,
                                     solution,
									 exact_solution,
                                     difference_per_cell,
                                     QGauss<dim>(degree+gauss_plus_error),
                                     VectorTools::H2_seminorm);
  H2_error = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::H2_seminorm);  

//   std::cout << "||Eh||_H2_Vol = " << H2_N;
  r_Npp = H2_error/H2_N;

}


template <int dim>
void Step4<dim>::output_results (const unsigned int cycle) const  
{
//   ComputeVelocity1<dim> velocity1;
//   ComputeVelocity2<dim> velocity2;
//   Compute2ndderivative1<dim> secondderivative1;
//   Compute2ndderivative2<dim> secondderivative2;
//   Compute2ndderivative3<dim> secondderivative3;
//   Compute2ndderivative4<dim> secondderivative4;
//   DataOut<dim> data_out;
// 
//   data_out.attach_dof_handler (dof_handler);
//   data_out.add_data_vector (solution, "solution");
// 
//   data_out.add_data_vector (solution, velocity1);
//   data_out.add_data_vector (solution, velocity2);
//   data_out.add_data_vector (solution, secondderivative1);
//   data_out.add_data_vector (solution, secondderivative2);
//   data_out.add_data_vector (solution, secondderivative3);
//   data_out.add_data_vector (solution, secondderivative4);
// 
//   data_out.build_patches ();
// 
//   std::ofstream output (dim == 1 ?
//                         "solution-1d.vtk" :
//                         "solution-2d.vtk");
//   data_out.write_vtk (output);
  
// ==========================================================       output results for every cycle
    
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ofstream output ("solution_TwoD_Pois_uexpexp-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk (output);
  }  
  
}


template <int dim>
void Step4<dim>::refine_grid ()                 
{
  std::cout << "==================== Adaptive refinement, " << "cycle " << cycle_global << std::endl;             
  
  std::cout << "    the number of active cells are:" << triangulation.n_active_cells() << std::endl;
  
  Vector<double> estimated_error_per_cell (triangulation.n_active_cells());
  
  KellyErrorEstimator<dim,dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);
  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.00);
  
//   std::cout << "estimated_error_per_cell: ";
//   estimated_error_per_cell.print(std::cout);
  
//   {
//     DataOut<dim> data_out;
//     data_out.attach_dof_handler (dof_handler);
//     data_out.add_data_vector (estimated_error_per_cell, "estimated_error_per_cell");
//     data_out.build_patches ();
//     std::ofstream output ("estimated_error_per_cell_step_4-" + std::to_string(cycle_global-1) + ".vtk");
//     data_out.write_vtk (output);
//   }    
  
  triangulation.execute_coarsening_and_refinement ();
}

// @sect4{Step4::run}

// This is the function which has the top-level control over everything. Apart
// from one line of additional output, it is the same as for the previous
// example.
template <int dim>
void Step4<dim>::run ()
{
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl << endl;
        
    for (unsigned int cycle=0; cycle<1;cycle++)              // used for adaptive mesh refinement
    {
                  
        cycle_global = cycle;
        
        if (cycle==0)
        {
            make_grid();
//             print_mesh_info (triangulation, "grid-1.eps");  
        }else
        {
            triangulation.refine_global (1);
//             refine_grid();                              // not done yet, Apr. 4, 2019
        }
                  

        setup_system ();                  
        assemble_system ();
        solve ();
//         compute_L2_norm ();
        
        compute_errors ();
        output_results (cycle);
//         
        my_CPU_time += computing_timer.return_total_cpu_time ();
        
        print_errors_and_CPU_time ();
        
        printToFile();
       
    }
  
}


template <int dim>
void Step4<dim>::print_errors_and_CPU_time ()
{
    
  cout << "\nprint errors and CPU time" << endl;

  cout << "solu_L2 = " << solu_L2 << endl;

  std::cout << "||Eh||_L2 = " << L2_error;  
  std::cout << ", r_N = " << r_N << std::endl;

  std::cout << "||Eh||_H1_seminorm = " << H1_error;
  std::cout << ", r_Np = " << r_Np << std::endl;
    
  std::cout << "||Eh||_H2_seminorm = " << H2_error;
  std::cout << ", r_Npp = " << r_Npp << std::endl;
    
  cout << "my_CPU_time is " << my_CPU_time << "\n" << "\n";
      
}


// @sect3{The <code>main</code> function}

// And this is the main function. It also looks mostly like in step-3, but if
// you look at the code below, note how we first create a variable of type
// <code>Step4@<2@></code> (forcing the compiler to compile the class template
// with <code>dim</code> replaced by <code>2</code>) and run a 2d simulation,
// and then we do the whole thing over in 3d.
//
// In practice, this is probably not what you would do very frequently (you
// probably either want to solve a 2d problem, or one in 3d, but not both at
// the same time). However, it demonstrates the mechanism by which we can
// simply change which dimension we want in a single place, and thereby force
// the compiler to recompile the dimension independent class templates for the
// dimension we request. The emphasis here lies on the fact that we only need
// to change a single place. This makes it rather trivial to debug the program
// in 2d where computations are fast, and then switch a single place to a 3 to
// run the much more computing intensive program in 3d for `real'
// computations.
//
// Each of the two blocks is enclosed in braces to make sure that the
// <code>laplace_problem_2d</code> variable goes out of scope (and releases
// the memory it holds) before we move on to allocate memory for the 3d
// case. Without the additional braces, the <code>laplace_problem_2d</code>
// variable would only be destroyed at the end of the function, i.e. after
// running the 3d problem, and would needlessly hog memory while the 3d run
// could actually use it.
int main (int argc, char *argv[])
{
//     std::remove("data_error.txt");

    double u_L2_inte = 1.0;
    int degree = 2;
    int refine = 3; 
    
//     std::cout << "Have " << argc << " arguments: ";
//     for (int i = 0; i < argc; ++i)
//     {
//         std::cout << argv[i][0] << " ";
//     }
//     std::cout << std::endl;
    
    if (argc != 4)
    {
        std::cout << "  ./step-4 <u_L2_inte> <degree> <refine>" << std::endl;
        exit(EXIT_FAILURE);
    }else
    {
        u_L2_inte = atof(argv[1]);
        degree=atoi(argv[2]);
        refine=atoi(argv[3]);  
    }
    
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~"
            << " Degree "
            << degree     
            << " Refine "
            << refine 
            << " ~~~~~~~~~~~~~~~~~~~~~~~"
            << std::endl;	  
    
  deallog.depth_console (0);

  /*{
    Step4<1> laplace_problem_1d;
    laplace_problem_1d.run ();
  }*/

 {
   Step4<2> laplace_problem_2d(u_L2_inte,degree,refine);
    laplace_problem_2d.run ();
  }

  return 0;
}
