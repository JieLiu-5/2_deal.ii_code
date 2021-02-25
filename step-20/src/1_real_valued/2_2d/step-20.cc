/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2005 - 2016 by the deal.II authors
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
 * Author: Wolfgang Bangerth, Texas A&M University, 2005, 2006
 */


// @sect3{Include files}

// Since this program is only an adaptation of step-4, there is not much new
// stuff in terms of header files. In deal.II, we usually list include files
// in the order base-lac-grid-dofs-fe-numerics, followed by C++ standard
// include files:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>        // Sep 13, 2018

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/timer_return_total_cpu_time.h>

#include <fstream>
#include <iostream>

// This is the only significant new header, namely the one in which the
// Raviart-Thomas finite element is declared:

#include <deal.II/base/tensor_function.h>

#define pi 3.141592653589793238462643

// The last step is as in all previous programs:
namespace Step20
{
  using namespace dealii;
  using namespace std;

  // @sect3{The <code>MixedLaplaceProblem</code> class template}

  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem (const unsigned int id_case, const double coeff_inner_x, const unsigned int id_scaling_scheme, const double u_L2_inte, const double up_L2_inte, const unsigned int degree, const unsigned int refine);
    void run ();

  private:
      
    const unsigned int id_case = 1;
    const double coeff_inner_x;
    const unsigned int id_scaling_scheme=1;
    const double u_L2_inte, up_L2_inte;
    
      
    double p_L2 = 0;
    double inte_p = 0;
    double r_p_L2 = 0;
    double u_L2, u_x_L2, u_y_L2= 0;
    double inte_u = 0;
    double r_u_L2 = 0;
    double du_L2 = 0;
    double inte_du = 0;
    double r_du_L2 = 0;
    
    const unsigned int is_constraints=1;
    const unsigned int is_UMFPACK = 1;
    unsigned int CG_iteration_schur = 1;
    
    
    double my_CPU_time;  
    int iteration_times;    
    
    
    void make_grid_and_dofs ();
    void print_mesh_info (const Triangulation<dim> &triangulation);
    void assemble_system ();
    void solve ();
    void compute_errors ();
    void print_errors_and_cpu_time () const;
    void output_results () const;

    const unsigned int   degree;
    const unsigned int   refine;
    
    unsigned int n_u=0;
    unsigned int n_p=0;

    Triangulation<dim>   triangulation;
    
//     FE_BDM<dim>          fe_bdm;
    FESystem<dim>        fe;
//     FESystem<dim>        fe_Q_try;
    
    DoFHandler<dim>      dof_handler;

    // The second difference is that the sparsity pattern, the system matrix,
    // and solution and right hand side vectors are now blocked. What this
    // means and what one can do with such objects is explained in the
    // introduction to this program as well as further down below when we
    // explain the linear solvers and preconditioners for this problem:
    
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double>       solution;
    BlockVector<double>       solution_0;
    BlockVector<double>       system_rhs;
    
    TimerOutput                               computing_timer;
    void printToFile() const;
    
    ConstraintMatrix constraints;
    
  };
  

  std::ofstream myfile;
  template <int dim>
  void MixedLaplaceProblem<dim>::printToFile() const
  {
      myfile.open ("data_error_TwoD_MM_uxm0p5square_feq_fedgq_P4P3.txt", std::ofstream::app);                   // _denom_1e2
      myfile << refine <<" ";
//       myfile << 1.0/(std::pow(triangulation.n_vertices(),0.5)-1.0) <<" ";  
      myfile << triangulation.n_vertices() <<" ";
      myfile << dof_handler.n_dofs() << " ";
      myfile << n_u << " ";
      myfile << n_p << " ";      
      myfile << p_L2 << " ";
//       myfile << r_p_L2 << " ";
      myfile << u_L2 << " ";
//       myfile << r_u_L2 << " ";
      myfile << du_L2 << " ";
//       myfile << r_du_L2 << " ";
      myfile << my_CPU_time << " ";
      if (is_UMFPACK == 1)
      {
          myfile << "UMF" << "\n";
      }else if (is_UMFPACK == 0)
      {
          myfile << CG_iteration_schur << "\n";
      }
  }

    template <int dim>
    class Common_var
    {
    public:
        Common_var(const unsigned int id_case);
//     protected:
        double coeff_outer_x, coeff_outer_y;
        double coeff_inner_y;
        double center_x, center_y;
        double const_inde;
    
    };
    
  
  // @sect3{Right hand side, boundary values, and exact solution}

  template <int dim>
  class RightHandSide : public Function<dim>,
  public Common_var<dim>
  {
  public:
    using Common_var<dim>::Common_var;
    RightHandSide (const unsigned int id_case, const double coeff_inner_x);       // : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  protected:
    const unsigned int id_case;
    const double coeff_inner_x;                          
  };
  

  template <int dim>
  class PressureBoundaryValues : public Function<dim>,
  public Common_var<dim>
  {
  public:
    using Common_var<dim>::Common_var;
    PressureBoundaryValues (const unsigned int id_case, const double coeff_inner_x);      // : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  protected:
    const unsigned int id_case;
    const double coeff_inner_x;                          
  };


  template <int dim>
  class ExactSolution : public Function<dim>,
  public Common_var<dim>
  {
  public:
    using Common_var<dim>::Common_var;
    ExactSolution (const unsigned int id_case, const double coeff_inner_x, const unsigned int id_scaling_scheme, const double u_L2_inte, const double up_L2_inte);          // : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    virtual void vector_gradient (const Point<dim> &p,
                               std::vector<Tensor<1,dim>>   &gradients) const;   
                               
  protected:
    const unsigned int id_case;
    const double coeff_inner_x;
    const unsigned int id_scaling_scheme;
    const double u_L2_inte, up_L2_inte;                                
  };
  
  
  template <int dim>
  RightHandSide<dim>::RightHandSide(const unsigned int id_case, const double coeff_inner_x): Function<dim>(dim+1),Common_var<dim>(id_case),
  id_case(id_case),
  coeff_inner_x (coeff_inner_x)
  {} 
  
  template <int dim>
  PressureBoundaryValues<dim>::PressureBoundaryValues(const unsigned int id_case, const double coeff_inner_x): Function<dim>(dim+1),Common_var<dim>(id_case),
  id_case(id_case),
  coeff_inner_x (coeff_inner_x)
  {}   
  
  template <int dim>
  ExactSolution<dim>::ExactSolution(const unsigned int id_case, const double coeff_inner_x, const unsigned int id_scaling_scheme, const double u_L2_inte, const double up_L2_inte): Function<dim>(dim+1),Common_var<dim>(id_case),
  id_case(id_case),
  coeff_inner_x (coeff_inner_x),
  id_scaling_scheme(id_scaling_scheme),
  u_L2_inte(u_L2_inte),
  up_L2_inte(up_L2_inte)
  {
      std::cout << "    using ExactSolution, " << "id_case: " << id_case << ", id_scaling_scheme: " << id_scaling_scheme << ", u_L2_inte: " << u_L2_inte << ", up_L2_inte: " << up_L2_inte << std::endl;
  } 
  
  
  
  
    template <int dim>
    Common_var<dim>::Common_var(const unsigned int id_case)
    {
      switch (id_case)
      {
          case 1:
              coeff_outer_x = 1;
              coeff_outer_y = 0;
              coeff_inner_y = 1;
              center_x = 0.5;
              center_y = 0.5;
              const_inde = 0.0;
              break;
          case 2:
              coeff_outer_x = 1;
              coeff_outer_y = 0;
              coeff_inner_y = 1;
              center_x = 0.5;
              center_y = 0.5;
              const_inde = 0.0;
              break;
          case 3:
              coeff_outer_x = 1;
              coeff_outer_y = 0;
              coeff_inner_y = 1;
              center_x = 0.5;
              center_y = 0.5;
              const_inde = 0.0;
              break;
          case 4:
              coeff_outer_x = 1;
              coeff_outer_y = 0;
              coeff_inner_y = 1;
              center_x = 0.5;
              center_y = 0.5;
              const_inde = 0.0;
              break;
          case 5:
              coeff_outer_x = 1;
              coeff_outer_y = 0;
              coeff_inner_y = 1;
              center_x = 0.5;
              center_y = 0.5;
              const_inde = 0.0;
              break;
          default:
              cout << "case does not exist" << endl;
              throw exception();              
      }        
    }
    
  
  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &p,
                                    const unsigned int /*component*/) const
  {
      switch (id_case)
      {
          case 1:
              return 1;
              break;
          case 2:
              return -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(((4.0*pow(this->coeff_outer_x*(p[0]-this->center_x),2)-2.0*this->coeff_outer_x*this -> coeff_inner_x)+(4.0*pow(this->coeff_outer_y*(p[1]-this->center_y),2)-2.0*this->coeff_outer_y*this -> coeff_inner_x))/pow(this -> coeff_inner_x,2));
              break;
          case 3:
              return 1;
              break;
          case 4:
              return 1;
              break;
          case 5:
              return -2.0*this->coeff_outer_x/pow(coeff_inner_x,2)-2.0*this->coeff_outer_y/pow(this->coeff_inner_y,2);              // this-> reference variables in the base class, coeff_inner_x is a variable inside the class
              break;
          default:
              cout << "case does not exist" << endl;
              throw exception();
              
      }
      return 0;
  }

  
  template <int dim>
  double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
  {
      
      switch (id_case)
      {
          case 1:
              return 1;
              break;
          case 2:
              return exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x);
              break;
          case 3:
              return 1;
              break;
          case 4:
              return 1;
              break;
          case 5:
              return this->coeff_outer_x*pow((p[0]-this->center_x)/coeff_inner_x,2)+this->coeff_outer_y*pow((p[1]-this->center_y)/this->coeff_inner_y,2)+this->const_inde;
              break;
          default:
              cout << "case does not exist" << endl;
              throw exception();
              
      }
      return 0;
      
  }    
  

  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
      
//     std::cout << "    ExactSolution::vector_value()" << std::endl;
    
    Assert (values.size() == dim+1,
            ExcDimensionMismatch (values.size(), dim+1));

      switch (id_case)
      {
          case 1:
              values(0)=1;
              values(1)=1;
              values(2)=1;
              break;
          case 2:
              values(0) = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(-2*this->coeff_outer_x*(p[0]-this->center_x)/this -> coeff_inner_x);          //-px
              values(1) = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(-2*this->coeff_outer_y*(p[1]-this->center_y)/this -> coeff_inner_x);          //-py
              values(2) = exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x);        //p
              break;
          case 3:
              values(0)=1;
              values(1)=1;
              values(2)=1;
              break;
          case 4:
              values(0)=1;
              values(1)=1;
              values(2)=1;
              break;
          case 5:
              values(0) = -2*(p[0]-this->center_x)*this->coeff_outer_x/coeff_inner_x/coeff_inner_x;              //-px
              values(1) = -2*(p[1]-this->center_y)*this->coeff_outer_y/this->coeff_inner_y/this->coeff_inner_y;                                  //-py
              values(2) = this->coeff_outer_x*pow((p[0]-this->center_x)/coeff_inner_x,2)+this->coeff_outer_y*pow((p[1]-this->center_y)/this->coeff_inner_y,2)+this->const_inde;        //p
              break;
          default:
              cout << "case does not exist" << endl;
              throw exception();              
      }
      

    
    if(id_scaling_scheme==1)
    {
        
        values(0)/=up_L2_inte;    
        values(1)/=up_L2_inte;    
        values(2)/=u_L2_inte;    
        
    }else if(id_scaling_scheme==2)
    {
        
        values(0)/=u_L2_inte;    
        values(1)/=u_L2_inte;    
        values(2)/=u_L2_inte;    
        
    }    
    
  }
  
  template <int dim>
  void
  ExactSolution<dim>::vector_gradient (const Point<dim> &p,
          	  	  	  	  	  	  	 std::vector<Tensor<1,dim>>   &gradients) const
  {
      
//     std::cout << "    ExactSolution::vector_gradient()" << std::endl;
    
      
      switch (id_case)
      {
          case 1:
              gradients[0][0] = 1;
              gradients[0][1] = 1;
              gradients[1][0] = 1;
              gradients[1][1] = 1;
              gradients[2][0] = 1;
              gradients[2][1] = 1;
              break;
          case 2:
              gradients[0][0] = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(((4.0*pow(this->coeff_outer_x*(p[0]-this->center_x),2)-2.0*this->coeff_outer_x*this -> coeff_inner_x))/pow(this -> coeff_inner_x,2));          //-pxx
              gradients[0][1] = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(((4.0*this->coeff_outer_x*(p[0]-this->center_x)*this->coeff_outer_y*(p[1]-this->center_y)))/pow(this -> coeff_inner_x,2));         //-pxy
              gradients[1][0] = gradients[0][1];         //-pyx
              gradients[1][1] = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*((4.0*pow(this->coeff_outer_y*(p[1]-this->center_y),2)-2.0*this->coeff_outer_y*this -> coeff_inner_x)/pow(this -> coeff_inner_x,2));          //-pyy
              gradients[2][0] = exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(-2*this->coeff_outer_x*(p[0]-this->center_x)/this -> coeff_inner_x);     //px
              gradients[2][1] = exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(-2*this->coeff_outer_y*(p[1]-this->center_y)/this -> coeff_inner_x);     //py
              break;
          case 3:
              gradients[0][0] = 1;
              gradients[0][1] = 1;
              gradients[1][0] = 1;
              gradients[1][1] = 1;
              gradients[2][0] = 1;
              gradients[2][1] = 1;
              break;
          case 4:
              gradients[0][0] = 1;
              gradients[0][1] = 1;
              gradients[1][0] = 1;
              gradients[1][1] = 1;
              gradients[2][0] = 1;
              gradients[2][1] = 1;
              break;
          case 5:
              gradients[0][0] = -2.0*this->coeff_outer_x/pow(coeff_inner_x,2);          //-pxx
              gradients[0][1] = 0.0;                          //-pxy
              gradients[1][0] = 0.0;                          //-pyx
              gradients[1][1] = -2.0*this->coeff_outer_y/pow(this->coeff_inner_y,2);           //-pyy
              gradients[2][0] = 2*(p[0]-this->center_x)*this->coeff_outer_x/pow(coeff_inner_x,2);     //px
              gradients[2][1] = 2*(p[1]-this->center_y)*this->coeff_outer_y/pow(this->coeff_inner_y,2);     //py
              break;
          default:
              cout << "case does not exist" << endl;
              throw exception();              
      }
    
    
    if(id_scaling_scheme==1)
    {
        
        gradients[0]/=up_L2_inte;
        gradients[1]/=up_L2_inte;
        gradients[2]/=u_L2_inte;          
        
    }else if(id_scaling_scheme==2)
    {
        
        gradients[0]/=u_L2_inte;
        gradients[1]/=u_L2_inte;
        gradients[2]/=u_L2_inte;
        
    }

  }






  
  
  
  
  
  
  
  
  
  
  
  
  
  
  template <int dim>                                         //create the boundary for the velocity
  class GradientBoundary : public Function<dim>,
  protected Common_var<dim>
  {
  public:
    GradientBoundary () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };
  
  template <int dim>
  void
  GradientBoundary<dim>::vector_value (const Point<dim> &/*p*/,
                                  Vector<double>   &values) const
  {
//     Assert (values.size() == 1+1,
//             ExcDimensionMismatch (values.size(), 1+1));

    values(0) = 0;      //u_real(-dp/dx)|x=1
    values(1) = 0;
    values(2) = 0;
//     values(3) = 0;
  }
  
  // @sect3{The inverse permeability tensor}

  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };

  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();

        for (unsigned int d=0; d<dim; ++d)
          values[p][d][d] = 1.;
      }
  }  
  

  // @sect3{MixedLaplaceProblem class implementation}

  // @sect4{MixedLaplaceProblem::MixedLaplaceProblem}

  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem (const unsigned int id_case, const double coeff_inner_x, const unsigned int id_scaling_scheme, const double u_L2_inte, const double up_L2_inte, const unsigned int degree, const unsigned int refine)
    :
    id_case(id_case),
    coeff_inner_x (coeff_inner_x),
    id_scaling_scheme(id_scaling_scheme),
    u_L2_inte (u_L2_inte),
    up_L2_inte (up_L2_inte),
    degree (degree),
    refine (refine),
        
    fe (FE_Q<dim>(degree), dim,
        FE_DGQ<dim>(degree-1), 1),

    dof_handler (triangulation),
	computing_timer  (std::cout, TimerOutput::summary, TimerOutput::cpu_times)
  {
      std::cout << "    coeff_inner_x: " << coeff_inner_x << ", id_scaling_scheme: " << id_scaling_scheme << ", u_L2_inte: " << u_L2_inte << ", up_L2_inte: " << up_L2_inte << std::endl;
      
}



  // @sect4{MixedLaplaceProblem::make_grid_and_dofs}

  // This next function starts out with well-known functions calls that create
  // and refine a mesh, and then associate degrees of freedom with it:
  template <int dim>
  void MixedLaplaceProblem<dim>::make_grid_and_dofs ()
  {
    std::cout << "Making grid... " << std::endl; 
    TimerOutput::Scope t(computing_timer, "make_grid_and_dofs");
//     std::cout << std::endl;
    
//     ~~~~~~~~~~~~~~~~~~~ using mesh from the embedded function in deal.ii
        
    const Point<dim> p1 (0.0, 0.0);                            // square
    const Point<dim> p2 (1.0, 1.0);    
    GridGenerator::hyper_cube (triangulation, 0, 1);        
    
//     triangulation.begin_active()->face(1)->set_boundary_id(2);       // right
    
    if(is_constraints==1)
    {
        
        std::cout << "    consisting of Neumann boundary conditions, constraints used" << std::endl;
        
        triangulation.begin_active()->face(2)->set_boundary_id(1);       // bottom
        triangulation.begin_active()->face(3)->set_boundary_id(1);       // top        
    }


//     const Point<dim> p1 (0.0, 0.0);                         // rectangle with specified step size
//     const Point<dim> p2 (1.0, 0.02);
//     const std::vector<unsigned int> repetitions={(unsigned int)(p2[0]/p2[1]),1};
//     GridGenerator::subdivided_hyper_rectangle (triangulation, repetitions, p1, p2);

    
    //     ~~~~~~~~~~~~~~~~~~~ using mesh from gmsh
    
    
/*    GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f("square_geometry.msh");          //  exponential_geometry        rectangular_geometry      square_geometry      
    gridin.read_msh(f); */   


//     ~~~~~~~~~~~~~~~~~~~ operate on the mesh

      Triangulation<2>::active_face_iterator face = triangulation.begin_active_face();      
      Triangulation<2>::active_face_iterator endface = triangulation.end_face();    
      
      for (; face!=endface; ++face)                                 // this iterates from the first active face to the last active face
      {
          if(face->at_boundary())
          {
                if(face->vertex(0)(0)==0 && face->vertex(1)(0)==0)              // denoting left boundary
                {
//                     std::cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the left boundary" << std::endl;
//                     face->set_boundary_id(1);
                }else if(face->vertex(0)(1)==p2[1] && face->vertex(1)(1)==p2[1])        // upper boundary
                {
//                     std::cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the upper boundary" << std::endl;
//                     face->set_boundary_id(1);
                }else if(face->vertex(0)(1)==p1[1] && face->vertex(1)(1)==p1[1])        // lower boundary
                {
//                     std::cout << "    vertex(0): (" << face->vertex(0) << "), vertex(1): (" << face->vertex(1) << "), I'm on the bottom boundary" << std::endl;
//                     face->set_boundary_id(1);
                }
          }
      }   


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~              // outputting data on the triangulation

    std::cout << "    vertices_per_cell: " << GeometryInfo< dim >::vertices_per_cell << std::endl;
    
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end(); 
    
    
    for (; cell!=endc; ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary())
            {
                if(cell->face(face)->boundary_id() == 1)		                                            // info on the Neumann boundary conditions
                {
                    std::cout << "    at cell " << cell->active_cell_index() << ", face " << face << ", the boundary id is 1";   
                    std::cout << ", coords of the face: ";   
                    for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
                        {
                            std::cout << cell->face(face)->vertex(vertex) << " | ";                   // << "      at vertex " << cell->face(face)->vertex(vertex) << ", coords: "
                        }
                    std::cout << std::endl;   
                }else
                {
                    std::cout << "    at cell " << cell->active_cell_index() << ", face " << face << ", the boundary id is 0";   
                    std::cout << ", coords of the face: ";   
                    for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
                        {
                            std::cout << cell->face(face)->vertex(vertex) << " | ";                   
                        }
//                     for (unsigned int dof=0; dof < GeometryInfo<dim>::vertices_per_face; ++dof)
//                         {
//                             std::cout << cell->face(face)->vertex(dof)->vertex_dofs << " | ";                   
//                         }
                        
                    std::cout << std::endl;  
                }
            }
          }
      }
    
    triangulation.refine_global (refine);
      
// ===================================================   // outputting data on fe
    
    dof_handler.distribute_dofs (fe);
    
    std::cout << "    fe.get_name: " << fe.get_name() << std::endl;
    
//     for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
//     {   
//         if (fe.is_primitive(i))
//         {
//             std::cout << i << "th dof is primitive";               // << ", shape function index is: " << ", system_to_component_index is: " << fe.system_to_component_index(i).first
//         }else
//         {
//             std::cout << i << "th dof is non-primitive";
//         }
//         std::cout << "   fe.system_to_base_index(i) " << fe.system_to_base_index(i).first.first << " " << fe.system_to_base_index(i).first.second << " " << fe.system_to_base_index(i).second << std::endl;
//     }
//     
//     for (unsigned int i = 0; i < fe.dofs_per_face; i++)
//     {   
//         std::cout << "   fe.face_system_to_base_index(i) " << fe.face_system_to_base_index(i).first.first << " " << fe.face_system_to_base_index(i).first.second << " " << fe.face_system_to_base_index(i).second << " " << std::endl;
//     }    
//     
//     std::cout << "    fe.n_dofs_per_vertex(): " << fe.n_dofs_per_vertex() << std::endl;
//     
//     std::cout << "    coords of support nondes in the reference cell: ";
//     for(unsigned int i = 0; i < fe.get_unit_support_points().size(); ++i)
//     {
//         std::cout << fe.get_unit_support_points()[i] << " | ";
//     }
//     std::cout << std::endl;
    

// ===================================================    
    
//     std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);
//     
//     typename DoFHandler<dim>::active_cell_iterator
//     cell_dof = dof_handler.begin_active(),
//     endc_dof = dof_handler.end();
//     
//     cell_dof->get_dof_indices (local_dof_indices);
//     std::cout << "    local_dof_indices reads ";
//     for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
//     {
//         std::cout << local_dof_indices[i] << " ";
//     }
//     std::cout << std::endl;  
//     
//     std::cout << "    indices for degrees of freedom located at vertices" << std::endl;
// 
//     for (; cell_dof!=endc_dof; ++cell_dof)
//       {
//           
//           std::cout << cell_dof->vertex(0)(0) << std::endl;
//           
//         for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//         {
//             for (unsigned int i = 0; i < 3; i++)
//             {
//                 std::cout << cell_dof->vertex_dof_index (v,i) << " ";                
//             }
//             std::cout << "; ";
//         }
//         std::cout << std::endl;
//             
//       }
    
//========================================Below are extracted from step-22==========================================
    
    
    
//     A_preconditioner.reset ();
//     system_matrix.clear ();
//     preconditioner_matrix.clear ();
    
    DoFRenumbering::Cuthill_McKee (dof_handler);
    std::vector<unsigned int> block_component (dim+1,0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise (dof_handler, block_component);                  // this changes the order in which dofs are numbered, which can be tested by outputting 'local_dof_indices'
    
/*    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)                             // this allows outputting the information on the dofs associated with fe
    {   
        if (fe.is_primitive(i))
        {
            std::cout << i << "th dof is primitive";               //
        }else
        {
            std::cout << i << "th dof is non-primitive";
        }
        std::cout << "   fe.system_to_base_index(i) " << fe.system_to_base_index(i).first.first << " " << fe.system_to_base_index(i).first.second << " " << fe.system_to_base_index(i).second 
        << ", coord: " << fe.get_unit_support_points()[i] << ", local_dof_index: " << local_dof_indices[i] << std::endl;
    }*/  

    if(is_constraints==1)
    {
        constraints.clear ();
        FEValuesExtractors::Vector velocities(0);                   // this acts as a local variable inside the braces
        DoFTools::make_hanging_node_constraints (dof_handler,
                                                constraints);

        //std::map<types::global_dof_index,double> boundary_values_Neu;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                ExactSolution<dim>(id_case, coeff_inner_x, id_scaling_scheme, 1, 1),
                                                constraints,
                                                fe.component_mask(velocities));
        
        std::cout << "    the number of constraints is: " << constraints.n_constraints() << endl;
        
    //     cout << ", of which the dof number is: ";
    //     for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
    //     {
    //         if(constraints.is_constrained(i))
    //         {
    //             std::cout << i << ", ";
    //         }
    //     }
    //     std::cout << std::endl;

    //      
    //     std::cout << "    the content of constraints is: " << std::endl;
    //     constraints.print(std::cout);    
    //     
    //     std::cout << "    ConstraintMatrix::write_dot is: " << std::endl;
    //     constraints.write_dot(std::cout); 
    
        constraints.close ();
    }
    
//========================================Above are extracted from step-22==========================================   
   


    
    std::vector<types::global_dof_index> dofs_per_block (2);
    
//     std::cout << "    accessed here 1" << std::endl;
    
    DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
    
//     std::cout << "    accessed here 2" << std::endl;

    n_u = dofs_per_block[0];
    n_p = dofs_per_block[1];
    
    
//     std::vector<types::global_dof_index> dofs_per_component (dim);
//     DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
//     n_u = 2*dofs_per_component[0];
//     n_p = dofs_per_component[dim];
    
    std::cout << "    Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "    Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')'
              << std::endl;
    
    {
      BlockDynamicSparsityPattern dsp (2,2);
      dsp.block(0,0).reinit (n_u, n_u);
      dsp.block(1,0).reinit (n_p, n_u);
      dsp.block(0,1).reinit (n_u, n_p);
      dsp.block(1,1).reinit (n_p, n_p);
      dsp.collect_sizes();   
      
      if(is_constraints==1)
      {
          DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);            // we still use that in Step-20 for the following two lines, Feb. 22, 2019
      }else if (is_constraints==0)
      {
//           std::cout << "    constraints not used" << std::endl;
          DoFTools::make_sparsity_pattern (dof_handler, dsp);              
      }
      
      sparsity_pattern.copy_from (dsp);
    
    system_matrix.reinit (sparsity_pattern);
//     preconditioner_matrix.reinit (preconditioner_sparsity_pattern);
    solution.reinit (2);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.collect_sizes ();
    
    solution_0.reinit (2);
    solution_0.block(0).reinit (n_u);
    solution_0.block(1).reinit (n_p);
    solution_0.collect_sizes ();   
    
    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();
  }
  }
  
  
  template <int dim>
void MixedLaplaceProblem<dim>::print_mesh_info(const Triangulation<dim> &triangulation)
{
  std::cout << "Mesh info:" << std::endl
            << "    dimension: " << dim << std::endl
            << "    no. of cells: " << triangulation.n_active_cells() << std::endl;

  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary())
            {
                boundary_count[cell->face(face)->boundary_id()]++;
            }
          }
      }

    std::cout << "    boundary indicators: ";
    for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
         it!=boundary_count.end();
         ++it)
      {
        std::cout << it->first << "(" << it->second << " times) ";
      }
    std::cout << std::endl;
  }
}

  // @sect4{MixedLaplaceProblem::assemble_system}

  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_system ()
  {
    std::cout << "Assembling... " << std::endl;  
    TimerOutput::Scope t(computing_timer, "assemble_system");
//     std::cout << std::endl;
    
    QGauss<dim>   quadrature_formula(degree+2);
    QGauss<dim-1> face_quadrature_formula(degree+2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values    | update_normal_vectors |
                                      update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

//     const unsigned int   dofs_per_face   = fe.dofs_per_face;
    
    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    std::vector<types::global_dof_index> local_face_dof_indices (fe.dofs_per_face);

    const RightHandSide<dim>          right_hand_side(id_case, coeff_inner_x);
    const PressureBoundaryValues<dim> pressure_boundary_values(id_case, coeff_inner_x);
    const KInverse<dim>               k_inverse;

    std::vector<double> rhs_values (n_q_points);
    std::vector<double> boundary_values (n_face_q_points);
    std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        
        fe_values.reinit (cell);
        local_matrix = 0;
        local_rhs = 0;
        
        std::cout << "    Cell no.: " << cell->active_cell_index() << "/" << triangulation.n_active_cells()-1 << std::endl;     
        
        for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
            {
                std::cout << "      at vertex " << cell->vertex_index(vertex) << ", coords: " << cell->vertex(vertex) << "\n";
            }
        std::cout << std::endl;           
        
        cell->get_dof_indices (local_dof_indices);        

        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);
        k_inverse.value_list (fe_values.get_quadrature_points(),
                              k_inverse_values);
        
        std::cout << "    local_dof_indices: ";
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
              std::cout << local_dof_indices[i] << " ";           
        }
        std::cout << std::endl;
        
//         int dof_index=0;
//         
//         for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//         {
// //             Assert (dof_handler.get_fe().dofs_per_cell ==
// //                     GeometryInfo<dim>::vertices_per_cell,
// //                     ExcNotImplemented());
//             std::cout << "    v is: " << v << std::endl; 
//             dof_index = cell->vertex_dof_index (1,0);
//             dof_index++;
//             std::cout << "    dof_index is: " << dof_index << std::endl; 
//         }
        
        
        
        
        for (unsigned int q=0; q</*1*/n_q_points; ++q)
        {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const Tensor<1,dim> phi_i_u     = fe_values[velocities].value (i, q);
              const double        div_phi_i_u = fe_values[velocities].divergence (i, q);
              const double        phi_i_p     = fe_values[pressure].value (i, q);
              
//                 std::cout << " -------------------- " << i << " th(rd) test function" << ", dof number: " << local_dof_indices[i] << ", component_index: " << fe.system_to_component_index(i).first << " -------------------- ";
//                 if (phi_i_u[0]!=0)
//                 {
//                     std::cout << "  x component of the real part of the velocity " << std::endl;
//                 }else if(phi_i_u[1]!=0)
//                 {
//                     std::cout << "  y component of the real part of the velocity " << std::endl;
//                 }else if(phi_i_p!=0)
//                 {
//                     std::cout << "  real part of the pressure " << std::endl;
//                 }else
//                 {
//                     std::cout << "  not a component" << std::endl;
//                 }
                
                
//                 std::cout << "      phi_i_u is " << phi_i_u << ", div_phi_i_u is " << div_phi_i_u << ", phi_i_p is " << phi_i_p << std::endl;
                              
                
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  const Tensor<1,dim> phi_j_u     = fe_values[velocities].value (j, q);
                  const double        div_phi_j_u = fe_values[velocities].divergence (j, q);
                  const double        phi_j_p     = fe_values[pressure].value (j, q);

                  local_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u
                                        - div_phi_i_u * phi_j_p
                                        - phi_i_p * div_phi_j_u)
                                       * fe_values.JxW(q);
                }

              local_rhs(i) += -phi_i_p *
                              rhs_values[q] *
                              fe_values.JxW(q);
            }
        }
            
//         std::cout << "    accessed here 1" <<  std::endl;            

//         bool cell_is_at_boundary;
        
//         for (unsigned int face_n=0;
//              face_n<GeometryInfo<dim>::faces_per_cell;
//              ++face_n)
//         {
//           if (cell->at_boundary(face_n))
//           {
//               cell_is_at_boundary=1;
//               break;
//           }
//         }
          
//           if (cell_is_at_boundary)
//           {
//               std::cout << "    cell " << cell->active_cell_index() << " is at boundaries";
//                 std::cout << ", of which the vertex coords are: ";   
//                 for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
//                     {
//                         std::cout << "vertex " << cell->vertex_index(vertex) << ": " << cell->vertex(vertex) << ", ";
//                     }
//                 std::cout << std::endl; 
//           }
 
        cell->get_dof_indices (local_dof_indices);
                
        for (unsigned int face_n=0; face_n< GeometryInfo<dim>::faces_per_cell; ++face_n)
        {
              if(cell->face(face_n)->boundary_id() == 0)                 //
            {
//                 std::cout << "    %%%%%%%%%%%%%%  face " << face_n << ": ";   
//                 for (unsigned int vertex=0; vertex < GeometryInfo<dim>::vertices_per_face; ++vertex)
//                     {
//                         std::cout << "(" << cell->face(face_n)->vertex(vertex) << ")";                   // 
//                     }
//                 std::cout  << ", the boundary id is 0 " << std::endl;
                
                cell->face(face_n)->get_dof_indices (local_face_dof_indices);
                
//                 std::cout << "      dof indices on face " << face_n << " is: ";
//                 for (unsigned int i=0; i<dofs_per_face; ++i)
//                 {
//                     std::cout << local_face_dof_indices[i] << ", ";
//                 }
//                 std::cout << std::endl;
                
                
              fe_face_values.reinit (cell, face_n);
              
//               std::cout << "      coords of the quadrature point and the value of normal vectors: " << std::endl;
//                 for(unsigned int q=0; q<n_face_q_points; ++q)
//                 {
//                     std::cout << "      " << fe_face_values.get_quadrature_points()[q] << ", " << fe_face_values.normal_vector(q) << "\n";
//                 }
                
//                 for(unsigned int q=0; q<n_face_q_points; ++q)
//                 {
//                     std::cout << "    fe_face_values at quad " << q << ": " << fe_face_values.get_quadrature_points()[q] << " is " << std::endl;
//                     for (unsigned int i=0; i<dofs_per_cell; ++i)
//                     {
//                         std::cout << "    dof index: " << local_dof_indices[i] << ", velocity: " << fe_face_values[velocities].value (i, q) << ", pressure: " << fe_face_values[pressure].value (i, q) << "\n";
//                     }
//                     std::cout << std::endl;
//                 }
              
              pressure_boundary_values
              .value_list (fe_face_values.get_quadrature_points(),
                           boundary_values);
              
//               std::cout << "      boundary_values at the quadrature points of the face are: ";
//               for(double n:boundary_values)
//               {
//                   std::cout << n << " ";
//               }
//               std::cout << std::endl;
              
              
//               std::cout << "      indices of dofs that contribute to the boundary integral are: " << std::endl;
//               for (unsigned int q=0; q<n_face_q_points; ++q)
//               {
//                 std::cout << "      quad " << q << ": \n";
//                 for (unsigned int i=0; i<dofs_per_cell; ++i)
//                 {
//                     if(fe_face_values[velocities].value (i, q) * fe_face_values.normal_vector(q) != 0)          // only show dofs that contributes to the rhs
//                     {
//                         std::cout << "      dof index: " << local_dof_indices[i] << ", velocity: " << fe_face_values[velocities].value (i, q) << ", pressure: " << fe_face_values[pressure].value (i, q) << "\n";
//                     }
//                 }
//                 std::cout << std::endl;
//               }
              
              
              for (unsigned int q=0; q<n_face_q_points; ++q)
              {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  local_rhs(i) += -(fe_face_values[velocities].value (i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values[q] *
                                    fe_face_values.JxW(q));
                }
              }
            }
        }
            
//         std::cout << "    accessed here 2" <<  std::endl;

//     if(cell->active_cell_index()==0)
//     {
//         std::cout << "    iterate cell to output dof indices" << std::endl;
//         cell->get_dof_indices (local_dof_indices);
//         
//         for (unsigned int i = 0; i < dofs_per_cell; i++)
//         {   
//             std::cout << "    i is: " << i << ", local_dof_indices[i] is: " << local_dof_indices[i] << std::endl;               
//         }
//     }

//         std::cout << "    the local_matrix reads" <<  std::endl;
//         local_matrix.print_formatted(std::cout);
//         
//         std::cout << "    the local_rhs reads" <<  std::endl;
//         local_rhs.print(std::cout);
        
        cell->get_dof_indices (local_dof_indices);
        
//         std::cout << "    local_dof_indices reads ";
//         for (unsigned int i=0; i<dofs_per_cell; ++i)
//         {
//             std::cout << local_dof_indices[i] << " ";
//         }
//         std::cout << std::endl;  
        
        
/*        std::cout << "system_matrix before assembling is " << std::endl;
        system_matrix.print_formatted(std::cout);
        std::cout << '\n';  */   


        if(is_constraints==1)
        {
//             std::cout << "    using constraints for assembling and dealing with Neumann BC" << std::endl;
        
            constraints.condense (system_matrix, system_rhs);
            
            constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                    local_dof_indices,
                                                    system_matrix, system_rhs);

        }else if (is_constraints==0)
        {
//             std::cout << "    normal assembling" << std::endl;
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    system_matrix.add (local_dof_indices[i],
                                    local_dof_indices[j],
                                    local_matrix(i,j));
                }
            }
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                system_rhs(local_dof_indices[i]) += local_rhs(i);
            }
        }
        
      }
        
//     std::cout << "system_matrix after applying BC is " << std::endl;
//     system_matrix.print_formatted(std::cout);
//     std::cout << '\n';
    // 
/*    std::cout << "system_rhs after applying BC is " << std::endl;
    system_rhs.print(std::cout); */     
      
      
      
//   std::ofstream output("TwoD_Pois_MM_system_matrix_"+std::to_string(degree)+"_"+std::to_string(refine)+".txt");                                           // store system_matrix to a file
//   for (unsigned int i=0;i<dof_handler.n_dofs();i++)
//   {
//     for (unsigned int j=0;j<dof_handler.n_dofs();j++)
//     {
//         output << system_matrix.el(i, j) << " "; // behaves like cout - cout is also a stream
//     }
//         output << "\n";
//   } 
//   output.close();
//   output.clear();
//   
//   std::ofstream output_rhs("TwoD_Pois_MM_system_rhs_"+std::to_string(degree)+"_"+std::to_string(refine)+".txt");                                           // store system_rhs to a file  
//   
//   for (unsigned int i=0;i<dof_handler.n_dofs();i++)
//   {
//       output_rhs << system_rhs[i] << " ";               // behaves like cout - cout is also a stream
//       output_rhs << "\n";
//   } 
// 
//   output_rhs.close();
//   output_rhs.clear();
      
  }


  // @sect3{Linear solvers and preconditioners}

  template <class MatrixType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType &m);

    void vmult(Vector<double>       &dst,
               const Vector<double> &src) const;

  private:
    const SmartPointer<const MatrixType> matrix;
  };


  template <class MatrixType>
  InverseMatrix<MatrixType>::InverseMatrix (const MatrixType &m)
    :
    matrix (&m)
  {}


  template <class MatrixType>
  void InverseMatrix<MatrixType>::vmult (Vector<double>       &dst,
                                         const Vector<double> &src) const
  {

    SolverControl solver_control (std::max(src.size(), static_cast<std::size_t> (200)),
                                  1e-16*src.l2_norm());
    SolverCG<>    cg (solver_control);

    dst = 0;

    cg.solve (*matrix, dst, src, PreconditionIdentity());
  }


  // @sect4{The <code>SchurComplement</code> class}

  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement (const BlockSparseMatrix<double>            &A,
                     const InverseMatrix<SparseMatrix<double> > &Minv);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
    const SmartPointer<const InverseMatrix<SparseMatrix<double> > > m_inverse;

    mutable Vector<double> tmp1, tmp2;
  };


  SchurComplement
  ::SchurComplement (const BlockSparseMatrix<double>            &A,
                     const InverseMatrix<SparseMatrix<double> > &Minv)
    :
    system_matrix (&A),
    m_inverse (&Minv),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void SchurComplement::vmult (Vector<double>       &dst,
                               const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    m_inverse->vmult (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }


  // @sect4{The <code>ApproximateSchurComplement</code> class}

  class ApproximateSchurComplement : public Subscriptor
  {
  public:
    ApproximateSchurComplement (const BlockSparseMatrix<double> &A);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;

    mutable Vector<double> tmp1, tmp2;
  };



  ApproximateSchurComplement::ApproximateSchurComplement
  (const BlockSparseMatrix<double> &A) :
    system_matrix (&A),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void
  ApproximateSchurComplement::vmult
  (Vector<double>       &dst,
   const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }

  // @sect4{MixedLaplace::solve}

  template <int dim>
  void MixedLaplaceProblem<dim>::solve ()
  {
      std::cout << "Solving... " << std::endl;
      TimerOutput::Scope t(computing_timer, "solve");
      
      if(id_scaling_scheme==1)
      {
          
          std::cout << "    scheme M1" << ", u_L2_inte: " << u_L2_inte << ", up_L2_inte: " << up_L2_inte << std::endl;
          system_matrix.block(0,1)*=(u_L2_inte/up_L2_inte);               // we scale B here
          system_rhs/=up_L2_inte;
          
      } else if(id_scaling_scheme==2)
      {
          
          std::cout << "    scheme M2" << ", u_L2_inte: " << u_L2_inte << std::endl;
          system_rhs/=u_L2_inte;

      } else
      {
          std::cout << "    no scaling scheme used" << std::endl;
      }
      
//       std::cout << "system_rhs: " << std::endl;
//       system_rhs.print(std::cout);
        
      
// ========================= reduce the precision of matrices from double to float =========================

/*    for (unsigned int i=0;i<dof_handler.n_dofs();i++)                 
    {
        system_rhs[i]=(float)system_rhs[i];
        
        for (unsigned int j=0;j<dof_handler.n_dofs();j++)
        {
            system_matrix.set(i,j, (float) system_matrix.el(i,j));
        }
    }     */ 
      
    

        if(is_constraints==1)                      // the difference with before is the arguments of ExactSolution, i.e. we need the exact solution after being scaled when performing distribution, Jan 4, 2020
        {
            constraints.clear ();
            FEValuesExtractors::Vector velocities(0);                   
            DoFTools::make_hanging_node_constraints (dof_handler,
                                                    constraints);

            VectorTools::interpolate_boundary_values (dof_handler,
                                                    1,
                                                    ExactSolution<dim>(id_case, coeff_inner_x, id_scaling_scheme, u_L2_inte, up_L2_inte),
                                                    constraints,
                                                    fe.component_mask(velocities));
            constraints.close ();
        }
    
    
        if (is_UMFPACK==1)
        {
            cout << "    UMFPACK for monolithic" << endl;
            
            SparseDirectUMFPACK  A_direct;		                      //SparseDirectUMFPACK
            A_direct.initialize(system_matrix);
            A_direct.vmult (solution, system_rhs);
        
            
            if(is_constraints==1)
            {
                constraints.distribute (solution);
            }
            
        } else if (is_UMFPACK == 0)
        {

            cout << "    Schur complement" << endl;
        
            InverseMatrix<SparseMatrix<double> > inverse_mass (system_matrix.block(0,0));
            Vector<double> tmp (solution.block(0).size());
            {
                SchurComplement schur_complement (system_matrix, inverse_mass);
                Vector<double> schur_rhs (solution.block(1).size());
                inverse_mass.vmult (tmp, system_rhs.block(0));
                system_matrix.block(1,0).vmult (schur_rhs, tmp);
                schur_rhs -= system_rhs.block(1);
                SolverControl solver_control (solution.block(1).size(),
                                                1e-16*schur_rhs.l2_norm());
                SolverCG<> cg (solver_control);
                ApproximateSchurComplement approximate_schur (system_matrix);
                InverseMatrix<ApproximateSchurComplement> approximate_inverse
                (approximate_schur);
                cg.solve (schur_complement, solution.block(1), schur_rhs,
                            approximate_inverse);
                
                constraints.distribute (solution);
                
                CG_iteration_schur = solver_control.last_step();
                
                std::cout << solver_control.last_step()
                            << " CG Schur complement iterations to obtain convergence."
                            << std::endl;
            }
            {
                system_matrix.block(0,1).vmult (tmp, solution.block(1));
                tmp *= -1;
                tmp += system_rhs.block(0);
                inverse_mass.vmult (solution.block(0), tmp);
                
                constraints.distribute (solution);
            
            }
        }
          
    std::cout << "solution reads " << std::endl;
    solution.print(std::cout);
    std::cout << std::endl;

  }


  // @sect3{MixedLaplaceProblem class implementation (continued)}

  // @sect4{MixedLaplace::compute_errors}

  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors ()
  {
    std::cout << "Computing errors... " << std::endl; 
    TimerOutput::Scope t(computing_timer, "compute_errors");
//     std::cout << std::endl;
    
    const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
    
    const ComponentSelectFunction<dim>
    velocity_mask(std::make_pair(0, dim), dim+1);             // for u
    
    const ComponentSelectFunction<dim>
    my_velocity_mask_1(0, 1, dim+1);                          // for u_x
    
    const ComponentSelectFunction<dim>
    my_velocity_mask_2(1, 1, dim+1);                          // for u_y


    ExactSolution<dim> exact_solution(id_case, coeff_inner_x, id_scaling_scheme, u_L2_inte, up_L2_inte);
    Vector<double> cellwise_errors (triangulation.n_active_cells());

    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature (q_trapez, degree+2);
    
//     double p_H1_error = 0;
//     double u_H1_error = 0;

    // With this, we can then let the library compute the errors and output
    // them to the screen:

    std::cout << "    solution" << std::endl;
    
    VectorTools::integrate_difference (dof_handler, solution_0, exact_solution,                             // L2_norm for the pressure
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    
    inte_p = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);    

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    p_L2 = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);
    r_p_L2 = p_L2/inte_p;
    
    
    
    std::cout << "    first derivative" << std::endl;
    
    VectorTools::integrate_difference (dof_handler, solution_0, exact_solution,                             // L2_norm for the velocity
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    inte_u = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);
    
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    u_L2 = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);
    r_u_L2 = u_L2/inte_u;
    
    
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &my_velocity_mask_1);
    u_x_L2 = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);    
    
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &my_velocity_mask_2);
    u_y_L2 = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);     
    
    
    std::cout << "    second derivative" << std::endl;
    
    VectorTools::integrate_difference (dof_handler, solution_0, exact_solution,                             // H1_seminorm for the velocity
                                       cellwise_errors, quadrature,
                                       VectorTools::H1_seminorm,
                                       &velocity_mask);
    inte_du = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::H1_seminorm);
    
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::H1_seminorm,
                                       &velocity_mask);
    du_L2 = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::H1_seminorm);   
    r_du_L2 = du_L2/inte_du;    
    

//     VectorTools::integrate_difference (dof_handler, solution, exact_solution,
//                                        cellwise_errors, quadrature,
//                                        VectorTools::H1_seminorm,
//                                        &pressure_mask);
//     p_H1_error = VectorTools::compute_global_error(triangulation,
//                                                                 cellwise_errors,
//                                                                 VectorTools::H1_seminorm);
// 
//     VectorTools::integrate_difference (dof_handler, solution, exact_solution,
//                                        cellwise_errors, quadrature,
//                                        VectorTools::H1_norm,
//                                        &velocity_mask);
//     u_H1_error = VectorTools::compute_global_error(triangulation,
//                                                                 cellwise_errors,
//                                                                 VectorTools::H1_norm);

  }
  
  template <int dim>
  void MixedLaplaceProblem<dim>::print_errors_and_cpu_time () const
  {
      
      std::cout << "print errors and cpu time" << std::endl;
      
    std::cout << "    inte_p = " << inte_p
              << ", inte_u = " << inte_u
              << ", inte_du = " << inte_du << std::endl;
    
    std::cout << "    p_L2 = " << p_L2
              << ", r_p_L2 = " << r_p_L2 << std::endl
              << "    u_L2 = " << u_L2
              << ", r_u_L2 = " << r_u_L2 // << std::endl
              << ", u_x_L2 = " << u_x_L2
              << ", u_y_L2 = " << u_y_L2 << endl
              << "    du_L2 = " << du_L2
              << ", r_du_L2 = " << r_du_L2 << std::endl;
              
//               << "||e_u||_H1_semi = " << u_H1_semi << std::endl;
// 			  << "||e_p||_H1 = " << p_H1_error
// 			  << ",   ||e_u||_H1 = " << u_H1_error
//               << std::endl;      
              
    std::cout << "    my_CPU_time is " << my_CPU_time << std::endl;  
                  
  }


  // @sect4{MixedLaplace::output_results}

  template <int dim>
  void MixedLaplaceProblem<dim>::output_results () const
  {
      std::cout << "output results to vtk" << std::endl;
      
    std::vector<std::string> solution_names(1,"u_x");
    solution_names.push_back ("u_y");
    solution_names.push_back ("p_custom");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (1,DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector (dof_handler, solution, solution_names, interpretation);

    data_out.build_patches (degree+1);

    std::ofstream output ("solution_TwoD_uxm0p5squared_MM_PPdisc.vtk");
    data_out.write_vtk (output);
  }



  // @sect4{MixedLaplace::run}

  // This is the final function of our main class. It's only job is to call
  // the other functions in their natural order:
  template <int dim>
  void MixedLaplaceProblem<dim>::run ()
  {
    make_grid_and_dofs();
    print_mesh_info(triangulation);
    assemble_system ();
    solve ();
    compute_errors ();
    output_results ();
    
    my_CPU_time += computing_timer.return_total_cpu_time ();

    print_errors_and_cpu_time ();
    printToFile();
    
    computing_timer.print_summary ();                     
    
  }
}


// @sect3{The <code>main</code> function}

int main (int argc, char *argv[])
{
    unsigned int id_case;
    double coeff_inner_x;
    unsigned int id_scaling_scheme;
    double u_L2_inte, up_L2_inte = 1.0;                              // define variables in 'main' function
	unsigned int degree = 1;
	unsigned int refine = 1;

	if ( argc != 8 ) {
		std::cout<<"usage: "<< argv[0] <<" <id_case> <coeff_inner_x> <id_scaling_scheme> <u_L2_inte> <up_L2_inte> <degree> <refinement>\n";
		exit(EXIT_FAILURE);
	} else {
        id_case = atoi(argv[1]);
        coeff_inner_x = atof(argv[2]);
        id_scaling_scheme = atoi(argv[3]);
        u_L2_inte = atof(argv[4]);
        up_L2_inte = atof(argv[5]);
		degree = atoi(argv[6]);
		refine = atoi(argv[7]);
	}
	
  std::cout << "================== \n" 
            << "  id_case: " << id_case << "\n"
//             << "  id_quad_assem_incre: " << id_quad_assem_incre << "\n"
            << "  coeff_inner_x: " << coeff_inner_x << "\n"
//             << "  tol_prm_schur: " << tol_prm_schur << "\n"
            << "  id_scaling_scheme: " << id_scaling_scheme << "\n"
            << "  u_L2_inte: " << u_L2_inte << "\n"
            << "  up_L2_inte: " << up_L2_inte << "\n"
            << "  degree is " << degree << " (P" << degree << "/P" << degree-1 << "^disc)" << "\n"
            << "  refine is " << refine << "\n"
            << "==================" 
            << std::endl;
  try
    {
      using namespace dealii;
      using namespace Step20;

      MixedLaplaceProblem<2> mixed_laplace_problem(id_case, coeff_inner_x, id_scaling_scheme, u_L2_inte, up_L2_inte, degree, refine);           // pass variables in main function to an object of class mixed_laplace_problem
      mixed_laplace_problem.run ();
//       std::cout << std::endl; 
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
