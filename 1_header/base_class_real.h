
#ifndef BASE_CLASS_REAL_H
#define BASE_CLASS_REAL_H


template <int dim>
class Coeff_Variable_Real                                                   // we put it here because we use it both in step-4 and step-20
{
  public:
    Coeff_Variable_Real(const unsigned int id_case, 
                        const double coeff_var_inner_x);
    
    double coeff_outer_x, coeff_outer_y = 0;
    double center_x, center_y = 0;
    double coeff_var_inner_y = 0;
    double coeff_var_inde = 0;
    
  protected:
    const double coeff_var_inner_x = 0.0;
};


template <int dim>
class Coeff_Diff_Real
{
  public:
    Coeff_Diff_Real (unsigned int id_case, 
                     const unsigned int id_coeff_value, 
                     const double coeff_diff_inner);

    void value (const vector<Point<dim>> &p,
                vector<Tensor<2,dim>>   &values) const;
    void inverse (const vector<Point<dim>> &p,
                  vector<Tensor<2,dim>>   &values) const;
    void gradient (const vector<Point<dim>> &p,
                   vector<Tensor<2,dim,Tensor<1,dim>>>   &values) const;       
    void gradient_x_direction (const vector<Point<dim>> &p,
                               vector<Tensor<2,dim>>   &values) const;       
    void gradient_y_direction (const vector<Point<dim>> &p,
                               vector<Tensor<2,dim>>   &values) const;        
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_diff_value;         // for investigating the influence of the diffusion coefficient on the error when applying the mixed fem on the 2d problems
    const double coeff_diff_inner;
};


template <int dim>
class Coeff_First_Derivative_A_Real
{
  public:
    Coeff_First_Derivative_A_Real (unsigned int id_case, 
                                   const unsigned int coeff_a_value);

    void value (const vector<Point<dim>> &p,
                vector<Tensor<1,dim>>   &values) const;              
  protected:
    const unsigned int id_case;
    const unsigned int coeff_a_value;
};


template <int dim>
class Coeff_Helm_Real
{
  public:
    Coeff_Helm_Real (unsigned int id_case,
                     const int id_coeff_helm_value,
                     const double coeff_helm_inner);            

    void value (const vector<Point<dim>> &p,
                vector<double>   &values) const;
           
  protected:
    const unsigned int id_case;
    const int id_coeff_helm_value;
    const double coeff_helm_inner;                                        // only 1 considered
};


template <int dim>
Coeff_Variable_Real<dim>::Coeff_Variable_Real(const unsigned int id_case, 
                                              const double coeff_var_inner_x):
coeff_var_inner_x(coeff_var_inner_x)
{
  
  switch (id_case)
  {                                           // Poisson equations
    case 1:
      break;
    case 2:                                   // including benchmark Poisson equation
//       coeff_outer_x = 1.0;
//       coeff_outer_y = 1.0;
      coeff_var_inner_y = 1.0;
      center_x = 0.5;
      center_y = 0.5;
      coeff_var_inde = 0.0;              
      break;
    case 3:
    case 4:
    case 5:
      break;
    case 61:                                  // p = a*((x-x_0)/c1)^2+b*((y-y_0)/c2)^2
      coeff_outer_x = 1.0;
      coeff_outer_y = 1.0;
      coeff_var_inner_y = 1.0*coeff_var_inner_x;
      center_x = 0.5;
      center_y = 0.5;
      coeff_var_inde = 0.0;
      break;
    case 62:                                  // p = (cx+100)^2
    case 63:                                  // p = x^2-c^2
    case 64:                                  // p = x^2+y^2-c^2
      break;
    
        
    case 21:                                  // benchmark diffusion, i.e. p=sin(2pix), MBC; extended to d=1+cx
      break;
    default:
      break;
  }       
}


template <int dim>
Coeff_Diff_Real<dim>::Coeff_Diff_Real (unsigned int id_case, 
                                       const unsigned int id_coeff_value, 
                                       const double coeff_diff_inner):
id_case(id_case),
id_coeff_diff_value(id_coeff_value),
coeff_diff_inner(coeff_diff_inner)
{}

template <int dim>
Coeff_First_Derivative_A_Real<dim>::Coeff_First_Derivative_A_Real (unsigned int id_case, 
                                                                   const unsigned int coeff_a_value):
id_case(id_case),
coeff_a_value(coeff_a_value)
{}


template <int dim>
Coeff_Helm_Real<dim>::Coeff_Helm_Real (unsigned int id_case,
                                       const int id_coeff_helm_value,
                                       const double coeff_helm_inner):
id_case(id_case),
id_coeff_helm_value(id_coeff_helm_value),
coeff_helm_inner(coeff_helm_inner)
{}


template <int dim>
void
Coeff_Diff_Real<dim>::value (const vector<Point<dim>> &p,
                             vector<Tensor<2,dim>> &values) const
{                                                                                   
                                                                                    // coefficient d also appears in 1. RightHandSide_Real
                                                                                    //                               2. GradientBoundary_Step_20_Real
                                                                                    //                               3. ExactSolution_Step_20_Real
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch (id_coeff_diff_value)
    {
      case 1:
        if(dim==1)
        {
          values[i][0][0] = coeff_diff_inner;  //    + 1.0*p[i][0] - 1.0*p[i][0]
        }else if(dim==2)
        {
          values[i][0][0] = coeff_diff_inner;
          values[i][0][1] = 0.0;
          values[i][1][0] = 0.0;
          values[i][1][1] = values[i][0][0];
        }        
        break;               
      case 2:
        if(dim==1)
        {            
            values[i][0][0] = 1.0 + coeff_diff_inner*p[i][0];          //  + pow(p[i][0],2.0)
        }else if(dim==2)
        {
            values[i][0][0] = 1.0 + coeff_diff_inner * (p[i][0] + p[i][1]);
            values[i][0][1] = p[i][0]*p[i][1];
            values[i][1][0] = values[i][0][1];
            values[i][1][1] = 1.0 + coeff_diff_inner * (p[i][0] + p[i][1]);
        }
        break;
      case 3:
        if(dim==1)
        {
          values[i][0][0] = 1.0+pow(p[i][0]-0.5,2.0);
        }else if(dim==2)
        {
          values[i][0][0] = 1.0+pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0);
          values[i][0][1] = 0.0;
          values[i][1][0] = values[i][0][1];
          values[i][1][1] = 1.0+pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0);  
        }
        break;
      case 4:
        if(dim==1)
        {
          values[i][0][0] = exp(- coeff_diff_inner * pow(p[i][0]-0.5, 2.0));
        }else if(dim==2)
        {
          values[i][0][0] = exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)));                    // cos(pi*(p[i][0]+p[i][1]))
          values[i][0][1] = 0.0;                                          // exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)))
          values[i][1][0] = values[i][0][1];
          values[i][1][1] = exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)));
        }
        break;
      case 5:
        for (unsigned int k=0; k<dim; ++k)
        {
//           values[i][k][k] = 1.0 + 0.5*sin(coeff_diff_inner* p[i][0]);
          values[i][k][k] = 0.5 + pow(cos(p[i][0]),2.0);
        }
        break; 
    }
  }
  
//   cout << "values:\n";
//   print_vector(values);
  
}

template <int dim>
void
Coeff_Diff_Real<dim>::gradient (const vector<Point<dim>> &p,
                                vector<Tensor<2,dim,Tensor<1,dim>>> &values) const
{
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch (id_coeff_diff_value)
    {
      case 1:
        for (unsigned int k=0; k<dim; ++k)
        {
          values[i][k][k] = 0.0;
        }
        break;               
      case 2:
        if(dim==1)
        {
          values[i][0][0][0] = coeff_diff_inner;         //  + 2.0*p[i][0]
        }else if(dim==2)
        {
          values[i][0][0][0] = coeff_diff_inner;                // gradient of D in the x direction
          values[i][0][1][0] = p[i][1];
          values[i][1][0][0] = values[i][0][1][0];
          values[i][1][1][0] = coeff_diff_inner;   
        
          values[i][0][0][1] = coeff_diff_inner;                // gradient of D in the y direction
          values[i][0][1][1] = p[i][0];
          values[i][1][0][1] = values[i][0][1][1];
          values[i][1][1][1] = coeff_diff_inner;                  
        }
        break;
      case 3:
        values[i][0][0][0] = 2.0*(p[i][0]-0.5);
        if(dim==2)
        {            
          values[i][0][1][0] = 0.0;
          values[i][1][0][0] = values[i][0][1][0];
          values[i][1][1][0] = 2.0*(p[i][0]-0.5);
        
          values[i][0][0][1] = 2.0*(p[i][1]-0.5);          
          values[i][0][1][1] = 0.0;
          values[i][1][0][1] = values[i][0][1][1];
          values[i][1][1][1] = 2.0*(p[i][1]-0.5);              
        }
        break;
      case 4:
        if(dim==1)
        {
          values[i][0][0][0] = exp(- coeff_diff_inner * pow(p[i][0]-0.5,2.0))*(- coeff_diff_inner * 2.0*(p[i][0]-0.5));
        }else if(dim==2)
        {            
          values[i][0][0][0] = exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)))*(-2.0*(p[i][0]-0.5));
          values[i][0][1][0] = 0.0;                                          // exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)))*(-2.0*(p[i][0]-0.5))
          values[i][1][0][0] = values[i][0][1][0];
          values[i][1][1][0] = exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)))*(-2.0*(p[i][0]-0.5));
        
          values[i][0][0][1] = exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)))*(-2.0*(p[i][1]-0.5));
          values[i][0][1][1] = 0.0;
          values[i][1][0][1] = values[i][0][1][1];
          values[i][1][1][1] = exp(-(pow(p[i][0]-0.5,2.0)+pow(p[i][1]-0.5,2.0)))*(-2.0*(p[i][1]-0.5));              
        }
        break;
      case 5:
        for (unsigned int k=0; k<dim; ++k)
        {
//           values[i][k][k][0] = 0.5*coeff_diff_inner*cos(coeff_diff_inner* p[i][0]);
          values[i][k][k][0] = 2.0*cos(p[i][0])*(-sin(p[i][0]));
        }
        break;
      default:
        break;
    }
  }
  
//   cout << "values:\n";
//   print_vector(values);  
  
}

template <int dim>
void
Coeff_Diff_Real<dim>::gradient_x_direction (const vector<Point<dim>> &p,
                                            vector<Tensor<2,dim>> &values) const
{
  vector<Tensor<2,dim,Tensor<1,dim>>> gradient_d(values.size(), Tensor<2,dim,Tensor<1,dim>>());
  
//   cout << "gradient_d:\n";
//   print_vector(gradient_d);
  
  Coeff_Diff_Real<dim>::gradient(p,gradient_d);
  
  for(unsigned int i=0; i<values.size(); ++i)
  {
    for (unsigned int j=0; j<dim; ++j)
    {
      for (unsigned int k=0; k<dim; ++k)
      {
        values[i][j][k] = gradient_d[i][j][k][0];    
      }
    }
  }
}

template <int dim>
void
Coeff_Diff_Real<dim>::gradient_y_direction (const vector<Point<dim>> &p,
                                            vector<Tensor<2,dim>> &values) const
{
  vector<Tensor<2,dim,Tensor<1,dim>>> gradient_d(values.size(), Tensor<2,dim,Tensor<1,dim>>());
  
  Coeff_Diff_Real<dim>::gradient(p,gradient_d);
  
  for(unsigned int i=0; i<values.size(); ++i)
  {
    for (unsigned int j=0; j<dim; ++j)
    {
      for (unsigned int k=0; k<dim; ++k)
      {
        values[i][j][k] = gradient_d[i][j][k][dim-1];                       // the value is equal to gradient in the x directio in 1d by default  
      }                                                                     // not used in 1d
    }
  }
}

template <int dim>
void
Coeff_Diff_Real<dim>::inverse (const vector<Point<dim>> &p,
                             vector<Tensor<2,dim>> &values) const
{
  vector<Tensor<2,dim>> coeff_diff_values (p.size(), Tensor<2,dim>());
  this->value(p,coeff_diff_values);
  
  for(unsigned int i=0; i<values.size(); ++i)
  {
    values[i]=invert(coeff_diff_values[i]);
  }
}

template <int dim>
void
Coeff_First_Derivative_A_Real<dim>::value (const vector<Point<dim>> &/*p*/,                 // coefficient a also appears in 1. RightHandSide_Real
                                           vector<Tensor<1,dim>> &values) const
{                                                                                   

  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch (id_case)
    {
      case 2:
        values[i][0] = 0.0;             // 1.0 + 1.0 * p[i][0];
//         if(dim==2)
//         {
//           values[i][1] = 1.0 + 1.0 * p[i][1];
//         }
        break;
      default:
        values[i][0] = 0.0;
        break;              
    }
  }
  
//   cout << "values:\n";
//   print_vector(values);
  
}


template <int dim>
void
Coeff_Helm_Real<dim>::value (const vector<Point<dim>> &p,
                             vector<double> &values) const
{
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch(id_coeff_helm_value)
    {
      case 1:
        values[i] = coeff_helm_inner;
        break;        
      case 2:
        values[i] = 1.0+coeff_helm_inner*p[i][0];
        break;
      case 3:
        values[i] = exp(-pow(p[i][0]-0.5,2.0));
        break;
      case 4:
        values[i] = 1.0+sin(coeff_helm_inner*p[i][0]);
        break;
      default:
        cout << "a default value for the Helmholtz coefficient chosen\n";
        break;
    }
  }
  
//   cout << "values of the Helmholtz coefficient: ";
//   print_vector_horizontally(values);
}


template <int dim>
class ComputeSurface_Real : public DataPostprocessorScalar<dim>
{
public:
  ComputeSurface_Real ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};


template <int dim>
class ComputeVelocity1_Real: 
public DataPostprocessorScalar<dim>
{
public:
  ComputeVelocity1_Real ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};


template <int dim>
class Compute2ndderivative1_Real:
public DataPostprocessorScalar<dim>
{
public:
  Compute2ndderivative1_Real ();

  virtual
  void
  evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const;
};


template <int dim>
ComputeSurface_Real<dim>::ComputeSurface_Real ()
:
DataPostprocessorScalar<dim> ("",        // solution
                              update_gradients)
{}

template <int dim>
void
ComputeSurface_Real<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
  for (unsigned int i=0; i<inputs.solution_values.size(); i++)
  {
    computed_quantities[i] = inputs.solution_values[i];
  }
}

template <int dim>
ComputeVelocity1_Real<dim>::ComputeVelocity1_Real ()
:
DataPostprocessorScalar<dim> ("",    // gradient
                              update_gradients)
{}

template <int dim>
void
ComputeVelocity1_Real<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
  for (unsigned int i=0; i<inputs.solution_values.size(); i++)
  {
    computed_quantities[i](0) = inputs.solution_gradients[i][0];
  }
}

template <int dim>
Compute2ndderivative1_Real<dim>::Compute2ndderivative1_Real ()
:
DataPostprocessorScalar<dim> ("",    // secondderivative
                              update_hessians)
{}

template <int dim>
void
Compute2ndderivative1_Real<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double> >               &computed_quantities) const
{
  for (unsigned int i=0; i<inputs.solution_values.size(); i++)
  {
    computed_quantities[i](0) = inputs.solution_hessians[i][0][0];
  }
}


#endif
