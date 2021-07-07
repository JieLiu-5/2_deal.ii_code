
#ifndef BASE_CLASS_COMPLEX_H
#define BASE_CLASS_COMPLEX_H

template <int dim>
class Coeff_Variable_Complex
{
  public:
    Coeff_Variable_Complex(const unsigned int id_case);
    
    complex<double> A = 0.1267946481444569561958246595168 - 0.1951839458217654310612232393396i;
    complex<double> B = 0.87320535185554282175957041545189 + 0.1951839458217654310612232393396i; 
    
    complex<double> r1 = 1.0 + 1.0i;
    complex<double> r2 = 0.0 - 1.0i;
    
    double coeff_y = pi;          // pi 0.0
    double centre_y = 0.5;
  protected:
    const unsigned int id_case;
};

template <int dim>
class Coeff_Diff_Complex
{
  public:
    Coeff_Diff_Complex (const unsigned int id_case,
                        const unsigned int id_coeff_diff_value,
                        const double coeff_diff_inner);

    void value (const vector<Point<dim>> &p,
                vector<Tensor<2,dim, complex<double>>>   &values) const;
                
    void value_real (const vector<Point<dim>> &p,
                     vector<Tensor<2,dim, double>>   &values) const;
                
    void inverse (const vector<Point<dim>> &p,
                  vector<Tensor<2,dim, complex<double>>>   &values) const;
    void gradient_x_direction (const vector<Point<dim>> &p,
                   vector<Tensor<2,dim, complex<double>>>   &values) const;              
                
    complex<double> constant_coeff_d = 1.0 + 1.0i;    
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_diff_value;
    const double coeff_diff_inner;
    
};


template <int dim>
class Coeff_Helm_Complex                
{
  public:
    Coeff_Helm_Complex (const unsigned int id_case,
                        const unsigned int id_coeff_helm_value,
                        const double coeff_Helm_inner);            

    void value (const vector<Point<dim>> &p,
                vector<complex<double>>   &values) const;
           
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_helm_value;
    const double coeff_Helm_inner;                                        // only 1 considered
};


template <int dim>
Coeff_Diff_Complex<dim>::Coeff_Diff_Complex (const unsigned int id_case,
                                             const unsigned int id_coeff_diff_value,
                                             const double coeff_diff_inner):
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value),
coeff_diff_inner(coeff_diff_inner)
{}    

template <int dim>
Coeff_Helm_Complex<dim>::Coeff_Helm_Complex (const unsigned int id_case,
                                             const unsigned int id_coeff_helm_value,
                                             const double coeff_Helm_inner):
id_case(id_case),
id_coeff_helm_value(id_coeff_helm_value),
coeff_Helm_inner(coeff_Helm_inner)
{}


template <int dim>
Coeff_Variable_Complex<dim>::Coeff_Variable_Complex(const unsigned int id_case):
id_case(id_case)
{}

template <int dim>
void
Coeff_Diff_Complex<dim>::value (const vector<Point<dim>> &p,                                // d only varies in the x direction by now
                                vector<Tensor<2,dim, complex<double>>> &values) const
{
                                                                                            // coefficient d also appears in 1. class of right-hand side
                                                                                            //                               2. class of gradient boundary
                                                                                            //                               3. class of exact solution for the mixed FEM
  for(unsigned int i=0; i<values.size(); ++i)
  {
    for(unsigned int k=0; k<dim; ++k)
    {
      switch(id_coeff_diff_value)
      {
        case 1:
          values[i][k][k] = coeff_diff_inner * (1.0 + 1.0i);
          break;
        case 2:          
          values[i][k][k] = 1.0 + coeff_diff_inner*p[i][0];
          break;
        case 3:
          values[i][k][k] = -(0.01+p[i][0])*(1.01-p[i][0]);
          break;            
        case 4:
          values[i][k][k] = -exp(-p[i][0]) + -exp(-p[i][0])* 1.0i;
          break;
        default:
          cout << "choice of diffusion coefficient does not exist in Coeff_Diff_Complex<dim>::value()\n";
          throw exception();
      }
    }
  }
  
//   cout << "values of the diffusion coefficient at ";
//   print_vector_horizontally(p);
//   cout << ": ";
//   print_vector_horizontally(values);  
  
}

template <int dim>
void
Coeff_Diff_Complex<dim>::value_real (const vector<Point<dim>> &p,
                                vector<Tensor<2,dim, double>> &values) const
{
  vector<Tensor<2,dim, complex<double>>> vector_value(p.size(),
                                                      Tensor<2,dim, complex<double>>());
  this->value(p, vector_value);
  
  
  for(unsigned int i=0; i<p.size(); ++i)
  {
    for(unsigned int j=0; j<dim; ++j)    
    {
      for(unsigned int k=0; k<dim; ++k)
      {
        values[i][j][k] = vector_value[i][j][k].real();
      }
    }
  }
  
}

template <int dim>
void
Coeff_Diff_Complex<dim>::inverse (const vector<Point<dim>> &p,
                                  vector<Tensor<2,dim, complex<double>>> &values) const
{
  vector<Tensor<2,dim,complex<double>>> coeff_diff_values (p.size(),
                                                           Tensor<2,dim,complex<double>>());
  this->value(p,coeff_diff_values);
  
  for(unsigned int i=0; i<values.size(); ++i)
  {
    values[i]=invert(coeff_diff_values[i]);
    
//     cout << "values[" << i << "]: " << values[i] << ", coeff_diff_values[" << i << "]: " << coeff_diff_values[i] << "\n";
    
  }
}

template <int dim>
void
Coeff_Diff_Complex<dim>::gradient_x_direction (const vector<Point<dim>> &p,                                 // used for deriving the right-hand side
                                   vector<Tensor<2,dim, complex<double>>> &values) const
{
  for(unsigned int i=0; i<values.size(); ++i)
  {
    for(unsigned int k=0; k<dim; ++k)
    {
        
      switch(id_coeff_diff_value)
      {
        case 1:
          values[i][k][k] = 0.0;
          break;
        case 2:          
          values[i][k][k] = coeff_diff_inner;
          break;
        case 3:
          values[i][k][k] = -(1.0 - 2.0*p[i][0]);
          break;            
        case 4:
          values[i][k][k] = exp(-p[i][0]) + exp(-p[i][0])* 1.0i;
          break;
        default:
          cout << "choice of diffusion coefficient does not exist in Coeff_Diff_Complex<dim>::value()\n";
          throw exception();
      }
    }
  }
}

template <int dim>
void
Coeff_Helm_Complex<dim>::value (const vector<Point<dim>> &p,
                                vector<complex<double>> &values) const
{
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch(id_coeff_helm_value)
    {
      case 1:
        values[i] = coeff_Helm_inner *(1.0 + 0.0i);
        break;
      case 2:
        values[i] = (1.0 + coeff_Helm_inner*p[i][0]) * (1.0 + 1.0i);
        break;          
      case 3:
        values[i] = 0.0 - 0.01 * 1.0i;  //              
        break;          
      case 4:
        values[i] = 2.0*exp(-p[i][0]) + 0.0 * 1.0i;
        break;
      default:
        cout << "choice of Helmholtz coefficient does not exist in Coeff_Helm_Complex<dim>::value()\n";
        throw exception();        
    }
  }
  
//   cout << "values of the Helmholtz coefficient at ";
//   print_vector_horizontally(p);
//   cout << ": ";
//   print_vector_horizontally(values);
  
}


#endif
