
#ifndef EXACT_SOLUTION_STEP_4_COMPLEX_H
#define EXACT_SOLUTION_STEP_4_COMPLEX_H


template <int dim>
class ExactSolution_Step_4_Complex_Both :
public Coeff_Variable_Complex<dim>
{
public:
  ExactSolution_Step_4_Complex_Both (const unsigned int id_case);
  virtual complex<double> value_both (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,dim,complex<double>> gradient_both (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim,complex<double>> hessian_both (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;     
protected:
  const unsigned int id_case;
};


template <int dim>
class ExactSolution_Step_4_Complex_Amp :
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>
{
public:
  ExactSolution_Step_4_Complex_Amp (const unsigned int id_case);
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
class ExactSolution_Step_4_Complex_Real :
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>
{
public:
  ExactSolution_Step_4_Complex_Real (const unsigned int id_case);
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim> hessian (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
};

template <int dim>
class ExactSolution_Step_4_Complex_Imag :
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>
{
public:
  ExactSolution_Step_4_Complex_Imag (const unsigned int id_case);
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim> hessian (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;         
};


template <int dim>
ExactSolution_Step_4_Complex_Both<dim>::ExactSolution_Step_4_Complex_Both(const unsigned int id_case):
Coeff_Variable_Complex<dim>(id_case),
id_case(id_case)
{}

template <int dim>
ExactSolution_Step_4_Complex_Amp<dim>::ExactSolution_Step_4_Complex_Amp(const unsigned int id_case):
Function<dim>(1),
ExactSolution_Step_4_Complex_Both<dim>(id_case)
{}

template <int dim>
ExactSolution_Step_4_Complex_Real<dim>::ExactSolution_Step_4_Complex_Real(const unsigned int id_case):
Function<dim>(1),
ExactSolution_Step_4_Complex_Both<dim>(id_case)
{}

template <int dim>
ExactSolution_Step_4_Complex_Imag<dim>::ExactSolution_Step_4_Complex_Imag(const unsigned int id_case):
Function<dim>(1),
ExactSolution_Step_4_Complex_Both<dim>(id_case)
{}


template <int dim>
complex<double> ExactSolution_Step_4_Complex_Both<dim> :: value_both (const Point<dim>   &p,
                                                        const unsigned int ) const
{
  complex<double> return_value;
  switch(id_case)
  {
    case 1:
//       cout << "  exact solution does not exist\n";
      break;
    case 2:
      if (dim==1)
      {
        return_value = this->A*std::exp(this->r1*p[0])+this->B*std::exp(this->r2*p[0]);       
      }else if(dim==2)
      {        
        return_value = (this->A*std::exp(this->r1*p[0])+this->B*std::exp(this->r2*p[0]))*cos(this->coeff_y*(p[1]-this->centre_y));
      }
      break;
    case 61:
      for(unsigned int i=0; i<dim; ++i)
      {
        return_value += pow(p[i]-0.5, 2.0);
      }
      break;
    default:
      cout << "  case does not exist\n";
      throw exception();    
  }
  return return_value;
}

template <int dim>
Tensor<1,dim,complex<double>> ExactSolution_Step_4_Complex_Both<dim>::gradient_both (const Point<dim>   &p,
                                                                const unsigned int) const
{
  Tensor<1,dim,complex<double>> return_value;
  switch(id_case)
  {
    case 1:
//       cout << "  exact solution does not exist\n";
      break;
    case 2:
      if (dim==1)
      {
        return_value[0] = this->A*this->r1*std::exp(this->r1*p[0])+this->B*this->r2*std::exp(this->r2*p[0]); 
      }else if(dim==2)
      {         
        return_value[0] = (this->A*this->r1*std::exp(this->r1*p[0])+this->B*this->r2*std::exp(this->r2*p[0]))*cos(this->coeff_y*(p[1]-this->centre_y));
        return_value[1] = (this->A*std::exp(this->r1*p[0])+this->B*std::exp(this->r2*p[0]))*(-this->coeff_y*sin(this->coeff_y*(p[1]-this->centre_y)));
      }
      break;
    case 61:
      for(unsigned int i=0; i<dim; ++i)
      {
        return_value[i] = 2.0 * (p[i]-0.5);
      }
      break;
    default:
      cout << "  case does not exist\n";
      throw exception();      
  }
  return return_value;
}

template <int dim>
SymmetricTensor<2,dim,complex<double>> ExactSolution_Step_4_Complex_Both<dim>::hessian_both (const Point<dim>   &p,
                                                                        const unsigned int) const
{
  SymmetricTensor<2,dim,complex<double>> return_value;
  switch(id_case)
  {
    case 1:
//       cout << "  exact solution does not exist\n";
      break;
    case 2:
      if (dim==1)
      {
        return_value[0][0] = this->A*pow(this->r1,2.0)*std::exp(this->r1*p[0])+this->B*pow(this->r2,2.0)*std::exp(this->r2*p[0]);   
      }else if(dim==2)
      {        
        return_value[0][0] = (this->A*pow(this->r1,2.0)*std::exp(this->r1*p[0])+this->B*pow(this->r2,2.0)*std::exp(this->r2*p[0]))*cos(this->coeff_y*(p[1]-this->centre_y)); // p_xx
        return_value[0][1] = (this->A*this->r1*std::exp(this->r1*p[0])+this->B*this->r2*std::exp(this->r2*p[0]))*(-this->coeff_y*sin(this->coeff_y*(p[1]-this->centre_y)));  // p_xy
        return_value[1][0] = (this->A*this->r1*std::exp(this->r1*p[0])+this->B*this->r2*std::exp(this->r2*p[0]))*(-this->coeff_y*sin(this->coeff_y*(p[1]-this->centre_y)));  // p_yx
        return_value[1][1] = (this->A*std::exp(this->r1*p[0])+this->B*std::exp(this->r2*p[0]))*(-pow(this->coeff_y,2.0)*cos(this->coeff_y*(p[1]-this->centre_y)));  // p_yy
      }
      break;
    case 61:
      for(unsigned int i=0; i<dim; ++i)
      {
        return_value[i][i] = 2.0;
      }
      break;
    default:
      cout << "  case does not exist\n";
      throw exception();      
  }
  return return_value;
}


template <int dim>
double ExactSolution_Step_4_Complex_Amp<dim> :: value (const Point<dim>   &p,
                                                   const unsigned int ) const
{
  return std::abs(ExactSolution_Step_4_Complex_Both<dim>::value_both(p));
}


template <int dim>
double ExactSolution_Step_4_Complex_Real<dim> :: value (const Point<dim>   &p,
                                                        const unsigned int ) const
{
  return ExactSolution_Step_4_Complex_Both<dim>::value_both(p).real();
}

template <int dim>
Tensor<1,dim> ExactSolution_Step_4_Complex_Real<dim>::gradient (const Point<dim>   &p,
                                                                const unsigned int) const
{
  Tensor<1,dim> return_value;
  for(unsigned int i=0; i<dim; ++i)
  {
    return_value[i] = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p)[i].real();
  }
  return return_value;
}

template <int dim>
SymmetricTensor<2,dim> ExactSolution_Step_4_Complex_Real<dim>::hessian (const Point<dim>   &p,
                                                                        const unsigned int) const
{
  SymmetricTensor<2,dim> return_value;
  for(unsigned int i=0; i<dim; ++i)
  {
    for(unsigned int j=0; j<dim; ++j)
    {
      return_value[i][j] = ExactSolution_Step_4_Complex_Both<dim>::hessian_both(p)[i][j].real();
    }
  }
  return return_value;
}

template <int dim>
double ExactSolution_Step_4_Complex_Imag<dim> :: value (const Point<dim>   &p,
                                                        const unsigned int ) const
{
  return (ExactSolution_Step_4_Complex_Both<dim>::value_both(p)).imag();
}

template <int dim>
Tensor<1,dim> ExactSolution_Step_4_Complex_Imag<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
  Tensor<1,dim> return_value;
  for(unsigned int i=0; i<dim; ++i)
  {
    return_value[i] = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p)[i].imag();
  }
  return return_value;
}

template <int dim>
SymmetricTensor<2,dim> ExactSolution_Step_4_Complex_Imag<dim>::hessian (const Point<dim>   &p,
                                       const unsigned int) const
{
  SymmetricTensor<2,dim> return_value;
  for(unsigned int i=0; i<dim; ++i)
  {
    for(unsigned int j=0; j<dim; ++j)
    {
      return_value[i][j] = ExactSolution_Step_4_Complex_Both<dim>::hessian_both(p)[i][j].imag();
    }
  }
  return return_value;
}

#endif
