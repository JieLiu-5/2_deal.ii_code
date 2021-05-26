
#ifndef EXACT_SOLUTION_STEP_4_REAL_H
#define EXACT_SOLUTION_STEP_4_REAL_H

template <int dim>
class ExactSolution_Step_4_Real_Base:                                       // for the inheritance by ExactSolution_Step_20_Real<dim>
public Coeff_Variable_Real<dim>
{
public:
  ExactSolution_Step_4_Real_Base (const unsigned int id_case,
                                  const double coeff_var_inner_x);
  
  virtual double value_base (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient_base (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim> hessian_base (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
protected:
  const unsigned int id_case;
  const double coeff_var_inner_x;
    
};

template <int dim>
class ExactSolution_Step_4_Real:
public Function<dim>,
public ExactSolution_Step_4_Real_Base<dim>
{
public:
  ExactSolution_Step_4_Real (const unsigned int id_case,
                             const double coeff_var_inner_x);
  
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
  virtual SymmetricTensor<2,dim> hessian (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
};


template <int dim>
ExactSolution_Step_4_Real_Base<dim>::ExactSolution_Step_4_Real_Base(const unsigned int id_case,
                                                                    const double coeff_var_inner_x):
Coeff_Variable_Real<dim>(id_case,coeff_var_inner_x),
id_case(id_case),
coeff_var_inner_x(coeff_var_inner_x)
{}

template <int dim>
ExactSolution_Step_4_Real<dim>::ExactSolution_Step_4_Real(const unsigned int id_case,
                                                          const double coeff_var_inner_x):
Function<dim>(1),
ExactSolution_Step_4_Real_Base<dim>(id_case,coeff_var_inner_x)
{}


template <int dim>
double ExactSolution_Step_4_Real_Base<dim> :: value_base (const Point<dim>   &p,
                                                const unsigned int ) const
{
  double return_value=0.0;
  switch (id_case)
  {
//     case 1:
//       return_value = pow(2.0*pi*this->coeff_var_inner_x, -2.0)*sin(2.0*pi*this->coeff_var_inner_x*p[0]);
//       break;
    case 2:
      if(dim==1)
      {
        return exp(-coeff_var_inner_x*pow(p[0]-0.5,2.0));  
      }else if(dim==2)
      {
        return exp(-(coeff_var_inner_x*pow(p[0]-0.5,2.0) + this->coeff_var_inner_y*pow(p[1]-0.5,2.0)));
      }
      break;
//     case 3:
//       return_value = std::pow(2.0*pi*this->coeff_var_inner_x, -2.0)*sin(2.0*pi*this->coeff_var_inner_x*p[0])-std::pow(p[0],2)/2.0;
//       break;
//     case 4:
//       return_value = pow(2.0*pi*this->coeff_var_inner_x, -1.0)*sin(2.0*pi*this->coeff_var_inner_x*p[0]);
//       break;
//     case 5:
//       return_value = pow(this->coeff_var_inner_x, -1.0)*p[0];
//       break;
    case 61:
      if (dim==1)
      {
        return_value = this->coeff_outer_x*pow((p[0]-this->center_x)/this->coeff_var_inner_x,2)+this->coeff_var_inde;
      }else if(dim==2)
      {
        return_value = this->coeff_outer_x*pow((p[0]-this->center_x)/this->coeff_var_inner_x,2)+this->coeff_outer_y*pow((p[1]-this->center_y)/this->coeff_var_inner_y,2) + (p[0]-0.5)*(p[1]-0.5) + this->coeff_var_inde;  
      }
      break;
//     case 611:
//       return_value = 2.0/3.0*pow(p[0],3.0) + 1.0/2.0*pow(p[0],2.0);
//       break;
//     case 612:
//       return_value = 1.0/12.0*pow(p[0],4.0) - 1.0/4.0*p[0];
//       break;
//     case 62:
//       return_value = pow(coeff_var_inner_x*p[0]+100.0,2.0);
//       break;
//     case 63:
//       return_value = pow(p[0],2.0)-pow(coeff_var_inner_x,2.0);
//       break;
//     case 64:
//       return_value = pow(p[0],2.0)+pow(p[1],2.0)-pow(coeff_var_inner_x,2.0);
//       break;      
//       
//       
//       
//     case 21:
//       return_value = sin(2.0*pi*p[0]);
//       break;
        
    default:
      cout << "case does not exist in ExactSolution_Step_4_Real_Base<dim>::value()\n";
      throw exception();
  }
    
  return return_value;
}

template <int dim>
Tensor<1,dim> ExactSolution_Step_4_Real_Base<dim>::gradient_base (const Point<dim>   &p,
                                       const unsigned int) const
{
  Tensor<1,dim> return_value;

  switch (id_case)
  {
//     case 1:
//       return_value[0] = pow(2.0*pi*this->coeff_var_inner_x, -1.0)*cos(2.0*pi*this->coeff_var_inner_x*p[0]);
//       break;
    case 2:
      if (dim==1)
      {
        return_value[0] = std::exp(-this->coeff_var_inner_x*std::pow(p[0]-0.5,2))*(-2.0*this->coeff_var_inner_x*(p[0]-0.5));
      }else if(dim==2)
      {
        return_value[0] = exp(-(coeff_var_inner_x*pow(p[0]-0.5,2.0) + this->coeff_var_inner_y*pow(p[1]-0.5,2.0)))*(-2.0*this->coeff_var_inner_x*(p[0]-0.5));
        return_value[1] = exp(-(coeff_var_inner_x*pow(p[0]-0.5,2.0) + this->coeff_var_inner_y*pow(p[1]-0.5,2.0)))*(-2.0*this->coeff_var_inner_y*(p[1]-0.5));
      }
      break;
//     case 3:
//       return_value[0] = pow(2.0*pi*this->coeff_var_inner_x, -1.0)*cos(2.0*pi*this->coeff_var_inner_x*p[0])-p[0];
//       break;
//     case 4:
//       return_value[0] = cos(2.0*pi*this->coeff_var_inner_x*p[0]);
//       break;
//     case 5:
//       return_value[0] = pow(this->coeff_var_inner_x, -1.0);
//       break;
    case 61:
      if (dim==1)
      {
        return_value[0] = 2*this->coeff_outer_x*(p[0]-this->center_x)/pow(this->coeff_var_inner_x,2.0);
      }else if(dim==2)
      {
        return_value[0] = 2*this->coeff_outer_x*(p[0]-this->center_x)/pow(this->coeff_var_inner_x,2.0) + (p[1]-0.5);
        return_value[1] = 2*this->coeff_outer_y*(p[1]-this->center_y)/pow(this->coeff_var_inner_y,2.0) + (p[0]-0.5);
      }        
      break;
//     case 611:
//       return_value[0] = 2.0*pow(p[0],2.0)+p[0];
//       break;      
//     case 612:
//       return_value[0] = 1.0/3.0*pow(p[0],3.0) - 1.0/4.0;
//       break;      
//     case 62:
//       return_value[0] = 2.0*coeff_var_inner_x*(coeff_var_inner_x*p[0]+100.0);
//       break;
//     case 63:
//       return_value[0] = 2.0*p[0];
//       break;
//     case 64:
//       return_value[0] = 2.0*p[0];
//       if(dim==2)
//       {
//         return_value[1] = 2.0*p[1];
//       }
//       break;       
//       
//     case 21:
//       return_value[0] = 2.0*pi*cos(2.0*pi*p[0]);
//       break;
        
    default:
      cout << "case does not exist in ExactSolution_Step_4_Real_Base<dim>::gradient()\n";
      throw exception();              
  }
  
  return return_value;
}

template <int dim>
SymmetricTensor<2,dim> ExactSolution_Step_4_Real_Base<dim>::hessian_base (const Point<dim>   &p,
                                       const unsigned int) const
{
  SymmetricTensor<2,dim> return_value;
  switch (id_case)
  {
//     case 1:
//       return_value[0][0] = -sin(2.0*pi*this->coeff_var_inner_x*p[0]);
//       break;
    case 2:
      if (dim==1)
      {
        return_value[0][0] = exp(-this->coeff_var_inner_x*std::pow(p[0]-0.5,2))*(std::pow(2*this->coeff_var_inner_x*(p[0]-0.5),2)-2.0*this->coeff_var_inner_x);                      //dp/dx
      }else if(dim==2)
      {
        return_value[0][0] = exp(-(coeff_var_inner_x*pow(p[0]-0.5,2.0) + this->coeff_var_inner_y*pow(p[1]-0.5,2.0)))*(std::pow(2*this->coeff_var_inner_x*(p[0]-0.5),2)-2.0*this->coeff_var_inner_x);
        return_value[0][1] = exp(-(coeff_var_inner_x*pow(p[0]-0.5,2.0) + this->coeff_var_inner_y*pow(p[1]-0.5,2.0)))*(-2.0*this->coeff_var_inner_x*(p[0]-0.5))*(-2.0*this->coeff_var_inner_y*(p[1]-0.5));
        return_value[1][0] = return_value[0][1];
        return_value[1][1] = exp(-(coeff_var_inner_x*pow(p[0]-0.5,2.0) + this->coeff_var_inner_y*pow(p[1]-0.5,2.0)))*(std::pow(2*this->coeff_var_inner_y*(p[1]-0.5),2)-2.0*this->coeff_var_inner_y);
      }
      break;
//     case 3:
//       return_value[0][0] = -(sin(2.0*pi*this->coeff_var_inner_x*p[0])+1.0);
//       break;
//     case 4:
//       return_value[0][0] = -2.0*pi*this->coeff_var_inner_x*sin(2.0*pi*this->coeff_var_inner_x*p[0]);
//       break;
//     case 5:
//       return_value[0][0] =  0.0;
//       break;
    case 61:
      if (dim==1)
      {
        return_value[0][0] = 2.0*this->coeff_outer_x/pow(this->coeff_var_inner_x,2);
      }else if(dim==2)
      {
        return_value[0][0] = 2.0*this->coeff_outer_x/pow(this->coeff_var_inner_x,2);  
        return_value[0][1] = 1.0;
        return_value[1][0] = 1.0;
        return_value[1][1] = 2.0*this->coeff_outer_y/pow(this->coeff_var_inner_y,2);
      }        
      break;
//     case 611:
//       return_value[0][0] = 4.0 * p[0] + 1.0;
//       break;        
//     case 612:
//       return_value[0][0] = pow(p[0],2.0);
//       break;        
//     case 62:
//       return_value[0][0] = 2.0*pow(coeff_var_inner_x,2.0);
//       break;
//     case 63:
//       return_value[0][0] = 2.0;
//       break;
//     case 64:
//       return_value[0][0] = 2.0;
//       if(dim==2)
//       {
//         return_value[1][1] = 2.0;
//       }
//       break;         
//       
//     case 21:
//       return_value[0][0] = -pow(2*pi,2.0)*sin(2.0*pi*p[0]);        // -2*pi*(this->coeff_var_inner_x*cos(2*pi*p[0])+(1.0+this->coeff_var_inner_x*p[0])*(-2*pi*sin(2*pi*p[0]))); 
//       break;
        
    default:
      cout << "case does not exist in ExactSolution_Step_4_Real_Base<dim>::hessian()\n";
      throw exception();              
  }
    
  return return_value;
}


template <int dim>
double ExactSolution_Step_4_Real<dim> :: value (const Point<dim>   &p,
                                                        const unsigned int ) const
{
  return ExactSolution_Step_4_Real_Base<dim>::value_base(p);
}

template <int dim>
Tensor<1,dim> ExactSolution_Step_4_Real<dim> :: gradient (const Point<dim>   &p,
                                                        const unsigned int ) const
{
  return ExactSolution_Step_4_Real_Base<dim>::gradient_base(p);
}

template <int dim>
SymmetricTensor<2,dim> ExactSolution_Step_4_Real<dim> :: hessian (const Point<dim>   &p,
                                                        const unsigned int ) const
{
  return ExactSolution_Step_4_Real_Base<dim>::hessian_base(p);
}




#endif
