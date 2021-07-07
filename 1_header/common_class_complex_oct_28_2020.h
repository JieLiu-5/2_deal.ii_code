
#ifndef COMMON_CLASS_COMPLEX_H
#define COMMON_CLASS_COMPLEX_H

template <int dim>
class Common_Var_Complex
{
  public:
    Common_Var_Complex(const unsigned int id_case);
    
    double length = 1.0;
    complex<double> constant_coeff_d = 1.0 + 1.0i;
    
    complex<double> A = 0.1267946481444569561958246595168 - 0.1951839458217654310612232393396i;
    complex<double> B = 0.87320535185554282175957041545189 + 0.1951839458217654310612232393396i; 
    
    complex<double> r1 = 1.0 + 1.0i;
    complex<double> r2 = 0.0 - 1.0i;
    
    double coeff_a=0.0;
    double coeff_b=0.0;
    
    double coeff_y = pi;          // pi 0.0
    double centre_y = 0.5;
    
    complex<double> T1 = 1.0 + 1.0i;
    complex<double> T2 = 1.0 + 1.0i;
    double T3 = -2.0;
    
};

template <int dim>
class Coeff_Diff_Complex
{
  public:
    Coeff_Diff_Complex (const unsigned int id_case, const unsigned int id_coeff_diff_value, const double coeff_diff_inner);

    void value (const vector<Point<dim>> &p,
                vector<Vector<double>>   &values) const;             
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_diff_value;
    const double coeff_diff_inner;
};

template <int dim>
class Coeff_Diff_Complex_Inverse
{
  public:
    Coeff_Diff_Complex_Inverse (const unsigned int id_case, const unsigned int id_coeff_diff_value, const double coeff_diff_inner);

    void value (const vector<Point<dim>> &p,
                vector<Vector<double>>   &values) const;             
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_diff_value;
    const double coeff_diff_inner;
};

template <int dim>
class Coeff_Helm_Complex:public Common_Var_Complex<dim>                
{
  public:
    using Common_Var_Complex<dim>::Common_Var_Complex;
    Coeff_Helm_Complex (const unsigned int id_case, const unsigned int id_coeff_helm_value, const double coeff_Helm_inner);            

    void value (const vector<Point<dim>> &p,
                vector<Vector<double>>   &values) const;
           
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_helm_value;
    const double coeff_Helm_inner;                                        // only 1 considered
};

template <int dim>
class RightHandSide_Complex : public Common_Var_Complex<dim>
{
  public:
    using Common_Var_Complex<dim>::Common_Var_Complex;
    RightHandSide_Complex (const unsigned int id_case, const unsigned int id_coeff_diff_value, const unsigned int id_coeff_helm_value);
    void vector_values (const vector<Point<dim>> &p,
                        vector<Vector<double>>   &value) const;
  protected:
    const unsigned int id_case;                    
    const unsigned int id_coeff_diff_value;                   
    const unsigned int id_coeff_helm_value;                   
};


template <int dim>
class PressureBoundaryValues_Complex : public Function<dim>,
public Common_Var_Complex<dim>
{
public:
  using Common_Var_Complex<dim>::Common_Var_Complex;
  PressureBoundaryValues_Complex(const unsigned int id_case);

  virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const;
protected:
  unsigned int id_case;
};

template <int dim>
Coeff_Diff_Complex<dim>::Coeff_Diff_Complex (const unsigned int id_case, const unsigned int id_coeff_diff_value, const double coeff_diff_inner):
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value),
coeff_diff_inner(coeff_diff_inner)
{}    

template <int dim>
Coeff_Diff_Complex_Inverse<dim>::Coeff_Diff_Complex_Inverse (const unsigned int id_case, const unsigned id_coeff_diff_value, const double coeff_diff_inner):
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value),
coeff_diff_inner(coeff_diff_inner)
{}

template <int dim>
Coeff_Helm_Complex<dim>::Coeff_Helm_Complex (const unsigned int id_case, const unsigned int id_coeff_helm_value, const double coeff_Helm_inner):Common_Var_Complex<dim>(id_case),
id_case(id_case),
id_coeff_helm_value(id_coeff_helm_value),
coeff_Helm_inner(coeff_Helm_inner)
{}

template <int dim>
RightHandSide_Complex<dim>::RightHandSide_Complex(const unsigned int id_case, const unsigned int id_coeff_diff_value, const unsigned int id_coeff_helm_value): Common_Var_Complex<dim>(id_case),
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value),
id_coeff_helm_value(id_coeff_helm_value)
{}

template <int dim>
PressureBoundaryValues_Complex<dim>::PressureBoundaryValues_Complex(const unsigned int id_case): Function<dim>(dim+1),Common_Var_Complex<dim>(id_case),
id_case(id_case)
{}


template <int dim>
Common_Var_Complex<dim>::Common_Var_Complex(const unsigned int id_case)
{
  switch (id_case)
  {
    case 1:
      break;
    case 2:
      coeff_a=1.0;
      coeff_b=0.0;
      break;
    default:
      cout << "case does not exist in Common_Var_Complex\n";
      throw exception();
  }
}

template <int dim>
void
Coeff_Diff_Complex<dim>::value (const vector<Point<dim>> &p,
                                vector<Vector<double>> &values) const
{
                                                                                            // coefficient d also appears in 1. class of right-hand side
                                                                                            //                               2. class of gradient boundary
                                                                                            //                               3. class of exact solution
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch (id_case)
    {
      case 1:
        values[i][0] = -(0.01+p[i][0])*(1.01-p[i][0]);
        values[i][1] = 0.0;
        break;
      case 2:
        switch(id_coeff_diff_value)
        {
          case 1:
            values[i][0] = 1.0;
            values[i][1] = 0.0;
            
//         values[i][0] = 2.0;
//         values[i][1] = 0.0;
        
//         values[i][0] = 2.0;
//         values[i][1] = 3.0;
            
            break;
          case 2:
            values[i][0] = 1.0+p[i][0];
            values[i][1] = 0.0;
            break;
          case 3:
            values[i][0] = 1.0+p[i][0];
            values[i][1] = 1.0+p[i][0];
            
/*            values[i][0] = -exp(-p[i][0]);
            values[i][1] = -exp(-p[i][0]);  */          
            break;
        }
        break;
    }
  }
}

template <int dim>
void
Coeff_Diff_Complex_Inverse<dim>::value (const vector<Point<dim>> &p,
                                        vector<Vector<double>> &values) const
{
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch (id_case)
    {
      case 1:
        values[i][0] = 1.0/(-(0.01+p[i][0])*(1.01-p[i][0]));
        values[i][1] = 0.0;
        break;
      case 2:
        switch(id_coeff_diff_value)
        {
          case 1:
            values[i][0] = 1.0;
            values[i][1] = 0.0;
//         
//         values[i][0] = 0.5;
//         values[i][1] = 0.0;

//         values[i][0] = 2.0/13.0;
//         values[i][1] = - 3.0/13.0;
            
            break;
          case 2:
            values[i][0] = 1.0/(1.0+p[i][0]);
            values[i][1] = 0.0;
            break;
          case 3:
            values[i][0] = 1.0/(2.0*(1.0+p[i][0]));
            values[i][1] = - 1.0/(2.0*(1.0+p[i][0]));
            
//         values[i][0] = -exp(p[i][0])/2.0;
//         values[i][1] = exp(p[i][0])/2.0;         
            break;
        }
        break;
    }
  }
}

template <int dim>
void
Coeff_Helm_Complex<dim>::value (const vector<Point<dim>> &p,                            // also affecting the parameter in the class of right-hand side
                                vector<Vector<double>> &values) const
{
  for(unsigned int i=0; i<values.size(); ++i)
  {
    switch (id_case)
    {
      case 1:
        values[i][0] = 0.0;
        values[i][1] = -0.01;
        break;
      case 2:
        switch(id_coeff_helm_value)
        {
          case 1:
            values[i][0] = 0.0;
            values[i][1] = 0.0;
            break;
          case 2:
            values[i][0] = 2.0*exp(-p[i][0]);
            values[i][1] = 0.0;
            break;
        }
        
        break;
    }
  }
}

template <int dim>
void
RightHandSide_Complex<dim>::vector_values (const vector<Point<dim>> &p,
                                           vector<Vector<double>>   &values) const
{
    
  complex<double> value_d = 0.0 + 0.0i;          
  complex<double> value_d_x = 0.0 + 0.0i;
  complex<double> value_d_y = 0.0 + 0.0i;
  
  complex<double> value_r = 0.0 + 0.0i;

  complex<double> value_p = 0.0 + 0.0i;
  complex<double> value_p_x = 0.0 + 0.0i;
  complex<double> value_p_y = 0.0 + 0.0i;
  complex<double> value_p_xx = 0.0 + 0.0i;
  complex<double> value_p_yy = 0.0 + 0.0i;  
  
  for(unsigned int i=0; i<values.size(); ++i)
  {
    
    switch(id_case)
    {
      case 1:
        values[i][0] = 1.0;
        values[i][1] = 0.0;
        break;
      case 2:                      
          
        switch (id_coeff_diff_value)
        {
          case 1:    
            value_d = 1.0 + 0.0i;                 //  2.0 + 0.0i      2.0 + 3.0i;
            value_d_x = 0.0 + 0.0i;
            break;
          case 2:
            value_d = 1.0+p[i][0] + 0.0i;
            value_d_x = 1.0 + 0.0i;
            break;
          case 3:
            value_d = this->constant_coeff_d*(1.0+p[i][0]);
            value_d_x = this->constant_coeff_d;          
            
        //           value_d = -this->constant_coeff_d*exp(-p[i][0]);
        //           value_d_x = this->constant_coeff_d*exp(-p[i][0]);
            
            break;
        }
        
        switch (id_coeff_helm_value)
        {
          case 1:
            value_r = 0.0 + 0.0i;
            break;
          case 2:
            value_r = 2.0*exp(-p[i][0]) + 0.0i;      
            break;
        }
        
          
        if (dim==1)
        {
          values[i][0] = 0.0;
          values[i][1] = 0.0;          
        }else if(dim==2)                                        // based on the code of the standard FEM for the problem
        {
            
          value_p = (this->A*exp(this->r1*p[i][0])+this->B*exp(this->r2*p[i][0]))*cos(this->coeff_y*(p[i][1]-this->centre_y));
          value_p_x = this->A*this->r1*exp(this->r1*p[i][0])+this->B*this->r2*exp(this->r2*p[i][0])*cos(this->coeff_y*(p[i][1]-this->centre_y));
          value_p_xx = (this->A*pow(this->r1,2)*exp(this->r1*p[i][0])+this->B*pow(this->r2,2)*exp(this->r2*p[i][0]))*cos(this->coeff_y*(p[i][1]-this->centre_y));
          value_p_yy = (this->A*exp(this->r1*p[i][0])+this->B*exp(this->r2*p[i][0]))*(-pow(this->coeff_y,2.0))*cos(this->coeff_y*(p[i][1]-this->centre_y));

            
          values[i][0] = //
                         (-((value_d_x*value_p_x + value_d_y*value_p_y)+value_d*(value_p_xx+value_p_yy))+value_r*value_p).real();
          
//                         ((this->T1*(this->A*pow(this->r1,2)*exp(this->r1*p[i][0])+this->B*pow(this->r2,2)*exp(this->r2*p[i][0]))).real()*cos(this->coeff_y*(p[i][1]-this->centre_y))
//                          +(this->T1*(this->A*exp(this->r1*p[i][0])+this->B*exp(this->r2*p[i][0]))).real()*cos(this->coeff_y*(p[i][1]-this->centre_y))*(-pow(this->coeff_y,2))
//                          -(this->T2*(this->A*this->r1*exp(this->r1*p[i][0])+this->B*this->r2*exp(this->r2*p[i][0]))).real()*cos(this->coeff_y*(p[i][1]-this->centre_y))
//                          -this->T3*(this->A*exp(this->r1*p[i][0])+this->B*exp(this->r2*p[i][0])).real()*cos(this->coeff_y*(p[i][1]-this->centre_y)))*exp(-p[i][0]);
                         
                         
          values[i][1] = //
                         (-((value_d_x*value_p_x + value_d_y*value_p_y)+value_d*(value_p_xx+value_p_yy))+value_r*value_p).imag();
                         
//                          ((this->T1*(this->A*pow(this->r1,2)*exp(this->r1*p[i][0])+this->B*pow(this->r2,2)*exp(this->r2*p[i][0]))).imag()*cos(this->coeff_y*(p[i][1]-this->centre_y))
//                          +(this->T1*(this->A*exp(this->r1*p[i][0])+this->B*exp(this->r2*p[i][0]))).imag()*cos(this->coeff_y*(p[i][1]-this->centre_y))*(-pow(this->coeff_y,2))
//                          -(this->T2*(this->A*this->r1*exp(this->r1*p[i][0])+this->B*this->r2*exp(this->r2*p[i][0]))).imag()*cos(this->coeff_y*(p[i][1]-this->centre_y))
//                          -this->T3*(this->A*exp(this->r1*p[i][0])+this->B*exp(this->r2*p[i][0])).imag()*cos(this->coeff_y*(p[i][1]-this->centre_y)))*exp(-p[i][0]);
        }
        break;
      default:
        cout << "  case does not exist \n";
    }
  }
}

template <int dim>
inline
void PressureBoundaryValues_Complex<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
{
  Assert (values.size() == 2, ExcDimensionMismatch (values.size(), 2));
  
  switch(id_case)
  {
    case 1:                                     // d = -(0.01+x)(1.01-x), r = -0.01i
      values(0) = this->coeff_a;                    
      values(1) = 0;
      break;
    case 2:                                     // d = -(1+1i)exp(-x), r=2*exp(-x)
      if (dim==1)
      {
        values(0) = this->coeff_a;
        values(1) = 0.0;
      }else if(dim==2)
      {      
        values(0) = (this->A*std::exp(this->r1*p[0])+this->B*std::exp(this->r2*p[0])).real()*cos(this->coeff_y*(p[1]-this->centre_y));
        values(1) = (this->A*std::exp(this->r1*p[0])+this->B*std::exp(this->r2*p[0])).imag()*cos(this->coeff_y*(p[1]-this->centre_y));                  // important to be correct
      }
      break;
    default:
      cout << "case does not exist\n";
      throw exception();
  }
}

#endif
