
#ifndef AUXILIARYCLASS_H
#define AUXILIARYCLASS_H





template <int dim>
class Common_Var
{
  public:
    Common_Var(const unsigned int id_case);
//     protected:
    double coeff_outer_x, coeff_outer_y = 0;
    double coeff_inner_y = 0;
    double center_x, center_y = 0;
    double const_inde = 0;
    
    double coeff_diff_inde = 1.0;
    double coeff_Helm_inde = 1.0;
};


  
template <int dim>
class RightHandSide : public Function<dim>,
public Common_Var<dim>
{
  public:
    using Common_Var<dim>::Common_Var;
    RightHandSide (const unsigned int id_case, const double coeff_inner_x);       // : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  protected:
    const unsigned int id_case;
    const double coeff_inner_x;                          
};
  

template <int dim>
class PressureBoundaryValues : public Function<dim>,
public Common_Var<dim>
{
  public:
    using Common_Var<dim>::Common_Var;
    PressureBoundaryValues (const unsigned int id_case, const double coeff_inner_x);      // : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  protected:
    const unsigned int id_case;
    const double coeff_inner_x;                          
};
  
  
template <int dim>                                                                                 //create the boundary for the velocity
class GradientBoundary : public Function<dim>,
public Common_Var<dim>  
{
  public:
    using Common_Var<dim>::Common_Var;
    GradientBoundary (const unsigned int id_case, const unsigned int id_loc_coeff_diff, const double coeff_inner_x);            //  {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  protected:
    const unsigned int id_case, id_loc_coeff_diff;
    const double coeff_inner_x;
};  
  
template <int dim>
class Coeff_Diff_Final:public Common_Var<dim>                // : public TensorFunction<2,dim,real_t>
{
  public:
    using Common_Var<dim>::Common_Var;
    Coeff_Diff_Final (unsigned int id_case, const double coeff_diff_inner);            // : TensorFunction<2,dim,real_t>() {}

    void value (const vector<Point<dim>> &p,
                Vector<double>   &values) const;
    void gradient (const vector<Point<dim>> &p,
                Vector<double>   &values) const;                
  protected:
    const unsigned int id_case;
    const double coeff_diff_inner;
};
  
template <int dim>
class Coeff_Helm_Final:public Common_Var<dim>                
{
  public:
    using Common_Var<dim>::Common_Var;
    Coeff_Helm_Final (unsigned int id_case, const double coeff_Helm_inner);            

    void value (const vector<Point<dim>> &p,
                Vector<double>   &values) const;
           
  protected:
    const unsigned int id_case;
    const double coeff_Helm_inner;                                        // only 1 considered
};      


  template <int dim>
  class ExactSolution : public Function<dim>,
  public Common_Var<dim>
  {
  public:
    using Common_Var<dim>::Common_Var;
    ExactSolution (const unsigned int id_case, const unsigned int id_loc_coeff_diff, const double coeff_inner_x);          // : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    virtual void vector_gradient (const Point<dim> &p,
                               std::vector<Tensor<1,dim>>   &gradients) const;   
                               
  protected:
    const unsigned int id_case, id_loc_coeff_diff;
    const double coeff_inner_x;                                
  };
  


  
  
  
  template <int dim>
  RightHandSide<dim>::RightHandSide(const unsigned int id_case, const double coeff_inner_x): Function<dim>(dim+1),Common_Var<dim>(id_case),
  id_case(id_case),
  coeff_inner_x (coeff_inner_x)
  {} 
  
  template <int dim>
  PressureBoundaryValues<dim>::PressureBoundaryValues(const unsigned int id_case, const double coeff_inner_x): Function<dim>(dim+1),Common_Var<dim>(id_case),
  id_case(id_case),
  coeff_inner_x (coeff_inner_x)
  {}   
  
  template<int dim>
  GradientBoundary<dim>::GradientBoundary(const unsigned int id_case, const unsigned int id_loc_coeff_diff, const double coeff_inner_x): 
  Function<dim>(dim),Common_Var<dim>(id_case),               // the argument of Function determines the number of componenets for GradientBoundary, Jan. 14, 2020
  id_case(id_case),
  id_loc_coeff_diff(id_loc_coeff_diff),
  coeff_inner_x(coeff_inner_x)
  {}
  
  
  template <int dim>
  Coeff_Diff_Final<dim>::Coeff_Diff_Final (unsigned int id_case, const double coeff_diff_inner):Common_Var<dim>(id_case),
  id_case(id_case),
  coeff_diff_inner(coeff_diff_inner)
  {}    
  
  template <int dim>
  Coeff_Helm_Final<dim>::Coeff_Helm_Final (unsigned int id_case, const double coeff_Helm_inner):Common_Var<dim>(id_case),
  id_case(id_case),
  coeff_Helm_inner(coeff_Helm_inner)
  {}  
  
  
  template <int dim>
  ExactSolution<dim>::ExactSolution(const unsigned int id_case, const unsigned int id_loc_coeff_diff, const double coeff_inner_x): Function<dim>(dim+1),Common_Var<dim>(id_case),
  id_case(id_case),
  id_loc_coeff_diff(id_loc_coeff_diff),
  coeff_inner_x (coeff_inner_x)
  {}   
    

  
    template <int dim>
    Common_Var<dim>::Common_Var(const unsigned int id_case)
    {
//         cout << "    constructor of Common_Var" << endl;
      switch (id_case)
      {
          case 1:
              break;
          case 2:                                   // including benchmark Poisson equation
              coeff_outer_x = 1;
              coeff_outer_y = 0;
              coeff_inner_y = 1;
              center_x = 0.5;
              center_y = 0.5;
              const_inde = 0.0;              
              
              break;
          case 3:
              break;
          case 4:
              break;
          case 5:
              break;
              
              
          case 21:                                  // benchmark diffusion, i.e. p=sin(2pix), MBC; extended to d=1+cx
              break;
              
                                                    
          case 22:                                  // diff, p = exp(-(x-0.5)^2), d=1+0.5sin(cx)
          case 24:                                  // diff, p = exp(-(x-0.5)^2), d=1+cx
          case 241:                                 // diff, p = exp(-(x-0.5)^2), d=c
              center_x = 0.5;
              break;
              
              
          case 25:                                  // diff, p = (2pic)^{-2}sin(2picx), d(x) is a constant
          case 26:                                  // diff, p = (2pic)^{-2}sin(2picx), d(x) is 1+10*x
              break;              
              
              
              
                                                   // Helmholtz equations
          case 8:                                  // Helm, p = exp(-(x-0.5)^2), d=1+0.5sin(x), r=c
          case 81:                                 // Helm, p = exp(-(x-0.5)^2), d=1.0, r=c
              center_x = 0.5;
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
              return sin(2.0*pi*coeff_inner_x*p[0]);
              break;
          case 2:
              return -exp(-coeff_inner_x*std::pow(p[0]-0.5,2))*(std::pow(2*coeff_inner_x*(p[0]-0.5),2)-2.0*coeff_inner_x);
              break;
          case 3:
              return sin(2.0*pi*coeff_inner_x*p[0])+1.0;
              break;
          case 4:
              return 2.0*pi*coeff_inner_x*sin(2.0*pi*coeff_inner_x*p[0]);
              break;
          case 5:
              return 0.0;
              break;
              
              
          case 21:
              return -2.0*pi*(coeff_inner_x*cos(2.0*pi*p[0])+(1.0+coeff_inner_x*p[0])*(-2.0*pi*sin(2.0*pi*p[0])));      // using -(coeff_inner_x*2.0*pi*cos(2.0*pi*p[0])+(1.0+coeff_inner_x*p[0])*(-pow(2.0*pi,2.0)*sin(2.0*pi*p[0]))), accuracy higher than using -2.0*pi*(coeff_inner_x*cos(2.0*pi*p[0])+(1.0+coeff_inner_x*p[0])*(-2.0*pi*sin(2.0*pi*p[0]))), Jan 17, 2020
              break;
          case 22:
              return -exp(-pow(p[0]-this->center_x,2.0))*                               // a common item
              (0.5*coeff_inner_x*cos(coeff_inner_x*p[0])*(-2*(p[0]-0.5))                // this->coeff_diff_inde   coeff_inner_x
              +(1.0+0.5*sin(coeff_inner_x*p[0]))*(pow(2*(p[0]-this->center_x),2)-2.0));    
              break;
          case 24:
              return -exp(-pow(p[0]-this->center_x,2.0))*                               
              (coeff_inner_x*(-2*(p[0]-0.5))                // this->coeff_diff_inde   coeff_inner_x
              +(1.0+coeff_inner_x*p[0])*(pow(2*(p[0]-this->center_x),2)-2.0));    
              break;              
          case 241:
              return -exp(-pow(p[0]-this->center_x,2.0))*                               
              (coeff_inner_x*(pow(2*(p[0]-this->center_x),2)-2.0));    
              break;              
              
              
          case 25:
              return sin(2.0*pi*coeff_inner_x*p[0])*this->coeff_diff_inde;                      
              break;
          case 26:
              return -this->coeff_diff_inde*pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0]) + (1.0+this->coeff_diff_inde*p[0])*sin(2.0*pi*coeff_inner_x*p[0]);
              break;


              
          case 8:
              return -exp(-pow(p[0]-this->center_x,2.0))*                               
              (0.5*this->coeff_diff_inde*cos(this->coeff_diff_inde*p[0])*(-2*(p[0]-0.5))                // this->coeff_diff_inde   coeff_inner_x
              +(1.0+0.5*sin(this->coeff_diff_inde*p[0]))*(pow(2*(p[0]-this->center_x),2)-2.0)
              -coeff_inner_x);    
              break;     
          case 81:
              return -exp(-pow(p[0]-this->center_x,2.0))*                               
              ((pow(2*(p[0]-this->center_x),2)-2.0)*1.0
              -coeff_inner_x);    
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
          case 25:
          case 26:
              return pow(2.0*pi*coeff_inner_x, -2.0)*sin(2.0*pi*coeff_inner_x*p[0]);
              break;
          case 2:
              return exp(-coeff_inner_x*std::pow(p[0]-0.5,2));
              break;
          case 3:
              return std::pow(2.0*pi*coeff_inner_x,-2)*sin(2.0*pi*coeff_inner_x*p[0])-std::pow(p[0],2)/2.0;
              break;
          case 4:
              return pow(2.0*pi*coeff_inner_x, -1.0)*sin(2.0*pi*coeff_inner_x*p[0]);
              break;
          case 5:
              return pow(coeff_inner_x, -1.0)*p[0];
              break;
          case 21:
              return sin(2.0*pi*p[0])+this->const_inde;
              break;      
          case 22:
          case 24:
          case 241:
          case 8:
          case 81:
              return exp(-std::pow(p[0]-0.5,2));
              break; 
              
          default:
              cout << "case does not exist" << endl;
              throw exception();
              
      }
      return 0;
      
  }    
  

  
  template <int dim>
  void
  GradientBoundary<dim>::vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const
  {
      
    cout << "  values.size(): " << values.size() << "\n";
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));

      
      switch (id_case)
      {
          case 1:
              values(0)=-pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0]);                                     // u(-dp/dx)
              break;
          case 2:
              values(0) = -exp(-coeff_inner_x*std::pow(p[0]-0.5,2))*(-2.0*coeff_inner_x*(p[0]-0.5));                 
              break;
          case 3:
              values(0)=-(pow(2.0*pi*coeff_inner_x, 2.0))*cos(2.0*pi*coeff_inner_x*p[0])+p[0];
              break;
          case 4:
              values(0)=-cos(2.0*pi*coeff_inner_x*p[0]); 
              break;
          case 5:
              values(0) = -pow(coeff_inner_x, -1.0);
              break;
              
              
          case 21:
              if(id_loc_coeff_diff==0 or id_loc_coeff_diff == 1)
              {
                  values(0) = -2.0*pi*cos(2.0*pi*p[0])*(p[0]*coeff_inner_x+1.0);
              }else if(id_loc_coeff_diff==2)
              {
                  values(0) = -2.0*pi*cos(2.0*pi*p[0]); 
              }              
              break;
          case 22:
              values(0) = -(1.0+0.5*sin(coeff_inner_x*p[0]))*exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              break;    
          case 24:
          case 241:
              values(0) = -(1.0+coeff_inner_x*p[0])*exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              break;                 
              
          case 25:
              values(0) = -pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0])*this->coeff_diff_inde;
              break;                 
          case 26:
              values(0) = -pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0])*(1.0+this->coeff_diff_inde*p[0]);
              break;                 

              
          case 8:
              values(0) = -(1.0+0.5*sin(p[0]))*exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              break;  
          case 81:
              values(0) = -exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              break;                

          default:
              cout << "case does not exist" << endl;
              throw exception();              
      }
  }
  

      

  template <int dim>
  void
  Coeff_Diff_Final<dim>::value (const vector<Point<dim>> &p,
                        Vector<double> &values) const
  {
    for(unsigned int i=0; i<values.size(); ++i)
    {
        switch (id_case)
        {
            case 1:
            case 2:
            case 81:
            case 3:
            case 4:
            case 5:
                values[i] = 1.0;
                break;
            
            case 21:
                values[i] = 1.0 + coeff_diff_inner* p[i][0];
                break;
            case 22:
                values[i] = 1.0 + 0.5*sin(coeff_diff_inner* p[i][0]);
                break;
            case 24:
                values[i] = 1.0 + coeff_diff_inner* p[i][0];
                break;
            case 241:
                values[i] = coeff_diff_inner;
                break;
            case 25:
                values[i] = this->coeff_diff_inde;
                break;
            case 26:
                values[i] = 1.0+this->coeff_diff_inde*p[i][0];
                break;



            case 8:
                values[i] = 1.0 + 0.5*sin(p[i][0]);
                break;


            default:
                cout << "case does not exist" << endl;
                throw exception();                    
        }
    }

  }
  
  template <int dim>
  void
  Coeff_Diff_Final<dim>::gradient (const vector<Point<dim>> &p,
                        Vector<double> &values) const
  {
    for(unsigned int i=0; i<values.size(); ++i)
    {
        switch (id_case)
        {
            case 1:
            case 2:
            case 81:
            case 3:
            case 4:
            case 5:
                values[i] = 0.0;
                break;
            
            case 21:
                values[i] = 1.0;
                break;
            case 22:
                values[i] = 0.5*coeff_diff_inner*cos(coeff_diff_inner* p[i][0]);
                break;
            case 24:
                values[i] = coeff_diff_inner;
                break;
            case 241:
                values[i] = 0;
                break;
            case 25:
                values[i] = 0.0;
                break;
            case 26:
                values[i] = this->coeff_diff_inde;
                break;
                
                
            case 8:
                values[i] = 0.5*cos(p[i][0]);
                break;
                
                
            default:
                cout << "case does not exist" << endl;
                throw exception();                    
        }
    }
  }  
  
  template <int dim>
  void
  Coeff_Helm_Final<dim>::value (const vector<Point<dim>> &/*p*/,
                        Vector<double> &values) const
  {
         
    for(unsigned int i=0; i<values.size(); ++i)
    {
      if (id_case == 8 or id_case == 81)
      {         
        values[i] = coeff_Helm_inner;
      }
    }       
      
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
              values(0)=-pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0]);                                    // independent u, which is equal to -dp/dx for poisson equations, and -adp/dx for diffusion equations, affecting error of the first derivative
                                                                                                                       // equal to gradient boundary
              values(1)=pow(2.0*pi*coeff_inner_x, -2.0)*sin(2.0*pi*coeff_inner_x*p[0]);            // p, affecting the error of the solution
              break;
          case 2:
              values(0) = -exp(-coeff_inner_x*std::pow(p[0]-0.5,2))*(-2.0*coeff_inner_x*(p[0]-0.5));   
              if (dim==1)
              {
                values(1) = exp(-coeff_inner_x*std::pow(p[0]-0.5,2));                                                    
              }else if(dim==2)
              {
                values(1) = 0.0;
                values(2) = exp(-coeff_inner_x*std::pow(p[0]-0.5,2));  
              }
              break;
          case 3:
              values(0)=-(pow(2.0*pi*coeff_inner_x, -1.0))*cos(2.0*pi*coeff_inner_x*p[0])+p[0];
              values(1)=std::pow(2.0*pi*coeff_inner_x, -2.0)*sin(2.0*pi*coeff_inner_x*p[0])-std::pow(p[0],2)/2.0;
              break;
          case 4:
              values(0)=-cos(2.0*pi*coeff_inner_x*p[0]); 
              values(1)=pow(2.0*pi*coeff_inner_x, -1.0)*sin(2.0*pi*coeff_inner_x*p[0]);
              break;
          case 5:
              values(0) = -pow(coeff_inner_x, -1.0);
              values(1) = pow(coeff_inner_x, -1.0)*p[0];
              break;
              
          case 21:
              if(id_loc_coeff_diff==0 or id_loc_coeff_diff == 1)
              {
                  values(0) = -2.0*pi*cos(2.0*pi*p[0])*(p[0]*coeff_inner_x+1.0);
              }else if(id_loc_coeff_diff==2)
              {
                  values(0) = -2.0*pi*cos(2.0*pi*p[0]); 
              }
              values(1) = sin(2.0*pi*p[0])+this->const_inde;
              break;
          case 22:
              values(0) = -(1.0+0.5*sin(coeff_inner_x*p[0]))*exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              values(1) = exp(-std::pow(p[0]-0.5,2));          
              break;
          case 24:
              values(0) = -(1.0+coeff_inner_x*p[0])*exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              values(1) = exp(-std::pow(p[0]-0.5,2));          
              break;
          case 241:
              values(0) = -(coeff_inner_x)*exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              values(1) = exp(-std::pow(p[0]-0.5,2));          
              break;
          case 25:
              values(0)=-pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0])*this->coeff_diff_inde;
              values(1)= pow(2.0*pi*coeff_inner_x, -2.0)*sin(2.0*pi*coeff_inner_x*p[0]);
              break;              
          case 26:
              values(0)=-pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0])*(1.0+this->coeff_diff_inde*p[0]);
              values(1)= pow(2.0*pi*coeff_inner_x, -2.0)*sin(2.0*pi*coeff_inner_x*p[0]);
              break;              
              
              
          case 8:
              values(0) = -(1.0+0.5*sin(p[0]))*exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              values(1) = exp(-std::pow(p[0]-0.5,2));          
              break;
          case 81:
              values(0) = -exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
              values(1) = exp(-std::pow(p[0]-0.5,2));     
              break;
              
              
          default:
              cout << "case does not exist" << endl;
              throw exception();              
      }
      
  }
  
template <int dim>
void
ExactSolution<dim>::vector_gradient (const Point<dim> &p,
        	  	  	  	  	  	  	 std::vector<Tensor<1,dim>>   &gradients) const
{      
//     std::cout << "      ExactSolution::vector_gradient()" << std::endl;
      
  switch (id_case)
  {
    case 1:
      gradients[0][0] = sin(2.0*pi*coeff_inner_x*p[0]);                                             // independent du/dx, affecting error of du/dx, 
                                                                                                            // value equal to rhs
      gradients[1][0] = pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0]);                  // dp/dx
      break;
    case 2:
      if(dim==1)
      {
        gradients[0][0] = -exp(-coeff_inner_x*std::pow(p[0]-0.5,2))*(std::pow(2*coeff_inner_x*(p[0]-0.5),2)-2.0*coeff_inner_x);
        gradients[1][0] = std::exp(-coeff_inner_x*std::pow(p[0]-0.5,2))*(-2.0*coeff_inner_x*(p[0]-0.5));  
      }else if(dim==2)
      {
        gradients[0][0] = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(((4.0*pow(this->coeff_outer_x*(p[0]-this->center_x),2)-2.0*this->coeff_outer_x*this -> coeff_inner_x))/pow(this -> coeff_inner_x,2));          //-pxx
        gradients[0][1] = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(((4.0*this->coeff_outer_x*(p[0]-this->center_x)*this->coeff_outer_y*(p[1]-this->center_y)))/pow(this -> coeff_inner_x,2));         //-pxy
        gradients[1][0] = gradients[0][1];         //-pyx
        gradients[1][1] = -exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*((4.0*pow(this->coeff_outer_y*(p[1]-this->center_y),2)-2.0*this->coeff_outer_y*this -> coeff_inner_x)/pow(this -> coeff_inner_x,2));          //-pyy
        gradients[2][0] = exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(-2*this->coeff_outer_x*(p[0]-this->center_x)/this -> coeff_inner_x);     //px
        gradients[2][1] = exp(-(this->coeff_outer_x*pow(p[0]-this->center_x,2)+this->coeff_outer_y*pow(p[1]-this->center_y,2))/this -> coeff_inner_x)*(-2*this->coeff_outer_y*(p[1]-this->center_y)/this -> coeff_inner_x);     //py
      }
      break;
    case 3:
        gradients[0][0] = sin(2.0*pi*coeff_inner_x*p[0])+1.0;
        gradients[1][0] = pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0])-p[0];
        break;
    case 4:
        gradients[0][0] = 2.0*pi*coeff_inner_x*sin(2.0*pi*coeff_inner_x*p[0]);
        gradients[1][0] = cos(2.0*pi*coeff_inner_x*p[0]);
        break;
    case 5:
        gradients[0][0] = 0.0;
        gradients[1][0] = pow(coeff_inner_x, -1.0);
        break;
        
        
    case 21:
        if(id_loc_coeff_diff==0 or id_loc_coeff_diff == 1)
        {
            gradients[0][0] = -2.0*pi*(cos(2.0*pi*p[0])*coeff_inner_x+(1.0+coeff_inner_x* p[0])*(-2.0*pi*sin(2.0*pi*p[0])));
        }else if (id_loc_coeff_diff==2)
        {    
            gradients[0][0] = pow(2.0*pi,2)*sin(2.0*pi*p[0]);
        }
        gradients[1][0] = 2.0*pi*cos(2.0*pi*p[0]);
        break;
    case 22:
        gradients[0][0] = -exp(-pow(p[0]-this->center_x,2.0))*(0.5*coeff_inner_x*cos(coeff_inner_x*p[0])*(-2*(p[0]-0.5))+(1.0+0.5*sin(coeff_inner_x*p[0]))*(pow(2*(p[0]-this->center_x),2)-2.0));
        gradients[1][0] = exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
        break;    
    case 24:
        gradients[0][0] = -exp(-pow(p[0]-this->center_x,2.0))*                              
                        (coeff_inner_x*(-2*(p[0]-0.5))               
                        +(1.0+coeff_inner_x*p[0])*(pow(2*(p[0]-this->center_x),2)-2.0));    
        gradients[1][0] = exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
        break;    
    case 241:
        gradients[0][0] = -exp(-pow(p[0]-this->center_x,2.0))*                               
                        (coeff_inner_x*(pow(2*(p[0]-this->center_x),2)-2.0));     
        gradients[1][0] = exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
        break;    
        
    case 25:
        gradients[0][0] = sin(2.0*pi*coeff_inner_x*p[0])*this->coeff_diff_inde;
        gradients[1][0] = pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0]);
        break;              
    case 26:
        gradients[0][0] = -this->coeff_diff_inde*pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0]) + (1.0+this->coeff_diff_inde*p[0])*sin(2.0*pi*coeff_inner_x*p[0]);
        gradients[1][0] = pow(2.0*pi*coeff_inner_x, -1.0)*cos(2.0*pi*coeff_inner_x*p[0]);
        break;
        
    case 8:
        gradients[0][0] = -exp(-pow(p[0]-this->center_x,2.0))*(0.5*cos(p[0])*(-2*(p[0]-0.5))+(1.0+0.5*sin(p[0]))*(pow(2*(p[0]-this->center_x),2)-2.0));
        gradients[1][0] = exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
        break;   
    case 81:
        gradients[0][0] = -exp(-pow(p[0]-this->center_x,2.0))*(pow(2*(p[0]-this->center_x),2)-2.0);     
        gradients[1][0] = exp(-std::pow(p[0]-0.5,2))*(-2.0*(p[0]-0.5));
        break;                 
        
        
    default:
        cout << "case does not exist" << endl;
        throw exception();              
  }
}
  
  
  template <int dim>
  class ComputeSurface : public DataPostprocessorScalar<dim>
  {
  public:
    ComputeSurface ();

    virtual
    void
    evaluate_scalar_field
    (const DataPostprocessorInputs::Scalar<dim> &inputs,
     std::vector<Vector<double> >               &computed_quantities) const;
  };

  template <int dim>
  ComputeSurface<dim>::ComputeSurface ()
  :
    DataPostprocessorScalar<dim> ("Surface",
  		  update_gradients)
  {}

  template <int dim>
  void
  ComputeSurface<dim>::evaluate_scalar_field
  (const DataPostprocessorInputs::Scalar<dim> &inputs,
   std::vector<Vector<double> >               &computed_quantities) const
  {
  	for (unsigned int i=0; i<inputs.solution_values.size(); i++)
  	{
  		 	  computed_quantities[i] = inputs.solution_values[i];
  	}
  }
  


  // @sect3{Linear solvers and preconditioners}

template <class MatrixType>
class InverseMatrix_CG : public Subscriptor
{
  public:
    InverseMatrix_CG(const MatrixType &m);

    void vmult(Vector<double>       &dst,
              const Vector<double> &src) const;
                
    void my_matrix_print () const;

  private:
    const SmartPointer<const MatrixType> matrix;
};


template <class MatrixType>
InverseMatrix_CG<MatrixType>::InverseMatrix_CG (const MatrixType &m)
:
matrix (&m)
{
    std::cout << "  InverseMatrix_CG" << std::endl;
//       std::cout << "    the type of the matrix recieved in the constructor of InverseMatrix_CG is: " << typeid(MatrixType).name() << '\n';
//       matrix->print_formatted(std::cout);      
}


  template <class MatrixType>
  void InverseMatrix_CG<MatrixType>::vmult (Vector<double>       &dst,                 // this function passes the private variable 'Matrix' and the arguments 'dst' and 'src' to the function cg.solve
                                         const Vector<double> &src) const
  {
      
//     std::cout << "              %%%%%% using vmult in the class InverseMatrix_CG";         // , which invokes the CG solver defined in it
    
    SolverControl solver_control (std::max(src.size()*1000, static_cast<std::size_t> (200)),
                                  1e-16*src.l2_norm());                                         // here the PreconditionIdentity is used, June 21, 2019
    SolverCG<>    cg (solver_control);
    
//     std::cout << "    tolerance: " << solver_control.tolerance() << std::endl;

    dst = 0;
    
//     std::cout << "        CG solver for the InverseMatrix_CG" << std::endl;
    
//     std::cout << "              the type of the matrix received is: " << typeid(MatrixType).name() << '\n';
    
    cg.solve (*matrix, dst, src, PreconditionIdentity());           // when the matrix is of ApproximateSchurComplement type that is used for the precondition, this function will invoke the vmult defined in the class ApproximateSchurComplement
                                                                    // when the matrix is of BlockSparseMatrix type, 
    
//     std::cout << "        finished" << std::endl;
    
/*    std::cout << "    "
            << solver_control.last_step()
            << " CG iterations to obtain convergence."              // Schur complement
//             << "\n"
            << std::endl; */   
    
//     std::cout << "        the use of the vmult in the class InverseMatrix_CG finished" << std::endl;
    
  }

  template <class MatrixType>
  void InverseMatrix_CG<MatrixType>::my_matrix_print() const
  {
//       std::cout << "    my_matrix_print" << std::endl;
      matrix->print_formatted(std::cout);
  }  
  
//   =================================================== Using the direct solver to solve the matrix equation when the lhs is 'M'
  
template <class MatrixType>
  class InverseMatrix_UMF : public Subscriptor
  {
  public:
    InverseMatrix_UMF(const MatrixType &m);

    void vmult(Vector<double>       &dst,
               const Vector<double> &src) const;

  private:
    const SmartPointer<const MatrixType> matrix;                            // to store the matrix received in the constructor
  };


  template <class MatrixType>
  InverseMatrix_UMF<MatrixType>::InverseMatrix_UMF (const MatrixType &m)            // m = 'M'
    :
    matrix (&m)
  {
      std::cout << "  InverseMatrix_UMF" << std::endl;
  }


  template <class MatrixType>                                               // used in SchurComplement::vmult, see line 1113
  void InverseMatrix_UMF<MatrixType>::vmult (Vector<double>       &dst,
                                         const Vector<double> &src) const
  {
    
//     std::cout << "InverseMatrix_UMF";                              

    dst = 0;
    
//     std::cout << "    *matrix is: ";                 
//     matrix->print(std::cout);

//     std::cout << " -> UMFPACK solver";                 
    
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(*matrix);
    A_direct.vmult (dst, src);
    
//     cout << endl;

  }    


//   typedef InverseMatrix my_matrix_inverse;
  
  // @sect4{The <code>SchurComplement</code> class}

  template <class T>
  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement (const BlockSparseMatrix<double>            &A,
                     const T &Minv);                     

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
    const SmartPointer<const T > m_inverse;

    mutable Vector<double> tmp1, tmp2;
  };


  template <class T>
  SchurComplement<T>
  ::SchurComplement (const BlockSparseMatrix<double>            &A,
                     const T &Minv)
    :
    system_matrix (&A),               // we only use B and B^T in the system_matrix
    m_inverse (&Minv),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}

  template <class T>
  void SchurComplement<T>::vmult (Vector<double>       &dst,
                               const Vector<double> &src) const
  {
    
//     std::cout << "\n    ###### using vmult in class SchurComplement" << std::endl;
//     
//     std::cout << "          --> B * p_k" << std::endl;
    
    system_matrix->block(0,1).vmult (tmp1, src);		                          // tmp1 = B * p_k, src is p_k in the CG solver
    
//     std::cout << "          --> M^-1 * B * p_k" << std::endl;
    
    
//     InverseMatrix<SparseMatrix<real_t> > inverse_mass (system_matrix->block(0,0));   // instead of using the argument inverse_mass, we create a local inverse_mass, and it works well
//     inverse_mass.vmult (tmp2, tmp1);
    
    m_inverse->vmult (tmp2, tmp1);				                                  // solve M * tmp2 = tmp1, which involves the CG solver defined in InverseMatrix
    
//     m_inverse->vmult (dst, src);
    
//     std::cout << "          --> B^T * M^-1 * B * p_k" << std::endl;
    
    system_matrix->block(1,0).vmult (dst, tmp2);		// dst = B^T * M^-1 * B * p_k, note that B^T * M^-1 * B is never explicit, therefore, we compute it for every loop for the Schur complement
    
//     std::cout << "    ###### " << std::endl;
    
//     std::cout << "	  A * p_k done for SchurComplement" << std::endl;
    
//     std::cout << "\n";
  }


  class ApproximateSchurComplement : public Subscriptor
  {
  public:
    ApproximateSchurComplement (const BlockSparseMatrix<real_t> &A);

    void vmult (Vector<real_t>       &dst,
                const Vector<real_t> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<real_t> > system_matrix;

    mutable Vector<real_t> tmp1, tmp2;
  };



  ApproximateSchurComplement::ApproximateSchurComplement
  (const BlockSparseMatrix<real_t> &A) :
    system_matrix (&A),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {
//       std::cout << "    the system_matrix recieved in the constructor of ApproximateSchurComplement reads: " << std::endl;
//       system_matrix->print_formatted(std::cout);
  }


  void
  ApproximateSchurComplement::vmult
  (Vector<real_t>       &dst,
   const Vector<real_t> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }  
  
  



template <int dim>
void print_constraints_info(ConstraintMatrix &constraints, DoFHandler<dim>& dof_handler)
{
  cout << "  @constraints info" << endl;

  cout << "  the constrained dofs are:";
    
  for (unsigned int i = 0; i<dof_handler.n_dofs(); ++i)
  {
    if(constraints.is_constrained(i))
    {
      std::cout << " " << i;
    }
  }
    
  cout << "(" << constraints.n_constraints() << " in total)" << std::endl;
    
  std::cout << "  the content of constraints is: " << std::endl;
  constraints.print(std::cout);    
}


template <typename T>
void scale_quad_coords_1_0(std::vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    quad_coords_vector[i](0)=(quad_coords_vector[i](0)-0.5)*2.0;
    quad_coords_vector[i](1)=quad_coords_vector[i](1)*2.0;
  }
}


template <typename T>
void scale_quad_coords_0_1(std::vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    quad_coords_vector[i](0)=quad_coords_vector[i](0)*2.0;
    quad_coords_vector[i](1)=(quad_coords_vector[i](1)-0.5)*2.0;
  }
}


template <typename T>
void scale_quad_coords_small(std::vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    for (unsigned int j=0; j<quad_coords_vector[0].dimension; ++j)
    {
      quad_coords_vector[i](j)=quad_coords_vector[i](j)*2.0;
    }
  }
}


template <typename T>
void scale_quad_coords_big(std::vector<T> &quad_coords_vector)
{
  for (unsigned int i=0; i<quad_coords_vector.size();++i)
  {
    for (unsigned int j=0; j<quad_coords_vector[0].dimension; ++j)
    {
      quad_coords_vector[i](j)=(quad_coords_vector[i](j)-0.5)*2.0;   
    }
  }
}



#endif


