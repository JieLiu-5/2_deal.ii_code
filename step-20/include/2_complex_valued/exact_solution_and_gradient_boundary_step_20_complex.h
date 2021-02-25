
#ifndef EXACT_SOLUTION_AND_GRADIENT_BOUNDARY_STEP_20_COMPLEX_H
#define EXACT_SOLUTION_AND_GRADIENT_BOUNDARY_STEP_20_COMPLEX_H

template <int dim>
class GradientBoundary_Step_20_Complex : 
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>,
public Coeff_Diff_Complex<dim>
{
  public:
    GradientBoundary_Step_20_Complex (const unsigned int id_case, const unsigned int id_coeff_diff_value);
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  protected:
    const unsigned int id_case;                                 
    const unsigned int id_coeff_diff_value;                                 
};

template <int dim>                                                          // this class and the following class are only for RT/DGQ elements
class GradientBoundary_Step_20_Complex_Real : 
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>,
public Coeff_Diff_Complex<dim>
{
  public:
    GradientBoundary_Step_20_Complex_Real (const unsigned int id_case, const unsigned int id_coeff_diff_value);
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  protected:
    const unsigned int id_case;                                 
    const unsigned int id_coeff_diff_value;                                 
};

template <int dim>
class GradientBoundary_Step_20_Complex_Imag : 
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>,
public Coeff_Diff_Complex<dim>
{
  public:
    GradientBoundary_Step_20_Complex_Imag (const unsigned int id_case, const unsigned int id_coeff_diff_value);
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  protected:
    const unsigned int id_case;                              
    const unsigned int id_coeff_diff_value;                              
};

template <int dim>
class ExactSolution_Step_20_Complex_Both :
public Function<dim>,
public ExactSolution_Step_4_Complex_Both<dim>,
public Coeff_Diff_Complex<dim>
{
  public:
    ExactSolution_Step_20_Complex_Both (const unsigned int id_case, const unsigned int id_coeff_diff_value);

    virtual void vector_value_both (const Point<dim> &p,
                               Vector<complex<double>>   &value) const;
    virtual void vector_gradient_both (const Point<dim> &p,
                               vector<Tensor<1,dim,complex<double>>>   &gradients) const;
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_diff_value;
};

template <int dim>
class ExactSolution_Step_20_Complex_Real :
public ExactSolution_Step_20_Complex_Both<dim>
{
  public:
    ExactSolution_Step_20_Complex_Real (const unsigned int id_case, const unsigned int id_coeff_diff_value);

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    virtual void vector_gradient (const Point<dim> &p,
                               vector<Tensor<1,dim>>   &gradients) const;
  protected:
    const unsigned int id_case;
    const unsigned int id_coeff_diff_value;
};

template <int dim>
class ExactSolution_Step_20_Complex_Imag :
public ExactSolution_Step_20_Complex_Both<dim>
{
  public:
    ExactSolution_Step_20_Complex_Imag (const unsigned int id_case, const unsigned int id_coeff_diff_value);

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    virtual void vector_gradient (const Point<dim> &p,
                               vector<Tensor<1,dim>>   &gradients) const;
  protected:
    const unsigned int id_case;                              
    const unsigned int id_coeff_diff_value;                              
};


template <int dim>
GradientBoundary_Step_20_Complex<dim>::GradientBoundary_Step_20_Complex(const unsigned int id_case, const unsigned int id_coeff_diff_value): 
Function<dim>(2*(dim+1)),
ExactSolution_Step_4_Complex_Both<dim>(id_case),
Coeff_Diff_Complex<dim>(id_case,id_coeff_diff_value,1.0),
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value)
{}

template <int dim>
GradientBoundary_Step_20_Complex_Real<dim>::GradientBoundary_Step_20_Complex_Real(const unsigned int id_case, const unsigned int id_coeff_diff_value): 
Function<dim>(2),
ExactSolution_Step_4_Complex_Both<dim>(id_case),
Coeff_Diff_Complex<dim>(id_case,id_coeff_diff_value,1.0),
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value)
{}

template <int dim>
GradientBoundary_Step_20_Complex_Imag<dim>::GradientBoundary_Step_20_Complex_Imag(const unsigned int id_case, const unsigned int id_coeff_diff_value): 
Function<dim>(2),
ExactSolution_Step_4_Complex_Both<dim>(id_case),
Coeff_Diff_Complex<dim>(id_case,id_coeff_diff_value,1.0),
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value)
{}

template <int dim>
ExactSolution_Step_20_Complex_Both<dim>::ExactSolution_Step_20_Complex_Both(const unsigned int id_case, const unsigned int id_coeff_diff_value):
Function<dim>(dim+1),
ExactSolution_Step_4_Complex_Both<dim>(id_case),
Coeff_Diff_Complex<dim>(id_case,id_coeff_diff_value,1.0),
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value)
{}

template <int dim>
ExactSolution_Step_20_Complex_Real<dim>::ExactSolution_Step_20_Complex_Real(const unsigned int id_case, const unsigned int id_coeff_diff_value):
ExactSolution_Step_20_Complex_Both<dim>(id_case,id_coeff_diff_value),
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value)
{}
  
template <int dim>
ExactSolution_Step_20_Complex_Imag<dim>::ExactSolution_Step_20_Complex_Imag(const unsigned int id_case, const unsigned int id_coeff_diff_value):
ExactSolution_Step_20_Complex_Both<dim>(id_case,id_coeff_diff_value),
id_case(id_case),
id_coeff_diff_value(id_coeff_diff_value)
{}


template <int dim>
void
GradientBoundary_Step_20_Complex<dim>::vector_value (const Point<dim> &p,
                                                     Vector<double>   &values) const
{
  vector<Point<dim>> vector_coords(1,Point<dim>());
  vector_coords[0] = p;
  
  vector<Tensor<2,dim, complex<double>>> vector_value_d(vector_coords.size(),Tensor<2,dim, complex<double>>());
  Coeff_Diff_Complex<dim>::value(vector_coords, vector_value_d);
  
  Tensor<1,dim,complex<double>> gradient_p;
  gradient_p = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p);
  
  Tensor<1,dim, complex<double>> coeff_diff_times_gradient;
  coeff_diff_times_gradient = vector_value_d[0]*gradient_p;
  
  if(dim==1)
  {
    values(0) = -coeff_diff_times_gradient[0].real();
    values(2) = -coeff_diff_times_gradient[0].imag();
  }else if(dim==2)
  {
    values(0) = -coeff_diff_times_gradient[0].real();                   // real -dp_x
    values(1) = -coeff_diff_times_gradient[1].real();                   // real -dp_y                                                                                              
    values(3) = -coeff_diff_times_gradient[0].imag();                   // imag -dp_x
    values(4) = -coeff_diff_times_gradient[1].imag();                   // imag -dp_y       
  }
}

template <int dim>
void
GradientBoundary_Step_20_Complex_Real<dim>::vector_value (const Point<dim> &p,                              // only for 2D
                                                          Vector<double>   &values) const
{
  Assert (2 == dim, ExcDimensionMismatch (2, dim));
  
  vector<Point<dim>> vector_coords(1,Point<dim>());
  vector_coords[0] = p;
  
  vector<Tensor<2,dim, complex<double>>> vector_value_d(vector_coords.size(),Tensor<2,dim, complex<double>>());
  Coeff_Diff_Complex<dim>::value(vector_coords, vector_value_d);
  
  Tensor<1,dim,complex<double>> gradient_p;
  gradient_p = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p);
  
  Tensor<1,dim, complex<double>> coeff_diff_times_gradient;
  coeff_diff_times_gradient = vector_value_d[0]*gradient_p;
  
  
  values(0) = -coeff_diff_times_gradient[0].real();  // real -dp_x
  values(1) = -coeff_diff_times_gradient[1].real();  // real -dp_y
}
  
template <int dim>
void
GradientBoundary_Step_20_Complex_Imag<dim>::vector_value (const Point<dim> &p,
                                                          Vector<double>   &values) const
{
  Assert (2 == dim, ExcDimensionMismatch (2, dim));
  
  vector<Point<dim>> vector_coords(1,Point<dim>());
  vector_coords[0] = p;
  
  vector<Tensor<2,dim, complex<double>>> vector_value_d(vector_coords.size(),Tensor<2,dim, complex<double>>());
  Coeff_Diff_Complex<dim>::value(vector_coords, vector_value_d);
  
  Tensor<1,dim,complex<double>> gradient_p;
  gradient_p = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p);
  
  Tensor<1,dim, complex<double>> coeff_diff_times_gradient;
  coeff_diff_times_gradient = vector_value_d[0]*gradient_p;
  
  
  values(0) = -coeff_diff_times_gradient[0].imag();  // imag -dp_x
  values(1) = -coeff_diff_times_gradient[1].imag();  // imag -dp_y
}

template <int dim>
void
ExactSolution_Step_20_Complex_Both<dim>::vector_value_both (const Point<dim> &p,
                                                       Vector<complex<double>>   &values) const
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));
  
  vector<Point<dim>> vector_coords(1,Point<dim>());
  vector_coords[0] = p;
  
  vector<Tensor<2,dim, complex<double>>> vector_value_d(vector_coords.size(),Tensor<2,dim, complex<double>>());
  Coeff_Diff_Complex<dim>::value(vector_coords, vector_value_d);  
  
  complex<double> value_p = 0.0 + 0.0i;
  value_p = ExactSolution_Step_4_Complex_Both<dim>::value_both(p);
  
  Tensor<1,dim,complex<double>> gradient_p;
  gradient_p = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p);
  
  Tensor<1,dim,complex<double>> coeff_diff_times_gradient_p;
  coeff_diff_times_gradient_p = vector_value_d[0]*gradient_p;
  
  
  if (dim == 1)
  {
    values(0) = -coeff_diff_times_gradient_p[0];
    values(1) = value_p;
  }else if(dim == 2)
  {
    values(0) = -coeff_diff_times_gradient_p[0];
    values(1) = -coeff_diff_times_gradient_p[1];
    values(2) = value_p;
  }
}

template <int dim>
void
ExactSolution_Step_20_Complex_Both<dim>::vector_gradient_both (const Point<dim> &p,
                                                               vector<Tensor<1,dim,complex<double>>>   &gradients) const
{
  Tensor<1,dim,complex<double>> gradient_p;
  gradient_p = ExactSolution_Step_4_Complex_Both<dim>::gradient_both(p);    
  Tensor<2,dim,complex<double>> hessian_p;
  hessian_p = ExactSolution_Step_4_Complex_Both<dim>::hessian_both(p);
  
//   cout << "gradient_p: " << gradient_p << "\n";
//   
//   cout << "hessian_p: " << hessian_p << "\n";
//   cout << "hessian_p[0]: " << hessian_p[0] << "\n";
//   cout << "hessian_p[1]: " << hessian_p[1] << "\n";
    
  vector<Point<dim>> vector_coords(1,Point<dim>());
  vector_coords[0] = p;
  
  vector<Tensor<2,dim, complex<double>>> vector_value_d(vector_coords.size(),Tensor<2,dim, complex<double>>());
  vector<Tensor<2,dim, complex<double>>> vector_gradient_d_x_direction(vector_coords.size(),Tensor<2,dim, complex<double>>());
  vector<Tensor<2,dim, complex<double>>> vector_gradient_d_y_direction(vector_coords.size(),Tensor<2,dim, complex<double>>());
  
  Coeff_Diff_Complex<dim>::value(vector_coords, vector_value_d);
  Coeff_Diff_Complex<dim>::gradient_x_direction(vector_coords, vector_gradient_d_x_direction);
  
  Tensor<1,dim,complex<double>> gradient_d_x_direction_times_gradient_p;
  Tensor<1,dim,complex<double>> gradient_d_y_direction_times_gradient_p;
  gradient_d_x_direction_times_gradient_p = vector_gradient_d_x_direction[0]*gradient_p;
  gradient_d_y_direction_times_gradient_p = vector_gradient_d_y_direction[0]*gradient_p;
  
  Tensor<1,dim,complex<double>> value_d_times_hessian_p_first_row;
  Tensor<1,dim,complex<double>> value_d_times_hessian_p_second_row;
  value_d_times_hessian_p_first_row = vector_value_d[0]*hessian_p[0];                       // note we assume p_xy = p_yx in this case
  value_d_times_hessian_p_second_row = vector_value_d[0]*hessian_p[1];
  
  if (dim == 1)
  {
    gradients[0][0] = -(vector_gradient_d_x_direction[0][0][0]*gradient_p[0] + vector_value_d[0][0][0]*hessian_p[0][0]);
    gradients[1][0] = gradient_p[0];
  }else if(dim == 2)
  {
    gradients[0][0] = -(gradient_d_x_direction_times_gradient_p[0] + value_d_times_hessian_p_first_row[0]);                // -(dp_x)_x
    gradients[0][1] = -(gradient_d_y_direction_times_gradient_p[0] + value_d_times_hessian_p_second_row[0]);                // -(dp_x)_y
    gradients[1][0] = -(gradient_d_x_direction_times_gradient_p[1] + value_d_times_hessian_p_first_row[0]);                // -(dp_y)_x
    gradients[1][1] = -(gradient_d_y_direction_times_gradient_p[1] + value_d_times_hessian_p_second_row[0]);                // -(dp_y)_y
    gradients[2][0] = gradient_p[0];                                                  // px
    gradients[2][1] = gradient_p[1];                                                  // py 
  }    
}

template <int dim>
void
ExactSolution_Step_20_Complex_Real<dim>::vector_value (const Point<dim> &p,
                                                       Vector<double>   &values) const
{
  Vector<complex<double>> values_both(dim+1);
  ExactSolution_Step_20_Complex_Both<dim>::vector_value_both(p, values_both);
  
  if (dim == 1)
  {
    values(0) = values_both[0].real();
    values(1) = values_both[1].real();
  }else if(dim == 2)
  {
    values(0) = values_both[0].real();
    values(1) = values_both[1].real();
    values(2) = values_both[2].real();
  }
}

template <int dim>
void
ExactSolution_Step_20_Complex_Real<dim>::vector_gradient (const Point<dim> &p,
                                                          vector<Tensor<1,dim>>   &gradients) const
{
  vector<Tensor<1,dim,complex<double>>> vector_gradient_both(dim+1,Tensor<1,dim,complex<double>>());
  ExactSolution_Step_20_Complex_Both<dim>::vector_gradient_both(p, vector_gradient_both);
  
  if (dim == 1)
  {
    gradients[0][0] = -vector_gradient_both[0][0].real();
    gradients[1][0] = vector_gradient_both[1][0].real();
  }else if(dim == 2)
  {
    gradients[0][0] = -vector_gradient_both[0][0].real();                // real -(dp_x)_x
    gradients[0][1] = -vector_gradient_both[0][1].real();                // real -(dp_x)_y
    gradients[1][0] = -vector_gradient_both[1][0].real();                // real -(dp_y)_x
    gradients[1][1] = -vector_gradient_both[1][1].real();                // real -(dp_y)_y
    gradients[2][0] = vector_gradient_both[2][0].real();                                                  // real px
    gradients[2][1] = vector_gradient_both[2][1].real();                                                  // real py 
  }
}
  
  
template <int dim>
void
ExactSolution_Step_20_Complex_Imag<dim>::vector_value (const Point<dim> &p,
                                Vector<double>   &values) const
{
  Vector<complex<double>> values_both(dim+1);
  ExactSolution_Step_20_Complex_Both<dim>::vector_value_both(p, values_both);
  
  if (dim == 1)
  {
    values(0) = values_both[0].imag();
    values(1) = values_both[1].imag();
  }else if(dim == 2)
  {
    values(0) = values_both[0].imag();
    values(1) = values_both[1].imag();
    values(2) = values_both[2].imag();
  }
}

template <int dim>
void
ExactSolution_Step_20_Complex_Imag<dim>::vector_gradient (const Point<dim> &p,
                                                        vector<Tensor<1,dim>>   &gradients) const
{
  vector<Tensor<1,dim,complex<double>>> vector_gradient_both(dim+1,Tensor<1,dim,complex<double>>());
  ExactSolution_Step_20_Complex_Both<dim>::vector_gradient_both(p, vector_gradient_both);
  
  if (dim == 1)
  {
    gradients[0][0] = -vector_gradient_both[0][0].imag();
    gradients[1][0] = vector_gradient_both[1][0].imag();
  }else if(dim == 2)
  {
    gradients[0][0] = -vector_gradient_both[0][0].imag();                // imag -(dp_x)_x
    gradients[0][1] = -vector_gradient_both[0][1].imag();                // imag -(dp_x)_y
    gradients[1][0] = -vector_gradient_both[1][0].imag();                // imag -(dp_y)_x
    gradients[1][1] = -vector_gradient_both[1][1].imag();                // imag -(dp_y)_y
    gradients[2][0] = vector_gradient_both[2][0].imag();                                                  // imag px
    gradients[2][1] = vector_gradient_both[2][1].imag();                                                  // imag py 
  }
}


#endif


