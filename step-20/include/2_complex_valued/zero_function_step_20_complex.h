
#ifndef ZERO_FUNCTION_STEP_20_COMPLEX_H
#define ZERO_FUNCTION_STEP_20_COMPLEX_H

template <int dim>
class Zero_Function_Custom : public Function<dim>
{
  public:
    Zero_Function_Custom () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    virtual void vector_gradient (const Point<dim> &p,
                               vector<Tensor<1,dim>>   &gradients) const;
};
  
template <int dim>
void
Zero_Function_Custom<dim>::vector_value (const Point<dim> &/*p*/,
                                  Vector<double>   &values) const
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));
  values(0) = 0.0;
  values(1) = 0.0;
}

template <int dim>
void
Zero_Function_Custom<dim>::vector_gradient (const Point<dim> &/*p*/,
                                          vector<Tensor<1,dim>>   &gradients) const
{
  gradients[0][0] = 0.0;
  gradients[1][0] = 0.0;
}

#endif


