



struct GridFunc_1D                                 // class for the mesh distortion
{
    
public:
    
  GridFunc_1D(unsigned int id_regular_grid_arg)
  {
    id_regular_grid = id_regular_grid_arg;
  }
    
  double trans(const double y) const
  {
      
    switch(id_regular_grid)
    {
      case 2:
      
        if(y<0.5)
        {
          return y/2.0;
        }else if(y>0.5)
        {
          return (y+1.0)/2.0;  
        }
        break;
      case 3:
        if(y<0.5)
        {
          return y/(1.5-y);
        }else if(y>0.5)
        {
          return 1.0 - (1.0 - y)/(0.5+y);  
        }
        break;
      default:
        break;
    }
    
    return y;
  }
  
  Point<1,double> operator()(const Point<1,double> &in) const
  {
    return Point<1,double>(trans(in(0)));
  }
  
private:
    
  unsigned int id_regular_grid;
    
};

#if 0
template <int dim>
void distorting_the_2d_mesh_like_a_sine_function(Triangulation<dim> &triangulation)
{
    
  GridTools::transform(
    [](const Point<2> &in) {
      return Point<2>(in[0], in[1] + std::sin(numbers::PI * in[0] / 5.0));
    },
    triangulation);
//   print_mesh_info(triangulation, "grid-5.vtu");
}


template <int dim>
void distorting_the_2d_mesh_moving_top_vertices_upwards(const Triangulation<dim> &triangulation)
{
      for (const auto &cell : triangulation.active_cell_iterators())
      {
        for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
        {
          Point<2> &v = cell->vertex(i);
          if (std::abs(v(1) - 1.0) < 1e-5)
          {
                v(1) += 0.5;
          }
        }
      }
}

#endif
