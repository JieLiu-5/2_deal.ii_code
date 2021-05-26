DEAL_II_NAMESPACE_OPEN

template<>
void AffineConstraints<double>::adjust_constraints (AffineConstraints &constraints)
{
    std::cout << "  adjusting constraints bassed on another constraint matrix\n";
      
    for (size_type i=0; i!=lines.size(); ++i)
    {
        if (lines[i].inhomogeneity == 0)
        {
            lines[i].inhomogeneity = constraints.lines[i].inhomogeneity;
        }
    }
}

template<>
void AffineConstraints<double>::adjust_constraints()
{
    std::cout << "    adjusting constraints customly\n";
    
    for (size_type i=0; i!=lines.size(); ++i)
    {
        lines[i].inhomogeneity = lines[i].inhomogeneity*2.0;
        std::cout << "    [" << i << "] " << lines[i].inhomogeneity << "\n";
    }
    
}


DEAL_II_NAMESPACE_CLOSE
