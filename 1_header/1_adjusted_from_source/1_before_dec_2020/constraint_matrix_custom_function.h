DEAL_II_NAMESPACE_OPEN

void ConstraintMatrix::adjust_constraints (ConstraintMatrix &constraints)
{
    std::cout << "    replacing the zero constraints components by the corresponding components of the argument constraints... ";
      
    for (size_type i=0; i!=lines.size(); ++i)
    {
        if (lines[i].inhomogeneity == 0)
        {
            lines[i].inhomogeneity = constraints.lines[i].inhomogeneity;
        }
    }
    std::cout << "-> done" << std::endl;  

}


DEAL_II_NAMESPACE_CLOSE
