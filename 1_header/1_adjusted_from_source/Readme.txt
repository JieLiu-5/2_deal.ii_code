
(Only for back-up purpose)
We have the following custom functions for deal.ii (Nov. 19, 2020):


1. /base

1.1 timer.h



2. /numerics

2.1 vector_tools_common.h
   We added the following terms on the second derivatives on January 18, 2021:
        H2_divnorm
        H2_seminorm_00
        H2_seminorm_01
        H2_seminorm_10
        H2_seminorm_11
   They also involve the change in vector_tools_integrate_difference.templates.h

2.2 vector_tools_integrate_difference.templates.h
        "H2_seminorm" appears 3 times in integrate_difference_inner()
                              2 times in do_integrate_difference()
                              1 time in compute_global_error()
                            
        "function_hessians" is defined and resized in IDScratchData and
                               initialized in do_integrate_difference()
        "psi_hessians" is defined and resized in IDScratchData and 
                          initialized in integrate_difference_inner()
             

2.3 vector_tools_boundary.h
    We added the functions for dealing with the Neumann boundary condition when using BDM elements. The functions are implemented in 2.3.1
    
2.3.1. vector_tools_boundary.templates.h


3. /lac

3.1. solver_cg1.h

3.2. affine_constraints.h

3.2.1 affine_constraints_custom_function.h


    

