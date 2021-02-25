
#ifndef SCHUR_COMPLEMENT_STEP_20_H
#define SCHUR_COMPLEMENT_STEP_20_H
  

template <class MatrixType>
class InverseMatrix_CG : public Subscriptor
{
  public:
    InverseMatrix_CG(const MatrixType &m);

    void vmult(Vector<double>       &dst,
              const Vector<double> &src) const;

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
    
  SolverControl solver_control (2000, 1e-12);                                         // here the PreconditionIdentity is used, June 21, 2019
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
  :matrix (&m)
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


#endif


