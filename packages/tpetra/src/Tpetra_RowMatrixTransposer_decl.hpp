
//@HEADER
/*
************************************************************************

              Tpetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef TPETRA_ROWMATRIXTRANSPOSER_HPP
#define TPETRA_ROWMATRIXTRANSPOSER_HPP
#include <Teuchos_RCP.hpp>
#include <Kokkos_DefaultNode.hpp>
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_ConfigDefs.hpp"

template <class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = Kokkos::DefaultNode::DefaultNodeType>
class Map;

//! Tpetra_RowMatrixTransposer: A class for transposing an Tpetra_RowMatrix object.

namespace Tpetra {
/*! This class provides capabilities to construct a transpose matrix of an existing Tpetra_RowMatrix
	  object and (optionally) redistribute it across a parallel distributed memory machine.
*/

template <class Scalar, class LocalOrdinal = int, class GlobalOrdinal = LocalOrdinal, class Node = Kokkos::DefaultNode::DefaultNodeType>
class RowMatrixTransposer {
    
  public:

  //! @name Constructors/destructors
  //@{ 
  //! Primary Tpetra_RowMatrixTransposer constructor.
  /*!
    \param origMatrix An existing Tpetra_RowMatrix object.  The Tpetra_RowMatrix, the LHS and RHS pointers
		       do not need to be defined before this constructor is called.

    \return Pointer to a Tpetra_RowMatrixTransposer object.

  */ 
  RowMatrixTransposer(const Teuchos::RCP<const RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > origMatrix);

  //! Tpetra_RowMatrixTransposer destructor.
  
  virtual ~RowMatrixTransposer();
  //@}
  
  //! @name Forward transformation methods
  //@{ 
  
  //! Generate a new Tpetra_CrsMatrix as the transpose of an Tpetra_RowMatrix passed into the constructor.
  /*! Constructs a new Tpetra_CrsMatrix that is a copy of the Tpetra_RowMatrix passed in to the constructor.
		
		\param optimizeTranspose Optimizes the storage of the newly created Transpose matrix
		\param transposeMatrix The matrix in which the result of the tranpose operation will be put.
		\param TransposeRowMap If this argument is defined, the transpose matrix will be distributed
		       using this map as the row map for the transpose.	If null, the function will evenly distribute
			   the rows of the tranpose matrix.
  */
  void createTranspose(const OptimizeOption optimizeTranspose, Teuchos::RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &transposeMatrix, Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > transposeRowMap = Teuchos::null);

	
 private: 
	//The original matrix to be transposed.
	const Teuchos::RCP<const RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > origMatrix_;
	//The matrix in which the result of the tranpose is placed.
	Teuchos::RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > transposeMatrix_;
	//Whether or not to optimize the storage of the transpose matrix.
	OptimizeOption optimizeTranspose_;	
	const Teuchos::RCP<const Teuchos::Comm<int> > comm_;
	GlobalOrdinal indexBase_;
};


}

#endif /* TPETRA_ROWMATRIXTRANSPOSER_DECL_HPP */