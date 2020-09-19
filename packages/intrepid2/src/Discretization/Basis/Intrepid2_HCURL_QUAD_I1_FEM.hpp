// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid2_HCURL_QUAD_I1_FEM.hpp
    \brief  Header file for the Intrepid2::Basis_HCURL_QUAD_I1_FEM class.
    \author Created by P. Bochev and D. Ridzal and K. Peterson.
            Kokkorized by Kyungjoo Kim
 */

#ifndef __INTREPID2_HCURL_QUAD_I1_FEM_HPP__
#define __INTREPID2_HCURL_QUAD_I1_FEM_HPP__

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_HVOL_C0_FEM.hpp"

namespace Intrepid2 {

  /** \class  Intrepid2::Basis_HCURL_QUAD_I1_FEM
      \brief  Implementation of the default H(curl)-compatible FEM basis of degree 1 on Quadrilateral cell

      Implements Nedelec basis of the first kind of degree 1 on the reference Quadrilateral cell.
      The basis has cardinality 4 and spans an INCOMPLETE bi-linear polynomial space. Basis functions
      are dual to a unisolvent set of degrees-of-freedom (DoF) defined and enumerated as follows:

      \verbatim
      ===================================================================================================
      |         |           degree-of-freedom-tag table                    |                            |
      |   DoF   |----------------------------------------------------------|       DoF definition       |
      | ordinal |  subc dim    | subc ordinal | subc DoF ord |subc num DoF |                            |
      |=========|==============|==============|==============|=============|============================|
      |    0    |       1      |       0      |       0      |      1      |   L_0(u) = (u.t)( 0,-1)    |
      |---------|--------------|--------------|--------------|-------------|----------------------------|
      |    1    |       1      |       1      |       0      |      1      |   L_1(u) = (u.t)( 1, 0)    |
      |---------|--------------|--------------|--------------|-------------|----------------------------|
      |    2    |       1      |       2      |       0      |      1      |   L_2(u) = (u.t)( 0, 1)    |
      |---------|--------------|--------------|--------------|-------------|----------------------------|
      |    3    |       1      |       3      |       0      |      1      |   L_3(u) = (u.t)(-1, 0)    |
      |=========|==============|==============|==============|=============|============================|
      |   MAX   |  maxScDim=1  |  maxScOrd=3  |  maxDfOrd=0  |      -      |                            |
      |=========|==============|==============|==============|=============|============================|
      \endverbatim

      \remarks
      \li     In the DoF functional \f${\bf t}\f$ is an edge tangent. Direction of edge
      tangents follows the vertex order of the edges in the cell topology and runs from
      edge vertex 0 to edge vertex 1, whereas their length is set equal to the edge length. For
      example, edge 1 of all Quadrilateral reference cells has vertex order {1,2}, i.e., its tangent
      runs from vertex 1 of the reference Quadrilateral to vertex 2 of that cell. On the reference
      Quadrilateral the coordinates of these vertices are (1,-1) and (1,1), respectively. Therefore,
      the tangent to edge 1 is (1,1) - (1,-1) = (0,2). Because its length already equals edge length,
      no further rescaling of the edge tangent is needed.

      \li     The length of the edge tangent equals the edge length. As a result, the DoF functional
      is the value of the tangent component of a vector field at the edge midpoint times the
      edge length. The resulting basis is equivalent to a basis defined by using the edge
      circulation as a DoF functional. Note that all edges of reference Quadrilateral<> cells
      have length 2.

  */

  namespace Impl {

    /**
      \brief See Intrepid2::Basis_HCURL_QUAD_I1_FEM
    */
    class Basis_HCURL_QUAD_I1_FEM {
    public:
      typedef struct Quadrilateral<4> cell_topology_type;

      /**
        \brief See Intrepid2::Basis_HCURL_QUAD_I1_FEM
      */
      template<EOperator opType>
      struct Serial {
        template<typename OutputViewType,
                 typename inputViewType>
        KOKKOS_INLINE_FUNCTION
        static void
        getValues(       OutputViewType output,
                   const inputViewType input );

      };

      template<typename ExecSpaceType,
               typename outputValueValueType, class ...outputValueProperties,
               typename inputPointValueType,  class ...inputPointProperties>
      static void
      getValues(       Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                 const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
                 const EOperator operatorType);

      /**
        \brief See Intrepid2::Basis_HCURL_QUAD_I1_FEM
      */
      template<typename outputValueViewType,
               typename inputPointViewType,
               EOperator opType>
      struct Functor {
              outputValueViewType _outputValues;
        const inputPointViewType  _inputPoints;

        KOKKOS_INLINE_FUNCTION
        Functor(      outputValueViewType outputValues_,
                      inputPointViewType inputPoints_)
          : _outputValues(outputValues_), _inputPoints(inputPoints_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_type pt) const {
          switch (opType) {
          case OPERATOR_VALUE : {
            auto       output = Kokkos::subview( _outputValues, Kokkos::ALL(), pt, Kokkos::ALL() );
            const auto input  = Kokkos::subview( _inputPoints,                 pt, Kokkos::ALL() );
            Serial<opType>::getValues( output, input );
            break;
          }

          case OPERATOR_CURL: {
            auto       output = Kokkos::subview( _outputValues, Kokkos::ALL(), pt );
            const auto input  = Kokkos::subview( _inputPoints,                 pt, Kokkos::ALL() );
            Serial<opType>::getValues( output, input );
            break;
          }
          default: {
            INTREPID2_TEST_FOR_ABORT( opType != OPERATOR_VALUE &&
                                      opType != OPERATOR_CURL,
                                      ">>> ERROR: (Intrepid2::Basis_HCURL_QUAD_CI_FEM::Serial::getVAlues) operator is not supported");
            break;
          }
          } //end switch
        }

      };
    };
  }

  template<typename ExecSpaceType = void,
           typename outputValueType = double,
           typename pointValueType = double>
  class Basis_HCURL_QUAD_I1_FEM : public Basis<ExecSpaceType,outputValueType, pointValueType> {
  public:
    using OrdinalTypeArray1DHost = typename Basis<ExecSpaceType,outputValueType,pointValueType>::OrdinalTypeArray1DHost;
    using OrdinalTypeArray2DHost = typename Basis<ExecSpaceType,outputValueType,pointValueType>::OrdinalTypeArray2DHost;
    using OrdinalTypeArray3DHost = typename Basis<ExecSpaceType,outputValueType,pointValueType>::OrdinalTypeArray3DHost;


    /** \brief  Constructor.
     */
    Basis_HCURL_QUAD_I1_FEM();

    using OutputViewType = typename Basis<ExecSpaceType,outputValueType,pointValueType>::OutputViewType;
    using PointViewType  = typename Basis<ExecSpaceType,outputValueType,pointValueType>::PointViewType;
    using ScalarViewType = typename Basis<ExecSpaceType,outputValueType,pointValueType>::ScalarViewType;

    using Basis<ExecSpaceType,outputValueType,pointValueType>::getValues;

    virtual
    void
    getValues(       OutputViewType outputValues,
               const PointViewType  inputPoints,
               const EOperator operatorType = OPERATOR_VALUE ) const {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify arguments
      Intrepid2::getValues_HCURL_Args(outputValues,
                                      inputPoints,
                                      operatorType,
                                      this->getBaseCellTopology(),
                                      this->getCardinality() );
#endif
      Impl::Basis_HCURL_QUAD_I1_FEM::
        getValues<ExecSpaceType>( outputValues,
                                  inputPoints,
                                  operatorType );
    }

    virtual
    void
    getDofCoords( ScalarViewType dofCoords ) const {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.rank() != 2, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HCURL_QUAD_I1_FEM::getDofCoords) rank = 2 required for dofCoords array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->basisCardinality_, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HCURL_QUAD_I1_FEM::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      // Verify 1st dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->basisCellTopology_.getDimension(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HCURL_QUAD_I1_FEM::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }
    
  virtual
  void
  getDofCoeffs( ScalarViewType dofCoeffs ) const {
#ifdef HAVE_INTREPID2_DEBUG
    // Verify rank of output array.
    INTREPID2_TEST_FOR_EXCEPTION( dofCoeffs.rank() != 2, std::invalid_argument,
        ">>> ERROR: (Intrepid2::Basis_HCURL_QUAD_I1_FEM::getDofCoeffs) rank = 2 required for dofCoeffs array");
    // Verify 0th dimension of output array.
    INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoeffs.extent(0)) != this->getCardinality(), std::invalid_argument,
        ">>> ERROR: (Intrepid2::Basis_HCURL_QUAD_I1_FEM::getDofCoeffs) mismatch in number of dof and 0th dimension of dofCoeffs array");
    // Verify 1st dimension of output array.
    INTREPID2_TEST_FOR_EXCEPTION( dofCoeffs.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
        ">>> ERROR: (Intrepid2::Basis_HCURL_QUAD_I1_FEM::getDofCoeffs) incorrect reference cell (1st) dimension in dofCoeffs array");
#endif
    Kokkos::deep_copy(dofCoeffs, this->dofCoeffs_);
  }

    virtual
    const char*
    getName() const {
      return "Intrepid2_HCURL_QUAD_I1_FEM";
    }

    virtual
    bool
    requireOrientation() const {
      return true;
    }

    /** \brief returns the basis associated to a subCell.

        The bases of the subCell are the restriction to the subCell of the bases of the parent cell,
        projected to the subCell line.

        \param [in] subCellDim - dimension of subCell
        \param [in] subCellOrd - position of the subCell among of the subCells having the same dimension
        \return pointer to the subCell basis of dimension subCellDim and position subCellOrd
     */
    BasisPtr<ExecSpaceType,outputValueType,pointValueType>
    getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override{
      if(subCellDim == 1)
        return Teuchos::rcp( new
          Basis_HVOL_C0_FEM<ExecSpaceType,outputValueType,pointValueType>(shards::getCellTopologyData<shards::Line<2> >()));

      INTREPID2_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Input parameters out of bounds");
    }

  };
}// namespace Intrepid2

#include "Intrepid2_HCURL_QUAD_I1_FEMDef.hpp"

#endif
