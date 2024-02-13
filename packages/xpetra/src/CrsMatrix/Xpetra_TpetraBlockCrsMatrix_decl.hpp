// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef XPETRA_TPETRABLOCKCRSMATRIX_DECL_HPP
#define XPETRA_TPETRABLOCKCRSMATRIX_DECL_HPP

/* this file is automatically generated - do not edit (see script/tpetra.py) */

#include "Xpetra_TpetraConfigDefs.hpp"

#include "Tpetra_BlockCrsMatrix.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_TpetraMap_decl.hpp"
#include "Xpetra_TpetraMultiVector_decl.hpp"
#include "Xpetra_TpetraVector_decl.hpp"
#include "Xpetra_TpetraCrsGraph_decl.hpp"
#include "Xpetra_Exceptions.hpp"

namespace Xpetra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
class TpetraBlockCrsMatrix
  : public CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>  //, public TpetraRowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>
{
  // The following typedef are used by the XPETRA_DYNAMIC_CAST() macro.
  typedef TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraBlockCrsMatrixClass;
  typedef TpetraVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraVectorClass;
  typedef TpetraImport<LocalOrdinal, GlobalOrdinal, Node> TpetraImportClass;
  typedef TpetraExport<LocalOrdinal, GlobalOrdinal, Node> TpetraExportClass;

 public:
  //! @name Constructor/Destructor Methods

  //! Constructor specifying fixed number of entries for each row (not implemented)
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap,
                       size_t maxNumEntriesPerRow,
                       const Teuchos::RCP<Teuchos::ParameterList> &params = Teuchos::null);

  //! Constructor specifying (possibly different) number of entries in each row (not implemented)
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap,
                       const ArrayRCP<const size_t> &NumEntriesPerRowToAlloc,
                       const Teuchos::RCP<Teuchos::ParameterList> &params = Teuchos::null);

  //! Constructor specifying column Map and fixed number of entries for each row (not implemented)
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &colMap,
                       size_t maxNumEntriesPerRow,
                       const Teuchos::RCP<Teuchos::ParameterList> &params = Teuchos::null);

  //! Constructor specifying column Map and number of entries in each row (not implemented)
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &colMap,
                       const ArrayRCP<const size_t> &NumEntriesPerRowToAlloc,
                       const Teuchos::RCP<Teuchos::ParameterList> &params = Teuchos::null);

  //! Constructor specifying a previously constructed graph ( not implemented )
  TpetraBlockCrsMatrix(const Teuchos::RCP<const CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > &graph,
                       const Teuchos::RCP<Teuchos::ParameterList> &params = Teuchos::null);

  //! Constructor specifying a previously constructed graph & blocksize
  TpetraBlockCrsMatrix(const Teuchos::RCP<const CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > &graph,
                       const LocalOrdinal blockSize);

  //! Constructor specifying a previously constructed graph, point maps & blocksize
  TpetraBlockCrsMatrix(const Teuchos::RCP<const CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > &graph,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &pointDomainMap,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &pointRangeMap,
                       const LocalOrdinal blockSize);

  //! Constructor for a fused import ( not implemented )
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &sourceMatrix,
                       const Import<LocalOrdinal, GlobalOrdinal, Node> &importer,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &domainMap = Teuchos::null,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rangeMap  = Teuchos::null,
                       const Teuchos::RCP<Teuchos::ParameterList> &params                           = Teuchos::null);

  //! Constructor for a fused export ( not implemented )
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &sourceMatrix,
                       const Export<LocalOrdinal, GlobalOrdinal, Node> &exporter,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &domainMap = Teuchos::null,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rangeMap  = Teuchos::null,
                       const Teuchos::RCP<Teuchos::ParameterList> &params                           = Teuchos::null);

  //! Constructor for a fused import ( not implemented )
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &sourceMatrix,
                       const Import<LocalOrdinal, GlobalOrdinal, Node> &RowImporter,
                       const Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Node> > DomainImporter,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &domainMap,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rangeMap,
                       const Teuchos::RCP<Teuchos::ParameterList> &params);

  //! Constructor for a fused export ( not implemented )
  TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &sourceMatrix,
                       const Export<LocalOrdinal, GlobalOrdinal, Node> &RowExporter,
                       const Teuchos::RCP<const Export<LocalOrdinal, GlobalOrdinal, Node> > DomainExporter,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &domainMap,
                       const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rangeMap,
                       const Teuchos::RCP<Teuchos::ParameterList> &params);

  //! Destructor.
  virtual ~TpetraBlockCrsMatrix();

  //! @name Insertion/Removal Methods

  //! Insert matrix entries, using global IDs (not implemented)
  void insertGlobalValues(GlobalOrdinal globalRow,
                          const ArrayView<const GlobalOrdinal> &cols,
                          const ArrayView<const Scalar> &vals);

  //! Insert matrix entries, using local IDs (not implemented)
  void insertLocalValues(LocalOrdinal localRow,
                         const ArrayView<const LocalOrdinal> &cols,
                         const ArrayView<const Scalar> &vals);

  //! Replace matrix entries, using global IDs (not implemented)
  void replaceGlobalValues(GlobalOrdinal globalRow,
                           const ArrayView<const GlobalOrdinal> &cols,
                           const ArrayView<const Scalar> &vals);

  //! Replace matrix entries, using local IDs.
  void replaceLocalValues(LocalOrdinal localRow,
                          const ArrayView<const LocalOrdinal> &cols,
                          const ArrayView<const Scalar> &vals);

  //! Set all matrix entries equal to scalarThis.
  void setAllToScalar(const Scalar &alpha);

  //! Scale the current values of a matrix, this = alpha*this (not implemented)
  void scale(const Scalar &alpha);

  //! Allocates and returns ArrayRCPs of the Crs arrays --- This is an Xpetra-only routine.
  //** \warning This is an expert-only routine and should not be called from user code. (not implemented)
  void allocateAllValues(size_t numNonZeros, ArrayRCP<size_t> &rowptr,
                         ArrayRCP<LocalOrdinal> &colind,
                         ArrayRCP<Scalar> &values);

  //! Sets the 1D pointer arrays of the graph (not impelmented)
  void setAllValues(const ArrayRCP<size_t> &rowptr,
                    const ArrayRCP<LocalOrdinal> &colind,
                    const ArrayRCP<Scalar> &values);

  //! Gets the 1D pointer arrays of the graph (not implemented)
  void getAllValues(ArrayRCP<const size_t> &rowptr,
                    ArrayRCP<const LocalOrdinal> &colind,
                    ArrayRCP<const Scalar> &values) const;

  //! Gets the 1D pointer arrays of the graph (not implemented)
  void getAllValues(ArrayRCP<Scalar> &values);

  //! @name Transformational Methods

  //!
  void resumeFill(const RCP<ParameterList> &params = null);

  //! Signal that data entry is complete, specifying domain and range maps.
  void fillComplete(const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &domainMap,
                    const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rangeMap, const RCP<ParameterList> &params = null);

  //! Signal that data entry is complete.
  void fillComplete(const RCP<ParameterList> &params = null);

  //!  Replaces the current domainMap and importer with the user-specified objects.
  void replaceDomainMapAndImporter(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &newDomainMap,
                                   Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Node> > &newImporter);

  //! Expert static fill complete
  void expertStaticFillComplete(const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &domainMap,
                                const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rangeMap,
                                const RCP<const Import<LocalOrdinal, GlobalOrdinal, Node> > &importer = Teuchos::null,
                                const RCP<const Export<LocalOrdinal, GlobalOrdinal, Node> > &exporter = Teuchos::null,
                                const RCP<ParameterList> &params                                      = Teuchos::null);

  //! @name Methods implementing RowMatrix

  //! Returns the Map that describes the row distribution in this matrix.
  const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getRowMap() const;

  //! Returns the Map that describes the column distribution in this matrix.
  const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getColMap() const;

  //! Returns the CrsGraph associated with this matrix.
  RCP<const CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > getCrsGraph() const;

  //! Number of global elements in the row map of this matrix.
  global_size_t getGlobalNumRows() const;

  //! Number of global columns in the matrix.
  global_size_t getGlobalNumCols() const;

  //! Returns the number of matrix rows owned on the calling node.
  size_t getLocalNumRows() const;

  //! Returns the number of columns connected to the locally owned rows of this matrix.
  size_t getLocalNumCols() const;

  //! Returns the global number of entries in this matrix.
  global_size_t getGlobalNumEntries() const;

  //! Returns the local number of entries in this matrix.
  size_t getLocalNumEntries() const;

  //! Returns the current number of entries on this node in the specified local row.
  size_t getNumEntriesInLocalRow(LocalOrdinal localRow) const;

  //! Returns the current number of entries in the (locally owned) global row.
  size_t getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const;

  //! Returns the maximum number of entries across all rows/columns on all nodes.
  size_t getGlobalMaxNumRowEntries() const;

  //! Returns the maximum number of entries across all rows/columns on this node.
  size_t getLocalMaxNumRowEntries() const;

  //! If matrix indices are in the local range, this function returns true. Otherwise, this function returns false.
  bool isLocallyIndexed() const;

  //! If matrix indices are in the global range, this function returns true. Otherwise, this function returns false.
  bool isGloballyIndexed() const;

  //! Returns true if the matrix is in compute mode, i.e. if fillComplete() has been called.
  bool isFillComplete() const;

  //! Returns true if the matrix is in edit mode.
  bool isFillActive() const;

  //! Returns the Frobenius norm of the matrix.
  typename ScalarTraits<Scalar>::magnitudeType getFrobeniusNorm() const;

  //! Returns true if getLocalRowView() and getGlobalRowView() are valid for this class.
  bool supportsRowViews() const;

  //! Extract a list of entries in a specified local row of the matrix. Put into storage allocated by calling routine.
  void getLocalRowCopy(LocalOrdinal LocalRow, const ArrayView<LocalOrdinal> &Indices, const ArrayView<Scalar> &Values, size_t &NumEntries) const;

  //! Extract a const, non-persisting view of global indices in a specified row of the matrix.
  void getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView<const GlobalOrdinal> &indices, ArrayView<const Scalar> &values) const;

  //! Extract a list of entries in a specified global row of this matrix. Put into pre-allocated storage.
  void getGlobalRowCopy(GlobalOrdinal GlobalRow, const ArrayView<GlobalOrdinal> &indices, const ArrayView<Scalar> &values, size_t &numEntries) const;

  //! Extract a const, non-persisting view of local indices in a specified row of the matrix.
  void getLocalRowView(LocalOrdinal LocalRow, ArrayView<const LocalOrdinal> &indices, ArrayView<const Scalar> &values) const;

  //! Returns true if globalConstants have been computed; false otherwise
  virtual bool haveGlobalConstants() const;

  //! @name Methods implementing Operator

  //! Computes the sparse matrix-multivector multiplication.
  void apply(const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X, MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS, Scalar alpha = ScalarTraits<Scalar>::one(), Scalar beta = ScalarTraits<Scalar>::zero()) const;

  //! Computes the matrix-multivector multiplication for region layout matrices (currently no block implementation)
  void apply(const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X, MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Y, Teuchos::ETransp mode, Scalar alpha, Scalar beta, bool sumInterfaceValues, const RCP<Import<LocalOrdinal, GlobalOrdinal, Node> > &regionInterfaceImporter, const Teuchos::ArrayRCP<LocalOrdinal> &regionInterfaceLIDs) const;

  //! Returns the Map associated with the domain of this operator. This will be null until fillComplete() is called.
  const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getDomainMap() const;

  //!
  const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getRangeMap() const;

  //! @name Overridden from Teuchos::Describable

  //! A simple one-line description of this object.
  std::string description() const;

  //! Print the object with some verbosity level to an FancyOStream object.
  void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const;

  //! @name Overridden from Teuchos::LabeledObject
  void setObjectLabel(const std::string &objectLabel);

  //! Get a copy of the diagonal entries owned by this node, with local row idices
  void getLocalDiagCopy(Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &diag) const;

  //! Get a copy of the diagonal entries owned by this node, with local row indices.
  void getLocalDiagCopy(Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &diag,
                        const Teuchos::ArrayView<const size_t> &offsets) const;

  //! Get offsets of the diagonal entries in the matrix.
  void getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const;

  //! Get a copy of the diagonal entries owned by this node, with local row indices, using row offsets.
  void getLocalDiagCopy(Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &diag, const Kokkos::View<const size_t *, typename Node::device_type, Kokkos::MemoryUnmanaged> &offsets) const;

  void replaceDiag(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &diag);

  //! Left scale operator with given vector values
  void leftScale(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &x);

  //! Right scale operator with given vector values
  void rightScale(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &x);

  //! Implements DistObject interface

  //! Access function for the Tpetra::Map this DistObject was constructed with.
  Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getMap() const;

  //! Import.
  void doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                const Import<LocalOrdinal, GlobalOrdinal, Node> &importer, CombineMode CM);

  //! Export.
  void doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                const Import<LocalOrdinal, GlobalOrdinal, Node> &importer, CombineMode CM);

  //! Import (using an Exporter).
  void doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                const Export<LocalOrdinal, GlobalOrdinal, Node> &exporter, CombineMode CM);

  //! Export (using an Importer).
  void doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                const Export<LocalOrdinal, GlobalOrdinal, Node> &exporter, CombineMode CM);

  void removeEmptyProcessesInPlace(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &newMap);

  //! @name Xpetra specific

  //! Does this have an underlying matrix
  bool hasMatrix() const;

  //! TpetraBlockCrsMatrix constructor to wrap a Tpetra::BlockCrsMatrix object
  TpetraBlockCrsMatrix(const Teuchos::RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &mtx);

  //! Get the underlying Tpetra matrix
  RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_BlockCrsMatrix() const;

  //! Get the underlying Tpetra matrix
  RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_BlockCrsMatrixNonConst() const;

#ifdef HAVE_XPETRA_TPETRA
  // using local_matrix_type = typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type;
  using local_matrix_type = typename CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type;

  local_matrix_type getLocalMatrixDevice() const;
  typename local_matrix_type::HostMirror getLocalMatrixHost() const;

  void setAllValues(const typename local_matrix_type::row_map_type &ptr,
                    const typename local_matrix_type::StaticCrsGraphType::entries_type::non_const_type &ind,
                    const typename local_matrix_type::values_type &val);
#endif  // HAVE_XPETRA_TPETRA

  //! Returns the block size of the storage mechanism
  LocalOrdinal GetStorageBlockSize() const { return mtx_->getBlockSize(); }

  //! Compute a residual R = B - (*this) * X
  void residual(const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X,
                const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &B,
                MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &R) const {
    using STS = Teuchos::ScalarTraits<Scalar>;
    R.update(STS::one(), B, STS::zero());
    this->apply(X, R, Teuchos::NO_TRANS, -STS::one(), STS::one());
  }

 private:
  RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > mtx_;

};  // TpetraBlockCrsMatrix class

}  // namespace Xpetra

#define XPETRA_TPETRABLOCKCRSMATRIX_SHORT
#endif  // XPETRA_TPETRABLOCKCRSMATRIX_DECL_HPP
