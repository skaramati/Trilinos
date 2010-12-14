//@HEADER
// ************************************************************************
// 
//               Tpetra: Linear Algebra Services Package 
//                 Copyright 2011 Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#ifndef __MatrixMarket_raw_hpp
#define __MatrixMarket_raw_hpp

#include "MatrixMarket_Banner.hpp"
#include "MatrixMarket_CoordDataReader.hpp"
#include "MatrixMarket_util.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <stdexcept>

namespace Tpetra {
  namespace MatrixMarket {
    namespace Raw {

      /// \class Element
      /// \author Mark Hoemmen
      /// \brief One structural nonzero of a sparse matrix
      ///
      template<class Scalar, class Ordinal>
      class Element {
      public:
	//! Default constructor: an invalid structural nonzero element
	Element () : rowIndex_ (-1), colIndex_ (-1), value_ (0) {}

	//! A structural nonzero element at (i,j) with value Aij
	Element (const Ordinal i, const Ordinal j, const Scalar& Aij) :
	  rowIndex_ (i), colIndex_ (j), value_ (Aij) {}

	//! Ignore the nonzero value for comparisons.
	bool operator== (const Element& rhs) {
	  return rowIndex_ == rhs.rowIndex_ && colIndex_ == rhs.colIndex_;
	}

	//! Ignore the nonzero value for comparisons.
	bool operator!= (const Element& rhs) {
	  return ! (*this == rhs);
	}

	//! Lex order first by row index, then by column index.
	bool operator< (const Element& rhs) const {
	  if (rowIndex_ < rhs.rowIndex_)
	    return true;
	  else if (rowIndex_ > rhs.rowIndex_)
	    return false;
	  else { // equal
	    return colIndex_ < rhs.colIndex_;
	  }
	}

	void merge (const Element& rhs, const bool replace=false) {
	  if (*this != rhs)
	    throw std::logic_error("Can only merge elements at the same "
				   "location in the sparse matrix");
	  else if (replace)
	    value_ = rhs.value_;
	  else
	    value_ += rhs.value_;
	}

	Ordinal rowIndex() const { return rowIndex_; }
	Ordinal colIndex() const { return colIndex_; }
	Scalar value() const { return value_; }

      private:
	Ordinal rowIndex_, colIndex_;
	Scalar value_;
      };

      //! Print out an Element to the given output stream
      template<class Scalar, class Ordinal>
      std::ostream& 
      operator<< (std::ostream& out, const Element<Scalar, Ordinal>& elt) 
      {
	typedef Teuchos::ScalarTraits<Scalar> STS;
	// Non-Ordinal types are floating-point types.  In order not to
	// lose information when we print a floating-point type, we have
	// to set the number of digits to print.  C++ standard behavior
	// in the default locale seems to be to print only five decimal
	// digits after the decimal point; this does not suffice for
	// double precision.  We solve the problem of how many digits to
	// print more generally below.  It's a rough solution so please
	// feel free to audit and revise it.
	//
	// FIXME (mfh 01 Feb 2011) 
	// This really calls for the following approach:
	//
	// Guy L. Steele and Jon L. White, "How to print floating-point
	// numbers accurately", 20 Years of the ACM/SIGPLAN Conference
	// on Programming Language Design and Implementation
	// (1979-1999): A Selection, 2003.
	if (! STS::isOrdinal)
	  {
	    // std::scientific, std::fixed, and default are the three
	    // output states for floating-point numbers.  A reasonable
	    // user-defined floating-point type should respect these
	    // flags; hopefully it does.
	    out << std::scientific;

	    // Decimal output is standard for Matrix Market format.
	    out << std::setbase (10);

	    // Compute the number of decimal digits required for expressing
	    // a Scalar, by comparing with IEEE 754 double precision (16
	    // decimal digits, 53 binary digits).  This would be easier if
	    // Teuchos exposed std::numeric_limits<T>::digits10, alas.
	    const double numDigitsAsDouble = 
	      16 * ((double) STS::t() / (double) Teuchos::ScalarTraits<double>::t());
	    // Adding 0.5 and truncating is a portable "floor".
	    const int numDigits = static_cast<int> (numDigitsAsDouble + 0.5);

	    // Precision to which a Scalar should be written.
	    out << std::setprecision (numDigits);
	  }
	out << elt.rowIndex() << " " << elt.colIndex() << " " << elt.value();
	return out;
      }

      template<class Scalar, class Ordinal>
      class Adder {
      public:
	typedef Ordinal index_type;
	typedef Scalar value_type;
	typedef Element<Scalar, Ordinal> element_type;
	typedef typename std::vector<element_type>::size_type size_type;

	Adder () : numRows_(0), numCols_(0), numNonzeros_(0) {}

	//! Add an element to the sparse matrix at location (i,j) (one-based indexing).
	void operator() (const Ordinal i, const Ordinal j, const Scalar& Aij) {
	  // i and j are 1-based
	  elts_.push_back (element_type (i-1, j-1, Aij));
	  // Keep track of the rightmost column containing a nonzero,
	  // and the bottommost row containing a nonzero.  This gives us
	  // a lower bound for the dimensions of the matrix, and a check
	  // for the reported dimensions of the matrix in the Matrix
	  // Market file.
	  numRows_ = std::max(numRows_, i);
	  numCols_ = std::max(numCols_, j);
	  numNonzeros_++;
	}

	/// \brief Print the sparse matrix data.  
	///
	/// We always print the data sorted.  You may also merge
	/// duplicate entries if you prefer.
	/// 
	/// \param out [out] Output stream to which to print
	/// \param doMerge [in] Whether to merge entries before printing
	/// \param replace [in] If merging, whether to replace duplicate
	///   entries; otherwise their values are added together.
	void print (std::ostream& out, const bool doMerge, const bool replace=false) {
	  if (doMerge)
	    merge (replace);
	  else
	    std::sort (elts_.begin(), elts_.end());
	  // Print out the results, delimited by newlines.
	  typedef std::ostream_iterator<element_type> iter_type;
	  std::copy (elts_.begin(), elts_.end(), iter_type (out, "\n"));
	}

	/// \brief Merge duplicate elements 
	///
	/// Merge duplicate elements of the sparse matrix, where
	/// "duplicate" means at the same (i,j) location in the sparse
	/// matrix.  Resize the array of elements to fit just the
	/// "unique" (meaning "nonduplicate") elements.
	///
	/// \return (# unique elements, # removed elements)
	std::pair<size_type, size_type>
	merge (const bool replace=false) 
	{
	  typedef typename std::vector<element_type>::iterator iter_type;

	  // Start with a sorted container.  It may be sorted already,
	  // but we just do the extra work.
	  std::sort (elts_.begin(), elts_.end());

	  // Walk through the array in place, merging duplicates and
	  // pushing unique elements up to the front of the array.  We
	  // can't use std::unique for this because it doesn't let us
	  // merge duplicate elements; it only removes them from the
	  // sequence.
	  size_type numUnique = 0;
	  iter_type cur = elts_.begin();
	  if (cur == elts_.end())
	    // There are no elements to merge
	    return std::make_pair (numUnique, size_type(0));
	  else {
	    iter_type next = cur;
	    ++next; // There is one unique element
	    ++numUnique;
	    while (next != elts_.end()) {
	      if (*cur == *next) {
		// Merge in the duplicated element *next
		cur->merge (*next, replace);
	      } else {
		// *cur is already a unique element.  Move over one to
		// *make space for the new unique element.
		++cur; 
		*cur = *next; // Add the new unique element
		++numUnique;
	      }
	      // Look at the "next" not-yet-considered element
	      ++next;
	    }
	    // Remember how many elements we removed before resizing.
	    const size_type numRemoved = elts_.size() - numUnique;
	    elts_.resize (numUnique);
	    return std::make_pair (numUnique, numRemoved);
	  }
	}

      private:
	Ordinal numRows_, numCols_, numNonzeros_;
	std::vector<element_type> elts_;
      };


      template<class Scalar, class Ordinal>
      class Reader {
      public:
	static void
	readFile (const std::string& filename,
		  const bool tolerant=false,
		  const bool debug=false)
	{
	  std::ifstream in (filename.c_str());
	  TEST_FOR_EXCEPTION(!in, std::runtime_error,
			     "Failed to open input file \"" + filename + "\".");
	  return read (in, tolerant, debug);
	}
      
	static void
	read (std::istream& in,	
	      const bool tolerant=false,
	      const bool debug=false)
	{
	  using std::cerr;
	  using std::cout;
	  using std::endl;
	  using Teuchos::RCP;
	  using Teuchos::Tuple;
	  typedef Teuchos::ScalarTraits<Scalar> STS;

	  // FIXME (mfh 01 Feb 2011) What about MPI?  Should only do
	  // this on Rank 0.
	  TEST_FOR_EXCEPTION(!in, std::invalid_argument,
			     "Input stream appears to be in an invalid state.");

	  std::string line;
	  if (! getline (in, line))
	    throw std::invalid_argument ("Failed to get first (banner) line");

	  Banner banner (line, tolerant);
	  if (banner.matrixType() != "coordinate")
	    throw std::invalid_argument ("Matrix Market input file must contain a "
					 "\"coordinate\"-format sparse matrix in "
					 "order to create a sparse matrix object "
					 "from it.");
	  else if (! STS::isComplex && banner.dataType() == "complex")
	    throw std::invalid_argument ("Matrix Market file contains complex-"
					 "valued data, but your chosen Scalar "
					 "type is real.");
	  else if (banner.dataType() != "real" && banner.dataType() != "complex")
	    throw std::invalid_argument ("Only real or complex data types (no "
					 "pattern or integer matrices) are "
					 "currently supported");
	  if (debug)
	    {
	      cout << "Banner line:" << endl
		   << banner << endl;
	    }
	  // The rest of the file starts at line 2, after the banner line.
	  size_t lineNumber = 2;

	  // Make an "Adder" that knows how to add sparse matrix entries,
	  // given a line of data from the file.
	  typedef Adder<Scalar, Ordinal> raw_adder_type;
	  // SymmetrizingAdder "advices" (yes, I'm using that as a verb)
	  // the original Adder, so that additional entries are filled
	  // in symmetrically, if the Matrix Market banner line
	  // specified a symmetry type other than "general".
	  typedef SymmetrizingAdder<raw_adder_type> adder_type;
	  RCP<raw_adder_type> rawAdder (new raw_adder_type);
	  adder_type adder (rawAdder, banner.symmType());
	  TEST_FOR_EXCEPTION(banner.dataType() == "complex" && ! STS::isComplex,
			     std::invalid_argument,
			     "The Matrix Market sparse matrix file contains "
			     "complex-valued data, but you are trying to read"
			     " the data into a sparse matrix of real values.");
	  // Make a reader that knows how to read "coordinate" format
	  // sparse matrix data.
	  typedef CoordDataReader<adder_type, Ordinal, Scalar, STS::isComplex> reader_type;
	  reader_type reader (adder);

	  // Read in the dimensions of the sparse matrix:
	  // (# rows, # columns, # structural nonzeros).
	  // The second element of the pair tells us whether the values
	  // were gotten successfully.
	  std::pair<Teuchos::Tuple<Ordinal, 3>, bool> dims = 
	    reader.readDimensions (in, lineNumber, tolerant);
	  TEST_FOR_EXCEPTION(! dims.second, std::invalid_argument,
			     "Error reading Matrix Market sparse matrix "
			     "file: failed to read coordinate dimensions.");
	  if (debug)
	    {
	      const Ordinal numRows = dims.first[0];
	      const Ordinal numCols = dims.first[1];
	      const Ordinal numNonzeros = dims.first[2];
	      cout << "Dimensions of matrix: " << numRows << " x " << numCols 
		   << ", with " << numNonzeros << " reported structural "
		"nonzeros." << endl;
	    }

	  // Read the sparse matrix entries.  "results" just tells us if
	  // and where there were any bad lines of input.  The actual
	  // sparse matrix entries are stored in the Adder object.
	  std::pair<bool, std::vector<size_t> > results = 
	    reader.read (in, lineNumber, tolerant, debug);
	  if (results.first)
	    cout << "Matrix Market file successfully read" << endl;
	  else 
	    cout << "Failed to read Matrix Market file" << endl;

	  // In tolerant mode, report any bad line number(s)
	  if (! results.first)
	    {
	      reportBadness (std::cerr, results);
	      if (! tolerant)
		throw std::invalid_argument("Invalid Matrix Market file");
	    }
	  // We're done reading in the sparse matrix.  Now print out the
	  // nonzero entry/ies.
	  if (debug)
	    {
	      const bool doMerge = false;
	      const bool replace = false;
	      rawAdder->print (std::cout, doMerge, replace);
	    }
	  std::cout << std::endl;
	}

	static void 
	reportBadness (std::ostream& out, 
		       const std::pair<bool, std::vector<size_t> >& results) 
	{
	  using std::endl;
	  const size_t numErrors = results.second.size();
	  const size_t maxNumErrorsToReport = 100;
	  out << numErrors << " errors when reading Matrix Market sparse matrix file." << endl;
	  if (numErrors > maxNumErrorsToReport)
	    out << "-- We do not report individual errors when there "
	      "are more than " << maxNumErrorsToReport << ".";
	  else if (numErrors == 1)
	    out << "Error on line " << results.second[0] << endl;
	  else if (numErrors > 1)
	    {
	      out << "Errors on lines {";
	      for (size_t k = 0; k < numErrors-1; ++k)
		out << results.second[k] << ", ";
	      out << results.second[numErrors-1] << "}" << endl;
	    }
	}
      };

    } // namespace Raw
  } // namespace MatrixMarket
} // namespace Tpetra

#endif // __MatrixMarket_raw_hpp