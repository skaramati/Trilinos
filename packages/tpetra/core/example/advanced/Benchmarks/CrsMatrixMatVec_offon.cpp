// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// @HEADER

// This benchmark exercises Tpetra::CrsMatrix's apply() method.
// Tpetra implements sparse matrix and dense vector data structures and
// computational kernels for users and other Trilinos data structures.
// Tpetra uses MPI (Message Passing Interface) for distributed-memory
// parallelism, and Kokkos for shared-memory parallelism within an MPI
// process.

#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "MatrixMarket_Tpetra.hpp"

#include "Kokkos_Random.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_oblackholestream.hpp"

namespace { // (anonymous)

// Options to read in from the command line
struct CmdLineOpts {
  // Do the benchmark this many times in a single timing loop, in case
  // the timer's granularity is too coarse to capture run time to
  // adequate precision.
  int numTrials;
  // Number of rows per MPI process (hence "local") in the graph;
  int lclNumRows;
  // Number of entries per row in the sparses graph;
  int numEntPerRow;
  // Bool that determines if a warm up apply is performed before timing
  bool warmUp;
  // String that points to a matrix market file to load the matrix
  std::string matrixFile;
  int matrixSize;
};

int isHost;
int pNum;
int numHost;

void setHost_dev(const Teuchos::Comm<int>& comm){
  char name[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(name, &len);
   /*if(name[4]=='b'){
        isHost = 0;
   }
   else{
        isHost = 1;
   }*/
   if(comm.getRank() < comm.getSize()/2) isHost = 1;
   else isHost = 0;
   MPI_Allreduce(&isHost, &numHost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   pNum = comm.getRank()%numHost;
   std::cout << name << " " << isHost << " " << pNum << std::endl;
   
}
// Use a utility from the Teuchos package of Trilinos to set up
// command-line options for reading, and set default values of
// command-line options.  clp is an output argument containing the
// set-up options.  It retains pointers to fields in 'opts'.  Reading
// the command-line options will update those fields in place.
void
setCmdLineOpts (CmdLineOpts& opts,
                Teuchos::CommandLineProcessor& clp)
{
  // Set default values of command-line options.

  opts.numTrials = 200;
  opts.lclNumRows = 4; //10000;
  opts.numEntPerRow = 2; //10;
  opts.warmUp = false; //true;
  opts.matrixFile = "m.mat"; //"./bcsstk38/bcsstk38.mtx";
  opts.matrixSize = 8;
 
  clp.setOption ("numTrials", &(opts.numTrials), "Number of trials per "
                 "timing loop (to increase timer precision).");
  clp.setOption ("lclNumRows", &(opts.lclNumRows), "Number of rows per MPI "
                 "process in the sparse graph.");
  clp.setOption ("numEntPerRow", &(opts.numEntPerRow), "Number of entries "
                 "per row in the sparse graph.");
  clp.setOption ("warm-up", "no-warm-up", &(opts.warmUp), "Perform a first un-timed apply"
                 " before running numTrials applies.");
  clp.setOption("matrixFile", &(opts.matrixFile), "Matrix market file containing matrix");
  clp.setOption("matrixSize", &(opts.matrixSize), "Matrix market file containing matrix");

}

// Actually read the command-line options from the command line,
// using the argc and argv arguments to main().  Use the clp output
// argument of setCmdLineOpts.  The options actually get written to
// the same CmdLineOpts struct instance that was passed into
// setCmdLineOpts above.
//
// Return 0 if successful, 1 if help printed due to the user
// specifying the "--help" option (indicates that the application
// shouldn't run), and -1 on error.
int
parseCmdLineOpts (Teuchos::CommandLineProcessor& clp, int argc, char* argv[])
{
  auto result = clp.parse (argc, argv);

  switch (result) {
  case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
    return 1;
  case Teuchos::CommandLineProcessor::PARSE_ERROR:
  case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
    return -1;
  case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
    return 0;
  default:
    return -1;
  }
}

// Check the command-line options that were read in by
// parseCmdLineOpts.  Return 0 if all correct, else return nonzero,
// using the LAPACK error reporting convention of the negative of the
// argument in its original order (starting with 1) as the error code.
// Print informative error messages to the given output stream \c out.
int
checkCmdLineOpts (std::ostream& out,
                  const Teuchos::Comm<int>& comm,
                  const CmdLineOpts& opts)
{
  int err = 0;

  if (opts.numTrials < 0) {
    out << "numTrials = " << opts.numTrials << " < 0." << std::endl;
    err = -1; // LAPACK error reporting convention
  }
  if (opts.lclNumRows < 0) {
    out << "lclNumRows = " << opts.lclNumRows << " < 0." << std::endl;
    err = -2; // LAPACK error reporting convention
  }
  if (opts.numEntPerRow < 0) {
    out << "numEntPerRow = " << opts.numEntPerRow << " < 0." << std::endl;
    err = -3; // LAPACK error reporting convention
  }

  return err;
}

// Print values of the command-line options, as read in by
// parseCmdLineOpts, to the given output stream.
void
printCmdLineOpts (Teuchos::FancyOStream& out,
                  const CmdLineOpts& opts)
{
  using std::endl;
  // Convention for FancyOStream is to push one tab before printing in
  // a scope.  OSTab pops the tab when leaving the scope.
  Teuchos::OSTab tab1 (out);
  out << "numTrials: " << opts.numTrials << endl
      << "lclNumRows: " << opts.lclNumRows << endl
      << "numEntPerRow: " << opts.numEntPerRow << endl
      << "warmUp: " << opts.warmUp << endl
      << "matrixFile: " << opts.matrixFile << endl
      << "matrixSize: " << opts.matrixSize << endl
      << endl;
}

// Return a pointer (RCP is like std::shared_ptr) to an output stream.
// It prints on Process 0 of the given MPI communicator, but ignores
// all output on other MPI processes.
Teuchos::RCP<Teuchos::FancyOStream>
getOutputStream (const Teuchos::Comm<int>& comm)
{
  using Teuchos::getFancyOStream;

  const int myRank = comm.getRank ();
  if (myRank == 0) {
    // Process 0 of the given communicator prints to std::cout.
    return getFancyOStream (Teuchos::rcpFromRef (std::cout));
  }
  else {
    // A "black hole output stream" ignores all output directed to it.
    return getFancyOStream (Teuchos::rcp (new Teuchos::oblackholestream ()));
  }
}

// Get a Tpetra::CrsGraph for use in benchmarks.  This method takes
// parameters that come from the command-line options read in by
// parseCmdLineOpts.
Teuchos::RCP<Tpetra::CrsGraph<> >
getTpetraGraph (const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                const CmdLineOpts& opts)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  typedef Tpetra::Map<> map_type;
  typedef Tpetra::CrsGraph<> graph_type;
  typedef Tpetra::Map<>::local_ordinal_type LO;
  typedef Tpetra::Map<>::global_ordinal_type GO;
  typedef Tpetra::global_size_t GST;

  LO lclNumRows = opts.lclNumRows;
  const GST gblNumRows = static_cast<GST> (opts.lclNumRows) *
    static_cast<GST> (comm->getSize ())/2;
  const GO indexBase = 0;

  // A Map describes a distribution of data over MPI processes.
  // This "row Map" will describe the distribution of rows of the
  // sparse graph that we will create.
  if(comm->getRank()>=comm->getSize()/2){
    lclNumRows = 0;
  }
  RCP<const map_type> rowMap =
    rcp (new map_type (gblNumRows, static_cast<size_t> (lclNumRows),
                       indexBase, comm));

                      
  
  const GO gblNumCols = static_cast<GO> (rowMap->getGlobalNumElements ());
  
  // Create the graph structure of the sparse matrix.
  RCP<graph_type> G =
    rcp (new graph_type (rowMap, opts.numEntPerRow));
  // Fill in the sparse graph.
  Teuchos::Array<GO> gblColInds (opts.numEntPerRow);
  for (LO lclRow = 0; lclRow < lclNumRows; ++lclRow) { // for each of my rows
     GO gblInd;
     gblInd = rowMap->getGlobalElement (lclRow);
   
    // Just put some entries in the graph.  The actual column
    // indices don't matter so much, as long as they make the
    // resulting matrix square and don't go out of bounds.
    LO lk=0;
    for (LO k = 0; k < static_cast<LO> (opts.numEntPerRow); ++k) {
      const GO curColInd = (gblInd + static_cast<GO> (3*k));
      if(comm->getRank()<comm->getSize()/2){
        if(curColInd >= rowMap->getGlobalElement (0) && curColInd <= rowMap->getGlobalElement (lclNumRows-1))
        {  gblColInds[lk] = curColInd;
            lk++;
        }
      }
      
      
    }
    G->insertGlobalIndices (gblInd, gblColInds ());
  }
  // Make the graph ready for use by CrsMatrix.
  G->fillComplete ();
  return G;
}

Teuchos::RCP<Tpetra::CrsGraph<> >
getTpetraGraph_OFF (const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                const CmdLineOpts& opts)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  typedef Tpetra::Map<> map_type;
  typedef Tpetra::CrsGraph<> graph_type;
  typedef Tpetra::Map<>::local_ordinal_type LO;
  typedef Tpetra::Map<>::global_ordinal_type GO;
  typedef Tpetra::global_size_t GST;

  LO lclNumRows = opts.lclNumRows;
  const GST gblNumRows = static_cast<GST> (opts.lclNumRows) *
    static_cast<GST> (comm->getSize ())/2;
  const GO indexBase = 0;

  // A Map describes a distribution of data over MPI processes.
  // This "row Map" will describe the distribution of rows of the
  // sparse graph that we will create.
  if(comm->getRank()<comm->getSize()/2){
    lclNumRows = 0;
  }
  RCP<const map_type> rowMap =
    rcp (new map_type (gblNumRows, static_cast<size_t> (lclNumRows),
                       indexBase, comm));

                      
  
  const GO gblNumCols = static_cast<GO> (rowMap->getGlobalNumElements ());
  
  // Create the graph structure of the sparse matrix.
  RCP<graph_type> G =
    rcp (new graph_type (rowMap, opts.numEntPerRow));
  // Fill in the sparse graph.
  Teuchos::Array<GO> gblColInds (opts.numEntPerRow);
  for (LO lclRow = 0; lclRow < lclNumRows; ++lclRow) { // for each of my rows
     GO gblInd;
     gblInd = rowMap->getGlobalElement (lclRow);
   
    // Just put some entries in the graph.  The actual column
    // indices don't matter so much, as long as they make the
    // resulting matrix square and don't go out of bounds.
    LO lk=0;
    for (LO k = 0; k < static_cast<LO> (opts.numEntPerRow); ++k) {
      const GO curColInd = (gblInd + static_cast<GO> (3*k));
     
      if(comm->getRank()>=comm->getSize()/2){
        
        if(curColInd < rowMap->getGlobalElement (0) || curColInd > rowMap->getGlobalElement (lclNumRows-1))
        {   
            std::cout << comm->getRank() << " " << lk << " " << curColInd << std::endl;
            gblColInds[lk] = curColInd;
            lk++;
        }
      }
      
      
    }
    G->insertGlobalIndices (gblInd, gblColInds ());
  }
  // Make the graph ready for use by CrsMatrix.
  G->fillComplete ();
  return G;
}

// Get a Tpetra::CrsMatrix for use in benchmarks.
// This method takes the result of getTpetraGraph() (above) and
// parameters that come from the command-line options read in by
// parseCmdLineOpts.
Teuchos::RCP<Tpetra::CrsMatrix<> >
getTpetraCrsMatrix (Teuchos::FancyOStream& out,
                    const Teuchos::RCP<const Tpetra::CrsGraph<> >& graph,
                    const CmdLineOpts& opts)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using std::endl;
  using matrix_type = Tpetra::CrsMatrix<>;
  //using device_type = matrix_type::device_type;
  using SC  = matrix_type::impl_scalar_type;
  //using KAT = Kokkos::ArithTraits<SC>;
  using LO  = Tpetra::Map<>::local_ordinal_type;
  //using host_device_type     = Kokkos::View<SC*, Kokkos::LayoutRight, device_type>::host_mirror_space;
  //using host_execution_space = host_device_type::execution_space;

  // We're filling on the host, so generate random numbers on the host.
  //using pool_type = Kokkos::Random_XorShift64_Pool<host_execution_space>;

  Teuchos::OSTab tab0 (out);
  out << "Create CrsMatrix for benchmark" << endl;
  Teuchos::OSTab tab1 (out);

  const auto meshRowMap = * (graph->getRowMap ());
  // Contrary to expectations, asking for the graph's number of
  // columns, or asking the column Map for the number of entries,
  // won't give the correct number of columns in the graph.
  // const GO gblNumCols = graph->getDomainMap ()->getGlobalNumElements ();
  const LO lclNumRows = meshRowMap.getLocalNumElements ();

  RCP<matrix_type> A = rcp (new matrix_type (graph));

  // Fill in the sparse matrix.
  out << "Fill the CrsMatrix" << endl;
  for (LO lclRow = 0; lclRow < lclNumRows; ++lclRow) { // for each of my rows
    matrix_type::local_inds_host_view_type lclColInds;
    graph->getLocalRowView (lclRow, lclColInds);

    // Put some entries in the matrix.
    matrix_type::values_host_view_type::non_const_type
                 lclValues("testLclValues", lclColInds.extent(0));
    Kokkos::deep_copy(lclValues, Teuchos::ScalarTraits<SC>::one());
    const LO err = A->replaceLocalValues (lclRow, lclColInds, lclValues);
    TEUCHOS_TEST_FOR_EXCEPTION(size_t(err) != lclColInds.size(),
                               std::logic_error, "Bug");
  }
  A->fillComplete();

  return A;
}

} // namespace (anonymous)


int
main (int argc, char* argv[])
{
  using Teuchos::RCP;
  using Teuchos::REDUCE_MIN;
  using Teuchos::reduceAll;
  using Teuchos::outArg;
  using Teuchos::TimeMonitor;
  using std::endl;
  typedef Tpetra::Vector<>::scalar_type SC;

  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  bool success = true;
  {
    auto comm = Tpetra::getDefaultComm ();

    // Output stream 'out' will ignore output not from Process 0.
    RCP<Teuchos::FancyOStream> pOut = getOutputStream (*comm);
    Teuchos::FancyOStream& out = *pOut;

    auto fout = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));

    // Read command-line options into the 'opts' struct.
    CmdLineOpts opts;
    {
      Teuchos::CommandLineProcessor clp;
      setCmdLineOpts (opts, clp);
      int result = parseCmdLineOpts (clp, argc, argv);
      if (result == 1) { // help printed
        return EXIT_SUCCESS;
      }
      else if (result == -1) { // parse error
        return EXIT_FAILURE;
      }
      result = checkCmdLineOpts (out, *comm, opts);
      if (result != 0) {
        return EXIT_FAILURE;
      }
    }

    out << "Command-line options:" << endl;
    printCmdLineOpts (out, opts);
    setHost_dev(*comm);
   
    // Create or read in the matrix
    RCP<Tpetra::CrsMatrix<> > A;
    RCP<Tpetra::CrsMatrix<> > A_OFF;
    RCP<Tpetra::CrsMatrix<> > A_ON;
    if(opts.matrixFile.empty()) {
      auto timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: getGraph");
      RCP<Tpetra::CrsGraph<> > G;
      RCP<Tpetra::CrsGraph<> > G_OFF;
      {
        TimeMonitor timeMon (*timer);
        G = getTpetraGraph (comm, opts);
        G_OFF = getTpetraGraph_OFF (comm, opts);
        
      }
      timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: getCrsMatrix");
      {
        TimeMonitor timeMon (*timer);
        A = getTpetraCrsMatrix (out, G, opts);
        A_OFF = getTpetraCrsMatrix (out, G_OFF, opts);
        A->describe(*fout, Teuchos::VERB_EXTREME);
        A_OFF->describe(*fout, Teuchos::VERB_EXTREME);
      }
    } else {
      auto timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: readCrsMatrix");
      {
        TimeMonitor timeMon (*timer);

        
        typedef Tpetra::Map<>::global_ordinal_type GO;
        using Teuchos::RCP;
        using Teuchos::rcp;
        typedef Tpetra::Map<> map_type;
        
        int matrix_size = opts.matrixSize;
        int localH = 0;
        int localBF = 0;

        
        int myRank = comm->getRank();
        if (isHost){
          localH=matrix_size/numHost + ((pNum < (matrix_size%numHost))?1:0);
        } 
        else {
          localBF=matrix_size/numHost + ((pNum < (matrix_size%numHost))?1:0);
        }
        
        RCP<const map_type> rowMap_Host =rcp (new map_type (matrix_size, static_cast<size_t> (localH),
                       0, comm));
        
        RCP<const map_type> rowMap_BF =rcp (new map_type (matrix_size, static_cast<size_t> (localBF),
                       0, comm));
        
        RCP<const map_type> nullmap;
        comm->barrier();
        std::cout << "Create A begin\n";
        A = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<> >::readSparseFile(opts.matrixFile, rowMap_Host, nullmap,rowMap_Host, rowMap_Host);
        comm->barrier();
        std::cout << "Create A end\n";
        RCP<const map_type> colmap_on;
        {
          Teuchos::Array<GO> GlobalElements_ON;
          RCP<Tpetra::CrsMatrix<> > A_temp = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<> >::readSparseFile(opts.matrixFile, rowMap_BF, nullmap, rowMap_BF, rowMap_BF);
          RCP<const map_type> colmap = A_temp->getColMap();
          auto GlobalElements = colmap->getMyGlobalIndices();
       
          if(!isHost){
            int low = rowMap_BF->getGlobalElement (0);
            int high = rowMap_BF->getGlobalElement (localBF-1);
            for (size_t i=0; i<GlobalElements.size(); ++i){
              if((GlobalElements[i]>=low) && (GlobalElements[i]<=high)) GlobalElements_ON.append(GlobalElements[i]);
              
            }
          }
          colmap_on = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), GlobalElements_ON(), 0, comm));
        }
        comm->barrier();
        std::cout << "Create A_on begin\n";
        //A_ON = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<> >::readSparseFile(opts.matrixFile, rowMap_BF, colmap_on, rowMap_BF, rowMap_BF);
        comm->barrier();
        std::cout << "Create A_on end\n";
        RCP<const map_type> colmap_off;
        {
          Teuchos::Array<GO> GlobalElements_OFF;
          RCP<Tpetra::CrsMatrix<> > A_temp = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<> >::readSparseFile(opts.matrixFile, rowMap_Host, nullmap, rowMap_Host, rowMap_Host);
          RCP<const map_type> colmap = A_temp->getColMap();
          auto GlobalElements = colmap->getMyGlobalIndices();
       
          if(isHost){
            int low = rowMap_Host->getGlobalElement (0);
            int high = rowMap_Host->getGlobalElement (localH-1);
            std::cout << low << " " << high << " "<<myRank<<std::endl; 
            for (size_t i=0; i<GlobalElements.size(); ++i){
              if(GlobalElements[i]<low) GlobalElements_OFF.append(GlobalElements[i]);
              if(GlobalElements[i]>high) GlobalElements_OFF.append(GlobalElements[i]);
            }
          }
          colmap_off = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), GlobalElements_OFF(), 0, comm));
        }
        comm->barrier();
        std::cout << "Create A_off begin\n";
        A_OFF = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<> >::readSparseFile(opts.matrixFile, rowMap_Host, colmap_off, rowMap_Host, rowMap_Host);
        comm->barrier();
        std::cout << "Create A_off end\n";
      }
    }
    
    Tpetra::Vector<> X (A->getRowMap ());
    Tpetra::Vector<> Y (A->getRowMap ());
    //Tpetra::Vector<> X1 (A_ON->getRowMap ());
    //Tpetra::Vector<> Y1 (A_ON->getRowMap ());
    Tpetra::Vector<> X2 (A_OFF->getRowMap ());
    Tpetra::Vector<> Y2 (A->getRowMap ());
    
    // Fill X with values that don't increase the max-norm of results.
    // That way, repeated mat-vecs won't overflow.  This matters
    // because some processors do a silly thing and handle Inf or NaN
    // (or even denorms) via traps.  This is very expensive, so if the
    // norms increase or decrease a lot, that might trigger the slow
    // case.
    auto timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: create vectors");
    {
      TimeMonitor timeMon (*timer);
      const SC X_val = static_cast<SC> (1.0) /
        static_cast<SC> (opts.numEntPerRow);
      X.putScalar (X_val);
      X2.putScalar (X_val);
      Y.putScalar (0.0);
    }

    // We first do a "warm-up" apply mainly to make sure UVM allocations
    // are already set by the time we get to the timer.
    // We make it an option to not do the warm-up in case someone wants to
    // quantify how "first pass" apply does.
    Tpetra::Details::Behavior::disable_timing();	
    Tpetra::Details::Behavior::enable_timing();	

    timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: apply (warm-up)");
    if(opts.warmUp) {
      TimeMonitor timeMon (*timer);
      A->apply (X, Y);
    }
    comm->barrier();
    std::cout << "**************************************************\n";
    std::cout << "*****************A->apply(X, Y)*******************\n";
    std::cout << "**************************************************\n";
    comm->barrier();
    timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: apply 1 (mat-vec)");
    {
      for (int trial = 0; trial < opts.numTrials; ++trial) {
        TimeMonitor timeMon (*timer);
        A->apply (X, Y);
      }
    }
    comm->barrier();
    std::cout << "**************************************************\n";
    std::cout << "**************A_ON->apply(X1, Y1)*****************\n";
    std::cout << "**************************************************\n";
    comm->barrier();
    timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: apply 2 (mat-vec)");
    {
      for (int trial = 0; trial < opts.numTrials; ++trial) {
        TimeMonitor timeMon (*timer);
        A->apply (X2, Y2);
      }
    }
    comm->barrier();
    std::cout << "**************************************************\n";
    std::cout << "**************A_OFF->apply(X2, Y2)****************\n";
    std::cout << "**************************************************\n";
    comm->barrier();
    timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: apply H-BF-off (mat-vec)");
    {
      for (int trial = 0; trial < opts.numTrials; ++trial) {
        TimeMonitor timeMon (*timer);
        A_OFF->apply (X2, Y2);
        
      }
    }
    std::cout << "**************************************************\n";
    std::cout << "***********************END************************\n";
    std::cout << "**************************************************\n";
    timer = TimeMonitor::getNewCounter ("Tpetra CrsMatrix Benchmark: apply H-BF-update (mat-vec)");
    {
      for (int trial = 0; trial < opts.numTrials; ++trial) {
        TimeMonitor timeMon (*timer);
        
        //Y1.update(1.0, Y2, 1.0);
      }
    }
    
    TimeMonitor::report (comm.ptr (), out);
  }

  if (success) {
    return EXIT_SUCCESS;
  }
  else {
    return EXIT_FAILURE;
  }
}

