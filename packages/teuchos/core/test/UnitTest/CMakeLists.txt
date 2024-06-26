#
# UnitTest_UnitTests
#


TRIBITS_ADD_EXECUTABLE(
  UnitTest_UnitTests
  SOURCES
    UnitTest_UnitTests.cpp
    ${TEUCHOS_STD_UNIT_TEST_MAIN}
  )


TRIBITS_ADD_TEST(
  UnitTest_UnitTests
  POSTFIX_AND_ARGS_0 test_names --details=TEST_NAMES
  POSTFIX_AND_ARGS_1 final_results --details=FINAL_RESULTS
  POSTFIX_AND_ARGS_2 all --details=ALL
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )


#
# Simple_UnitTests
#


TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Simple_UnitTests
  SOURCES
    Int_UnitTests.cpp
    PrintDouble_UnitTests.cpp
    TemplateFunc_UnitTests.cpp
    vector_UnitTests.cpp
    ${TEUCHOS_STD_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )

IF (${PARENT_PACKAGE_NAME}_ENABLE_COMPLEX AND ${PARENT_PACKAGE_NAME}_ENABLE_FLOAT)
  SET(PASS_STRING "Summary: total = 19, run = 18, passed = 18, failed = 0")
ELSEIF (${PARENT_PACKAGE_NAME}_ENABLE_COMPLEX AND NOT ${PARENT_PACKAGE_NAME}_ENABLE_FLOAT)
  SET(PASS_STRING "Summary: total = 17, run = 16, passed = 16, failed = 0")
ELSEIF (NOT ${PARENT_PACKAGE_NAME}_ENABLE_COMPLEX AND ${PARENT_PACKAGE_NAME}_ENABLE_FLOAT)
  SET(PASS_STRING "Summary: total = 17, run = 16, passed = 16, failed = 0")
ELSEIF (NOT ${PARENT_PACKAGE_NAME}_ENABLE_COMPLEX AND NOT ${PARENT_PACKAGE_NAME}_ENABLE_FLOAT)
  SET(PASS_STRING "Summary: total = 16, run = 15, passed = 15, failed = 0")
ENDIF()


TRIBITS_ADD_TEST(
  Simple_UnitTests
  NAME_POSTFIX minus_one
  ARGS "--not-unit-test=vector_float_constAt_UnitTest"
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "${PASS_STRING}"
  )


#
# UnitTest_BadUnitTest
#


TRIBITS_ADD_EXECUTABLE(
  UnitTest_BadUnitTest
  COMM serial mpi  
  SOURCES
    Int_UnitTests.cpp
    TemplateFunc_UnitTests.cpp
    Failing_UnitTest.cpp
    ${TEUCHOS_STD_UNIT_TEST_MAIN}
  )


TRIBITS_ADD_TEST(
  UnitTest_BadUnitTest
  NAME UnitTest_BadUnitTest_final_results
  ARGS "--show-test-details=FINAL_RESULTS"
  NUM_MPI_PROCS 1
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "Int_BadAssignment_UnitTest"
  )


TRIBITS_ADD_TEST(
  UnitTest_BadUnitTest
  NAME UnitTest_BadUnitTest_end_result_failed
  ARGS "--teuchos-suppress-startup-banner"
  NUM_MPI_PROCS 1
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "End Result: TEST FAILED"
  )


TRIBITS_ADD_TEST(
  UnitTest_BadUnitTest
  NAME UnitTest_BadUnitTest_end_result_totals
  ARGS "--teuchos-suppress-startup-banner"
  NUM_MPI_PROCS 1
  COMM serial mpi
  PASS_REGULAR_EXPRESSION ", failed = 2"
  )


TRIBITS_ADD_TEST(
  UnitTest_BadUnitTest
  NAME UnitTest_BadUnitTest_Int_BadAssignment_failed
  ARGS
    "--show-test-details=FINAL_RESULTS --group-name=Int --test-name=BadAssignment"
    "--show-test-details=TEST_NAMES --group-name=Int --test-name=BadAssignment"
  NUM_MPI_PROCS 1
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "[FAILED]"
  )


#
# Hashtable unit tests
#


TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Hashtable_UnitTest
  SOURCES
    Hashtable_UnitTests.cpp
    ${TEUCHOS_STD_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )


#
# Range1D unit tests
#


TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Range1D_UnitTests
  SOURCES
    Range1D_UnitTests.cpp
    ${TEUCHOS_STD_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )


#
# VerbosityLevel unit tests
#

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  VerbosityLevel_UnitTests
  SOURCES
    VerbosityLevel_UnitTests.cpp
    ${TEUCHOS_STD_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )


#
# GlobalMPISession unit tests
#


TRIBITS_ADD_EXECUTABLE_AND_TEST(
  GlobalMPISessionUnitTests
  SOURCES GlobalMPISession_UnitTests.cpp
    ${TEUCHOS_STD_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  )
TRIBITS_ADD_TEST(
  GlobalMPISessionUnitTests
  COMM mpi
  STANDARD_PASS_OUTPUT
  )
