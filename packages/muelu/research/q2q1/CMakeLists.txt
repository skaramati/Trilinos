TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})



IF ( NOT MueLu_ENABLE_EXPLICIT_INSTANTIATION OR (MueLu_ENABLE_EXPLICIT_INSTANTIATION AND MueLu_TPETRA_INST_INT_INT) )

  IF (${PACKAGE_NAME}_ENABLE_ThyraTpetraAdapters AND ${PACKAGE_NAME}_ENABLE_Teko)
    IF ((${PACKAGE_NAME}_ENABLE_Ifpack2 AND ${PACKAGE_NAME}_ENABLE_Amesos2) OR
        (${PACKAGE_NAME}_ENABLE_Epetra AND ${PACKAGE_NAME}_ENABLE_EpetraExt AND ${PACKAGE_NAME}_ENABLE_Ifpack  AND ${PACKAGE_NAME}_ENABLE_Amesos))

      TRIBITS_ADD_EXECUTABLE(
        Q2Q1
        SOURCES Q2Q1.cpp MueLu_Q2Q1PFactory.hpp MueLu_Q2Q1uPFactory.hpp
        COMM serial mpi
        )

      TRIBITS_COPY_FILES_TO_BINARY_DIR(Q2Q1_cp
        SOURCE_FILES driver.xml status.py

        Q2Q1_9x9_A.mm         Q2Q1_9x9_B.mm               Q2Q1_9x9_Bt.mm              Q2Q1_9x9_BigA.mm
        Q2Q1_9x9_p2vMap.mm    Q2Q1_9x9_PresCoords.mm      Q2Q1_9x9_VelCoords.mm       Q2Q1_9x9_AForPat.mm
        )
    ENDIF()

    # LBV on Sept 18th 2019: commenting out the test as there is no will to fix Thyra to have it work...
    # #
    # # Tests
    # #

    # INCLUDE(XpetraAddTest)

    # IF (${PACKAGE_NAME}_ENABLE_Ifpack2 AND ${PACKAGE_NAME}_ENABLE_Amesos2 AND ${PACKAGE_NAME}_ENABLE_ThyraTpetraAdapters AND ${PACKAGE_NAME}_ENABLE_Teko)
    #   XPETRA_ADD_TEST_TPETRA(Q2Q1 1)
    # ENDIF()

    # IF (${PACKAGE_NAME}_ENABLE_Epetra AND ${PACKAGE_NAME}_ENABLE_Ifpack  AND ${PACKAGE_NAME}_ENABLE_Amesos  AND ${PACKAGE_NAME}_ENABLE_ThyraTpetraAdapters AND ${PACKAGE_NAME}_ENABLE_Teko)
    #   # Disabling Epetra as Vanka is unavailable
    #   # XPETRA_ADD_TEST_EPETRA(Q2Q1 1)
    # ENDIF()

  ENDIF()

ENDIF()
