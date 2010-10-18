
INCLUDE(SetCacheOnOffEmpty)
INCLUDE(MultilineSet)
INCLUDE(AdvancedOption)


#
# This module defines the datastructure for the list of packages
# ${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY which has the form:
#
#   repo0_nameDir  repo0_giturl   repo0_classification
#   repo1_nameDir  repo1_giturl   repo1_classification
#   ...
#
# There are 3 fields per row all stored in a flat array.
# 


SET(ERP_REPO_DIRNAME_OFFSET 0)
SET(ERP_REPO_GITURL_OFFSET 1)
SET(ERP_REPO_CLASSIFICATION_OFFSET 2)

SET(ERP_NUM_FIELDS_PER_REPO 3)


#
# Macro that processes the list varaible contents in
# ${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY into sperate arrays
# ${PROJECT_NAME}_EXTRA_REPOSITORIES_DEFAULT and
# ${PROJECT_NAME}_EXTRA_REPOSITORIES_GITURLS
#
# The macro responds to ${PROJECT_NAME}_ENABLE_KNOWN_EXTERNAL_REPOS_TYPE
# to match the categories.
#
#

MACRO(PACKAGE_ARCH_PROCESS_EXTRAREPOS_LISTS)

  # A) Get the total number of extrarepos defined  

  #PRINT_VAR(${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY)
  ASSERT_DEFINED(${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY)
  LIST(LENGTH ${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY
    ${PROJECT_NAME}_NUM_EXTRAREPOS_AND_FIELDS )
  MATH(EXPR ${PROJECT_NAME}_NUM_EXTRAREPOS
    "${${PROJECT_NAME}_NUM_EXTRAREPOS_AND_FIELDS}/${ERP_NUM_FIELDS_PER_REPO}")
  #PRINT_VAR(${PROJECT_NAME}_NUM_EXTRAREPOS)
  MATH(EXPR ${PROJECT_NAME}_LAST_EXTRAREPO_IDX "${${PROJECT_NAME}_NUM_EXTRAREPOS}-1")
  #PRINT_VAR(${PROJECT_NAME}_LAST_EXTRAREPO_IDX)

  # B) Process the list of extra repos

  SET(${PROJECT_NAME}_EXTRA_REPOSITORIES_DEFAULT)
  SET(${PROJECT_NAME}_EXTRA_REPOSITORIES_GITURLS)

  FOREACH(EXTRAREPO_IDX RANGE ${${PROJECT_NAME}_LAST_EXTRAREPO_IDX})

    # B.1) Extract the fields for the current extrarepo row

    # DIRNAME
    MATH(EXPR EXTRAREPO_DIRNAME_IDX
      "${EXTRAREPO_IDX}*${ERP_NUM_FIELDS_PER_REPO}+${ERP_REPO_DIRNAME_OFFSET}")
    LIST(GET ${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY
      ${EXTRAREPO_DIRNAME_IDX} EXTRAREPO_DIRNAME )
    #PRINT_VAR(EXTRAREPO_DIRNAME)

    # GITURL
    MATH(EXPR EXTRAREPO_GITURL_IDX
      "${EXTRAREPO_IDX}*${ERP_NUM_FIELDS_PER_REPO}+${ERP_REPO_GITURL_OFFSET}")
    LIST(GET ${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY
      ${EXTRAREPO_GITURL_IDX} EXTRAREPO_GITURL )
    #PRINT_VAR(EXTRAREPO_GITURL)

    # CLASSIFICATION
    MATH(EXPR EXTRAREPO_CLASSIFICATION_IDX
      "${EXTRAREPO_IDX}*${ERP_NUM_FIELDS_PER_REPO}+${ERP_REPO_CLASSIFICATION_OFFSET}")
    LIST(GET ${PROJECT_NAME}_EXTRAREPOS_DIR_GITREPOURL_CATEGORY
      ${EXTRAREPO_CLASSIFICATION_IDX} EXTRAREPO_CLASSIFICATION )
    #PRINT_VAR(EXTRAREPO_CLASSIFICATION)

    # B.2) Determine the match of the classification

    SET(ADD_EXTRAREPO FALSE)
    #ASSERT_DEFINED(${PROJECT_NAME}_ENABLE_KNOWN_EXTERNAL_REPOS_TYPE)
    #PRINT_VAR(${PROJECT_NAME}_ENABLE_KNOWN_EXTERNAL_REPOS_TYPE)
    IF (${PROJECT_NAME}_ENABLE_KNOWN_EXTERNAL_REPOS_TYPE STREQUAL "Continuous" AND
        EXTRAREPO_CLASSIFICATION STREQUAL "Continuous"
      )
      SET(ADD_EXTRAREPO TRUE)
    ELSEIF (${PROJECT_NAME}_ENABLE_KNOWN_EXTERNAL_REPOS_TYPE STREQUAL "Nightly" AND
        (EXTRAREPO_CLASSIFICATION STREQUAL "Continuous" OR EXTRAREPO_CLASSIFICATION STREQUAL "Nightly")
      )
      SET(ADD_EXTRAREPO TRUE)
    ENDIF()
    #PRINT_VAR(ADD_EXTRAREPO)


    # B.3) Add the extrarepo to the list if the classification matches

    IF (ADD_EXTRAREPO)
      MESSAGE("-- " "Adding extra ${EXTRAREPO_CLASSIFICATION} repository ${EXTRAREPO_DIRNAME} ...")
      LIST(APPEND ${PROJECT_NAME}_EXTRA_REPOSITORIES_DEFAULT ${EXTRAREPO_DIRNAME})
      LIST(APPEND ${PROJECT_NAME}_EXTRA_REPOSITORIES_GITURLS ${EXTRAREPO_GITURL})
    ENDIF()

  ENDFOREACH()

  # C) Get the actual number of active extra repos

  LIST(LENGTH ${PROJECT_NAME}_EXTRA_REPOSITORIES_DEFAULT ${PROJECT_NAME}_NUM_EXTRAREPOS )
  #PRINT_VAR(${PROJECT_NAME}_NUM_EXTRAREPOS)
  MATH(EXPR ${PROJECT_NAME}_LAST_EXTRAREPO_IDX "${${PROJECT_NAME}_NUM_EXTRAREPOS}-1")

  # D) Print the final set of extrarepos in verbose mode
  
  IF (${PROJECT_NAME}_VERBOSE_CONFIGURE)
    PRINT_VAR(${PROJECT_NAME}_EXTRA_REPOSITORIES_DEFAULT)
    PRINT_VAR(${PROJECT_NAME}_EXTRA_REPOSITORIES_GITURLS)
  ENDIF()

ENDMACRO()


#
# Extract the final name of the extra repo
#

FUNCTION(GET_EXTRAREPO_BASE_NAME  EXTRAREPO_DIRNAME EXTRAREPO_NAME_OUT)
  GET_FILENAME_COMPONENT(EXTRAREPO_NAME "${EXTRAREPO_DIRNAME}" NAME)
  SET(${EXTRAREPO_NAME_OUT} "${EXTRAREPO_NAME}" PARENT_SCOPE)
ENDFUNCTION()