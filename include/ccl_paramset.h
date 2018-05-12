/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once
// Max. no. of paramsets allowed in the list
#define CCL_PARAMSET_LIST_SIZE 30

// Status variables
#define CCL_PARAMSET_NOT_FOUND 1
#define CCL_PARAMSET_ALREADY_EXISTS 2

typedef struct ccl_paramset_list {
    void* paramset[CCL_PARAMSET_LIST_SIZE];
    int num_sets;
    char name[CCL_PARAMSET_LIST_SIZE][128];
} ccl_paramset_list;


int ccl_paramset_list_init(ccl_paramset_list* pslist);
int ccl_has_paramset(ccl_paramset_list* pslist, char* name);
void* ccl_get_paramset(ccl_paramset_list* pslist, char* name);
void ccl_add_paramset(ccl_paramset_list* pslist, char* name, void* params, 
                      int* status);
void ccl_remove_paramset(ccl_paramset_list* pslist, char* name, int* status);
void ccl_list_paramsets(ccl_paramset_list* pslist);

#ifdef __cplusplus
}
#endif
