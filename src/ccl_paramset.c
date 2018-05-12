#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "ccl_paramset.h"


/* ------- ROUTINE: ccl_paramset_list_init ------
   INPUTS: ccl_paramset_list pslist
   TASK: Initialize a ccl_paramset_list.   
*/
int ccl_paramset_list_init(ccl_paramset_list* pslist){
    // Create a new paramset_list
    pslist->num_sets = 0;
}


/* ------- ROUTINE: ccl_has_paramset ------
   INPUTS: ccl_paramset_list pslist
           char* name
   OUTPUT: bool has_paramset
   TASK:   Return true if the paramset_list contains a paramset with this name
*/
int ccl_has_paramset(ccl_paramset_list* pslist, char* name){
    // Check if the parameter dictionary contains a given parameter set
    for(int i=0; i < pslist->num_sets; i++){
        if(strcmp(pslist->name[i], name) == 0) return true; // found
    }
    return false;
}


/* ------- ROUTINE: ccl_get_paramset ------
   INPUTS: ccl_paramset_list pslist
           char* name
   OUTPUT: void* (pointer to user-defined parameters struct)
   TASK:   Return a pointer to a user-defined parameters structure, specified 
           by 'name'. This is returned as a void pointer, which must be cast to 
           the correct type. If the paramset is not found, NULL is returned.
*/
void* ccl_get_paramset(ccl_paramset_list* pslist, char* name){
    // Get a paramset by name (if it exists)
    
    // Loop over available paramsets to find the requested one
    for(int i=0; i < pslist->num_sets; i++){
        if(strcmp(pslist->name[i], name) == 0) return pslist->paramset[i];
    }
    
    // paramset doesn't exist, return NULL
    return NULL;
}


/* ------- ROUTINE: ccl_add_paramset ------
   INPUTS: ccl_paramset_list pslist
           char* name
           void* params
           int* status
   TASK:   Add a pointer to a user-defined parameter structure to the 
           paramset_list. If a paramset with this name already exists, status 
           is returned with value CCL_PARAMSET_ALREADY_EXISTS.
*/
void ccl_add_paramset(ccl_paramset_list* pslist, char* name, void* params, 
                      int* status)
{
    // Add a parameter struct to the paramset_list
    
    // Check to see if paramset with this name already exists
    if (has_paramset(pslist, name)){
        *status = CCL_PARAMSET_ALREADY_EXISTS;
        return;
    }
    
    // Add new paramset
    int idx = pslist->num_sets;
    strcpy(pslist->name[idx], name); // Name of paramset_list
    pslist->paramset[idx] = params; // Assign pointer to params struct
    pslist->num_sets = idx + 1;
    
    *status = 0;
    return;
}


/* ------- ROUTINE: ccl_remove_paramset ------
   INPUTS: ccl_paramset_list pslist
           char* name
           int* status
   TASK:   Remove a paramset from the list. The paramset is not deallocated by 
           this function; the reference to it is simply removed from the list. 
           If a paramset with this name does not exist, status is returned with 
           value CCL_PARAMSET_NOT_FOUND.
*/
void ccl_remove_paramset(ccl_paramset_list* pslist, char* name, int* status){
    // Remove a named paramset, if it exists
    
    // Find the index of the paramset to be removed
    int idx = -1;
    for(int i=0; i < pslist->num_sets; i++){
        if(strcmp(pslist->name[i], name) == 0){
            idx = i;
            break;
        }
    }
    
    // Return errorcode if no paramset was found
    if(idx == -1){
        *status = CCL_PARAMSET_NOT_FOUND;
        return;
    }
    
    // Shuffle paramsets up in the list, replacing the paramset to be removed
    for(int i=idx+1; i < pslist->num_sets; i++){
        strcpy(pslist->name[i-1], pslist->name[i]);
        pslist->paramset[i-1] = pslist->paramset[i];
    }
    pslist->num_sets--; // Decrement paramset count
    *status = 0;
    return;
}


/* ------- ROUTINE: ccl_list_paramsets ------
   INPUTS: ccl_paramset_list pslist
   TASK:   Print a list of all paramsets stored in the list.
*/
void ccl_list_paramsets(ccl_paramset_list* pslist){
    // Check if the parameter dictionary contains a given parameter set
    for(int i=0; i < pslist->num_sets; i++)
        printf("%3d: %s\n", i, pslist->name[i]);
}

