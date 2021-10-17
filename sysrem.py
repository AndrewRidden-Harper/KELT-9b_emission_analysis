import numpy as np
import astropy.io.fits as pyfits 


#def sysrem(r_ij, s_ij, a_j,maxiter=50, eps=1e-3):
#def sysrem(r_ij, s_ij, a_j,maxiter=10000000, eps=1e-50):
#def sysrem(r_ij, s_ij, a_j,maxiter=10000, eps=1e-15):
#def sysrem(r_ij, s_ij, a_j,maxiter=1000, eps=1e-5):
def sysrem(r_ij, s_ij, a_j,maxiter=1000, eps=1e-3):
    
    '''
    Inputs: 
    r_ij = input matrix 
    s_ij = matrix of uncertainties 
    a_j = intial guess of vector to solved for.  Needs length numcols of r_ij.
    
    I have modified it from Geert Jan's original version to get it to generate 
    a vector of 1s of length numcols to serve as the a_j vector since a_j 
    can be anything and it should still converge if allowed enough iterations.
    As stated in: The Sys-Rem Detrending Algorith: Implementation and testing 
    
    T. Mazeh, O. Tamuz, S. Zucker, arXiv 
    
    It was tested using a random initial a_j and an initial a_j of 1s 
    and they both found the same first component with a max of 20 iterations 
    
    I also changed the eps to 1e-4 from 1e-3 and maxiter to 30 from 20
    
    '''
    
   
    
    # if a_j == None:
    #     a_j = np.ones(np.shape(r_ij)[1]) #an initial vector of 1s of length number of columns in data


    #chi2_new = np.nansum(r_ij**2./s_ij**2.)
    chi2_new = 0
    chi2_old = chi2_new + 100.*eps
    #diff = np.abs((chi2_new-chi2_old))

    niter = 0
    while (np.abs(chi2_new-chi2_old)>eps) & (niter < maxiter):

        chi2_old = chi2_new

        c_i = np.nansum(r_ij*a_j/s_ij**2, axis=1)/np.nansum(a_j**2/s_ij**2, axis=1)  
        #r_ij*a_j multiplies each column of r_ij by the corresponding colulmn of a_j
        
        a_j = np.nansum(r_ij*c_i[:,np.newaxis]/s_ij**2, axis=0)/np.nansum(c_i[:,np.newaxis]**2/s_ij**2, axis=0)

        chi2_new = np.nansum((r_ij-np.outer(c_i,a_j))**2./s_ij**2.)

        niter += 1
        #print('change in chi2: %.5e'%(chi2_new-chi2_old))
    if np.abs(chi2_new-chi2_old)<eps:
        #print('Stopped iterating because change in chi2 of %.10e was lower than threshold of %.3e after %d iterations' % (chi2_new-chi2_old,eps,niter))
        pass
    
    if niter == maxiter:
        #print('stopped iterating because max number of iterations of %d was reached with delta chi2 of %f' % (niter,chi2_new-chi2_old))
        pass
    # print('c_i')
    # print(c_i)
    # print('a_j')
    # print(a_j)
    #print('Final chi2: %.5e'%(chi2_new))
    return c_i, a_j
    
def sysrem_sub(data,uncertainty_matrix,components_to_remove,a_j='default'):

    
    '''
    
    Remove the "components_to_remove" largest components.
    The vectors which minimise the global expression to fit linear 
    trends to the data are found with sysrem.
    The matrix that is removed is the outer product of the two vectors c_i and a_j
    
    '''       
    
    results_cube  = np.empty((data.shape[0],data.shape[1],components_to_remove))
    
    # hdu=pyfits.PrimaryHDU(np.array([]))
    
    # hdulist_removed=pyfits.HDUList([hdu])    
    # hdulist_data = pyfits.HDUList([hdu])
    
    # original_var = np.var(data)
    
    # f = open('sysrem_variance_changes.txt','w')
    
    # f.write('original variance = %f\n' % (original_var))
    
    
    data = data.T
    uncertainty_matrix = uncertainty_matrix.T
    
    removed_components = 0 
    
    if a_j == 'default':
        a_j = np.ones(np.shape(data)[1])
    
    while removed_components < components_to_remove: 
                
        c_i, a_j = sysrem(data,uncertainty_matrix,a_j)
        
        data = data - np.outer(c_i,a_j)    
        
        results_cube[:,:,removed_components] = data.T
        
        ## store the output in a multi-extension data cube 
        # newhdu_removed=pyfits.ImageHDU(np.outer(c_i,a_j))        
        # hdulist_removed.append(newhdu_removed)
        
        # newhdu_data = pyfits.ImageHDU(data)
        # hdulist_data.append(newhdu_data)       
        
        removed_components += 1
        ##f.write('variance after component %d removed = %f (current percent of original variance = %f, decreased by %f percent)\n' % (removed_components, np.var(data),100.0*np.var(data)/original_var,(100.0-100.0*np.var(data)/original_var)))
        
    #f.close()        
    # hdulist_data.writeto('data_after_removal_photon_unc.fits',clobber=True)
    # hdulist_removed.writeto('removed_components_photon_unc.fits',clobber=True)

    #data = data.T    
    
    #return data
    return results_cube
        
        
        