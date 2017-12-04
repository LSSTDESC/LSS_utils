import healpy as hp
import numpy as np

def binned_statistic(x, values, func, nbins, range):
    '''The usage is approximately the same as the scipy one
    from https://stackoverflow.com/questions/26783719/effic
    iently-get-indices-of-histogram-bins-in-python'''
    from scipy.sparse import csr_matrix
    r0, r1 = range
    mask = (x > r0) &  (x < r1)
    x = x[mask]
    values = values[mask]
    N = len(values)
    digitized = (float(nbins) / (r1-r0) * (x-r0)).astype(int)
    S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))
    return np.array([func(group) for group in np.split(S.data, S.indptr[1:-1])])

def make_hp_map(ra,dec,nside=2048):
    good = np.logical_or(np.logical_not(np.isnan(ra)),np.logical_not(np.isnan(dec)))
    pix_nums = hp.ang2pix(nside,np.pi/2.-dec[good]*np.pi/180,ra[good]*np.pi/180)
    pix_counts = np.bincount(pix_nums,minlength=12*nside**2)
    return pix_counts

def get_depth_map(ra,dec,mags,nside=2048):
    good = np.logical_or(np.logical_not(np.isnan(ra)),np.logical_not(np.isnan(dec)))
    pix_nums = hp.ang2pix(nside,np.pi/2.-dec[good]*np.pi/180,ra[good]*np.pi/180)
    map_out = np.zeros(12*nside**2)
    for px in np.unique(pix_nums):
        mask = px==pix_nums
        if np.count_nonzero(mask)>0:
            map_out[px]=np.max(mags[mask])
        else:
            map_out[px]=0.
    return map_out

def depth_map_snr(ra,dec,mags,snr,nside=2048):
    good = np.logical_or(np.logical_not(np.isnan(ra)),np.logical_not(np.isnan(dec)))
    pix_nums = hp.ang2pix(nside,np.pi/2.-dec[good]*np.pi/180,ra[good]*np.pi/180)
    map_out = np.zeros(12*nside**2)
    #Binned statistic 2d is awfully slow (because it doesn't use the fact that all bins are equal width
    #median_snr, xed, _, _ = binned_statistic_2d(mags,pix_nums,snr,statistic='median',bins=(50,12*nside**2),range=[(20,30),(0,12*nside**2)])
    #bin_centers = 0.5*xed[1:]+0.5*xed[:-1]
    #depth = bin_centers[np.argmin(np.fabs(median_snr-5),axis=0)]
    map_out = np.zeros(12*nside**2)
    bin_centers = np.linspace(22+6/30.,28-6/30.,30.)
    for px in np.unique(pix_nums):
        mask = px==pix_nums
        if np.count_nonzero(mask)>0:
            median_snr = binned_statistic(mags[mask],snr[mask],np.nanmedian,nbins=30,range=(22,28))
            mask2 = np.isnan(median_snr)==False
            if np.count_nonzero(mask2)>0:
                depth = bin_centers[mask2][np.argmin(np.fabs(median_snr[mask2]-5.))] 
                map_out[px]=depth
            else:
                map_out[px]=0
        else:
            map_out[px]=0.
    return map_out

def gen_random_fast(nrandom,mask):
    """ This method approximates using a higher resolution healpix map
    to place the random points. It should be fine for measurements larger
    than the mask pixel scale. We take advantage of the equal area nature
    of the healpixels. The downside of this method is that it needs a lot
    of memory for large masks"""

    nside=hp.get_nside(mask)
    nside2=4*nside
    ra=[]
    th=[]
    filled_pixels = np.where((mask>0) & (np.isnan(mask)==False))[0]
    densities = mask[filled_pixels]
    kpix = np.random.choice(filled_pixels,size=nrandom,p=densities/np.sum(densities))
    bincounts = np.bincount(kpix)
    kpix2 = np.unique(kpix)
    counts=bincounts[bincounts>0]
    hh=nside2**2/nside**2
    i=0
    for i,c in enumerate(counts):
        rpix=np.random.randint(0,high=hh,size=c)
        nestpix=hp.ring2nest(nside,kpix2[i])
        theta, phi = hp.pix2ang(nside2,hh*nestpix+rpix,nest=True)
        theta=90.-theta*180./np.pi
        phi=phi*180./np.pi
        for j in range(0,len(theta)):
            ra.append(phi[j])
            th.append(theta[j])
    ra=np.array(ra)
    dec=np.array(th)
    return ra,dec

def gen_random_uniform(nrandom,mask):
    """This method generates a uniform random catalog
    over certain region. This method is slow"""
    theta,phi = hp.pix2ang(hp.get_nside(mask),np.where(mask>0)[0])
    y_min = np.cos(np.min(theta))
    y_max = np.cos(np.max(theta))
    if y_min>y_max:
        aux = y_min
        aux2 = y_max
        y_min = aux2
        y_max = aux
    phi_min = np.min(phi)
    phi_max = np.max(phi)
    rphi = np.random.random(phi_min,phi_max,size=nrandom)
    ry = np.random.random(y_min,y_max,size=nrandom)
    pixnum = hp.ang2pix(hp.get_nside(mask),np.cos(ry),rphi)
    good = mask[pixnum]>0
    dec = 180/np.pi*np.arcsin(ry[good])
    ra = rphi[good]*180/np.pi
    return ra,dec
