import numpy as np

#%% Domain class

class Domain():
    def __init__(self, dim, lim, tlim):
        self.dim = dim
        self.lim = np.array(lim)
        self.tlim = tlim
        
    def getMesh(self):
        raise Exception("This function must be redefined in the subclass!")
    

class Domain2D(Domain):
    def __init__(self,
                 vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
                 tInterval = None
                 ):
        """
        Initializes the attributes of the class. The default domain is a 1x1 square.

        Inputs:
            vertices [vertexNum x dim]: vertices of the domain (one vertex per row)
            
        Attributes:
            dim: domain dimension
            lim [2 x dim]: domain limits
        """
        dim = 2
        if np.shape(vertices)[1]!=dim:
            raise ValueError('Vertex dimensions are incompatible with domain dimension!')
        
        lim = np.vstack([np.min(vertices, axis=0), np.max(vertices, axis=0)])
        tlim = tInterval
        
        super().__init__(dim, lim, tlim)
           
    def getMesh(self, discNum=100, tDiscNum=[]):
        """
        This function creates a structured mesh of the domain

        Args:
            discNum [1 x dim]: number of spatial training points (each dimension)
        """
        # Error handling:
        if not (np.size(discNum) == 1 or np.size(discNum) == 2):
            raise ValueError('\'discNum\' dimension incompatible!')
        elif np.size(discNum) == 1:
            discNum = [discNum, discNum]
            
        lim = self.lim
        coord = []                              # spatial training points
        hs = []                                 # spatial element size
        for d in range(np.size(discNum)):
            dof = discNum[d]
            h = (lim[1,d] - lim[0,d])/(dof)     # divide "unit"
            hs.append(h)
            coordTmp = np.arange(lim[0,d] + h, lim[1,d], h)
            coordTmp = np.reshape(coordTmp, [dof-1, 1])
            coord.append(coordTmp)
        hs = np.array(hs)
        
        tlim = self.tlim
        tdof = tDiscNum
        
        ht = (tlim[1] - tlim[0])/(tdof)
        t_coord = np.arange(tlim[0] + ht, tlim[1], ht)
        t_coord = np.reshape(t_coord, [tdof-1, 1])
        
        return hs, coord, ht, t_coord