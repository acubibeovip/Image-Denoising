class ELMData():
    def __init__(self, 
                 PDE,
                 discNum=20,
                 bDiscNum=[],
                 tDiscNum=[]):
        
        self.dim = PDE.dim
        self.PDE = PDE
        self.discNum = discNum
        self.bDiscNum = bDiscNum
        self.tDiscNum = tDiscNum
        
        self.trainingPoints()
        
    def trainingPoints(self):
        PDE = self.PDE
        domain = PDE.domain
        discNum = self.discNum
        bDiscNum = self.bDiscNum
        
        mesh = domain.getMesh(discNum, bDiscNum)
        print("Debug")