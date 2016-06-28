import pdb
from scipy.io import loadmat

def loadMeta(inFile):
    m = loadmat(inFile)

    vals = list(m['synsets'][0, :])

    #Dictionary comprehension
    #Wordnet idx to list idx
    wnToIdx = {str(d[1][0]): d[0][0,0]-1 for d in vals}
    #Index to name
    idxToName = [d[2][0] for d in vals]

    return (wnToIdx, idxToName)

