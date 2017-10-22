# Part 1: Data + Architecture
import numpy as np

# I = (hours sleeping, hours studying) , j = Score on test
I = np.array(([3,5], [5,1] , [10,2]), dtype=float)
j = np.array(([75], [82], [93]), dtype=float)

# Normalize
I = I / np.amax(I, axis=0)
j = j/100 #Max test score is 100
