from network import MCNN
import sys
                
if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 test.py A(or B)')
    exit()
                            
mcnn = MCNN(dataset)
mcnn.test()