import sys
from model import *

def run_model(my_model):
    model = my_model()
    param_count = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM (%.5f)" % (param_count / 1e6, param_count))
    print('Trainable parameter count: {:,d} -> {:.2f} MB'.format(param_count, param_count * 32 / 8 / (2 ** 20)))
    
    # ## check model output
    # x = torch.randn((8, 2, 301, 161), dtype=torch.float32)
    # out = model(x)
    # print('{} -> {}'.format(x.shape, out['est_comp'].shape))

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num

if __name__ == '__main__':
    file = sys.argv[0]
    test_model = sys.argv[1]
    test_model = eval(test_model)
    run_model(test_model)