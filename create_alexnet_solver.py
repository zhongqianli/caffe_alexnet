from caffe.proto import caffe_pb2

# net: "models/bvlc_alexnet/train_val.prototxt"
# test_iter: 1000
# test_interval: 1000
# base_lr: 0.01
# lr_policy: "step"
# gamma: 0.1
# stepsize: 100000
# display: 20
# max_iter: 450000
# momentum: 0.9
# weight_decay: 0.0005
# snapshot: 10000
# snapshot_prefix: "models/bvlc_alexnet/caffe_alexnet_train"
# solver_mode: GPU

def create_solver(net_name):
    s = caffe_pb2.SolverParameter()
    s.net = "{0}_train_test.prototxt".format(net_name)

    s.test_interval = 1000
    s.test_iter.append(1000)

    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 0.0005

    s.lr_policy = "step"
    s.gamma = 0.1
    s.stepsize = 100000

    s.display = 20

    s.max_iter = 450000

    s.snapshot = 10000

    s.snapshot_prefix = "snapshot/{0}".format(net_name)

    s.type = "SGD"

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open("{0}_solver.prototxt".format(net_name), "w") as f:
        f.write(str(s))

if __name__ == "__main__":
    create_solver("std_alexnet")