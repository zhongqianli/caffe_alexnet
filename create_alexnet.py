import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

# input: 227x227
def create_alexnet():
    net_name = "std_alexnet"
    net_filename = "{0}_train_test.prototxt".format(net_name)
    with open(net_filename, "w") as f:
        f.write('name: "{0}"\n'.format(net_name))

    net = caffe.NetSpec()
    batch_size = 256
    lmdb = "examples/imagenet/ilsvrc12_train_lmdb"
    net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                      transform_param=dict(mirror=True,
                                                           crop_size=227,
                                                           mean_file="data/ilsvrc12/imagenet_mean.binaryproto"),
                                 ntop=2,include=dict(phase=caffe_pb2.Phase.Value("TRAIN")))

    with open(net_filename, "a") as f:
        f.write(str(net.to_proto()))

    del net
    net = caffe.NetSpec()
    batch_size = 50
    lmdb = "examples/imagenet/ilsvrc12_test_lmdb"
    net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                 transform_param=dict(mirror=False,
                                                      crop_size=227,
                                                      mean_file="data/ilsvrc12/imagenet_mean.binaryproto"),
                                 ntop=2, include=dict(phase=caffe_pb2.Phase.Value("TEST")))

    net.conv1 = L.Convolution(net.data, kernel_size=11, stride=4, num_output=96,
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.norm1 = L.LRN(net.relu1, local_size=5, alpha=0.0001, beta=0.75)
    net.pool1 = L.Pooling(net.norm1, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    net.conv2 = L.Convolution(net.pool1, kernel_size=5, num_output=256, pad=2, group=2,
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.norm2 = L.LRN(net.relu2, local_size=5, alpha=0.0001, beta=0.75)
    net.pool2 = L.Pooling(net.norm2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    net.conv3 = L.Convolution(net.pool2, kernel_size=3, num_output=384, pad=1,
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu3 = L.ReLU(net.conv3, in_place=True)

    net.conv4 = L.Convolution(net.relu3, kernel_size=3, num_output=384, pad=1,
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0.1),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu4 = L.ReLU(net.conv4, in_place=True)

    net.conv5 = L.Convolution(net.relu4, kernel_size=3, num_output=256, pad=1, group=2,
                              weight_filler=dict(type="gaussian", std=0.01),
                              bias_filler=dict(type="constant", value=0.1),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu5 = L.ReLU(net.conv5, in_place=True)
    net.pool5 = L.Pooling(net.relu5, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    net.fc6 = L.InnerProduct(net.pool5, num_output=4096,
                             weight_filler=dict(type="gaussian", std=0.005),
                             bias_filler=dict(type="constant", value=0.1),
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu6 = L.ReLU(net.fc6, in_place=True)
    net.dropout6 = L.Dropout(net.relu6, dropout_param=dict(dropout_ratio=0.5), in_place=True)

    net.fc7 = L.InnerProduct(net.dropout6, num_output=4096
                             , weight_filler=dict(type="gaussian", std=0.005),
                             bias_filler=dict(type="constant", value=0.1),
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu7 = L.ReLU(net.fc7, in_place=True)
    net.drop7 = L.Dropout(net.relu7, dropout_param=dict(dropout_ratio=0.5), in_place=True)

    net.fc8 = L.InnerProduct(net.drop7, num_output=10,
                             weight_filler=dict(type="gaussian", std=0.01),
                             bias_filler=dict(type="constant", value=0),
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.accuracy = L.Accuracy(net.fc8, net.label, include=dict(phase=caffe_pb2.Phase.Value("TEST")))
    net.loss =  L.SoftmaxWithLoss(net.fc8, net.label)

    with open(net_filename, "a") as f:
        f.write(str(net.to_proto()))


if __name__ == "__main__":
    create_alexnet()
