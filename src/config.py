'''
This class saves configuration options for networks
Could be transformed into a yaml
'''
class NetConfig():
    def __init__(self, gen_model, dis_model):
        # Generator
        self.gen_filters = 64             # number of filters in the bottommost layer
        self.gen_activation = 'relu'      # activation function [relu/lrelu/prelu/selu/tanh]
        self.gen_n_downsamples = 2        # number of downsampling layers in content encoder
        self.gen_n_res = 2                # number of residual blocks in content encoder/decoder
        self.gen_pad_type = 'reflection'  # padding type [zero/reflect]
        self.gen_norm = 'layer'           # normalization type
        self.gen_bias = False              # use bias
        self.gen_dropout = True           # use dropout on upsampling layers
        
        self.dis_filters = 64             # number of filters in the bottommost layer
        self.dis_norm = 'none'            # normalization layer [none/bn/in/ln]
        self.dis_activation = 'lrelu'     # activation function [relu/lrelu/prelu/selu/tanh]
        self.dis_n_layer = 2              # number of layers in D
        self.dis_bias = False
        self.dis_pad_type = 'reflection'  # padding type [zero/reflect]

        # Data options
        self.input_channels = 3           # number of image channels [1/3]

        # General Network config
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.decay_epoch = 60
        self.lmb = 10
        self.id_coef = 2
        self.init_weights = 'normal'
        self.loss_mode = 'bce'
        self.norm = 'instance'
        self.enc_sw = True
        self.dec_sw = False
        self.penalisation = 1            # 1 = no impact