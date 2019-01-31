from keras.layers import Input, Conv3D, PReLU, Dense, Flatten, Conv3DTranspose, Reshape, concatenate, average
from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2

from custom_metrics import precision, euclidean_distance_loss


def autoencoder(part = "all", pretrained_model_path = None):

    """ Defines the autoencoder model.

    Parameters
    -------------
    


    Returns
    -------------
    keras training Model instance
    
    """

    if part not in ["all", "encoder", "decoder"]:
        raise ValueError("Parameter 'part' value not in allowed parameters. Set 'all' for Autoencoder model, 'encoder' for Encoder part of Autoencoder, and 'decoder' for Decoder part of Autoencoder.")


    # Encoder 
    input_vox = Input(shape=(20,20,20,1,), name = "input_ac")

    conv1 = Conv3D(96, 7, name = "conv1_ac")(input_vox)
    act1 = PReLU(name = "p_re_lu_1")(conv1)

    conv2 = Conv3D(256, 5, name = "conv2_ac")(act1)
    act2 = PReLU(name = "p_re_lu_2")(conv2)

    conv3 = Conv3D(384, 3, name = "conv3_ac")(act2)
    act3 = PReLU(name = "p_re_lu_3")(conv3)

    conv4 = Conv3D(256, 3, name = "conv4_ac")(act3)
    act4 = PReLU(name = "p_re_lu_4")(conv4)

    flat = Flatten(name = "flat_ac")(act4)
    dens = Dense(64, name = "dense1_ac")(flat) # Embedding layer

    if part == "encoder":
        encoder = Model(inputs = input_vox, outputs = dens)

        if pretrained_model_path:
            load_pretrained_weights(pretrained_model_path, encoder)
        
        return encoder

    # Decoder

    if part == "decoder":
        input_decoder = Input(shape = (64,), name = "input_decoder")
        dens2 = Dense(216, name = "dense2_ac")(input_decoder)
    else:
        dens2 = Dense(216, name = "dense2_ac")(dens)

    reshp = Reshape((6, 6, 6, -1), name = "reshape_ac")(dens2)

    deconv1 = Conv3DTranspose(256, 3, name = "deconv1_ac")(reshp)
    dact1 = PReLU(name = "p_re_lu_5")(deconv1)

    deconv2 = Conv3DTranspose(384, 3, name = "deconv2_ac")(dact1)
    dact2 = PReLU(name = "p_re_lu_6")(deconv2)

    deconv3 = Conv3DTranspose(256, 5, name = "deconv3_ac")(dact2)
    dact3 = PReLU(name = "p_re_lu_7")(deconv3)

    deconv4 = Conv3DTranspose(96, 7, name = "deconv4_ac")(dact3)
    dact4 = PReLU(name = "p_re_lu_8")(deconv4)

    out = Conv3DTranspose(1, 1, activation = "sigmoid", name = "output_ac")(dact4)

    if part == "decoder":
        decoder = Model(inputs = input_decoder, outputs = out)

        if pretrained_model_path:
            load_pretrained_weights(pretrained_model_path, decoder)

        return decoder

    # Autoencoder
    autoencoder = Model(inputs = input_vox, outputs = out)

    if pretrained_model_path:
        load_pretrained_weights(pretrained_model_path, autoencoder)

    return autoencoder


def mobile_net(pretrained_model_path = None):

    """Simple, lightweight convnet."""

    if pretrained_model_path:
        return load_model(pretrained_model_path, custom_objects = {"euclidean_distance_loss": euclidean_distance_loss})

    else:
        mobilenet_v2 = MobileNetV2(include_top = False, input_shape = (224,224,3))

        flat = Flatten(name = "flat")(mobilenet_v2.output)
        x = Dense(64, name = "dense_out")(flat)
        mobilenet_v2_regress = Model(inputs = mobilenet_v2.input, outputs = x)

        # leave only last 2 block trainable
        for layer in mobilenet_v2_regress.layers[:-22]:
            layer.trainable = False

        return mobilenet_v2_regress



def joint_model_shared():

    """Joint model with shared embedding layer between autoencoder and convnet. Train from scratch."""

    input_vox = Input(shape=(20,20,20,1,), name = "input_ac")
    conv1 = Conv3D(96, 7, name = "conv1_ac")(input_vox)
    act1 = PReLU(name = "p_re_lu_1")(conv1)
    conv2 = Conv3D(256, 5, name = "conv2_ac")(act1)
    act2 = PReLU(name = "p_re_lu_2")(conv2)
    conv3 = Conv3D(384, 3, name = "conv3_ac")(act2)
    act3 = PReLU(name = "p_re_lu_3")(conv3)
    conv4 = Conv3D(256, 3, name = "conv4_ac")(act3)
    act4 = PReLU(name = "p_re_lu_4")(conv4)
    flat = Flatten(name = "flat_ac")(act4)
    embedding_ac = Dense(64, name = "dense1_ac")(flat)


    mobilenet_v2 = MobileNetV2(include_top = False, input_shape = (224,224,3))
    flat_mobilenet = Flatten(name = "flat")(mobilenet_v2.output)
    embedding_mn = Dense(64, name = "dense_out")(flat_mobilenet)

    embedding = Dense(64, name = "embedding") # Embedding layer

    shared_embedding_ac = embedding(embedding_ac)
    shared_embedding_mn = embedding(embedding_mn)
    merged_embedding = average([shared_embedding_ac, shared_embedding_mn])

    dens2 = Dense(216, name = "dense2_ac")(merged_embedding)
    reshp = Reshape((6, 6, 6, -1), name = "reshape_ac")(dens2)
    deconv1 = Conv3DTranspose(256, 3, name = "deconv1_ac")(reshp)
    dact1 = PReLU(name = "p_re_lu_5")(deconv1)
    deconv2 = Conv3DTranspose(384, 3, name = "deconv2_ac")(dact1)
    dact2 = PReLU(name = "p_re_lu_6")(deconv2)
    deconv3 = Conv3DTranspose(256, 5, name = "deconv3_ac")(dact2)
    dact3 = PReLU(name = "p_re_lu_7")(deconv3)
    deconv4 = Conv3DTranspose(96, 7, name = "deconv4_ac")(dact3)
    dact4 = PReLU(name = "p_re_lu_8")(deconv4)
    out = Conv3DTranspose(1, 1, activation = "sigmoid", name = "output_ac")(dact4)

    joint_model = Model(inputs = [input_vox, mobilenet_v2.input], outputs = out)
    
    print("Setting non-trainable layers...")
    for layer in joint_model.layers[:-22]:
        print(layer.name)
            layer.trainable = False

    return joint_model


def load_pretrained_weights(pretrained_model_path, training_model, model_type = "autoencoder"):
    print("Loading pretrained model...")

    if model_type == "autoencoder":
        custom_objects = {"precision": precision}
    elif model_type == "convnet":
        custom_objects = {"euclidean_distance_loss": euclidean_distance_loss}

    pretrained_model = load_model(pretrained_model_path, custom_objects = custom_objects)
    
    for layer in training_model.layers:
        try:
            layer.set_weights(pretrained_model.get_layer(layer.name).get_weights())
            print("Setting pretrained weights. (layer : ", layer.name, ")")
        except ValueError:
            print("Pretrained model '", pretrained_model.name ,"'"," is missing layer: ", layer.name)