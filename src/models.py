def autoencoder(part = "all"):

    """ Defines the autoencoder model.

    Parameters
    -------------



    Returns
    -------------
    
    
    """

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
        return Model(inputs = input_vox, outputs = dens)

    # Decoder
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
        return Model(inputs = dens2, outputs = out)


    autoencoder = Model(inputs = input_vox, outputs = out)
    return autoencoder
