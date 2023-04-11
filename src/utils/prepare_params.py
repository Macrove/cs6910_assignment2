def get_cnn_params(n_filters:int, filter_organisation:int, kernel_size:int, stride:int = 1, padding:int = 1):
    padding = 1
    cnn_params = []

    if n_filters < 32 and filter_organisation == 2:
        print("**********NOTE**************\nToo small filter size for \
              given filter organisation. Therefore changing filter organisation to 0")
        filter_organisation = 0
        
    cnn_params.append({
        "in_features": 3,
        "out_features": n_filters,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding
    })
    for i in range(1, 5):
        if filter_organisation == 0:
            cnn_params.append({
                "in_features": n_filters,
                "out_features": n_filters,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding
            })
        elif filter_organisation == 1:
            cnn_params.append({
                "in_features": cnn_params[-1]["out_features"],
                "out_features": int(cnn_params[-1]["out_features"] * 2),
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding
            })
        elif filter_organisation == 2:
            cnn_params.append({
                "in_features": cnn_params[-1]["out_features"],
                "out_features": int(cnn_params[-1]["out_features"] / 2),
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding
            })
    return cnn_params

    