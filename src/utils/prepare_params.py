def get_cnn_params(n_filters, filter_organisation, kernel_size, stride, padding):
    cnn_params = []
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
                "out_features": cnn_params[-1]["out_features"] * 2,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding
            })
        elif filter_organisation == 2:
            cnn_params.append({
                "in_features": cnn_params[-1]["out_features"],
                "out_features": cnn_params[-1]["out_features"] / 2,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding
            })
    return cnn_params

    