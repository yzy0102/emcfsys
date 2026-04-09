def upsert_image_layer(viewer, data, name: str):
    if name in viewer.layers:
        viewer.layers[name].data = data
        return viewer.layers[name]
    return viewer.add_image(data, name=name)


def upsert_labels_layer(viewer, data, name: str):
    if name in viewer.layers:
        viewer.layers[name].data = data
        return viewer.layers[name]
    return viewer.add_labels(data, name=name)
