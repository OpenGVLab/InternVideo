def image_aug(images, image_transform):
    # print(image_transform)
    # I've no idea what the fuck this is doing
    # TODO: Maybe remove the second view?
    global_transform = image_transform[0]
    # local_transform = image_transform[0][1]
    global_images_tensor = []
    # 2 GLOBAL views
    for i in range(2):
        global_images_tensor.append(global_transform(images).unsqueeze(0))
    return global_images_tensor
    # # 3 LOCAL VIEWS
    # local_images_tensor = []
    # for i in range(3):
    #     local_images_tensor.append(local_transform(images).unsqueeze(0))
    # return [global_images_tensor, local_images_tensor]