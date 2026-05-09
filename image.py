import matplotlib.pyplot as plt

def show_image(flat_image, label_name):

    image = flat_image.reshape(3, 32, 32)
    image = image.transpose(1, 2, 0)

    plt.imshow(image)
    plt.title(label_name)
    plt.show()

show_image(X[0], class_names[y[0]])