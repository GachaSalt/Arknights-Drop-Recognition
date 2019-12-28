import cv2
import matplotlib.pyplot as plt

__all__ = ('plot', 'show')


def plot(data):
    """
    Plot a 1D sequence.
    :param data: 1D sequence
    """
    X = [i for i in range(len(data))]
    plt.xlim([0, len(X) - 1])
    plt.plot(X, data, '.-')
    plt.show()


def show(img, mode='BGR'):
    import matplotlib.pyplot as plt
    if mode.upper() == 'BGR':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(new_img)
        plt.show()
        return
    if mode.upper() == 'GRAY':
        plt.imshow(img, plt.cm.gray)
        plt.show()
        return
    plt.imshow(img)
    plt.show()
