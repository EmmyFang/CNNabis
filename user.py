import torch
import os
from tkinter import *
from tkinter import filedialog
from PIL import Image
from torchvision import transforms
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def get_image(path, size):

    user_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ])
    image = plt.imread(path)
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = user_transforms(image)
    return image


def use(model, image):

    model.eval()
    image = torch.unsqueeze(image, 0)
    output = model(image)
    output = output.squeeze()
    out_prob = torch.nn.functional.softmax(output)
    output = output.detach().numpy()
    index = np.argsort(output)
    return out_prob, index


class Window(Frame):

    def __init__(self, master=None):

        Frame.__init__(self, master)
        self.master = master
        self.model = torch.load(os.getcwd() + '/model/best.pt', map_location='cpu')
        self.labels = [
            'bigleaf', 'box elder', 'cannabis', 'japanese maple', 'norway maple', 'red maple', 'silver maple',
            'sugar maple'
        ]
        self.text = Text(master)
        self.init_window()

    def init_window(self):

        self.master.title('CNNabis')
        self.pack(fill=BOTH, expand=1)
        self.text.insert(INSERT,
                         'Welcome to CNNabis. This software will help you distinguish \n'
                         '7 types of commonly found maple leaves and cannabis leaves.\n'
                         'Please press Begin and load your image.\n'
                         )
        self.text.pack()
        quit = Button(self, text='Exit', command=self.exit)
        quit.place(x=400, y=200)
        start = Button(self, text='Begin', command=self.get_file)
        start.place(x=200, y=200)

    def get_file(self):

        path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Please choose image of your leaf",
                                                   filetypes=(("jpeg files","*.jpg"),("png files","*.png")))
        image = get_image(path, 200)
        out_prob, index = use(self.model, image)
        predict = index[-1]
        sec_most_prob = index[-2]
        if predict == 2:
            self.text.insert(INSERT, 'Oh NO! I think this is a cannabis leaf (probability: {})'.format(
                format(np.around(out_prob[predict].detach().numpy(), decimals=3), '.3f')
            ) + '\n')
            self.text.insert(END, 'It might be a {} (probability {})'.format(
                self.labels[sec_most_prob], format(np.around(out_prob[sec_most_prob].detach().numpy(), decimals=3), '.3f')
            ) + '\n')
        else:
            self.text.insert(INSERT, 'I think this is a {} (probability {})'.format(
                self.labels[predict], format(np.around(out_prob[predict].detach().numpy(), decimals=3), '.3f')
            ) + '\n')
            self.text.insert(END, 'It might be a {} (probability {})'.format(
                self.labels[sec_most_prob], format(np.around(out_prob[sec_most_prob].detach().numpy(), decimals=3), '.3f')
            ) + '\n')

        return 0

    def exit(self):
        self.text.insert(END, 'Thanks for using CNNabis. Wish you a nice day.')
        exit()


if __name__ == '__main__':
    root = Tk()
    root.geometry('1280x720')
    app = Window(root)
    root.mainloop()
