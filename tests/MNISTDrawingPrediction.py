import tkinter
from PIL import Image, ImageDraw
import numpy as np
from sys import path
path.insert(1, "..")
from sobek.network import network



class Sketchpad(tkinter.Canvas):
    def __init__(self, parent, predictionLabel, **kwargs, ):
        super().__init__(parent, **kwargs)
        self.bind("<Button-3>", self.test)
        self.bind("<B1-Motion>", self.add_line)
        self.PILImage = Image.new("F", (560, 560), 100)
        self.draw = ImageDraw.Draw(self.PILImage)
        self.MNISTNN = network.networkFromFile("MNISTtest2")
        self.predictionLabel = predictionLabel

    def add_line(self, event):
        self.create_oval((event.x+24, event.y+24, event.x-24, event.y-24), fill="black")
        self.draw.ellipse([event.x-24, event.y-24, event.x+24, event.y+24], fill="black")
        smallerImage = self.PILImage.reduce(20)
        imageAsArray = np.array(smallerImage.getdata())
        imageAsArray = (100 - imageAsArray)/100
        predictionArray = self.MNISTNN.process(imageAsArray)
        print(predictionArray)
        prediction = np.argmax(predictionArray)
        self.predictionLabel['text'] =  ( "Predicted number : "  + str(prediction) + " with confidence : " + str(predictionArray[prediction]))

    def test(self, event):
        self.PILImage = Image.new("F", (560, 560), 100)
        self.draw = ImageDraw.Draw(self.PILImage)
        self.delete("all")

window = tkinter.Tk()
window.title("Number guesser")
window.resizable(False, False)
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)

predictionLabel = tkinter.Label(window, text="Predicted number :")

sketch = Sketchpad(window, predictionLabel, width=560, height=560)
sketch.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))
predictionLabel.grid(column=0, row=1)

window.mainloop()