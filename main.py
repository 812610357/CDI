import ioput as io
import matplotlib.pyplot as plt

data=io.readimage("./data/Lenna.png")





io.showimage(data)
io.writeimage(data,"./data/Lenna_gray.png")
