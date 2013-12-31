import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math
import random

def scaleSpectrum(A):
   return numpy.real(numpy.log10(numpy.absolute(A) + numpy.ones(A.shape)))


def randomVector(seed, length):
   random.seed(secretKey)
   return [random.choice([0,1]) for _ in range(length)]


def applyWatermark(imageMatrix, watermarkMatrix, alpha):
   shiftedDFT = fftshift(fft2(imageMatrix))
   watermarkedDFT = shiftedDFT + alpha * watermarkMatrix

   #misc.imsave("images/watermarked-spectrum.png", scaleSpectrum(watermarkedDFT))
   watermarkedImage = ifft2(ifftshift(watermarkedDFT))

   return watermarkedImage


def makeWatermark(imageShape, radius, secretKey, vectorLength=50):
   watermark = numpy.zeros(imageShape)
   center = (int(imageShape[0] / 2) + 1, int(imageShape[1] / 2) + 1)

   vector = randomVector(secretKey, vectorLength)

   x = lambda t: center[0] + int(radius * math.cos(t * 2 * math.pi / vectorLength))
   y = lambda t: center[1] + int(radius * math.sin(t * 2 * math.pi / vectorLength))
   indices = [(x(t), y(t)) for t in range(vectorLength)]

   for i,location in enumerate(indices):
      watermark[location] = vector[i]

   return watermark


def decodeWatermark(image, secretKey):
   pass


if __name__ == "__main__":
   filename = "images/hance-up-sw.jpg"
   secretKey = 57846874321257
   alpha = 0

   theImage = ndimage.imread(filename, flatten=True)
   watermark = makeWatermark(theImage.shape, min(theImage.shape) / 4, secretKey=secretKey)

   misc.imsave("images/watermark-spectrum.png", watermark)

   watermarked = numpy.real(applyWatermark(theImage, watermark, alpha))
   misc.imsave("%s-watermarked-%.3f.png" % (filename.split('.')[0], alpha), watermarked)

