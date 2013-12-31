import numpy
from scipy import misc
from scipy import ndimage
import math
import matplotlib.pyplot as plt


def ifftExample():
   shift = 4
   spectrum = numpy.zeros((513,513))
   spectrum[256,256] = 1
   spectrum[256,256 + shift] = 1
   spectrum[256,256 - shift] = 1

   plt.clf()
   plt.imshow(spectrum, cmap=plt.cm.gray)
   plt.show()


   A = numpy.fft.ifft2(spectrum)
   img = numpy.array([[numpy.absolute(x) for x in row] for row in A])

   plt.clf()
   plt.imshow(img, cmap=plt.cm.gray)
   plt.show()


def fourierSpectrumExample(filename):
   A = ndimage.imread(filename, flatten=True)

   unshiftedfft = numpy.fft.fft2(A)
   spectrum = numpy.log10(numpy.absolute(unshiftedfft) + numpy.ones(A.shape))
   misc.imsave("images/%s-spectrum-unshifted.png" % (filename.split('.')[0]), spectrum)

   shiftedFFT = numpy.fft.fftshift(numpy.fft.fft2(A))
   spectrum = numpy.log10(numpy.absolute(shiftedFFT) + numpy.ones(A.shape))
   misc.imsave("images/%s-spectrum.png" % (filename.split('.')[0]), spectrum)


# create a list of 2d indices of A in decreasing order by the size of the
# (real) entry of A that they index to
def sortedIndices(A):
   indexList = [(i,j) for i in range(A.shape[0]) for j in range(A.shape[1])]
   indexList.sort(key=lambda x: -A[x])
   return numpy.array(indexList)


def animation(filename):
   A = ndimage.imread(filename, flatten=True)

   # subtract the mean so that the DC component is zero
   A = A - numpy.mean(A)
   ffted = numpy.fft.fft2(A)

   magnitudes = numpy.absolute(ffted)
   frame = numpy.zeros(ffted.shape, dtype=numpy.complex)

   t = 0
   decreasingIndices = sortedIndices(magnitudes)

   # only process every other index because every frequency has the
   # same magnitude as its symmetric opposite about the origin.
   for i,j in decreasingIndices[::2]:
      wave = numpy.zeros(A.shape)

      entry = ffted[i,j]
      frame[i, j] = wave[i, j] = entry
      frame[-i, -j] = wave[-i, -j] = entry.conjugate()

      ifftFrame = numpy.fft.ifft2(numpy.copy(frame))
      ifftFrame = [[x.real for x in row] for row in ifftFrame]
      misc.imsave('frames/%06d.png' % t, ifftFrame)

      ifftWave = numpy.fft.ifft2(numpy.copy(wave))
      ifftWave = [[x.real for x in row] for row in ifftWave]
      misc.imsave('waves/%06d.png' % t, ifftWave)

      t += 1


#ifftExample()
fourierSpectrumExample('sherlock.jpg')
#animation('images/hance-up-sw.png')


