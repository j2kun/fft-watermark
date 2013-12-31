import cmath
import numpy
import matplotlib.pyplot as plt
import random

def memoize(f):
   cache = {}

   def memoizedFunction(*args):
      if args not in cache:
         cache[args] = f(*args)
      return cache[args]

   memoizedFunction.cache = cache
   return memoizedFunction


@memoize
def omega(p, q):
   return cmath.exp((2.0 * cmath.pi * 1j * q) / p)


def pad(inputList):
   k = 0
   while 2**k < len(inputList):
      k += 1
   return numpy.concatenate((inputList, ([0] * (2**k - len(inputList)))))


def fft(signal):
   n = len(signal)
   if n == 1:
      return signal
   else:
      Feven = fft([signal[i] for i in xrange(0, n, 2)])
      Fodd = fft([signal[i] for i in xrange(1, n, 2)])

      combined = [0] * n
      for m in xrange(n/2):
         combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
         combined[m + n/2] = Feven[m] - omega(n, -m) * Fodd[m]

      return combined


def fft2d(matrix):
   fftRows = numpy.array([fft(row) for row in matrix])
   return numpy.array([fft(row) for row in fftRows.transpose()]).transpose()


def testCase():
   A = numpy.array([[0,0,0,0], [0,1,0,0], [0,0,0,0], [0,0,0,0]])
   for row in fft2d(A):
      print(', '.join(['%.3f + %.3fi' % (x.real, x.imag) for x in row]))


norm = lambda x: cmath.polar(x)[0]

# playing with images

def frequencyFilter(signal):
   for i in range(20000, len(signal)-20000):
      signal[i] = 0


def processWithNumpy(imageData):
   transformedImage = numpy.fft.fft2(imageData)
   frequencyFilter(transformedImage)

   cleanedImage = numpy.fft.ifft2(transformedImage)
   return numpy.array(cleanedImage, dtype=numpy.float64)


