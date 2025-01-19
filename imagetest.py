import numpy
import matplotlib.pyplot as plt


def normalize(A):
    return numpy.interp(A, (A.min(), A.max()), (0, 1))


def ifftExample():
    shift = 4
    n = 1024
    spectrum = numpy.zeros((n + 1, n + 1))
    spectrum[n // 2, n // 2] = 1
    spectrum[n//2, n//2 + shift] = 1
    spectrum[n//2 , n//2 - shift] = 1

    plt.imsave("images/example_spectrum.png", spectrum, cmap=plt.cm.gray)
    A = numpy.fft.ifft2(spectrum)
    img = numpy.array([[numpy.absolute(x) for x in row] for row in A])
    plt.imsave("images/example_spatial_domain.png", img, cmap=plt.cm.gray)


def fourierSpectrumExample(filename):
    A = plt.imread(filename)

    unshiftedfft = numpy.fft.fft2(A)
    spectrum = numpy.log10(numpy.absolute(unshiftedfft) + numpy.ones(A.shape))
    spectrum = normalize(spectrum)
    plt.imsave(
        "images/%s-spectrum-unshifted.png" % (filename.split("/")[-1].split(".")[0]),
        spectrum,
    )

    shiftedFFT = numpy.fft.fftshift(numpy.fft.fft2(A))
    spectrum = numpy.log10(numpy.absolute(shiftedFFT) + numpy.ones(A.shape))
    spectrum = normalize(spectrum)
    plt.imsave(
        "images/%s-spectrum.png" % (filename.split("/")[-1].split(".")[0]),
        spectrum,
    )


# create a list of 2d indices of A in decreasing order by the size of the
# (real) entry of A that they index to
def sortedIndices(A):
    indexList = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])]
    indexList.sort(key=lambda x: -A[x])
    return numpy.array(indexList)


def animation(filename):
    A = plt.imread(filename)

    # subtract the mean so that the DC component is zero
    A = A - numpy.mean(A)
    ffted = numpy.fft.fft2(A)

    magnitudes = numpy.absolute(ffted)
    frame = numpy.zeros(ffted.shape, dtype=numpy.complex)

    t = 0
    decreasingIndices = sortedIndices(magnitudes)

    # only process every other index because every frequency has the
    # same magnitude as its symmetric opposite about the origin.
    for i, j in decreasingIndices[::2]:
        wave = numpy.zeros(A.shape)

        entry = ffted[i, j]
        frame[i, j] = wave[i, j] = entry
        frame[-i, -j] = wave[-i, -j] = entry.conjugate()

        ifftFrame = numpy.fft.ifft2(numpy.copy(frame))
        ifftFrame = [[x.real for x in row] for row in ifftFrame]
        ifftFrame = normalize(ifftFrame)
        plt.imsave("frames/%06d.png" % t, ifftFrame)

        ifftWave = numpy.fft.ifft2(numpy.copy(wave))
        ifftWave = [[x.real for x in row] for row in ifftWave]
        ifftWave = normalize(ifftWave)
        plt.imsave("waves/%06d.png" % t, ifftWave)

        t += 1


ifftExample()
# fourierSpectrumExample("images/sherlock.jpg")
# animation('images/hance-up-sw.png')
