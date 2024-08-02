import dutils
dutils.init()
def test_save_histogram():
    vals = np.random.randn(1000)
    dutils.save_histogram(y=vals,x=vals, savename='test_histogram1.png')
    pass
if __name__ == '__main__':
    test_save_histogram()
