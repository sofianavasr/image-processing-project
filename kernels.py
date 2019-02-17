import numpy as np

#gaussian kernels
kernel053 = np.matrix('24879 107973 24879; \
                       107973 468592 107973; \
                       24879 107973 24879')                       
kernel055 = np.matrix('2 212 922 212 2; \
                       212 24745 107391 24745 212; \
                       922 107391 466066 107391 922; \
                       212 24745 107391 24745 212; \
                       2 212 922 212 2')
kernel057 = np.matrix('0 0 0 0 0 0 0; \
                       0 2 212 922	212 2 0; \
                       0 212 24745 10739 24745	212 0; \
                       0 922 10739 466064 10739 922	0; \
                       0 212 24745 10739 24745	212 0; \
                       0 2 212 922 212 2 0; \
                       0 0 0 0 0 0 0')
kernel0511 = np.matrix('0 0 0 0 0 0 0 0 0 0 0; \
                        0 0 0 0 0 0 0 0 0 0 0; \
                        0 0 0 0 0 0 0 0 0 0 0; \
                        0 0 0 2 212	922 212 2 0 0 0; \
                        0 0 0 212 24745 10739 24745 212 0 0 0; \
                        0 0 0 922 10739 466064 10739 922 0 0 0; \
                        0 0 0 212 24745 10739 24745 212 0 0 0; \
                        0 0 0 2 212	922 212 2 0 0 0; \
                        0 0 0 0 0 0 0 0 0 0 0; \
                        0 0 0 0 0 0 0 0 0 0 0; \
                        0 0 0 0 0 0 0 0 0 0 0')
kernel13 = np.matrix('77847 123317 77847; \
                      123317 195346 123317; \
                      77847 123317	77847')
kernel15 = np.matrix('3765 15019 23792 15019 3765; \
                      15019 59912 94907 59912 15019; \
                      23792 94907 150342 94907 23792; \
                      15019 59912 94907 59912 15019; \
                      3765 15019 23792 15019 3765')
kernel17 = np.matrix('36 363 1446 0.002291 1446 363 36; \
                      363 3676 14662 23226 14662 3676 363; \
                      1446 14662 58488 92651 58488 14662 1446; \
                      2291 23226 92651 146768 92651 23226 2291; \
                      1446 14662 58488 92651 58488 14662 1446; \
                      363 3676 14662 23226 14662 3676 363; \
                      36 363 1446 2291 1446 363 36')
kernel111 = np.matrix('0 0 0 0 1 1 1 0 0 0 0; \
                       0 0 1 14 55 88 55 14 1 0	0; \
                       0 1 36 362 1445 2289 1445 362 36 1 0; \
                       0 14 362 3672 14648 23204 14648 3672 362 14 0; \
                       1 55 1445 14648 58433 92564 58433 14648 1445 55 1; \
                       1 88 2289 23204 92564 146632 92564 23204 2289 88 1; \
                       1 55 1445 14648 58433 92564 58433 14648 1445 55 1; \
                       0 14 362 3672 14648 23204 14648 3672 362 14 0; \
                       0 1 36 362 1445 2289 1445 362 36 1 0; \
                       0 0 1 14 55 88 55 14 1 0 0; \
                       0 0 0 0 1 1 1 0 0 0 0')
kernel153 = np.matrix('95332 118095 95332; \
                       118095 146293 118095; \
                       95332 118095 95332')
kernel155 = np.matrix('15026 28569 35391 28569 15026; \
                       28569 54318 67288 54318 28569; \
                       35391 67288 83355 67288 35391; \
                       28569 54318 67288 54318 28569; \
                       15026 28569 35391 28569 15026')
kernel157 = np.matrix('15 438 8328 10317 8328 438 15; \
                       438 12788 24314 3012 24314 12788 438; \
                       8328 24314 46228 57266 46228 24314 8328; \
                       10317 3012 57266 7094 57266 3012 10317; \
                       8328 24314 46228 57266 46228	24314 8328; \
                       438 12788 24314 3012 24314 12788 438; \
                       15 438 8328 10317 8328 438 15')
kernel1511 = np.matrix('2 1 47 136 259 32 259 136 47 1 2; \
                        1 72 322 939 1785 2212 1785 939 322 72 1; \
                        47 322 1443 4212 8008 9921 8008 4212 1443 322 47; \
                        136 939 4212 12297 2338 28963 2338 12297 4212 939 136; \
                        259 1785 8008 2338 44453 55067 44453 2338 8008 1785 259; \
                        32 2212 9921 28963 55067 68216 55067 28963 9921 2212 32; \
                        259 1785 8008 2338 44453 55067 44453 2338 8008 1785 259; \
                        136 939 4212 12297 2338 28963 2338 12297 4212 939 136; \
                        47 322 1443 4212 8008 9921 8008 4212 1443 322 47; \
                        1 72 322 939 1785 2212 1785 939 322 72 1; \
                        2 1 47 136 259 32 259 136 47 1 2')
#rayleight kernels
ray13Factor = 1.2995755734477323
ray13 = np.matrix('0 0 0; \
                   0 36787944 16417000; \
                   0 16417000 7326256')

#sobel kernels
sobelx = np.matrix('-1 0 1; -2 0 2; -1 0 1')
sobely = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')


""" img = np.matrix('180 180 50 14; 206 100 5 124; 194 68 197 251; 172 106 207 233; 180 88 179 209')

def conv(matrix, kernel, borderSize, borderType):
    shape = np.shape(matrix)
    rowsLimit = shape[0] - borderSize
    columnsLimit = shape[1] - borderSize
    convMatrix = np.copy(matrix)
    
    for i in range(borderSize, rowsLimit):
        for j in range(borderSize, columnsLimit):
            submatrix = matrix[i-borderSize:i+borderSize+1:1,j-borderSize:j+borderSize+1:1]              
            convMatrix[i,j] = np.sum(np.multiply(submatrix, kernel))
    
    if borderType == 3:
        finalMatrix = convMatrix
    else:
        finalMatrix = convMatrix[borderSize:rowsLimit:1, borderSize:columnsLimit:1]   
      
    return finalMatrix


gradientY = conv(img, sobely, 1, 3)
gradientX = conv(img, sobelx, 1, 3)
    
gradient = np.absolute(gradientX) + np.absolute(gradientY)

print(gradient) """