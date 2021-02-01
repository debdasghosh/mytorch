import torch
import numpy as np
import matplotlib.pyplot as plt

""" 
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2.)
c = torch.tensor(3., requires_grad=True)
d = a + a * b
e = (d + c) + 3

e.backward()
print(a.grad)
print(b.grad)
print(c.grad)


grad = np.ones((2,2)) * np.asarray([[1.,1.],[1.,1.]])
print(grad)

grad = np.ones(()) * np.asarray([1.])
print(grad)

kk = np.ones((1,2,3))
print(np.reshape(kk, (-1)))


print("========")
arr1 = np.asarray([1,2,3,4])
arr2 = np.asarray([[1,1,1,1],[1,1,1,1]])
print(arr1 * arr2)
#print(np.pad(arr1, (0,4), mode='symmetric') )

print("#########")
mat = np.asarray([[1,2,3,4],[5,6,7,8]])
print( np.matmul( np.transpose(mat), mat) )
print(np.float_power(mat, -1)) 

"""
""" 
array1 = np.array([[1, 1], [2, 2], [3, 3]])
array2 = np.array([1, 2, 3])

shuffler = np.random.permutation(len(array1))
array1_shuffled = array1[shuffler]
array2_shuffled = array2[shuffler]

print(array1_shuffled)
print(array2_shuffled)

array1 = np.array([
    [5,3,8],
    [1,8,2],
    [7,6,2],
    [1,2,3],
    [10,50,1]

])

print(np.argmax(array1, axis=1)) 

"""

# input_size = 64
# kernel_size = 3
# stride = 2

# print(((input_size - kernel_size)//stride) + 1) 



# x = np.asarray([1,2,3,4,5,6,7,8,9])
# print(x[1:4])

# x = torch.randn(4, 3, 2)
# layer = torch.nn.Flatten(0)
# out = layer(x)
# print(out.shape)

# kk = np.load('autograder/hw2_autograder/weights/mlp_weights_part_b.npy', allow_pickle = True)
# print(kk[0].shape)
# print(kk[1].shape)
# print(kk[2].shape)




# letters = 'cpn'
# guess = 'champion'
# i = 0
# for w in guess:
#     if w == letters[i]:
#         i += 1
#     if i == len(letters):
#         break
# if i == len(letters):
#     print ("yes")
# else:
#     print("no")


# for x in thelist: 
#     if a%3: continue
#     yield x
#     a += 1



# def createGenerator(thelist):
#     a = 0
#     for x in thelist: 
#         if a%3: continue
#         yield x
#         a += 1
# for i in createGenerator(thelist):
#     if not a%3: a += 1
# kk = [a+1 for i in createGenerator(thelist) if not a%3][0]
# print(kk)
# thelist = [1,2,3,4,5,6,7]
# myList = createGenerator(thelist)
# for num in myList:
#     print(num)
# print(next(myList))
# print(next(myList))
# print(next(myList))
# print(next(myList))



# a = np.asarray([[1, 2, 3],[4, 5, 6]])
# print(a.shape)

# b = np.reshape(a, (1,2,3))
# print(b.shape)
# a = np.asarray([[[1, 2, 3],[4, 5, 6]]])
# print(a.shape)
# a = np.asarray([[[1, 2, 3]],[[4, 5, 6]]])
# print(a.shape)
# a = np.asarray([[[1],
#             [2],
#             [3]],
            
#             [[4],
#             [5],
#             [6]]])
# print(a.shape)

# myList = np.asarray([5,8,2,9])
# ranked = np.argsort(myList) # 2,0,1,3
# print(ranked[::-1])
# print(myList[3:4])


# x=torch.randn(35, 64, 128)
# print(x.shape)
# batch_size = x.size(1)
# seq_len = x.size(0)
# feature_dim = x.size(2)

# if x.shape[0] % 2 > 0:
#     x = x[:-1]

# x = x.contiguous().view(int(seq_len/2),batch_size,feature_dim*2)
# print(x.shape)

torch.manual_seed(1234)
sequence_length = 6
batch_size = 2
embedding_size = 4
x = torch.arange(sequence_length*batch_size*embedding_size).reshape(sequence_length, batch_size, embedding_size)
y = x.permute(1,0,2)
y = y.contiguous().view(batch_size,int(sequence_length/2),embedding_size*2)
y = y.permute(1,0,2)
print(y)

batch_size = x.size(1)
seq_len = x.size(0)
feature_dim = x.size(2)

#int(seq_len/2),batch_size,feature_dim*2


for i in range(int(seq_len/2)):
    cnt = i * 2
    if i == 0:
        c = torch.cat((x[cnt,:,:], x[cnt+1,:,:] ), dim=1)
    else:
        c = torch.cat((c, torch.cat((x[cnt,:,:], x[cnt+1,:,:] ), dim=1) ))
c = c.reshape(int(seq_len/2),batch_size,feature_dim*2)
print(c)


'''
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7]],

        [[ 8,  9, 10, 11],
         [12, 13, 14, 15]],

        [[16, 17, 18, 19],
         [20, 21, 22, 23]],

        [[24, 25, 26, 27],
         [28, 29, 30, 31]],

        [[32, 33, 34, 35],
         [36, 37, 38, 39]],

        [[40, 41, 42, 43],
         [44, 45, 46, 47]]])

tensor([[[ 0,  1,  2,  3,  8,  9, 10, 11],
         [ 4,  5,  6,  7, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 24, 25, 26, 27],
         [20, 21, 22, 23, 28, 29, 30, 31]],
        [[32, 33, 34, 35, 40, 41, 42, 43],
         [36, 37, 38, 39, 44, 45, 46, 47]]])
'''

