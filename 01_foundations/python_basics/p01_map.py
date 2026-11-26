def multiply(x):
    return x * 10

numbers = [1,2,3,4]

result  = map(multiply,numbers)
print(result)
# <map object at 0x00000192B982ABF0>

print(list(result))
