class A():
    def __init__(self, a):
        self.a = a
        
    def get_a(self):
        return self.a
    
iA = A(4)
print(iA.a)

b = iA.get_a()
b = 5

print(iA.a)

iA.a = 5
print(iA.a)
        