class TreeNode:
    def __init__ (self, data): 
        self.left = None 
        self.right = None 
        self.data = data

def preOrder(rootNode):
    if rootNode != None:
        print(rootNode.data) 
        preOrder(rootNode.left) 
        preOrder(rootNode.right)

def inOrder(rootNode): 
    if rootNode != None:
        inOrder(rootNode.left) 
        print(rootNode.data) 
        inOrder(rootNode.right)

def postOrder(rootNode): 
    if rootNode != None:
        postOrder(rootNode.left) 
        postOrder(rootNode.right) 
        print(rootNode.data)

a = TreeNode(10) 
b = TreeNode(20) 
c = TreeNode(30) 
d = TreeNode(40) 
e = TreeNode(50) 
f = TreeNode(60) 
g = TreeNode(70) 
d.left = b 
d.right = f 
b.left = a 
b.right = c 
f.left = e 
f.right = g
print("Pre-order Traversal :")
preOrder(d)
print("In-order Traversal :") 
inOrder(d)
print("Post-order Traversal :") 
postOrder(d)
