import numpy as np

def softmax(data_in):
    x=np.array(data_in,dtype=np.float64)
    e=x-np.max(x)
    exp_e=np.exp(e)
    return exp_e/np.sum(exp_e)

np.random.seed(3)
# Number of inputs
N = 3
# Dimension of input
D = 4
# Empty list for input
X = []
# Create random input vectors x_n and append to X
for n in range(N):
  X.append(np.random.normal(size=(D,1)))
# Print out the list
print(X)

np.random.seed(0)

# Choose random values for the parameters
omega_q = np.random.normal(size=(D,D))
omega_k = np.random.normal(size=(D,D))
omega_v = np.random.normal(size=(D,D))
beta_q = np.random.normal(size=(D,1))
beta_k = np.random.normal(size=(D,1))
beta_v = np.random.normal(size=(D,1))

all_queries = []
all_keys = []
all_values = []


for x in X:
    query=beta_q+np.matmul(omega_q,x)
    key=beta_k+np.matmul(omega_k,x)
    value=beta_v+np.matmul(omega_v,x)

    all_queries.append(query)
    all_keys.append(key)
    all_values.append(value)


all_x_prime=[]
for n in range(N):
    all_km_qn=[]
    for key in all_keys:
        dot_product=np.dot(key.T,all_queries[n])
        all_km_qn.append(dot_product)
    attention=softmax(all_km_qn)
    print("Attentions for output ", n)
    print(attention)

    x_prime=np.zeros((D,1))
    for weight, value in zip(attention,all_values):
        x_prime+=weight*value
    all_x_prime.append(x_prime)




print("x_prime_0_calculated:", all_x_prime[0].transpose())
print("x_prime_1_calculated:", all_x_prime[1].transpose())
print("x_prime_2_calculated:", all_x_prime[2].transpose())













