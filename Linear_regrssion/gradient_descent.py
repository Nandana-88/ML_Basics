import pandas as pd
df=pd.read_csv("Salary_dataset.csv")
df.head()
df.info()
x=df["YearsExperience"]
y=df["Salary"]
w=0.0
b=0.0
alpha=0.01
num_itr=1000
for i in range(num_itr):
    y_prd=[]
    for val_x in x:
        y_prd.append(w*val_x+b)
    #intilizing the gradient 
    dw=0
    db=0
    n=len(x)
    for j in range(n):
        error=y_prd[j]-y[j]
        dw+=error*x[j]
        db+=error
    dw/=n
    db/=n
    #update
    w-=dw*alpha
    b-=db*alpha
#(Optional) Print cost or parameters periodically to observe convergence
   if i % 100 == 0:
      cost = 0.0
    for k in range(n):
        cost += (y_pred[k] - y[k]) ** 2
    cost/=(2 * n)
print(w,b,cost)
def prediction(inp):
    return w*inp+b
inp=int(input("ENTER THE YEAR EXPERIENCE"))
predicted_salary=prediction(inp)
print("predicted_salary is",predicted_salary)