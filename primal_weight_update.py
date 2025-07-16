# Put this at the start of a new restart
# x_prev and y_prev are the initialization of the previous restart
# x = x^{n,0}        y = y^{n,0}
# x_prev = x^{n-1,0} y_prev = y^{n-1,0}
# where n is the current restart

omega = torch.sqrt(torch.sqrt(torch.linalg.norm(y_prev - y, 2) / torch.linalg.norm(x_prev - x, 2)) * omega)
tau = eta / omega
sigma = eta * omega
x_prev = x.clone()
y_prev = y.clone()
