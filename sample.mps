NAME          JuMP_model
ROWS
 N  COST
 G  C1
 G  C2
COLUMNS
    X         COST                12.0   C1                   6.0
    X         C2                   7.0
    Y         COST                20.0   C1                   8.0
    Y         C2                  12.0
RHS
    RHS1      C1                 100.0
    RHS1      C2                 120.0
BOUNDS
 LO BND1      X                    0.0
 LO BND1      Y                    0.0
 UP BND1      Y                    3.0
ENDATA
