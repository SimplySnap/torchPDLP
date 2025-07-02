NAME          Testprob
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  EQN
COLUMNS
    X1        COST               1.0   LIM1               1.0
    X1        LIM2               1.0
    X2        COST               4.0   LIM1               1.0
    X2        EQN               -1.0
    X3        COST               9.0   LIM2               1.0
    X3        EQN                1.0
RHS
    RHS1      LIM1               5.0   LIM2              10.0
    RHS1      EQN                7.0
BOUNDS
 UP BND1      X1                 4.0
 LO BND1      X2                -1.0
 UP BND1      X2                 1.0
 FR BND1      X3
ENDATA