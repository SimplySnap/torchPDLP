How to install and run PaPILO. Run the papilo_setup.py program in the packages folder by
    %run PDLP-AMD-RIPS/Packages/papilo_setup.py
Note that this will take almost 10 minutes.

Or if that doesn't work, build the executable manually like this.
First, clone the PaPILO github:
    git clone https://github.com/scipopt/papilo.git
  
In the papilo folder, make a folder called build to build the cmake files into. This must be done in the command line and not through a notebook cell:
    cd papilo
    mkdir build
    cd build
    cmake ..
  
Compile the PaPILO executable into the bin folder, this is the step that takes a while:
    make
  
Move the executable into the local bin folder so calling it works properly:
    sudo mv /content/papilo/build/bin/papilo /usr/local/bin/

Once this is done, running
    papilo --help
shows all the commands that can be run but the main one is presolve run by
    papilo presolve -f problem.mps -r reduced.mps -v reduced.postsolve
where problem.mps is the path to the origninal LP, reduced.mps is the path to output the presolved LP, and reduces.postsolve is the path to output the postsolving data.

You can also import the presolve function from prepostsolve.py and it will work with python to do presolving through subprocess.
