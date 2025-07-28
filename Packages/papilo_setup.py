from subprocess import run
'''
Be aware, this will take like ten minutes to build the papilo executable.
'''

command = "rm -rf papilo && git clone https://github.com/scipopt/papilo.git && cd papilo && mkdir build && cd build && cmake .. && make && sudo mv /content/papilo/build/bin/papilo /usr/local/bin/ && papilo --help"
result = run(command, shell=True, check=True)
