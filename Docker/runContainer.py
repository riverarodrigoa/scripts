import os
import sys
from pathlib import Path
import argparse

def checkPortsInUse():
    containersRunning = os.popen('docker container ls -q').read().strip('\n').split('\n')
    portsUsed = []
    if containersRunning != ['']:
        for container in containersRunning:
            p = os.popen(f"docker port {container}").read().strip('\n').split(':')
            portsUsed.append(p[1])

    newPort = 8888
    while str(newPort) in portsUsed:
        newPort += 1
    return newPort

def checkContainerName():
    path = Path().absolute()
    name = path.parts[-1]
    containerNames = os.popen('docker ps --format "{{.Names}}"').read().strip('\n').split('\n')
    newName = name
    counter = 1
    if containerNames != ['']:
        while newName in containerNames:
            newName += f'_{counter}'
            counter += 1
    return newName


if __name__ == '__main__':
    # Parsing arguments if they exist
    parser = argparse.ArgumentParser(description='Run a Dev container automatically.')
    parser.add_argument('-i', metavar='Image', dest='image', type=str, help='Name of the Docker Image.')
    parser.add_argument('-v', metavar='Version', dest='version', type=str, help='Version of the Docker Image.')
    parser.add_argument('-n', metavar='Name', dest='cname', type=str, help='Name of the container.')
    parser.add_argument('-p', metavar='Port', dest='port', type=str, help='Forwarded port of the container.')

    args = vars(parser.parse_args())

    image = 'figaro' if args['image'] is None else args['image']
    version = 0.6 if args['version'] is None else args['version']
    name = checkContainerName() if args['cname'] is None else args['cname']
    port = checkPortsInUse() if args['port'] is None else args['port']
    print(f'Starting a new docker container...')
    print(f'IMAGE: {image}:{version}\nNAME: {name}\tPORT: {port}')
    path = Path().absolute()
    os.system(f"docker run -it --rm -p {port}:8888 --name {name} -v  {path / '01_Preprocessing'}:/dsProject/01_Preprocessing/ -v  {path / '02_Modelling'}:/dsProject/02_Modelling/ -v  {path /'03_Models'}:/dsProject/03_Models/ -v  {path /'04_Reports'}:/dsProject/04_Reports/ -v  {path / 'data'}:/dsProject/data/ {image}:{version}")


    #docker run -it --rm -p %port%:8888 --name %name% -v  %cd%/01_Preprocessing/:/dsProject/01_Preprocessing/ -v  %cd%/02_Modelling/:/dsProject/02_Modelling/ -v  %cd%/03_Models/:/dsProject/03_Models/ -v  %cd%/04_Reports/:/dsProject/04_Reports/ -v  %cd%/data/:/dsProject/data/ %image%:%version%
# os.system(f'.\\run_container.bat {newPort}')

