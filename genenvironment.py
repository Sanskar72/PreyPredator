# Imports
import copy
import numpy as np
import random
from collections import deque
size = 50

def genEnvironment():
    """_summary_
        Function to create the environment for the project that is a graph of nodes connected by edges.
        Picking nodes with degree less than 3, add an edge between it and one node within 5 steps forward or backward along the primary loop.  
        (So node 10 might get connected to node 7 or node 15, but not node 16.)
    Returns: Nodes dictionary which contains the data of the nodes its neighbours and the degree of that node which must be less than or equal to 3
        _type_: 2D Dictionary
    """
    nodes = dict()
    for i in range(size):
        temp = dict()
        temp["degree"] = 2
        temp["neighbours"] = [(i-1)%size,(i+1)%size]
        nodes[i] = temp

    variation = 5
    visited = set(np.arange(size))
    while len(visited)>0:
        i = random.choice(list(visited))
        # Checking for nodes with degree<3
        if nodes[i]["degree"] < 3:
            neighList = list()
            # Running the boundary=5 condition
            for j in range(-variation, variation+1):
                neigh3 = (i+j) % size
                invalids = [i, (i+1)%size,(i-1)%size]
                # Checking if the new neighbour fits the vriteria and adding it to the dictionary
                if nodes[neigh3]["degree"] < 3 and neigh3 not in invalids:
                    neighList.append(neigh3)
            if len(neighList)>0:
                ind = np.random.randint(len(neighList))
                nodes[i]["neighbours"].append(neighList[ind])
                nodes[neighList[ind]]["neighbours"].append(i)
                nodes[i]["degree"] += 1
                nodes[neighList[ind]]["degree"] += 1
                visited.remove(neighList[ind])
                visited.remove(i)
                #print(i, neighList[ind], "removed")
            else:
                visited.remove(i)
                #print(i, "removed")

    return nodes, size

def spawnCreatures():
    """_summary_
        Function for the initial spawning of the Predator, Agent and Prey

    Returns: The spawn location of predator, agent, and prey
        _type_: integer variables with node location
    """
    agent = np.random.randint(size)
    predator = np.random.randint(size-1)
    prey = np.random.randint(size-1)
    # Check if Predator Spawn is not the same as Agent Spawn
    if predator>=agent:
        predator = (predator + 1)%size    
    # Check if Agent Spawn is not the same as Prey Spawn
    if prey>=agent:
        prey = (prey + 1)%size

    return predator, agent, prey

def BFS(nodes, start, goal):
    """_summary_
        Performing Breadth First Search to reach to the goal node location using the shortest path
    Args:
        nodes (2D Dictionary): Dictionary with all the node information in the graph
        start (int): The stat node position i.e. the Predator Position for traversal
        goal (int): The goal node i.e Agent Position

    Returns: StatusCode if there is a successful way for the agent to reach its goal and the path which is to be followed
        _type_: Dictionary
    """
    # Stores indices of the location
    visited = dict()
    visited[start] = True
    startQ = deque()
    finalPath = list()
    
    # Mark the starting cell as visited and push it into the goal queue
    startQ.append([start])
    
    # Iterate while the queue is not empty
    while startQ:
        path = startQ.popleft()
        x = path[-1]
        if x == goal:
            if len(finalPath) == 0:
                finalPath.append(path)
            elif len(finalPath[-1]) == len(path):
                finalPath.append(path)
            elif len(finalPath[-1]) > len(path):
                finalPath.clear()
                finalPath.append(path)

        # Go to the adjacent nodes in the graph
        for i in range(nodes[x]["degree"]):
            childX = nodes[x]["neighbours"][i]
            if not visited.get(childX, False):
                newPath = list(path)
                newPath.append(childX)
                startQ.append(newPath)
                visited[childX] = True
                
    return {"statusCode": 200, "path":finalPath}

def preyMovement(nodes, preyPos):
    """_summary_
        Function to move the Prey inside the system
    Args:
        nodes (2D Dictionary): Dictionary with all the node information in the graph
        preyPos (int): Initial Spawn location of the prey

    Returns: New location of the prey after random movement
        _type_: int
    """
    nextSteps = copy.deepcopy(nodes[preyPos]["neighbours"])
    nextSteps.append(preyPos)
    # print(nextSteps)
    nextStep = random.choice(nextSteps)
    preyPos = nextStep
    return preyPos


def predatorMovement(agentPos, predatorPos, nodes):
    """_summary_
        Function for the movement of the Predator based on the Agent position
    Args:
        agentPos (int): The location of the agent on the graph
        predatorPos (int): The location of the predator on the graph
        nodes (2D Dictionary): Dictionary with all the node information in the graph

    Returns:
        _type_: _description_
    """
    bestNeigh = list()
    minLen = 50
    if agentPos != predatorPos:
        for neigh in nodes[predatorPos]["neighbours"]:
            paths = BFS(nodes, neigh, agentPos)["path"]
            path = random.choice(paths)
            if len(bestNeigh)==0:
                bestNeigh.append([neigh, len(path)])
                minLen = len(path)
            elif len(path) == minLen:
                bestNeigh.append([neigh, len(path)])
            elif len(path) < minLen:
                bestNeigh.clear()
                bestNeigh.append([neigh, len(path)])
                minLen = len(path)
            #print("BN: ",bestNeigh)
        
        predArr = random.choice(bestNeigh)
        predatorPos = predArr[0]
        return {"statusCode":200, "predatorPos":predatorPos}
        
    else:
        return {"statusCode": 400, "predatorPos":agentPos}

# # __Driver Code__
# nodes, size = genEnvironment()
# for i in range(50):
#     print(nodes[i]["neighbours"])



# for j in range(10000):
#     aa = list()
#     nodes, size = genEnvironment()
#     sum = 0
#     for i in range(50):
#         if nodes[i]["degree"] == 2:
#             sum += 1   
#     aa.append(sum)
# print(np.max(aa))
