import networkx
import tweepy
import json
from tweepy import OAuthHandler
import os
import csv
import matplotlib.pyplot as plt

# 1: cyberbullying;  categroy[0] is  normal; categroy[1] is cyberbullying
# record users' retweet id and times based on userInfoWithLabelAndRetweetList.csv
consumer_key = 'T35ai1b9pf9vxeUcYhhqEuAU3'
consumer_secret = 'IDQdQWs3TGgeqElIQYXYPOiICyUmYj5wgg5qf15Hstp2rrO1Kb'
access_token = '1311423992125816832-F7a4EbYykGTTzvsvzquVvqZEGdogfZ'
access_secret = 'Hyh8w6To0et3XlWlkodlUaDTQjdGJpDsSN40ljatTCiyW'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def generateUserRetweetInfo():
    userRetweetInfo = {}  # record users and retweets,key: user, value: retweetlist
    userFile = open('userInfoWithLabelAndRetweetList.csv')
    userLine = userFile.readline()
    while userLine:
        userLine = userLine.strip('\n')
        info = userLine.split(',')
        if info[2]:
            retweetlist = info[2].split('&')
            if info[0] in userRetweetInfo.keys():
                templist = userRetweetInfo.get(info[0]) + retweetlist
                userRetweetInfo[info[0]] = templist
            else:
                userRetweetInfo[info[0]] = retweetlist
        userLine = userFile.readline()
    userFile.close()
    # print(len(userRetweetInfo))
    # print(userRetweetInfo)
    return userRetweetInfo


def generateUserCateInfo():
    userCateInfo = {}  # key:user, value: category(cyberbullying or normal)
    with open('userInfoWithLabelAndRetweetList.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for info in csv_reader:
            if info[0] in userCateInfo.keys():
                userCateInfo[info[0]].append(info[1])  # 2: normal; 0,1 : cyberbullying
            else:
                userCateInfo[info[0]] = [info[1]]
    user2label = {}
    for id in userCateInfo.keys():
        normal = 1
        bully = 1
        label_list = userCateInfo[id]
        for label in label_list:
            if label == '2':
                normal = normal + 1
            else:
                bully = bully + 1
        if normal > bully:  # normal:1
            user2label[id] = 1
        else:  # bully:0
            user2label[id] = 0
    with open("user_label.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["userId", "label"])
        for id in user2label.keys():
            write.writerow([id, user2label[id]])

    return user2label


def generateNodes():
    # add nodes
    NODES = {}  # dict
    nodefile = open('userInfoWithUserName.csv')
    nodeline = nodefile.readline()
    while nodeline:
        nodeline = nodeline.strip('\n')
        nodeline = nodeline.split(',')
        NODES[nodeline[0]] = nodeline[1]
        nodeline = nodefile.readline()
    nodefile.close()
    return NODES


def getFollowerInfo():
    followerInfo = {}  # key: userid, value:
    with open('followersInfo.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='&')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count + 1
            else:
                row[0] = row[0]
                row[1] = row[1]
                followerlist = list(row[1].split(','))
                followerInfo[row[0]] = followerlist
                line_count = line_count + 1
    return followerInfo


# generate undirected graph
def generateEdgesGraph(Nodes, followerInfo, userRetweetInfo):
    # add edges
    EDGES = {}
    i = 0
    for x in Nodes.keys():
        if x in followerInfo.keys():
            followerList = followerInfo[x]
            followers = set([str(f) for f in followerList])

        retweetList = userRetweetInfo.get(x)
        for y in Nodes.keys():
            weight = 1
            retweight = 0
            if retweetList and y in retweetList:
                retweight = 1
            if y in followers:
                if (y, x) in EDGES.keys():
                    val = EDGES[(y, x)]
                    EDGES[(y, x)] = val + max(weight, weight + retweight)
                else:
                    EDGES[(x, y)] = max(weight, weight + retweight)
    return EDGES


def generateEdgesDigraph(Nodes, followerInfo, userRetweetInfo):
    # add edges
    EDGES = set()
    i = 0
    for x in Nodes.keys():
        if x in followerInfo.keys():
            followerList = followerInfo[x]
            followers = set([str(f) for f in followerList])

        retweetList = userRetweetInfo.get(x)
        for y in Nodes.keys():
            weight = 1
            retweight = 0
            if retweetList and y in retweetList:
                retweight = 1
                # EDGES.add((x, y, retweight))
            if y in followers:
                EDGES.add((x, y, max(weight, weight + retweight)))
            else:
                if retweetList and y in retweetList:
                    EDGES.add((x, y, retweight))
    # print(len(EDGES))
    return EDGES


def convertEdgeDicToSet(edgeDic):
    EDGES = set()
    # print(edgeDic.keys())
    for edge in edgeDic.keys():
        EDGES.add((edge[0], edge[1], edgeDic[edge]))
    return EDGES


def generateGraph(nodes, edges):
    GRAPH = networkx.Graph()
    for nodeId in nodes.keys():
        GRAPH.add_node(nodeId)
    GRAPH.add_weighted_edges_from(edges)
    writeGraphToFile(edges)
    # print(GRAPH.edges)
    return GRAPH


def generateDigraph(nodes, edges):
    GRAPH = networkx.DiGraph()
    for nodeId in nodes.keys():
        GRAPH.add_node(nodeId)
    GRAPH.add_weighted_edges_from(edges)
    # print(GRAPH.edges)
    # print(len(GRAPH.edges))
    return GRAPH


def writeGraphToFile(edges):
    with open("socialNetwork_graph.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["point_from", "point_to", "weight"])
        for x in edges:
            write.writerow([x[0], x[1], x[2]])


def main():
    # Nodes = generateNodes()
    # userRetweetInfo = generateUserRetweetInfo()
    # followInfo = getFollowerInfo()
    # Edges = generateEdgesGraph(Nodes,followInfo,userRetweetInfo)
    # Edges = convertEdgeDicToSet(Edges)
    # G = generateGraph(Nodes, Edges)
    print('generate user label')
    generateUserCateInfo()


if __name__ == "__main__":
    main()
