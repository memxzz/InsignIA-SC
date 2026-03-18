from io import StringIO

import mediapipe as mp
import os
import sys
import cv2
import math
import numpy as np
import json
import inspect
import keyboard
drawPile = []
fingers = {
    "THUMB": {"mag":0.5,"pos":[0,0]},
    "INDEX": {"mag":0.5,"pos":[0,0]},
    "MIDDLE": {"mag":0.5,"pos":[0,0]},
    "RING": {"mag":0.5,"pos":[0,0]},
    "PINKY": {"mag":0.5,"pos":[0,0]},
    "orientation": np.array([0.0,0.0,0.0]),
}
expressions = {}
expMove = {}
actualExpr = ""

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

expreJsonROUTE  = resource_path("expressions.json")
expreMoveJsonROUTE  = resource_path("expressionsMove.json")

with open(expreJsonROUTE, "r") as f:
    expressions = json.load(f)
with open(expreMoveJsonROUTE , "r") as f:
    expMove = json.load(f)

for letter in expressions:
    expressions[letter]["orientation"] = np.array(
        expressions[letter]["orientation"]
    )
#--------------------------------------    
handScale = 1
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
frame = {}
lastTarget = ""
point2Draw = [0,0]
moveTestAccerts = 0
#---------------------------------
def EXPdrawPoints(expr,expName,mo):
    if type(frame) != np.ndarray:
        return
    if not mo:
        return
    global lastTarget
    global point2Draw
    global moveTestAccerts
    points = expr["points"]
    h, w, _ = frame.shape        
    fx = fingers[expr["handler"]]["pos"][0]
    fy = fingers[expr["handler"]]["pos"][1]
    fingerX = int(fx *w)
    fingerY = int(fy *h)
    cv2.circle( #finger
        frame,
        (fingerX,fingerY), 
        radius = 8, 
        color=(0,0,255),
        thickness =-1
    )
    #print(fingerX,fingerY,x,y) 
    cv2.circle( #point
        frame,
        (point2Draw[0],point2Draw[1]), 
        radius = 8, 
        color=(255,0,0),
        thickness =-1
    )
    
def MCsetParams(expr,exprName,recalculate,movingOne):
    h, w, _ = frame.shape
    global lastTarget
    global point2Draw
    global moveTestAccerts
    if not movingOne: lastTarget = ""; return 
    if not "handler" in expr: return
    if not recalculate and lastTarget == exprName:return
    if not recalculate:lastTarget = exprName
    if not recalculate:moveTestAccerts = 0

    points = expr["points"] 
    point = points[moveTestAccerts]
    fx = fingers[expr["handler"]]["pos"][0]
    fy = fingers[expr["handler"]]["pos"][1]
    print(handScale)
    x = int(fx *w)+ int((point[0])/handScale)
    y = int(fy *h)+ int((point[1])/handScale)
    point2Draw = [x,y]
def goMoveCheck(exprName): #puede que optimize esta funcion
    h, w, _ = frame.shape
    expr = expressions[exprName]
    old = expr
    movingOne = False
    global lastTarget
    global point2Draw
    global moveTestAccerts
    moveExprName = ""
    pointRadius = 40
    #-------------------------
    if "lookFor" in expr:
        if expr["lookFor"] in expMove:
            old = expr
            moveExprName = expr["lookFor"]
            expr = expMove[expr["lookFor"]]
            
            movingOne = True
    else:
        if exprName in expMove:
            print("this case")
            expr = expMove[exprName]
            movingOne = True
            moveExprName = exprName
    #-------------------------      
    
    
    MCsetParams(expr,exprName,False,movingOne)
    if not "handler" in expr: return exprName,movingOne
    if not "points" in expr:  return exprName,movingOne
    print("ey")
    points = expr["points"]
    point = points[moveTestAccerts]
    handler = fingers[expr["handler"]]
    handlerPos = [
        int(handler["pos"][0]*w),
        int(handler["pos"][1]*h)
    ]
    point2handlerMag = magnitud2d({'x':handlerPos[0],'y':handlerPos[1]},{'x':point2Draw[0],'y':point2Draw[1]})
    if point2handlerMag <= pointRadius and moveTestAccerts+1 < len(points):
        moveTestAccerts += 1
        if moveTestAccerts > len(points):
            moveTestAccerts = len(point) #para prevenir un bug que odio
        MCsetParams(expr,exprName,True,movingOne)
        return "",True
    elif moveTestAccerts >= len(points)-1:
        return moveExprName,True #jeje
    EXPdrawPoints(expr,exprName,movingOne)
    if movingOne: exprName = ""
    return exprName,movingOne
    
def get_distance(expr, fingers,exprName):
    total = 0
    for finger in fingers:
        if finger == "orientation":
            continue
        diff = fingers[finger]["mag"] - expr[finger]["mag"]
        total += diff ** 2
    diff_vec = fingers["orientation"] - expr["orientation"]
    total += np.sum(diff_vec ** 2)
    return math.sqrt(total)


def get_closest_expression():
    best_match = None
    smallest_distance = float("inf")
    
    for name, expr in expressions.items():
        distance = get_distance(expr, fingers,name)
        
        if distance < smallest_distance:#ignore this error, he is stupid.
            smallest_distance = distance
            best_match = name
    best_match,movingOne = goMoveCheck(best_match)
    if best_match == None or type(best_match) != str:
        best_match = ""
    return best_match, smallest_distance
def magnitud2d(a,b):
    XX = (b["x"]-a["x"])**2
    YY = (b['y']-a['y'])**2
    sum = XX+YY
    return math.sqrt(sum)
def magnitud(a,b):
    XX = (b.x-a.x)**2
    YY = (b.y-a.y)**2
    ZZ = (b.z-a.z)**2
    sum = XX+YY+ZZ
    return math.sqrt(sum)
def rest2Vectors(a, b):
    return np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ])
def getHandOrientation(mark):
    markLand = mark.landmark
    mpHandLand = mp_hands.HandLandmark
    #WRIST
    A = markLand[mpHandLand["WRIST"]]
    B = markLand[mpHandLand["INDEX_FINGER_MCP"]]
    C = markLand[mpHandLand["PINKY_MCP"]]
    points = np.asarray([A,B,C])
    normal_vector = np.cross(rest2Vectors(points[2], points[0]), rest2Vectors(points[1], points[2]))
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector

def setFingersValue(results):
    if results.multi_hand_landmarks is None:
        #print("No results, cant continue.")
        return
    for mark in results.multi_hand_landmarks:
        global handScale
        markLand = mark.landmark
        mpHandLand = mp_hands.HandLandmark
        #WRIST
        wrist = markLand[mpHandLand["WRIST"]]
        #THUMB
        thumbPointUp = markLand[mpHandLand["THUMB_TIP"]]
        #INDEX
        indexPointUp = markLand[mpHandLand["INDEX_FINGER_TIP"]]
        indexPointDown = markLand[mpHandLand["INDEX_FINGER_PIP"]]
        indexPointMCP = markLand[mpHandLand["INDEX_FINGER_MCP"]]
        #MIDDLE
        middlePointUp = markLand[mpHandLand["MIDDLE_FINGER_TIP"]]
        middlePointDown = markLand[mpHandLand["MIDDLE_FINGER_PIP"]]
        #RING
        ringPointUp = markLand[mpHandLand["RING_FINGER_TIP"]]
        ringPointDown = markLand[mpHandLand["RING_FINGER_PIP"]]
        #PINKY
        pinkyPointUp = markLand[mpHandLand["PINKY_TIP"]]
        pinkyPointDown = markLand[mpHandLand["PINKY_PIP"]]
        pinkyPointMCP = markLand[mpHandLand["PINKY_MCP"]]
        handScale = magnitud(wrist, indexPointMCP)
        fingers["THUMB"]["mag"] = magnitud(thumbPointUp,pinkyPointMCP)/handScale
        fingers["THUMB"]["pos"] = [thumbPointUp.x,thumbPointUp.y]

        fingers["INDEX"]["mag"] = magnitud(indexPointUp,wrist)/handScale
        fingers["INDEX"]["pos"] = [indexPointUp.x,indexPointUp.y]

        fingers["MIDDLE"]["mag"] = magnitud(middlePointUp,wrist)/handScale
        fingers["MIDDLE"]["pos"] = [middlePointUp.x,middlePointUp.y]

        fingers["RING"]["mag"] = magnitud(ringPointUp,wrist)/handScale
        fingers["RING"]["pos"] = [ringPointUp.x,ringPointUp.y]

        fingers["PINKY"]["mag"] = magnitud(pinkyPointUp,wrist)/handScale
        fingers["PINKY"]["pos"] = [pinkyPointUp.x,pinkyPointUp.y]
        
        fingers["orientation"] = getHandOrientation(mark)
        print(fingers)
        upsideDown = False
        if wrist.y < indexPointDown.y:
            upsideDown = True
        
def keysCheck():
    global  moveTestAccerts
    if keyboard.is_pressed("3"):
        moveTestAccerts += 1
        print(moveTestAccerts)
    if keyboard.is_pressed("2"):
        moveTestAccerts -= 1
        print(moveTestAccerts)

with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.5,
) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _  = frame.shape
        frame = cv2.flip(frame,1)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_RGB)
        setFingersValue(results)
        actualExpr = get_closest_expression()[0]
        frame = cv2.putText(frame, actualExpr, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #keys
        keysCheck()
        if results.multi_hand_landmarks is not None:
            for mark in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(
                    frame,
                    mark,
                    mp_hands.HAND_CONNECTIONS,
                )

        cv2.imshow("muerte...",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break



cap.release()
cv2.destroyAllWindows()