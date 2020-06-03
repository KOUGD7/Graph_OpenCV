__author__  = """KOUGD7 """; __email__   = "cougahdere@yahoo.com"; __date__    = "03/2020";
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
import math

# Find Labels using conected component
def get_connected_components(I, min, max):
    # Convert to gray
    Igray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    # Threshold
    ret, Ithresh = cv2.threshold(Igray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep only small components but not to small
    output = cv2.connectedComponentsWithStats(Ithresh)

    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    labelStats = output[2]
    labelAreas = labelStats[:, 4]

    rects = []  # List, its length is the number of CCs
    num_labels2 = num_labels
    labels2 = labels
    stats2 = []
    centroids2 = []

    for i in range(1, len(centroids)):

        x0 = stats[i][0]
        y0 = stats[i][1]

        w = stats[i][2]
        h = stats[i][3]

        r = [(0, 0), (0, 0)]
        if min <= labelAreas[i] <= max:
            # r = Rectangle(x0, y0, x0 + w, y0 + h)
            r = [(x0, y0), (x0 + w, y0 + h)]
            rects.append(r)

    # A numpy array of size (n, 2) where n is the number of CCs
    centroids_ = np.ndarray((centroids.shape[0] - 1, centroids.shape[1]))
    for i in range(1, len(centroids)):
        centroids_[i - 1] = centroids[i]

    # get center from centroid
    for point in centroids_:
        x = point[0]
        y = point[1]
        center = (int(x), int(y))
        # cv2.circle(img, center, 2, (0, 0, 255), 3)

    # get rectangle from array
    for point in rects:
        start = point[0]
        end = point[1]
        #cv2.rectangle(img, start, end, (250, 255, 2), 2)

    # remove all labels
    sizes = stats[1:, -1];
    nb_components = num_labels - 1
    output = labels
    img2 = np.zeros((output.shape))

    img3=img2.copy()
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if not (min <= sizes[i] <= max):
            #IMAGE WITHOUT LABELS
            img2[output == i + 1] = 255
        else:
            #IMAGE WITH ONLY LABELS
            img3[output == i + 1] = 255


    return rects, centroids2, img2, img3  # , num_labels2, labels2, stats2


def get_states(I, min, max, off):
    # Convert to gray
    Igray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    # Threshold
    ret, Ithresh = cv2.threshold(Igray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Reduce thickness of the lines
    kernel = np.ones((2, 2), np.uint8)
    Ithresh = cv2.dilate(Ithresh, kernel, iterations=1)
    Ithresh = cv2.erode(Ithresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(Ithresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #contours = [contours[i] for i in range(len(contours)) if hierarchy[i][3] < 0]

    quality = 0.78

    i = 0
    radii = []
    centres = []
    contours1 = []
    # Find all the contours in the iamge
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.0001 * cv2.arcLength(contour, True), True)
        cv2.drawContours(I, [approx], 0, (0, 255, 0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        # if len(approx) <= 36000 and len(approx) >0:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        # print(aspectRatio)

        # ratio of the states high to width
        if aspectRatio >= 0.60 and aspectRatio <= 1.40 and len(approx) > 20:
            i = i + 1

            # get the bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            # draw a green rectangle to visualize the bounding rect
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # get the min area rect
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            # cv2.drawContours(img, [box], 0, (0, 0, 255))

            # finally, get the min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # convert all values to int
            center = (int(x), int(y))
            radius = int(radius)

            areaCon = cv2.contourArea(contour)
            areaCir = np.pi * (radius ** 2)

            eff = areaCon / areaCir
            # print(eff)

            # and draw the circle in blue
            if (min <= radius <= max) and eff > quality:
                centres.append(center)
                radii.append(radius)
                contours1.append(contour)

    # draw circle
    circles = list(zip(radii, centres, contours1))
    circles.sort(key=lambda x: 10 * (x[1][1] + x[1][0]) + x[0])

    #       [r,(x,y)]
    prev = [0, (0, 0)]
    state = 1

    # capture circles after filtering unwanted circles
    radii2 = []
    centres2 = []
    contours2 = []
    for radius, centre, contour in circles:
        x, y = centre
        # print (prev)
        px = prev[1][0]
        py = prev[1][1]
        pr = prev[0]

        # remove circle that that are close has off
        if not ((px - off < x < px + off) and (py - off < y < py + off) and (pr - off < radius < pr + off)):
            cv2.circle(I, centre, radius, (255, 0, 0), 2)
            cv2.circle(I, centre, 2, (0, 0, 255), 1)
            cv2.putText(I, "S " + str(state), centre, cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))
            state += 1
            centres2.append(centre)
            radii2.append(radius)
            contours2.append(contour)
            pass
        prev = radius, centre

    return radii2, centres2, contours2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_arrows(binary):
    Ithresh = np.uint8(binary)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(Ithresh)

    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    img2 = np.zeros((output.shape))

    #remove component less than 50
    for i in range(0, nb_components):
        if not (50 >= sizes[i]):
            img2[output == i + 1] = 255

    # Closing possible gaps Arrows
    Ithresh = np.uint8(img2)
    kernel = np.ones((3, 3), np.uint8)
    Ithresh = cv2.morphologyEx(Ithresh, cv2.MORPH_CLOSE, kernel)

    thrash2 = np.float32(Ithresh)

    # Arrow Detection
    kernel = np.ones((2, 2), np.uint8)
    Ithresh = cv2.morphologyEx(Ithresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(Ithresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0

    Arrows = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # center of mass
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.circle(img, (cx, cy), 3, (200, 10, 200), -1)

        # compute the rotated bounding box of the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        # box = np.int0(box)
        # draw a red 'nghien' rectangle
        # .drawContours(img, [box], 0, (0, 0, 255))

        # www.pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)

        # unpack the ordered bounding box, then compute the
        # midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-right and
        # bottom-right

        (tl, tr, br, bl) = box
        if dist.euclidean(tl, tr) < dist.euclidean(tl, bl):
            # midpoint of narrow end
            (topX, topY) = midpoint(br, bl)
            (bottomX, bottomY) = midpoint(tr, tl)
            # midpoint of long end
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
        else:
            # midpoint of narrow end
            (topX, topY) = midpoint(tl, bl)
            (bottomX, bottomY) = midpoint(tr, br)
            # midpoint of long end
            (tlblX, tlblY) = midpoint(br, bl)
            (trbrX, trbrY) = midpoint(tr, tl)

        if dist.euclidean((cx, cy), (topX, topY)) < dist.euclidean((cx, cy), (bottomX, bottomY)):
            start_point = (bottomX, bottomY)
            end_point = (topX, topY)
            pass
        else:
            start_point = (topX, topY)
            end_point = (bottomX, bottomY)

        #convert elements to int
        start_point = tuple(int(t) for t in start_point)
        end_point = tuple(int(t) for t in end_point)

        cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), 3, 5)
        cv2.circle(img, (cx, cy), 3, (200, 10, 200), -1)

        Arrows.append((start_point, end_point))
    return Ithresh, Arrows


def compareLables(alpha1, label1):
    #np.resize(alpha, (200, 200*m))
    #np.resize(label, (200, 200 * m))
    #print(m)
    alpha, recA = alpha1
    label, recL = label1

    upperCornerA, lowerCornerA = recA
    xA, yA = upperCornerA
    xwA, yhA = lowerCornerA

    upperCorner, lowerCorner = recL
    x, y = upperCorner
    xw, yh = lowerCorner

    xScale  = (xw - x)/(xwA - xA)
    yScale = (yh - y) / (yhA - yA)
    #print((xScale, yScale))

    alpha = alpha.copy()
    alpha = cv2.resize(alpha, (0, 0), fx=xScale, fy=yScale)

    kernel = np.ones((3, 3), np.uint8)
    #alpha = cv2.dilate(alpha, kernel, iterations=1)
    #label=  cv2.dilate(label, kernel, iterations=1)

    intersect = cv2.bitwise_and(alpha, label)
    #intersect = cv2.morphologyEx(intersect, cv2.MORPH_OPEN, kernel)

    intersect = np.uint8(intersect)
    """cv2.imshow('TEST', alpha)
    cv2.imshow('TEST1', label)
    cv2.imshow('TEST2', intersect)

    key = cv2.waitKey(3000)  # pauses for 3 seconds before fetching next image
    if key == 27:  # if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        #break"""

    output = cv2.connectedComponentsWithStats(intersect)

    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    labelStats = output[2]
    labelAreas = labelStats[:, 4]

    height, width = intersect.shape

    #print([labelAreas, (height * width)])
    simularity_idex = sum(labelAreas[1:])/(height * width)

    return simularity_idex


def detect_alphabet(labels, alphabet, alpharange):
    # TESTING FOR TEMPLATE MATCHING
    rectsL, centroidsL, _, imgL = labels
    imgL = np.array(imgL, dtype=np.uint8)

    alphaimg = alphabet

    #THIS IS PRINT TO MAIN IMG. FIX THIS!!!!!!!!!!!!!!!!!!!!!!!
    AA = get_connected_components(alphaimg, 0, alpharange)
    rectsAA, centroidsAA, _, imgAA = AA
    imgAA = np.array(imgAA, dtype=np.uint8)


    templates = []
    subimages = []
    for rect in rectsAA:
        upperCorner, lowerCorner = rect
        x, y = upperCorner
        xw, yh = lowerCorner
        templates.append((imgAA[y:yh, x:xw], rect))


    for rec in rectsL:
        upperCorner, lowerCorner = rec
        x, y = upperCorner
        xw, yh = lowerCorner
        subimages.append((imgL[y:yh, x:xw], rec))

    mapping = {}
    newRecs = []
    countS = 0
    for s in subimages:
        countT = 0
        maxIndex = 0
        alphaindex = -1
        for t in templates:
            mapping[countT] = t
            #subT, rT = t

            sindex = compareLables(t, s)
            #print(["SI: ",sindex])

            if sindex > maxIndex:
                maxIndex = sindex
                alphaindex = countT
            countT+=1

        subS, recS = s
        upperCorner, lowerCorner = recS
        cv2.putText(img, "L" + str(alphaindex), upperCorner, cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))

        newRecs.append((alphaindex, recS))
        countS+=1

    #print(("num alphabet/ numberlabels", len(templates), len(subimages)))

    cv2.imshow('ConnectL', imgL)
    cv2.imshow('ConnectAA', imgAA)
    return mapping, newRecs


#CLASSES FOR REPRESENTATION OF DFA/ GRAPH
class State:
    def __init__(self, r, c):
        self.radius = r
        self.centre = c
        self.out_arrows = {}
        self.accept = False

    """def __init__(self, r, c, l):
        self.radius = r
        self.centre = c
        self.label = l
        self.out_arrows = []
        self.accept = False"""

    def add_arrow(self, key, value):
        self.out_arrows[key] = value

    def set_final(self):
        self.accept = True

class Arrow:
    def __init__(self, t, h):
        self.tail = t
        self.head = h
        self.label = None
        self.next = None

    def get_mid(self):
        return midpoint(self.tail, self.head)

    def add_next(self, s):
        self.next = s

    def add_label(self, l):
        self.label = l


class Label:
    def __init__(self, v, r):
        self.value = v
        self.rec = r

    def get_centre(self):
        start, end = self.rec
        return midpoint(start, end)

    def min_circle(self):
        #print(self.rec)
        centre, radius = cv2.minEnclosingCircle(np.float32(self.rec))
        return centre, radius



def distance (A, B):
    x1, y1 = A
    x2, y2 = B
    return math.sqrt( ((x2-x1)**2) + ((y2-y1)**2) )

def associator(states, arrows, labels):

    root = []
    #testing
    ls = len(states[0])
    la = len(arrows)
    ll = len(labels[1])

    radii, centres, contours = states
    states = zip(radii, centres)
    #set for O(1) deletion
    Ostates = set()
    for s in states:
        radius, centre= s
        state = State(radius, centre)
        Ostates.add(state)

    Oarrows = []
    for a in arrows:
        start, end  = a
        arrow = Arrow(start, end)
        Oarrows.append(arrow)

    Olabels = set()
    mapping, labels1 = labels
    for l in labels1:
        value, rec = l
        label = Label(value, rec)
        Olabels.add(label)

    #DETECTING FINAL STATES
    removal =[]
    for state in Ostates:
        for stateN in Ostates:
            dist = distance(state.centre, stateN.centre)
            #print(dist)
            if dist < (state.radius - stateN.radius):
                #highlight final state
                cv2.circle(img, stateN.centre, stateN.radius, (255,0,255), 2)
                #cv2.circle(img, state.centre, state.radius, (255, 0, 255), 2)
                removal.append(stateN)
                state.set_final()
    for el in removal:
        if el in Ostates:
            Ostates.remove(el)

    #REMOVAL OF LABELS IN STATE
    labelremoval = []
    for state in Ostates:
        for label in Olabels:
            lcentre, lradius = label.min_circle()
            dist = distance(state.centre, lcentre)
            if dist < (state.radius - lradius):
                # highlight removed labels
                start, end = label.rec
                cv2.rectangle(img, start, end, (250, 255, 2), 2)
                labelremoval.append(label)
        for el in labelremoval:
            if el in Olabels:
                Olabels.remove(el)

    #ADD LABELS TO ARROWS
    for a in Oarrows:
        mindist = 10**10
        minlabel = None
        for l in Olabels:
            dist = distance(a.get_mid(), l.get_centre())
            if dist < mindist and dist< distance(a.head, a.tail):
                minlabel = l
                mindist = dist
        if minlabel:
            a.add_label(minlabel)
            Olabels.remove(minlabel)

    #DETECTING CONNECTION BETWEEN STATES AND ARROWS
    for a in Oarrows:
        minhead =10**10
        mintail = 10**10
        headState = None
        tailState = None
        for s in Ostates:
            disthead = distance(s.centre, a.head) - s.radius
            disttail = distance(s.centre, a.tail) - s.radius
            if disthead < minhead:
                headState = s
                minhead = disthead
            if disttail < mintail:
                tailState = s
                mintail = disttail

        a.add_next(headState)
        #If state has label (not start state)
        if a.label:
            tailState.add_arrow(a.label.value, a)
        else:
            root.append(a)


    """for state in Ostates:
        print((state.centre, state.accept, state.out_arrows))
    for arrow in Oarrows:
        print((arrow.head, arrow.next, arrow.label))

    print(("#states #arrows #labels", ls, la, ll))
    print((len(Ostates), len(Oarrows), len(Olabels)))"""

    print(root)
    return root


def nothing(x):
    #print(x)
    pass

def assoc(x):
    #print(x)
    pass

if __name__ == "__main__":
    
    cv2.namedWindow('Connect')
    cv2.resizeWindow('Connect', 600, 600)
    cv2.createTrackbar('Max Radius', 'Connect', 0, 1000, nothing)
    cv2.createTrackbar('Min Radius', 'Connect', 0, 1000, nothing)
    cv2.createTrackbar('Max Area', 'Connect', 0, 1000, nothing)
    cv2.createTrackbar('Min Area', 'Connect', 0, 1000, nothing)
    cv2.createTrackbar('Alphabet', 'Connect', 0, 10000, nothing)
    cv2.createTrackbar('Circle Off.', 'Connect', 0, 100, nothing)
    cv2.createTrackbar('Graph', 'Connect', 0, 1, assoc)

    while (1):

        maxA = cv2.getTrackbarPos('Max Area', 'Connect')
        minA = cv2.getTrackbarPos('Min Area', 'Connect')
        maxR = cv2.getTrackbarPos('Max Radius', 'Connect')
        minR = cv2.getTrackbarPos('Min Radius', 'Connect')
        maxAlpha = cv2.getTrackbarPos('Alphabet', 'Connect')
        offset_between_Cirle = cv2.getTrackbarPos('Circle Off.', 'Connect')

        # Read image
        img = cv2.imread("statemachineone.jpg")
        #img = cv2.imread("test1.jpg")

        # create different copy to use to labels from the circles
        cimg = img.copy()

        labels = get_connected_components(cimg, minA, maxA)

        shapes = get_states(img, minR, maxR, offset_between_Cirle)

        # REMOVING CIRCLES AND LABELS
        # retrieves binary img with removed labels
        Ithresh = labels[2]

        # draw mask with contour for circles on binary image
        mask = np.ones(cimg.shape[:2], dtype="uint8") * 255
        state_Contour = shapes[2]
        for c in state_Contour:
            #change thickness based on the width of lines in the image
            cv2.drawContours(mask, [c], -1, 0, 5)

        # Combine mask with binary image
        Ithresh = cv2.bitwise_and(Ithresh, Ithresh, mask=mask)

        Ithresh, arrows = get_arrows(Ithresh)

        cv2.imshow('Connect2', Ithresh)

        alphaimg = cv2.imread('alphabet.jpg')
        newLabels = detect_alphabet(labels, alphaimg, maxAlpha)
        #map, recs = testaa
        #print(map)

        graph_check = cv2.getTrackbarPos('Graph', 'Connect')
        if graph_check > 0:
            graph = associator(shapes, arrows, newLabels)

        cv2.imshow('Connect', img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

