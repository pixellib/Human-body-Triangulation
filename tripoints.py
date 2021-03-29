

import matplotlib.pyplot as plt
import numpy as np
import cv2

####################################################
def countorplot(mergimg):
    contours, hierarchy = cv2.findContours(mergimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, 0, (0,255,0), 3)
    cnt = contours[0]
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    xv=[]
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # cv2.line(rgb,start,end,[0,255,0],2)
        cv2.circle(frame,far,5,[0,0,255],-1)
        # print('circle',far)
        xv.append(far)
    # print('far',np.shape(xv),'shape',xv,'xv',xv[0])
    # cv2.drawContours(rgb, contours, 0, (0,255,0), 3)
    res = xv[::-1] 
    for vc in res:
        gh=tuple(vc)
        cv2.circle(frame,gh,5,[255,0,0],-1)
    #############################################33
    from shapely import geometry
    import matplotlib.pyplot as plt

    # your variables
    coords = [(0, 0), (0, 100), (20, 100), (30, 60), (40, 100), (60, 100), (60, 0), (40, 10), (40, 40), (20, 40), (20, 10)]
    coords=res
    lines = [[coords[i-1], coords[i]] for i in range(len(coords))]
    # print('coord',type(coords)) 
    # your factor of 10%
    # Note: with 20% the polygon becomes a multi-polygon, so a loop for plotting would be needed.
    factor = 0.1010  ## 0.30021

    # code from nathan
    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)

    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*factor

    # assert abs(shrink_distance - center.distance(max_corner)) < 0.0001
    assert abs(shrink_distance - center.distance(max_corner)) > 0.8

    my_polygon = geometry.Polygon(coords)
    my_polygon_shrunken = my_polygon.buffer(-shrink_distance)


    x, y = my_polygon.exterior.xy
    plt.plot(x,y)
    x, y = my_polygon_shrunken.exterior.xy

    vb=[]
    xx=[]
    yy=[]
    for xc,yc in zip(x,y):
        xc=int(xc)
        yc=int(yc)
        
        # print('xc',int(xc),'yc',yc)
        vb.append(np.hstack(((xc),(yc))))
        xx.append(xc)
        yy.append(yc)
    # print('vstack',vb,type(vb))
    # print('vstack type',type(vb))
    vbx=np.asarray(vb)
    gx=[]
    for vc in vbx:
        gh=tuple(vc)
        cv2.circle(frame,gh,5,[0,0,255],-1)

        gx.append(gh)


#####################

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}'.format(x, y)





rgb=cv2.imread('425.jpg')
gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
ret,thresh3 =cv2.threshold(gray,21,150,cv2.THRESH_BINARY)
ed = cv2.Canny(thresh3,12,45)

################################################
img_gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 21, 150,0)
kernel = np.ones((5,5),np.uint8)

dilation = cv2.dilate(thresh3,kernel,iterations = 1)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(rgb, contours, -1, (0,255,0), 1)
cnt = contours[-1]

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
# print(np.shape(hull))
# print('hull',cnt)
# for xc in cnt:
#     print('cntr',cnt)

pnts=[]
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(rgb,start,end,[0,255,0],1)
    cv2.circle(rgb,far,5,[0,0,255],-1)
    pnts.append(np.array(far))
    # print('far',far)


    
pnts=np.array(pnts)
# print('Points',pnts)
# cv2.imshow('img',rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
######################Tril


###########################################
# countorplot(thresh3)
# newarr = np.reshape(len(contours), 2)
# contours, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(rgb, contours, 0, (0,255,0), 3)
# hull = cv2.convexHull(contours,returnPoints = False)
# cv2.imshow('labeled.png', rgb)
# cv2.waitKey(0)

# hull = cv2.convexHull(cnt,returnPoints = False)
# defects = cv2.convexityDefects(cnt,hull)
# xv=[]
# for i in range(defects.shape[0]):
#     s,e,f,d = defects[i,0]
#     start = tuple(cnt[s][0])
#     end = tuple(cnt[e][0])
#     far = tuple(cnt[f][0])

#     # cv2.line(rgb,start,end,[0,255,0],2)
#     cv2.circle(rgb,far,5,[0,0,255],-1)




###########################################


# fig, ax = plt.subplots()
# im = ax.imshow(thresh3, interpolation='none')
# ax.format_coord = Formatter(im)
# plt.show()



# points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
# print('points',np.shape(points))
from scipy.spatial import Delaunay
tri = Delaunay(pnts)
# print('points',pnts[:,0])
import matplotlib.pyplot as plt
tr=plt.triplot(pnts[:,0], pnts[:,1], tri.simplices)
print('delanu points',tri)




# plt.triplot(pnts[:,0], pnts[:,1], tri.simplices)
# plt.plot(pnts[:,0], pnts[:,1], 'o')
# plt.show()
############## matplotlib tri
x=pnts[:,0]
y=pnts[:,1]
n_angles = 36
n_radii = 8
min_radius = 0.25
import matplotlib.tri as tri
import matplotlib.pyplot as plt

triang = tri.Triangulation(x, y)
xtri = tri.Triangulation(x[0:3], y[0:3])
print('x tri',x[0:3], y[0:3])
# triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
#                          y[triang.triangles].mean(axis=1))
#                 < min_radius)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(xtri, 'bo-', lw=1)
ax1.set_title('sk triangulation')
plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#### Polygon shape 
# img=rgb
# points=pnts
# mask = np.zeros(img.shape[0:2], dtype=np.uint8)
# res = cv2.bitwise_and(img,img,mask = mask)

# cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
# rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect

# cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
# wbg = np.ones_like(img, np.uint8)*255
# cv2.bitwise_not(wbg,wbg, mask=mask)
# # overlap the resulted cropped image on the white background
# dst = wbg+res
# cv2.imshow("poly image", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()