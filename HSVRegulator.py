import cv2
import win32ui
import win32api
import win32con
import win32gui
import numpy as np

from pydantic import BaseModel


class Region(BaseModel):
    width: int
    height: int
    left: int
    top: int


def validate_region(region: Region | None) -> Region:
    if region is None:
        return Region(
            width=win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            height=win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN),
            left=win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN),
            top=win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
        )

    elif type(region) is Region:
        # region.width = region.width - region.left + 1
        # region.height = region.height - region.top + 1

        return region

    raise NotImplementedError(
        'Region should have type of Region class'
    )


def grab_screen(region: Region) -> np.ndarray:
    hwin = win32gui.GetDesktopWindow()

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()

    bmp.CreateCompatibleBitmap(
        srcdc, region.width, region.height
    )
    memdc.SelectObject(bmp)
    memdc.BitBlt(
        (0, 0), (region.width, region.height),
        srcdc, (region.left, region.top), win32con.SRCCOPY
    )

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype=np.uint8)
    img.shape = (region.height, region.width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return np.array(img)


# [Hmin, Smin, Vmin]
LOWER_RANGE_HSV_ARRAY_PORTAL = [0, 145, 125]
# [Hmax, Smax, Vmax]
HIGHER_RANGE_HSV_ARRAY_PORTAL = [3, 255, 255]

# SHOW_RECTANGLES = False
CAPTURING_CHECK: bool = False
IMG_PATHS = []

if not CAPTURING_CHECK:
    # Load image
    images = [
        cv2.imread(img_path)
        for img_path in IMG_PATHS
    ]
    assert len(images) >= 1

# Create a window
cv2.namedWindow('image')

# if SHOW_RECTANGLES:
#     cv2.namedWindow('image_debug')
#     image_debug_original = cv2.imread(IMG_PATHS)
#     image_debug = image_debug_original.copy()

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar(
    'HMin', 'image', LOWER_RANGE_HSV_ARRAY_PORTAL[0], 360, lambda x: ...
)
cv2.createTrackbar(
    'SMin', 'image', LOWER_RANGE_HSV_ARRAY_PORTAL[1], 255, lambda x: ...
)
cv2.createTrackbar(
    'VMin', 'image', LOWER_RANGE_HSV_ARRAY_PORTAL[2], 255, lambda x: ...
)
cv2.createTrackbar(
    'HMax', 'image', HIGHER_RANGE_HSV_ARRAY_PORTAL[0], 360, lambda x: ...
)
cv2.createTrackbar(
    'SMax', 'image', HIGHER_RANGE_HSV_ARRAY_PORTAL[1], 255, lambda x: ...
)
cv2.createTrackbar(
    'VMax', 'image', HIGHER_RANGE_HSV_ARRAY_PORTAL[2], 255, lambda x: ...
)

# Set default value for Max HSV trackbars
# cv2.setTrackbarPos('HMax', 'image', 179)
# cv2.setTrackbarPos('SMax', 'image', 255)
# cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while (1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    if CAPTURING_CHECK:
        images = [grab_screen(validate_region(
            Region(width=1920, height=1080, left=0, top=0)))]

    results = []

    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # if SHOW_RECTANGLES:
        #     is_rectangled = False
        #     for contour in cv2.findContours(
        #         mask, cv2.RETR_TREE, cv2.CIRCLES_GRID_FINDER_PARAMETERS_ASYMMETRIC_GRID
        #     )[0]:
        #         if cv2.contourArea(contour) > PORTAL_AREA_RADIUS:
        #             x, y, w, h = cv2.boundingRect(contour)
        #             rectangle_image = cv2.rectangle(
        #                 image_debug.copy(), (x, y), (x + w, y + h), (0, 0, 255), 1
        #             )
        #             is_rectangled = True
        #             print("RECTANGLE WAS FOUNDED")

        #     if not is_rectangled:
        #         rectangle_image = image_debug_original.copy()

        results.append(cv2.bitwise_and(image, image, mask=mask))

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
        print("MIN [%d, %d, %d], MAX [%d, %d, %d]" % (
            hMin, sMin, vMin, hMax, sMax, vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image

    for i, result in enumerate(results):
        cv2.imshow(f'image {i+1}', result)

    # if SHOW_RECTANGLES:
    #     cv2.imshow('image_debug', rectangle_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
