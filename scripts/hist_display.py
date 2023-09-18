import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import os

current_image_index = 0

def display_optimized_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    maskh = np.zeros((480,640),dtype='uint8')
    h=int(0.8*480)
    polyh = np.array([[(0,h),(640,h),(640,480),(0,480)]]) # polyh might need adjustment
    cv2.fillPoly(maskh,polyh,255)
    masks = np.zeros((480,640),dtype='uint8')
    polys = np.array([[(0,300),(640,300),(640,340),(0,340)]]) # polys might need adjustment
    cv2.fillPoly(masks,polys,255)
    maskc = np.zeros((480,640),dtype='uint8')
    h=int(0.5*480)
    polyc = np.array([[(0,h),(640,h),(640,480),(0,480)]]) # polyh might need adjustment
    cv2.fillPoly(maskc,polyc,255)

    def optimized_histogram(image, show=False):
        """
        Extract the lanes from the image using the histogram method
        :param image: Image to extract the lanes from
        :param show: Boolean to show the image
        :return: The steering angle
        """
        stopline = False
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # brightness = np.mean(img_gray)
        h = 480
        w = 640
        img_roi = cv2.bitwise_and(img_gray,maskh)
        t = np.max(img_roi)-55
        if t<30:
            t=30
        np.clip(t,30,200)
        print(t)
        ret, thresh = cv2.threshold(img_roi, t, 255, cv2.THRESH_BINARY)
        hist=np.zeros((1,w))
        for i in range(w):
            hist[0,i]=np.sum(thresh[:,i])

        # get lane marking delimiters
        lanes=[]
        p=0
        for i in range(w):
            if hist[0,i]>=1500 and p==0:
                lanes.append(i)
                p=255
            elif hist[0,i]<=1500 and p==255:
                lanes.append(i)
                p=0
        if len(lanes)%2==1:
            lanes.append(w-1)
        # print(lanes)

        # get lane markings
        centers=[]
        for i in range(int(len(lanes)/2)):
            if abs(lanes[2*i]-lanes[2*i+1])>3: # and abs(lanes[2*i]-lanes[2*i+1])<100: #exclude large lanes
                centers.append((lanes[2*i]+lanes[2*i+1])/2)
            if abs(lanes[2*i]-lanes[2*i+1])>350 and t>50:
                stopline = True
        return centers, stopline
    
    # Define a function to go to the next image
    def next_image(event):
        global current_image_index
        current_image_index = (current_image_index + 1) % 615
        on_submit(str(current_image_index))

    # Define a function to go to the previous image
    def previous_image(event):
        global current_image_index
        current_image_index = (current_image_index - 1) % 615
        on_submit(str(current_image_index))

    _, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.ravel()
    plt.subplots_adjust(bottom=0.2)
    # Add the buttons
    button_previous_ax = plt.axes([0.25, 0.05, 0.1, 0.075])
    button_previous = plt.Button(button_previous_ax, label='Previous')
    button_previous.on_clicked(previous_image)

    button_next_ax = plt.axes([0.65, 0.05, 0.1, 0.075])
    button_next = plt.Button(button_next_ax, label='Next')
    button_next.on_clicked(next_image)

    text_box = TextBox(plt.axes([0.25, 0.95, 0.5, 0.05]), 'Enter image path:')
    def on_submit(text):
        image_path = text
        global current_image_index
        current_image_index = int(text)
        # print(image_path)
        try:
            image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/images/images11/'+image_path+'.jpg')
        except:
            print("invalid path")
        image = cv2.resize(image, (640, 480))
        # original image
        # ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # ax[0].set_title('Original Image')

        ax[2].clear()
        ax[3].clear()

        # img_gray
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ax[0].imshow(img_gray, cmap='gray')
        ax[0].set_title('Grayscale Image')

        # img_roi
        img_roi = cv2.bitwise_and(img_gray, maskh)
        # ax[2].imshow(img_roi, cmap='gray')
        # ax[2].set_title('ROI Image')

        # threshold
        threshold_value = np.clip(np.max(img_roi) - 55, 30, 200)
        _, thresh = cv2.threshold(img_roi, threshold_value, 255, cv2.THRESH_BINARY)
        ax[1].imshow(thresh, cmap='gray')
        ax[1].set_title('Threshold Image')

        # lanes
        centers, stopline = optimized_histogram(image)

        # hist
        if not stopline:
            hist = np.sum(thresh, axis=0)
            avg = np.full((640), np.average(hist))
            ax[2].plot(hist)
            ax[2].plot(avg)
            print(np.average(hist))
            ax[2].set_title('Histogram')
        else:
            img_roi = cv2.bitwise_and(img_gray,maskc)
            t = np.max(img_roi)-55
            if t<30:
                t=30
            np.clip(t,30,200)
            print(t)
            ret, thresh = cv2.threshold(img_roi, t, 255, cv2.THRESH_BINARY)
            hist = np.sum(thresh, axis=0)
            avg = np.full((640), np.average(hist))
            ax[2].plot(hist)
            ax[2].plot(avg)
            print(np.average(hist))
            ax[2].set_title('Histogram')

        # get lane centers based on 4 cases
        if len(centers)==0: # no lane detected
            center = 640/2
            # case = 0
        elif len(centers)==1: # one lane detected
            # case = 1
            if centers[0]>640/2:
                center = (centers[0]-0)/2
            else:
                center = (centers[0]*2+640)/2
                # center = (centers[0]+640)/2
        elif abs(centers[0]-centers[len(centers)-1])<200: # the left most lane and the right most lane are close together (fuse them)
            # case = 2
            if (centers[0]+centers[len(centers)-1])>640:
                center = ((centers[0]+centers[len(centers)-1])/2+0)/2
            else:
                center = ((centers[0]+centers[len(centers)-1])+640)/2
        else: # the left most lane and the right most lane are far (avg them)
            # case = 3
            center = (centers[0]+centers[len(centers)-1])/2
        print(centers)
        print(center)
        ax[3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i in range(0, len(centers)):
            ax[3].plot([centers[i],centers[i]], [480, 380], 'r-', linewidth=5)
        ax[3].plot([center, center], [480, 380], 'g-', linewidth=5)
        ax[3].set_title('Detected Lanes and Lane center')
        if stopline:
            ax[3].text(64, 48, "stopline!", fontsize=12, color='red')
        plt.draw()
    text_box.on_submit(on_submit)

    # original image
    # ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # ax[0].set_title('Original Image')

    # img_gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ax[0].imshow(img_gray, cmap='gray')
    ax[0].set_title('Grayscale Image')

    # img_roi
    img_roi = cv2.bitwise_and(img_gray, maskh)
    # ax[2].imshow(img_roi, cmap='gray')
    # ax[2].set_title('ROI Image')

    # threshold
    threshold_value = np.clip(np.max(img_roi) - 55, 30, 200)
    _, thresh = cv2.threshold(img_roi, threshold_value, 255, cv2.THRESH_BINARY)
    ax[1].imshow(thresh, cmap='gray')
    ax[1].set_title('Threshold Image')

    # hist
    hist = np.sum(thresh, axis=0)
    avg = np.full((640), np.average(hist))
    ax[2].plot(hist)
    ax[2].plot(avg)
    print(np.average(hist))
    ax[2].set_title('Histogram')

    # lanes
    centers, stopline = optimized_histogram(image)
    # get lane centers based on 4 cases
    if len(centers)==0: # no lane detected
        center = 640/2
        # case = 0
    elif len(centers)==1: # one lane detected
        # case = 1
        if centers[0]>640/2:
            center = (centers[0]-0)/2
        else:
            # center = (centers[0]*2+640)/2
            center = (centers[0]+640)/2
    elif abs(centers[0]-centers[len(centers)-1])<200: # the left most lane and the right most lane are close together (fuse them)
        # case = 2
        if (centers[0]+centers[len(centers)-1])>640:
            center = ((centers[0]+centers[len(centers)-1])/2+0)/2
        else:
            center = ((centers[0]+centers[len(centers)-1])+640)/2
    else: # the left most lane and the right most lane are far (avg them)
        # case = 3
        center = (centers[0]+centers[len(centers)-1])/2
    print(centers)
    print(center)
    ax[3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i in range(0, len(centers)):
        ax[3].plot([centers[i],centers[i]], [480, 380], 'r-', linewidth=5)
    ax[3].plot([center, center], [480, 380], 'g-', linewidth=5)
    ax[3].set_title('Detected Lanes and Lane center')
    ax[3].text(center, 350, str(center), fontsize=12, color='green')
    if stopline:
        ax[3].text(64, 48, "stopline!", fontsize=12, color='red')

    plt.show()

if __name__ == '__main__':
    image_path = os.path.dirname(os.path.realpath(__file__))+'/images/images11/0.jpg'
    display_optimized_histogram(image_path)