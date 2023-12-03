import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math


img = cv2.cvtColor(cv2.imread("8.jpg"), cv2.COLOR_BGR2RGB)
background = cv2.cvtColor(cv2.imread('back.jpg'), cv2.COLOR_BGR2RGB)
image= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (250,250))
image = cv2.resize(image, (250,250))
background = cv2.resize(background, (250,250))
#scribble 
#my photo
#image = img[1500:4000, 0:3000]
fg = cv2.imread('9.jpg', cv2.IMREAD_GRAYSCALE)
fg = cv2.resize(fg, (250,250))
bg = cv2.imread('10.jpg', cv2.IMREAD_GRAYSCALE)
bg = cv2.resize(bg, (250,250))
_, fgmask = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
_, bgmask = cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY)

#image = image[350:1450,300:1000]

def create_graph(image, fg, bg):
    rows, cols = image.shape[:2]
    G = nx.Graph()
    
    source = (-1, -1)
    sink = (-2, -2)
    G.add_node(source)
    G.add_node(sink)  
    
    hist_f = cv2.calcHist([fg], [0], cv2.bitwise_not(fg), [256], [0, 256])
    fprob = hist_f / hist_f.sum()
    hist_b = cv2.calcHist([bg], [0], cv2.bitwise_not(bg), [256], [0, 256])
    bprob = hist_b / hist_b.sum()
    variance = np.var(image)

    def capacity_nodes(u, v, image, variance):
        return int(100 * np.exp(-np.square(int(image[u]) - int(image[v])) / (2 * variance)))
    
    sigma = np.var(image)
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            pixel = image[i, j]
            pixel_node = (i, j)

           # P_f_i = hist_f[pixel][0]
           # P_b_i = hist_b[pixel][0]
                
                # Set capacities with a scaling factor to avoid unbounded issues
            #capacity_scale = 100  # You can adjust this value as needed
                
                # Ensure capacities are bounded to avoid numerical instability
            #capacity_f = -capacity_scale * np.log(P_f_i + 1e-5)  # Added small epsilon to avoid log(0)
            #capacity_b = -capacity_scale * np.log(P_b_i + 1e-5)  # Added small epsilon to avoid log(0)
            if fgmask[(i,j)] == 0:
                G.add_edge(source, (i, j), weight=float('inf'))
            elif bgmask[(i, j)] == 0:
                G.add_edge(source, (i, j), weight=0)
            else:
                G.add_edge(source, (i, j), weight=int(-10*np.log10(bprob[image[(i,j)]] + 1e-5)))

             # Node to Sink
            if bgmask[(i, j)] == 0:
                G.add_edge((i, j), sink, weight=float('inf'))
            elif fgmask[(i, j)] == 0:
                G.add_edge((i, j), sink, weight=0)
            else:
                G.add_edge((i, j), sink, weight=int(-10*np.log10(fprob[image[(i,j)]] + 1e-5)))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(i > 0):
                G.add_edge((i, j), (i-1, j), weight=capacity_nodes((i, j), (i-1,j), image, variance))
            if(j > 0):
                G.add_edge((i, j), (i, j-1), weight=capacity_nodes((i, j), (i, j-1), image, variance))
            if(i < image.shape[0] - 1):
                G.add_edge((i, j), (i+1, j), weight=capacity_nodes((i, j), (i+1, j), image, variance))
            if(j < image.shape[1] - 1):
                G.add_edge((i, j), (i, j+1), weight=capacity_nodes((i, j), (i, j+1), image, variance))  

    return G

G = create_graph(image, fgmask, bgmask)
source = (-1, -1)
sink = (-2, -2)
flow_value, flow_dict = nx.minimum_cut(G, source, sink, capacity='weight')
#print(flow_dict)
# segmentation mask
rows, cols = image.shape[:2]
segmentation_mask = np.zeros((rows, cols), dtype=np.uint8)



segment_image =  np.zeros(image.shape, dtype=np.uint8)
for v in flow_dict[1]:
    segment_image[v] = 255

# Display the segmentation result
plt.imshow(segment_image, cmap='gray')
plt.title("Segmentation Mask")
plt.show()

b, g, r = cv2.split(img)
# Apply bitwise AND operation between each channel and the binary image
result_b = cv2.bitwise_and(b, segment_image)
result_g = cv2.bitwise_and(g, segment_image)
result_r = cv2.bitwise_and(r, segment_image)

# Merge the individual channels back into a color image
foreground = cv2.merge((result_b, result_g, result_r))
plt.imshow(foreground)
plt.title("Foreground")
plt.show()


# Overlay foreground image over background image
b, g, r = cv2.split(img)
bg_b, bg_g, bg_r = cv2.split(background)

# Apply the binary mask to each channel of the overlay image
b_masked = cv2.bitwise_and(b, b, mask=segment_image)
g_masked = cv2.bitwise_and(g, g, mask=segment_image)
r_masked = cv2.bitwise_and(r, r, mask=segment_image)

# Invert the binary mask and apply it to each channel of the background image
inverse_mask = cv2.bitwise_not(segment_image)
bg_b_masked = cv2.bitwise_and(bg_b, bg_b, mask=inverse_mask)
bg_g_masked = cv2.bitwise_and(bg_g, bg_g, mask=inverse_mask)
bg_r_masked = cv2.bitwise_and(bg_r, bg_r, mask=inverse_mask)

# Combine the masked overlay channels with the masked background channels
final_b = cv2.add(b_masked, bg_b_masked)
final_g = cv2.add(g_masked, bg_g_masked)
final_r = cv2.add(r_masked, bg_r_masked)

# Merge the combined channels to create the final overlay image
final_overlay_image = cv2.merge((final_b, final_g, final_r))

fig, axes = plt.subplots(1, 3, figsize=(18, 18))

axes[0].set_title('Original Image')
axes[0].imshow(img)

axes[1].set_title('Segmeted Foreground Image')
axes[1].imshow(foreground)
# Display the final overlay image
axes[2].set_title('Foreground image overlayed over Taj-Mahal')
axes[2].imshow(final_overlay_image)
final_overlay_image = cv2.cvtColor(final_overlay_image, cv2.COLOR_BGR2RGB)
plt.imshow(final_overlay_image)
plt.title("Final")
plt.show()
cv2.imwrite('Final.png', final_overlay_image)

plt.show()
