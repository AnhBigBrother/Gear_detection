import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.axis("off")


# ảnh đầu vào
img_path = "gear4.jpg"
img = cv2.imread(img_path)

# chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# lọc trung vị
gray = cv2.medianBlur(gray, 5)

# nhị phân hóa ảnh với phương pháp otsu
(thres, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# hiển thị ảnh đã nhị phân hóa
plt.figure(figsize=(8, 8))
imshow(bw)
plt.title("Ảnh nhị phân")
plt.show()

floodfill = bw.copy()

# lấp đầy khoảng trống ảnh trong bánh răng
h, w = bw.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

cv2.floodFill(floodfill, mask, (0, 0), 0);

# đảo ảnh nhị phân
floodfill_inv = cv2.bitwise_not(floodfill)

# cộng ảnh đó với ảnh gốc
fill = bw & floodfill_inv

# hiển thị ảnh sau khi lấp đầy
plt.figure(figsize=(15, 12))

ax = plt.subplot(1, 3, 1)
imshow(floodfill)
plt.title("Floodfill")

ax = plt.subplot(1, 3, 2)
imshow(floodfill_inv)
plt.title("Inverted Floodfill")

ax = plt.subplot(1, 3, 3)
imshow(fill)
plt.title("Filled Image")

plt.show()


# đảo ảnh
fill = cv2.bitwise_not(fill)
# tạo ma trận kenel dạng elip 100*100
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(100, 100))
# mở rộng kenel từ tâm ra đường kinh trong
open = cv2.morphologyEx(fill, cv2.MORPH_OPEN, kernel)

# hiển thị ảnh
plt.figure(figsize=(15, 12))
plt.subplot(1, 2, 1)
imshow(open)
plt.title("Opened region")

plt.subplot(1, 2, 2)
overlap = cv2.addWeighted(fill, 0.5, open, 0.5, 0.0)
imshow(overlap)
plt.title("ảnh chồng lấp")
plt.show()

# Tính đường kính ngoài và trong
outer_diameter = 0
inner_diameter = 0
contours1, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours1:
    (x, y, w, h) = cv2.boundingRect(contour)
    diameter1 = w if w > h else h
    if diameter1 > outer_diameter:
        outer_diameter = diameter1
contours2, _ = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours2:
    (x, y, w, h) = cv2.boundingRect(contour)
    diameter2 = w if w > h else h
    if diameter2 > inner_diameter:
        inner_diameter = diameter2

# phân đoạn răng
img_cogs = open | bw

# hiển thị răng phân đoạn
plt.figure(figsize=(8, 8))
imshow(img_cogs)
plt.title("răng sau khi phân đoạn")
plt.show()

# đảo ngược ảnh
img_cogs = cv2.bitwise_not(img_cogs)
# nhận diện các vùng liên kết với nhau
count, labels, stats, centroids = cv2.connectedComponentsWithStats(img_cogs)
areas = stats[:, 4]
median = np.median(areas)

# hiển thị răng đã nhận diện

img_centroids = img.copy()
cog_count = 0
for i in range(0, count):
    # Filter the correct area
    if areas[i] > median - 100 and areas[i] < median + 100:
        cv2.circle(img_centroids, (int(centroids[i, 0]), int(centroids[i, 1])), 0, (0, 255, 0), 5)
        cog_count += 1

plt.figure(figsize=(12, 12))
imshow(img_centroids)
plt.title(f"Số răng : {cog_count} \nĐường kính ngoài: {outer_diameter} \nĐường kính trong: {inner_diameter}", fontsize=15)
plt.show()